"""Train the Geometry Dash agent with PPO, or watch a trained one play.

    python train.py --sim                 # no game needed: train against the simulator
    python train.py --sim --envs 64       # 64 courses at once — batched inference
    python train.py                       # real GD, whatever level you are sitting in
    python train.py --level 1 --speed 8   # load Stereo Madness, run 8x wall-clock
    python train.py --practice            # checkpoint curriculum: grind segments
    python train.py --play runs/gd/best.pt --speed 1   # watch it, no learning

There are two rollout backends and one learner. Live Geometry Dash is a single
environment stepped inside a frame handshake, so it collects one transition at a
time. The simulator is stepped as N courses at once, so it collects N, and the
policy decides for all of them in a single forward pass. `--rollout` is the
number of *transitions* per PPO update either way, so the sample budget behind an
update does not change when you change `--envs`.

The loop itself never draws anything. Open the URL it prints to watch the network
live; close the tab and training goes straight back to full speed.
"""

import argparse
import csv
import os
import time
import webbrowser
from collections import deque

import numpy as np
import torch

from gdbot import env as gdenv
from gdbot import policy as P
from gdbot import ppo
from gdbot.bridge import OFFICIAL_LEVELS
from gdbot.env import BridgeLost, LiveEnv, VersionMismatch
from gdbot.obs import (CHANNEL_NAMES, GRID_SHAPE, N_SCALARS, PLAYER_COL,
                       PLAYER_ROW, SCALAR_NAMES)
from gdbot.sim_env import SimEnv
from gdbot.telemetry import Telemetry, b64
from gdbot.vec_env import VecSimEnv

HERE = os.path.dirname(os.path.abspath(__file__))

# Consecutive bridge stalls to absorb before giving up. Generous on purpose: an
# overnight run should survive hitches, and a genuinely dead game fails fast
# anyway because recover() cannot reconnect.
MAX_STALLS = 20

# Bins in the advantage histogram the viewer draws. Enough to show the shape of
# the distribution, few enough that the payload stays a rounding error.
ADV_BINS = 32


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", action="store_true",
                   help="train against the simulator instead of real GD")
    # Defaults to 1, not to something fast. Measured at a fixed sample budget,
    # 8 parallel courses learn markedly *worse* than one — see
    # docs/mathematical-report.md section 7.1. The speed is real and useful when
    # wall-clock is the binding constraint; it is not free, so it is opt-in.
    p.add_argument("--envs", type=int, default=1,
                   help="simulator courses stepped in parallel: faster per second, "
                        "but worse per sample (sim only; live is always 1)")
    p.add_argument("--play", metavar="CKPT",
                   help="load a checkpoint and play greedily, without learning")
    p.add_argument("--run", default="gd", help="run name under runs/")
    p.add_argument("--level", type=int, default=0,
                   help="official level id 1-21 to load before training")
    p.add_argument("--speed", type=int, default=4,
                   help="physics steps per rendered frame (wall-clock multiplier)")
    p.add_argument("--step-hz", type=int, default=60,
                   help="agent decisions per second of GAME time (keep fixed per run)")
    p.add_argument("--practice", action="store_true",
                   help="practice mode + auto checkpoints (segment curriculum)")
    p.add_argument("--rollout", type=int, default=2048,
                   help="transitions per PPO update (split across --envs)")
    p.add_argument("--minibatch", type=int, default=256, help="PPO minibatch size")
    p.add_argument("--updates", type=int, default=100000, help="PPO updates to run")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--entropy", type=float, default=0.01)
    # Reward shaping is a hyperparameter like any other, and the analysis in
    # docs/mathematical-report.md turns on being able to move these.
    p.add_argument("--progress-reward", type=float, default=None,
                   help="reward for clearing a whole level (default 10)")
    p.add_argument("--death-penalty", type=float, default=None,
                   help="cost of dying (default 1)")
    p.add_argument("--complete-bonus", type=float, default=None,
                   help="bonus for finishing a level (default 10)")
    p.add_argument("--clip", type=float, default=0.2, help="PPO clip range")
    p.add_argument("--target-kl", type=float, default=0.03,
                   help="stop an update once one epoch moves this far")
    p.add_argument("--no-value-clip", action="store_true",
                   help="disable the clipped value loss")
    p.add_argument("--anneal", type=int, default=0, metavar="N",
                   help="linearly decay lr and entropy to a tenth over N updates")
    p.add_argument("--epochs", type=int, default=4, help="PPO epochs per update")
    p.add_argument("--hidden", type=int, default=256, help="dense trunk width")
    p.add_argument("--seed", type=int, default=1,
                   help="simulator course seed (and torch seed)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-viewer", action="store_true", help="do not serve the viewer")
    p.add_argument("--open", action="store_true", help="open the viewer in a browser")
    p.add_argument("--fps", type=float, default=20.0, help="viewer frames per second")
    p.add_argument("--http-port", type=int, default=8765)
    p.add_argument("--ws-port", type=int, default=8766)
    return p.parse_args()


def pick_device(choice: str, n_envs: int) -> str:
    """CPU unless there is both a GPU and a batch big enough to feed it.

    Measured on this network: at batch 1 CUDA is 3.7x *slower* than the CPU,
    because 216k parameters cannot cover a kernel launch. It only starts paying
    around batch 32. Picking "auto" used to mean "cuda if present", which quietly
    made single-environment runs worse.
    """
    if choice != "auto":
        return choice
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda" if n_envs >= 32 else "cpu"


class RunLog:
    """Two CSVs — one row per episode, one per PPO update. Appends across runs."""

    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        self.dir = run_dir
        self._ep_f, self._ep = self._open("episodes.csv", [
            "wall", "env_steps", "episode", "return", "best_percent", "steps",
            "complete", "dead"])
        self._up_f, self._up = self._open("updates.csv", [
            "wall", "env_steps", "update", "mean_return", "mean_percent",
            "best_percent", "pi_loss", "v_loss", "entropy", "kl", "clip_frac",
            "steps_per_sec", "explained_variance", "grad_norm", "lr", "epochs"])

    def _open(self, name, header):
        path = os.path.join(self.dir, name)
        fresh = not os.path.exists(path) or os.path.getsize(path) == 0
        f = open(path, "a", newline="")
        w = csv.writer(f)
        if fresh:
            w.writerow(header)
        return f, w

    def episode(self, row):
        self._ep.writerow(row)

    def update(self, row):
        self._up.writerow(row)
        self._ep_f.flush()
        self._up_f.flush()

    def close(self):
        self._ep_f.close()
        self._up_f.close()


class State:
    """The counters the two rollout backends both advance."""

    def __init__(self, env_steps: int, best_ever: float):
        self.env_steps = env_steps
        self.episodes = 0
        self.best_ever = best_ever
        self.ep_return = 0.0            # live backend only; vec tracks its own
        self.returns = deque(maxlen=200)
        self.percents = deque(maxlen=200)
        self.stalls = 0
        self.obs = None                 # Obs (live) or (grid, scalars) (vec)


def _reset_with_recovery(env, tries: int = MAX_STALLS):
    """env.reset(), but a stall retries instead of ending the run."""
    for attempt in range(tries):
        try:
            return env.reset()
        except BridgeLost as exc:
            print(f"\n[stall on reset {attempt + 1}/{tries}] {exc}\n  recovering...")
            if not env.recover():
                raise
            print("  recovered")
    raise BridgeLost(f"could not start an attempt after {tries} recoveries")


def make_env(args):
    if args.sim:
        if args.envs > 1:
            return (VecSimEnv(n_envs=args.envs, seed=args.seed,
                              reseed_each_episode=True),
                    f"simulator x{args.envs}")
        return SimEnv(seed=args.seed, reseed_each_episode=True), "simulator"
    env = LiveEnv(speed=args.speed, step_hz=args.step_hz, practice=args.practice)
    print("Waiting for the GDBot Bridge (start GD, enter a level)...")
    if not env.connect(timeout=120):
        raise SystemExit("Bridge never connected — is the mod installed and GD in a level?")
    if args.level:
        print(f"Loading level {args.level}: {OFFICIAL_LEVELS.get(args.level, '?')}")
        env.load_level(args.level)
    else:
        # Remember whatever level you are already sitting in, so a recovery that
        # lands on the menu can put us back into the right one.
        env.level_id = int(env.b.read().get("level_id", 0))
    return env, "Geometry Dash"


# --- the wire format ---------------------------------------------------------
def _wire_learning(learning) -> dict:
    """Base64 the kernel block; the rest of the learning snapshot is plain JSON."""
    if not learning:
        return None
    k = learning["kernels"]
    return {"layers": learning["layers"], "grads": learning.get("grads", {}),
            "kernels": {"n": k["n"], "c": k["c"], "kh": k["kh"], "kw": k["kw"],
                        "data": b64(k["data"])}}


def snapshot(grid, scalars, viz, action, percent, hud, series,
             learning=None, saliency=None, adv_hist=None) -> dict:
    """Everything the viewer draws for one frame, already wire-ready.

    `learning`, `saliency` and `adv_hist` all change once per PPO update rather
    than once per frame, so they are computed there and simply carried along.
    """
    return {
        "learning": _wire_learning(learning),
        "grid": b64((np.asarray(grid) > 0).astype(np.uint8).ravel()),
        "saliency": b64(saliency) if saliency is not None else None,
        "adv_hist": adv_hist,
        "scalars": [round(float(v), 4) for v in scalars],
        "probs": [round(p, 4) for p in viz["probs"]],
        "value": round(viz["value"], 3),
        "action": int(action),
        "hidden": b64(viz["hidden"]),
        "head": b64(viz["head"]),
        "stages": [{"n": s["n"], "h": s["h"], "w": s["w"], "data": b64(s["data"])}
                   for s in viz["stages"]],
        "percent": round(float(percent), 4),
        "hud": hud,
        "series": series,
    }


def _hud(args, st, update_i, level_name, t_start, t0, steps0, extra) -> dict:
    elapsed = max(1e-6, time.time() - t0)
    hud = {
        "update": update_i, "env_steps": st.env_steps, "episodes": st.episodes,
        "best_ever": round(st.best_ever * 100, 2),
        "recent": round(float(np.mean(st.percents)) * 100, 2) if st.percents else 0.0,
        "speed": args.speed, "step_hz": args.step_hz, "envs": args.envs if args.sim else 1,
        "level": level_name,
        "sps": round((st.env_steps - steps0) / elapsed, 1),
        "wall": int(time.time() - t_start),
    }
    hud.update(extra)
    return hud


def _series(st, hist) -> dict:
    return {"returns": [round(r, 2) for r in st.returns],
            "percents": [round(p * 100, 2) for p in st.percents],
            "entropy": [round(e, 4) for e in hist["entropy"]],
            "kl": [round(k, 5) for k in hist["kl"]],
            "v_loss": [round(v, 4) for v in hist["v_loss"]],
            "clip_frac": [round(c, 4) for c in hist["clip_frac"]],
            "ev": [round(e, 4) for e in hist["ev"]],
            "grad_norm": [round(g, 4) for g in hist["grad_norm"]]}


# --- rollout backends --------------------------------------------------------
def collect_live(env, model, buf, tele, st, args, ctx):
    """One rollout from the real game (or the scalar simulator): one step at a time."""
    while not buf.full:
        capture = tele.should_capture()
        action, logp, value, viz = model.act(st.obs, greedy=ctx["playing"],
                                             introspect=capture)
        try:
            next_obs, reward, done, info = env.step(action)
        except BridgeLost as exc:
            # A stall is a pause, not an ending. This exact failure — one hitch
            # past the frame timeout — killed a 7.12h run that was still
            # improving, with the game perfectly healthy throughout.
            st.stalls += 1
            print(f"\n[stall {st.stalls}/{MAX_STALLS}] {exc}\n"
                  f"  recovering (re-attaching, reloading the level if needed)...")
            if st.stalls > MAX_STALLS or not env.recover():
                raise
            # Close the episode at the discontinuity. Without this, GAE
            # bootstraps credit straight across the gap and blends two unrelated
            # episodes into one advantage estimate.
            if buf.n:
                buf.dones[buf.n - 1] = 1.0
            st.ep_return = 0.0
            try:
                st.obs = _reset_with_recovery(env)
            except BridgeLost:
                continue          # stalled again mid-reset; recover again
            print("  recovered — continuing")
            continue

        st.env_steps += 1
        st.ep_return += reward

        # A truncated episode is not a failure, so bootstrap its tail instead of
        # teaching the policy that surviving ends the world.
        if done and info.get("timeout") and not info.get("dead"):
            _a, _lp, v_next, _ = model.act(next_obs, greedy=True)
            reward += args.gamma * v_next

        # Fill the buffer even when playing: it is what ends the inner loop, and
        # one preallocated row costs a memcpy.
        buf.add(st.obs, action, logp, value, reward, done)

        if capture:
            tele.publish(snapshot(
                st.obs.grid, st.obs.scalars, viz, action, info.get("percent", 0.0),
                _hud(args, st, ctx["update_i"], ctx["level_name"], ctx["t_start"],
                     ctx["t0"], ctx["steps0"],
                     {"checkpoints": info.get("checkpoints", 0),
                      # When no --level was given we are training in whatever
                      # level the game was already sitting in, and only the frame
                      # knows which one that is.
                      "level": ctx["level_name"] or f"level {info.get('level_id', 0)}"}),
                _series(st, ctx["hist"]), learning=ctx["learning"][0],
                saliency=model.saliency(st.obs.grid, st.obs.scalars, action),
                adv_hist=ctx["adv_hist"][0]))

        st.obs = next_obs
        if done:
            st.episodes += 1
            bp = info.get("best_percent", 0.0)
            st.returns.append(st.ep_return)
            st.percents.append(bp)
            st.best_ever = max(st.best_ever, bp)
            if ctx["log"]:
                ctx["log"].episode([round(time.time() - ctx["t_start"], 1),
                                    st.env_steps, st.episodes, round(st.ep_return, 3),
                                    round(bp, 4), info.get("steps", 0),
                                    int(info.get("complete", False)),
                                    int(info.get("dead", False))])
            st.ep_return = 0.0
            st.obs = _reset_with_recovery(env)


def collect_vec(env, model, buf, tele, st, args, ctx):
    """One rollout from N simulators: one batched decision per step, for all of them."""
    while not buf.full:
        grid, scalars = st.obs
        actions, logps, values = model.act_batch(grid, scalars, greedy=ctx["playing"])
        (next_grid, next_scalars), rewards, dones, info = env.step(actions)
        st.env_steps += env.n

        # Bootstrap the tail of any episode that ended on the step limit rather
        # than on a death, using the frame the auto-reset threw away.
        trunc = info.get("truncated")
        if trunc is not None:
            idx, tg, ts = trunc
            _a, _lp, v_next = model.act_batch(tg, ts, greedy=True)
            rewards = rewards.copy()
            rewards[idx] += args.gamma * v_next

        buf.add(grid, scalars, actions, logps, values, rewards, dones)

        if tele.should_capture():
            # Environment 0 is the one on screen. `viz_one` re-runs the forward
            # for that state instead of re-deciding, so the action drawn is the
            # action the simulator actually received.
            viz = model.viz_one(grid[0], scalars[0])
            tele.publish(snapshot(
                grid[0], scalars[0], viz, int(actions[0]), info["percent"][0],
                _hud(args, st, ctx["update_i"], ctx["level_name"], ctx["t_start"],
                     ctx["t0"], ctx["steps0"], {"checkpoints": 0}),
                _series(st, ctx["hist"]), learning=ctx["learning"][0],
                saliency=model.saliency(grid[0], scalars[0], int(actions[0])),
                adv_hist=ctx["adv_hist"][0]))

        st.obs = (next_grid, next_scalars)
        for ep in info["episodes"]:
            st.episodes += 1
            st.returns.append(ep["return"])
            st.percents.append(ep["best_percent"])
            st.best_ever = max(st.best_ever, ep["best_percent"])
            if ctx["log"]:
                ctx["log"].episode([round(time.time() - ctx["t_start"], 1),
                                    st.env_steps, st.episodes, round(ep["return"], 3),
                                    round(ep["best_percent"], 4), ep["steps"],
                                    int(ep["complete"]), int(ep["dead"])])


def main():
    args = parse_args()
    playing = bool(args.play)
    if playing or not args.sim:
        args.envs = 1              # one game, or one greedy demonstration
    device = pick_device(args.device, args.envs)
    run_dir = os.path.join(HERE, "runs", args.run)
    vectorised = args.sim and args.envs > 1

    # Applied before any environment exists, and applied to the module the two
    # reward functions read at call time, so every backend agrees.
    for flag, const in (("progress_reward", "PROGRESS_REWARD"),
                        ("death_penalty", "DEATH_PENALTY"),
                        ("complete_bonus", "COMPLETE_BONUS")):
        v = getattr(args, flag)
        if v is not None:
            setattr(gdenv, const, v)
            print(f"reward: {const} = {v}")

    torch.manual_seed(args.seed)          # so a sweep config is reproducible
    env, env_name = make_env(args)
    if playing:
        model, _ck = P.load(args.play, device=device)
        model.eval()
        print(f"Playing {args.play} on {env_name} ({device}) — no learning.")
    else:
        model = P.ConvActorCritic(hidden=args.hidden).to(device)
        os.makedirs(run_dir, exist_ok=True)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    start_update, env_steps, best_ever, resumed_mean = 0, 0, 0.0, 0.0
    latest = os.path.join(run_dir, "latest.pt")
    if not playing and os.path.exists(latest):
        model, ck = P.load(latest, device=device)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
        if ck.get("optim"):
            optim.load_state_dict(ck["optim"])
        start_update = ck.get("update", 0)
        env_steps = ck.get("env_steps", 0)
        best_ever = ck.get("best_percent", 0.0)
        resumed_mean = ck.get("best_mean", 0.0)
        print(f"Resumed {latest}: update {start_update}, {env_steps:,} env steps, "
              f"best {best_ever * 100:.1f}%")
        if model.n_scalars != N_SCALARS:
            print(f"  note: checkpoint predates the {N_SCALARS}-scalar observation "
                  f"({model.n_scalars} scalars) — the extra inputs are ignored")

    log = None if playing else RunLog(run_dir)
    tele = Telemetry(http_port=args.http_port, ws_port=args.ws_port, fps=args.fps,
                     enabled=not args.no_viewer)
    url = tele.start({
        "grid_shape": list(GRID_SHAPE), "channels": list(CHANNEL_NAMES),
        "player_col": PLAYER_COL, "player_row": PLAYER_ROW,
        "scalar_names": list(SCALAR_NAMES), "n_scalars": N_SCALARS,
        "stage_shapes": [list(s) for s in model.map_shapes],
        "viz_maps": P.VIZ_MAPS, "hidden": int(model.fc.out_features),
        "env": env_name, "device": device, "run": args.run,
        "envs": args.envs, "params": sum(p.numel() for p in model.parameters()),
        "mode": "play" if playing else "train",
    })
    if url:
        print(f"\n  viewer:  {url}\n")
        if args.open:
            webbrowser.open(url)

    # --rollout is a transition budget, so an update sees the same number of
    # samples whatever --envs is set to and a sweep stays comparable.
    if vectorised:
        per_env = max(1, args.rollout // args.envs)
        buf = ppo.VecRollout(per_env, args.envs, GRID_SHAPE, N_SCALARS)
        print(f"Rollout: {per_env} steps x {args.envs} envs = "
              f"{per_env * args.envs} transitions per update")
        # GAE decays as (gamma*lam)^k, so it has an effective horizon of
        # 1/(1-gamma*lam) steps — 17 at the defaults. A rollout shorter than a
        # few of those is truncated before the advantage estimate has converged,
        # and no amount of extra throughput buys back the credit it loses.
        horizon = 1.0 / max(1e-6, 1.0 - args.gamma * args.lam)
        if per_env < 4 * horizon:
            print(f"  warning: {per_env} steps/env is short against a GAE horizon "
                  f"of {horizon:.0f}; prefer --envs <= {int(args.rollout / (4 * horizon))} "
                  f"or a larger --rollout")
    else:
        buf = ppo.Rollout(args.rollout, GRID_SHAPE, N_SCALARS)

    st = State(env_steps, best_ever)
    best_mean = resumed_mean   # best rolling-mean percent so far; gates best.pt

    # Learning-view state: the last weights fingerprint to diff against, and the
    # PPO health history the viewer sparklines. Both are cheap and only touched
    # once per update, so an unwatched run barely notices them.
    fingerprint = model.weight_fingerprint()
    learning = [model.learning_snapshot(None)]     # boxed so the loop can swap it
    adv_hist = [None]
    hist = {k: deque(maxlen=200)
            for k in ("entropy", "kl", "v_loss", "clip_frac", "ev", "grad_norm")}
    t_start = time.time()
    level_name = env_name if args.sim else OFFICIAL_LEVELS.get(args.level, "")
    update_i = start_update
    collect = collect_vec if vectorised else collect_live

    try:
        # Inside the try on purpose. This used to sit outside it, so a stall while
        # waiting for the very first attempt killed the run before training began
        # AND skipped the finally, leaving the mod attached with the game stranded.
        st.obs = env.reset() if vectorised else _reset_with_recovery(env)

        for update_i in range(start_update, args.updates):
            buf.reset()
            t0, steps0 = time.time(), st.env_steps
            frac = (update_i - start_update) / args.anneal if args.anneal else 0.0
            lr_now = ppo.anneal(args.lr, args.lr * 0.1, frac)
            ent_now = ppo.anneal(args.entropy, args.entropy * 0.1, frac)

            ctx = {"playing": playing, "update_i": update_i, "level_name": level_name,
                   "t_start": t_start, "t0": t0, "steps0": steps0, "hist": hist,
                   "learning": learning, "adv_hist": adv_hist, "log": log}
            collect(env, model, buf, tele, st, args, ctx)

            sps = (st.env_steps - steps0) / max(1e-6, time.time() - t0)
            mret = float(np.mean(st.returns)) if st.returns else 0.0
            mpct = float(np.mean(st.percents)) if st.percents else 0.0

            if playing:         # collected, but there is nothing to learn from
                print(f"[play] episodes {st.episodes:>5}  mean% {mpct * 100:5.1f}  "
                      f"best% {st.best_ever * 100:5.1f}  return {mret:7.2f}  "
                      f"{sps:6.0f} steps/s")
                continue

            if vectorised:
                _a, _lp, last_v = model.act_batch(st.obs[0], st.obs[1], greedy=True)
                adv, ret = ppo.compute_gae_vec(buf.rewards[:buf.n], buf.values[:buf.n],
                                               buf.dones[:buf.n], last_v,
                                               gamma=args.gamma, lam=args.lam)
            else:
                _a, _lp, last_v, _ = model.act(st.obs, greedy=True)
                adv, ret = ppo.compute_gae(buf.rewards[:buf.n], buf.values[:buf.n],
                                           buf.dones[:buf.n], last_v,
                                           gamma=args.gamma, lam=args.lam)

            stats = ppo.update(model, optim, buf, adv, ret, epochs=args.epochs,
                               clip=args.clip, batch=args.minibatch,
                               ent_coef=ent_now, target_kl=args.target_kl,
                               clip_value=not args.no_value_clip, lr=lr_now,
                               collect_grads=bool(tele.viewers))

            hist["entropy"].append(stats["entropy"])
            hist["kl"].append(stats["kl"])
            hist["v_loss"].append(stats["v_loss"])
            hist["clip_frac"].append(stats["clip_frac"])
            hist["ev"].append(stats["explained_variance"])
            hist["grad_norm"].append(stats["grad_norm"])

            # Diff the weights against the pre-update copy: this is what turns the
            # viewer from "what it sees" into "what it is learning".
            if tele.viewers:
                snap = model.learning_snapshot(fingerprint)
                snap["grads"] = stats.get("grads", {})
                learning[0] = snap
                counts, edges = np.histogram(np.asarray(adv).ravel(), bins=ADV_BINS)
                adv_hist[0] = {"counts": counts.tolist(),
                               "lo": float(edges[0]), "hi": float(edges[-1])}
            fingerprint = model.weight_fingerprint()

            print(f"[{update_i:5d}] steps {st.env_steps:>9,}  eps {st.episodes:>5}  "
                  f"return {mret:7.2f}  mean% {mpct * 100:5.1f}  best% "
                  f"{st.best_ever * 100:5.1f}  ent {stats['entropy']:.3f}  "
                  f"kl {stats['kl']:.4f}  ev {stats['explained_variance']:+.2f}  "
                  f"{sps:6.0f} steps/s"
                  f"{'  [viewer]' if tele.viewers else ''}")
            log.update([round(time.time() - t_start, 1), st.env_steps, update_i,
                        round(mret, 3), round(mpct, 4), round(st.best_ever, 4),
                        round(stats["pi_loss"], 4), round(stats["v_loss"], 4),
                        round(stats["entropy"], 4), round(stats["kl"], 5),
                        round(stats["clip_frac"], 4), round(sps, 1),
                        round(stats["explained_variance"], 4),
                        round(stats["grad_norm"], 4), round(stats["lr"], 8),
                        stats["epochs"]])

            extra = {"optim": optim.state_dict(), "update": update_i + 1,
                     "env_steps": st.env_steps, "best_percent": st.best_ever,
                     "best_mean": best_mean}
            P.save(model, latest, extra)
            # Gate best.pt on the rolling mean, not one lucky episode — a single
            # good attempt says more about variance than about the policy.
            if st.percents and mpct > best_mean:
                best_mean = mpct
                P.save(model, os.path.join(run_dir, "best.pt"), extra)

    except (KeyboardInterrupt, BridgeLost, VersionMismatch) as exc:
        print(f"\nstopped: {exc}")
    finally:
        if not playing:
            P.save(model, latest, {"optim": optim.state_dict(),
                                   "update": update_i + 1, "env_steps": st.env_steps,
                                   "best_percent": st.best_ever,
                                   "best_mean": best_mean})
            print(f"saved {latest}")
        if log:
            log.close()
        tele.stop()
        env.close()


if __name__ == "__main__":
    main()

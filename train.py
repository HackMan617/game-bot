"""Train the Geometry Dash agent with PPO, or watch a trained one play.

    python train.py --sim                 # no game needed: train against the simulator
    python train.py                       # real GD, whatever level you are sitting in
    python train.py --level 1 --speed 8   # load Stereo Madness, run 8x wall-clock
    python train.py --practice            # checkpoint curriculum: grind segments
    python train.py --play runs/gd/best.pt --speed 1   # watch it, no learning

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

from gdbot import policy as P
from gdbot import ppo
from gdbot.bridge import OFFICIAL_LEVELS
from gdbot.env import BridgeLost, LiveEnv, VersionMismatch
from gdbot.obs import (CHANNEL_NAMES, GRID_SHAPE, N_SCALARS, PLAYER_COL,
                       PLAYER_ROW, SCALAR_NAMES)
from gdbot.sim_env import SimEnv
from gdbot.telemetry import Telemetry, b64

HERE = os.path.dirname(os.path.abspath(__file__))

# Consecutive bridge stalls to absorb before giving up. Generous on purpose: an
# overnight run should survive hitches, and a genuinely dead game fails fast
# anyway because recover() cannot reconnect.
MAX_STALLS = 20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", action="store_true",
                   help="train against the simulator instead of real GD")
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
    p.add_argument("--rollout", type=int, default=2048, help="steps per PPO update")
    p.add_argument("--updates", type=int, default=100000, help="PPO updates to run")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--entropy", type=float, default=0.01)
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


def pick_device(choice: str) -> str:
    if choice != "auto":
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"


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
            "steps_per_sec"])

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


def _reset_with_recovery(env, args, tries: int = MAX_STALLS):
    """env.reset(), but a stall retries instead of ending the run.

    The simulator cannot stall, so this is a no-op there.
    """
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


def _wire_learning(learning) -> dict:
    """Base64 the kernel block; the rest of the learning snapshot is plain JSON."""
    if not learning:
        return None
    k = learning["kernels"]
    return {"layers": learning["layers"],
            "kernels": {"n": k["n"], "c": k["c"], "kh": k["kh"], "kw": k["kw"],
                        "data": b64(k["data"])}}


def snapshot(obs, viz, action, info, hud, series, learning=None) -> dict:
    """Everything the viewer draws for one frame, already wire-ready.

    `learning` is the weights-and-deltas block. It changes once per PPO update,
    not once per frame, so it is computed there and simply carried along here.
    """
    return {
        "learning": _wire_learning(learning),
        "grid": b64((obs.grid > 0).astype(np.uint8).ravel()),
        "scalars": [round(float(v), 4) for v in obs.scalars],
        "probs": [round(p, 4) for p in viz["probs"]],
        "value": round(viz["value"], 3),
        "action": int(action),
        "hidden": b64(viz["hidden"]),
        "head": b64(viz["head"]),
        "stages": [{"n": s["n"], "h": s["h"], "w": s["w"], "data": b64(s["data"])}
                   for s in viz["stages"]],
        "percent": round(float(info.get("percent", 0.0)), 4),
        "hud": hud,
        "series": series,
    }


def main():
    args = parse_args()
    device = pick_device(args.device)
    run_dir = os.path.join(HERE, "runs", args.run)
    playing = bool(args.play)

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
        "mode": "play" if playing else "train",
    })
    if url:
        print(f"\n  viewer:  {url}\n")
        if args.open:
            webbrowser.open(url)

    buf = ppo.Rollout(args.rollout, GRID_SHAPE, N_SCALARS)
    returns, percents = deque(maxlen=200), deque(maxlen=200)
    ep_return, episodes = 0.0, 0
    best_mean = resumed_mean   # best rolling-mean percent so far; gates best.pt
    stalls = 0                 # bridge stalls survived this run

    # Learning-view state: the last weights fingerprint to diff against, and the
    # PPO health history the viewer sparklines. Both are cheap and only touched
    # once per update, so an unwatched run barely notices them.
    fingerprint = model.weight_fingerprint()
    learning = [model.learning_snapshot(None)]     # boxed so the loop can swap it
    hist = {k: deque(maxlen=200) for k in ("entropy", "kl", "v_loss", "clip_frac")}
    t_start = time.time()
    level_name = "simulator" if args.sim else OFFICIAL_LEVELS.get(args.level, "")
    update_i = start_update

    try:
        # Inside the try on purpose. This used to sit outside it, so a stall while
        # waiting for the very first attempt killed the run before training began
        # AND skipped the finally, leaving the mod attached with the game stranded.
        obs = _reset_with_recovery(env, args)

        for update_i in range(start_update, args.updates):
            buf.reset()
            t0, steps0 = time.time(), env_steps

            while not buf.full:
                capture = tele.should_capture()
                action, logp, value, viz = model.act(obs, greedy=playing,
                                                     introspect=capture)
                try:
                    next_obs, reward, done, info = env.step(action)
                except BridgeLost as exc:
                    # A stall is a pause, not an ending. This exact failure — one
                    # hitch past the frame timeout — killed a 7.12h run that was
                    # still improving, with the game perfectly healthy throughout.
                    stalls += 1
                    print(f"\n[stall {stalls}/{MAX_STALLS}] {exc}\n"
                          f"  recovering (re-attaching, reloading the level if needed)...")
                    if stalls > MAX_STALLS or not env.recover():
                        raise
                    # Close the episode at the discontinuity. Without this, GAE
                    # bootstraps credit straight across the gap and blends two
                    # unrelated episodes into one advantage estimate.
                    if buf.n:
                        buf.dones[buf.n - 1] = 1.0
                    ep_return = 0.0
                    try:
                        obs = _reset_with_recovery(env, args)
                    except BridgeLost:
                        continue          # stalled again mid-reset; recover again
                    print("  recovered — continuing")
                    continue
                env_steps += 1
                ep_return += reward

                # A truncated episode is not a failure, so bootstrap its tail
                # instead of teaching the policy that surviving ends the world.
                if done and info.get("timeout") and not info.get("dead"):
                    _a, _lp, v_next, _ = model.act(next_obs, greedy=True)
                    reward += args.gamma * v_next

                # Fill the buffer even when playing: it is what ends the inner
                # loop, and one preallocated row costs a memcpy.
                buf.add(obs, action, logp, value, reward, done)

                if capture:
                    elapsed = max(1e-6, time.time() - t0)
                    hud = {
                        "update": update_i, "env_steps": env_steps,
                        "episodes": episodes, "best_ever": round(best_ever * 100, 2),
                        "recent": round(float(np.mean(percents)) * 100, 2) if percents else 0.0,
                        "speed": args.speed, "step_hz": args.step_hz,
                        "level": level_name or f"level {info.get('level_id', 0)}",
                        "checkpoints": info.get("checkpoints", 0),
                        "sps": round((env_steps - steps0) / elapsed, 1),
                        "wall": int(time.time() - t_start),
                    }
                    series = {"returns": [round(r, 2) for r in returns],
                              "percents": [round(p * 100, 2) for p in percents],
                              "entropy": [round(e, 4) for e in hist["entropy"]],
                              "kl": [round(k, 5) for k in hist["kl"]],
                              "v_loss": [round(v, 4) for v in hist["v_loss"]],
                              "clip_frac": [round(c, 4) for c in hist["clip_frac"]]}
                    tele.publish(snapshot(obs, viz, action, info, hud, series,
                                          learning=learning[0]))

                obs = next_obs
                if done:
                    episodes += 1
                    bp = info.get("best_percent", 0.0)
                    returns.append(ep_return)
                    percents.append(bp)
                    best_ever = max(best_ever, bp)
                    if log:
                        log.episode([round(time.time() - t_start, 1), env_steps,
                                     episodes, round(ep_return, 3), round(bp, 4),
                                     info.get("steps", 0),
                                     int(info.get("complete", False)),
                                     int(info.get("dead", False))])
                    ep_return = 0.0
                    obs = _reset_with_recovery(env, args)

            sps = (env_steps - steps0) / max(1e-6, time.time() - t0)
            mret = float(np.mean(returns)) if returns else 0.0
            mpct = float(np.mean(percents)) if percents else 0.0

            if playing:         # collected, but there is nothing to learn from
                print(f"[play] episodes {episodes:>5}  mean% {mpct * 100:5.1f}  "
                      f"best% {best_ever * 100:5.1f}  return {mret:7.2f}  "
                      f"{sps:6.0f} steps/s")
                continue

            _a, _lp, last_value, _ = model.act(obs, greedy=True)
            adv, ret = ppo.compute_gae(buf.rewards[:buf.n], buf.values[:buf.n],
                                       buf.dones[:buf.n], last_value,
                                       gamma=args.gamma, lam=args.lam)
            stats = ppo.update(model, optim, buf, adv, ret, epochs=args.epochs,
                               ent_coef=args.entropy)

            for k in hist:
                hist[k].append(stats[k])
            # Diff the weights against the pre-update copy: this is what turns the
            # viewer from "what it sees" into "what it is learning".
            if tele.viewers:
                learning[0] = model.learning_snapshot(fingerprint)
            fingerprint = model.weight_fingerprint()

            print(f"[{update_i:5d}] steps {env_steps:>9,}  eps {episodes:>5}  "
                  f"return {mret:7.2f}  mean% {mpct * 100:5.1f}  best% "
                  f"{best_ever * 100:5.1f}  ent {stats['entropy']:.3f}  "
                  f"kl {stats['kl']:.4f}  {sps:6.0f} steps/s"
                  f"{'  [viewer]' if tele.viewers else ''}")
            log.update([round(time.time() - t_start, 1), env_steps, update_i,
                        round(mret, 3), round(mpct, 4), round(best_ever, 4),
                        round(stats["pi_loss"], 4), round(stats["v_loss"], 4),
                        round(stats["entropy"], 4), round(stats["kl"], 5),
                        round(stats["clip_frac"], 4), round(sps, 1)])

            extra = {"optim": optim.state_dict(), "update": update_i + 1,
                     "env_steps": env_steps, "best_percent": best_ever,
                     "best_mean": best_mean}
            P.save(model, latest, extra)
            # Gate best.pt on the rolling mean, not one lucky episode — a single
            # good attempt says more about variance than about the policy.
            if percents and mpct > best_mean:
                best_mean = mpct
                P.save(model, os.path.join(run_dir, "best.pt"), extra)

    except (KeyboardInterrupt, BridgeLost, VersionMismatch) as exc:
        print(f"\nstopped: {exc}")
    finally:
        if not playing:
            P.save(model, latest, {"optim": optim.state_dict(),
                                   "update": update_i + 1, "env_steps": env_steps,
                                   "best_percent": best_ever,
                                   "best_mean": best_mean})
            print(f"saved {latest}")
        if log:
            log.close()
        tele.stop()
        env.close()


if __name__ == "__main__":
    main()

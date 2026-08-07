"""End-to-end checks for the learning stack, with no game and no browser.

Everything here runs against SimEnv, so it covers what used to be testable only
by launching Geometry Dash: the observation contract both backends share, the
conv policy's shapes and introspection payload, a real PPO update, and the
telemetry channel actually delivering a frame to a websocket client.

    python tests/test_stack.py
"""

import asyncio
import base64
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdbot import policy as P                                       # noqa: E402
from gdbot import ppo                                               # noqa: E402
from gdbot.env import (COMPLETE_BONUS, DEATH_PENALTY,                # noqa: E402
                       PROGRESS_REWARD, shape_reward)
from gdbot.obs import (CHANNEL_NAMES, GRID_SHAPE, N_SCALARS,         # noqa: E402
                       PLAYER_COL, PLAYER_ROW, SCALAR_NAMES, grid_to_ascii)
from gdbot.sim_env import SimEnv, make_course                        # noqa: E402
from gdbot.telemetry import Telemetry, b64                           # noqa: E402
from gdbot.vec_env import VecSimEnv                                   # noqa: E402

_failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def close(a, b, tol=1e-5):
    return abs(a - b) <= tol


# --- observation contract ----------------------------------------------------
def test_obs_contract():
    check("channel names cover the grid", len(CHANNEL_NAMES) == GRID_SHAPE[0])
    check("scalar names cover the vector", len(SCALAR_NAMES) == N_SCALARS)

    obs = SimEnv(seed=3).reset()
    check("sim emits the shared grid shape", obs.grid.shape == GRID_SHAPE,
          str(obs.grid.shape))
    check("grid is float32 occupancy", obs.grid.dtype == np.float32
          and set(np.unique(obs.grid)) <= {0.0, 1.0})
    check("scalar vector is the declared length", obs.scalars.shape == (N_SCALARS,))

    # Unbounded inputs are the classic silent killer of a conv policy, so sweep a
    # whole run of random play and watch the extremes.
    rng = np.random.default_rng(0)
    env = SimEnv(seed=7)
    obs = env.reset()
    lo, hi = obs.scalars.copy(), obs.scalars.copy()
    for _ in range(3000):
        obs, _r, done, _i = env.step(int(rng.integers(2)))
        np.minimum(lo, obs.scalars, out=lo)
        np.maximum(hi, obs.scalars, out=hi)
        if done:
            obs = env.reset()
    check("scalars stay bounded over a full run", lo.min() >= -1.001 and hi.max() <= 2.001,
          f"min={lo.min():.3f} max={hi.max():.3f}")


def test_grid_geometry():
    """The player's cell is the anchor the whole grid is drawn around. If terrain
    leaks into it, the network is quietly looking at a shifted world."""
    env = SimEnv(seed=11)
    obs = env.reset()
    clean, worst = True, None
    for _ in range(1500):
        # Terrain at or above the player's row is normal *ahead* of it — that is
        # what a step looks like. In its own column it would mean the cube is
        # buried in the wall it is standing on.
        if obs.grid[0, PLAYER_ROW:, PLAYER_COL].sum() != 0.0:
            clean, worst = False, obs.grid
            break
        obs, _r, done, _i = env.step(0)
        if done:
            obs = env.reset()
    check("the cube's own column is clear at and above its row", clean,
          "" if clean else "\n" + grid_to_ascii(worst))

    obs = SimEnv(seed=5).reset()            # starts standing on flat ground
    check("floor sits directly under the cube",
          obs.grid[0, PLAYER_ROW - 1, PLAYER_COL] == 1.0)
    check("the ground fills every row below the cube",
          obs.grid[0, :PLAYER_ROW, PLAYER_COL].sum() == PLAYER_ROW)
    # The generator never builds anything taller than a 2-block step, so
    # anything up here means the vertical mapping has drifted.
    check("nothing floats above the tallest possible step",
          obs.grid[:, PLAYER_ROW + 4:, :].sum() == 0.0)

    env = SimEnv(make_course(seed=2))
    obs = env.reset()
    hazards = 0
    for _ in range(2000):
        hazards += int(obs.grid[1].sum() > 0)
        obs, _r, done, _i = env.step(0)
        if done:
            break
    check("spikes appear in the hazard channel", hazards > 0, f"{hazards} frames")


# --- reward ------------------------------------------------------------------
def test_reward():
    check("progress pays", close(shape_reward(0.1, False, False), PROGRESS_REWARD * 0.1))
    check("backwards motion pays nothing", shape_reward(-0.5, False, False) == 0.0)
    check("death costs", close(shape_reward(0.0, True, False), -DEATH_PENALTY))
    check("completion pays the bonus", close(shape_reward(0.0, False, True), COMPLETE_BONUS))

    env = SimEnv(make_course(seed=4))
    env.reset()
    total, info = 0.0, {}
    for _ in range(5000):
        _o, r, done, info = env.step(0)
        total += r
        if done:
            break
    expect = PROGRESS_REWARD * info["percent"] - DEATH_PENALTY * int(info["dead"])
    check("an episode's return is exactly its progress minus its death",
          close(total, expect, 0.2), f"{total:.3f} vs {expect:.3f}")


# --- policy ------------------------------------------------------------------
def test_policy():
    model = P.ConvActorCritic()
    obs = SimEnv(seed=1).reset()

    action, logp, value, viz = model.act(obs)
    check("act returns a binary action", action in (0, 1))
    check("act returns a log-prob", logp <= 0.0 and math.isfinite(logp))
    check("act returns a finite value", math.isfinite(value))
    check("introspection is opt-in, not free", viz is None)

    _a, _lp, _v, viz = model.act(obs, introspect=True)
    check("probs are a distribution",
          len(viz["probs"]) == 2 and close(sum(viz["probs"]), 1.0))
    check("hidden activations match the dense width",
          viz["hidden"].shape == (model.fc.out_features,))
    check("head weights cover both actions",
          viz["head"].shape == (2 * model.fc.out_features,))
    check("one viz stage per conv layer", len(viz["stages"]) == len(model.map_shapes))

    shapes_ok = True
    for stage, (c, h, w) in zip(viz["stages"], model.map_shapes):
        shapes_ok &= (stage["n"] == min(c, P.VIZ_MAPS) and (stage["h"], stage["w"]) == (h, w)
                      and stage["data"].size == stage["n"] * h * w
                      and stage["data"].dtype == np.uint8)
    check("feature maps are uint8 and correctly shaped", bool(shapes_ok),
          str(model.map_shapes))

    g = torch.as_tensor(obs.grid).unsqueeze(0)
    s = torch.as_tensor(obs.scalars).unsqueeze(0)
    agrees = True
    with torch.no_grad():
        for a in (0, 1):
            lp, _ent, v = model.evaluate(g, s, torch.tensor([a]))
            logits, v2, _ = model(g, s)
            agrees &= close(float(lp[0]), float(torch.log_softmax(logits, -1)[0, a]), 1e-5)
            agrees &= close(float(v[0]), float(v2[0]), 1e-5)
    check("batched evaluate agrees with a single forward", bool(agrees))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ck.pt")
        P.save(model, path, {"update": 7})
        loaded, ck = P.load(path)
        check("checkpoint keeps its metadata", ck["update"] == 7)
        check("a reloaded policy decides identically",
              model.act(obs, greedy=True)[0] == loaded.act(obs, greedy=True)[0])


# --- PPO ---------------------------------------------------------------------
def test_ppo():
    rewards = np.ones(3, dtype=np.float32)
    values = np.zeros(3, dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    adv, ret = ppo.compute_gae(rewards, values, dones, 0.0, gamma=1.0, lam=1.0)
    # Zero baseline, no discount: advantage is just reward-to-go.
    check("GAE reduces to reward-to-go", np.allclose(adv, [3, 2, 1]) and np.allclose(ret, [3, 2, 1]),
          str(adv))

    dones = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    adv, _ = ppo.compute_gae(np.ones(4, np.float32), np.zeros(4, np.float32),
                             dones, 0.0, gamma=1.0, lam=1.0)
    check("credit does not cross a terminal step",
          close(float(adv[0]), 2.0) and close(float(adv[1]), 1.0), str(adv))

    buf = ppo.Rollout(4, GRID_SHAPE, N_SCALARS)
    obs = SimEnv(seed=1).reset()
    for i in range(4):
        buf.add(obs, i % 2, -0.5, 1.0, 0.25, i == 3)
    check("rollout stores what it was given",
          buf.full and list(buf.actions) == [0, 1, 0, 1] and list(buf.dones) == [0, 0, 0, 1])
    buf.reset()
    check("rollout reset clears it", buf.n == 0 and not buf.full)

    torch.manual_seed(0)
    model = P.ConvActorCritic()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    env = SimEnv(seed=1)
    buf = ppo.Rollout(128, GRID_SHAPE, N_SCALARS)
    obs = env.reset()
    while not buf.full:
        action, logp, value, _ = model.act(obs)
        obs, reward, done, _i = env.step(action)
        buf.add(obs, action, logp, value, reward, done)
        if done:
            obs = env.reset()

    before = [p.detach().clone() for p in model.parameters()]
    adv, ret = ppo.compute_gae(buf.rewards[:buf.n], buf.values[:buf.n], buf.dones[:buf.n], 0.0)
    stats = ppo.update(model, optim, buf, adv, ret, epochs=2, batch=32)
    check("entropy is a valid two-action entropy",
          0.0 < stats["entropy"] <= math.log(2) + 1e-6, f"{stats['entropy']:.4f}")
    check("losses are finite",
          math.isfinite(stats["pi_loss"]) and math.isfinite(stats["v_loss"]),
          f"pi={stats['pi_loss']:.4f} v={stats['v_loss']:.4f}")
    check("the update actually moved the weights",
          any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))


# --- telemetry ---------------------------------------------------------------
def test_telemetry():
    tele = Telemetry(fps=1000.0)
    check("an unwatched run never pays for viz", tele.should_capture() is False)
    tele._clients = 1
    tele.enabled = False
    check("--no-viewer disables capture entirely", tele.should_capture() is False)

    tele = Telemetry(fps=20.0)
    tele._clients = 1
    first = tele.should_capture()
    second = tele.should_capture()
    time.sleep(1.0 / 20 + 0.02)
    third = tele.should_capture()
    check("capture is rate-limited to viewer fps", first and not second and third)

    arr = np.arange(256, dtype=np.uint8)
    check("base64 round-trips",
          np.array_equal(np.frombuffer(base64.b64decode(b64(arr)), dtype=np.uint8), arr))


def test_websocket():
    """The real thing: start the server, connect a client, receive a snapshot."""
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        try:
            from websockets import connect
        except ImportError:
            check("websocket delivers meta then frames", False, "websockets not installed")
            return

    tele = Telemetry(http_port=8791, ws_port=8792, fps=60.0)
    tele.start({"grid_shape": list(GRID_SHAPE), "hello": "world"})
    time.sleep(0.4)                      # let the asyncio thread bind

    result = {}

    async def client():
        async with connect("ws://127.0.0.1:8792") as ws:
            meta = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            result["meta"] = meta.get("type") == "meta" and meta.get("hello") == "world"
            for _ in range(100):         # the trainer publishes only when watched
                if tele.should_capture():
                    tele.publish({"grid": b64(np.zeros(4, np.uint8)), "n": 1})
                    break
                await asyncio.sleep(0.02)
            frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            result["frame"] = frame.get("type") == "frame" and frame.get("n") == 1

    try:
        asyncio.run(client())
    except Exception as exc:
        check("websocket delivers meta then frames", False, repr(exc))
        return
    finally:
        tele.stop()

    check("viewer receives the meta handshake", result.get("meta", False))
    check("viewer receives a published frame", result.get("frame", False))


# --- the vectorised simulator ------------------------------------------------
def test_vec_env():
    """VecSimEnv must be SimEnv, N at a time — not merely something similar.

    A vectorised environment that drifts from the scalar one is the worst kind of
    bug: nothing crashes, the policy just trains on a slightly different world
    than the one the tests and the live game describe. So this is exact equality
    on the grid, not a tolerance.
    """
    rng = np.random.default_rng(0)
    a = SimEnv(seed=1, reseed_each_episode=False)
    b = VecSimEnv(n_envs=1, seed=1, reseed_each_episode=False)
    oa = a.reset()
    (gb, sb) = b.reset()

    bad, episodes = None, 0
    if not np.array_equal(oa.grid, gb[0]):
        bad = "grid differs at t=0"
    for t in range(6000):
        if bad:
            break
        act = int(rng.integers(2))
        oa, ra, da, ia = a.step(act)
        (gb, sb), rb, db, ib = b.step(np.array([act]))
        if abs(ra - float(rb[0])) > 1e-5:
            bad = f"reward at t={t}: {ra} vs {float(rb[0])}"
        elif bool(da) != bool(db[0]):
            bad = f"done at t={t}"
        elif da:
            episodes += 1
            ep = ib["episodes"][0]
            if abs(ep["best_percent"] - ia["best_percent"]) > 1e-6:
                bad = f"episode stats at t={t}"
            # The scalar env returns the terminal frame and resets on the next
            # call; the vectorised one auto-resets and returns the fresh frame.
            # Line them up before comparing.
            oa = a.reset()
        if not bad and not np.array_equal(oa.grid, gb[0]):
            bad = f"grid at t={t}"
        if not bad and not np.allclose(oa.scalars, sb[0], atol=1e-6):
            bad = f"scalars at t={t}"
    check("vec env matches the scalar sim step for step", bad is None,
          bad or f"6000 steps, {episodes} episodes")

    # Independence: two environments on different courses must not interfere.
    v = VecSimEnv(n_envs=4, seed=1)
    g, s = v.reset()
    check("vec obs has the batch shapes the policy expects",
          g.shape == (4, *GRID_SHAPE) and s.shape == (4, N_SCALARS),
          f"{g.shape} {s.shape}")
    courses_differ = not np.array_equal(v.ground[0], v.ground[1])
    check("each environment gets its own course", courses_differ)

    # Auto-reset: a finished environment must begin a new episode in place,
    # without disturbing the ones still running.
    for _ in range(4000):
        (g, s), r, d, info = v.step(np.zeros(4, dtype=np.int64))
        if d.any():
            break
    check("a finished environment auto-resets", bool(d.any()) and
          np.all(v.percent[np.nonzero(d)[0]] < 0.05),
          f"{int(d.sum())} of 4 done")
    check("finished environments report their episode", len(info["episodes"]) == int(d.sum()))


def test_vec_ppo():
    """GAE over (T, N) must agree with the scalar recursion column by column."""
    rng = np.random.default_rng(1)
    T, N = 12, 5
    rewards = rng.normal(size=(T, N)).astype(np.float32)
    values = rng.normal(size=(T, N)).astype(np.float32)
    dones = (rng.random((T, N)) < 0.2).astype(np.float32)
    last = rng.normal(size=N).astype(np.float32)

    adv_v, ret_v = ppo.compute_gae_vec(rewards, values, dones, last)
    agrees = True
    for j in range(N):
        adv_s, ret_s = ppo.compute_gae(rewards[:, j], values[:, j], dones[:, j],
                                       float(last[j]))
        agrees &= np.allclose(adv_s, adv_v[:, j], atol=1e-5)
        agrees &= np.allclose(ret_s, ret_v[:, j], atol=1e-5)
    check("vectorised GAE agrees with the scalar recursion", bool(agrees))

    buf = ppo.VecRollout(4, 3, GRID_SHAPE, N_SCALARS)
    env = VecSimEnv(n_envs=3, seed=2)
    g, s = env.reset()
    for i in range(4):
        acts = np.array([i % 2] * 3)
        buf.add(g, s, acts, np.full(3, -0.5, np.float32), np.ones(3, np.float32),
                np.full(3, 0.25, np.float32), np.zeros(3, np.float32))
        (g, s), _r, _d, _i = env.step(acts)
    fg, fs, fa, flp, fv = buf.flat()
    check("vec rollout flattens to one batch",
          buf.full and buf.transitions == 12 and fg.shape == (12, *GRID_SHAPE)
          and fs.shape == (12, N_SCALARS) and fa.shape == (12,),
          f"{fg.shape}")

    # A real update through the vectorised buffer, exercising the same code path
    # `update()` takes for the flat one.
    torch.manual_seed(0)
    model = P.ConvActorCritic()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    buf = ppo.VecRollout(32, 4, GRID_SHAPE, N_SCALARS)
    env = VecSimEnv(n_envs=4, seed=3)
    g, s = env.reset()
    while not buf.full:
        acts, logps, vals = model.act_batch(g, s)
        (ng, ns), rew, done, _i = env.step(acts)
        buf.add(g, s, acts, logps, vals, rew, done)
        g, s = ng, ns
    adv, ret = ppo.compute_gae_vec(buf.rewards[:buf.n], buf.values[:buf.n],
                                   buf.dones[:buf.n], np.zeros(4, np.float32))
    stats = ppo.update(model, optim, buf, adv, ret, epochs=2, batch=32,
                       collect_grads=True)
    check("a vectorised update produces finite diagnostics",
          all(math.isfinite(stats[k]) for k in
              ("pi_loss", "v_loss", "entropy", "kl", "explained_variance",
               "grad_norm")),
          f"ev={stats['explained_variance']:.3f} gn={stats['grad_norm']:.3f}")
    check("k3 KL is never negative", stats["kl"] >= 0.0, f"{stats['kl']:.6f}")
    check("per-layer gradients are reported when asked",
          set(stats["grads"]) == {n for n, _ in model.named_layers()},
          str(sorted(stats["grads"])))

    ev = ppo.explained_variance(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    check("explained variance is 1 for a perfect critic", close(ev, 1.0))
    ev0 = ppo.explained_variance(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    check("explained variance is 0 for a mean-predicting critic", close(ev0, 0.0))


def test_batched_policy():
    """act_batch and saliency — the two things the vectorised viewer depends on."""
    model = P.ConvActorCritic()
    env = VecSimEnv(n_envs=6, seed=4)
    g, s = env.reset()

    acts, logps, vals = model.act_batch(g, s)
    check("act_batch returns one decision per environment",
          acts.shape == (6,) and logps.shape == (6,) and vals.shape == (6,)
          and set(np.unique(acts)) <= {0, 1}, str(acts.shape))

    a_g, lp_g, _v = model.act_batch(g, s, greedy=True)
    a_g2, _lp, _v2 = model.act_batch(g, s, greedy=True)
    check("greedy act_batch is deterministic", np.array_equal(a_g, a_g2))

    with torch.no_grad():
        logits, value, _ = model(torch.as_tensor(g), torch.as_tensor(s))
    check("act_batch values agree with a direct forward",
          np.allclose(vals, value.numpy(), atol=1e-5))
    check("greedy act_batch takes the argmax",
          np.array_equal(a_g, logits.argmax(-1).numpy()))

    sal = model.saliency(g[0], s[0], int(acts[0]))
    C, H, W = GRID_SHAPE
    check("saliency is one uint8 magnitude per grid cell",
          sal.shape == (H, W) and sal.dtype == np.uint8 and sal.max() == 255,
          f"{sal.shape} max={sal.max()}")

    # Saliency must not leave gradients on the parameters for the next PPO
    # update to pick up — that would silently corrupt the very first minibatch.
    dirty = [n for n, p in model.named_parameters() if p.grad is not None]
    check("saliency leaves no gradient on the parameters", not dirty, str(dirty))

    viz = model.viz_one(g[0], s[0])
    check("viz_one packs the same payload as introspection",
          len(viz["probs"]) == 2 and close(sum(viz["probs"]), 1.0)
          and len(viz["stages"]) == len(model.map_shapes))

    # A narrower checkpoint (pre prev_action/air_time) must still run against the
    # current 19-scalar observation rather than dying on a shape mismatch.
    narrow = P.ConvActorCritic(n_scalars=N_SCALARS - 2)
    a2, _lp2, _v2, _viz2 = narrow.act(SimEnv(seed=1).reset())
    check("an older, narrower checkpoint still decides", a2 in (0, 1))


def main():
    for fn in (test_obs_contract, test_grid_geometry, test_reward, test_policy,
               test_batched_policy, test_ppo, test_vec_env, test_vec_ppo,
               test_telemetry, test_websocket):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        return 1
    print("all stack tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

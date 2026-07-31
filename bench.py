"""Benchmarks for the gdbot stack.

    python bench.py components     # where the per-decision budget goes (~2 min)
    python bench.py sweep          # hyperparameter sweep in the simulator (~35 min)
    python bench.py live           # throughput/latency against real GD (needs a level)

`components` also verifies the two claims the architecture rests on: that an
unwatched run pays nothing for visualisation, and that inference is small next to
the environment. `sweep` runs the *real trainer* as a subprocess per config, so it
measures the shipping code path rather than a reimplementation of it.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gdbot import policy as P                                        # noqa: E402
from gdbot import ppo                                                # noqa: E402
from gdbot.obs import GRID_SHAPE, N_SCALARS                          # noqa: E402
from gdbot.sim_env import SimEnv                                     # noqa: E402
from gdbot.telemetry import Telemetry, b64                           # noqa: E402


# --- helpers ------------------------------------------------------------------
def timeit(fn, n, warmup=None):
    """Median, p99 and mean of `fn`, in microseconds."""
    for _ in range(warmup if warmup is not None else max(10, n // 10)):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e6)
    ts.sort()
    return statistics.median(ts), ts[int(len(ts) * 0.99)], sum(ts) / len(ts)


def row(name, med, p99, mean, extra=""):
    print(f"  {name:<34} {med:>9.1f} {p99:>9.1f} {mean:>9.1f}   {extra}")


def header(title):
    print(f"\n{title}")
    print(f"  {'':<34} {'p50 us':>9} {'p99 us':>9} {'mean us':>9}")


# --- components ---------------------------------------------------------------
def bench_components():
    # Deliberately NOT overriding the thread count: train.py doesn't either, and
    # forcing all 16 made act() 2.3x slower here than the trainer actually runs.
    # Section 7 measures that effect rather than hiding it.
    print("=" * 78)
    print("gdbot component benchmarks")
    print("=" * 78)
    print(f"torch {torch.__version__}   cuda={torch.cuda.is_available()}   "
          f"threads={torch.get_num_threads()} (torch default)   cpus={os.cpu_count()}")

    # 1. environment
    header("1. Environment (SimEnv)")
    env = SimEnv(seed=1, reseed_each_episode=True)
    env.reset()
    rng = np.random.default_rng(0)

    def env_step():
        _o, _r, d, _i = env.step(1 if rng.random() < 0.1 else 0)
        if d:
            env.reset()

    med, p99, mean = timeit(env_step, 20000)
    row("step() incl. grid rasterise", med, p99, mean, f"{1e6 / mean:,.0f}/s ceiling")
    env.reset()
    m, p, mg = timeit(lambda: env._grid(), 20000)
    row("  of which _grid()", m, p, mg, f"{mg / mean * 100:.0f}% of step")
    m, p, mo = timeit(lambda: env._obs(), 20000)
    row("  of which _obs()", m, p, mo)

    # 2. policy
    header("2. Policy inference (CPU)")
    model = P.ConvActorCritic()
    model.eval()
    obs = env.reset()
    m, p, mean_a = timeit(lambda: model.act(obs), 3000)
    row("act() sample", m, p, mean_a, f"{1e6 / mean_a:,.0f}/s ceiling")
    m, p, mn = timeit(lambda: model.act(obs, greedy=True), 3000)
    row("act() greedy", m, p, mn)
    m, p, mean_i = timeit(lambda: model.act(obs, introspect=True), 2000)
    row("act(introspect=True)", m, p, mean_i, f"{mean_i / mean_a:.1f}x act()")

    g = torch.as_tensor(obs.grid).unsqueeze(0)
    s = torch.as_tensor(obs.scalars).unsqueeze(0)
    with torch.no_grad():
        m, p, mean_f = timeit(lambda: model(g, s), 3000)
    row("raw forward()", m, p, mean_f,
        f"{(mean_a - mean_f) / mean_a * 100:.0f}% of act() is overhead")

    fp = model.weight_fingerprint()
    m, p, mean_l = timeit(lambda: model.learning_snapshot(fp), 500)
    row("learning_snapshot() (per update)", m, p, mean_l, "not per step")

    print(f"\n  parameters: {sum(q.numel() for q in model.parameters()):,}   "
          f"conv out {model.conv_out}   dense {model.fc.out_features}")

    header("3. Batched inference (CPU)")
    for bs in (1, 8, 32, 128):
        gb, sb = torch.zeros(bs, *GRID_SHAPE), torch.zeros(bs, N_SCALARS)
        with torch.no_grad():
            m, p, mn = timeit(lambda: model(gb, sb), 400)
        row(f"batch {bs:>3}", m, p, mn, f"{bs * 1e6 / mn:,.0f} obs/s")

    # 4. GPU
    if torch.cuda.is_available():
        header("4. Policy inference (CUDA)")
        gm = P.ConvActorCritic().cuda().eval()
        torch.cuda.synchronize()

        def cuda_act():
            gm.act(obs)
            torch.cuda.synchronize()

        m, p, mn = timeit(cuda_act, 800)
        row("act() single obs", m, p, mn, f"{mean_a / mn:.2f}x vs CPU")
        for bs in (32, 128, 512):
            gb = torch.zeros(bs, *GRID_SHAPE, device="cuda")
            sb = torch.zeros(bs, N_SCALARS, device="cuda")

            def fwd():
                with torch.no_grad():
                    gm(gb, sb)
                torch.cuda.synchronize()

            m, p, mn = timeit(fwd, 250)
            row(f"batch {bs:>3}", m, p, mn, f"{bs * 1e6 / mn:,.0f} obs/s")
        print(f"\n  device: {torch.cuda.get_device_name(0)}")
    else:
        print("\n4. CUDA not available — skipped")

    # 5. telemetry: is an unwatched run really free?
    header("5. Telemetry overhead per step")
    tele = Telemetry(enabled=True, fps=20.0)
    m, p, mean_n = timeit(tele.should_capture, 200000)
    row("should_capture(), no viewer", m, p, mean_n,
        f"{mean_n / mean * 100:.4f}% of an env step")
    tele._clients = 1
    m, p, mean_c = timeit(tele.should_capture, 200000)
    row("should_capture(), viewer on", m, p, mean_c, "(mostly rate-limited)")

    _a, _lp, _v, viz = model.act(obs, introspect=True)
    payload = {
        "grid": b64((obs.grid > 0).astype(np.uint8).ravel()),
        "scalars": [float(x) for x in obs.scalars], "probs": viz["probs"],
        "value": viz["value"], "action": 0, "hidden": b64(viz["hidden"]),
        "head": b64(viz["head"]),
        "stages": [{"n": t["n"], "h": t["h"], "w": t["w"], "data": b64(t["data"])}
                   for t in viz["stages"]],
        "percent": 0.5, "hud": {}, "series": {"returns": [1.0] * 200},
    }
    m, p, mean_p = timeit(lambda: tele.publish(payload), 2000)
    row("publish() a full snapshot", m, p, mean_p)
    size = len(json.dumps({"type": "frame", **payload}))
    print(f"\n  snapshot {size / 1024:.1f} KB -> {size * 20 / 1024:.0f} KB/s at 20 fps")
    watched = mean_c + (mean_i - mean_a + mean_p) / (1e6 / mean / 20)
    print(f"  amortised while watched: ~{watched:.2f} us/step "
          f"({watched / mean * 100:.2f}% of an env step)")
    tele.stop()

    # 6. PPO update
    header("6. PPO update (ms, not us)")
    for size in (512, 1024, 2048, 4096):
        buf = ppo.Rollout(size, GRID_SHAPE, N_SCALARS)
        e2, m2 = SimEnv(seed=2), P.ConvActorCritic()
        opt = torch.optim.Adam(m2.parameters(), lr=3e-4)
        o = e2.reset()
        while not buf.full:
            a, lp, v, _ = m2.act(o)
            o, r, d, _ = e2.step(a)
            buf.add(o, a, lp, v, r, d)
            if d:
                o = e2.reset()
        adv, ret = ppo.compute_gae(buf.rewards[:buf.n], buf.values[:buf.n],
                                   buf.dones[:buf.n], 0.0)
        t = time.perf_counter()
        ppo.update(m2, opt, buf, adv, ret, epochs=4, batch=256)
        dt = (time.perf_counter() - t) * 1000
        collect = size * (mean + mean_a) / 1000
        print(f"  rollout {size:>5}: update {dt:>7.1f} ms   collect ~{collect:>7.0f} ms"
              f"   update = {dt / (dt + collect) * 100:>4.1f}% of the cycle")

    # 7. threads. This net is tiny, so torch's intra-op parallelism can cost more
    # in synchronisation than it saves — worth knowing before setting OMP_NUM_THREADS.
    header("7. Torch thread count (this net is small enough for it to matter)")
    default_threads = torch.get_num_threads()
    m3 = P.ConvActorCritic()
    m3.eval()
    o3 = SimEnv(seed=3).reset()
    buf3 = ppo.Rollout(1024, GRID_SHAPE, N_SCALARS)
    e3 = SimEnv(seed=3)
    ob = e3.reset()
    while not buf3.full:
        a, lp, v, _ = m3.act(ob)
        ob, r, d, _ = e3.step(a)
        buf3.add(ob, a, lp, v, r, d)
        if d:
            ob = e3.reset()
    adv3, ret3 = ppo.compute_gae(buf3.rewards[:buf3.n], buf3.values[:buf3.n],
                                 buf3.dones[:buf3.n], 0.0)
    for nt in (1, 2, 4, 8, 16):
        if nt > (os.cpu_count() or 1):
            continue
        torch.set_num_threads(nt)
        m, p, mn = timeit(lambda: m3.act(o3), 800)
        opt3 = torch.optim.Adam(m3.parameters(), lr=1e-5)
        t = time.perf_counter()
        ppo.update(m3, opt3, buf3, adv3, ret3, epochs=2, batch=256)
        upd = (time.perf_counter() - t) * 1000
        mark = "  <- default" if nt == default_threads else ""
        row(f"threads {nt:>2}: act()", m, p, mn,
            f"{1e6 / mn:>6,.0f}/s   update(1024,2ep) {upd:>6.0f} ms{mark}")
    torch.set_num_threads(default_threads)
    print("\n" + "=" * 78)


# --- sweep --------------------------------------------------------------------
# One axis at a time off a shared baseline. Full grid search would be 3^6 runs for
# very little more information than this, and the simulator is only a prior for
# the live game anyway.
BASE = dict(lr=3e-4, entropy=0.01, rollout=2048, epochs=4, gamma=0.99, hidden=256)
AXES = {
    "lr":      [1e-4, 3e-4, 1e-3],
    "entropy": [0.0, 0.005, 0.01, 0.02],
    "rollout": [512, 1024, 2048, 4096],
    "epochs":  [2, 4, 8],
    "gamma":   [0.97, 0.99, 0.995],
    "hidden":  [128, 256, 512],
}


def sweep_configs():
    seen, out = set(), []
    for axis, values in AXES.items():
        for v in values:
            cfg = dict(BASE)
            cfg[axis] = v
            key = tuple(sorted(cfg.items()))
            if key in seen:
                continue
            seen.add(key)
            name = "baseline" if cfg == BASE else f"{axis}-{v}"
            out.append((name, axis, cfg))
    return out


def score_run(run_dir, tail=50):
    """Read a finished run's updates.csv back and score its last `tail` updates."""
    path = os.path.join(run_dir, "updates.csv")
    rows = []
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                float(r["mean_percent"]), float(r["entropy"])
                rows.append(r)
            except (ValueError, KeyError, TypeError):
                continue
    if not rows:
        return None
    last = rows[-tail:]
    return {
        "updates": len(rows),
        "mean_pct": sum(float(r["mean_percent"]) for r in last) / len(last) * 100,
        "best_pct": max(float(r["best_percent"]) for r in rows) * 100,
        "entropy": sum(float(r["entropy"]) for r in last) / len(last),
        "sps": sum(float(r["steps_per_sec"]) for r in last) / len(last),
    }


def completion_rate(run_dir):
    path = os.path.join(run_dir, "episodes.csv")
    if not os.path.exists(path):
        return 0.0
    n = c = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                c += int(r["complete"])
                n += 1
            except (ValueError, KeyError, TypeError):
                continue
    return 100.0 * c / max(1, n)


def bench_sweep(steps, jobs, tag):
    cfgs = sweep_configs()
    print("=" * 88)
    print(f"gdbot hyperparameter sweep — {len(cfgs)} configs x {steps:,} steps, "
          f"{jobs} at a time")
    print(f"baseline: {BASE}")
    print("=" * 88)

    out_root = os.path.join(HERE, "runs")
    queue, running, results = list(cfgs), [], {}
    t0 = time.time()

    def launch(name, cfg):
        run = f"{tag}-{name}"
        d = os.path.join(out_root, run)
        if os.path.isdir(d):     # a stale run would resume and poison the result
            import shutil
            shutil.rmtree(d, ignore_errors=True)
        cmd = [sys.executable, "train.py", "--sim", "--no-viewer", "--device", "cpu",
               "--run", run, "--seed", "1",
               "--lr", str(cfg["lr"]), "--entropy", str(cfg["entropy"]),
               "--rollout", str(cfg["rollout"]), "--epochs", str(cfg["epochs"]),
               "--gamma", str(cfg["gamma"]), "--hidden", str(cfg["hidden"]),
               "--updates", str(max(1, steps // cfg["rollout"]))]
        # 4 threads x 3 jobs = 12 of 16 cores. Section 7 of `components` shows this
        # net loses badly to oversubscription, so leave headroom rather than
        # letting three jobs each grab every core.
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        p = subprocess.Popen(cmd, cwd=HERE, env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return {"name": name, "proc": p, "dir": d, "cfg": cfg}

    while queue or running:
        while queue and len(running) < jobs:
            name, _axis, cfg = queue.pop(0)
            running.append(launch(name, cfg))
            print(f"  [{len(results) + len(running)}/{len(cfgs)}] started {name}")
        time.sleep(2.0)
        for job in running[:]:
            if job["proc"].poll() is None:
                continue
            running.remove(job)
            err = job["proc"].stderr.read().decode("utf8", "replace")[-300:]
            s = score_run(job["dir"])
            if s is None:
                print(f"  ! {job['name']} produced no data. {err.strip()[:200]}")
                continue
            s["complete_pct"] = completion_rate(job["dir"])
            results[job["name"]] = (job["cfg"], s)
            print(f"  = {job['name']:<16} mean {s['mean_pct']:>5.1f}%  "
                  f"complete {s['complete_pct']:>5.1f}%  ent {s['entropy']:.3f}")

    print(f"\nswept {len(results)} configs in {(time.time() - t0) / 60:.1f} min\n")
    print(f"{'axis':<9} {'config':<16} {'mean%':>7} {'complete%':>10} "
          f"{'best%':>7} {'entropy':>8} {'steps/s':>9}")
    print("-" * 88)
    for axis in AXES:
        for name, _a, cfg in cfgs:
            if _a != axis or name not in results:
                continue
            _c, s = results[name]
            mark = " *" if name == "baseline" else "  "
            print(f"{axis:<9} {name:<16}{mark}{s['mean_pct']:>5.1f} "
                  f"{s['complete_pct']:>10.1f} {s['best_pct']:>7.1f} "
                  f"{s['entropy']:>8.3f} {s['sps']:>9.0f}")
    print("-" * 88)
    print("* baseline. Ranked by mean% over the last 50 updates.")
    if results:
        best = max(results.items(), key=lambda kv: kv[1][1]["mean_pct"])
        print(f"\nbest: {best[0]} -> {best[1][1]['mean_pct']:.1f}% mean, "
              f"{best[1][1]['complete_pct']:.1f}% completion")
    print("\nNOTE: the simulator is markedly easier than the real game and has no "
          "orbs,\nportals or non-cube modes. Treat this as a prior, not proof.")

    with open(os.path.join(out_root, f"{tag}-results.json"), "w") as f:
        json.dump({k: {"cfg": v[0], "score": v[1]} for k, v in results.items()}, f,
                  indent=2)
    print(f"wrote runs/{tag}-results.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["components", "sweep", "live"])
    ap.add_argument("--steps", type=int, default=250000, help="steps per sweep config")
    ap.add_argument("--jobs", type=int, default=3, help="concurrent sweep runs")
    ap.add_argument("--tag", default="sweep", help="run-name prefix for sweep output")
    args = ap.parse_args()

    if args.mode == "components":
        bench_components()
    elif args.mode == "sweep":
        bench_sweep(args.steps, args.jobs, args.tag)
    else:
        from gdbot.bridge import _bench
        _bench()


if __name__ == "__main__":
    main()

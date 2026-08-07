"""The experiments docs/mathematical-report.md cites, so its numbers are checkable.

    python experiments.py parity    # vectorised vs scalar rollouts, equal samples
    python experiments.py lam       # GAE lambda — tests the credit-horizon argument
    python experiments.py reward    # which reward term is actually doing the work

Each experiment shells out to `train.py` rather than reimplementing the loop, so
what gets measured is the trainer people actually run. Results come back as a
mean over the tail of each run's `episodes.csv`, with two error estimates:

* **across seeds** — the honest one, but with only a handful of seeds it is a
  wide interval and should be read as such;
* **moving-block bootstrap** within a run — because consecutive episodes are
  strongly correlated (measured lag-20 autocorrelation ~0.6 on real runs), the
  naive standard error of an N-episode mean understates the true uncertainty by
  roughly a factor of two. The block bootstrap resamples contiguous blocks, so
  the correlation survives resampling instead of being averaged away.

Nothing here is a substitute for more seeds. It is a way of not fooling yourself
about how much a 5-point gap between two single runs means.
"""

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

# Enough samples that the tail is past the initial climb, short enough that a
# whole experiment finishes while you are still interested in it.
DEFAULT_UPDATES = 120
DEFAULT_SEEDS = (1, 2, 3)
TAIL_EPISODES = 400
BLOCK = 25              # bootstrap block length, in episodes


def run_one(name: str, seed: int, updates: int, extra: list) -> str:
    """Train one config from scratch and return its run directory."""
    run_dir = os.path.join(RUNS, name)
    shutil.rmtree(run_dir, ignore_errors=True)     # never resume: latest.pt would
    cmd = [sys.executable, os.path.join(HERE, "train.py"), "--sim",
           "--run", name, "--seed", str(seed), "--updates", str(updates),
           "--no-viewer"] + extra
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise SystemExit(f"{name} failed")
    print(f"    {name:28s} {time.time() - t0:6.1f}s")
    return run_dir


def tail_percents(run_dir: str, tail: int = TAIL_EPISODES) -> np.ndarray:
    path = os.path.join(run_dir, "episodes.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["best_percent"]) for r in rows[-tail:]]) * 100.0


def block_bootstrap(x: np.ndarray, block: int = BLOCK, draws: int = 4000) -> tuple:
    """95% CI for the mean of an autocorrelated series.

    Resampling single episodes would assume independence we do not have; drawing
    contiguous blocks keeps the local correlation structure intact.
    """
    n = len(x)
    if n < block * 2:
        return float(x.mean()), float(x.mean())
    starts_n = int(np.ceil(n / block))
    rng = np.random.default_rng(0)
    means = np.empty(draws)
    for i in range(draws):
        starts = rng.integers(0, n - block, size=starts_n)
        means[i] = np.concatenate([x[s:s + block] for s in starts])[:n].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(title: str, results: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"{'config':22s} {'mean%':>7s} {'seed spread':>16s} {'block-boot 95% CI':>22s}")
    for name, per_seed in results.items():
        pooled = np.concatenate(list(per_seed.values()))
        mean = pooled.mean()
        seed_means = [v.mean() for v in per_seed.values()]
        spread = (f"{min(seed_means):.1f}–{max(seed_means):.1f}"
                  if len(seed_means) > 1 else "n=1")
        lo, hi = block_bootstrap(pooled)
        print(f"{name:22s} {mean:7.1f} {spread:>16s} {f'[{lo:.1f}, {hi:.1f}]':>22s}")
    print("\nOverlapping intervals mean the experiment did not separate those "
          "configs — not that they are equal.")


def sweep(title: str, configs: dict, seeds, updates: int) -> None:
    results = {}
    for name, extra in configs.items():
        print(f"  {name}")
        results[name] = {}
        for seed in seeds:
            run_dir = run_one(f"x-{name}-s{seed}", seed, updates, extra)
            results[name][seed] = tail_percents(run_dir)
            shutil.rmtree(run_dir, ignore_errors=True)
    report(title, results)


def exp_parity(args):
    """Does batching the rollout cost anything in learning, at equal samples?

    --rollout is a transition budget, so both arms see 2048 fresh transitions per
    update. If the vectorised arm is worse, the shorter per-environment segments
    are hurting GAE; if it matches, the speedup is free.
    """
    sweep("Vectorised vs scalar rollouts (2048 transitions/update either way)", {
        "scalar-1env": ["--envs", "1", "--rollout", "2048"],
        "vec-8env": ["--envs", "8", "--rollout", "2048"],
        "vec-32env": ["--envs", "32", "--rollout", "2048"],
    }, args.seeds, args.updates)


def exp_lam(args):
    """The credit-horizon prediction.

    GAE weights the k-th TD residual by (gamma*lam)^k, so it has an effective
    horizon of 1/(1 - gamma*lam) steps: 16.8 at the defaults. One jump arc is
    ~4.5 blocks, which at 10.3 blocks/s and 60 steps/s is ~26 steps — longer than
    the horizon. If that argument is right, raising lam should help.

    Every arm runs on the single-environment rollout. The first version of this
    ran on `--envs 8` and every arm landed within a point of 14%, because that
    backend barely learns at this budget (see `parity`) — an experiment with no
    dynamic range cannot resolve anything, and reporting it as "no effect" would
    have been wrong.
    """
    sweep("GAE lambda (gamma = 0.99)", {
        "lam-0.90": ["--envs", "1", "--lam", "0.90"],
        "lam-0.95": ["--envs", "1", "--lam", "0.95"],
        "lam-0.97": ["--envs", "1", "--lam", "0.97"],
        "lam-0.99": ["--envs", "1", "--lam", "0.99"],
    }, args.seeds, args.updates)


def exp_reward(args):
    """Which term of the reward is carrying the run?

    The player auto-scrolls at a fixed speed, so progress is an affine function
    of survival time and the progress reward and the death penalty encode the
    same preference. This asks whether either alone trains as well as both.
    """
    sweep("Reward ablation", {
        "both": ["--envs", "1"],
        "progress-only": ["--envs", "1", "--death-penalty", "0"],
        "death-only": ["--envs", "1", "--progress-reward", "0"],
    }, args.seeds, args.updates)


EXPERIMENTS = {"parity": exp_parity, "lam": exp_lam, "reward": exp_reward}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=sorted(EXPERIMENTS) + ["all"])
    p.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = p.parse_args()

    names = sorted(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in names:
        EXPERIMENTS[name](args)


if __name__ == "__main__":
    main()

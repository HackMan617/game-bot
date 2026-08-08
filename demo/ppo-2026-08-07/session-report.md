# gdbot — session report, 2026-08-07

First live run of the post-refactor stack: the 19-scalar observation, the
per-epoch KL brake, the k3 estimator, the clipped value loss, and λ = 0.99 taken
from section 4.3 of `docs/mathematical-report.md`. The point was to find out
whether a pile of individually-justified corrections adds up to a better agent on
the real game.

**Headline: it does not — and that is the result.** Against the 2026-07-30 run at
equal samples, the new stack is statistically indistinguishable: best 19.6% vs
19.5%, final-200 mean 16.1% vs 15.5%, bootstrap intervals overlapping across most
of their width. What the run did produce is a much sharper picture of *what is
actually stopping the agent*, and it is not the learning algorithm.

**2.24 h · 432 128 steps · 211 updates · 732 attempts · 0 completions · GD 2.2081 / Geode 5.8.2**

The run was cut short of its 350-update budget by the mod hanging the game — the
third occurrence of a known issue, discussed in section 5.

---

## 1. What this run was testing

Everything below landed after the last live run, so none of it had ever met the
real game:

| change | expected effect | source |
|---|---|---|
| `prev_action`, `air_time` in the observation | disambiguates a fresh click from a held one, which GD treats differently | commit 100a36c |
| per-epoch KL brake | old code averaged since the update began, engaging at 1.6× the configured target at 4 epochs | maths report §6 |
| k3 KL estimator | SE at the 0.03 threshold: k1 51-83%, k3 1-35% | maths report §6 |
| clipped value loss, explained variance | critic diagnostics that did not previously exist | commit 100a36c |
| **λ 0.95 → 0.99** | GAE horizon 16.8 → 50 steps; one cube jump is 26 steps, so 0.95 cannot span the jump it is grading | maths report §4.3 |

Configuration: `--level 1 --run live-v2 --lam 0.99 --updates 350`, everything else
at defaults (lr 3e-4, γ 0.99, entropy 0.01, clip 0.2, target-KL 0.03, 4 epochs,
rollout 2048, minibatch 256, hidden 256, CPU).

---

## 2. The comparison

Both runs are Stereo Madness from scratch. The baseline is truncated to this
run's sample count, because it had 3.3× the wall-clock budget. Intervals are
moving-block bootstrap (block 25) over the final 200 attempts, which respects the
lag-20 ≈ 0.63 episode autocorrelation instead of assuming independence.

| | live-v2 (new) | 2026-07-30 @ 432k steps | 2026-07-30 (full run) |
|---|---|---|---|
| env steps | 431 732 | 431 003 | 1 431 757 |
| attempts | 732 | 846 | 1 783 |
| PPO updates | **211** | **424** | 1 400 |
| best attempt | 19.60% | 19.50% | 52.41% |
| final-200 mean | 16.1% | 15.5% | 23.8% |
| 95% CI | [15.0, 16.8] | [13.9, 16.6] | [18.8, 28.5] |
| completions | 0 | 0 | 0 |

The intervals overlap over most of their width. A +0.6 pp difference in mean
against a between-seed SD of ~7 pp from the earlier λ sweep is not a measurement
of anything; one seed per arm cannot separate configurations this close, and the
maths report already established it would take ~120 seeds to try.

An early read at 52k steps showed live-v2 ahead by +3.5 pp with non-overlapping
intervals. It evaporated completely by 306k. Recording it here because it is a
clean example of why a mid-run lead on this task is not evidence.

### The one asymmetry worth keeping

live-v2 matched the baseline using **211 gradient updates against its 424**, because
the rollout default moved 1024 → 2048 between the runs. Same outcome at half the
gradient steps runs against the sweep's finding that everything buying more
gradient steps per sample wins. This is confounded — three things differ between
the runs, not one — so it is a hypothesis, not a result. It is a cheap one to
test properly in sim.

### Trajectory

Tenths of each run, matched samples:

| tenth | live-v2 mean% | live-v2 best% | baseline mean% | baseline best% |
|---|---|---|---|---|
| 1 | 3.42 | 8.96 | 3.50 | 8.86 |
| 2 | 4.50 | 10.83 | 3.70 | 5.93 |
| 3 | 5.00 | 14.21 | 4.27 | 10.79 |
| 4 | 8.89 | 14.26 | 5.69 | 12.44 |
| 5 | 12.74 | 19.41 | 7.39 | 12.44 |
| 6 | 15.99 | 19.49 | 11.74 | 19.41 |
| 7 | 16.77 | 19.49 | 15.66 | 19.47 |
| 8 | 15.64 | 19.47 | 16.91 | 19.54 |
| 9 | 16.79 | 19.49 | 14.38 | 19.50 |
| 10 | 15.92 | 19.60 | 15.97 | 19.49 |

live-v2 reaches the plateau roughly one tenth earlier and then sits on it for
half the run. Both end in the same place.

---

## 3. The wall is one obstacle, and it is the whole story

The two runs agree on where the agent dies to a resolution of half a percent:

| band | live-v2 | baseline (full) | block |
|---|---|---|---|
| 17.0-17.5% | 33 | 33 | 151 |
| 18.0-18.5% | 11 | 41 | 160 |
| **19.0-19.5%** | **193** | **433** | **169** |
| 19.5-20.0% | 1 | 24 | 174 |
| 20.0-27.0% | **0** | 189 | 178-236 |

**26% of every attempt this run made ended inside a single 0.5 pp band at block
169**, and nothing at all got past 19.6%. Two independently-built observation
vectors, two credit horizons differing by 3×, and two policies converging on the
same half-percent of level is not a property of either learner. It is one
obstacle.

The baseline's distribution shows what lies beyond: a scatter through 20-25% and
then a second pile-up of 104 attempts at 26.5-27.0% (block 236). The staircase is
real; this run simply never got up the first step, and the baseline needed
roughly 1M more samples than we had to do it.

The best-ever series makes the same point. Fifteen records in 732 attempts, and
the last meaningful one lands at 138k steps:

```
steps   128,423   best 19.41%
steps   138,162   best 19.49%
steps   380,892   best 19.60%
```

242 000 steps — 56% of the run — bought 0.11 pp. The mean kept improving while
the ceiling did not, which is the signature of a policy consolidating a route it
already has rather than searching for a new one.

---

## 4. PPO was healthy the whole time, and that is the problem

By decile:

| update | entropy | KL | clip_frac | expl. var | grad_norm |
|---|---|---|---|---|---|
| 0 | 0.598 | 0.0026 | 0.022 | 0.39 | 0.41 |
| 63 | 0.432 | 0.0041 | 0.047 | 0.74 | 0.37 |
| 126 | 0.492 | 0.0034 | 0.037 | 0.76 | 0.34 |
| 189 | 0.503 | 0.0038 | 0.043 | 0.77 | 0.35 |
| 210 | 0.492 | 0.0022 | 0.025 | 0.51 | 0.37 |

Nothing here is broken. Explained variance climbs to 0.85 — the critic predicts
returns well. Gradient norms are flat. Entropy settles at ~0.49 against ln 2 =
0.693 and never collapses, so the policy stays stochastic.

The diagnostic that matters:

> **The KL early-stop engaged on 0 of 211 updates.** Median KL 0.0034 against a
> configured target of 0.03 — a 9× headroom that was never touched.

The trust region is not binding, so the corrected KL brake — the fix this run was
partly built to validate — had nothing to do. Every update ran its full 4 epochs.
What bounds the step size here is the learning rate, not the trust region, and
3e-4 leaves an order of magnitude on the table. That is consistent with the
sweep's preference for lr 1e-3 and gives it a mechanism rather than just a
ranking.

Against the baseline at matched samples, the per-epoch fix shows up only in the tail:

| | median KL | p90 | max |
|---|---|---|---|
| live-v2 (per-epoch brake) | 0.0034 | 0.0051 | 0.0093 |
| baseline (cumulative brake) | 0.0036 | 0.0079 | 0.0217 |

Identical at the median, 2.3× hotter at the maximum. The direction matches the
predicted 1.6× inflation at 4 epochs, but with λ and rollout also differing this
is corroboration, not measurement.

---

## 5. The mod hung the game again

The run ended at update 210 when the trainer reported `no physics frame for 30s`
and entered its recovery loop. State at that moment:

- `GeometryDash.exe` alive, `Responding: True`, 6.6 CPU-hours accumulated over a
  2.2-hour run
- `state_seq` frozen — `CCScheduler::update` had stopped
- `attached = 1`, so the in-mod watchdog could not fire, exactly as before

This is the third occurrence, and the second and third both happened at
`--speed 4`; it has still never been seen at `--speed 1`. Recovery needs a GD
restart — the trainer's own stall recovery cannot help, because there is nothing
alive on the other side to re-attach to.

No data was lost. `latest.pt` is written every update and both CSVs are flushed
every update, so the archive is complete through update 210.

Worth noting that `--speed 4` buys nothing but risk on this workload: measured
throughput was 54.9 steps/s median against the mod's hard 60-step/s ceiling. The
next live run should use `--speed 1` and accept the ~40% sample cost, or the hang
should be fixed first.

---

## 6. What this changes

1. **Stop tuning the learner for this wall.** Two architectures, two credit
   horizons, same half-percent of level. The 19.0-19.5% band at block 169 is a
   representation-or-exploration problem, not an optimisation one.
2. **`--practice` is now the obvious next experiment.** The segment curriculum
   was built for exactly this and has still never been tested live. It targets
   the wall directly instead of hoping a better gradient estimator walks through it.
3. **Raise the learning rate.** A 9× unused KL headroom is the clearest actionable
   number in the run, and it explains the sweep's lr 1e-3 preference mechanically.
4. **λ = 0.99 is neither confirmed nor refuted.** It cost nothing and the run
   reached the plateau marginally sooner, but this design cannot separate a 0.6 pp
   effect. It stays on for its dispersion argument, not for a demonstrated gain.
5. **Fix the hang or drop to `--speed 1`.** At 54.9 of a possible 60 steps/s,
   speed 4 is buying almost nothing and has now cost two runs.

## 7. Reproducing

```
python train.py --level 1 --run live-v2 --lam 0.99 --updates 350
python report.py live-v2
```

`live-episodes.csv` (one row per attempt) and `live-updates.csv` (one row per PPO
update, 16 columns) are the raw logs; `live-report.pdf` is the generated
learning-curve and diagnostics view. Checkpoints are not archived — `*.pt` is
gitignored — but `runs/live-v2/best.pt` and `latest.pt` exist locally.

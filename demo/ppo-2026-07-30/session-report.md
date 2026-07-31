# gdbot — session report, 2026-07-30

Rewrote the agent from NEAT to conv-PPO, rebuilt the visualisation as a browser
dashboard, took the whole thing to the real game for the first time, and left it
training unattended for seven hours.

**Headline: the bot reached 52.4% of Stereo Madness on real Geometry Dash**,
learning from scratch in a single 7.12-hour run — against ~19% for the NEAT
version it replaced. It had not plateaued when the run ended.

---

## 1. What changed, and why

### The old stack could not go further

| | NEAT (before) | conv PPO (now) |
|---|---|---|
| What it sees | 22 hand-picked numbers | 24×16×4 occupancy grid + 17 scalars |
| Saw blades / orbs / pads / portals | **invisible** | four dedicated channels |
| Optimiser | neuroevolution | PPO (GAE, clipped surrogate, KL early stop) |
| Watching it | pygame, **inside** the decision loop | browser, off the hot path |
| Cost of watching | capped training at 120 decisions/s | ~0.3 µs/step |

Two hard ceilings forced the change. The 22-number observation could not
*represent* most of what kills you in Geometry Dash, and neuroevolution cannot
grow topologies over the 1 536 grid inputs that can. Separately, the old viewer
redrew the network on every decision behind a `clock.tick(120)`, so watching the
bot throttled it and the redraw ran whether or not anyone was looking.

### The rewrite

`bridge → GDEnv → {LiveEnv, SimEnv} → ConvActorCritic → PPO`, with telemetry
hanging off the side. Nothing in the observation encodes *where in the level* the
player is — no x, no percent — because a policy that can read the clock memorises
a level instead of learning to see it.

---

## 2. Real Geometry Dash

**7.12 h · 1 412 608 steps · 1 760 attempts · 0 completions · GD 2.2081 / Geode 5.8.2**

Mean progress went 0 → 25.4%, best attempt 52.4%, entropy 0.693 → 0.238.

### Learning is a staircase

The agent clears an obstacle, piles up against the next, then breaks through.

| tenth of run | mean % | p90 % | best % |
|---|---|---|---|
| 1 | 3.60 | 5.46 | 8.86 |
| 2 | 5.14 | 8.86 | 12.44 |
| 3 | 10.40 | 15.81 | 19.41 |
| 4 | 16.53 | 19.47 | 19.54 |
| 5 | 15.52 | 19.47 | 19.49 |
| 6 | 17.47 | 19.48 | 19.73 |
| 7 | 20.85 | 26.71 | 26.79 |
| 8 | 17.86 | 26.71 | 27.89 |
| 9 | 24.63 | 36.12 | 48.86 |
| 10 | **25.00** | **45.34** | **52.41** |

Deciles 4–6 are a textbook plateau: mean stuck at ~16%, p90 pinned at exactly
19.47% for three tenths of the run — the agent hitting one obstacle it could not
solve. Then it broke through twice in a row.

### Where it dies

| band | deaths | what it is |
|---|---|---|
| 2–5% | 405 | the opening spike pair |
| 10–11% | 199 | second cluster |
| **18–19%** | **498** | **the wall — 28% of every attempt** |
| 26–27% | 127 | fourth obstacle |

Reached ≥20% in 19.6% of attempts, ≥30% in 5.9%, ≥40% in 2.6%, ≥50% in 0.28%.

### It was still accelerating when it stopped

The p90 gained more in the last tenth (36.1 → 45.3) than in any earlier one. The
run did not converge — it was **cut short by a bridge stall**. That single fact
sets the top priority: make runs survive stalls, then run much longer.

---

## 3. Simulator

Two runs, freshly generated course every episode so there is nothing to memorise:

| run | steps | wall | mean % | best % | completions |
|---|---|---|---|---|---|
| `simlong` | 819 200 | 7.4 min | 16 → 84.2 | 99.94 | 455 / 1 265 (36.0%) |
| `demo` (resumed) | 1 382 400 | 15.1 min | → 89.5 | 99.94 | 921 / 1 900 (48.5%) |

Plateaus at 85–89% mean. Greedy play completes 99.9% of unseen courses.

**The simulator is 33× faster than the live game** (1 850 vs 56 steps/s) — the
entire argument for keeping it. It is also markedly easier, and has no orbs,
pads, portals or non-cube gamemodes, so it is a development harness and a prior,
never proof.

---

## 4. Three bugs the real game found

None of these were reachable without running against GD.

1. **`fast_respawn` death loop.** `resetLevel()` called from inside `postUpdate`
   lands mid death-sequence, so the respawned player died again immediately —
   4 000+ attempts at 0.00%, the level unusable, and it survived detaching the
   agent. It also silently poisoned an entire benchmark round before it was
   spotted. Deferring the reset to the top of the frame cut it from unrecoverable
   to ~9× too many respawns; it now defaults **off** and the mod disables it
   itself after 20 dead respawns.
2. **Handshake could desync permanently.** One timeout left a stale signal that
   satisfied the next wait instantly, failed the sequence check, and stopped the
   mod waiting *forever* — the game free-ran at ~240 fps with 1 650 of every
   2 250 frames unanswered. Now self-healing.
3. **There is no wall-clock speedup.** `--speed` does not multiply physics steps;
   it skips presents, worth 0.6× → 1.0× real time and nothing beyond. GD gates its
   own stepping, so `postUpdate` never fires faster however much render work is
   skipped. The docs claiming a speedup were wrong and were corrected.

A fourth, fixed this session: **a stall ended the 7-hour run.** `BridgeLost`
propagated out of the training loop and the process exited, while the game was
perfectly healthy (verified still publishing at exactly 60 fps afterwards).

---

## 5. Benchmarks

`python bench.py components` — all figures p50 unless noted, CPU is a 16-thread
box with torch's default 8 threads, GPU is an RTX 5060 Ti.

### Where a decision goes

| stage | time | note |
|---|---|---|
| `SimEnv.step()` | 42.5 µs | 23 516/s ceiling |
| ‣ grid rasterise | 28.2 µs | 66% of the step |
| `act()` sample | 341.2 µs | 2 931/s ceiling |
| `act()` greedy | 291.4 µs | |
| `act(introspect=True)` | 501.5 µs | 1.5× — only paid when watched |
| raw `forward()` | 139.8 µs | 59% of `act()` is Python/numpy overhead |

**The policy dominates in the simulator** — 341 µs of inference against 42.5 µs of
environment, so sim training is *inference-bound*, not env-bound. Live is the
reverse: a frame is 16 667 µs at 60 Hz, so inference is ~2% and the game is the
bottleneck. Same code, opposite profile.

Predicted cycle at rollout 2048: 786 ms collect + 326 ms update = **1 842 steps/s**.
Measured end-to-end in `simlong`: **1 853 steps/s**. The model of the system is right.

### Batched inference — CPU vs GPU

| batch | CPU | CUDA |
|---|---|---|
| 1 | 7 009 obs/s | 796 obs/s (**0.27× — slower**) |
| 32 | 61 803 obs/s | 82 916 obs/s |
| 128 | 134 383 obs/s | 329 966 obs/s |
| 512 | — | **1 208 802 obs/s** |

**The GPU is 3.7× slower than the CPU for a single observation** — kernel-launch
overhead swamps a 216 k-parameter net. It only pays above batch ~32. Single-env
training belongs on CPU; a vectorised simulator is what would unlock the GPU, and
the ceiling there is ~47× the current single-env rate.

### Telemetry — is an unwatched run really free?

| | time | share of an env step |
|---|---|---|
| `should_capture()`, no viewer | **0.1 µs** | 0.20% |
| `publish()` a full snapshot | 49.8 µs | only at 20 fps |
| amortised while watched | 0.30 µs/step | 0.70% |

Claim verified. A snapshot is 14.9 KB → 298 KB/s at 20 fps.

### PPO update cost

| rollout | update | collect | update share |
|---|---|---|---|
| 512 | 102.6 ms | 196 ms | 34.3% |
| 1024 | 184.9 ms | 393 ms | 32.0% |
| 2048 | 326.4 ms | 786 ms | 29.3% |
| 4096 | 645.0 ms | 1 572 ms | 29.1% |

### Torch threads — a trap worth knowing

| threads | `act()` | update (1024, 2 epochs) |
|---|---|---|
| 1 | 2 898/s | 201 ms |
| 4 | 2 836/s | 121 ms |
| **8 (default)** | **2 958/s** | **86 ms** |
| 16 | 2 433/s | **301 ms — 3.5× worse** |

This net is small enough that intra-op parallelism costs more than it saves.
Forcing all 16 threads made the first benchmark run report a 1 981 ms update and
2 171 act/s — numbers that contradicted the trainer's own measured throughput.
**Do not set `OMP_NUM_THREADS` above 8 for this workload.**

### Live bridge

| `--speed` | steps/s | vs real time | missed | handshake p50 |
|---|---|---|---|---|
| 1 | 35 | 0.6× | 0 | 13 µs |
| 2 | 59 | 1.0× | 0 | 10 µs |
| 4 | 60 | 1.0× | 0 | 11 µs |
| 16 | 60 | 1.0× | 0 | 11 µs |

Zero missed frames at every setting; the ceiling is GD's, not ours.

---

## 6. Hyperparameter sweep

15 configurations, one axis at a time off a shared baseline, 250 000 simulator
steps each, three at a time — 18.2 minutes total. Each config runs the *real
trainer* as a subprocess, so this measures the shipping code path. Scored on mean
% over the last 50 updates.

| axis | config | mean % | completion % | entropy |
|---|---|---|---|---|
| lr | 1e-4 | 12.9 | 0.0 | 0.491 |
| lr | **3e-4** *(baseline)* | 48.3 | 12.0 | 0.444 |
| lr | **1e-3** | **65.5** | **25.5** | 0.537 |
| entropy | 0.0 | 15.4 | 0.7 | 0.368 |
| entropy | **0.005** | **56.2** | **17.8** | 0.441 |
| entropy | 0.02 | 31.2 | 4.6 | 0.521 |
| rollout | 512 | 59.6 | 11.3 | 0.412 |
| rollout | 1024 | 36.8 | 5.7 | 0.358 |
| rollout | 4096 | 35.0 | 13.0 | 0.546 |
| epochs | 2 | 12.5 | 0.0 | 0.519 |
| epochs | **8** | **63.2** | **24.1** | 0.494 |
| gamma | 0.97 | 15.8 | 0.2 | 0.483 |
| gamma | 0.995 | 48.5 | 10.8 | 0.485 |
| hidden | 128 | 21.1 | 2.4 | 0.449 |
| hidden | 512 | 51.1 | 14.3 | 0.444 |

### One finding explains most of the table

**At a fixed sample budget, everything that buys more gradient steps per sample
wins.** The top three — lr 1e-3, epochs 8, rollout 512 — are all "optimise harder
per sample", and the bottom three — lr 1e-4, epochs 2, gamma 0.97 — are all
"optimise less, or credit less". The default configuration is simply
*under-optimised* at 250 k steps.

That matters most where samples are expensive. Live runs at 60 steps/s, so the
data is ~33× costlier than in the simulator, and extracting more learning per
frame is worth more there than anywhere.

Secondary results:

- **Entropy 0.005 beats the 0.01 default** (56.2 vs 48.3), and 0.0 collapses
  (15.4, entropy → 0.368). Some exploration is essential; the default is a touch
  too much of it.
- **Short horizons fail.** gamma 0.97 scored 15.8; 0.99 and 0.995 are equivalent.
  With ~800-step episodes, a 33-step effective horizon cannot reach the payoff.
- **Width matters below 256.** hidden 128 scored 21.1; 512 was marginally better
  than 256 but well inside noise.

### What this does not establish

- **One seed per config.** Differences under ~10 points are not separable from
  noise. Only the large gaps (lr, epochs, gamma) are safe to act on.
- **This measures learning *speed* at 250 k steps, not final quality.** A high
  learning rate that wins early can plateau lower later.
- **The simulator is a prior, not proof** — markedly easier than the real game,
  and with no orbs, pads, portals or non-cube modes.

---

## 7. Honest limitations

- **Zero completions on the real game.** 52.4% is halfway, not a clear.
- **Live entropy fell to 0.238** with no completions — the policy is becoming
  deterministic while still failing, a premature-convergence risk.
- **Live training is ~1× real time.** Anything meaningful is an overnight job.
- **The simulator cannot pretrain most of a real level** — no orbs, pads, portals
  or ship/wave/ball sections.
- **`fast_respawn` is still ~9× too many respawns**, so it stays off.
- **The game window freezes while an agent is attached.** Training is unaffected;
  the frame-pacing collapse that causes it is what keeps the handshake in lockstep.
- **Sim results are a prior for live, not evidence about it.**

---

## 8. Next steps, in priority order

1. **Long live runs, now that they survive stalls.** The run was accelerating when
   it died. Cheapest remaining gain by a wide margin.
2. **Practice-mode curriculum against the 18–19% wall.** The mod already
   implements `CMD_SET_PRACTICE`, `CMD_RESPAWN_CP` and `cp_pct`, and `LiveEnv`
   already accepts `practice=True` — but it has **never been run live**. 28% of
   attempts die in one 2% band; grinding that segment instead of replaying the
   first 10% every time attacks it directly.
3. **Entropy floor.** Guard against the policy going deterministic before it can
   finish a level.
4. **Vectorised simulator + GPU.** Benchmarks say batch-512 inference is ~1.2 M
   obs/s against 7 k single-env on CPU. Parallel envs are the one change that
   makes sim iteration dramatically faster.
5. **Simulator fidelity: orbs, pads, then a ship section.** Required before the
   simulator can pretrain anything past the first third of a real level.
6. **Fix `fast_respawn` properly** — worth ~8% of wall-clock, so it ranks below
   all of the above.

---

## Files

| path | what |
|---|---|
| `live-episodes.csv` | every one of the 1 760 live attempts |
| `live-updates.csv` | per-PPO-update diagnostics for the 7.12 h run |
| `live-report.pdf` | generated learning curve, PPO health, attempt distribution |
| `sim-updates.csv`, `sim-report.pdf` | the `simlong` simulator run |

Reproduce with `python report.py live` / `python bench.py components`.

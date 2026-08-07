# gdbot: a mathematical account of the learning system

This document derives what the agent is actually optimising, works out the
numbers that follow from the constants in the code, and reports what happened
when the predictions were tested. It is written against the repository as it
stands — every constant cited is a real symbol in a real file, and every measured
number is reproducible with a command given in the text.

Three things fell out of the derivations that changed the code:

1. The PPO trust-region brake was measuring the wrong average and engaging up to
   **2× later** than configured (§5.4).
2. The GAE credit horizon is **16.8 steps**, while one jump takes **26 steps** —
   the advantage estimator cannot span a single jump (§4.3).
3. Because the player auto-scrolls, progress is an affine function of survival
   time, so the progress reward and the death penalty encode the *same*
   preference — and within the discount horizon the death term is **4.7× the
   larger** of the two (§3.3).

And one thing fell out of *testing* them that contradicted the analysis:

4. The vectorised rollout built in §6 is **2.5× worse per sample** than the
   single-environment one it was meant to replace, despite being 4.5× faster per
   second (§7.1). The throughput argument was correct and the conclusion drawn
   from it was wrong. `--envs` now defaults to 1.

Point 4 also invalidated two of the three experiments in this document, because
they had been run on the faster backend before that backend had been validated.
The broken versions are left visible next to their replacements, because the
failure mode — an experiment whose arms all sit on a floor, and therefore reports
tight confident intervals around nothing — is more instructive than the result
would have been.

Rerun properly, the λ prediction of point 2 comes out **monotone in the predicted
direction across all four arms, and still not statistically resolvable**: the
effect is 2.5 pp against a between-seed SD of ~7 pp, so separating it would need
roughly 120 seeds per arm rather than three (§7.2). It is reported that way
rather than as a confirmation.

---

## 1. The problem, formally

### 1.1 It is a POMDP, not an MDP

Geometry Dash is a deterministic dynamical system, but the agent does not observe
its state. It observes a 24 × 16 window of occupancy around the player plus a
handful of kinematic scalars. Level geometry outside that window, the exact
sub-cell position of the player, and the internal state of triggers are all
hidden. So the object being solved is a partially observable Markov decision
process

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, T, R, \Omega, O, \gamma \rangle$$

with $\mathcal{A} = \{0, 1\}$ (release, hold), $\gamma = 0.99$, and an
observation function $O : \mathcal{S} \to \Omega$ given by `LiveEnv._obs` /
`VecSimEnv._obs`.

The policy is memoryless — $\pi_\theta(a \mid o)$, not $\pi_\theta(a \mid h)$ for
a history $h$. That is a deliberate restriction, and it is only sound to the
extent that $O$ preserves enough state to make the *observed* process
approximately Markov. Two places where it did not, and now does:

| Hidden variable | Why it matters | Fix |
|---|---|---|
| Whether the button was already down | GD fires an orb on a **fresh** click and ignores a held one. "Holding" and "pressing" are different states of the world that the old observation rendered identically. | `prev_action` scalar |
| How long the player has been airborne | $v_y = 0$ both on the ground and at the apex of a jump. `on_ground` separates those two, but nothing said how long a fall had been running. | `air_time` scalar |

Neither is cosmetic. The first one is a genuine aliasing of two states with
different dynamics, and no amount of training can fix an aliased observation —
the optimal memoryless policy on an aliased POMDP is strictly worse than the
optimal policy on the underlying MDP, and can be worse than *both* of the
deterministic policies it is forced to average over.

The observation vector is **19 scalars** (`obs.N_SCALARS`), up from 17.

### 1.2 What is deliberately *not* observed

Nothing in $\Omega$ encodes absolute position: no $x$, no percent, no timer. This
is a design constraint, not an oversight. With a position input the policy can
represent "at $x = 4130$, jump", which is a lookup table for one level and
transfers to nothing. Removing it makes the policy a function of local geometry
only, which is the definition of the reactive generalist the project is after.

The cost is that the value function is also denied position, so $V(o)$ cannot
represent "I am 90% through and about to finish". §3.3 shows this matters less
than it sounds, because with auto-scroll the progress signal is almost entirely
redundant with the survival signal anyway.

---

## 2. The network

### 2.1 Geometry

Input $4 \times 16 \times 24$: four channels (solid, hazard, orb/pad, portal),
16 rows, 24 columns. Each cell is `bridge.CELL` $= 30$ GD units $= 1$ block. The
player sits at column `GRID_BEHIND` $= 2$, so the window covers **2 blocks behind
and 21 ahead**.

| Layer | Kernel | Stride | Output $(C, H, W)$ | Params | MACs |
|---|---|---|---|---|---|
| conv1 | $3\times3$ | 1 | $16 \times 16 \times 24$ | 592 | 221,184 |
| conv2 | $3\times3$ | 2 | $32 \times 8 \times 12$ | 4,640 | 442,368 |
| conv3 | $3\times3$ | 2 | $32 \times 4 \times 6$ | 9,248 | 221,184 |
| flatten | — | — | 768 | — | — |
| dense | $787 \to 256$ | — | 256 | 201,728 | 201,472 |
| policy | $256 \to 2$ | — | 2 | 514 | 512 |
| value | $256 \to 1$ | — | 1 | 257 | 256 |
| **total** | | | | **216,979** | **1,086,976** |

The parameter total is exact — `sum(p.numel() for p in model.parameters())`
reports 216,979, and the viewer prints it in its header.

### 2.2 The dense layer is 93% of the parameters and 19% of the arithmetic

$$\frac{201{,}728}{216{,}979} = 92.97\% \text{ of parameters}, \qquad
\frac{201{,}472}{1{,}086{,}976} = 18.53\% \text{ of MACs}$$

The three convolutions carry 6.7% of the parameters and do 81% of the work. This
asymmetry is the single clearest structural weakness in the network: almost all
of the capacity sits in one fully-connected layer that reads a flattened
$32 \times 4 \times 6$ tensor, and that layer is *not* translation-equivariant.
The convolutions learn "spike at head height" once and share that detector across
the whole window; the dense layer then has to learn what that feature *means*
separately at each of the $4 \times 6 = 24$ pooled positions conv3 leaves — each
of which covers a $4 \times 4$ block region of the original grid, since two
stride-2 layers downsample by 4 in each axis.

Two standard remedies, neither yet implemented, both of which would cut the
parameter count by roughly an order of magnitude:

* **Global pooling over width.** Replace `flatten` with a mean or max over the
  width axis, giving $32 \times 4 = 128$ features. Dense becomes $147 \to 256$,
  i.e. 37,888 parameters, and the whole network drops to ~53k. This throws away
  *where* along the window a feature was, which for this task is probably too
  destructive — distance to the hazard is the whole game.
* **A $1\times1$ bottleneck before the flatten.** $32 \to 8$ channels costs 264
  parameters and reduces the flatten to 192, making dense $211 \to 256 =
  54{,}272$. Total ~69k, with spatial information intact.

The second is the better bet and is the recommended next architectural change.
It is not made here because it invalidates every existing checkpoint and this
document's job is to say *why* it should be made, with numbers.

### 2.3 Receptive field

For a stack of convolutions the receptive field $r$ propagates backwards as
$r_{l-1} = s_l (r_l - 1) + k_l$. Starting from a single unit of conv3's output:

$$r = 1 \xrightarrow{\text{conv3}(k{=}3,s{=}2)} 3
      \xrightarrow{\text{conv2}(k{=}3,s{=}2)} 7
      \xrightarrow{\text{conv1}(k{=}3,s{=}1)} 9$$

So one conv3 unit sees a $9 \times 9$ cell patch $= 270$ GD units. At the
simulator's forward speed of `VX` $= 10.3$ blocks/s $= 309$ units/s, that is
**0.87 s of travel** per conv3 unit. The dense layer sees all of them, so the
policy's full field of view is the 21 blocks ahead $= 630$ units $= 2.04$ s.

Keep that 2.04 s in mind; §4.3 shows the credit-assignment horizon is far
shorter, and that mismatch is the actionable finding.

---

## 3. The reward

### 3.1 Definition and the telescoping property

From `env.shape_reward`, with $p_t \in [0,1]$ the fraction of the level reached:

$$r_t = R_p \max(0,\, p_t - p_{t-1}) - R_d \mathbf{1}[\text{dead}] + R_c \mathbf{1}[\text{complete}]$$

with $R_p = 10$, $R_d = 1$, $R_c = 10$.

Because a live player's $p_t$ is non-decreasing, the $\max(0, \cdot)$ is inactive
and the progress terms telescope:

$$\sum_{t=1}^{T} R_p (p_t - p_{t-1}) = R_p\, p_T$$

so an episode's undiscounted return is exactly $10 p_T - \mathbf{1}[\text{dead}]
+ 10\,\mathbf{1}[\text{complete}]$. This is asserted directly in
`tests/test_stack.py` ("an episode's return is exactly its progress minus its
death"), which is why the clamp is safe: it exists only to stop a policy farming
reward by oscillating across a percent boundary, and on a monotone trajectory it
changes nothing.

### 3.2 Progress and survival are the same quantity

The player auto-scrolls at a fixed speed. In the simulator this is exact —
`px += VX * DT` every step, so

$$p_t = \frac{1 + t \cdot v_x \Delta t}{L}$$

is *affine in $t$*. In the real game it is piecewise affine, with breakpoints
only at speed portals.

Therefore **progress carries no information that survival time does not**. The
two reward terms are not two objectives being traded off; they are two encodings
of "stay alive longer". The interesting question is not which one to prefer, but
what their relative magnitudes do to the value function.

### 3.3 The death term dominates the value function by 4.7×

Take Stereo Madness: length 26,724 units, forward speed 309 units/s, so
$T \approx 86.5$ s $\approx 5{,}190$ decisions at 60 Hz. A full clear pays
$R_p = 10$ spread over those steps:

$$\bar{r} = \frac{10}{5190} = 1.927 \times 10^{-3} \text{ per step}$$

The discounted value of an immortal policy making steady progress is

$$V_\infty = \frac{\bar r}{1 - \gamma} = \frac{1.927\times10^{-3}}{0.01} = 0.193$$

The value of a state that dies in $n$ steps is

$$V(n) = \bar r\,\frac{1 - \gamma^n}{1-\gamma} - \gamma^{\,n-1} R_d$$

| $n$ (steps to death) | progress term | death term | $V(n)$ |
|---|---|---|---|
| 10 | $+0.018$ | $-0.914$ | $-0.895$ |
| 26 (one jump) | $+0.044$ | $-0.778$ | $-0.733$ |
| 100 (one horizon) | $+0.122$ | $-0.370$ | $-0.247$ |
| $\infty$ | $+0.193$ | $0$ | $+0.193$ |

The value function's dynamic range is $1.088$, of which the progress term
contributes at most $0.193$ (17.7%) and the death term $0.914$ (84.0%):

$$\frac{\text{death range}}{\text{progress range}} = \frac{0.914}{0.193} = 4.74$$

So the critic is overwhelmingly a *time-to-death predictor*, and $R_p = 10$ is
close to decorative. This is not obviously wrong — §3.2 says the two signals
agree — but it has a concrete consequence: **tuning $R_p$ is nearly a no-op, and
tuning $R_d$ is the lever that actually rescales the advantages.** Anyone tuning
reward shaping on this project should know that before spending compute on it.

`experiments.py reward` tests the prediction directly by zeroing one term at a
time; results in §7.3.

### 3.4 The completion bonus is unreachable shaping

$R_c = 10$ fires at $p \geq 0.999$. Discounted back from a start state 5,190
steps away it is worth $10 \times 0.99^{5189} \approx 2 \times 10^{-22}$, which
is indistinguishable from zero. Until the policy can already finish a level, the
completion bonus contributes nothing to any gradient. It is a scoreboard, not an
incentive. (In the simulator, courses are ~1,280 steps, so $0.99^{1280} \approx
3\times10^{-6}$ — still negligible, which is consistent with the sweep's
completion rates being driven by policy quality rather than by the bonus.)

---

## 4. Credit assignment

### 4.1 GAE

The advantage estimator (`ppo.compute_gae`, `ppo.compute_gae_vec`) is

$$\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l},
\qquad \delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$

$\lambda = 0$ gives the one-step TD advantage: minimum variance, but biased by
however wrong $V$ is. $\lambda = 1$ gives the Monte-Carlo advantage: unbiased
given any state-dependent baseline, maximum variance. Intermediate $\lambda$
interpolates, and the implementation computes it by the standard backwards
recursion in $O(T)$.

The vectorised form runs the identical recursion with $N$ accumulators, one per
environment column. Credit never crosses between environments because they are
separate accumulators, and never crosses an episode boundary within one because
$d_t$ zeroes the recursion — the same guarantee, checked by the same test.

### 4.2 Two different horizons

Discounting and GAE have *different* effective horizons and it is easy to
conflate them:

$$H_\gamma = \frac{1}{1 - \gamma} = 100 \text{ steps},
\qquad H_{\text{GAE}} = \frac{1}{1 - \gamma\lambda} = \frac{1}{1 - 0.9405} = 16.8 \text{ steps}$$

$H_\gamma$ is how far into the future the *objective* looks. $H_{\text{GAE}}$ is
how far the *advantage estimate* propagates a surprise before the value function
takes over. Converting both to game distance at 60 steps/s and 10.3 blocks/s:

| Quantity | Steps | Seconds | Blocks |
|---|---|---|---|
| Sight (grid ahead) | — | 2.04 | 21.0 |
| Discount horizon $H_\gamma$ | 100 | 1.67 | 17.2 |
| **One jump arc** | **26** | **0.44** | **4.46** |
| GAE horizon $H_{\text{GAE}}$ | 16.8 | 0.28 | 2.89 |

Sight (21 blocks) slightly exceeds the discount horizon (17.2 blocks), which is
a good place to be: the agent can see marginally further than it is asked to
care about.

### 4.3 The GAE horizon is shorter than a single jump

The jump arc is derived from the simulator's own integrator rather than
estimated. With `JUMP_V` $= 20$ blocks/s and `GRAVITY` $= 90$ blocks/s² at
$\Delta t = 1/60$, the height after $n$ ticks is

$$y_n = \frac{1}{60}\sum_{k=1}^{n}(20 - 1.5k) = \frac{20n - 0.75\,n(n+1)}{60}$$

which returns to zero at $n(19.25 - 0.75n) = 0$, i.e. $n = 25.67 \Rightarrow$
the cube lands on tick **26**, having travelled $26 \times 10.3/60 = 4.46$
blocks, with an apex of 2.06 blocks at tick 13. (These match the code comment's
"~2.2 high, ~4.5 wide" to within the discretisation.)

So $H_{\text{GAE}} = 16.8 < 26$: **the advantage estimator's credit decays to
$1/e$ before the jump it is evaluating has even landed.** The decision to jump
and the outcome of that jump are separated by more than the estimator's reach,
so the connection between them has to be carried by the value function — which,
per §3.3, is mostly a time-to-death predictor and cannot represent "this jump
will clear that spike" without position information.

Solving for the $\lambda$ that covers one jump:

$$\frac{1}{1 - \gamma\lambda} \geq 26
\;\Longrightarrow\; \gamma\lambda \geq \frac{25}{26}
\;\Longrightarrow\; \lambda \geq \frac{0.96154}{0.99} = 0.9713$$

**Prediction: $\lambda \approx 0.97$–$0.98$ should outperform the default 0.95.**
Tested in §7.2.

This also explains an otherwise odd entry in the existing sweep: `gamma-0.97`
scored 15.8% against the baseline's 48.3%, a far larger drop than a modest change
of objective should cause. But $\gamma$ enters $H_{\text{GAE}}$ too —
$\gamma\lambda = 0.9215$, so $H_{\text{GAE}}$ falls from 16.8 to 12.7 steps, less
than half a jump. The sweep was reading a credit-assignment failure as a
discounting result.

### 4.4 Truncation

An episode that hits the step limit has not failed, and treating it as terminal
teaches the policy that surviving ends the world. Both backends bootstrap
instead:

$$r_T \leftarrow r_T + \gamma V(s_{T+1})$$

For the single-environment path this is straightforward. For the vectorised path
it is not, because auto-reset means $s_{T+1}$ has already been overwritten by a
new episode's first frame — so `VecSimEnv.step` hands back the discarded frame
in `info["truncated"]` for exactly those slots. Skipping this would apply a
silent $-V(s_{T+1})$ bias to every truncated episode.

### 4.5 A constraint on how far the rollout can be parallelised

$N$ environments and a fixed transition budget $B$ give $B/N$ steps per
environment, and GAE is truncated at that length. Truncating well inside
$H_{\text{GAE}}$ throws away the credit the estimator was built to propagate, so

$$\frac{B}{N} \gtrsim 4 H_{\text{GAE}} \;\Longrightarrow\;
N \lesssim \frac{B\,(1 - \gamma\lambda)}{4}$$

At $B = 2048$, $\gamma\lambda = 0.9405$: $N \lesssim 30$. `train.py` prints this
as a warning when it is violated, which is how the 128-environment configuration
in §6 was flagged as fast-but-wrong rather than quietly accepted.

---

## 5. The PPO update

### 5.1 The objective

With $\rho_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ and $\epsilon = 0.2$:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\;
\text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\Big]$$

$$L(\theta) = -L^{\text{CLIP}} + c_v L^{\text{VF}} - c_e\,\mathbb{E}[\mathcal{H}(\pi_\theta)]$$

with $c_v = 0.5$, $c_e = 0.01$. Advantages are normalised over the whole rollout
before the epoch loop, not per minibatch, so every minibatch in an update sees a
consistent scale.

The $\min$ is what makes this a pessimistic bound rather than a symmetric trust
region: when $\hat A_t > 0$ the objective stops rewarding increases in $\rho$
past $1+\epsilon$, and when $\hat A_t < 0$ it stops rewarding decreases past
$1-\epsilon$ — but in both cases the *unclipped* branch is still available if it
is worse, so the update can always undo an over-large step.

### 5.2 The clipped value loss

Added in this revision:

$$V^{\text{clip}} = V_{\text{old}} + \text{clip}(V_\theta - V_{\text{old}}, -\epsilon, +\epsilon)$$
$$L^{\text{VF}} = \mathbb{E}\big[\max\big((V_\theta - \hat R)^2,\; (V^{\text{clip}} - \hat R)^2\big)\big]$$

The advantages that drive $L^{\text{CLIP}}$ were computed from $V_{\text{old}}$.
Without this, four epochs of unconstrained regression can move the critic far
enough that the advantages being optimised no longer correspond to any value
function the learner holds. The $\max$ keeps it pessimistic in the same sense as
the policy loss.

### 5.3 KL: why the estimator was changed

Sampling $x \sim \pi_{\text{old}}$ and writing $\rho = \pi_{\text{new}}/\pi_{\text{old}}$, two unbiased estimators of $\mathrm{KL}(\pi_{\text{old}} \Vert \pi_{\text{new}})$ are

$$k_1 = -\log \rho, \qquad k_3 = (\rho - 1) - \log \rho$$

Both have expectation $\mathrm{KL}$, since $\mathbb{E}[\rho] = 1$ makes the extra
term a mean-zero control variate. But $k_3 \geq 0$ *pointwise* (because
$\log \rho \leq \rho - 1$ for all $\rho > 0$), while $k_1$ is routinely negative
on individual samples.

The variance argument is the one that matters. Write $\rho = 1 + \delta$ and
expand:

$$k_1 = -\log(1+\delta) \approx -\delta + \tfrac{1}{2}\delta^2, \qquad
k_3 = \delta - \log(1+\delta) \approx \tfrac{1}{2}\delta^2$$

**$k_1$ is first order in the policy change while its mean is second order.** Its
leading term is pure noise that cancels only in expectation, so its relative
error is worst exactly when the policy has barely moved — which is the regime the
early-stop test lives in. $k_3$ is second order throughout.

For $k_1$ this gives $\mathrm{SD}(k_1) \approx \sqrt{2\,\mathrm{KL}}$, which
holds to three digits. For $k_3$ the answer depends on the *distribution* of
$\delta$, and the usual Gaussian shortcut ($\mathbb{E}[\delta^4] = 3\sigma^4$,
giving $\sqrt{2}\,\mathrm{KL}$) is badly wrong here: this is a **two-action**
policy, so $\rho$ takes exactly two values, $\delta^2$ is nearly constant, and
$k_3 \approx \delta^2/2$ barely varies at all. The numbers below are measured
(4M samples per row, $\pi_{\text{old}}$ Bernoulli, $\pi_{\text{new}}$ chosen to
put the true KL at the configured threshold of 0.03, minibatch 256):

| $\pi_{\text{old}}(a{=}1)$ | entropy | SE of $k_1$ | SE of $k_3$ | variance reduction |
|---|---|---|---|---|
| 0.50 | 0.693 | 51.3% | **1.0%** | 50.7× |
| 0.70 | 0.611 | 53.2% | **6.4%** | 8.3× |
| 0.90 | 0.325 | 57.5% | **17.5%** | 3.3× |
| 0.97 | 0.135 | 65.9% | **35.1%** | 1.9× |
| 0.99 | 0.056 | 82.9% | **62.3%** | 1.3× |

(SE columns are the standard error of the minibatch estimate as a percentage of
the 0.03 threshold.)

Two things follow. First, **$k_1$'s error at the threshold is 51–83% across the
entire range** — a brake measured that way is a coin flip regardless of how the
policy is behaving. Second, $k_3$'s advantage is largest precisely when the
policy is still exploring, and decays as it commits. Runs in this project settle
around entropy 0.36–0.55, i.e. $\pi \approx 0.63$–$0.85$, so the operating regime
is a 3–10× variance reduction. $k_3$ costs one subtraction.

Both are reported: `stats["kl"]` is $k_3$ and drives the brake,
`stats["kl_k1"]` is kept so the two can be compared on a real run.

### 5.4 The early-stop bug

The original test was

```python
stats["kl"] += kl.item(); batches += 1
...
if batches and stats["kl"] / batches > target_kl: break
```

`stats["kl"]` accumulates across *every minibatch since the update began*, so by
epoch 4 the comparison is against a mean that includes epoch 1's near-zero KL.
Let $\kappa_e$ be the true KL during epoch $e$; early in an update it grows
roughly linearly, $\kappa_e \approx e\kappa$. The test after epoch $E$ compares

$$\frac{1}{E}\sum_{e=1}^{E} e\kappa = \frac{\kappa(E+1)}{2} \;\;\text{against}\;\; \text{target}$$

so it fires when $\kappa_E = E\kappa$ has reached

$$\frac{2E}{E+1} \times \text{target}$$

| epochs | brake fires at |
|---|---|
| 2 | 1.33 × target |
| 4 | **1.60 × target** |
| 8 | **1.78 × target** |
| $\to\infty$ | 2 × target |

The configured 0.03 was operating as 0.048 at the default 4 epochs, and as 0.053
at the `--epochs 8` setting the existing sweep found best — so the sweep's
preference for more epochs was partly a preference for a *looser trust region*,
not for more gradient steps per sample. The fix tracks KL within the current
epoch only.

### 5.5 Explained variance

$$\text{EV} = 1 - \frac{\mathrm{Var}(\hat R - V_{\text{old}})}{\mathrm{Var}(\hat R)}$$

Now computed every update and plotted live. It is the only cheap statistic that
distinguishes "the critic is learning something" from "the advantages are
Monte-Carlo noise around a constant": at $\text{EV} = 0$ the value head is no
better than predicting the mean return, and every $\hat A_t$ is then just a
noisy return. Observed behaviour on a short simulator run: $-1.88$ at
initialisation, crossing zero within two updates, $+0.99$ by update 200.

---

## 6. Throughput

### 6.1 The batch-1 problem

One decision costs 1.09 MMAC $\approx$ 2.17 MFLOP (§2.1). A CPU core sustains
tens of GFLOP/s on this kind of work, so the arithmetic in a single decision is
worth tens of microseconds at most — yet the measured cost at batch 1 was ~341 µs
(`python bench.py components`), i.e. an effective 6.4 GFLOP/s. The gap is fixed
overhead per call: Python dispatch, tensor allocation, and kernel launch, none of
which scale with batch size. §6.3 puts that overhead at 96% of a batch-1
decision, measured rather than inferred.

This is also why CUDA was measured **3.7× slower than the CPU** at batch 1 on
this network. A GPU's fixed cost per launch is larger, and 216k parameters cannot
amortise it. The conclusion drawn at the time — "single-env training belongs on
CPU" — was correct but incomplete. The real conclusion is that *batch 1 is the
problem*, and the fix is to stop running at batch 1.

`train.pick_device` now encodes this: `auto` selects CUDA only when
`--envs >= 32`, rather than whenever a GPU exists.

### 6.2 The vectorised simulator

`gdbot/vec_env.py` holds $N$ courses as arrays and steps them with array
operations, including the grid rasterisation — which is where most of the time
goes, since the physics is a handful of flops against 1,536 cells of occupancy.

Environment throughput alone (`python -m gdbot.vec_env`):

| $N$ | env-steps/s | µs per step |
|---|---|---|
| scalar `SimEnv` | 15,823 | 63.2 |
| vec, 1 | 10,587 | 94.5 |
| vec, 8 | 75,286 | 13.3 |
| vec, 32 | 225,237 | 4.4 |
| vec, 128 | 448,991 | 2.2 |

Note the vectorised environment is **slower than the scalar one at $N = 1$** —
numpy's per-call overhead on length-1 arrays exceeds the Python loop it replaces.
This is why `train.py` uses `SimEnv` for `--envs 1` and only switches to
`VecSimEnv` above that.

End-to-end rollout collection (`--rollout 2048`, so the sample budget per update
is identical in every row):

| `--envs` | collect steps/s | µs per decision | speedup |
|---|---|---|---|
| 1 | 2,045 | 489 | 1.0× |
| 8 | 9,118 | 110 | 4.5× |
| 32 | 17,210 | 58 | 8.4× |
| 128 | 50,900 | 19.6 | **24.9×** |

The 128 row is quoted for completeness but violates the horizon constraint of
§4.5 ($B/N = 16 < 4 H_{\text{GAE}}$), and `train.py` prints a warning for it.

**Everything in this section is about steps per second, and steps per second
turned out to be the wrong objective.** §7.1 measures what these configurations
actually learn at a fixed sample budget, and the answer reverses the ranking:
$N = 1$ scores 37.0% where $N = 8$ scores 14.5%. The throughput numbers above are
real and reproducible; the inference that a faster rollout is a better one is
not. This is why the section that follows exists.

### 6.3 How much of a batch-1 decision was overhead

Subtracting the environment cost from the total gives the per-decision inference
cost: **395 µs at $N=1$ against 17.4 µs at $N=128$**. The arithmetic per decision
is identical in both cases, so the 378 µs difference is entirely fixed cost that
batching amortises — **96% of what a batch-1 decision cost was not arithmetic.**
That is the measurement behind §6.1's claim, rather than an inference from FLOP
counts.

Inference still dominates collection at every batch size, which is the opposite
of the live backend's profile — there a decision costs ~2% of a 16,667 µs frame
and the game is the bottleneck. Same code, opposite constraints, which is why the
two rollout paths are written separately rather than forced through one
abstraction.

---

## 7. Experiments

Run with `python experiments.py {parity,lam,reward}`. Each config is trained from
scratch for 120 PPO updates (~246k transitions) on three seeds, and scored by the
mean furthest-reached percent over the last 400 episodes.

**On error bars.** Consecutive episodes are strongly correlated — measured lag-20
autocorrelation of 0.63 on `runs/demo` — so the naive standard error of an
$n$-episode mean is optimistic by roughly $\sqrt{(1+\rho)/(1-\rho)} \approx 2.1$.
The intervals below come from a **moving-block bootstrap** (block length 25
episodes), which resamples contiguous runs and so preserves that correlation.
The across-seed spread is also reported, and with three seeds it should be read
as a range, not an interval.

### 7.1 Parallel rollouts cost more than they buy, at a fixed sample budget

`python experiments.py parity`. Every arm trains on the same 245,760 transitions;
only the shape of the rollout differs.

| config | mean % | seed spread | block-bootstrap 95% CI | collect steps/s |
|---|---|---|---|---|
| **scalar, 1 env** | **37.0** | 30.7–41.3 | [29.0, 44.6] | 2,045 |
| vec, 8 envs | 14.5 | 13.1–15.2 | [14.0, 14.9] | 9,118 |
| vec, 32 envs | 14.7 | 14.0–15.1 | [14.2, 15.1] | 17,210 |

This is the opposite of what §6 set out to achieve, and it is not close: the
intervals are nowhere near overlapping, and all three seeds of each arm agree.
**Parallelising the rollout makes the agent learn markedly worse per sample.**

It is not an environment bug. `VecSimEnv` is asserted equal to `SimEnv` step for
step, and the two arms match on every aggregate that does not involve learning:
over 92,160 steps the scalar run completed 548 episodes of mean length 168, the
vectorised run 547 of mean length 167.

Nor is it present from the start. Tracking a matched pair of seed-1 runs:

| update | scalar mean % | entropy | vec mean % | entropy |
|---|---|---|---|---|
| 0 | 11.3 | 0.693 | 11.1 | 0.691 |
| 10 | 12.6 | 0.673 | 13.0 | 0.656 |
| 20 | 12.8 | 0.680 | 14.0 | 0.667 |
| 30 | 12.6 | 0.629 | 13.6 | 0.630 |
| 44 | 15.1 | 0.577 | 13.4 | 0.647 |
| 64 | 28.4 | — | 13.5 | — |
| 119 | **65.1** | — | **13.1** | — |

Wall-clock for those two runs: **152 s** for the scalar arm, **71 s** for the
vectorised one. The vectorised rollout is 2.1× faster end-to-end and finishes
five times worse.

The obvious rejoinder is that a faster backend should simply be given more
samples. It does not rescue it: the same `--envs 8` configuration run out to 277
updates — **2.3× the samples the scalar arm needed to reach 65%, and more
wall-clock than it used** — was still oscillating around 13–15%. This is not a
slower ascent up the same curve; it is a different curve that has flattened.

The two are indistinguishable — if anything the vectorised run leads — for the
first ~45 updates. Then the scalar run's entropy keeps falling and its score
takes off, while the vectorised run's entropy drifts back up and its score does
not. That seed-1 scalar run finished at a 65.1% rolling mean and was **still
climbing steeply** at the cutoff, so 120 updates truncates the scalar arm
mid-ascent and the tail-400 metric lags it; the gap in the table above is if
anything an underestimate.

The shape of that divergence points at *what a batch contains* rather than at
arithmetic. A single-environment rollout of 2048 consecutive steps is a
concentrated run of experience against whichever obstacle is currently blocking
progress; the whole gradient points at solving that one thing. Eight parallel
courses spread the same 2048 samples across eight different blocking obstacles,
and the averaged gradient is a compromise between them. Decorrelating the batch
is normally a virtue — it is the standard argument for parallel actors — but on a
task whose progress is gated by clearing one specific obstacle at a time, the
concentration appears to be worth more than the variance reduction.

That explanation is a hypothesis consistent with the data, not something this
experiment establishes. What the experiment does establish is the cost, and that
is enough to set the default: **`--envs` defaults to 1.** The speedup is real and
remains available for the case where wall-clock, not sample count, is the binding
constraint — but it is opt-in, and the help text says what it costs.

### 7.2 The λ prediction points the right way, but the experiment cannot resolve it

Rerun on the single-environment backend, three seeds, 120 updates each:

| config | $H_{\text{GAE}}$ (steps) | mean % | seed spread | block-bootstrap 95% CI |
|---|---|---|---|---|
| λ = 0.90 | 9.2 | 36.1 | 29.0–40.1 | [27.9, 43.6] |
| λ = 0.95 | 16.8 | 37.0 | 30.7–41.3 | [29.0, 44.6] |
| λ = 0.97 | 24.1 | 37.8 | 29.2–42.8 | [29.4, 46.0] |
| λ = 0.99 | 50.3 | **38.6** | 37.9–39.3 | [31.5, 45.3] |

The ordering is **monotone in λ and in the direction §4.3 predicted** — every
increase in the credit horizon helps, and the arm whose horizon first exceeds the
26-step jump (λ = 0.97, $H_{\text{GAE}} = 24.1$; λ = 0.99, $H_{\text{GAE}} =
50.3$) sits at the top. That is four out of four in the predicted order, which a
coin would manage one time in 24.

It is still not evidence. The whole range spans 2.5 pp while the seed spread
within a single arm spans 11–14 pp. Estimating the across-seed SD from those
ranges ($\sigma \approx 7$ pp), separating a 2.5 pp effect at conventional power
would take

$$n \approx \frac{2\sigma^2 (z_{\alpha/2} + z_\beta)^2}{\Delta^2}
= \frac{2 (7)^2 (2.80)^2}{(2.5)^2} \approx 120 \text{ seeds per arm}$$

— about 21 hours of compute for this sweep, against the 31 minutes it actually
got. **Three seeds cannot resolve this and no amount of bootstrapping the
episodes inside them will change that**, because the dominant variance is
between seeds, not within them.

One thing in the table is not subtle, though: λ = 0.99's seeds land within 1.4 pp
of each other where every other arm spreads over 11–14. With $n = 3$ a range is a
poor statistic and this may be luck, but "a longer credit horizon makes the run
less seed-dependent" is a sharper and cheaper hypothesis to test than the mean
effect, and it is the one worth chasing next.

The honest verdict: **the analysis in §4.3 survives contact with the data and is
not confirmed by it.**

#### The version of this experiment that was wrong

The first run produced this instead:

| config | mean % | block-bootstrap 95% CI |
|---|---|---|
| λ = 0.90 | 13.7 | [13.3, 14.0] |
| λ = 0.95 | 14.5 | [14.0, 14.9] |
| λ = 0.97 | 14.5 | [14.1, 15.0] |
| λ = 0.99 | 14.1 | [13.8, 14.5] |

It is tempting to read that as "λ does not matter, §4.3 was wrong" — and note
that it is *also* monotone-ish and would have supported a story if one had been
wanted. It measured nothing. Every arm ran on `--envs 8`, which §7.1 then showed
lands at ≈14.5% *whatever* it is configured to do; that is the backend's ceiling
at this budget, not a response to λ. An experiment whose arms all sit on a floor
has no dynamic range and cannot resolve anything, and the tight intervals it
reports are a measure of how reliably nothing happened.

This is a methodological mistake worth recording rather than quietly deleting:
the λ and reward experiments were run on a backend before that backend had been
validated, and the parity result that invalidated them came out of the same
batch. `experiments.py` now pins every arm of both to `--envs 1`.

### 7.3 The reward ablation is inconclusive for the same reason

| config | mean % | block-bootstrap 95% CI |
|---|---|---|
| both terms | 14.5 | [14.0, 14.9] |
| progress only (`--death-penalty 0`) | 13.9 | [13.5, 14.3] |
| death only (`--progress-reward 0`) | 14.3 | [13.9, 14.8] |

Removing the progress reward entirely changed nothing measurable — which is what
§3.3 predicts, but it is also what a run that is not learning would produce, and
these runs were not learning. The prediction stands as a prediction. Rerunning on
the single-environment backend, where the dynamic range is 11% → 37%, is the test
that would actually settle it.

---

## 8. What the existing hyperparameter sweep can and cannot support

`runs/sweep-results.json` compares 15 configurations at one seed each, scoring
each by a 200-episode rolling mean. Applying §7's variance analysis to that
metric: on `runs/demo` the per-episode SD of furthest-reached is 39.6 pp, so a
200-episode mean has a naive SE of 2.80 pp, inflating to ≈5.9 pp once the
autocorrelation is accounted for — a 95% interval of about **±11.5 pp per arm**,
and **±16.3 pp for the difference between two arms**.

Against that threshold:

| Config | Δ vs baseline | Resolvable? |
|---|---|---|
| lr 1e-3 | +17.2 | marginally |
| epochs 8 | +14.9 | no |
| rollout 512 | +11.3 | no |
| entropy 0.005 | +8.0 | no |
| hidden 512 | +2.8 | no |
| gamma 0.995 | +0.2 | no |
| entropy 0.02 | −17.1 | yes |
| hidden 128 | −27.2 | yes |
| gamma 0.97 | −32.5 | yes |
| entropy 0.0 | −32.9 | yes |
| lr 1e-4 | −35.4 | yes |
| epochs 2 | −35.8 | yes |

**The sweep reliably identifies what is bad and barely identifies what is good.**
That is the expected shape: bad configurations fail by large margins, good ones
differ by amounts comparable to the noise. The original conclusion — "everything
that buys more gradient steps per sample wins" — is a reasonable reading of the
positive side, but it rests on gaps that a single seed cannot separate, and it
should be treated as a hypothesis rather than a finding.

One caveat in the other direction: some of the measured autocorrelation is
*trend* — the policy really is improving during the window — rather than
stationary noise, so the 2.1× inflation is closer to an upper bound than a point
estimate. The honest summary is that the true intervals lie somewhere between the
naive ±5.5 pp and the corrected ±11.5 pp, and that the ordering of the top four
configurations is not established either way.

---

## 9. Summary of changes this analysis produced

| § | Finding | Change |
|---|---|---|
| 1.1 | Held vs fresh button press was aliased; orbs only fire on a fresh click | `prev_action`, `air_time` added to the observation |
| 2.2 | Dense layer is 93% of parameters, 19% of MACs, and not translation-equivariant | Documented with a concrete $1\times1$-bottleneck proposal; not yet implemented |
| 3.3 | Death term is 4.7× the progress term inside the discount horizon | Reward constants exposed as CLI flags; ablation in §7.3 |
| 3.4 | Completion bonus discounts to $2\times10^{-22}$ | Documented as a scoreboard, not an incentive |
| 4.3 | GAE horizon (16.8 steps) is shorter than one jump (26 steps) | $\lambda$ experiment; §7.2 |
| 4.4 | Auto-reset destroys the frame needed to bootstrap a truncated episode | `info["truncated"]` returns it |
| 4.5 | Parallelism is bounded by the GAE horizon, not by hardware | Warning printed when $B/N < 4H_{\text{GAE}}$ |
| 5.2 | Value function could drift outside the advantages' trust region | Clipped value loss |
| 5.3 | KL estimator had ±51% error at the threshold | $k_3$ estimator |
| 5.4 | Trust-region brake engaged at up to 2× the configured KL | Per-epoch measurement |
| 5.5 | No way to tell a working critic from a broken one | Explained variance, logged and plotted |
| 6.1 | Batch 1 wastes 96% of the per-decision cost on overhead | Vectorised simulator; `auto` device now batch-aware |
| 7.1 | …but the vectorised rollout is 2.5× worse **per sample** | `--envs` defaults to 1; the speedup is opt-in and its cost is in the help text |
| 7.2 | Two experiments were run on an unvalidated backend and had no dynamic range | `experiments.py` pins them to `--envs 1`; broken versions kept alongside the reruns |
| 7.2 | Rerun, λ is monotone in the predicted direction but 2.5 pp against 7 pp of seed noise | Reported as unresolved; ~120 seeds/arm would be needed |

## 10. Open questions

* **Why exactly does the parallel rollout lose?** §7.1 measures the cost and
  offers a batch-composition story, but does not test it. Two experiments would
  separate the candidates: hold the number of parallel courses at 8 while giving
  every environment the *same* course seed (isolating decorrelation from course
  diversity), and sweep `--rollout` upward at fixed `--envs` (isolating segment
  length from parallelism). Until then the mechanism is a guess.
* **Does a longer credit horizon mainly raise the mean, or mainly shrink the
  seed-to-seed spread?** §7.2 found λ = 0.99's three seeds within 1.4 pp of each
  other where every other arm spread over 11–14 pp. If that survives more seeds it
  is both a bigger effect than the 2.5 pp mean shift and far cheaper to detect,
  since variance differences need fewer samples than mean differences of this size.
* **Does the $1\times1$ bottleneck (§2.2) hold up?** It should cut the parameter
  count 3× with no loss of spatial information. Untested because it invalidates
  existing checkpoints.
* **Does any of this transfer to the live game?** Every experiment here runs
  against the simulator, whose cube-only physics is a strict subset of Geometry
  Dash. The simulator has no ship, no wave, no orbs, no portals — and two of the
  four occupancy channels are never populated. A $\lambda$ that is right for a
  4.46-block jump arc may be wrong for a ship segment.
* **Is the entropy floor safe at scale?** A short vectorised run at
  `--rollout 512 --envs 8` collapsed to $\mathcal{H} = 0.003$ (deterministic
  release) by update 30 and never recovered, at a configuration that also
  violated §4.5's horizon bound. Whether that is the horizon violation, the
  entropy coefficient, or both has not been separated.

"""PPO — rollout buffers, GAE, and the clipped-surrogate update.

Deliberately small and dependency-light (torch + numpy). The rollout buffers are
preallocated because a live rollout runs inside the frame handshake: every
allocation in that loop is latency the mod spends waiting on us, and a frame we
answer late is a frame the game spends idle.

Two buffers, one update path. `Rollout` is the flat single-environment buffer the
live game fills one frame at a time; `VecRollout` is the (T, N) buffer the
vectorised simulator fills N rows at a time. Both expose `flat()`, so `update()`
never needs to know which one it was handed.

What changed from the first version, and why:

* **The KL brake now measures one epoch, not the whole update.** It used to
  compare `sum(kl) / batches` — an average over every minibatch since the update
  began — against `target_kl`. Epoch 1's KL is near zero by construction, so that
  average stays below the threshold long after the *current* policy has walked
  well past it. The brake fired late or not at all, which is the opposite of what
  a trust region is for.
* **KL is the k3 estimator, not k1.** Both are unbiased for KL(pi_old || pi_new),
  but k1 = E[-log r] has high variance and goes negative on ordinary samples,
  which is a poor thing to threshold. k3 = E[r - 1 - log r] adds a mean-zero
  control variate, is provably non-negative term by term, and costs one
  subtraction. See docs/mathematical-report.md section 5.3.
* **The value loss is clipped** the same way the policy loss is, so one update
  cannot move the critic arbitrarily far from the values the advantages were
  computed against.
* **Explained variance is reported.** It is the one number that says whether the
  critic is doing anything: at 0 the value head is no better than predicting the
  mean return, and every advantage is pure Monte-Carlo noise.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class Rollout:
    """A fixed-size buffer of transitions, preallocated once and reused."""

    def __init__(self, size: int, grid_shape, n_scalars: int):
        self.size = size
        self.grid_shape = tuple(grid_shape)
        self.grid = np.zeros((size, *grid_shape), dtype=np.float32)
        self.scalars = np.zeros((size, n_scalars), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.logp = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.n = 0

    def add(self, obs, action, logp, value, reward, done) -> None:
        i = self.n
        self.grid[i] = obs.grid
        self.scalars[i] = obs.scalars
        self.actions[i] = action
        self.logp[i] = logp
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = 1.0 if done else 0.0
        self.n += 1

    @property
    def full(self) -> bool:
        return self.n >= self.size

    @property
    def transitions(self) -> int:
        return self.n

    def reset(self) -> None:
        self.n = 0

    def flat(self):
        """(grid, scalars, actions, logp, values) as one flat batch."""
        n = self.n
        return (self.grid[:n], self.scalars[:n], self.actions[:n],
                self.logp[:n], self.values[:n])


class VecRollout:
    """The same buffer for N environments stepped together: everything is (T, N).

    Kept as a separate class rather than a `n_envs=1` mode on `Rollout` because
    the live backend's buffer is genuinely one-dimensional and giving it a
    trailing axis of 1 would put a reshape in the frame handshake for nothing.
    """

    def __init__(self, size: int, n_envs: int, grid_shape, n_scalars: int):
        self.size = size
        self.n_envs = n_envs
        self.grid_shape = tuple(grid_shape)
        self.grid = np.zeros((size, n_envs, *grid_shape), dtype=np.float32)
        self.scalars = np.zeros((size, n_envs, n_scalars), dtype=np.float32)
        self.actions = np.zeros((size, n_envs), dtype=np.int64)
        self.logp = np.zeros((size, n_envs), dtype=np.float32)
        self.values = np.zeros((size, n_envs), dtype=np.float32)
        self.rewards = np.zeros((size, n_envs), dtype=np.float32)
        self.dones = np.zeros((size, n_envs), dtype=np.float32)
        self.n = 0

    def add(self, grid, scalars, actions, logp, values, rewards, dones) -> None:
        i = self.n
        self.grid[i] = grid
        self.scalars[i] = scalars
        self.actions[i] = actions
        self.logp[i] = logp
        self.values[i] = values
        self.rewards[i] = rewards
        self.dones[i] = dones
        self.n += 1

    @property
    def full(self) -> bool:
        return self.n >= self.size

    @property
    def transitions(self) -> int:
        return self.n * self.n_envs

    def reset(self) -> None:
        self.n = 0

    def flat(self):
        n = self.n
        return (self.grid[:n].reshape(-1, *self.grid_shape),
                self.scalars[:n].reshape(n * self.n_envs, -1),
                self.actions[:n].reshape(-1),
                self.logp[:n].reshape(-1),
                self.values[:n].reshape(-1))


def compute_gae(rewards, values, dones, last_value: float,
                gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over one rollout."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae, next_value = 0.0, last_value
    for t in reversed(range(n)):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
        next_value = values[t]
    return adv, adv + np.asarray(values, dtype=np.float32)


def compute_gae_vec(rewards, values, dones, last_values,
                    gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """GAE over a (T, N) rollout — the same recursion, N accumulators at once.

    Each column is an independent environment, so credit never crosses between
    them, and `dones` already stops it crossing an episode boundary within one.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    t_steps = rewards.shape[0]
    adv = np.zeros_like(rewards)
    gae = np.zeros(rewards.shape[1], dtype=np.float32)
    next_values = np.asarray(last_values, dtype=np.float32)
    for t in reversed(range(t_steps)):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
        next_values = values[t]
    return adv, adv + values


def explained_variance(values, returns) -> float:
    """1 - Var(returns - values) / Var(returns).

    1.0 is a perfect critic, 0.0 is a critic no better than the mean return, and
    a negative number means the value head is actively worse than a constant —
    which, since advantages are `return - value`, means the policy gradient is
    being steered by noise.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    returns = np.asarray(returns, dtype=np.float64).ravel()
    var = returns.var()
    if var < 1e-12:
        return 0.0
    return float(1.0 - (returns - values).var() / var)


def anneal(start: float, end: float, frac: float) -> float:
    """Linear schedule. `frac` runs 0 -> 1 over the run."""
    frac = min(max(frac, 0.0), 1.0)
    return start + (end - start) * frac


def update(model, optim, buf, adv, ret, *, epochs: int = 4,
           clip: float = 0.2, batch: int = 256, ent_coef: float = 0.01,
           vf_coef: float = 0.5, max_grad_norm: float = 0.5,
           target_kl: float = 0.03, clip_value: bool = True,
           lr: Optional[float] = None, collect_grads: bool = False) -> dict:
    """One PPO update over a collected rollout. Returns a dict of diagnostics.

    Stops early once the *current* policy has moved `target_kl` away from the one
    that collected the data — with a live game behind the environment, wasting
    epochs on a stale rollout is expensive in wall-clock, not just in sample
    efficiency.
    """
    if lr is not None:
        for group in optim.param_groups:
            group["lr"] = lr

    d = model.device
    g_np, s_np, a_np, lp_np, v_np = buf.flat()
    n = len(a_np)
    grid = torch.as_tensor(g_np, device=d)
    scalars = torch.as_tensor(s_np, device=d)
    actions = torch.as_tensor(a_np, device=d)
    old_logp = torch.as_tensor(lp_np, device=d)
    old_v = torch.as_tensor(v_np, device=d)

    adv = np.asarray(adv, dtype=np.float32).ravel()
    ret = np.asarray(ret, dtype=np.float32).ravel()
    adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=d)
    ret_t = torch.as_tensor(ret, device=d)

    stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0,
             "kl_k1": 0.0, "clip_frac": 0.0, "grad_norm": 0.0, "epochs": 0,
             "lr": float(optim.param_groups[0]["lr"]),
             "explained_variance": explained_variance(v_np, ret),
             "adv_mean": float(adv.mean()), "adv_std": float(adv.std()),
             "samples": int(n), "stopped_early": 0}
    batches = 0
    grads: dict = {}

    for epoch in range(epochs):
        perm = torch.randperm(n, device=d)
        epoch_kl, epoch_batches = 0.0, 0
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            logp, entropy, value = model.evaluate(grid[b], scalars[b], actions[b])
            log_ratio = logp - old_logp[b]
            ratio = torch.exp(log_ratio)

            s1 = ratio * adv_t[b]
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[b]
            pi_loss = -torch.min(s1, s2).mean()

            if clip_value:
                # Same trust region as the policy: the critic may not move
                # further than `clip` from the values the advantages were built
                # on, so one update cannot invalidate its own targets.
                v_clipped = old_v[b] + torch.clamp(value - old_v[b], -clip, clip)
                v_loss = torch.max((value - ret_t[b]) ** 2,
                                   (v_clipped - ret_t[b]) ** 2).mean()
            else:
                v_loss = ((value - ret_t[b]) ** 2).mean()

            ent = entropy.mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent

            optim.zero_grad(set_to_none=True)
            loss.backward()
            # Read the gradient before clipping — after it, every norm is either
            # the true one or exactly max_grad_norm, which tells you nothing.
            total_norm = float(nn.utils.clip_grad_norm_(model.parameters(),
                                                        max_grad_norm))
            # Per-layer norms are only for the viewer, so an unwatched run does
            # not pay for them — the same bargain the telemetry channel makes.
            if collect_grads and hasattr(model, "grad_snapshot"):
                for k, v in model.grad_snapshot().items():
                    grads[k] = grads.get(k, 0.0) + v
            optim.step()

            with torch.no_grad():
                # k3: unbiased like k1, but non-negative and far lower variance.
                kl = ((ratio - 1.0) - log_ratio).mean()
                kl_k1 = (-log_ratio).mean()
                clipped = ((ratio - 1.0).abs() > clip).float().mean()
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["kl"] += kl.item()
                stats["kl_k1"] += kl_k1.item()
                stats["clip_frac"] += clipped.item()
                stats["grad_norm"] += total_norm
                epoch_kl += kl.item()
                epoch_batches += 1
            batches += 1

        stats["epochs"] = epoch + 1
        # Measured over *this* epoch only. Averaging across epochs dilutes the
        # signal with epoch 1's near-zero KL and the brake never engages.
        if epoch_batches and epoch_kl / epoch_batches > target_kl:
            stats["stopped_early"] = 1
            break

    if batches:
        for k in ("pi_loss", "v_loss", "entropy", "kl", "kl_k1", "clip_frac",
                  "grad_norm"):
            stats[k] /= batches
        stats["grads"] = {k: v / batches for k, v in grads.items()}
    else:
        stats["grads"] = {}
    return stats

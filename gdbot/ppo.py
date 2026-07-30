"""PPO — rollout buffer, GAE, and the clipped-surrogate update.

Deliberately small and dependency-light (torch + numpy). The rollout buffer is
preallocated because a live rollout runs inside the frame handshake: every
allocation in that loop is latency the mod spends waiting on us, and a frame we
answer late is a frame the game spends idle.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class Rollout:
    """A fixed-size buffer of transitions, preallocated once and reused."""

    def __init__(self, size: int, grid_shape, n_scalars: int):
        self.size = size
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

    def reset(self) -> None:
        self.n = 0


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


def update(model, optim, buf: Rollout, adv, ret, *, epochs: int = 4,
           clip: float = 0.2, batch: int = 256, ent_coef: float = 0.01,
           vf_coef: float = 0.5, max_grad_norm: float = 0.5,
           target_kl: float = 0.03) -> dict:
    """One PPO update over a collected rollout. Returns a dict of diagnostics.

    Stops early once the policy has moved `target_kl` away from the one that
    collected the data — with a live game behind the environment, wasting epochs
    on a stale rollout is expensive in wall-clock, not just in sample efficiency.
    """
    d = model.device
    n = buf.n
    grid = torch.as_tensor(buf.grid[:n], device=d)
    scalars = torch.as_tensor(buf.scalars[:n], device=d)
    actions = torch.as_tensor(buf.actions[:n], device=d)
    old_logp = torch.as_tensor(buf.logp[:n], device=d)
    adv = np.asarray(adv, dtype=np.float32)
    adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=d)
    ret_t = torch.as_tensor(np.asarray(ret, dtype=np.float32), device=d)

    stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0,
             "clip_frac": 0.0, "epochs": 0}
    batches = 0
    for epoch in range(epochs):
        perm = torch.randperm(n, device=d)
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            logp, entropy, value = model.evaluate(grid[b], scalars[b], actions[b])
            ratio = torch.exp(logp - old_logp[b])
            s1 = ratio * adv_t[b]
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[b]
            pi_loss = -torch.min(s1, s2).mean()
            v_loss = ((value - ret_t[b]) ** 2).mean()
            ent = entropy.mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optim.step()

            with torch.no_grad():
                kl = (old_logp[b] - logp).mean()
                clipped = ((ratio - 1.0).abs() > clip).float().mean()
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["kl"] += kl.item()
                stats["clip_frac"] += clipped.item()
            batches += 1

        stats["epochs"] = epoch + 1
        if batches and stats["kl"] / batches > target_kl:
            break

    if batches:
        for k in ("pi_loss", "v_loss", "entropy", "kl", "clip_frac"):
            stats[k] /= batches
    return stats

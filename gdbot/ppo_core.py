"""Minimal, self-contained PPO for the jump/no-jump policy.

Works on any GDEnv (SimEnv for offline dev, LiveEnv for the real game). The
policy is a small MLP: observation -> 2 logits (no-jump / jump). Kept tiny and
dependency-light (just torch + numpy) so it can be warm-started by cloning a NEAT
champion and then fine-tuned to push past the plateau.
"""

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi_head = nn.Linear(hidden, 2)   # logits: [no-jump, jump]
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi_head(h), self.v_head(h).squeeze(-1)


@torch.no_grad()
def act(model: ActorCritic, obs, greedy: bool = False):
    """Return (action, log_prob, value) for a single observation."""
    x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    logits, v = model(x)
    dist = torch.distributions.Categorical(logits=logits)
    a = torch.argmax(logits) if greedy else dist.sample()
    return int(a.item()), float(dist.log_prob(a).item()), float(v.item())


def compute_gae(rews, vals, dones, last_val, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation over one rollout."""
    n = len(rews)
    adv = np.zeros(n, dtype=np.float32)
    gae, next_val = 0.0, last_val
    for t in reversed(range(n)):
        nonterm = 1.0 - dones[t]
        delta = rews[t] + gamma * next_val * nonterm - vals[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
        next_val = vals[t]
    ret = adv + np.asarray(vals, dtype=np.float32)
    return adv, ret


def ppo_update(model, optim, obs, acts, old_logp, adv, ret,
               epochs=4, clip=0.2, batch=256, ent_coef=0.01, vf_coef=0.5):
    """One PPO update over a collected rollout. Returns (pi_loss, v_loss)."""
    obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    acts = torch.as_tensor(np.asarray(acts), dtype=torch.int64)
    old_logp = torch.as_tensor(np.asarray(old_logp), dtype=torch.float32)
    adv = np.asarray(adv, dtype=np.float32)
    adv = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32)
    ret = torch.as_tensor(np.asarray(ret), dtype=torch.float32)

    n = len(obs)
    pi_loss = v_loss = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            logits, v = model(obs[b])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(acts[b])
            ratio = torch.exp(logp - old_logp[b])
            s1 = ratio * adv[b]
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[b]
            pl = -torch.min(s1, s2).mean()
            vl = ((v - ret[b]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pl + vf_coef * vl - ent_coef * ent
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            pi_loss, v_loss = float(pl.item()), float(vl.item())
    return pi_loss, v_loss


def behavioral_clone(model, obs_list, act_list, epochs=8, batch=256, lr=1e-3):
    """Supervised warm-start: train the policy head to imitate (obs -> action)
    pairs collected from a NEAT champion, so PPO starts near the plateau."""
    if not obs_list:
        return
    X = torch.as_tensor(np.asarray(obs_list), dtype=torch.float32)
    Y = torch.as_tensor(np.asarray(act_list), dtype=torch.int64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            logits, _ = model(X[b])
            loss = loss_fn(logits, Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return float(loss.item())

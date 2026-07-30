"""The policy: a small conv net over the occupancy grid, plus a scalar branch.

The grid is a picture, so the network that reads it is a convnet — three conv
layers shrink 24x16 down to 6x4 while widening the channels, the flattened result
is concatenated with the kinematic scalars, and a shared trunk feeds a policy head
(hold / release) and a value head.

This is the part NEAT could not do. Evolving topologies over 1536 grid inputs is
hopeless, so the old live agent was fed 22 hand-picked numbers and could not see
saw blades, orbs or pads at all. A conv net reads the whole picture, and weight
sharing means "spike two blocks ahead at head height" is learned once rather than
once per position.

`forward(..., introspect=True)` also returns every intermediate activation. That
is what the viewer draws, and it costs nothing when nobody is watching because
the trainer only asks for it when a browser is actually connected.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .obs import GRID_SHAPE, N_SCALARS

# How many feature maps the viewer is allowed to show per conv stage. Keeping a
# cap here rather than in the viewer means we never serialise more than we draw.
VIZ_MAPS = 16


def _init(layer: nn.Module, gain: float = np.sqrt(2), bias: float = 0.0):
    """Orthogonal init — the standard PPO recipe, and it matters for stability."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class ConvActorCritic(nn.Module):
    def __init__(self, grid_shape: Tuple[int, int, int] = GRID_SHAPE,
                 n_scalars: int = N_SCALARS, hidden: int = 256):
        super().__init__()
        c, h, w = grid_shape
        self.grid_shape = grid_shape
        self.n_scalars = n_scalars

        self.conv1 = _init(nn.Conv2d(c, 16, 3, stride=1, padding=1))
        self.conv2 = _init(nn.Conv2d(16, 32, 3, stride=2, padding=1))
        self.conv3 = _init(nn.Conv2d(32, 32, 3, stride=2, padding=1))
        self.relu = nn.ReLU()

        with torch.no_grad():   # derive the flat size instead of hard-coding it
            probe = self._conv(torch.zeros(1, c, h, w))
            self.map_shapes = [tuple(t.shape[1:]) for t in probe]
            self.conv_out = int(np.prod(probe[-1].shape[1:]))

        self.fc = _init(nn.Linear(self.conv_out + n_scalars, hidden))
        self.pi = _init(nn.Linear(hidden, 2), gain=0.01)   # small: start near-uniform
        self.v = _init(nn.Linear(hidden, 1), gain=1.0)

    # --- forward -------------------------------------------------------------
    def _conv(self, g: torch.Tensor):
        a1 = self.relu(self.conv1(g))
        a2 = self.relu(self.conv2(a1))
        a3 = self.relu(self.conv3(a2))
        return [a1, a2, a3]

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor,
                introspect: bool = False):
        maps = self._conv(grid)
        flat = maps[-1].flatten(1)
        h = self.relu(self.fc(torch.cat([flat, scalars], dim=1)))
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        if not introspect:
            return logits, value, None
        return logits, value, {"maps": maps, "hidden": h}

    # --- helpers used by the trainer -----------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _tensors(self, grid, scalars):
        d = self.device
        g = torch.as_tensor(np.asarray(grid), dtype=torch.float32, device=d)
        s = torch.as_tensor(np.asarray(scalars), dtype=torch.float32, device=d)
        if g.dim() == 3:
            g, s = g.unsqueeze(0), s.unsqueeze(0)
        return g, s

    @torch.no_grad()
    def act(self, obs, greedy: bool = False, introspect: bool = False):
        """One decision. Returns (action, log_prob, value, viz-or-None)."""
        g, s = self._tensors(obs.grid, obs.scalars)
        logits, value, inner = self(g, s, introspect=introspect)
        dist = torch.distributions.Categorical(logits=logits)
        a = torch.argmax(logits, dim=-1) if greedy else dist.sample()
        viz = self._pack_viz(logits, value, inner) if introspect else None
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item()), viz

    def evaluate(self, grid, scalars, actions):
        """Batched log-probs / entropy / values for a PPO update."""
        logits, value, _ = self(grid, scalars)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    # --- introspection for the viewer ----------------------------------------
    def _pack_viz(self, logits, value, inner) -> dict:
        """Quantise activations to uint8 so a frame is a few KB, not a few MB.

        Each map is scaled by its own max, which is what makes the picture
        readable: absolute ReLU magnitudes drift over training, but "which cells
        of this filter fired hardest right now" is the thing worth watching.
        """
        probs = torch.softmax(logits, dim=-1)[0]
        out = {
            "probs": [float(p) for p in probs],
            "value": float(value.item()),
            "hidden": _quantise(inner["hidden"][0]),
            # The decision layer is the one place where individual weights still
            # mean something to a human, so it goes over signed for the viewer to
            # draw as green/red edges — the MarI/O picture, at the only layer
            # where that picture is honest.
            "head": _quantise_signed(self.pi.weight),
            "stages": [],
        }
        for m in inner["maps"]:
            m = m[0][:VIZ_MAPS]                  # (n, h, w)
            out["stages"].append({
                "n": int(m.shape[0]), "h": int(m.shape[1]), "w": int(m.shape[2]),
                "data": _quantise(m.flatten()),
            })
        return out

    def head_weights(self) -> np.ndarray:
        """Policy-head weights (2, hidden) — the one layer worth drawing as edges."""
        return self.pi.weight.detach().cpu().numpy()


def _quantise(t: torch.Tensor) -> np.ndarray:
    """Magnitudes -> 0..255, scaled by the tensor's own peak."""
    a = t.detach().cpu().numpy().astype(np.float32)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak <= 1e-8:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip(np.abs(a) / peak * 255.0, 0, 255).astype(np.uint8)


def _quantise_signed(t: torch.Tensor) -> np.ndarray:
    """Signed values -> 0..255 around a midpoint of 128, so sign survives."""
    a = t.detach().cpu().numpy().astype(np.float32).ravel()
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak <= 1e-8:
        return np.full(a.shape, 128, dtype=np.uint8)
    return np.clip(a / peak * 127.0 + 128.0, 0, 255).astype(np.uint8)


def save(model: ConvActorCritic, path: str, extra: Optional[dict] = None) -> None:
    torch.save({"model": model.state_dict(),
                "grid_shape": model.grid_shape,
                "n_scalars": model.n_scalars,
                **(extra or {})}, path)


def load(path: str, device: str = "cpu") -> Tuple[ConvActorCritic, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ConvActorCritic(tuple(ck["grid_shape"]), ck["n_scalars"]).to(device)
    model.load_state_dict(ck["model"])
    return model, ck

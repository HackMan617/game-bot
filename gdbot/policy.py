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

    def _fit(self, s: torch.Tensor) -> torch.Tensor:
        """Match the scalar vector to the width this network was built for.

        The observation gained two scalars (prev_action, air_time) after the
        first live run. Rather than orphan every checkpoint trained before that,
        a narrower network simply ignores the tail and a wider one sees zeros —
        one place, so no caller has to know which era a checkpoint came from.
        """
        have = s.shape[-1]
        if have == self.n_scalars:
            return s
        if have > self.n_scalars:
            return s[..., :self.n_scalars]
        pad = torch.zeros(*s.shape[:-1], self.n_scalars - have,
                          dtype=s.dtype, device=s.device)
        return torch.cat([s, pad], dim=-1)

    def _tensors(self, grid, scalars):
        d = self.device
        g = torch.as_tensor(np.asarray(grid), dtype=torch.float32, device=d)
        s = torch.as_tensor(np.asarray(scalars), dtype=torch.float32, device=d)
        if g.dim() == 3:
            g, s = g.unsqueeze(0), s.unsqueeze(0)
        return g, self._fit(s)

    @torch.no_grad()
    def act(self, obs, greedy: bool = False, introspect: bool = False):
        """One decision. Returns (action, log_prob, value, viz-or-None)."""
        g, s = self._tensors(obs.grid, obs.scalars)
        logits, value, inner = self(g, s, introspect=introspect)
        dist = torch.distributions.Categorical(logits=logits)
        a = torch.argmax(logits, dim=-1) if greedy else dist.sample()
        viz = self._pack_viz(logits, value, inner) if introspect else None
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item()), viz

    @torch.no_grad()
    def act_batch(self, grid: np.ndarray, scalars: np.ndarray,
                  greedy: bool = False):
        """N decisions in one forward pass. Returns numpy (a, logp, value).

        This is the whole point of the vectorised environment: at batch 1 this
        network is kernel-launch bound and CUDA is measurably *slower* than the
        CPU, but the same 216k parameters amortise across a batch almost for
        free. One call here replaces N calls to `act`.
        """
        g, s = self._tensors(grid, scalars)
        logits, value, _ = self(g, s)
        dist = torch.distributions.Categorical(logits=logits)
        a = torch.argmax(logits, dim=-1) if greedy else dist.sample()
        return (a.cpu().numpy().astype(np.int64),
                dist.log_prob(a).cpu().numpy().astype(np.float32),
                value.cpu().numpy().astype(np.float32))

    @torch.no_grad()
    def viz_one(self, grid, scalars) -> dict:
        """The introspection payload for a single state, without re-deciding.

        The vectorised trainer has already sampled its actions in one batched
        pass; calling `act(introspect=True)` to draw the picture would sample a
        *second*, different action and the viewer would show a decision the game
        never received. This just re-runs the forward for one state and packs the
        activations.
        """
        g, s = self._tensors(grid, scalars)
        logits, value, inner = self(g, s, introspect=True)
        return self._pack_viz(logits, value, inner)

    def evaluate(self, grid, scalars, actions):
        """Batched log-probs / entropy / values for a PPO update."""
        logits, value, _ = self(grid, self._fit(scalars))
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

    def saliency(self, grid, scalars, action: Optional[int] = None) -> np.ndarray:
        """Which cells of the grid actually drove this decision. Returns (H, W).

        This is the gradient of the chosen action's log-probability with respect
        to the occupancy grid, |d log pi(a|s) / d grid|, summed over the four
        channels. Feature maps show what each filter responds to; this shows what
        the *decision* depended on, which is the question a human watching
        actually has. A policy that has learned to see hazards lights up on the
        spike it is about to jump; one that is still guessing lights up on the
        floor it is standing on, or on nothing in particular.

        `torch.autograd.grad` rather than `.backward()` on purpose: this runs
        inside rollout collection, and backward() would leave gradients sitting
        on the parameters for the next PPO update to trip over.
        """
        g, s = self._tensors(grid, scalars)
        g = g.detach().requires_grad_(True)
        logits, _value, _ = self(g, s)
        if action is None:
            action = int(torch.argmax(logits, dim=-1).item())
        chosen = torch.log_softmax(logits, dim=-1)[0, int(action)]
        grad = torch.autograd.grad(chosen, g)[0]
        return _quantise(grad[0].abs().sum(0))          # (C,H,W) -> (H,W)

    @torch.no_grad()
    def grad_snapshot(self) -> dict:
        """Per-layer gradient norms, read between backward() and the optimiser step.

        Gradient norm is the honest answer to "is this layer still learning?".
        A conv stage whose gradient has collapsed two orders of magnitude below
        the dense layer's is, for practical purposes, frozen.
        """
        return {name: (float(torch.linalg.vector_norm(w.grad))
                       if w.grad is not None else 0.0)
                for name, w in self.named_layers()}

    # --- watching it learn ---------------------------------------------------
    # Everything above shows what the network SEES on this frame. The rest of this
    # class shows how the network itself is CHANGING — the part you actually want
    # when the question is "is it learning?" rather than "what is it looking at?".

    def named_layers(self):
        """The weight tensors worth reporting on, in forward order."""
        return [("conv1", self.conv1.weight), ("conv2", self.conv2.weight),
                ("conv3", self.conv3.weight), ("dense", self.fc.weight),
                ("policy", self.pi.weight), ("value", self.v.weight)]

    @torch.no_grad()
    def weight_fingerprint(self) -> dict:
        """A cheap copy of every layer's weights, for diffing against later."""
        return {n: w.detach().clone() for n, w in self.named_layers()}

    @torch.no_grad()
    def learning_snapshot(self, previous: Optional[dict] = None) -> dict:
        """How the weights look now, and how far they moved since `previous`.

        conv1's kernels go over in full: they are only 16x4x3x3, they are the one
        layer whose weights are directly interpretable as picture detectors, and
        watching them sharpen out of noise is the clearest visual signal that
        learning is happening at all.
        """
        layers = []
        for name, w in self.named_layers():
            wn = w.detach()
            delta = 0.0
            if previous is not None and name in previous:
                delta = float(torch.linalg.vector_norm(wn - previous[name]))
            layers.append({
                "name": name,
                "shape": list(wn.shape),
                "norm": float(torch.linalg.vector_norm(wn)),
                "delta": delta,
                "std": float(wn.std()),
                "absmax": float(wn.abs().max()),
            })

        k = self.conv1.weight.detach()            # (out, in, kh, kw)
        return {
            "layers": layers,
            "kernels": {
                "n": int(k.shape[0]), "c": int(k.shape[1]),
                "kh": int(k.shape[2]), "kw": int(k.shape[3]),
                "data": _quantise_signed(k),
            },
        }


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
    # The width goes in the checkpoint too, so a run trained at a non-default
    # width reloads instead of failing on a shape mismatch.
    torch.save({"model": model.state_dict(),
                "grid_shape": model.grid_shape,
                "n_scalars": model.n_scalars,
                "hidden": int(model.fc.out_features),
                **(extra or {})}, path)


def load(path: str, device: str = "cpu") -> Tuple[ConvActorCritic, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ConvActorCritic(tuple(ck["grid_shape"]), ck["n_scalars"],
                            hidden=int(ck.get("hidden", 256))).to(device)
    model.load_state_dict(ck["model"])
    return model, ck

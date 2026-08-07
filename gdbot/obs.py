"""The observation contract every backend produces and the policy consumes.

One observation is a pair:

    grid    (C, H, W) float32 occupancy — what the bot *sees* ahead of it
    scalars (S,)      float32 kinematics and mode flags — how it *moves*

The grid comes straight from the mod (`bridge.GRID_*`) so its layout is the same
in Sim and Live, and it is channel-first so it feeds a conv net with no reshape.

Nothing here encodes *where in the level* the player is — no x, no percent. That
is deliberate: the goal is a reactive generalist, and a policy that can read the
clock can memorise a level instead of learning to see it.

`build_scalars` is shape-polymorphic: pass floats and it returns `(S,)`, pass
arrays of length N and it returns `(N, S)`. That is what lets the single-env
backends and the vectorised simulator share one definition of the input scaling
instead of maintaining two that can drift apart.
"""

from dataclasses import dataclass

import numpy as np

from .bridge import CELL, GRID_BEHIND, GRID_C, GRID_H, GRID_W

# Gamemode ids — must match `currentMode()` in geode-mod/src/main.cpp.
CUBE, SHIP, BALL, UFO, WAVE, ROBOT, SPIDER, SWING = range(8)
N_MODES = 8
MODE_NAMES = ("cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing")

# Grid channels, in the order the mod writes them.
CHANNEL_NAMES = ("solid", "hazard", "orb/pad", "portal")

GRID_SHAPE = (GRID_C, GRID_H, GRID_W)
PLAYER_COL = GRID_BEHIND      # the player's own column in the grid
PLAYER_ROW = GRID_H // 2      # the player's own row

# Scalar layout. Order is fixed — the viewer labels bars with these names.
#
# `prev_action` and `air_time` were added after the first live run. They close a
# genuine hole in the state: GD activates an orb on a *fresh* click and ignores a
# held one, so "holding" and "pressing" are different states of the world that
# the old vector rendered identically. Without them the process the policy sees
# is not Markov and no amount of training can fix that. `air_time` is the same
# argument for timing: vy is 0 both on the ground and at the apex of a jump, and
# while `on_ground` separates those two, only a clock says how long a fall has
# been going.
SCALAR_NAMES = (
    "on_ground", "vy", "upside_down", "speed", "gravity", "mini",
    "floor_gap", "has_ceiling", "ceiling_gap",
    "prev_action", "air_time",
) + tuple(f"mode:{m}" for m in MODE_NAMES)
N_SCALARS = len(SCALAR_NAMES)
MODE0 = SCALAR_NAMES.index("mode:cube")    # first index of the gamemode one-hot

_VY_NORM = 1000.0             # GD y-velocity units/sec at which we saturate
_GAP_NORM = CELL * (GRID_H / 2)   # half a grid height, in GD units
_AIR_NORM = 60.0              # frames airborne at which air_time saturates (1s)


@dataclass(frozen=True)
class Obs:
    """One frame as the network sees it."""

    grid: np.ndarray      # (C, H, W) float32, 0.0 / 1.0
    scalars: np.ndarray   # (S,) float32

    def __post_init__(self):
        assert self.grid.shape == GRID_SHAPE, f"grid {self.grid.shape} != {GRID_SHAPE}"
        assert self.scalars.shape == (N_SCALARS,), f"scalars {self.scalars.shape}"


def build_scalars(*, on_ground, vy, upside_down, player_speed, gravity_mod,
                  vehicle_size, y, ground_y, ceiling_y, gamemode,
                  prev_action=0.0, air_time=0.0) -> np.ndarray:
    """Pack per-frame kinematics into the fixed scalar vector.

    Every environment calls this, so Sim, Live and VecSim can never drift into
    producing differently-scaled inputs for the same physical situation.
    Distances are in GD units (1 block = 30), which is why the simulators convert
    their block-space velocities before calling in.

    Scalar arguments give a `(S,)` vector; arrays of length N give `(N, S)`.
    """
    on_ground = np.asarray(on_ground, dtype=np.float32)
    batch = on_ground.shape
    s = np.zeros((*batch, N_SCALARS), dtype=np.float32)

    s[..., 0] = on_ground
    s[..., 1] = np.clip(np.asarray(vy, np.float32) / _VY_NORM, -1.0, 1.0)
    s[..., 2] = np.asarray(upside_down, np.float32)
    # 0.7..1.6x -> -0.3..0.6
    s[..., 3] = np.clip(np.asarray(player_speed, np.float32) - 1.0, -1.0, 1.0)
    s[..., 4] = np.clip(np.asarray(gravity_mod, np.float32) / 2.0, -1.0, 1.0)
    # 0 normal, -0.5 mini
    s[..., 5] = np.clip(np.asarray(vehicle_size, np.float32) - 1.0, -1.0, 1.0)
    s[..., 6] = np.clip((np.asarray(y, np.float32) - np.asarray(ground_y, np.float32))
                        / _GAP_NORM, 0.0, 2.0)
    ceiling_y = np.asarray(ceiling_y, np.float32)
    has_ceiling = ceiling_y > 0.0
    s[..., 7] = has_ceiling.astype(np.float32)
    s[..., 8] = np.where(has_ceiling,
                         np.clip((ceiling_y - np.asarray(y, np.float32)) / _GAP_NORM,
                                 0.0, 2.0), 0.0)
    s[..., 9] = np.clip(np.asarray(prev_action, np.float32), 0.0, 1.0)
    s[..., 10] = np.clip(np.asarray(air_time, np.float32) / _AIR_NORM, 0.0, 1.0)

    # One-hot the gamemode without a Python branch, so the batched path stays
    # vectorised and the single-frame path lands on the identical index.
    mode = np.asarray(gamemode, dtype=np.int64)
    valid = ((mode >= 0) & (mode < N_MODES)).astype(np.float32)
    idx = MODE0 + np.clip(mode, 0, N_MODES - 1)
    np.put_along_axis(s, idx[..., None], valid[..., None], axis=-1)
    return s


def grid_to_ascii(grid: np.ndarray) -> str:
    """Render a grid the way `python -m gdbot.bridge --grid` does.

    Used by tests and by `--sim` sanity checks, so a bad perception change shows
    up as a picture rather than a shape assertion.
    """
    marks = "#!oP"
    rows = []
    for r in range(GRID_H - 1, -1, -1):          # top row first
        line = []
        for w in range(GRID_W):
            hit = next((c for c in range(GRID_C) if grid[c, r, w]), None)
            if r == PLAYER_ROW and w == PLAYER_COL:
                line.append("@")
            else:
                line.append(marks[hit] if hit is not None else ".")
        rows.append("".join(line))
    return "\n".join(rows)

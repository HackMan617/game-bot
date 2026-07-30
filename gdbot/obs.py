"""The observation contract every backend produces and the policy consumes.

One observation is a pair:

    grid    (C, H, W) float32 occupancy — what the bot *sees* ahead of it
    scalars (S,)      float32 kinematics and mode flags — how it *moves*

The grid comes straight from the mod (`bridge.GRID_*`) so its layout is the same
in Sim and Live, and it is channel-first so it feeds a conv net with no reshape.

Nothing here encodes *where in the level* the player is — no x, no percent. That
is deliberate: the goal is a reactive generalist, and a policy that can read the
clock can memorise a level instead of learning to see it.
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
SCALAR_NAMES = (
    "on_ground", "vy", "upside_down", "speed", "gravity", "mini",
    "floor_gap", "has_ceiling", "ceiling_gap",
) + tuple(f"mode:{m}" for m in MODE_NAMES)
N_SCALARS = len(SCALAR_NAMES)

_VY_NORM = 1000.0             # GD y-velocity units/sec at which we saturate
_GAP_NORM = CELL * (GRID_H / 2)   # half a grid height, in GD units


@dataclass(frozen=True)
class Obs:
    """One frame as the network sees it."""

    grid: np.ndarray      # (C, H, W) float32, 0.0 / 1.0
    scalars: np.ndarray   # (S,) float32

    def __post_init__(self):
        assert self.grid.shape == GRID_SHAPE, f"grid {self.grid.shape} != {GRID_SHAPE}"
        assert self.scalars.shape == (N_SCALARS,), f"scalars {self.scalars.shape}"


def build_scalars(*, on_ground: bool, vy: float, upside_down: bool,
                  player_speed: float, gravity_mod: float, vehicle_size: float,
                  y: float, ground_y: float, ceiling_y: float,
                  gamemode: int) -> np.ndarray:
    """Pack per-frame kinematics into the fixed scalar vector.

    Both environments call this, so Sim and Live can never drift into producing
    differently-scaled inputs for the same physical situation. Distances are in
    GD units (1 block = 30), which is why SimEnv converts its block-space
    velocities before calling in.
    """
    s = np.zeros(N_SCALARS, dtype=np.float32)
    s[0] = 1.0 if on_ground else 0.0
    s[1] = np.clip(vy / _VY_NORM, -1.0, 1.0)
    s[2] = 1.0 if upside_down else 0.0
    s[3] = np.clip(player_speed - 1.0, -1.0, 1.0)      # 0.7..1.6x -> -0.3..0.6
    s[4] = np.clip(gravity_mod / 2.0, -1.0, 1.0)
    s[5] = np.clip(vehicle_size - 1.0, -1.0, 1.0)      # 0 normal, -0.5 mini
    s[6] = np.clip((y - ground_y) / _GAP_NORM, 0.0, 2.0)
    has_ceiling = ceiling_y > 0.0
    s[7] = 1.0 if has_ceiling else 0.0
    s[8] = np.clip((ceiling_y - y) / _GAP_NORM, 0.0, 2.0) if has_ceiling else 0.0
    mode = int(gamemode)
    if 0 <= mode < N_MODES:
        s[9 + mode] = 1.0
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

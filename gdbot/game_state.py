"""Shared game-state schema produced by every environment backend.

Both SimEnv and (later) LiveEnv emit a GameState each tick.  The observation
builder (observation.py) consumes ONLY this schema, so the same network inputs
work whether the state came from the simulator or from real-game memory reads.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

# Gamemode ids (GD 2.2 has 8; only CUBE is implemented for now).
CUBE = 0
SHIP = 1
BALL = 2
UFO = 3
WAVE = 4
ROBOT = 5
SPIDER = 6
SWING = 7  # new in 2.2


@dataclass
class GameState:
    """A single frame of game state, backend-agnostic.

    Coordinates are in "blocks" (1 block = 1 grid cell). `player_x` grows to the
    right as the level auto-scrolls; `player_y` is height above the floor.
    """

    player_x: float
    player_y: float
    vy: float            # vertical velocity, blocks/sec (+ = upward)
    on_ground: bool
    gamemode: int        # one of the ids above; CUBE for now
    dead: bool
    complete: bool
    percent: float       # progress through the level, 0.0 .. 1.0

    # lookahead(k) -> list of (surface_height, is_spike) for the k columns
    # immediately ahead of the player. This is how the network "sees" upcoming
    # hazards, and is what makes a *reactive* (generalizing) policy possible.
    # For SimEnv this reads the course grid; for LiveEnv it will read the
    # pre-parsed static hazard grid of the real level.
    lookahead: Callable[[int], List[Tuple[float, bool]]]

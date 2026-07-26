"""Turn a GameState into the fixed-length feature vector fed to the network.

Keeping this separate from the environments means Sim and Live share the exact
same network input layout. If you change LOOKAHEAD / features here, update
`num_inputs` in neat_config.txt to match OBS_SIZE.
"""

import numpy as np

from .game_state import GameState

# How many columns ahead of the player the network can "see".
LOOKAHEAD = 10

# Feature layout: for each lookahead column -> (relative surface height, spike),
# then two scalars: vertical velocity and on-ground flag.
OBS_SIZE = LOOKAHEAD * 2 + 2

# Normalization constants (keep inputs roughly in [-1, 1]).
_MAX_HEIGHT_DIFF = 8.0
_MAX_VY = 25.0


def build_observation(state: GameState) -> np.ndarray:
    """Build the network input vector from a GameState."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    cells = state.lookahead(LOOKAHEAD)
    for i, (surface, spike) in enumerate(cells):
        # Height of the ground ahead, relative to where the player currently is.
        diff = surface - state.player_y
        obs[i * 2] = np.clip(diff / _MAX_HEIGHT_DIFF, -1.0, 1.0)
        obs[i * 2 + 1] = 1.0 if spike else 0.0
    obs[LOOKAHEAD * 2] = float(np.clip(state.vy / _MAX_VY, -1.0, 1.0))
    obs[LOOKAHEAD * 2 + 1] = 1.0 if state.on_ground else 0.0
    return obs

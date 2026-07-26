"""Abstract environment interface shared by SimEnv (now) and LiveEnv (Phase 2+).

Deliberately Gym-like so the same code drives NEAT today and Stable-Baselines3
PPO/DQN later (Phase 6) with no changes to the environment contract.
"""

from abc import ABC, abstractmethod
from typing import Tuple

from .game_state import GameState


class GDEnv(ABC):
    """A Geometry-Dash-like environment.

    action: 0 = do nothing, 1 = jump/hold (cube: tap-to-jump).
    step() returns (state, reward, done, info).
    """

    @abstractmethod
    def reset(self) -> GameState:
        ...

    @abstractmethod
    def step(self, action: int) -> Tuple[GameState, float, bool, dict]:
        ...

    def close(self) -> None:  # optional; overridden by backends that hold resources
        pass

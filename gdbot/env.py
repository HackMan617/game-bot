"""The environment interface, and the live Geometry Dash backend.

`GDEnv` is the only thing the learning code talks to, so the same trainer drives
the deterministic simulator (`sim_env.SimEnv`) and the real game (`LiveEnv`).

LiveEnv is built on the v5 bridge handshake rather than the old polling loop: the
mod publishes a frame and blocks until we answer it, so the agent cannot silently
miss frames and the game runs exactly as fast as we can think. One answered frame
is one fixed 1/step_hz slice of game time, so raising `speed` buys wall-clock
throughput without changing a single thing the policy experiences.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from .bridge import Bridge, BridgeLost, VersionMismatch  # noqa: F401  (re-exported)
from .obs import Obs, build_scalars

# Reward shaping. Progress dominates: clearing a whole level sums to
# PROGRESS_REWARD, so the completion bonus is worth about one more level and a
# death costs roughly a tenth of one. Only *forward* progress pays, so a policy
# cannot farm reward by oscillating across a percent boundary.
PROGRESS_REWARD = 10.0
DEATH_PENALTY = 1.0
COMPLETE_BONUS = 10.0

START_PCT = 0.03      # a fresh attempt is anything below this
COMPLETE_PCT = 0.999


def shape_reward(gained: float, dead: bool, complete: bool) -> float:
    """The reward both backends pay out, so Sim and Live score identically."""
    r = PROGRESS_REWARD * max(0.0, gained)
    if dead:
        r -= DEATH_PENALTY
    if complete:
        r += COMPLETE_BONUS
    return r


class GDEnv(ABC):
    """A Geometry-Dash-like environment.

    action: 0 = release, 1 = hold jump.
    step() returns (obs, reward, done, info); info always carries at least
    `percent`, `dead` and `complete`.
    """

    @abstractmethod
    def reset(self) -> Obs:
        ...

    @abstractmethod
    def step(self, action: int) -> Tuple[Obs, float, bool, dict]:
        ...

    def close(self) -> None:
        pass


class LiveEnv(GDEnv):
    """Real Geometry Dash, driven through the GDBot Bridge mod."""

    def __init__(self, bridge: Optional[Bridge] = None, *, speed: int = 4,
                 step_hz: int = 60, practice: bool = False,
                 fast_respawn: bool = True, mute: bool = True,
                 max_steps: int = 12000, stall_steps: int = 600):
        self.b = bridge or Bridge(attach=False)
        self.practice = practice
        self.max_steps = max_steps
        self.stall_steps = stall_steps
        self._cfg = dict(speed=speed, step_hz=step_hz,
                         fast_respawn=fast_respawn, mute=mute)
        self._pending_seq: Optional[int] = None
        self._grid = np.zeros((0,), dtype=np.uint8)
        self._last_attempt: Optional[int] = None
        self._last_pct = 0.0
        self._best_pct = 0.0
        self._steps = 0
        self._stalled = 0

    # --- connection ----------------------------------------------------------
    def connect(self, timeout: float = 120.0) -> bool:
        """Wait for the mod, then take the wheel and apply our run settings."""
        if not self.b.wait_connected(timeout=timeout):
            return False
        self.b.set_step_hz(self._cfg["step_hz"])
        self.b.set_speed(self._cfg["speed"])
        self.b.set_fast_respawn(self._cfg["fast_respawn"])
        self.b.set_mute(self._cfg["mute"])
        self.b.attach()
        if self.practice:
            self.b.set_practice(True)
        return True

    def set_speed(self, n: int) -> None:
        """Change the wall-clock multiplier mid-run; the policy is unaffected."""
        self._cfg["speed"] = n
        self.b.set_speed(n)

    # --- the frame handshake -------------------------------------------------
    def _pump(self, jump: bool) -> dict:
        """Answer the frame we are holding, then block for the next one.

        The grid is copied out *before* we answer, because the moment the mod is
        released it starts overwriting the shared block with the next frame.
        """
        if self._pending_seq is not None:
            self.b.send_action(jump, self._pending_seq)
        st = self.b.wait_frame()
        self._pending_seq = st["state_seq"]
        self._grid = self.b.read_grid()
        return st

    # --- GDEnv ---------------------------------------------------------------
    def reset(self) -> Obs:
        """Block until a fresh attempt is underway, then hand back frame one.

        Normal mode waits for the run to be back near 0%; practice mode accepts
        any live frame after a respawn, because there the whole point is to
        restart at the checkpoint the curriculum left us on.
        """
        if self._last_attempt is None and not self.practice:
            self.b.reset_from_start()      # guarantee a clean 0% on the first episode

        deadline = time.time() + 60.0
        while True:
            st = self._pump(False)
            if st["in_level"] and not st["dead"]:
                fresh = self._last_attempt is None or st["attempt"] != self._last_attempt
                if fresh and (self.practice or st["percent"] < START_PCT):
                    break
            if time.time() > deadline:
                raise BridgeLost("no fresh attempt started within 60s "
                                 "(is the level paused, or the run finished?)")

        self._last_attempt = st["attempt"]
        self._last_pct = st["percent"]
        self._best_pct = st["percent"]
        self._steps = 0
        self._stalled = 0
        return self._obs(st)

    def step(self, action: int) -> Tuple[Obs, float, bool, dict]:
        st = self._pump(bool(action))
        self._steps += 1

        dead = bool(st["dead"])
        complete = st["percent"] >= COMPLETE_PCT
        gained = max(0.0, st["percent"] - self._last_pct)
        self._last_pct = st["percent"]
        self._best_pct = max(self._best_pct, st["percent"])

        reward = shape_reward(gained, dead, complete)

        # A live player always advances, so a long flat stretch of percent means
        # something is wrong (paused, level over, respawn animation) rather than
        # a bad policy — end the episode instead of feeding the buffer garbage.
        self._stalled = self._stalled + 1 if gained <= 0.0 else 0
        timeout = self._steps >= self.max_steps or self._stalled >= self.stall_steps
        done = dead or complete or timeout

        if done:
            self._last_attempt = st["attempt"]

        info = {"percent": st["percent"], "best_percent": self._best_pct,
                "dead": dead, "complete": complete, "timeout": timeout,
                "steps": self._steps, "attempt": st["attempt"],
                "x": st["x"], "y": st["y"], "level_id": st["level_id"],
                "checkpoints": st["checkpoint_count"], "raw": st}
        return self._obs(st), reward, done, info

    def close(self) -> None:
        try:
            if self._pending_seq is not None:
                self.b.send_action(False, self._pending_seq)
        except Exception:
            pass
        self.b.close()

    # --- curriculum ----------------------------------------------------------
    def load_level(self, level_id: int) -> bool:
        ok = self.b.load_level(level_id)
        self._last_attempt = None
        return ok

    def respawn_at(self, index: int) -> bool:
        return self.b.respawn_at(index)

    # --- observation ---------------------------------------------------------
    def _obs(self, st: dict) -> Obs:
        return Obs(
            grid=self._grid.astype(np.float32),
            scalars=build_scalars(
                on_ground=bool(st["on_ground"]), vy=st["vy"],
                upside_down=bool(st["upside_down"]),
                player_speed=st["player_speed"], gravity_mod=st["gravity_mod"],
                vehicle_size=st["vehicle_size"],
                y=st["y"], ground_y=st["ground_y"], ceiling_y=st["ceiling_y"],
                gamemode=st["gamemode"],
            ),
        )

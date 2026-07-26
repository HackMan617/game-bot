"""LiveEnv — a GDEnv backed by the GDBot Bridge Geode mod (shared memory).

The trained/evolving network drives real Geometry Dash through this, using the
same reset()/step() contract as SimEnv. GD auto-retries on death, so each genome
evaluation is one attempt.

v1 is "blind / per-level": the observation is player kinematics + progress, so a
network learns THIS level's timing (like DashBot / MarI/O-for-one-level). True
reactive lookahead arrives when the mod also streams nearby hazards — then this
env fills GameState.lookahead and the sim-trained reactive brain transfers over.
"""

import time

from .env_base import GDEnv
from .game_state import GameState
from .live_shared import LiveShared


class LiveEnv(GDEnv):
    def __init__(self, poll_hz: float = 300.0):
        self.sh = LiveShared()
        self._period = 1.0 / poll_hz
        self._last_frame = -1

    # --- connection helpers ----------------------------------------------------
    def wait_connected(self, timeout: float = None) -> bool:
        """Block until the mod stamps the shared block (GD in a level)."""
        t0 = time.time()
        while not self.sh.connected():
            if timeout is not None and time.time() - t0 > timeout:
                return False
            time.sleep(0.1)
        return True

    def _next_frame(self) -> dict:
        """Block until the mod advances one physics frame; return the state."""
        while True:
            st = self.sh.read()
            if st["frame"] != self._last_frame and st["magic"] != 0:
                self._last_frame = st["frame"]
                return st
            time.sleep(self._period)

    # --- GDEnv interface -------------------------------------------------------
    def reset(self) -> GameState:
        """Wait for a fresh attempt to begin (alive, near the start)."""
        self.sh.set_action(False)
        while True:
            st = self._next_frame()
            if st["in_level"] and not st["dead"] and st["percent"] < 0.03:
                return self._state(st)

    def step(self, action: int):
        self.sh.set_action(bool(action))
        st = self._next_frame()
        done = bool(st["dead"]) or st["percent"] >= 0.999
        reward = -1.0 if st["dead"] else (10.0 if st["percent"] >= 0.999 else 0.01)
        return self._state(st), reward, done, st

    def close(self) -> None:
        self.sh.set_action(False)
        self.sh.close()

    # --- state -----------------------------------------------------------------
    def _state(self, st: dict) -> GameState:
        # No hazard data yet -> empty lookahead. When the mod streams nearby
        # objects, fill this from them and the reactive observation lights up.
        def lookahead(k):
            return [(0.0, False)] * k

        return GameState(
            player_x=st["x"], player_y=st["y"], vy=st["vy"],
            on_ground=bool(st["on_ground"]), gamemode=st["gamemode"],
            dead=bool(st["dead"]), complete=st["percent"] >= 0.999,
            percent=st["percent"], lookahead=lookahead,
        )

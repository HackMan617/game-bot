"""SimEnv — a fast, deterministic, headless cube simulator.

This exists so the trainer, the policy and the viewer can be developed, tested
and demonstrated with Geometry Dash closed. It models the essence of cube mode:
constant auto-scroll, gravity, a fixed-impulse jump that only fires from the
ground, spikes to clear and blocks to land on (running into a wall kills).

It emits the *same* `Obs` as `LiveEnv` — the identical 24x16x4 grid layout and
the identical scalar vector — so a policy trained here loads and runs against the
real game unchanged. Internally the sim works in blocks; velocities are scaled to
GD units (1 block = 30) on the way out so the normalisation matches Live.

Courses are generated from a seed and are solvable by construction, so training
runs and regression tests are reproducible.
"""

import random
from typing import List, Tuple

import numpy as np

from .env import COMPLETE_PCT, GDEnv, shape_reward
from .obs import (CUBE, GRID_BEHIND, GRID_C, GRID_H, GRID_W, Obs, PLAYER_ROW,
                  build_scalars)

# --- Physics constants (units: blocks, seconds) -------------------------------
DT = 1.0 / 60.0     # fixed timestep — GD physics are frame-locked
VX = 10.3           # forward speed (~GD "normal" speed)
GRAVITY = 90.0      # downward acceleration
JUMP_V = 20.0       # upward impulse on jump (arc ~= 2.2 blocks high, ~4.5 wide)
SPIKE_KILL_H = 1.0  # must clear a spike by more than this to survive
UNITS_PER_BLOCK = 30.0   # GD's cell size, used to report Live-scaled velocities


class Course:
    """A level as parallel arrays indexed by integer column.

    ground_height[c] = height of the top surface the cube stands on at column c.
    is_spike[c]      = a deadly spike sits on that surface.
    """

    def __init__(self, ground_height: List[float], is_spike: List[bool]):
        self.ground_height = ground_height
        self.is_spike = is_spike
        self.length = len(ground_height)

    def surface(self, col: int) -> float:
        col = max(0, min(self.length - 1, col))
        return self.ground_height[col]

    def spike(self, col: int) -> bool:
        if col < 0 or col >= self.length:
            return False
        return self.is_spike[col]


def make_course(seed: int = 0, length: int = 220) -> Course:
    """Generate a deterministic, solvable cube course.

    Obstacles are spaced far enough apart that one well-timed jump clears each,
    so a purely reactive policy (see hazard -> jump) can always win.
    """
    rng = random.Random(seed)
    ground = [0.0] * length
    spike = [False] * length

    col = 12  # flat run-up so the cube can settle before the first obstacle
    while col < length - 12:
        col += rng.randint(6, 12)  # spacing between obstacles = landing room
        if col >= length - 12:
            break
        kind = rng.choice(["spike", "spike2", "step", "step"])
        if kind == "spike":
            spike[col] = True
        elif kind == "spike2":
            spike[col] = True
            spike[col + 1] = True
        else:  # step: a short raised block to jump onto, then fall off
            h = rng.choice([1.0, 2.0])
            blen = rng.randint(2, 4)
            for c in range(col, min(col + blen, length)):
                ground[c] = h
            col += blen
    return Course(ground, spike)


class SimEnv(GDEnv):
    """Headless cube simulator speaking the `GDEnv` contract."""

    def __init__(self, course: Course = None, seed: int = 0,
                 max_steps: int = 5000, reseed_each_episode: bool = False):
        self.course = course if course is not None else make_course(seed)
        self.seed = seed
        self.max_steps = max_steps
        # Rotating the course per episode is what stops the policy memorising a
        # single layout — the sim analogue of training across many real levels.
        self.reseed_each_episode = reseed_each_episode
        self._episode = 0
        self.reset()

    # --- GDEnv ---------------------------------------------------------------
    def reset(self) -> Obs:
        if self.reseed_each_episode and self._episode:
            self.course = make_course(self.seed + self._episode)
        self._episode += 1
        self.px = 1.0
        self.py = self.course.surface(1)
        self.vy = 0.0
        self.on_ground = True
        self.dead = False
        self.complete = False
        self.percent = 0.0
        self._best_pct = 0.0
        self._steps = 0
        self._prev_action = 0
        self._air = 0
        return self._obs()

    def step(self, action: int) -> Tuple[Obs, float, bool, dict]:
        if self.dead or self.complete:
            return self._obs(), 0.0, True, self._info(timeout=False)

        prev_py = self.py       # height at the end of the last tick (detects landings)
        prev_pct = self.percent
        self._steps += 1

        # 1. Jump only fires from the ground (classic cube).
        if action and self.on_ground:
            self.vy = JUMP_V
            self.on_ground = False

        # 2. Gravity integrates only while airborne.
        if not self.on_ground:
            self.vy -= GRAVITY * DT
            self.py += self.vy * DT

        # 3. Auto-scroll forward.
        self.px += VX * DT
        col = int(self.px)
        surface = self.course.surface(col)

        # 4. Resolve against the surface.
        if self.py <= surface:
            if prev_py >= surface - 1e-6:
                self.py = surface          # descended onto it -> land
                self.vy = 0.0
                self.on_ground = True
            else:
                self.dead = True           # body below the column we walked into
        else:
            self.on_ground = False

        # 5. Spikes are deadly unless cleared by more than SPIKE_KILL_H.
        if self.course.spike(col) and self.py <= surface + SPIKE_KILL_H:
            self.dead = True

        # 6. Progress / completion.
        self.percent = min(1.0, self.px / self.course.length)
        self._best_pct = max(self._best_pct, self.percent)
        if self.percent >= COMPLETE_PCT:
            self.complete = True

        # Book-keeping the observation needs but the physics does not.
        self._prev_action = 1 if action else 0
        self._air = 0 if self.on_ground else self._air + 1

        timeout = self._steps >= self.max_steps
        reward = shape_reward(self.percent - prev_pct, self.dead, self.complete)
        done = self.dead or self.complete or timeout
        return self._obs(), reward, done, self._info(timeout)

    # --- observation ---------------------------------------------------------
    def _grid(self) -> np.ndarray:
        """Rasterise the course around the player into the shared grid layout.

        Row PLAYER_ROW is the player's own row and column GRID_BEHIND is its own
        column, matching what the mod publishes, so the two backends put the same
        feature in the same cell.
        """
        g = np.zeros((GRID_C, GRID_H, GRID_W), dtype=np.float32)
        base_col = int(self.px) - GRID_BEHIND
        for w in range(GRID_W):
            surface = self.course.surface(base_col + w)
            # Channel 0 (solid): everything below the top surface is ground.
            top_row = PLAYER_ROW + int(np.floor(surface - self.py))
            lo = max(0, min(GRID_H, top_row))
            if lo > 0:
                g[0, :lo, w] = 1.0
            # Channel 1 (hazard): a spike sits on the surface cell itself.
            if self.course.spike(base_col + w) and 0 <= top_row < GRID_H:
                g[1, top_row, w] = 1.0
        return g

    def _obs(self) -> Obs:
        return Obs(
            grid=self._grid(),
            scalars=build_scalars(
                on_ground=self.on_ground,
                vy=self.vy * UNITS_PER_BLOCK,          # blocks/s -> GD units/s
                upside_down=False,
                player_speed=1.0, gravity_mod=1.0, vehicle_size=1.0,
                y=self.py * UNITS_PER_BLOCK, ground_y=0.0, ceiling_y=0.0,
                gamemode=CUBE,
                prev_action=float(self._prev_action), air_time=float(self._air),
            ),
        )

    def _info(self, timeout: bool) -> dict:
        return {"percent": self.percent, "best_percent": self._best_pct,
                "dead": self.dead, "complete": self.complete, "timeout": timeout,
                "steps": self._steps, "x": self.px, "y": self.py,
                "level_id": 0, "checkpoints": 0}


if __name__ == "__main__":   # eyeball the grid the same way `bridge --grid` does
    import time

    from .obs import grid_to_ascii

    env = SimEnv(seed=1)
    obs = env.reset()
    while True:
        print("\033[H\033[J", end="")
        print(f"x={env.px:6.1f}  y={env.py:5.2f}  {env.percent * 100:5.1f}%\n")
        print(grid_to_ascii(obs.grid))
        obs, _r, done, _i = env.step(0)
        if done:
            obs = env.reset()
        time.sleep(0.05)

"""VecSimEnv — N cube simulators advanced as numpy arrays, not as a Python loop.

The single-environment loop is inference-bound: measured at batch 1 the policy
costs ~341us per decision against ~42us for the environment, and at that batch
size CUDA is 3.7x *slower* than the CPU because a 216k-parameter network cannot
fill a kernel launch. Both problems have the same cause and the same fix —
decide for many states at once.

So this module holds the whole population as arrays. Every environment shares
one code path through the physics, one rasterisation of the occupancy grid, and
one forward pass through the network. The trainer's inner loop goes from N
python-level steps plus N forward passes to one of each.

Two invariants make it safe to trust:

* The physics is a line-by-line transliteration of `SimEnv.step`, in the same
  order, including the detail that a jump integrates gravity on the same tick it
  fires. `tests/test_stack.py` runs a 3000-step parity check against SimEnv and
  requires exact agreement, so a divergence is a test failure rather than a
  quietly different training distribution.
* The observation is built by the same `build_scalars` the other backends call,
  in its batched form. There is no second definition of the input scaling to
  drift out of sync.

Environments auto-reset: when one finishes, its slot immediately begins a new
episode and the returned observation is that episode's first frame. `dones[i]`
still marks the transition as terminal, which is all GAE needs to avoid
bootstrapping across the boundary.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .env import COMPLETE_PCT, shape_reward_vec
from .obs import (CUBE, GRID_BEHIND, GRID_C, GRID_H, GRID_W, PLAYER_ROW,
                  build_scalars)
from .sim_env import (DT, GRAVITY, JUMP_V, SPIKE_KILL_H, UNITS_PER_BLOCK, VX,
                      make_course)

# Seed stride between environments. Prime and far larger than any plausible
# episode count, so env i's course sequence never collides with env j's.
SEED_STRIDE = 100003


class VecSimEnv:
    """`n_envs` independent cube courses, stepped together.

    `reset()` returns `(grid, scalars)` with leading dimension `n_envs`;
    `step(actions)` takes an int array of that length and returns
    `((grid, scalars), rewards, dones, info)`.
    """

    def __init__(self, n_envs: int = 16, seed: int = 0, max_steps: int = 5000,
                 reseed_each_episode: bool = True,
                 course_length: Optional[int] = None):
        self.n = int(n_envs)
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.reseed_each_episode = reseed_each_episode

        probe = make_course(seed) if course_length is None else make_course(seed, course_length)
        self.length = probe.length
        self._course_length = self.length

        self.ground = np.zeros((self.n, self.length), dtype=np.float32)
        self.spike = np.zeros((self.n, self.length), dtype=bool)
        self._episode = np.zeros(self.n, dtype=np.int64)
        self._rows = np.arange(self.n)

        # Allocated once and written in place — a rollout does this thousands of
        # times, and a fresh (N, 4, 16, 24) array per step is pure garbage churn.
        self._grid = np.zeros((self.n, GRID_C, GRID_H, GRID_W), dtype=np.float32)
        self._cols = np.arange(GRID_W)[None, :]
        self._rowspan = np.arange(GRID_H)[None, None, :]

        for i in range(self.n):
            self._load_course(i)
        self.reset()

    # --- courses -------------------------------------------------------------
    def _course_seed(self, i: int) -> int:
        """Env i, episode e uses seed + i*STRIDE + e.

        With i = 0 that is exactly `SimEnv(seed=s, reseed_each_episode=True)`'s
        sequence, which is what makes the parity test meaningful.
        """
        return self.seed + i * SEED_STRIDE + int(self._episode[i])

    def _load_course(self, i: int) -> None:
        c = make_course(self._course_seed(i), self._course_length)
        self.ground[i] = np.asarray(c.ground_height, dtype=np.float32)
        self.spike[i] = np.asarray(c.is_spike, dtype=bool)

    # --- state ---------------------------------------------------------------
    def _reset_idx(self, idx: np.ndarray) -> None:
        """Begin a fresh episode in the given slots, in place."""
        if idx.size == 0:
            return
        if self.reseed_each_episode:
            for i in idx:
                self._episode[i] += 1
                self._load_course(int(i))
        self.px[idx] = 1.0
        self.py[idx] = self.ground[idx, 1]
        self.vy[idx] = 0.0
        self.on_ground[idx] = True
        self.dead[idx] = False
        self.complete[idx] = False
        self.percent[idx] = 0.0
        self.best_pct[idx] = 0.0
        self.steps[idx] = 0
        self.prev_action[idx] = 0.0
        self.air[idx] = 0.0
        self.ep_return[idx] = 0.0

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        # The kinematic state is float64, not float32, and that is deliberate.
        # `SimEnv` integrates in Python floats, and the grid row a column lands on
        # is `floor(surface - py)` — an integer boundary. At float32 the two
        # backends disagree on that floor within ~14 steps and start training on
        # subtly different worlds. float64 here costs six N-length arrays; the
        # grid, which is the array that actually matters for bandwidth, stays
        # float32.
        self.px = np.ones(self.n, dtype=np.float64)
        self.py = self.ground[:, 1].astype(np.float64)
        self.vy = np.zeros(self.n, dtype=np.float64)
        self.on_ground = np.ones(self.n, dtype=bool)
        self.dead = np.zeros(self.n, dtype=bool)
        self.complete = np.zeros(self.n, dtype=bool)
        self.percent = np.zeros(self.n, dtype=np.float64)
        self.best_pct = np.zeros(self.n, dtype=np.float64)
        self.steps = np.zeros(self.n, dtype=np.int64)
        self.prev_action = np.zeros(self.n, dtype=np.float32)
        self.air = np.zeros(self.n, dtype=np.float32)
        self.ep_return = np.zeros(self.n, dtype=np.float32)
        return self._obs()

    # --- physics -------------------------------------------------------------
    def step(self, actions) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray,
                                     np.ndarray, Dict]:
        a = np.asarray(actions).reshape(self.n) != 0
        prev_py = self.py.copy()
        prev_pct = self.percent.copy()
        self.steps += 1

        # 1. Jump only fires from the ground (classic cube).
        jump = a & self.on_ground
        self.vy = np.where(jump, JUMP_V, self.vy)
        self.on_ground &= ~jump

        # 2. Gravity integrates only while airborne — including on the very tick
        #    a jump fires, because step 1 already cleared on_ground.
        airborne = ~self.on_ground
        self.vy = np.where(airborne, self.vy - GRAVITY * DT, self.vy)
        self.py = np.where(airborne, self.py + self.vy * DT, self.py)

        # 3. Auto-scroll forward.
        self.px += VX * DT
        col = self.px.astype(np.int64)
        cc = np.clip(col, 0, self.length - 1)
        surface = self.ground[self._rows, cc]

        # 4. Resolve against the surface: descending onto it lands, walking into
        #    it kills.
        below = self.py <= surface
        landed = below & (prev_py >= surface - 1e-6)
        self.py = np.where(landed, surface, self.py)
        self.vy = np.where(landed, 0.0, self.vy)
        self.on_ground = np.where(below, landed, False)
        self.dead |= below & ~landed

        # 5. Spikes are deadly unless cleared by more than SPIKE_KILL_H.
        in_course = (col >= 0) & (col < self.length)
        self.dead |= (self.spike[self._rows, cc] & in_course
                      & (self.py <= surface + SPIKE_KILL_H))

        # 6. Progress / completion.
        self.percent = np.minimum(1.0, self.px / self.length)
        np.maximum(self.best_pct, self.percent, out=self.best_pct)
        self.complete = self.percent >= COMPLETE_PCT

        self.prev_action = a.astype(np.float32)
        self.air = np.where(self.on_ground, 0.0, self.air + 1.0).astype(np.float32)

        timeout = self.steps >= self.max_steps
        reward = shape_reward_vec(self.percent - prev_pct, self.dead, self.complete)
        self.ep_return += reward
        done = self.dead | self.complete | timeout

        info: Dict = {"percent": self.percent.copy(),
                      "best_percent": self.best_pct.copy(),
                      "dead": self.dead.copy(), "complete": self.complete.copy(),
                      "timeout": timeout.copy(), "episodes": self._finished(done, timeout)}

        # Hitting the step limit is not a failure, but auto-reset makes it look
        # like one: the next observation belongs to a different episode, so the
        # trainer cannot bootstrap V(s_T) from it. Hand back the frame we are
        # about to throw away for exactly those slots. Rare enough (a course runs
        # ~1300 steps against a 5000 limit) that the extra rasterise is free.
        trunc = np.nonzero(timeout & ~self.dead & ~self.complete)[0]
        if trunc.size:
            g, s = self._obs()
            info["truncated"] = (trunc, g[trunc].copy(), s[trunc].copy())

        # Auto-reset *after* the episode records are taken, so the stats describe
        # the episode that ended rather than the one that just began.
        self._reset_idx(np.nonzero(done)[0])
        return self._obs(), reward, done, info

    def _finished(self, done: np.ndarray, timeout: np.ndarray) -> List[dict]:
        return [{"return": float(self.ep_return[i]),
                 "best_percent": float(self.best_pct[i]),
                 "steps": int(self.steps[i]),
                 "dead": bool(self.dead[i]),
                 "complete": bool(self.complete[i]),
                 # A truncated episode is not a failure; the trainer bootstraps
                 # its tail rather than teaching the policy that surviving ends
                 # the world.
                 "timeout": bool(timeout[i] and not self.dead[i] and not self.complete[i])}
                for i in np.nonzero(done)[0]]

    # --- observation ---------------------------------------------------------
    def _obs(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._rasterise(), self._scalars()

    def _scalars(self) -> np.ndarray:
        return build_scalars(
            on_ground=self.on_ground, vy=self.vy * UNITS_PER_BLOCK,
            upside_down=np.zeros(self.n, dtype=np.float32),
            player_speed=np.ones(self.n, dtype=np.float32),
            gravity_mod=np.ones(self.n, dtype=np.float32),
            vehicle_size=np.ones(self.n, dtype=np.float32),
            y=self.py * UNITS_PER_BLOCK,
            ground_y=np.zeros(self.n, dtype=np.float32),
            ceiling_y=np.zeros(self.n, dtype=np.float32),
            gamemode=np.full(self.n, CUBE, dtype=np.int64),
            prev_action=self.prev_action, air_time=self.air,
        )

    def _rasterise(self) -> np.ndarray:
        """Draw every environment's local window in one pass.

        The scalar version walks 24 columns per environment in Python. This does
        the same arithmetic as three array ops over an (N, 24) block, which is
        where most of the speedup over a loop of SimEnv actually comes from —
        the physics is a handful of flops, the grid is 1536 cells.
        """
        g = self._grid
        g.fill(0.0)

        base = self.px.astype(np.int64) - GRID_BEHIND            # (N,)
        cols = base[:, None] + self._cols                        # (N, W)
        cc = np.clip(cols, 0, self.length - 1)
        surf = self.ground[self._rows[:, None], cc]              # (N, W)

        # Channel 0 (solid): every row below the top surface is ground.
        top = PLAYER_ROW + np.floor(surf - self.py[:, None]).astype(np.int64)
        lo = np.clip(top, 0, GRID_H)                             # (N, W)
        solid = self._rowspan < lo[:, :, None]                   # (N, W, H)
        g[:, 0] = solid.transpose(0, 2, 1).astype(np.float32)

        # Channel 1 (hazard): a spike sits on the surface cell itself.
        hit = (self.spike[self._rows[:, None], cc] & (cols >= 0) & (cols < self.length)
               & (top >= 0) & (top < GRID_H))
        ni, wi = np.nonzero(hit)
        if ni.size:
            g[ni, 1, top[ni, wi], wi] = 1.0
        return g

    def close(self) -> None:
        pass


if __name__ == "__main__":      # a quick throughput read against the scalar sim
    import time

    from .sim_env import SimEnv

    for n in (1, 8, 32, 128):
        env = VecSimEnv(n_envs=n, seed=1)
        env.reset()
        acts = np.zeros(n, dtype=np.int64)
        t0 = time.perf_counter()
        for _ in range(2000):
            env.step(acts)
        dt = time.perf_counter() - t0
        print(f"vec n={n:4d}  {2000 * n / dt:12,.0f} env-steps/s")

    env, obs = SimEnv(seed=1), None
    obs = env.reset()
    t0 = time.perf_counter()
    for _ in range(2000):
        _o, _r, done, _i = env.step(0)
        if done:
            env.reset()
    print(f"scalar      {2000 / (time.perf_counter() - t0):12,.0f} env-steps/s")

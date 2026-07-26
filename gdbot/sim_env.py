"""SimEnv — a fast, deterministic, headless cube simulator.

This is the training ground for the network before the real game is wired up.
It models the essence of GD cube mode: constant forward auto-scroll, gravity,
a fixed-impulse jump that only fires from the ground, spikes you must jump over,
and steps/blocks you must jump onto (running into a wall = death). Same feel as
the jump mechanic prototyped in `input test.py`, but headless and physics-driven.

Courses are generated deterministically from a seed and are solvable by
construction, so training and regression tests are fully reproducible.
"""

import random
from typing import List, Tuple

from .env_base import GDEnv
from .game_state import CUBE, GameState

# --- Physics constants (units: blocks, seconds) ---------------------------------
DT = 1.0 / 60.0     # fixed timestep — GD physics are frame-locked
VX = 10.3           # forward speed (~GD "normal" speed)
GRAVITY = 90.0      # downward acceleration
JUMP_V = 20.0       # upward impulse on jump (arc ~= 2.2 blocks high, ~4.5 wide)
SPIKE_KILL_H = 1.0  # must clear a spike by more than this to survive


class Course:
    """A level as parallel arrays indexed by integer column.

    ground_height[c] = height of the top surface the cube stands on at column c.
    is_spike[c]      = a deadly spike sits on the surface at column c.
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

    Obstacles are spaced far enough apart that a single well-timed jump clears
    each one, so a reactive policy (spike/step ahead -> jump) can always win.
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
        else:  # step: a short raised block you must jump onto, then fall off
            h = rng.choice([1.0, 2.0])
            blen = rng.randint(2, 4)
            for c in range(col, min(col + blen, length)):
                ground[c] = h
            col += blen
    return Course(ground, spike)


class SimEnv(GDEnv):
    def __init__(self, course: Course, render: bool = False):
        self.course = course
        self.render = render
        self.render_fps = 60      # raise for snappier demos (e.g. watch_learn.py)
        self._screen = None
        self._clock = None
        self._font = None
        self.reset()

    # --- GDEnv interface --------------------------------------------------------
    def reset(self) -> GameState:
        self.px = 1.0
        self.py = self.course.surface(1)
        self.vy = 0.0
        self.on_ground = True
        self.dead = False
        self.complete = False
        self.percent = 0.0
        return self._state()

    def step(self, action: int):
        if self.dead or self.complete:
            return self._state(), 0.0, True, {}

        prev_py = self.py  # height at end of previous tick (used to detect landings)

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
                # Descended onto the surface (or already on it) -> land.
                self.py = surface
                self.vy = 0.0
                self.on_ground = True
            else:
                # Body is below the top of the column we walked into -> wall hit.
                self.dead = True
        else:
            self.on_ground = False

        # 5. Spikes: deadly unless cleared by more than SPIKE_KILL_H.
        if self.course.spike(col) and self.py <= surface + SPIKE_KILL_H:
            self.dead = True

        # 6. Progress / completion.
        self.percent = min(1.0, self.px / self.course.length)
        if self.px >= self.course.length - 1:
            self.complete = True

        done = self.dead or self.complete
        return self._state(), self._reward(), done, {}

    def close(self) -> None:
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

    # --- helpers ----------------------------------------------------------------
    def _reward(self) -> float:
        """Dense reward for the (later) RL path; NEAT uses max-percent instead."""
        if self.dead:
            return -1.0
        if self.complete:
            return 10.0
        return VX * DT * 0.1  # small reward for surviving forward progress

    def _state(self) -> GameState:
        course = self.course
        base_col = int(self.px)

        def lookahead(k: int):
            return [
                (course.surface(base_col + i), course.spike(base_col + i))
                for i in range(1, k + 1)
            ]

        return GameState(
            player_x=self.px,
            player_y=self.py,
            vy=self.vy,
            on_ground=self.on_ground,
            gamemode=CUBE,
            dead=self.dead,
            complete=self.complete,
            percent=self.percent,
            lookahead=lookahead,
        )

    # --- optional visualization -------------------------------------------------
    def draw_game(self, surface, rect, overlay: str = None) -> None:
        """Draw the course + cube into `rect` on an existing surface.

        Factored out so a combined view (watch.py) can render the game and the
        neural-network panel into one window.
        """
        import pygame

        PX = 22  # pixels per block
        floor_y = rect.y + rect.height - 40

        prev_clip = surface.get_clip()
        surface.set_clip(rect)
        pygame.draw.rect(surface, (18, 18, 28), rect)

        cam = self.px - 5  # keep the cube ~5 blocks from the left edge
        for c in range(max(0, int(cam) - 1), min(self.course.length, int(cam) + 44)):
            sx = rect.x + int((c - cam) * PX)
            top = floor_y - int(self.course.surface(c) * PX)
            pygame.draw.rect(surface, (55, 55, 78), (sx, top, PX + 1, rect.bottom - top))
            if self.course.spike(c):
                pygame.draw.polygon(
                    surface, (225, 70, 70),
                    [(sx, top), (sx + PX // 2, top - PX), (sx + PX, top)],
                )

        cube_x = rect.x + int((self.px - cam) * PX)
        cube_y = floor_y - int(self.py * PX) - PX
        color = (225, 70, 70) if self.dead else (90, 220, 130)
        pygame.draw.rect(surface, color, (cube_x, cube_y, PX, PX))
        surface.set_clip(prev_clip)

        if overlay:
            if self._font is None:
                self._font = pygame.font.SysFont("consolas", 18)
            surface.blit(self._font.render(overlay, True, (235, 235, 245)),
                         (rect.x + 10, rect.y + 8))

    def render_frame(self, overlay: str = None) -> None:
        if not self.render:
            return
        import pygame

        W, H = 900, 420
        if self._screen is None:
            pygame.init()
            pygame.font.init()
            self._screen = pygame.display.set_mode((W, H))
            self._clock = pygame.time.Clock()
            pygame.display.set_caption("gdbot — SimEnv")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.render = False
                pygame.quit()
                self._screen = None
                return

        self._screen.fill((10, 10, 16))
        self.draw_game(self._screen, pygame.Rect(0, 0, W, H), overlay=overlay)
        pygame.display.flip()
        self._clock.tick(self.render_fps)

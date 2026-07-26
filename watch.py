"""Watch the bot — game on top, live MarI/O-style neural network below.

    python watch.py play  [seed]   # the trained winner playing (default seed 99, unseen)
    python watch.py learn [seed]   # evolve from scratch and watch each generation improve

Close the window (or Esc) to stop.
"""

import os
import pickle
import sys

import neat
import pygame

from gdbot import netviz
from gdbot.neat_core import eval_genomes
from gdbot.observation import build_observation
from gdbot.sim_env import SimEnv, make_course

W, H = 1040, 660
GAME_RECT = (0, 0, W, 300)
NET_RECT = (0, 302, W, H - 302)
HERE = os.path.dirname(os.path.abspath(__file__))


def _config():
    return neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        os.path.join(HERE, "neat_config.txt"),
    )


def _make_window():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("gdbot — game + neural network")
    return screen, pygame.time.Clock()


def _run_episode(screen, clock, genome, config, seed, label, fps=60):
    """Play one attempt, rendering game + network each frame. Returns False if
    the user closed the window."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = SimEnv(make_course(seed=seed))
    state = env.reset()
    done = False
    while not done:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return False
        action = 1 if net.activate(build_observation(state))[0] > 0.5 else 0
        state, _reward, done, _ = env.step(action)

        screen.fill((10, 10, 16))
        env.draw_game(screen, pygame.Rect(*GAME_RECT), overlay=f"{label}    {state.percent:5.0%}")
        netviz.draw_network(screen, net, pygame.Rect(*NET_RECT))
        pygame.display.flip()
        clock.tick(fps)

    # Hold the final frame briefly so a completion/death is visible.
    for _ in range(int(fps * 0.6)):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
        clock.tick(fps)
    return True


def play(seed=99):
    screen, clock = _make_window()
    config = _config()
    with open(os.path.join(HERE, "winner.pkl"), "rb") as f:
        genome = pickle.load(f)
    _run_episode(screen, clock, genome, config, seed, f"TRAINED BOT — unseen seed {seed}")
    pygame.quit()


def learn(seed=3, generations=40):
    """Evolve from scratch; after each generation, watch its best network play."""
    screen, clock = _make_window()
    config = _config()
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(False))

    for g in range(1, generations + 1):
        pop.run(eval_genomes, 1)              # evolve one generation
        best = pop.best_genome
        if best is None or best.fitness is None:
            continue
        label = f"LEARNING — generation {g}   best fitness {best.fitness:.2f}"
        # Faster playback early (attempts are short); readable throughout.
        if not _run_episode(screen, clock, best, config, seed, label, fps=120):
            break
        if best.fitness >= config.fitness_threshold:
            _run_episode(screen, clock, best, config, seed, f"SOLVED at generation {g}!", fps=60)
            break
    pygame.quit()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "play"
    seed_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if mode == "learn":
        learn(seed=seed_arg if seed_arg is not None else 3)
    else:
        play(seed=seed_arg if seed_arg is not None else 99)

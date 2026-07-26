"""NEAT training core — evolves networks that map observation -> jump.

Faithful to the MarI/O approach: a population of networks each attempts the
level(s); fitness = how far they get, plus a completion bonus. To force a
*reactive, generalizing* policy (rather than a memorized single-level macro),
each genome is evaluated across SEVERAL different courses and scored on the mean.
"""

import os

import neat

from .observation import build_observation
from .sim_env import SimEnv, make_course

# Multiple courses => the network must react to what it sees, not memorize.
TRAIN_SEEDS = [1, 2, 3, 4, 5]
MAX_STEPS = 5000  # safety cap per attempt (a course is ~1300 ticks)


def eval_single(net, seed: int) -> float:
    """Run one genome's network on one course; return fitness in [0, 2]."""
    env = SimEnv(make_course(seed=seed))
    state = env.reset()
    best_percent = 0.0
    for _ in range(MAX_STEPS):
        action = 1 if net.activate(build_observation(state))[0] > 0.5 else 0
        state, _reward, done, _ = env.step(action)
        best_percent = max(best_percent, state.percent)
        if done:
            break
    fitness = best_percent
    if state.complete:
        fitness += 1.0  # completion bonus (max per-course fitness = 2.0)
    return fitness


def eval_genomes(genomes, config) -> None:
    for _genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total = sum(eval_single(net, seed) for seed in TRAIN_SEEDS)
        genome.fitness = total / len(TRAIN_SEEDS)


def run(config_path: str, generations: int = 100, checkpoint_dir: str = "checkpoints"):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    os.makedirs(checkpoint_dir, exist_ok=True)
    pop.add_reporter(
        neat.Checkpointer(10, filename_prefix=os.path.join(checkpoint_dir, "neat-"))
    )

    winner = pop.run(eval_genomes, generations)
    return winner, stats, config

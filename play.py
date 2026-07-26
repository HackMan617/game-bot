"""Watch the trained bot play — on a course it was NOT trained on, to show it
generalizes rather than memorizes.

    python play.py [seed]

Defaults to seed 99 (outside the training seeds 1..5).
"""

import os
import pickle
import sys

import neat

from gdbot.observation import build_observation
from gdbot.sim_env import SimEnv, make_course

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 99

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        os.path.join(HERE, "neat_config.txt"),
    )
    with open(os.path.join(HERE, "winner.pkl"), "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = SimEnv(make_course(seed=seed), render=True)
    state = env.reset()
    done = False
    while not done and env.render:
        action = 1 if net.activate(build_observation(state))[0] > 0.5 else 0
        state, _reward, done, _ = env.step(action)
        env.render_frame()

    env.close()
    print(f"seed={seed}  percent={state.percent:.1%}  complete={state.complete}")


if __name__ == "__main__":
    main()

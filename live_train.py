"""Evolve a NEAT network that plays REAL Geometry Dash via the Geode bridge,
with a separate window showing the live neural network as it learns.

    python live_train.py

Requires: the GDBot Bridge mod installed, GD running, and you sitting inside a
level (e.g. Stereo Madness) with auto-retry on. Each genome drives one attempt;
fitness = furthest % reached. Watch the overlay window: nodes light up and
connections (green +, red -) strengthen as it learns the level's timing.

v1 observation is kinematics-only (percent, y, vy, on-ground) — the network
learns THIS level. Reactive/generalist play comes when the mod streams hazards.
"""

import os
import pickle
import sys

import neat
import numpy as np
import pygame

from gdbot import netviz
from gdbot.live_env import LiveEnv

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE_OBS = 4
_hud = {"gen": 0, "genome": 0, "best": 0.0, "pct": 0.0}
_win = {"screen": None, "clock": None, "font": None}


def build_obs(state) -> np.ndarray:
    return np.array([
        state.percent,
        float(np.clip(state.player_y / 500.0, -1.0, 1.0)),
        float(np.clip(state.vy / 1000.0, -1.0, 1.0)),
        1.0 if state.on_ground else 0.0,
    ], dtype=np.float32)


def _overlay(net):
    w = _win
    if w["screen"] is None:
        pygame.init(); pygame.font.init()
        w["screen"] = pygame.display.set_mode((760, 460))
        w["clock"] = pygame.time.Clock()
        w["font"] = pygame.font.SysFont("consolas", 18)
        pygame.display.set_caption("gdbot — live neural network (real GD)")
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            raise KeyboardInterrupt
    s = w["screen"]
    s.fill((10, 10, 16))
    hud = (f"gen {_hud['gen']}   genome {_hud['genome']}   "
           f"best {_hud['best']:.2f}   now {_hud['pct']*100:4.1f}%")
    s.blit(w["font"].render(hud, True, (235, 235, 245)), (14, 12))
    netviz.draw_network(s, net, pygame.Rect(0, 44, 760, 416))
    pygame.display.flip()
    w["clock"].tick(120)


def eval_genomes(genomes, config):
    for gid, genome in genomes:
        _hud["genome"] = gid
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        state = env.reset()
        best = 0.0
        while True:
            action = 1 if net.activate(build_obs(state))[0] > 0.5 else 0
            _overlay(net)
            state, _r, done, _info = env.step(action)
            best = max(best, state.percent)
            _hud["pct"] = state.percent
            if done:
                break
        genome.fitness = best + (1.0 if state.complete else 0.0)
        _hud["best"] = max(_hud["best"], genome.fitness)


def main():
    global env
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        os.path.join(HERE, "live_neat_config.txt"),
    )
    env = LiveEnv()
    print("Waiting for the GDBot Bridge (start GD, enter a level)...")
    if not env.wait_connected(timeout=120):
        print("Bridge never connected — is the mod installed and are you in a level?")
        sys.exit(1)
    print("Connected. Evolving — watch the overlay window.")

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))

    class GenTick(neat.reporting.BaseReporter):
        def start_generation(self, gen):
            _hud["gen"] = gen
    pop.add_reporter(GenTick())

    try:
        winner = pop.run(eval_genomes, 200)
        with open(os.path.join(HERE, "live_winner.pkl"), "wb") as f:
            pickle.dump(winner, f)
        print(f"Saved live_winner.pkl (fitness={winner.fitness:.3f})")
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

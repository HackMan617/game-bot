"""Evolve a NEAT network that plays REAL Geometry Dash via the Geode bridge,
with a compact live network overlay and a CSV log of every generation/genome.

    python live_train.py

Requires: the GDBot Bridge mod installed, GD running, and you sitting inside a
level (e.g. Stereo Madness) with auto-retry on. Each genome drives one attempt;
fitness = furthest % reached (+1 for completing).

Reactive observation (22 inputs): for each of 10 look-ahead cells the network
sees (surface height, spike), plus on-ground and vertical velocity. So it reacts
to spikes to jump over AND blocks to jump onto — not memorized positions.

Logs (live_log/):
  generations.csv  one row per generation: best/mean fitness, best %, running best
  genomes.csv      one row per genome: fitness, max %, completed, steps survived
"""

import csv
import os
import pickle
import sys

import neat
import numpy as np
import pygame

from gdbot import netviz
from gdbot.live_env import BridgeLost, LiveEnv

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "live_log")
LIVE_OBS = 22  # 10 cells x (surface, spike) + on-ground + vy

_hud = {"gen": 0, "genome": 0, "best": 0.0, "pct": 0.0, "obest": 0.0}
_win = {"screen": None, "clock": None, "font": None}
_genome_stats = {}                      # gid -> {percent, steps, complete}
_log = {"gw": None, "sw": None, "gf": None, "sf": None, "obest": 0.0}


def build_obs(state) -> np.ndarray:
    """Reactive: (surface height, spike) per look-ahead cell, plus motion."""
    obs = []
    for surf, spike in state.lookahead(10):
        obs.append(float(np.clip(surf / 6.0, -1.0, 1.0)))  # relative height (blocks)
        obs.append(1.0 if spike else 0.0)
    obs.append(1.0 if state.on_ground else 0.0)
    obs.append(float(np.clip(state.vy / 1000.0, -1.0, 1.0)))
    return np.array(obs, dtype=np.float32)


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
    hud = (f"gen {_hud['gen']}  genome {_hud['genome']}  best {_hud['best']:.2f}  "
           f"now {_hud['pct']*100:4.1f}%  best% {_hud['obest']*100:4.1f}")
    s.blit(w["font"].render(hud, True, (235, 235, 245)), (14, 12))
    netviz.draw_network(s, net, pygame.Rect(0, 44, 760, 416))
    pygame.display.flip()
    w["clock"].tick(120)


def eval_genomes(genomes, config):
    for gid, genome in genomes:
        _hud["genome"] = gid
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        state = env.reset()
        best, steps = 0.0, 0
        while True:
            action = 1 if net.activate(build_obs(state))[0] > 0.5 else 0
            _overlay(net)
            state, _r, done, _info = env.step(action)
            best = max(best, state.percent)
            steps += 1
            _hud["pct"] = state.percent
            if done:
                break
        genome.fitness = best + (1.0 if state.complete else 0.0)
        _hud["best"] = max(_hud["best"], genome.fitness)
        _genome_stats[gid] = {"percent": best, "steps": steps, "complete": state.complete}


class GenLogger(neat.reporting.BaseReporter):
    """Write per-generation and per-genome CSV rows + a console summary."""

    def start_generation(self, gen):
        self._gen = gen
        _hud["gen"] = gen

    def post_evaluate(self, config, population, species, best_genome):
        gen = getattr(self, "_gen", 0)
        fits, best_pct = [], 0.0
        for gid, g in population.items():
            st = _genome_stats.get(gid, {})
            pct = st.get("percent", 0.0)
            fits.append(g.fitness if g.fitness is not None else 0.0)
            best_pct = max(best_pct, pct)
            _log["gw"].writerow([gen, gid, round(g.fitness or 0.0, 4), round(pct, 4),
                                 int(st.get("complete", False)), st.get("steps", 0)])
        _log["obest"] = max(_log["obest"], best_pct)
        _hud["obest"] = _log["obest"]
        mean = sum(fits) / len(fits) if fits else 0.0
        bf = best_genome.fitness if best_genome else 0.0
        _log["sw"].writerow([gen, len(population), round(bf, 4), round(mean, 4),
                             best_genome.key if best_genome else -1,
                             round(best_pct, 4), round(_log["obest"], 4)])
        _log["gf"].flush(); _log["sf"].flush()
        _genome_stats.clear()
        print(f"[gen {gen}] best_fit={bf:.3f}  best%={best_pct*100:.1f}  "
              f"mean_fit={mean:.3f}  overall_best%={_log['obest']*100:.1f}")


def main():
    global env
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        os.path.join(HERE, "live_neat_config.txt"),
    )
    os.makedirs(LOG_DIR, exist_ok=True)
    _log["gf"] = open(os.path.join(LOG_DIR, "genomes.csv"), "w", newline="")
    _log["sf"] = open(os.path.join(LOG_DIR, "generations.csv"), "w", newline="")
    _log["gw"] = csv.writer(_log["gf"])
    _log["gw"].writerow(["gen", "genome_id", "fitness", "max_percent", "complete", "steps"])
    _log["sw"] = csv.writer(_log["sf"])
    _log["sw"].writerow(["gen", "n_genomes", "best_fitness", "mean_fitness",
                         "best_genome_id", "best_percent", "overall_best_percent"])

    env = LiveEnv()
    print("Waiting for the GDBot Bridge (start GD, enter a level)...")
    if not env.wait_connected(timeout=120):
        print("Bridge never connected — is the mod installed and are you in a level?")
        sys.exit(1)
    print(f"Connected. Evolving — watch the overlay. Logs -> {LOG_DIR}")

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(GenLogger())

    try:
        winner = pop.run(eval_genomes, 300)
        with open(os.path.join(HERE, "live_winner.pkl"), "wb") as f:
            pickle.dump(winner, f)
        print(f"Saved live_winner.pkl (fitness={winner.fitness:.3f})")
    except (KeyboardInterrupt, BridgeLost) as exc:
        print(f"\nstopped: {exc}")
        best = getattr(pop, "best_genome", None)
        if best is not None:
            with open(os.path.join(HERE, "live_winner.pkl"), "wb") as f:
                pickle.dump(best, f)
            print(f"Saved best-so-far genome (fitness={best.fitness})")
    finally:
        env.close()
        for k in ("gf", "sf"):
            if _log[k]:
                _log[k].close()
        print("logs finalized in", LOG_DIR)


if __name__ == "__main__":
    main()

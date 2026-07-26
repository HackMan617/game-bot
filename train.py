"""Train the NEAT bot against the deterministic simulator.

    python train.py [generations]

Saves the best genome to winner.pkl and a fitness plot to fitness.png.
No real game required — this proves the network can learn to play.
"""

import os
import pickle
import sys

from gdbot.neat_core import run

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    generations = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    config_path = os.path.join(HERE, "neat_config.txt")

    winner, stats, _config = run(config_path, generations=generations)

    with open(os.path.join(HERE, "winner.pkl"), "wb") as f:
        pickle.dump(winner, f)
    print(f"\nBest genome saved to winner.pkl (fitness={winner.fitness:.3f})")

    # Optional: plot fitness over generations if matplotlib is available.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        best = [c.fitness for c in stats.most_fit_genomes]
        mean = stats.get_fitness_mean()
        plt.figure(figsize=(9, 4))
        plt.plot(best, label="best")
        plt.plot(mean, label="mean")
        plt.xlabel("generation")
        plt.ylabel("fitness (max 2.0)")
        plt.title("gdbot NEAT training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, "fitness.png"))
        print("Fitness plot saved to fitness.png")
    except Exception as exc:  # plotting is a nicety, never fatal
        print(f"(skipped fitness plot: {exc})")


if __name__ == "__main__":
    main()

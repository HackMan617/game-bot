"""Generate a PDF report of a live training run from the CSV logs.

    python report.py

Reads live_log/generations.csv + live_log/genomes.csv and writes
live_log/report.pdf: a learning curve, fitness curves, the genome-result
distribution, and a table of the top genomes. Colours are the Okabe-Ito
colourblind-safe pair (blue/orange); text stays in neutral ink.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "live_log")

BLUE, ORANGE = "#0072B2", "#E69F00"          # Okabe-Ito, CVD-safe pair
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"


def _read(path, required):
    """Read a CSV, skipping rows whose required fields aren't fully parseable
    (a hard-killed run can leave a truncated final line)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                for key, cast in required.items():
                    cast(r[key])
                rows.append(r)
            except (ValueError, KeyError, TypeError):
                continue
    return rows


def main():
    gens = _read(os.path.join(LOG, "generations.csv"),
                 {"gen": int, "best_fitness": float, "mean_fitness": float,
                  "best_percent": float, "overall_best_percent": float})
    genomes = _read(os.path.join(LOG, "genomes.csv"),
                    {"gen": int, "genome_id": int, "max_percent": float,
                     "complete": int, "steps": int})
    if not gens:
        print("No generation data yet — run live_train.py first.")
        return

    g = [int(r["gen"]) for r in gens]
    best_fit = [float(r["best_fitness"]) for r in gens]
    mean_fit = [float(r["mean_fitness"]) for r in gens]
    best_pct = [float(r["best_percent"]) * 100 for r in gens]
    obest_pct = [float(r["overall_best_percent"]) * 100 for r in gens]
    gm_pct = [float(r["max_percent"]) * 100 for r in genomes]
    n_complete = sum(1 for r in genomes if int(r["complete"]))
    best_row = max(genomes, key=lambda r: float(r["max_percent"]))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlecolor": INK, "figure.facecolor": "white",
    })

    pdf = PdfPages(os.path.join(LOG, "report.pdf"))

    # --- Page 1: summary + learning curve -------------------------------------
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("gdbot — Live Training Report", fontsize=20, fontweight="bold", y=0.965)
    fig.text(0.5, 0.925, "Reactive NEAT on real Geometry Dash — Stereo Madness",
             ha="center", color=MUTED, fontsize=12)
    summary = [
        f"Generations: {len(g)}        Genomes evaluated: {len(genomes)}",
        f"Best reached — gen 0: {best_pct[0]:.1f}%     final gen: {best_pct[-1]:.1f}%"
        f"     overall best: {max(obest_pct):.1f}%",
        f"Top genome: gen {best_row['gen']}, id {best_row['genome_id']} "
        f"→ {float(best_row['max_percent'])*100:.1f}%     level completed: "
        f"{'yes' if n_complete else 'no'}",
    ]
    fig.text(0.09, 0.87, "\n".join(summary), va="top", fontsize=12.5, color=INK,
             linespacing=1.7)

    ax = fig.add_axes([0.09, 0.09, 0.85, 0.60])
    ax.plot(g, obest_pct, color=BLUE, lw=2.2, label="overall best % (running max)")
    ax.plot(g, best_pct, color=ORANGE, lw=1.4, label="best % this generation")
    ax.set_xlabel("generation")
    ax.set_ylabel("% of level reached")
    ax.set_title("Learning curve — how far the best network gets over generations")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.annotate(f"{max(obest_pct):.1f}%", xy=(g[-1], obest_pct[-1]),
                xytext=(-6, 6), textcoords="offset points", color=BLUE, fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 2: fitness curves + genome distribution -------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.subplots_adjust(hspace=0.4, left=0.09, right=0.94, top=0.93, bottom=0.09)
    a1.plot(g, best_fit, color=BLUE, lw=2.2, label="best fitness")
    a1.plot(g, mean_fit, color=ORANGE, lw=1.6, label="mean fitness")
    a1.set_title("Fitness over generations")
    a1.set_xlabel("generation")
    a1.set_ylabel("fitness (percent + completion bonus)")
    a1.grid(True, color=GRID, lw=0.6)
    a1.set_axisbelow(True)
    a1.legend(frameon=False)

    a2.hist(gm_pct, bins=30, color=BLUE)
    a2.set_title("Distribution of every genome's result — the stars vs the flops")
    a2.set_xlabel("% of level reached")
    a2.set_ylabel("number of genomes")
    a2.grid(True, color=GRID, lw=0.6, axis="y")
    a2.set_axisbelow(True)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 3: top genomes table --------------------------------------------
    top = sorted(genomes, key=lambda r: float(r["max_percent"]), reverse=True)[:15]
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Top 15 genomes by furthest progress", fontsize=16, fontweight="bold", y=0.95)
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.85])
    ax.axis("off")
    rows = [[i + 1, r["gen"], r["genome_id"], f"{float(r['max_percent'])*100:.1f}%",
             r["steps"], "yes" if int(r["complete"]) else "—"] for i, r in enumerate(top)]
    tbl = ax.table(cellText=rows,
                   colLabels=["rank", "generation", "genome id", "best %", "steps survived", "completed"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.7)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(fontweight="bold", color=INK)
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()
    print("wrote", os.path.join(LOG, "report.pdf"))


if __name__ == "__main__":
    main()

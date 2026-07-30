"""Generate a PDF report of a training run from its CSV logs.

    python report.py [run_name]        # default: gd

Reads runs/<name>/updates.csv + runs/<name>/episodes.csv and writes
runs/<name>/report.pdf: the learning curve, the PPO diagnostics that say whether
the run is healthy, and the distribution of individual attempts. Colours are the
Okabe-Ito colourblind-safe pair (blue/orange); text stays in neutral ink.
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE, ORANGE = "#0072B2", "#E69F00"          # Okabe-Ito, CVD-safe pair
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"


def _read(path, required):
    """Read a CSV, skipping rows whose required fields aren't fully parseable
    (a hard-killed run can leave a truncated final line)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                for key, cast in required.items():
                    cast(r[key])
                rows.append(r)
            except (ValueError, KeyError, TypeError):
                continue
    return rows


def _col(rows, key, cast=float, scale=1.0):
    return [cast(r[key]) * scale for r in rows]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "gd"
    log_dir = os.path.join(HERE, "runs", run)

    ups = _read(os.path.join(log_dir, "updates.csv"),
                {"update": int, "env_steps": int, "mean_return": float,
                 "mean_percent": float, "best_percent": float, "entropy": float,
                 "kl": float, "v_loss": float, "steps_per_sec": float})
    eps = _read(os.path.join(log_dir, "episodes.csv"),
                {"episode": int, "env_steps": int, "return": float,
                 "best_percent": float, "steps": int, "complete": int})
    if not ups:
        print(f"No update data in {log_dir} — run `python train.py --run {run}` first.")
        return

    steps = _col(ups, "env_steps", int)
    msteps = [s / 1e6 for s in steps]
    mean_pct = _col(ups, "mean_percent", scale=100)
    best_pct = _col(ups, "best_percent", scale=100)
    mean_ret = _col(ups, "mean_return")
    entropy = _col(ups, "entropy")
    kl = _col(ups, "kl")
    v_loss = _col(ups, "v_loss")
    sps = _col(ups, "steps_per_sec")

    ep_pct = _col(eps, "best_percent", scale=100) if eps else []
    n_complete = sum(int(r["complete"]) for r in eps) if eps else 0
    wall_h = (float(ups[-1]["wall"]) / 3600.0) if "wall" in ups[-1] else 0.0

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlecolor": INK, "figure.facecolor": "white",
    })
    pdf = PdfPages(os.path.join(log_dir, "report.pdf"))

    # --- Page 1: summary + learning curve -------------------------------------
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("gdbot — PPO Training Report", fontsize=20, fontweight="bold", y=0.965)
    fig.text(0.5, 0.925, f'run "{run}" — conv policy over the 24x16x4 occupancy grid',
             ha="center", color=MUTED, fontsize=12)
    summary = [
        f"PPO updates: {len(ups):,}        Environment steps: {steps[-1]:,}"
        f"        Episodes: {len(eps):,}        Wall clock: {wall_h:.1f} h",
        f"Mean progress — first update: {mean_pct[0]:.1f}%     final: {mean_pct[-1]:.1f}%"
        f"     best single attempt: {max(best_pct):.1f}%",
        f"Completions: {n_complete:,}        Throughput: {sum(sps)/len(sps):,.0f} steps/s"
        f"        Final policy entropy: {entropy[-1]:.3f} (max 0.693)",
    ]
    fig.text(0.09, 0.87, "\n".join(summary), va="top", fontsize=12, color=INK,
             linespacing=1.7)

    ax = fig.add_axes([0.09, 0.09, 0.85, 0.60])
    ax.plot(msteps, best_pct, color=BLUE, lw=2.2, label="best attempt so far (running max)")
    ax.plot(msteps, mean_pct, color=ORANGE, lw=1.4, label="mean of the last 200 attempts")
    _style(ax, "Learning curve — how far the agent gets as it trains",
           "environment steps (millions)", "% of level reached")
    ax.legend(frameon=False)
    ax.annotate(f"{mean_pct[-1]:.1f}%", xy=(msteps[-1], mean_pct[-1]),
                xytext=(-6, 6), textcoords="offset points", color=ORANGE, fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 2: the diagnostics that say whether the run is healthy ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.subplots_adjust(hspace=0.42, wspace=0.28, left=0.09, right=0.95,
                        top=0.90, bottom=0.09)
    fig.suptitle("Is the run healthy?", fontsize=16, fontweight="bold")

    a = axes[0][0]
    a.plot(msteps, mean_ret, color=BLUE, lw=1.6)
    _style(a, "Mean episode return", "steps (M)", "return")

    a = axes[0][1]
    a.plot(msteps, entropy, color=BLUE, lw=1.6)
    a.axhline(0.693, color=MUTED, lw=0.8, ls="--")
    a.annotate("uniform (0.693)", xy=(msteps[0], 0.693), xytext=(2, 3),
               textcoords="offset points", color=MUTED, fontsize=9)
    # Entropy falling means the policy is committing; a flatline at 0.693 means
    # it is still coin-flipping, and a crash to 0 means it stopped exploring.
    _style(a, "Policy entropy — is it committing?", "steps (M)", "nats")

    a = axes[1][0]
    a.plot(msteps, kl, color=ORANGE, lw=1.4)
    _style(a, "KL per update — step size", "steps (M)", "KL divergence")

    a = axes[1][1]
    a.plot(msteps, v_loss, color=ORANGE, lw=1.4)
    _style(a, "Value loss — can the critic predict returns?", "steps (M)", "MSE")
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 3: individual attempts ------------------------------------------
    if ep_pct:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.subplots_adjust(hspace=0.4, left=0.09, right=0.94, top=0.93, bottom=0.09)

        a1.scatter(_col(eps, "env_steps", int), ep_pct, s=4, alpha=0.28, color=BLUE,
                   edgecolors="none")
        _style(a1, "Every attempt — the spread is what the mean above hides",
               "environment steps", "% reached")

        a2.hist(ep_pct, bins=40, color=BLUE)
        _style(a2, "Distribution of attempts — where the agent keeps dying",
               "% of level reached", "attempts")
        a2.grid(True, color=GRID, lw=0.6, axis="y")
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print("wrote", os.path.join(log_dir, "report.pdf"))


if __name__ == "__main__":
    main()

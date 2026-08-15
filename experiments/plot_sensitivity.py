"""Plot QUBO energy against compute budget — does more search rescue the annealer?

One question, drawn: **is the penalty barrier a matter of budget?** The x-axis is
wall-clock seconds actually spent, the y-axis is the QUBO energy reached — the objective
the sampler is minimising, computed from the BQM it was handed, so a solver that loses
here has lost on its own terms.

Two horizontal reference lines carry the argument. `greedy_topk` and `mmr` do **no search
at all**; they are single deterministic passes. Any point above those lines is a solver
that spent compute to do worse than not searching.

Usage::

    python experiments/plot_sensitivity.py --sensitivity results/sensitivity.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STYLE = {
    "qubo_sa": ("tab:red", "o", "QUBO + neal SA"),
    "qubo_tabu": ("tab:orange", "s", "QUBO + tabu"),
    "qubo_feasible": ("tab:blue", "D", "QUBO + swap annealing"),
}
REFERENCE = {
    "greedy_topk": ("black", "greedy top-k (no search)"),
    "mmr": ("dimgray", "MMR (no search)"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sensitivity", type=Path,
                    default=ROOT / "results" / "sensitivity.csv")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    frame = pd.read_csv(args.sensitivity)
    n_seeds = frame["repeat"].nunique()

    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    for name, (colour, label) in REFERENCE.items():
        rows = frame[frame.method == name]
        if len(rows):
            ax.axhline(rows["energy"].mean(), color=colour, ls="--", lw=1.3, alpha=0.8)
            ax.text(0.13, rows["energy"].mean(), f"  {label}", color=colour,
                    va="bottom", fontsize=9)

    for name, (colour, marker, label) in STYLE.items():
        rows = frame[frame.method == name]
        if not len(rows):
            continue
        stats = rows.groupby("budget", sort=False).agg(
            energy=("energy", "mean"),
            spread=("energy", "std"),
            seconds=("seconds", "mean"),
        ).sort_values("seconds")

        ax.errorbar(stats.seconds, stats.energy, yerr=stats.spread.fillna(0.0),
                    fmt=marker + "-", color=colour, markersize=8, capsize=3,
                    lw=1.6, label=label)

    ax.axhline(0.0, color="black", lw=0.8, alpha=0.35)
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock seconds spent (log scale)  →  more compute")
    ax.set_ylabel("QUBO energy reached  →  lower is better")
    ax.set_title(
        "Does more compute rescue the penalty-encoded annealer?\n"
        f"barrier instance, n=200, k=10, λ=4, mean of {n_seeds} seeds",
        fontsize=12.5,
    )
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False, fontsize=9.5, loc="center right")

    fig.text(
        0.5, 0.005,
        "neal never reaches zero at any budget; the constraint-preserving solver is "
        "already below it after 0.2 s.",
        ha="center", fontsize=9, color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    out = args.out or (ROOT / "results" / "sensitivity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

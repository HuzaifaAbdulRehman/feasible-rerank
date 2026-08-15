"""Plot the fairness-budget curve produced by ``protocol.py``.

One question, drawn: **at a given fairness requirement, how accurate can each method
be?** The x-axis is the declared budget `tau` -- an upper bound on mean exposure-parity
deviation -- and the y-axis is NDCG@k on users that played no part in choosing the
configuration being scored.

Two things this figure does that the Pareto plot cannot:

* **Selection and evaluation are separated.** Every point is a configuration chosen on
  a disjoint half of the users. A Pareto scatter over a sweep shows the best cell found;
  this shows what that cell is worth on data it was not chosen on, which is a smaller
  and more honest number.
* **Infeasibility is visible.** A method that cannot meet the budget at any setting is
  drawn hollow and cut from the line rather than plotted at its least-bad point. A
  method that cannot satisfy the requirement has not competed, and a solid marker would
  say otherwise.

Usage::

    python experiments/plot_protocol.py --protocol results/amazon_lb_protocol.csv
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
    "greedy_topk": ("black", "X", "greedy top-k"),
    "mmr": ("dimgray", "P", "MMR"),
    "quota_mmr": ("darkgreen", "*", "quota-MMR"),
    "qubo_sa": ("tab:red", "o", "QUBO + neal SA"),
    "qubo_tabu": ("tab:orange", "s", "QUBO + tabu"),
    "qubo_feasible": ("tab:blue", "D", "QUBO + swap annealing"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    frame = pd.read_csv(args.protocol)
    n_seeds = frame["seed"].nunique()

    fig, (ax_ndcg, ax_parity) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    for method, rows in frame.groupby("method", sort=False):
        colour, marker, label = STYLE.get(method, ("tab:purple", "o", method))

        stats = rows.groupby("tau").agg(
            ndcg=("ndcg@k", "mean"),
            ndcg_sd=("ndcg@k", "std"),
            parity=("exposure_parity", "mean"),
            parity_sd=("exposure_parity", "std"),
            # A budget counts as met only if it was met on every seed. Anything less is
            # a method that sometimes cannot satisfy the requirement, which for a
            # service-level constraint is the same as not satisfying it.
            feasible=("feasible", "all"),
        ).reset_index()

        met = stats[stats.feasible]
        missed = stats[~stats.feasible]

        # The line connects only budgets the method actually met.
        if len(met) > 1:
            ax_ndcg.plot(met.tau, met.ndcg, color=colour, lw=1.6, alpha=0.9, zorder=3)
        if len(met):
            ax_ndcg.errorbar(
                met.tau, met.ndcg, yerr=met.ndcg_sd.fillna(0.0),
                fmt=marker, color=colour, markersize=9, capsize=3, lw=1.2,
                label=label, zorder=4,
            )
        if len(missed):
            # Hollow: shown so the reader can see the method was entered and failed,
            # not silently dropped -- but never joined to the line.
            ax_ndcg.scatter(
                missed.tau, missed.ndcg, facecolors="none", edgecolors=colour,
                marker=marker, s=90, linewidths=1.4, zorder=4,
                label=None if len(met) else label,
            )

        # No label: the left panel already carries the method legend, and repeating it
        # here covered the data it was meant to explain.
        ax_parity.errorbar(
            stats.tau, stats.parity, yerr=stats.parity_sd.fillna(0.0),
            fmt=marker, color=colour, markersize=8, capsize=3, lw=1.2, alpha=0.9,
        )

    taus = sorted(frame["tau"].unique())
    ax_parity.plot(taus, taus, color="crimson", ls="--", lw=1.2, alpha=0.7,
                   label="the budget itself (parity = tau)", zorder=1)

    ax_ndcg.set_xlabel("fairness budget  tau  (max mean exposure-parity deviation)\nlooser ->")
    ax_ndcg.set_ylabel("NDCG@10 on held-out users  ->  better")
    ax_ndcg.set_title("Accuracy available at each fairness requirement", fontsize=12)
    ax_ndcg.grid(alpha=0.25, linestyle=":")
    ax_ndcg.legend(frameon=False, fontsize=9, loc="lower right")

    ax_parity.set_xlabel("fairness budget  tau")
    ax_parity.set_ylabel("achieved exposure parity on held-out users  ->  worse")
    ax_parity.set_title("Was the budget actually met?", fontsize=12)
    ax_parity.grid(alpha=0.25, linestyle=":")
    ax_parity.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle(
        args.title
        or f"Tuned and evaluated on disjoint users -- {args.protocol.stem} "
           f"({n_seeds} seeds)",
        fontsize=13,
    )
    fig.text(
        0.5, 0.005,
        "Hollow markers: no configuration met the budget on every seed. "
        "Points above the dashed line in the right panel would be violations.",
        ha="center", fontsize=8.5, color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))

    out = args.out or (ROOT / "results" / f"{args.protocol.stem}_budget.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    # Printed alongside the figure so the numbers behind it are visible without a
    # spreadsheet -- the figure is for reading, the table is for checking.
    print("\nNDCG@10 on held-out users, by fairness budget (methods that met it):\n")
    pivot = (
        frame[frame.feasible]
        .pivot_table(index="method", columns="tau", values="ndcg@k", aggfunc="mean")
        .reindex([m for m in STYLE if m in set(frame.method)])
    )
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}", na_rep="  --  "))


if __name__ == "__main__":
    main()

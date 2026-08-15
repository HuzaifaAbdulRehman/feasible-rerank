"""Does the feasibility result hold on more than one catalogue?

The single-dataset finding is that below a fairness budget of roughly 0.25, no classical
baseline can satisfy the constraint at any setting of its own hyperparameters, while the
QUBO can. Stated that way it is a claim about *methods*. It could just as easily be a
claim about Amazon Luxury Beauty's popularity structure, and one dataset cannot tell the
two apart.

This reads every ``*_protocol.csv`` in ``results/`` and asks one question of each:

    **What is the tightest fairness budget each method can actually meet?**

That number -- call it the method's *reach* -- is the whole finding in one figure. If the
QUBO's reach is consistently tighter than every classical baseline's across catalogues
that differ in size, density and difficulty, the result is about the method. If it moves
around, it is about the data.

The datasets are deliberately run with identical budgets, grids and seeds, so nothing
here is tuned per-catalogue to look better.

Usage::

    python experiments/compare_datasets.py
    python experiments/compare_datasets.py --protocol results/amazon_lb_protocol.csv ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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

#: Printed alongside each dataset so the reader can see what was varied. Density and
#: catalogue size are the two properties the finding could plausibly depend on.
DATASET_NOTES = {
    "amazon_lb": "Luxury Beauty  1,366 items  density 0.0067  hit rate 0.49",
    "amazon_software": "Software  729 items  density 0.0096  hit rate 0.28",
    "amazon_giftcards": "Gift Cards  147 items  density 0.0442  n=100 candidates",
}


def reach(frame: pd.DataFrame) -> pd.DataFrame:
    """Tightest budget each method meets on *every* seed, per dataset.

    "On every seed" rather than "on average": a fairness budget is a service-level
    constraint, and a method that meets it two runs in three has not met it. NaN means
    the method never met any budget in the grid.
    """
    rows = []
    for (dataset, method), block in frame.groupby(["dataset", "method"], sort=False):
        met = block.groupby("tau")["feasible"].all()
        feasible_taus = met[met].index
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "reach": float(feasible_taus.min()) if len(feasible_taus) else np.nan,
                # Accuracy at that tightest budget -- reach is only interesting if the
                # method is still returning usable lists when it gets there.
                "ndcg_at_reach": (
                    float(block[block.tau == feasible_taus.min()]["ndcg@k"].mean())
                    if len(feasible_taus)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def load(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["dataset"] = path.stem.replace("_protocol", "")
        frames.append(frame)
    if not frames:
        raise SystemExit("no protocol CSVs found; run experiments/protocol.py first")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", nargs="+", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paths = args.protocol or sorted((ROOT / "results").glob("*_protocol.csv"))
    frame = load(paths)
    datasets = list(dict.fromkeys(frame["dataset"]))
    print(f"datasets: {', '.join(datasets)}\n")

    reaches = reach(frame)

    # ---- the synthesis table -------------------------------------------------
    print("Tightest fairness budget each method can meet, on every seed")
    print("(lower is a stronger guarantee; -- means it never met any budget tested)\n")
    pivot = reaches.pivot(index="method", columns="dataset", values="reach")
    pivot = pivot.reindex([m for m in STYLE if m in set(reaches.method)])
    print(pivot.to_string(float_format=lambda v: f"{v:.2f}", na_rep="  --"))

    print("\nNDCG@10 delivered at that tightest budget\n")
    acc = reaches.pivot(index="method", columns="dataset", values="ndcg_at_reach")
    acc = acc.reindex([m for m in STYLE if m in set(reaches.method)])
    print(acc.to_string(float_format=lambda v: f"{v:.4f}", na_rep="  --"))

    # ---- the verdict, stated per dataset rather than in aggregate ------------
    print("\nPer dataset: is any QUBO variant's reach strictly tighter than every "
          "classical baseline's?\n")
    verdicts = []
    for dataset in datasets:
        block = reaches[reaches.dataset == dataset]
        qubo = block[block.method.str.startswith("qubo")]["reach"].dropna()
        classical = block[~block.method.str.startswith("qubo")]["reach"].dropna()

        if qubo.empty:
            verdict = "no QUBO variant met any budget"
        elif classical.empty:
            verdict = f"YES -- QUBO reaches {qubo.min():.2f}; no baseline met any budget"
        elif qubo.min() < classical.min():
            verdict = f"YES -- QUBO {qubo.min():.2f} vs best baseline {classical.min():.2f}"
        else:
            verdict = f"NO -- QUBO {qubo.min():.2f}, best baseline {classical.min():.2f}"
        verdicts.append(verdict.startswith("YES"))
        print(f"  {dataset:<20} {verdict}")

    held = sum(verdicts)
    print(
        f"\nHolds on {held} of {len(datasets)} datasets. "
        + (
            "The result is about the method."
            if held == len(datasets)
            else "It is not a property of the method alone -- say so."
        )
    )

    # ---- small multiples -----------------------------------------------------
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5.0 * len(datasets), 5.0), squeeze=False, sharey=True
    )

    for ax, dataset in zip(axes[0], datasets, strict=True):
        block = frame[frame.dataset == dataset]
        for method, rows in block.groupby("method", sort=False):
            colour, marker, label = STYLE.get(method, ("tab:purple", "o", method))
            stats = rows.groupby("tau").agg(
                ndcg=("ndcg@k", "mean"), feasible=("feasible", "all")
            ).reset_index()

            met, missed = stats[stats.feasible], stats[~stats.feasible]
            if len(met) > 1:
                ax.plot(met.tau, met.ndcg, color=colour, lw=1.5, alpha=.9, zorder=3)
            if len(met):
                ax.scatter(met.tau, met.ndcg, color=colour, marker=marker, s=62,
                           zorder=4, label=label)
            if len(missed):
                ax.scatter(missed.tau, missed.ndcg, facecolors="none", edgecolors=colour,
                           marker=marker, s=62, linewidths=1.3, zorder=4,
                           label=None if len(met) else label)

        ax.set_title(DATASET_NOTES.get(dataset, dataset), fontsize=9.5)
        ax.set_xlabel("fairness budget  tau\nlooser ->")
        ax.grid(alpha=.25, linestyle=":")

    axes[0][0].set_ylabel("NDCG@10 on held-out users  ->  better")

    handles, labels = [], []
    for ax in axes[0]:
        for h, s in zip(*ax.get_legend_handles_labels(), strict=True):
            if s not in labels:
                handles.append(h)
                labels.append(s)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(.5, -.03), fontsize=9)

    fig.suptitle(
        "Does the feasibility result survive a change of catalogue?  "
        "Identical budgets, grids and seeds throughout.",
        fontsize=12.5,
    )
    fig.text(.5, .005, "Hollow markers: the method could not meet that budget on every seed.",
             ha="center", fontsize=8.5, color="dimgray")
    fig.tight_layout(rect=(0, .05, 1, .94))

    out = args.out or (ROOT / "results" / "datasets_budget.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")

    table_out = out.with_suffix(".csv")
    reaches.to_csv(table_out, index=False)
    print(f"wrote {table_out}")


if __name__ == "__main__":
    main()

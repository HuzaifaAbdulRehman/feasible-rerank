"""Plot the accuracy / diversity / fairness trade-off traced by ``sweep.py``.

Three panels, one per cost axis, all sharing NDCG on the y-axis:

* **Gini** -- popularity bias across the whole catalogue. The measure the QUBO does
  *not* optimise, which makes it the honest one.
* **exposure parity** -- deviation from equal group exposure. This one the fairness
  term does optimise directly, so a good number here is necessary but not impressive.
* **intra-list similarity** -- redundancy within a single list, what ``lam`` targets.

The classical baselines are drawn as fixed reference points. The question the figure
has to answer is not "does the QUBO move" -- any reranker moves -- but whether any
point on its curve sits above and to the left of the baselines. Where it does not, the
sweep has cost compute to reproduce MMR, and the write-up should say so.

Usage::

    python experiments/plot_pareto.py --sweep results/amazon_lb_sweep.csv \
        --baselines results/amazon_lb.csv
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

#: ``(column, axis label, what optimises it)``. The last entry is only plotted when the
#: sweep carries individual-item fairness, which requires held-out relevance labels and
#: so exists on the real benchmark but not the synthetic one.
COST_AXES = [
    ("gini", "Gini of item exposure", "popularity bias (not optimised)"),
    ("exposure_parity", "group exposure parity", "optimised by mu"),
    ("intra_list_sim", "intra-list similarity", "optimised by lam"),
    ("iif", "individual item unfairness (II-F)", "not optimised; Rampisela et al."),
]

BASELINE_STYLE = {
    "greedy_topk": ("black", "X"),
    "mmr": ("dimgray", "P"),
    "quota_mmr": ("darkgreen", "*"),
}

SWEEP_STYLE = {
    "qubo_sa": ("tab:red", "o"),
    "qubo_tabu": ("tab:orange", "s"),
    "qubo_feasible": ("tab:blue", "D"),
}


def pareto_front(frame: pd.DataFrame, cost: str) -> pd.DataFrame:
    """Points with no other point that is both cheaper and more accurate."""
    frame = frame.dropna(subset=[cost, "ndcg@k"])
    keep = []
    for _, row in frame.iterrows():
        dominated = (
            (frame[cost] <= row[cost])
            & (frame["ndcg@k"] >= row["ndcg@k"])
            & ((frame[cost] < row[cost]) | (frame["ndcg@k"] > row["ndcg@k"]))
        ).any()
        if not dominated:
            keep.append(row)
    return pd.DataFrame(keep).sort_values(cost)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True, type=Path)
    ap.add_argument("--baselines", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    sweep = pd.read_csv(args.sweep)

    # Drop any classical baseline rows carried in the sweep file. They ignore lam and
    # mu, so a sweep that included them holds one point repeated once per grid cell --
    # which would be drawn as a dense cluster masquerading as a curve, and would crowd
    # the Pareto front listing with identical rows. The baselines belong on the plot as
    # the fixed reference markers below, and only once.
    baseline_rows = sweep["method"].isin(BASELINE_STYLE)
    if baseline_rows.any():
        dropped = sorted(sweep.loc[baseline_rows, "method"].unique())
        print(f"ignoring baseline rows in the sweep file: {', '.join(dropped)}")
        sweep = sweep[~baseline_rows]

    # Default to the baselines sweep.py wrote for this exact benchmark. Passing a
    # headline results CSV instead is valid only when it used the same user sample --
    # catalogue-level metrics (Gini, catalogue coverage) do not transfer between sample
    # sizes, so mixing them puts two different scales on one axis.
    baseline_path = args.baselines
    if baseline_path is None:
        candidate = args.sweep.with_name(f"{args.sweep.stem}_baselines.csv")
        baseline_path = candidate if candidate.exists() else None
    baselines = pd.read_csv(baseline_path) if baseline_path else None
    if baseline_path:
        print(f"baselines: {baseline_path}")

    axes_to_plot = [
        axis
        for axis in COST_AXES
        if axis[0] in sweep.columns and sweep[axis[0]].notna().any()
    ]

    fig, axes = plt.subplots(
        1, len(axes_to_plot), figsize=(5.3 * len(axes_to_plot), 5.2), squeeze=False
    )
    axes = axes[0]

    for ax, (cost, xlabel, note) in zip(axes, axes_to_plot, strict=True):
        for method, group in sweep.groupby("method"):
            colour, marker = SWEEP_STYLE.get(method, ("tab:purple", "o"))
            ax.scatter(
                group[cost],
                group["ndcg@k"],
                c=colour,
                marker=marker,
                s=46,
                alpha=0.65,
                edgecolors="none",
                label=method,
            )
            front = pareto_front(group, cost)
            if len(front) > 1:
                ax.plot(front[cost], front["ndcg@k"], color=colour, lw=1.4, alpha=0.9)

        if baselines is not None:
            for _, row in baselines.iterrows():
                if row["method"] in BASELINE_STYLE and pd.notna(row.get(cost)):
                    colour, marker = BASELINE_STYLE[row["method"]]
                    ax.scatter(
                        row[cost],
                        row["ndcg@k"],
                        c=colour,
                        marker=marker,
                        s=210,
                        zorder=5,
                        label=row["method"],
                    )

        ax.set_xlabel(f"{xlabel}  ->  worse\n({note})")
        ax.set_ylabel("NDCG@k  ->  better")
        ax.grid(alpha=0.25, linestyle=":")

    # One legend for the figure; the three panels share every series.
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=len(unique),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        args.title or f"Accuracy vs diversity and fairness -- {args.sweep.stem}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    out = args.out or (ROOT / "results" / f"{args.sweep.stem}_pareto.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    for cost, _, _ in axes_to_plot:
        front = pareto_front(sweep, cost)
        print(f"\nPareto front on {cost}:")
        print(
            front[["method", "lam", "mu", "ndcg@k", cost]].to_string(
                index=False, float_format=lambda v: f"{v:.4f}"
            )
        )


if __name__ == "__main__":
    main()

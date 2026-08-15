"""Render the README's result tables straight from the CSVs in ``results/``.

The tables in a README are the part a reader actually checks, and they are also the
part most likely to drift: numbers get copied by hand, a run is repeated, one cell is
updated and its neighbour is not. This repo has already shipped one such error -- a
claimed II-F spread of 0.003% that was really 2.6% -- which is precisely the kind of
mistake that survives proofreading, because a plausible number in a well-formatted
table does not look wrong.

So the tables are generated. Anything printed here is a direct read of the CSV the
experiment wrote, and the arrows and bolding are applied mechanically.

Usage::

    python experiments/tables.py                          # every table
    python experiments/tables.py --repeats amazon_lb      # just the repeat study
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# The "lower is better" arrows are U+2193, which the default Windows console codepage
# (cp1252) cannot encode -- printing a table would raise UnicodeEncodeError rather
# than produce output. Markdown destined for a README is UTF-8 by definition, so the
# stream is reconfigured rather than the arrows removed.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

#: ``column -> (header, lower-is-better, decimals)``. Order here is column order out.
COLUMNS: dict[str, tuple[str, bool, int]] = {
    "ndcg@k": ("NDCG@10", False, 4),
    "recall@k": ("recall@10", False, 3),
    "category_coverage": ("cat. cov.", False, 3),
    "exposure_parity": ("parity", True, 4),
    "intra_list_sim": ("ILS", True, 4),
    "catalogue_coverage": ("cat. coverage", False, 3),
    "gini": ("Gini", True, 4),
    "aif": ("AI-F", True, -1),
    "ibo": ("IBO", False, 3),
    "seconds": ("secs", True, 2),
    "kwh": ("kWh", True, -1),
}

#: Methods in the order they should appear -- baselines first, then QUBO variants in
#: the order the write-up discusses them. Anything not listed is appended.
METHOD_ORDER = [
    "greedy_topk",
    "mmr",
    "quota_mmr",
    "qubo_sa",
    "qubo_tabu",
    "qubo_feasible",
]


def _fmt(value: float, decimals: int) -> str:
    if pd.isna(value):
        return "--"
    if decimals < 0:  # scientific, for the very small columns
        return f"{value:.1e}"
    return f"{value:.{decimals}f}"


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    return frame.sort_values(
        "method", key=lambda s: s.map(lambda m: rank.get(m, len(rank)))
    )


def markdown_table(frame: pd.DataFrame) -> str:
    """One row per method, best value per column in bold.

    'Best' follows the direction in :data:`COLUMNS`, and ties are all bolded -- on the
    synthetic benchmark several methods hit the arithmetic floor of exposure parity
    together, and silently bolding only the first would invent a winner.
    """
    frame = _order(frame)
    cols = [c for c in COLUMNS if c in frame.columns and frame[c].notna().any()]

    header = ["method"] + [
        f"{COLUMNS[c][0]}{' ↓' if COLUMNS[c][1] else ''}" for c in cols
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    best = {
        c: (frame[c].min() if COLUMNS[c][1] else frame[c].max())
        for c in cols
    }

    for _, row in frame.iterrows():
        cells = [str(row["method"])]
        for c in cols:
            text = _fmt(row[c], COLUMNS[c][2])
            # Compare on the rendered string, not the float: two values that both
            # print as 0.2667 should both be bold, whatever they differ by at 1e-9.
            if text != "--" and text == _fmt(best[c], COLUMNS[c][2]):
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def repeats_table(frame: pd.DataFrame, metrics: list[str]) -> str:
    """mean ± std across seeds, so the reader can see which gaps are real.

    A gap smaller than the two methods' combined spread is not a result, and the point
    of this table is to make that visible rather than leaving it to be assumed.
    """
    n_seeds = frame["seed"].nunique()
    stats = frame.groupby("method", sort=False)[metrics].agg(["mean", "std"])

    # Reorder on the index rather than via reset_index(): agg() produces MultiIndex
    # columns, so a reset would make the method column ('method', '') and indexing it
    # by name would return a Series instead of the name.
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    stats = stats.reindex(sorted(stats.index, key=lambda m: rank.get(m, len(rank))))

    header = ["method"] + [COLUMNS.get(m, (m, False, 4))[0] for m in metrics]
    lines = [
        f"Mean ± std over {n_seeds} seeds.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]

    for method, row in stats.iterrows():
        cells = [str(method)]
        for m in metrics:
            decimals = COLUMNS.get(m, (m, False, 4))[2]
            mean, std = row[(m, "mean")], row[(m, "std")]
            cells.append(f"{_fmt(mean, decimals)} ± {_fmt(std, decimals)}")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stem",
        nargs="+",
        default=["synthetic", "amazon_lb"],
        help="results/<stem>.csv basenames to render",
    )
    args = ap.parse_args()

    for stem in args.stem:
        headline = RESULTS / f"{stem}.csv"
        if headline.exists():
            print(f"\n### {stem}\n")
            print(markdown_table(pd.read_csv(headline)))

        repeats = RESULTS / f"{stem}_repeats.csv"
        if repeats.exists():
            frame = pd.read_csv(repeats)
            # Skip all-NaN columns: the synthetic benchmark has no held-out purchase,
            # so recall is undefined there and a column of "-- ± --" is worse than no
            # column at all.
            metrics = [
                m
                for m in ["ndcg@k", "recall@k", "exposure_parity", "gini", "seconds"]
                if m in frame.columns and frame[m].notna().any()
            ]
            print(f"\n### {stem} -- repeat study\n")
            print(repeats_table(frame, metrics))


if __name__ == "__main__":
    main()

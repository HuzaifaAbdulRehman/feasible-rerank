"""Benchmark driver: run every reranker over a population of users and tabulate.

Usage::

    python experiments/run_experiment.py --config configs/synthetic.yaml
    python experiments/run_experiment.py --config configs/amazon_lb.yaml

Writes a per-method results CSV to ``results/`` and prints a summary table.

Both benchmarks are reduced to the same :class:`Benchmark` container so the evaluation
loop is shared. The one thing that genuinely differs is the index space: on the
synthetic benchmark every user draws from the *same* 60 items, so a candidate index is
a catalogue index. On real data each user gets their own top-200, so candidate index 3
is a different product for every user. Catalogue-level metrics -- Gini, coverage -- are
therefore computed on ``instance.to_catalogue_ids(selection)`` throughout. Computing
them on raw candidate indices would report a flattering and completely meaningless
number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.loader import Benchmark, load_benchmark  # noqa: E402
from benchmarks.synthetic import make_users  # noqa: E402
from qubo_rerank.metrics.dpfr import (  # noqa: E402
    individual_item_fairness,
    item_better_off,
)
from qubo_rerank.metrics.energy import measure_energy  # noqa: E402
from qubo_rerank.metrics.fairness import (  # noqa: E402
    catalogue_coverage,
    category_coverage,
    exposure_parity,
    gini,
    intra_list_similarity,
    item_exposure,
)
from qubo_rerank.metrics.relevance import ndcg_at_k, recall_at_k  # noqa: E402
from qubo_rerank.solvers import (  # noqa: E402
    MMR,
    FeasibleAnnealing,
    GreedyTopK,
    QuotaMMR,
    SimulatedAnnealing,
    TabuSearch,
)


def build_solvers(cfg: dict) -> list:
    """Instantiate the rerankers named in the config."""
    s = cfg.get("solvers", {})
    solvers = []
    if s.get("greedy", True):
        solvers.append(GreedyTopK())
    if s.get("mmr", True):
        solvers.append(MMR(lam=s.get("mmr_lam", 0.5)))
    if s.get("quota_mmr", True):
        solvers.append(QuotaMMR(lam=s.get("mmr_lam", 0.5)))
    if s.get("qubo_sa", True):
        solvers.append(
            SimulatedAnnealing(
                num_reads=s.get("num_reads", 100), seed=cfg.get("seed", 0)
            )
        )
    if s.get("qubo_tabu", False):
        solvers.append(
            TabuSearch(num_reads=s.get("tabu_reads", 10), seed=cfg.get("seed", 0))
        )
    if s.get("qubo_feasible", False):
        solvers.append(
            FeasibleAnnealing(
                num_restarts=s.get("num_restarts", 8),
                num_sweeps=s.get("num_sweeps", 120),
                seed=cfg.get("seed", 0),
            )
        )
    return solvers


def build_benchmark(cfg: dict) -> Benchmark:
    """Construct the benchmark named by ``cfg['dataset']['source']``.

    Defaults to ``synthetic`` so existing configs keep working untouched.
    """
    data = cfg["data"]
    dataset = cfg.get("dataset", {})
    source = dataset.get("source", "synthetic")

    if source == "synthetic":
        n_items = data.get("n_items", 60)
        users = make_users(
            n_users=data.get("n_users", 20),
            n_items=n_items,
            n_groups=data.get("n_groups", 6),
            k=data.get("k", 10),
            dominance=data.get("dominance", 0.6),
            intra_group_similarity=data.get("intra_group_similarity", 0.7),
            lam=cfg.get("lam", 1.0),
            mu=cfg.get("mu", 0.0),
            seed=cfg.get("seed", 0),
        )
        return Benchmark(
            instances=users,
            # No held-out ground truth exists by construction; relevance *is* the
            # target. Recall is reported as blank rather than as a fabricated zero.
            relevant=[set() for _ in users],
            n_catalogue=n_items,
            catalogue_groups=users[0].groups,
            name="synthetic",
            stats={"sampled_users": len(users), "items": n_items},
        )

    if source == "amazon":
        return load_benchmark(
            path=ROOT / dataset.get("path", "data/amazon_lb/Luxury_Beauty.csv"),
            n_users=data.get("n_users", 200),
            n_candidates=data.get("n_items", 200),
            k=data.get("k", 10),
            n_groups=data.get("n_groups", 4),
            min_interactions=dataset.get("min_interactions", 5),
            topk_neighbours=dataset.get("topk_neighbours", 100),
            shrink=dataset.get("shrink", 100.0),
            binary=dataset.get("binary", True),
            lam=cfg.get("lam", 1.0),
            mu=cfg.get("mu", 0.0),
            seed=cfg.get("seed", 0),
        )

    raise ValueError(f"unknown dataset source {source!r}; expected 'synthetic' or 'amazon'")


def present(instance, selection: list[int]) -> list[int]:
    """Order a selected set for display: most relevant first.

    A QUBO selects a *set*; it says nothing about what order to show it in. The
    solvers return their sets index-sorted, and NDCG -- like every position-discounted
    measure -- is order-sensitive, so scoring the raw output charges the QUBO methods
    for an ordering they never claimed to produce. On the synthetic benchmark that
    penalty is worth up to 0.09 NDCG, which is larger than several of the differences
    being reported.

    Sorting by descending relevance is both the fair resolution and what any deployed
    system does with a chosen set. It is applied uniformly to every solver -- including
    the baselines, where it is worth a further ~0.03 to MMR -- so the comparison is
    strictly about *which items were chosen*.

    (On the Amazon benchmark this is a no-op: ``loader.py`` returns candidates already
    sorted by descending relevance, so index order is already relevance order.)
    """
    relevance = np.asarray(instance.relevance, dtype=float)
    return sorted((int(i) for i in selection), key=lambda i: -relevance[i])


def evaluate_solver(solver, bench: Benchmark, measure: bool = True) -> dict:
    """Run one reranker over every instance and return a row of mean metrics.

    Shared with :mod:`experiments.sweep` so the two scripts cannot drift into
    computing "the same" metric two different ways.
    """
    has_ground_truth = any(bench.relevant)

    # Solve everything first, score afterwards. Scoring is not cheap -- intra-list
    # similarity alone is O(k^2) in Python per user -- and folding it into the measured
    # window would charge evaluation cost to the solver. That barely moves the QUBO
    # methods, which spend hundreds of milliseconds per user, but it is most of the
    # measured time for the sub-second baselines, i.e. exactly the comparison the
    # "energy-aware" claim rests on.
    raw_selections: list[list[int]] = []
    with measure_energy(solver.name, enabled=measure) as m:
        for inst in bench.instances:
            raw_selections.append(solver.solve(inst).selection)

    reading = m.reading.as_dict() if m.reading else {}

    catalogue_selections: list[list[int]] = []
    catalogue_relevant: list[set[int]] = []
    ndcgs, covs, parities, ils, recalls = [], [], [], [], []

    for inst, relevant, raw in zip(bench.instances, bench.relevant, raw_selections):
        sel = present(inst, raw)

        ndcgs.append(ndcg_at_k(inst.relevance, sel))
        covs.append(category_coverage(inst.groups, sel))
        parities.append(exposure_parity(inst.groups, sel))
        ils.append(intra_list_similarity(inst.similarity, sel))
        if has_ground_truth:
            # Users whose held-out item never entered the candidate set score 0 rather
            # than being dropped -- dropping them would quietly inflate recall by
            # excluding exactly the hardest cases.
            recalls.append(recall_at_k(sel, relevant))

        catalogue_selections.append(inst.to_catalogue_ids(sel))
        catalogue_relevant.append(set(inst.to_catalogue_ids(sorted(relevant))))

    exposure = item_exposure(catalogue_selections, bench.n_catalogue)

    # Individual-item fairness needs relevance labels, so it is only defined where
    # there is a held-out ground truth -- i.e. on the real benchmark, not the
    # synthetic one whose "relevance" is the generator's own score.
    if has_ground_truth:
        iif, aif = individual_item_fairness(
            catalogue_selections, catalogue_relevant, bench.n_catalogue
        )
        ibo = item_better_off(catalogue_selections, catalogue_relevant, bench.n_catalogue)
    else:
        iif = aif = ibo = None

    return {
        "method": solver.name,
        "ndcg@k": float(np.mean(ndcgs)),
        "recall@k": float(np.mean(recalls)) if recalls else None,
        "category_coverage": float(np.mean(covs)),
        "exposure_parity": float(np.mean(parities)),
        "intra_list_sim": float(np.mean(ils)),
        "catalogue_coverage": catalogue_coverage(catalogue_selections, bench.n_catalogue),
        "gini": gini(exposure),
        "iif": iif,
        "aif": aif,
        "ibo": ibo,
        "seconds": reading.get("seconds"),
        "kwh": reading.get("kwh"),
        "co2_kg": reading.get("co2_kg"),
        "energy_measured": reading.get("energy_measured", False),
    }


def run(cfg: dict, bench: Benchmark) -> pd.DataFrame:
    measure = cfg.get("measure_energy", True)
    return pd.DataFrame(
        [evaluate_solver(solver, bench, measure=measure) for solver in build_solvers(cfg)]
    )


#: Metrics that are averages over a fixed population and so should be stable across
#: seeds. ``seconds`` and ``kwh`` are deliberately excluded from that expectation --
#: they are properties of the machine on the day, not of the method.
def run_repeats(cfg: dict, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run the whole benchmark under ``repeats`` different seeds.

    Every number this project reports is a mean over users, which says nothing about
    whether it would survive a different draw of users, a different synthetic instance,
    or a different solver seed. Repeating with ``seed = base + r`` varies all three at
    once -- the benchmark is rebuilt each time, so the user sample changes too -- and
    the spread across repeats is what tells you which gaps are real.

    Observed on this hardware: quality metrics are stable to ~0.01, while ``seconds``
    varies by up to 3x between identical runs. Timing conclusions need many more
    repeats than quality conclusions, or a quieter machine.

    Returns:
        ``(per_repeat, summary)`` -- the raw rows, and mean/std/min/max per method.
    """
    base_seed = cfg.get("seed", 0)
    frames = []

    for r in range(repeats):
        repeat_cfg = dict(cfg)
        repeat_cfg["seed"] = base_seed + r
        bench = build_benchmark(repeat_cfg)
        frame = run(repeat_cfg, bench)
        frame.insert(0, "repeat", r)
        frame.insert(1, "seed", repeat_cfg["seed"])
        frames.append(frame)
        print(f"  repeat {r + 1}/{repeats} (seed={repeat_cfg['seed']}) done")

    per_repeat = pd.concat(frames, ignore_index=True)

    numeric = [
        c
        for c in per_repeat.columns
        if c not in {"repeat", "seed", "method", "energy_measured"}
        and pd.api.types.is_numeric_dtype(per_repeat[c])
    ]
    summary = per_repeat.groupby("method", sort=False)[numeric].agg(
        ["mean", "std", "min", "max"]
    )
    return per_repeat, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="re-run under this many seeds and report mean/std per method; "
        "the benchmark is rebuilt each time, so the user sample varies too",
    )
    ap.add_argument(
        "--n-users",
        type=int,
        default=None,
        help="override the config's user count; useful for repeat studies, where a "
        "full-size run times the number of repeats is not worth the wall-clock",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.n_users:
        cfg["data"]["n_users"] = args.n_users

    if args.repeats > 1:
        print(f"running {args.repeats} repeats of {args.config}\n")
        per_repeat, summary = run_repeats(cfg, args.repeats)

        raw_out = args.out or (ROOT / "results" / f"{args.config.stem}_repeats.csv")
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        per_repeat.to_csv(raw_out, index=False)

        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 100)
        headline = ["ndcg@k", "exposure_parity", "gini", "seconds"]
        available = [c for c in headline if c in summary.columns.get_level_values(0)]
        print("\nmean +/- std across repeats:\n")
        for metric in available:
            block = summary[metric]
            print(f"  {metric}")
            for method, row in block.iterrows():
                spread = "" if pd.isna(row["std"]) else f" +/- {row['std']:.4g}"
                print(
                    f"    {method:<16} {row['mean']:.4g}{spread}"
                    f"   [{row['min']:.4g}, {row['max']:.4g}]"
                )
            print()

        print(f"wrote {raw_out}")
        print(
            "\nquality metrics should be stable across seeds; 'seconds' is a property "
            "of the machine on the day and is expected to move much more."
        )
        return

    bench = build_benchmark(cfg)
    df = run(cfg, bench)

    out = args.out or (ROOT / "results" / f"{args.config.stem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print(f"\nconfig:  {args.config}")
    print(f"dataset: {bench.name}")
    for key, value in bench.stats.items():
        print(f"  {key:20s} {value}")
    if any(bench.relevant):
        print(f"  {'candidate_hit_rate':20s} {bench.candidate_hit_rate:.4f}   <- ceiling on recall@k")
    print(
        f"\nlam={cfg.get('lam')}  mu={cfg.get('mu')}  users={len(bench.instances)}  "
        f"n={bench.instances[0].n}  k={bench.instances[0].k}\n"
    )

    columns = ["method", "ndcg@k"]
    if any(bench.relevant):
        columns.append("recall@k")
    columns += [
        "category_coverage",
        "exposure_parity",
        "intra_list_sim",
        "catalogue_coverage",
        "gini",
    ]
    if any(bench.relevant):
        columns += ["iif", "aif", "ibo"]
    columns += ["seconds", "kwh"]
    print(df[columns].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print(f"\nwrote {out}")
    print(
        "\nreading the table: higher ndcg, recall, category_coverage, "
        "catalogue_coverage and ibo are better; lower exposure_parity, "
        "intra_list_sim, gini, iif and aif are better."
    )
    if any(bench.relevant):
        print(
            "iif/aif/ibo are the individual-item fairness measures of Rampisela et al.; "
            "unlike exposure_parity they are not what the QUBO optimises."
        )


if __name__ == "__main__":
    main()

"""Sweep lambda and mu to trace the relevance / diversity / fairness trade-off.

This is the Pareto analysis in miniature. It answers the question the single-config
run cannot: *at what weights does the QUBO actually beat the classical baselines, and
what does each unit of fairness cost in NDCG?*

The benchmark is loaded **once** and the weights are then set on the instances in
place. On real data that saves re-running the ItemKNN fit for every grid point, but
more importantly it guarantees every point on the curve is scored against byte-identical
candidate sets. Regenerating the data per point leaves the comparison at the mercy of
the sampling seed, which is a good way to publish a trade-off curve that is really a
plot of RNG noise.

Usage::

    python experiments/sweep.py --config configs/amazon_lb.yaml \
        --lam 0.0 1.0 4.0 16.0 --mu 0.0 1.0 4.0 --solver qubo_feasible
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_experiment import (  # noqa: E402
    build_benchmark,
    build_solvers,
    evaluate_solver,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--lam", nargs="+", type=float, default=[0.0, 1.0, 4.0, 16.0])
    ap.add_argument("--mu", nargs="+", type=float, default=[0.0, 1.0, 4.0])
    ap.add_argument(
        "--solver",
        nargs="+",
        default=None,
        help="solver names to sweep; defaults to every solver enabled in the config",
    )
    ap.add_argument(
        "--n-users",
        type=int,
        default=None,
        help="override the config's user count; a grid point costs one full pass, so "
        "the sweep is usually run on fewer users than the headline benchmark",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.n_users:
        cfg["data"]["n_users"] = args.n_users
    bench = build_benchmark(cfg)

    # Default to the QUBO solvers only. The classical baselines ignore lam and mu, so
    # sweeping them re-evaluates an identical list at every grid point -- pure cost,
    # and it fills the sweep CSV with rows that look like a curve but are one point
    # repeated. They are run once, separately, in the baseline block below.
    solvers = build_solvers(cfg)
    if args.solver:
        wanted = set(args.solver)
        missing = wanted - {s.name for s in solvers}
        if missing:
            raise SystemExit(f"solver(s) not enabled in {args.config}: {sorted(missing)}")
        solvers = [s for s in solvers if s.name in wanted]
    else:
        solvers = [s for s in solvers if s.name.startswith("qubo")]
        if not solvers:
            raise SystemExit(f"no qubo solvers enabled in {args.config}")

    print(f"dataset: {bench.name}   users={len(bench.instances)}   "
          f"n={bench.instances[0].n}   k={bench.instances[0].k}")
    print(f"solvers: {', '.join(s.name for s in solvers)}")
    print(f"grid:    {len(args.lam)} lam x {len(args.mu)} mu = "
          f"{len(args.lam) * len(args.mu) * len(solvers)} evaluations\n")

    out = args.out or (ROOT / "results" / f"{args.config.stem}_sweep.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Evaluate the classical baselines on *this* benchmark, not the headline one.
    #
    # The baselines ignore lam and mu entirely, so they are a fixed reference point and
    # only need running once. But they must be run over the same user sample as the
    # sweep: Gini and catalogue coverage are catalogue-level aggregates, and a 40-user
    # run spreads exposure over less of the catalogue than a 200-user run. Plotting the
    # sweep against baselines from a different sample size silently compares numbers
    # that are not on the same scale -- and Gini is the headline fairness axis.
    baseline_solvers = [s for s in build_solvers(cfg) if not s.name.startswith("qubo")]
    if baseline_solvers:
        for inst in bench.instances:
            inst.lam = cfg.get("lam", 1.0)
            inst.mu = cfg.get("mu", 0.0)
        baseline_rows = [
            evaluate_solver(s, bench, measure=False) for s in baseline_solvers
        ]
        baseline_out = out.with_name(f"{out.stem}_baselines.csv")
        pd.DataFrame(baseline_rows).to_csv(baseline_out, index=False)
        print(f"baselines on the same {len(bench.instances)} users -> {baseline_out}")
        for row in baseline_rows:
            print(
                f"{'':<22}{row['method']:<14} ndcg={row['ndcg@k']:.4f} "
                f"parity={row['exposure_parity']:.4f} gini={row['gini']:.4f}"
            )
        print()

    rows = []
    for lam in args.lam:
        for mu in args.mu:
            # Weights live on the instance, so the grid point is applied in place --
            # no regeneration, no reload, identical candidate sets throughout.
            for inst in bench.instances:
                inst.lam = lam
                inst.mu = mu

            for solver in solvers:
                row = {"lam": lam, "mu": mu, **evaluate_solver(solver, bench, measure=False)}
                rows.append(row)
                print(
                    f"lam={lam:<6} mu={mu:<6} {row['method']:<14} "
                    f"ndcg={row['ndcg@k']:.4f} cov={row['category_coverage']:.4f} "
                    f"parity={row['exposure_parity']:.4f} ils={row['intra_list_sim']:.4f} "
                    f"gini={row['gini']:.4f}"
                )

                # Rewritten after every grid point rather than once at the end. A full
                # sweep is tens of minutes; losing all of it to an interrupt on the last
                # point is an avoidable way to waste an afternoon, and a partial curve
                # is still a usable curve.
                pd.DataFrame(rows).to_csv(out, index=False)

    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()

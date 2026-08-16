"""Do the formulation's design choices actually matter? Ablate them and find out.

Three decisions were made early in this project and then never revisited, each of which a
reviewer could reasonably question:

**1. Equal group targets.** The fairness term defaults to ``k / |C|`` per group -- every
group gets the same number of slots regardless of how much of the catalogue it holds.
``exposure_targets_proportional`` has existed since the first commit and has never been
compared against it. The two encode genuinely different notions of fairness: equal targets
give a 20-item group the same exposure as a 700-item one, which is either the point or
absurd depending on whether groups are protected classes or catalogue partitions.

**2. k = 10, always.** Every result in the repo reranks to a list of ten. Ten is
conventional but arbitrary, and the cardinality penalty scales with ``k``, so the barrier
could plausibly behave differently at 5 or 20.

**3. n = 100-200, always.** The candidate-set size is the QUBO's problem size. It is
already known that the barrier worsens with n (see the optimality experiment at
n=14..30); this checks whether the *downstream* conclusions move too.

None of these is expected to overturn anything. The point of an ablation is that
"expected" is not evidence, and a formulation whose design choices were never varied is
one nobody has checked.

Usage::

    python experiments/ablation.py --config configs/amazon_lb.yaml --what targets k n
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

from experiments.run_experiment import (
    build_benchmark,
    build_solvers,
    evaluate_solver,
    present,
)
from qubo_rerank.formulations.fairness import exposure_targets_proportional
from qubo_rerank.metrics.fairness import exposure_parity


def group_sizes(bench) -> str:
    counts = np.bincount(np.asarray(bench.catalogue_groups).ravel())
    return " / ".join(str(int(c)) for c in counts)


def ablate_targets(cfg: dict, solver_names: list[str]) -> pd.DataFrame:
    """Equal vs proportional group targets, scored under *both* definitions.

    Scoring under both is not optional here. ``exposure_parity`` defaults to equal
    targets, so a solver given proportional targets is otherwise measured against an
    objective it was never asked to optimise -- which makes it look worse by
    construction rather than by result. The first version of this ablation did exactly
    that and appeared to show a clean accuracy/fairness trade-off that was pure artefact.

    On these benchmarks the two definitions diverge more than the catalogue suggests.
    Popularity tiers are equal-sized *in the catalogue* (342/341/341/341), but a user's
    candidate set is skewed toward the short head, and proportional targets are computed
    from the candidate set. So "proportional" here means "match the shortlist's own
    skew", which is a materially weaker fairness requirement.
    """
    rows = []
    for label in ("equal", "proportional"):
        bench = build_benchmark(cfg)
        per_instance_targets = []
        for inst in bench.instances:
            inst.lam = cfg["lam"]
            inst.mu = cfg["mu"]
            # None means "equal shares" inside build_fairness; proportional is derived
            # from each candidate set's own group composition, so it is per-instance.
            targets = (
                None if label == "equal"
                else exposure_targets_proportional(inst.groups, inst.k)
            )
            inst.targets = targets
            per_instance_targets.append(targets)

        for solver in build_solvers(cfg):
            if solver.name not in solver_names:
                continue
            scored = evaluate_solver(solver, bench, measure=False)

            # Re-score parity under the other definition, so both are on the table.
            selections = [
                present(inst, solver.solve(inst).selection) for inst in bench.instances
            ]
            under_equal = float(np.mean([
                exposure_parity(inst.groups, sel)
                for inst, sel in zip(bench.instances, selections, strict=True)
            ]))
            under_proportional = float(np.mean([
                exposure_parity(
                    inst.groups, sel, exposure_targets_proportional(inst.groups, inst.k)
                )
                for inst, sel in zip(bench.instances, selections, strict=True)
            ]))

            rows.append({
                "ablation": "targets", "setting": label, **scored,
                "parity_vs_equal": under_equal,
                "parity_vs_proportional": under_proportional,
            })
            print(f"  optimised={label:<13} {rows[-1]['method']:<14} "
                  f"ndcg={scored['ndcg@k']:.4f}   "
                  f"parity[equal]={under_equal:.4f}  "
                  f"parity[prop]={under_proportional:.4f}")
    return pd.DataFrame(rows)


def ablate_scalar(cfg: dict, solver_names: list[str], key: str, values: list[int]) -> pd.DataFrame:
    """Vary ``data[key]`` -- k or n_items -- rebuilding the benchmark each time."""
    rows = []
    for value in values:
        local = {**cfg, "data": {**cfg["data"], key: value}}
        bench = build_benchmark(local)
        for inst in bench.instances:
            inst.lam, inst.mu = local["lam"], local["mu"]

        for solver in build_solvers(local):
            if solver.name not in solver_names:
                continue
            rows.append({
                "ablation": key, "setting": str(value),
                **evaluate_solver(solver, bench, measure=False),
            })
            print(f"  {key}={value:<6} {rows[-1]['method']:<14} "
                  f"ndcg={rows[-1]['ndcg@k']:.4f}  parity={rows[-1]['exposure_parity']:.4f}"
                  f"  ils={rows[-1]['intra_list_sim']:.4f}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--what", nargs="+", default=["targets", "k", "n"],
                    choices=["targets", "k", "n"])
    ap.add_argument("--solver", nargs="+",
                    default=["greedy_topk", "quota_mmr", "qubo_tabu", "qubo_feasible"])
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--n-users", type=int, default=60)
    ap.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--n-values", nargs="+", type=int, default=[50, 100, 200])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cfg["data"]["n_users"] = args.n_users
    cfg["lam"], cfg["mu"] = args.lam, args.mu

    # Run at the operating point the protocol selects, not the headline config: an
    # ablation of the fairness formulation at mu=0 would vary something switched off.
    print(f"ablations on {args.config.stem} at lam={args.lam}, mu={args.mu}, "
          f"{args.n_users} users")
    print(f"group sizes: {group_sizes(build_benchmark(cfg))}\n")

    frames = []
    if "targets" in args.what:
        print("targets: equal vs proportional")
        frames.append(ablate_targets(cfg, args.solver))
        print()
    if "k" in args.what:
        print("list size k")
        frames.append(ablate_scalar(cfg, args.solver, "k", args.k_values))
        print()
    if "n" in args.what:
        print("candidate-set size n")
        frames.append(ablate_scalar(cfg, args.solver, "n_items", args.n_values))
        print()

    frame = pd.concat(frames, ignore_index=True)
    out = args.out or (ROOT / "results" / f"{args.config.stem}_ablation.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    # State whether each choice mattered, rather than leaving it to be eyeballed.
    print("Does the choice change the ranking of methods?\n")
    for ablation, block in frame.groupby("ablation", sort=False):
        orders = {
            setting: tuple(rows.sort_values("ndcg@k", ascending=False)["method"])
            for setting, rows in block.groupby("setting", sort=False)
        }
        distinct = set(orders.values())
        verdict = "unchanged" if len(distinct) == 1 else f"CHANGES ({len(distinct)} orderings)"
        print(f"  {ablation:<10} {verdict}")
        if len(distinct) > 1:
            for setting, order in orders.items():
                print(f"      {setting:<14} {' > '.join(order)}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

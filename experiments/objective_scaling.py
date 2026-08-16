"""Does the QUBO's advantage over apportionment grow with the diversity weight?

This experiment exists because the claim it tests was, for a short time, the only
unreproducible number in the repository. After the apportionment baseline refuted the
original feasibility headline, what replaced it was a claim about *where* the QUBO still
earns its compute: the pairwise diversity term. ``BalancedQuota`` fills each group
greedily, which is optimal for a separable objective and not for one carrying
``lam * sum_ij s_ij x_i x_j``. So the prediction is specific and falsifiable -- the two
should tie at ``lam = 0`` and diverge as ``lam`` grows.

The README quoted "+5.5% at lam=4" from an ad-hoc run with no script and no saved CSV,
which is exactly the defect an audit had just flagged elsewhere (a published table with
no driver). This module is that driver.

**Why the reported quantity is not a percentage.** The obvious summary -- one solver's
energy divided by another's -- is a ratio of two *signed* numbers whose zero is arbitrary.
The QUBO objective can be positive or negative depending on lam and the instance, so that
ratio is not a percentage of anything: the same defect made an exact-MIP comparison report
146.6% at n=20. The margin here is therefore anchored the way
:mod:`experiments.optimality` anchors its recovery figure, on a *difference*::

    advantage = (apportionment_energy - qubo_energy) / (random_energy - qubo_energy)

Random feasible selection is 0 and the QUBO is 1, so the number answers "how much of the
distance from no-optimisation-at-all to the QUBO's solution does apportionment fail to
cover". Adding a constant to the objective leaves it unchanged.

Both the absolute energy gap and the anchored fraction are written out, so a reader who
prefers the raw difference is not forced through the normalisation.

Usage::

    python experiments/objective_scaling.py --lam 0 1 2 4 8 --repeats 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.loader import load_benchmark
from qubo_rerank.formulations.builder import build_problem
from qubo_rerank.solvers import BalancedQuota, FeasibleAnnealing, QuotaMMR, TabuSearch


def problem_for(instance):
    """Compose the instance's BQM once.

    Built per instance rather than per evaluation: the random-feasible anchor scores
    dozens of subsets against the same model, and rebuilding a 200-variable dense BQM
    each time dominated the runtime of this experiment for no benefit.
    """
    return build_problem(
        relevance=instance.relevance,
        similarity=instance.similarity,
        k=instance.k,
        groups=instance.groups,
        lam=instance.lam,
        mu=instance.mu,
        targets=instance.targets,
    )


def energy_of(rp, n: int, selection) -> float:
    """BQM energy of a selection, offset included.

    The offset matters: the cardinality penalty expands to include ``P * k^2``, which
    lands there. Every solver in this repository reports energy on this scale, and one
    that did not was the subject of its own bug fix.
    """
    chosen = set(int(i) for i in selection)
    return float(rp.bqm.energy({i: (1 if i in chosen else 0) for i in range(n)}))


def random_feasible_energy(rp, instance, trials: int, seed: int) -> float:
    """Mean energy of a uniformly random k-subset: the "no optimisation" anchor."""
    rng = np.random.default_rng(seed)
    return float(np.mean([
        energy_of(rp, instance.n,
                  sorted(rng.choice(instance.n, instance.k, replace=False)))
        for _ in range(trials)
    ]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path,
                    default=ROOT / "data/amazon_lb/Luxury_Beauty.csv")
    ap.add_argument("--lam", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--n-users", type=int, default=40)
    ap.add_argument("--n-candidates", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    print(f"objective scaling: lam in {args.lam}, mu={args.mu}, "
          f"{args.repeats} seeds x {args.n_users} users\n")

    rows = []
    for lam in args.lam:
        for seed in range(args.repeats):
            bench = load_benchmark(
                args.data, n_users=args.n_users, n_candidates=args.n_candidates,
                k=args.k, seed=seed, lam=lam, mu=args.mu,
            )
            methods = {
                "quota_mmr": QuotaMMR(lam=0.5),
                "balanced_quota": BalancedQuota(lam=0.0),
                "balanced_quota_div": BalancedQuota(lam=0.5),
                "qubo_tabu": TabuSearch(num_reads=10, seed=seed),
                "qubo_feasible": FeasibleAnnealing(num_restarts=8, num_sweeps=120,
                                                   seed=seed),
            }
            problems = [problem_for(i) for i in bench.instances]

            for name, solver in methods.items():
                energies = [energy_of(rp, inst.n, solver.solve(inst).selection)
                            for rp, inst in zip(problems, bench.instances, strict=True)]
                rows.append({"lam": lam, "mu": args.mu, "seed": seed,
                             "method": name, "energy": float(np.mean(energies))})

            rnd = float(np.mean([
                random_feasible_energy(rp, inst, trials=50, seed=seed)
                for rp, inst in zip(problems, bench.instances, strict=True)
            ]))
            rows.append({"lam": lam, "mu": args.mu, "seed": seed,
                         "method": "random_feasible", "energy": rnd})
            print(f"  lam={lam:<5g} seed {seed}  done")

    frame = pd.DataFrame(rows)
    out = args.out or (ROOT / "results" / "objective_scaling.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    # ---- the table the claim is made from -----------------------------------
    print("\nObjective energy by lam (mean over seeds; lower is better)\n")
    pivot = frame.pivot_table(index="method", columns="lam", values="energy")
    order = ["random_feasible", "quota_mmr", "balanced_quota", "balanced_quota_div",
             "qubo_tabu", "qubo_feasible"]
    print(pivot.reindex([m for m in order if m in pivot.index])
          .to_string(float_format=lambda v: f"{v:+.5f}"))

    print("\nHow much of the QUBO's improvement over random does the best classical")
    print("method fail to capture?  0 = it matches the QUBO, 1 = it is no better than")
    print("random. Offset-invariant by construction.\n")
    print(f"  {'lam':>6}{'best classical':>18}{'gap to qubo_tabu':>20}{'shortfall':>12}")
    for lam in args.lam:
        block = frame[frame.lam == lam].groupby("method")["energy"].mean()
        qubo = block["qubo_tabu"]
        rnd = block["random_feasible"]
        classical = {m: block[m] for m in
                     ["quota_mmr", "balanced_quota", "balanced_quota_div"] if m in block}
        best_name = min(classical, key=classical.get)
        best = classical[best_name]
        span = rnd - qubo
        shortfall = (best - qubo) / span if abs(span) > 1e-12 else float("nan")
        print(f"  {lam:>6g}{best_name:>18}{best - qubo:>+20.6f}{shortfall:>12.3f}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

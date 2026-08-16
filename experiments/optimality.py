"""How far from the true optimum is each solver? Brute-force small instances and find out.

Every other comparison in this project is **relative**: one solver's energy against
another's, or against a greedy baseline. That supports statements like "``neal`` is worse
than tabu" but not "``neal`` recovers 40% of what is available", and the second is a much
stronger thing to be able to say — it is the difference between ranking methods and
measuring them.

At small ``n`` the optimum is not a matter of opinion. Choosing ``k`` items from ``n``
admits ``C(n, k)`` subsets: 53,130 at n=25, k=5. Enumerating them gives the exact minimum
of the QUBO, and every solver can then be scored as a **gap to optimal** rather than as a
position in a league table.

Two things this is expected to establish, both of which the relative comparisons cannot:

* **Whether the constraint-preserving solvers are actually good**, or merely better than a
  broken one. A solver that reaches the exact optimum on every small instance has earned
  the benefit of the doubt at larger ``n``; one that plateaus at 80% has not.
* **How much of the objective ``neal`` recovers.** The barrier argument predicts it should
  land near what a *random feasible set* achieves, since that is effectively what it
  returns. Random selection is included as a reference line to test exactly that.

The instances are small by necessity, and small instances are the *easy* case for every
method here — the penalty barrier grows with ``n``. So a solver failing at n=20 is
strong evidence; a solver succeeding is weak evidence, and is reported as such.

Usage::

    python experiments/optimality.py --n-items 22 --k 5 --repeats 20
"""

from __future__ import annotations

import argparse
import itertools
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.sensitivity import barrier_instance, energy_of
from qubo_rerank.formulations.builder import build_problem
from qubo_rerank.solvers import (
    MMR,
    FeasibleAnnealing,
    GreedyTopK,
    QuotaMMR,
    SimulatedAnnealing,
    SimulatedBifurcation,
    TabuSearch,
)


def exact_optimum(instance) -> tuple[float, list[int]]:
    """Minimum QUBO energy over all C(n, k) feasible selections, by enumeration.

    Enumerates only *feasible* subsets, which is the honest comparison: every solver here
    is being asked for exactly k items, so the reference should be the best k-subset
    rather than the unconstrained minimum of the penalised objective. Those differ --
    the unconstrained minimum of a penalty-encoded problem is not necessarily feasible.
    """
    rp = build_problem(
        relevance=instance.relevance,
        similarity=instance.similarity,
        k=instance.k,
        groups=instance.groups,
        lam=instance.lam,
        mu=instance.mu,
        targets=instance.targets,
    )
    linear = np.array([rp.bqm.linear[i] for i in range(instance.n)])
    quadratic = np.zeros((instance.n, instance.n))
    for (u, v), bias in rp.bqm.quadratic.items():
        quadratic[int(u), int(v)] = bias
        quadratic[int(v), int(u)] = bias

    # The offset is not optional. A dimod BQM carries a constant term, and the
    # cardinality penalty P(sum(x) - k)^2 expands to include P*k^2, which lands there.
    # Omitting it shifts every enumerated energy by ~2.78 on these instances while
    # leaving the *ranking* of subsets untouched -- so the optimum looks plausible and
    # every gap-to-optimal is nonsense. Caught only because the gaps came out uniformly
    # positive against a negative optimum.
    offset = float(rp.bqm.offset)

    best_energy, best_selection = np.inf, []
    for subset in itertools.combinations(range(instance.n), instance.k):
        index = list(subset)
        energy = linear[index].sum() + 0.5 * quadratic[np.ix_(index, index)].sum() + offset
        if energy < best_energy:
            best_energy, best_selection = energy, index
    return float(best_energy), best_selection


def random_feasible_energy(instance, trials: int, seed: int) -> float:
    """Mean energy of a uniformly random k-subset -- what "no optimisation" looks like.

    This is the line the barrier argument predicts ``neal`` should sit near: if the
    penalty dominates the objective, the sampler returns a feasible set chosen for
    reasons unrelated to the objective, which is a random feasible set in all but name.
    """
    rng = np.random.default_rng(seed)
    return float(np.mean([
        energy_of(instance, sorted(rng.choice(instance.n, instance.k, replace=False)))
        for _ in range(trials)
    ]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # A *list*. The write-up publishes a ladder over n = 14/18/22/26/30 to support
    # "the barrier grows with n", and that table was previously unreproducible from this
    # repository: the flag took a single int, the CSV had no n column, and no driver
    # script existed. Sweeping here, and recording n per row, makes the mechanism claim
    # regenerable by the documented command.
    ap.add_argument("--n-items", type=int, nargs="+", default=[22])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    for n in args.n_items:
        print(f"enumerating C({n}, {args.k}) = {comb(n, args.k):,} subsets per instance, "
              f"{args.repeats} instances")
    print()

    rows = []
    for n_items, repeat in itertools.product(args.n_items, range(args.repeats)):
        instance = barrier_instance(n_items, args.k, args.lam, seed=repeat)
        instance.mu = args.mu

        optimum, _ = exact_optimum(instance)
        random_energy = random_feasible_energy(instance, trials=200, seed=repeat)

        solvers = [
            GreedyTopK(),
            MMR(lam=0.5),
            QuotaMMR(lam=0.5),
            SimulatedAnnealing(num_reads=100, seed=repeat),
            TabuSearch(num_restarts=50, timeout=None, seed=repeat),
            FeasibleAnnealing(num_restarts=8, num_sweeps=120, seed=repeat),
            SimulatedBifurcation(num_restarts=8, num_steps=600, seed=repeat),
        ]
        for solver in solvers:
            result = solver.solve(instance)
            feasible = len(result.selection) == instance.k
            energy = energy_of(instance, result.selection) if feasible else np.nan
            rows.append({
                "n": n_items,
                "repeat": repeat, "method": solver.name,
                "energy": energy, "optimum": optimum, "random": random_energy,
                "feasible": feasible,
                "optimal": bool(feasible and abs(energy - optimum) < 1e-9),
            })

        rows.append({"n": n_items, "repeat": repeat, "method": "random_feasible",
                     "energy": random_energy, "optimum": optimum,
                     "random": random_energy, "feasible": True, "optimal": False})

        print(f"  n={n_items:>3} instance {repeat:>2}  optimum {optimum:+.6f}   "
              f"random {random_energy:+.6f}")

    frame = pd.DataFrame(rows)

    # Fraction of the available improvement each solver captures. Anchored on the random
    # feasible set rather than on zero: "how much better than not optimising at all",
    # which is the quantity the barrier argument is about.
    span = frame["random"] - frame["optimum"]
    frame["recovered"] = np.where(
        span > 0, (frame["random"] - frame["energy"]) / span, np.nan
    )

    out = args.out or (ROOT / "results" / "optimality.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    # `recovered` is NaN wherever a solver returned an infeasible set, so its mean is
    # taken over the feasible runs only, while `feasible` is a fraction of *all* runs.
    # Printed side by side without saying so, "18.9% recovered | 50% feasible" reads as
    # though both were out of twelve -- when the first is the average over the six runs
    # that worked. The denominator is now shown next to the number that depends on it.
    order = ["greedy_topk", "mmr", "quota_mmr", "balanced_quota", "qubo_sa", "qubo_sb",
             "qubo_tabu", "qubo_feasible", "random_feasible"]

    # The ladder the write-up quotes, printed directly rather than assembled by hand from
    # several single-size runs -- which is how it became unreproducible in the first place.
    if len(args.n_items) > 1:
        print("\n% of the available improvement recovered, by catalogue size")
        header = "".join(f"{n:>8}" for n in args.n_items)
        print(f"{'method':<18}{header}")
        for name in order:
            block = frame[frame.method == name]
            if not len(block):
                continue
            cells = "".join(
                f"{100 * block[block.n == n]['recovered'].mean():>7.1f}%"
                for n in args.n_items
            )
            print(f"{name:<18}{cells}")

    print(f"\npooled over n = {args.n_items}")
    print(f"{'method':<18}{'gap to optimal':>16}{'% recovered':>14}{'(of runs)':>11}"
          f"{'exactly optimal':>17}{'feasible':>10}")
    for name in order:
        block = frame[frame.method == name]
        if not len(block):
            continue
        gap = (block["energy"] - block["optimum"]).mean()
        scored = int(block["recovered"].notna().sum())
        print(f"{name:<18}{gap:>+16.6f}{100 * block['recovered'].mean():>13.1f}%"
              f"{f'{scored}/{len(block)}':>11}"
              f"{100 * block['optimal'].mean():>16.0f}%"
              f"{100 * block['feasible'].mean():>9.0f}%")

    print("\nSmall instances are the easy case -- the penalty barrier grows with n.")
    print("A solver failing here is strong evidence; one succeeding is weak evidence.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

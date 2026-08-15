"""Does the penalty barrier survive more compute? Sweep every solver's budget and see.

The central finding of this project is that penalty-encoded cardinality defeats
simulated annealing. The obvious objection is the cheap one: **you did not run it long
enough.** Every result elsewhere in the repo uses one budget per solver, so on the
evidence presented that objection is not answerable.

This script answers it. Each solver is run across two orders of magnitude of compute and
scored on **QUBO energy** -- its own objective, not a downstream metric -- against two
fixed reference lines:

* **greedy MMR**, a hand-written heuristic that does no search at all. If an annealer
  given 100x the compute still cannot beat a greedy pass, the problem is not the budget.
* **the corrected solvers**, which reach a better energy at their default settings.

Energy is the right measure precisely because it removes the usual escape route. NDCG
depends on the evaluation protocol and could in principle be argued about; the QUBO
energy is the number the sampler is *trying* to minimise, computed from the same BQM the
sampler was handed. A solver that loses on it has lost on its own terms.

The expected shape, if the barrier explanation is right: `neal`'s energy is flat in the
budget. Not slowly improving -- flat, or drifting slightly worse, because the search is
resolving a landscape whose relevant structure sits entirely below the penalty scale.

Usage::

    python experiments/sensitivity.py --n-items 200 --k 10 --repeats 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.synthetic import make_instance
from qubo_rerank.formulations.builder import build_problem
from qubo_rerank.solvers import (
    MMR,
    FeasibleAnnealing,
    GreedyTopK,
    SimulatedAnnealing,
    TabuSearch,
)


def barrier_instance(n_items: int, k: int, lam: float, seed: int):
    """A candidate set with the sparse, spiky similarity structure of real data.

    ``rng.random(...) ** 6`` concentrates most similarity near zero with a light tail,
    which is what the Amazon matrices look like (mean ~0.005) and is far less forgiving
    than the synthetic generator's dense block structure. The barrier is a property of
    the *encoding*, so it should appear here too -- and this instance is the one the
    penalty-barrier tests in tests/test_solvers.py already use.
    """
    rng = np.random.default_rng(seed)
    instance = make_instance(n_items=n_items, n_groups=4, k=k, seed=seed)
    similarity = (rng.random((n_items, n_items)) ** 6) * 0.5
    instance.similarity = (similarity + similarity.T) / 2
    np.fill_diagonal(instance.similarity, 0.0)
    instance.lam = lam
    return instance


def energy_of(problem, selection: list[int]) -> float:
    """QUBO energy of a selection, from the same BQM the solvers were handed."""
    rp = build_problem(
        relevance=problem.relevance,
        similarity=problem.similarity,
        k=problem.k,
        groups=problem.groups,
        lam=problem.lam,
        mu=problem.mu,
        targets=problem.targets,
    )
    chosen = set(selection)
    return float(rp.bqm.energy({i: (1 if i in chosen else 0) for i in range(problem.n)}))


def budgets(solver_name: str) -> list[dict]:
    """Compute ladders, chosen so each solver spans ~2 orders of magnitude of work.

    The ladders are not equal in wall-clock -- they cannot be, since the solvers do
    different things per unit of work. That is fine and in fact the point: the question
    is whether *any* budget lets neal catch up, not whether the budgets match.
    """
    if solver_name == "qubo_sa":
        return [
            {"num_reads": r, "num_sweeps": s}
            for r, s in [(10, 100), (100, 100), (100, 1000), (100, 10_000),
                         (1000, 1000), (100, 100_000), (1000, 10_000)]
        ]
    if solver_name == "qubo_tabu":
        return [{"num_restarts": r} for r in (5, 20, 50, 200, 1000)]
    if solver_name == "qubo_feasible":
        return [
            {"num_restarts": r, "num_sweeps": s}
            for r, s in [(2, 30), (4, 60), (8, 120), (16, 240), (32, 480)]
        ]
    raise ValueError(solver_name)


def make_solver(name: str, params: dict, seed: int):
    if name == "qubo_sa":
        return SimulatedAnnealing(
            num_reads=params["num_reads"], num_sweeps=params["num_sweeps"], seed=seed
        )
    if name == "qubo_tabu":
        # timeout=None so the ladder measures *work*, not a wall-clock budget -- the
        # whole experiment would otherwise be measuring the clock.
        return TabuSearch(num_restarts=params["num_restarts"], timeout=None, seed=seed)
    if name == "qubo_feasible":
        return FeasibleAnnealing(
            num_restarts=params["num_restarts"],
            num_sweeps=params["num_sweeps"],
            seed=seed,
        )
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--solver", nargs="+",
                    default=["qubo_sa", "qubo_tabu", "qubo_feasible"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for repeat in range(args.repeats):
        instance = barrier_instance(args.n_items, args.k, args.lam, seed=repeat)

        # Reference lines. Neither searches; both are computed once per instance.
        for name, solver in (("greedy_topk", GreedyTopK()), ("mmr", MMR(lam=0.5))):
            result = solver.solve(instance)
            rows.append({
                "repeat": repeat, "method": name, "budget": "-", "work": 0,
                "energy": energy_of(instance, result.selection),
                "seconds": float("nan"),
                "cardinality_ok": len(result.selection) == instance.k,
            })

        for name in args.solver:
            for params in budgets(name):
                solver = make_solver(name, params, seed=repeat)
                start = time.perf_counter()
                result = solver.solve(instance)
                elapsed = time.perf_counter() - start

                # A single scalar for the x-axis. Crude by design: it only has to order
                # the ladder, and reporting seconds alongside keeps it honest.
                work = int(np.prod([v for v in params.values()]))
                rows.append({
                    "repeat": repeat, "method": name,
                    "budget": ", ".join(f"{k}={v}" for k, v in params.items()),
                    "work": work,
                    "energy": energy_of(instance, result.selection),
                    "seconds": elapsed,
                    "cardinality_ok": len(result.selection) == instance.k,
                })
                print(f"  seed {repeat}  {name:<14} {rows[-1]['budget']:<28} "
                      f"energy={rows[-1]['energy']:+.5f}  {elapsed:6.1f}s")

    frame = pd.DataFrame(rows)
    out = args.out or (ROOT / "results" / "sensitivity.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    print("\nMean QUBO energy by budget (lower is better):\n")
    reference = frame[frame.method.isin(["greedy_topk", "mmr"])].groupby("method")["energy"].mean()
    for name, value in reference.items():
        print(f"  {name:<14} {value:+.5f}   (no search)")
    print()
    for name in args.solver:
        block = frame[frame.method == name]
        for budget, rows_ in block.groupby("budget", sort=False):
            print(f"  {name:<14} {budget:<28} {rows_['energy'].mean():+.5f}"
                  f"   {rows_['seconds'].mean():7.2f}s")
        print()

    # The question the script exists to answer, stated as a verdict rather than left
    # for the reader to compute.
    sa = frame[frame.method == "qubo_sa"]
    if len(sa):
        best_sa = sa.groupby("budget")["energy"].mean().min()
        best_ref = reference.min()
        print(f"neal's best energy at ANY budget : {best_sa:+.5f}")
        print(f"best no-search heuristic         : {best_ref:+.5f}")
        print(
            "\n-> "
            + (
                "neal never beats a heuristic that does no search at all, at any budget "
                "tried. More compute is not the missing ingredient."
                if best_sa > best_ref
                else "neal overtakes the heuristic at some budget; the barrier claim "
                     "needs qualifying by compute."
            )
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

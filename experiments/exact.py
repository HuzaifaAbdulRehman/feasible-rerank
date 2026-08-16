"""The obvious counter-hypothesis: why not just use an exact solver?

Selecting ``k`` items to minimise a linear-plus-quadratic objective under a cardinality
constraint is a **mixed-integer quadratic program**. That is a solved problem class with
mature off-the-shelf solvers, and it is the first question any reviewer of a QUBO paper
asks. Until it is answered, "our heuristic beats other heuristics" is not the same claim
as "this problem needs a heuristic" -- and only the second justifies the framing.

This script answers it, and the answer is allowed to be unflattering: if an exact solver
returns the optimum at ``n=200`` in milliseconds, the honest conclusion is that QUBO
machinery buys nothing at this scale.

**Formulation.** The quadratic term is linearised with the standard McCormick
substitution ``y_ij = x_i * x_j``. Because every diversity coefficient is non-negative
and the problem is a *minimisation*, only the lower bounds are needed --

    y_ij >= x_i + x_j - 1,    y_ij >= 0

-- since the solver will push each ``y`` down on its own. Adding ``y <= x_i`` and
``y <= x_j`` would be correct but doubles the constraint count for nothing.

**And note what the constraint becomes.** Cardinality here is ``sum(x) == k``: a genuine
linear constraint, enforced exactly by the solver. There is no penalty weight, so there
is no barrier. That makes this the constraint-aware approach taken to its limit, and a
second, independent test of the explanation in the README: if the penalty encoding is
what breaks single-flip annealing, a solver that never encodes it should not struggle.

Two outcomes, both worth having:

* **Exact solving is tractable at reranking sizes** -- then the honest recommendation is
  to use a MIP solver, and the QUBO contribution is the *formulation*, not the solving.
* **It blows up somewhere below n=200** -- then heuristics are necessary rather than
  merely convenient, and the size at which that happens is a real result.

Usage::

    python experiments/exact.py --n-items 20 40 80 150 200 --k 10 --repeats 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.sensitivity import barrier_instance
from qubo_rerank.solvers import FeasibleAnnealing, GreedyTopK, SimulatedAnnealing, TabuSearch


def objective_energy(instance, selection) -> float:
    """Relevance + diversity only, with no cardinality penalty and no normalisation.

    This is the quantity every method is actually competing on. Because all of them
    return exactly ``k`` items, the cardinality penalty contributes the same constant to
    each and cannot change the ranking -- so including it only obscures the comparison.

    Dropping it matters for more than tidiness. The penalty couples *every* pair, so a
    MILP built from the penalty-encoded BQM carries C(n,2) auxiliary variables for a
    constraint it enforces natively in one line. Scoring on the objective lets the exact
    solver be given the problem it should be given, rather than a handicapped
    transcription of someone else's encoding.
    """
    index = list(selection)
    relevance = np.asarray(instance.relevance, dtype=float)
    similarity = np.asarray(instance.similarity, dtype=float)
    block = similarity[np.ix_(index, index)]
    return float(-relevance[index].sum() + instance.lam * 0.5 * block.sum())


def solve_exact(instance, time_limit: float = 300.0, solver_name: str = "HiGHS"):
    """Exact MILP for the relevance + diversity objective with hard cardinality.

    Returns ``(selection, seconds, status)``. ``status`` is PuLP's, so an unsolved model
    is reported as such rather than quietly returning whatever the solver had reached --
    a time-limited run that returns its incumbent looks exactly like a solved one.

    The fairness term is omitted here on purpose: ``(sum_c x - t)^2`` needs a further
    linearisation of its own, and the question this script exists to answer -- can an
    exact solver handle the reranking problem at all -- is settled by the relevance and
    diversity terms, which are what make it quadratic.
    """
    n, k = instance.n, instance.k
    relevance = np.asarray(instance.relevance, dtype=float)
    similarity = np.asarray(instance.similarity, dtype=float)
    lam = instance.lam

    problem = pulp.LpProblem("rerank", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]

    # Only pairs that actually contribute. On real data the similarity matrix is ~99%
    # zeros, so this is the difference between a model that builds and one that does
    # not -- and it is available only because the cardinality penalty, which couples
    # every pair, is not in this objective.
    pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if similarity[i, j] != 0.0
    ]
    y = {(i, j): pulp.LpVariable(f"y{i}_{j}", lowBound=0, upBound=1) for i, j in pairs}

    problem += (
        -pulp.lpSum(relevance[i] * x[i] for i in range(n))
        + lam * pulp.lpSum(similarity[i, j] * y[(i, j)] for i, j in pairs)
    )
    problem += pulp.lpSum(x) == k          # the constraint, exactly, with no penalty

    # McCormick lower bound only, valid because similarity is non-negative and this is a
    # minimisation: the solver pushes each y down unaided.
    if (similarity < 0).any():
        raise ValueError("negative similarity; the one-sided relaxation is invalid")
    for i, j in pairs:
        problem += y[(i, j)] >= x[i] + x[j] - 1

    solver = (
        pulp.HiGHS(msg=False, timeLimit=time_limit)
        if solver_name == "HiGHS"
        else pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    )

    start = time.perf_counter()
    problem.solve(solver)
    elapsed = time.perf_counter() - start

    status = pulp.LpStatus[problem.status]
    selection = sorted(i for i in range(n) if x[i].value() and x[i].value() > 0.5)
    return selection, elapsed, status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-items", nargs="+", type=int, default=[20, 40, 80, 150, 200])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--solver", default="HiGHS", choices=["HiGHS", "CBC"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or (ROOT / "results" / "exact.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in args.n_items:
        for repeat in range(args.repeats):
            instance = barrier_instance(n, args.k, args.lam, seed=repeat)
            instance.mu = 0.0                      # matches solve_exact's objective

            selection, seconds, status = solve_exact(
                instance, time_limit=args.time_limit, solver_name=args.solver
            )
            feasible = len(selection) == args.k
            exact_energy = objective_energy(instance, selection) if feasible else np.nan
            rows.append({
                "n": n, "repeat": repeat, "method": f"exact_{args.solver}",
                "energy": exact_energy, "seconds": seconds,
                "status": status, "feasible": feasible,
            })
            print(f"  n={n:<4} seed {repeat}  exact  {status:<10} "
                  f"energy={exact_energy:+.6f}  {seconds:7.2f}s")

            # A random feasible k-subset: the "no optimisation at all" reference.
            #
            # Without it the only normalised statement available is heuristic/optimum,
            # and that is a quotient of two *signed* energies whose zero is arbitrary --
            # add a constant to the objective and the percentage moves. It is not merely
            # imprecise: at n=20 the optimum here is positive, so the same formula yields
            # 146.6%, which is why that column was never quoted. Anchoring on random
            # instead gives (random - energy) / (random - optimum), a ratio of
            # *differences*, which is invariant to the offset. optimality.py already
            # scores this way; this makes the exact comparison consistent with it.
            rng = np.random.default_rng(repeat)
            random_energy = float(np.mean([
                objective_energy(
                    instance,
                    sorted(rng.choice(instance.n, args.k, replace=False).tolist()),
                )
                for _ in range(200)
            ]))
            rows.append({
                "n": n, "repeat": repeat, "method": "random_feasible",
                "energy": random_energy, "seconds": 0.0,
                "status": "reference", "feasible": True,
            })

            for solver in (
                GreedyTopK(),
                SimulatedAnnealing(num_reads=100, seed=repeat),
                TabuSearch(num_restarts=50, timeout=None, seed=repeat),
                FeasibleAnnealing(num_restarts=8, num_sweeps=120, seed=repeat),
            ):
                start = time.perf_counter()
                result = solver.solve(instance)
                rows.append({
                    "n": n, "repeat": repeat, "method": solver.name,
                    "energy": objective_energy(instance, result.selection),
                    "seconds": time.perf_counter() - start,
                    "status": "-", "feasible": len(result.selection) == args.k,
                })

            # Flushed after every instance. Exact solving at n=200 can run for an hour,
            # and losing the whole ladder because the last rung timed out is an
            # avoidable way to waste an afternoon -- a partial ladder is still a result.
            pd.DataFrame(rows).to_csv(out, index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)

    print("\nmean energy (lower is better) and seconds, by problem size\n")
    print(f"{'n':>5}  {'method':<16}{'energy':>12}{'seconds':>10}{'solved':>9}")
    for n in args.n_items:
        block = frame[frame.n == n]
        for method, rows_ in block.groupby("method", sort=False):
            solved = (
                f"{100 * (rows_['status'] == 'Optimal').mean():.0f}%"
                if method.startswith("exact") else "-"
            )
            print(f"{n:>5}  {method:<16}{rows_['energy'].mean():>+12.6f}"
                  f"{rows_['seconds'].mean():>10.2f}{solved:>9}")
        print()

    # The verdict, stated rather than left to the reader.
    exact = frame[frame.method.str.startswith("exact")]
    proven = exact[exact.status == "Optimal"]
    if len(proven):
        largest = proven["n"].max()
        at_largest = proven[proven.n == largest]
        print(f"Exact optimum proven up to n={largest} "
              f"(mean {at_largest['seconds'].mean():.1f}s).")
        heuristic = frame[(frame.n == largest) & (frame.method == "qubo_feasible")]
        if len(heuristic):
            gap = heuristic["energy"].mean() - at_largest["energy"].mean()
            print(f"qubo_feasible gap to proven optimum at n={largest}: {gap:+.8f} "
                  f"in {heuristic['seconds'].mean():.2f}s")

        # Offset-invariant scoring, per size. Reported instead of "% of the optimum",
        # which is a ratio of signed energies and therefore not a percentage of anything.
        print("\n% of the available improvement recovered (random = 0%, optimum = 100%)")
        sizes = sorted(proven["n"].unique())
        print(f"{'method':<18}" + "".join(f"{n:>9}" for n in sizes))
        for name in ["greedy_topk", "qubo_sa", "qubo_tabu", "qubo_feasible"]:
            cells = ""
            for n in sizes:
                opt = proven[proven.n == n]["energy"].mean()
                rnd = frame[(frame.n == n) & (frame.method == "random_feasible")]
                got = frame[(frame.n == n) & (frame.method == name)]
                if not len(rnd) or not len(got) or rnd["energy"].mean() == opt:
                    cells += f"{'--':>9}"
                    continue
                span = rnd["energy"].mean() - opt
                cells += f"{100 * (rnd['energy'].mean() - got['energy'].mean()) / span:>8.1f}%"
            print(f"{name:<18}{cells}")
    unsolved = exact[exact.status != "Optimal"]
    if len(unsolved):
        print(f"Not proven optimal within {args.time_limit:.0f}s at n="
              f"{sorted(unsolved['n'].unique())}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""Simulated annealing that never leaves the feasible set.

**The problem this solves.** Encoding "exactly k" as a penalty ``P(sum x - k)^2`` and
handing the result to a single-spin-flip sampler creates a trap. Two feasible k-sets
are never adjacent: getting from one to another means removing an item and adding
another, and the intermediate state has ``k-1`` or ``k+1`` items and therefore costs
``P``. So the sampler faces a choice with no good side:

* cold enough to resolve the objective (differences here are ~1e-3 of the penalty
  scale) and it cannot cross the barrier at all -- it freezes in whichever feasible
  set it first stumbled into;
* hot enough to cross the barrier and the objective is thermal noise.

Measured on Amazon Luxury Beauty (n=200, k=10), ``neal`` returns lists that are
*always* the right length and *nearly arbitrary* among lists of that length: its mean
QUBO energy is worse than a greedy MMR baseline evaluated on the same BQM. Raising
``num_sweeps`` tenfold makes it slightly worse, not better, which is the giveaway --
more search does not help when the search is looking at the wrong scale. Lowering the
penalty strength does not help either, because the barrier height *is* the strength,
and a strength low enough to cross is too low to enforce the constraint.

**The fix.** Anneal over k-subsets rather than over binary strings. The move is a swap
-- one item out, one item in -- so every state visited is feasible by construction,
the penalty term is a constant that never enters a single acceptance decision, and the
barrier does not exist. This is the standard remedy for hard-constrained QUBOs and is
why D-Wave ships constrained (CQM) solvers rather than expecting users to tune penalty
weights.

The honest caveat: this is no longer a *generic* QUBO sampler, and it is not what a
physical annealer does. It is a classical heuristic exploiting the constraint
structure. Reported alongside :class:`SimulatedAnnealing` precisely so the cost of
insisting on a generic sampler is visible rather than hidden.
"""

from __future__ import annotations

from math import exp
from typing import Any

import numpy as np

from ..formulations.builder import build_problem
from .base import SolveResult, timed


def _dense(bqm, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(h, J)`` as dense arrays, ``J`` symmetric with a zero diagonal."""
    h = np.zeros(n, dtype=float)
    J = np.zeros((n, n), dtype=float)
    for v, bias in bqm.linear.items():
        h[int(v)] = bias
    for (u, v), bias in bqm.quadratic.items():
        J[int(u), int(v)] = bias
        J[int(v), int(u)] = bias
    return h, J


class FeasibleAnnealing:
    """Metropolis annealing over the k-subsets of the candidate set.

    Args:
        num_restarts: independent anneals; the best is returned. Cheap insurance
            against a bad random start, and the spread across restarts is a usable
            signal of how rugged the landscape is.
        num_sweeps: temperature steps per restart. Each sweep proposes ``n`` swaps.
        seed: RNG seed.
        beta_range: ``(hot, cold)`` inverse temperatures. Derived from the actual
            spread of swap energies when omitted -- hand-set values do not transfer
            between the synthetic benchmark and real data, whose objective scales
            differ by two orders of magnitude.
        warm_start: begin one restart from the greedy top-k list instead of a random
            subset. Never *worse* than random on average, and it makes the "does the
            QUBO improve on greedy" question a strict comparison.
    """

    name = "qubo_feasible"

    def __init__(
        self,
        num_restarts: int = 8,
        num_sweeps: int = 120,
        seed: int | None = None,
        beta_range: tuple[float, float] | None = None,
        warm_start: bool = True,
    ) -> None:
        self.num_restarts = num_restarts
        self.num_sweeps = num_sweeps
        self.seed = seed
        self.beta_range = beta_range
        self.warm_start = warm_start

    # -- energy bookkeeping ------------------------------------------------------
    #
    # The search runs on objective + fairness only. On the feasible set the
    # cardinality term contributes an identical constant to every state, so it cannot
    # affect a single acceptance decision; excluding it also avoids differencing two
    # numbers ~700x larger than the quantity of interest. The energy *reported* is
    # still that of the full composed BQM, so it stays comparable with every other
    # solver in the benchmark.

    def _estimate_beta_range(
        self, h: np.ndarray, J: np.ndarray, n: int, k: int, rng: np.random.Generator
    ) -> tuple[float, float]:
        """Pick hot and cold temperatures from the observed scale of swap deltas."""
        if k >= n:
            # No swap exists, so there is no delta scale to estimate. The anneal will
            # short-circuit anyway; any finite range keeps geomspace happy.
            return 1.0, 100.0

        probes = []
        for _ in range(64):
            chosen = rng.choice(n, size=k, replace=False)
            x = np.zeros(n)
            x[chosen] = 1.0
            field = J @ x
            out = chosen[rng.integers(k)]
            outside = np.flatnonzero(x == 0.0)
            inn = outside[rng.integers(outside.size)]
            probes.append(abs(h[inn] - h[out] + (field[inn] - J[inn, out]) - field[out]))

        scale = float(np.mean(probes))
        if scale <= 0 or not np.isfinite(scale):
            return 1.0, 100.0
        # Hot: a typical uphill swap is accepted about half the time. Cold: it is
        # accepted about once in a million. Two and a half decades of cooling.
        return float(np.log(2.0) / scale), float(np.log(1e6) / scale)

    def _anneal(
        self,
        h: np.ndarray,
        J: np.ndarray,
        n: int,
        k: int,
        start: np.ndarray,
        betas: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, int, float]:
        chosen = np.array(sorted(start), dtype=int)
        inside = np.zeros(n, dtype=bool)
        inside[chosen] = True
        outside = np.flatnonzero(~inside)

        x = inside.astype(float)
        field = J @ x
        energy = float(h[chosen].sum() + 0.5 * x @ field)

        best_selection = chosen.copy()
        best_energy = energy
        accepted = 0

        n_out = outside.size  # a swap preserves both set sizes, so this is invariant
        if n_out == 0:
            # k == n: there is exactly one feasible set and nothing to search.
            return best_selection, energy, 0, 0.0

        for beta in betas:
            # Drawing the whole sweep's randomness in three vector calls rather than
            # 3n scalar ones is worth roughly 3x here; the loop body is otherwise
            # dominated by RNG overhead rather than by any actual arithmetic.
            a_idx = rng.integers(k, size=n)
            b_idx = rng.integers(n_out, size=n)
            uniform = rng.random(n)

            for step in range(n):
                a = a_idx[step]
                b = b_idx[step]
                out = chosen[a]
                inn = outside[b]

                delta = h[inn] - h[out] + (field[inn] - J[inn, out]) - field[out]

                if delta <= 0.0 or uniform[step] < exp(-beta * delta):
                    chosen[a], outside[b] = inn, out
                    field += J[:, inn] - J[:, out]
                    energy += float(delta)
                    accepted += 1

                    # Annealing accepts uphill moves, so the final state is not
                    # necessarily the best one seen. Track the best explicitly --
                    # forgetting to do this is why a warm start can end up worse
                    # than the list it started from.
                    if energy < best_energy:
                        best_energy = energy
                        best_selection = chosen.copy()

        # The running energy is accumulated from ~1e5 swap deltas, so it drifts. That is
        # harmless inside a restart -- every accept/reject compares deltas, not totals --
        # but restarts are ranked against each other by total, and a drifted total could
        # crown the wrong one. Recompute the winner exactly; it costs one O(n^2) pass per
        # restart. The drift is returned so it can be asserted on rather than assumed.
        x_best = np.zeros(n)
        x_best[best_selection] = 1.0
        exact = float(h[best_selection].sum() + 0.5 * x_best @ (J @ x_best))

        return best_selection, exact, accepted, abs(exact - best_energy)

    def solve(self, problem) -> SolveResult:  # noqa: ANN001
        stats: dict[str, Any] = {
            "num_restarts": self.num_restarts,
            "num_sweeps": self.num_sweeps,
        }
        n, k = problem.n, problem.k

        with timed(stats, "build_time"):
            rp = build_problem(
                relevance=problem.relevance,
                similarity=problem.similarity,
                k=k,
                groups=problem.groups,
                lam=problem.lam,
                mu=problem.mu,
                targets=problem.targets,
            )
            search = rp.parts["objective"].copy()
            search.update(rp.parts["fairness"])
            h, J = _dense(search, n)

        rng = np.random.default_rng(self.seed)
        hot, cold = self.beta_range or self._estimate_beta_range(h, J, n, k, rng)
        betas = np.geomspace(hot, cold, self.num_sweeps)

        starts = []
        if self.warm_start:
            starts.append(np.argsort(-np.asarray(problem.relevance))[:k])
        while len(starts) < self.num_restarts:
            starts.append(rng.choice(n, size=k, replace=False))

        best_selection, best_energy, total_accepted, worst_drift = None, np.inf, 0, 0.0
        with timed(stats, "solve_time"):
            for start in starts:
                selection, energy, accepted, drift = self._anneal(
                    h, J, n, k, start, betas, rng
                )
                total_accepted += accepted
                worst_drift = max(worst_drift, drift)
                if energy < best_energy:
                    best_selection, best_energy = selection, energy

        selection = sorted(int(i) for i in best_selection)

        stats["search_energy"] = float(best_energy)
        # Accumulated-vs-recomputed discrepancy across restarts. Should sit at float
        # noise; anything larger means the swap bookkeeping is wrong, not imprecise.
        stats["energy_drift"] = float(worst_drift)
        stats["energy"] = float(rp.bqm.energy({i: (1 if i in set(selection) else 0) for i in range(n)}))
        stats["accepted_moves"] = total_accepted
        stats["beta_range"] = (hot, cold)
        stats["n_selected"] = len(selection)
        # True by construction here -- kept so the stats dict matches the other
        # solvers' and so a regression in the swap bookkeeping would surface.
        stats["cardinality_ok"] = len(selection) == k
        stats["energy_breakdown"] = rp.energy_breakdown(selection)

        return SolveResult(selection=selection, stats=stats)

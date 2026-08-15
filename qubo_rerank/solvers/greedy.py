"""Classical baselines.

These are what the QUBO has to beat. Two of them:

* **Greedy top-k** -- what essentially every deployed recommender does: sort by score,
  take the first k. Maximal relevance, no diversity or fairness consideration at all.
* **MMR** (Maximal Marginal Relevance, Carbonell & Goldstein 1998) -- the standard
  diversity-aware reranker, and the honest comparison point. Claiming QUBO beats
  greedy top-k on diversity is trivial; beating a tuned MMR is the real test.

An optional group quota variant of MMR gives a classical fairness baseline too, so the
QUBO is not the only method being asked to satisfy exposure constraints.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import SolveResult, timed


class GreedyTopK:
    """Sort by relevance, take the first k."""

    name = "greedy_topk"

    def solve(self, problem) -> SolveResult:  # noqa: ANN001 - duck-typed
        stats: dict = {}
        with timed(stats, "solve_time"):
            order = np.argsort(-np.asarray(problem.relevance, dtype=float))
            selection = [int(i) for i in order[: problem.k]]
        stats["n_selected"] = len(selection)
        return SolveResult(selection=selection, stats=stats)


class MMR:
    """Maximal Marginal Relevance.

    At each step pick the item maximising::

        (1 - lam) * relevance_i  -  lam * max_{j in selected} similarity_ij

    ``lam=0`` degenerates to greedy top-k; ``lam=1`` ignores relevance entirely.
    """

    name = "mmr"

    def __init__(self, lam: float = 0.5) -> None:
        self.lam = lam

    def solve(self, problem) -> SolveResult:  # noqa: ANN001
        stats: dict = {"lam": self.lam}
        rel = _unit_scale(np.asarray(problem.relevance, dtype=float))
        sim = _unit_scale(np.asarray(problem.similarity, dtype=float))
        n, k = len(rel), problem.k

        with timed(stats, "solve_time"):
            selected: list[int] = []
            remaining = set(range(n))
            while len(selected) < k and remaining:
                best, best_score = None, -np.inf
                for i in remaining:
                    penalty = max((sim[i, j] for j in selected), default=0.0)
                    score = (1.0 - self.lam) * rel[i] - self.lam * penalty
                    if score > best_score:
                        best, best_score = i, score
                selected.append(int(best))
                remaining.discard(int(best))

        stats["n_selected"] = len(selected)
        return SolveResult(selection=selected, stats=stats)


class QuotaMMR:
    """MMR with a hard per-group cap -- a classical fairness baseline.

    Each group may occupy at most ``ceil(k / n_groups)`` slots. This is the obvious
    thing a practitioner would do without any optimisation machinery, so it is the
    fair comparison for the QUBO fairness term. If the QUBO cannot beat this, the
    honest conclusion is that the QUBO is not worth its compute -- and that is a
    perfectly publishable result.
    """

    name = "quota_mmr"

    def __init__(self, lam: float = 0.5) -> None:
        self.lam = lam

    def solve(self, problem) -> SolveResult:  # noqa: ANN001
        stats: dict = {"lam": self.lam}
        if problem.groups is None:
            raise ValueError("QuotaMMR requires group labels")

        rel = _unit_scale(np.asarray(problem.relevance, dtype=float))
        sim = _unit_scale(np.asarray(problem.similarity, dtype=float))
        groups = np.asarray(problem.groups).ravel()
        n, k = len(rel), problem.k

        n_groups = len(np.unique(groups))
        cap = int(np.ceil(k / n_groups))
        used: dict[int, int] = {}

        with timed(stats, "solve_time"):
            selected: list[int] = []
            remaining = set(range(n))
            while len(selected) < k and remaining:
                best, best_score = None, -np.inf
                for i in remaining:
                    g = int(groups[i])
                    if used.get(g, 0) >= cap:
                        continue
                    penalty = max((sim[i, j] for j in selected), default=0.0)
                    score = (1.0 - self.lam) * rel[i] - self.lam * penalty
                    if score > best_score:
                        best, best_score = i, score
                if best is None:
                    # Every remaining item is in a saturated group. Relax the cap
                    # rather than return a short list -- a short list would silently
                    # break the comparison against fixed-k methods.
                    cap += 1
                    continue
                selected.append(int(best))
                used[int(groups[best])] = used.get(int(groups[best]), 0) + 1
                remaining.discard(int(best))

        stats["n_selected"] = len(selected)
        stats["cap"] = cap
        return SolveResult(selection=selected, stats=stats)


def _unit_scale(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo <= 0.0:
        return np.zeros_like(a, dtype=float)
    return (a - lo) / (hi - lo)

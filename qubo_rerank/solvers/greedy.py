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

import numpy as np

from .base import SolveResult, timed


class GreedyTopK:
    """Sort by relevance, take the first k."""

    name = "greedy_topk"

    def solve(self, problem) -> SolveResult:
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

    def solve(self, problem) -> SolveResult:
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

    def solve(self, problem) -> SolveResult:
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


class BalancedQuota:
    """Largest-remainder apportionment, then greedy MMR inside each group.

    **This is the baseline that refutes the naive reading of the feasibility claim, and
    it exists because an independent audit wrote it in fifteen lines.** :class:`QuotaMMR`
    enforces ``ceil(k / |C|)`` as an *upper* bound only, with no lower bound and no
    remainder rule, so it can finish at 3/3/3/1 over four groups and has no way back --
    exposure parity 0.30 against an arithmetic floor of 0.20. That is a defect of that
    particular heuristic, not a property of classical reranking, and reporting it as the
    latter would have been the most serious error in this project.

    The fix is textbook apportionment. Give group ``c`` a base quota of
    ``floor(target_c)``, then hand the ``k - sum(floor)`` leftover slots to the groups
    with the largest fractional parts. That allocation provably minimises
    ``sum_c |quota_c - target_c|``, which *is* the exposure-parity numerator, so this
    baseline attains the parity floor by construction whenever each group has enough
    candidates -- deterministically, on every user, in milliseconds.

    Two details keep it honest rather than strawmanned in the other direction:

    * **Ties are broken by relevance.** With equal targets every fractional part is the
      same, so the choice of which groups get an extra slot is free on parity and is
      spent on relevance instead.
    * **It reads the target vector.** Unlike :class:`QuotaMMR` it handles proportional
      targets, so the expressiveness argument has to be made against *this*, not against
      the weaker heuristic.

    Where the QUBO still earns its compute is the pairwise term: this baseline fills each
    group greedily, which is optimal for a separable objective and not for one containing
    ``lam * sum s_ij x_i x_j``. Set ``lam > 0`` and the two diverge.
    """

    name = "balanced_quota"

    def __init__(self, lam: float = 0.5) -> None:
        self.lam = lam

    def solve(self, problem) -> SolveResult:
        stats: dict = {"lam": self.lam}
        if problem.groups is None:
            raise ValueError("BalancedQuota requires group labels")

        rel = _unit_scale(np.asarray(problem.relevance, dtype=float))
        sim = _unit_scale(np.asarray(problem.similarity, dtype=float))
        groups = np.asarray(problem.groups).ravel()
        k = problem.k
        uniq = np.unique(groups)

        if problem.targets is not None:
            targets = np.array([float(problem.targets[int(g)]) for g in uniq])
        else:
            targets = np.full(len(uniq), k / len(uniq), dtype=float)

        with timed(stats, "solve_time"):
            # A group can never supply more items than the candidate set holds of it.
            # Ignoring that is what made a first version of this baseline miss the floor
            # on the users whose retrieval happened to skew -- it allocated slots that
            # could not be filled, then spilled them arbitrarily.
            capacity = np.array([int(np.sum(groups == g)) for g in uniq])

            base = np.minimum(np.floor(targets).astype(int), capacity)
            quota = base.copy()

            # Hand out the remaining slots one at a time, each to the group currently
            # furthest below its target and still able to take one. Greedy on the
            # largest shortfall is exactly largest-remainder when capacity does not
            # bind, and degrades gracefully into the best attainable split when it does.
            best_in_group = np.array([
                rel[groups == g].max() if np.any(groups == g) else -np.inf for g in uniq
            ])
            for _ in range(int(k - quota.sum())):
                room = quota < capacity
                if not room.any():
                    break
                shortfall = np.where(room, targets - quota, -np.inf)
                # Ties on shortfall are free in parity terms, so spend them on relevance.
                j = int(np.lexsort((-best_in_group, -shortfall))[0])
                quota[j] += 1

            selected: list[int] = []
            for j, g in enumerate(uniq):
                pool = [int(i) for i in np.flatnonzero(groups == g)]
                for _ in range(int(quota[j])):
                    if not pool:
                        break
                    scores = [
                        (1.0 - self.lam) * rel[i]
                        - self.lam * max((sim[i, s] for s in selected), default=0.0)
                        for i in pool
                    ]
                    pick = pool[int(np.argmax(scores))]
                    selected.append(pick)
                    pool.remove(pick)

            # A group short of candidates leaves the list under-length; top it up from
            # whatever is left rather than return fewer than k, which would silently
            # break every comparison against fixed-k methods.
            if len(selected) < k:
                taken = set(selected)
                for i in np.argsort(-rel, kind="stable"):
                    if int(i) not in taken:
                        selected.append(int(i))
                        if len(selected) == k:
                            break

            # NDCG here is position-weighted, so the list is returned best-first. Any
            # real reranker presents it that way, and leaving it in group order would
            # penalise this baseline for a presentation choice rather than a decision.
            selected = sorted(selected[:k], key=lambda i: -rel[i])

        stats["n_selected"] = len(selected)
        stats["quota"] = {int(g): int(q) for g, q in zip(uniq, quota, strict=True)}
        return SolveResult(selection=selected, stats=stats)


def _unit_scale(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo <= 0.0:
        return np.zeros_like(a, dtype=float)
    return (a - lo) / (hi - lo)

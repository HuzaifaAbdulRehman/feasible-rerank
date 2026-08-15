"""Fairness and diversity metrics.

Two families here:

* **Group exposure** -- how evenly the k slots are spread over categories or sellers.
  This is what the QUBO fairness term directly optimises, so reporting it alone would
  be self-serving.
* **Distributional** -- Gini and coverage over the item catalogue across all users.
  These are the standard measures in the recsys fairness literature and are *not*
  what the penalty optimises, which makes them the honest check.

For publication-grade individual-item fairness, the measures in Rampisela et al.
(WWW 2025, MIT-licensed, see reference/DPFR-fairness-eval) should be reported
alongside these; they are more carefully motivated than anything hand-rolled here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def group_counts(groups: np.ndarray, selection: Sequence[int]) -> dict[int, int]:
    """Number of selected items per group."""
    groups = np.asarray(groups).ravel()
    counts: dict[int, int] = {int(g): 0 for g in np.unique(groups)}
    for i in selection:
        counts[int(groups[int(i)])] += 1
    return counts


def category_coverage(groups: np.ndarray, selection: Sequence[int]) -> float:
    """Fraction of available groups represented at least once in the list."""
    groups = np.asarray(groups).ravel()
    n_groups = len(np.unique(groups))
    if n_groups == 0:
        return 0.0
    present = {int(groups[int(i)]) for i in selection}
    return len(present) / n_groups


def exposure_parity(
    groups: np.ndarray, selection: Sequence[int], targets: dict[int, float] | None = None
) -> float:
    """Mean absolute deviation from target exposure, normalised by list length.

    ``0.0`` is perfect parity. Larger is worse. Normalising by ``k`` keeps the value
    comparable across list sizes.
    """
    groups = np.asarray(groups).ravel()
    counts = group_counts(groups, selection)
    k = len(list(selection))
    if k == 0:
        return 0.0

    if targets is None:
        targets = {g: k / len(counts) for g in counts}

    deviation = sum(abs(counts[g] - targets.get(g, 0.0)) for g in counts)
    return float(deviation / k)


def gini(values: Sequence[float]) -> float:
    """Gini coefficient of an exposure distribution.

    ``0`` is perfectly equal, approaching ``1`` is maximally concentrated. Applied to
    per-item exposure counts aggregated over all users, this is the standard measure
    of popularity bias in a recommender.
    """
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(np.abs(x))
    n = x.size
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))


def catalogue_coverage(selections: Sequence[Sequence[int]], n_items: int) -> float:
    """Fraction of the catalogue that appears in at least one recommendation list."""
    if n_items == 0:
        return 0.0
    seen: set[int] = set()
    for sel in selections:
        seen.update(int(i) for i in sel)
    return len(seen) / n_items


def item_exposure(selections: Sequence[Sequence[int]], n_items: int) -> np.ndarray:
    """Per-item exposure counts across all users -- the input to :func:`gini`."""
    counts = np.zeros(n_items, dtype=float)
    for sel in selections:
        for i in sel:
            counts[int(i)] += 1.0
    return counts


def intra_list_similarity(similarity: np.ndarray, selection: Sequence[int]) -> float:
    """Mean pairwise similarity within the list -- lower means more diverse.

    This is the quantity the diversity term of the QUBO minimises, reported directly
    so the effect of ``lam`` is visible rather than inferred.
    """
    sel = [int(i) for i in selection]
    if len(sel) < 2:
        return 0.0
    sim = np.asarray(similarity, dtype=float)
    total = 0.0
    pairs = 0
    for a in range(len(sel)):
        for b in range(a + 1, len(sel)):
            total += float(sim[sel[a], sel[b]])
            pairs += 1
    return total / pairs if pairs else 0.0

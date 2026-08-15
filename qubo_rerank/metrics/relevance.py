"""Relevance metrics.

Deliberately small and dependency-free so the core library can be tested without a
RecBole install. RecBole's own evaluator is used for the full benchmark runs; these
exist so unit tests can assert on known values.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with log2 position discount."""
    gains = np.asarray(gains, dtype=float)
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevance: np.ndarray, selection: Sequence[int], k: int | None = None) -> float:
    """NDCG of a selection, using relevance scores as graded gains.

    The ideal ranking is the top-``k`` items by relevance from the same candidate set,
    so a greedy top-k reranker scores exactly 1.0 by construction. That is the point:
    it makes the relevance cost of adding diversity or fairness directly readable.
    """
    relevance = np.asarray(relevance, dtype=float)
    selection = list(selection)
    k = len(selection) if k is None else k

    actual = dcg(relevance[selection][:k])
    ideal = dcg(np.sort(relevance)[::-1][:k])
    return float(actual / ideal) if ideal > 0 else 0.0


def recall_at_k(selection: Sequence[int], relevant: set[int], k: int | None = None) -> float:
    """Fraction of ground-truth relevant items appearing in the selection."""
    if not relevant:
        return 0.0
    sel = list(selection)[: (k if k is not None else len(selection))]
    return len(set(sel) & relevant) / len(relevant)


def precision_at_k(selection: Sequence[int], relevant: set[int], k: int | None = None) -> float:
    """Fraction of the selection that is relevant."""
    sel = list(selection)[: (k if k is not None else len(selection))]
    if not sel:
        return 0.0
    return len(set(sel) & relevant) / len(sel)

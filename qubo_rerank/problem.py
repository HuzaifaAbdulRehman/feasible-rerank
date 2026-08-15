"""The reranking instance handed to every solver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class RerankInstance:
    """One user's candidate set, ready to rerank.

    Attributes:
        relevance: ``(n,)`` scores from the retrieval model.
        similarity: ``(n, n)`` pairwise item similarity.
        k: list size to return.
        groups: ``(n,)`` group label per item (category, seller). Optional, but
            required by the fairness term and by :class:`QuotaMMR`.
        lam: diversity weight, passed through to the QUBO objective and MMR.
        mu: fairness weight.
        targets: optional per-group target counts.
        item_ids: original catalogue ids, so results can be mapped back out of the
            candidate-set index space.
    """

    relevance: np.ndarray
    similarity: np.ndarray
    k: int
    groups: np.ndarray | None = None
    lam: float = 1.0
    mu: float = 0.0
    targets: Mapping[int, float] | None = None
    item_ids: Sequence[int] | None = None

    def __post_init__(self) -> None:
        self.relevance = np.asarray(self.relevance, dtype=float).ravel()
        self.similarity = np.asarray(self.similarity, dtype=float)
        if self.groups is not None:
            self.groups = np.asarray(self.groups).ravel()

        n = self.relevance.shape[0]
        if self.similarity.shape != (n, n):
            raise ValueError(
                f"similarity must be ({n}, {n}), got {self.similarity.shape}"
            )
        if self.groups is not None and self.groups.shape[0] != n:
            raise ValueError(
                f"groups must have length {n}, got {self.groups.shape[0]}"
            )
        if self.k > n:
            raise ValueError(f"k={self.k} exceeds candidate count n={n}")

    @property
    def n(self) -> int:
        return int(self.relevance.shape[0])

    def to_catalogue_ids(self, selection: Sequence[int]) -> list[int]:
        """Map candidate-set indices back to original item ids."""
        if self.item_ids is None:
            return [int(i) for i in selection]
        return [int(self.item_ids[i]) for i in selection]

"""Common solver interface.

Every reranking strategy -- QUBO-based or classical -- returns the same thing: a list
of selected item indices plus a dict of stats. That uniformity is what lets the
experiment driver treat greedy top-k and simulated annealing as interchangeable, which
is the whole point of the benchmark.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class SolveResult:
    """Outcome of one reranking call.

    Attributes:
        selection: indices of the chosen items, in the candidate-set index space.
        stats: solver-reported numbers -- wall-clock, energy, number of reads. Energy
            is only comparable between solvers running on the *same* BQM.
    """

    selection: list[int]
    stats: dict[str, Any] = field(default_factory=dict)


class Solver(Protocol):
    """Structural type implemented by every reranker."""

    name: str

    def solve(self, problem: Any) -> SolveResult:  # pragma: no cover - protocol
        ...


class timed:
    """Context manager recording wall-clock seconds into a dict.

    Follows the per-stage timing practice in CQFS, where build time and sample time
    are recorded separately -- without that split you cannot tell whether a solver is
    slow or whether you merely built a big matrix.
    """

    def __init__(self, into: dict[str, Any], key: str) -> None:
        self.into = into
        self.key = key

    def __enter__(self) -> timed:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.into[self.key] = time.perf_counter() - self._t0


def selection_from_sample(sample: dict[int, int], expected_k: int | None = None) -> list[int]:
    """Extract selected indices from a dimod sample.

    Args:
        sample: mapping of variable -> 0/1.
        expected_k: if given, the selection is truncated or reported as-is when the
            solver violates the cardinality constraint. We do **not** silently pad:
            a wrong-length result means the penalty weight was too low and that needs
            to surface, not be papered over.
    """
    selection = sorted(int(v) for v, val in sample.items() if val == 1)
    if expected_k is not None and len(selection) != expected_k:
        # Caller decides what to do; recorded in stats by the solver wrappers.
        pass
    return selection


def ndcg_safe_scores(relevance: np.ndarray, selection: list[int]) -> np.ndarray:
    """Relevance of the selected items, in selection order."""
    return np.asarray(relevance, dtype=float)[selection]

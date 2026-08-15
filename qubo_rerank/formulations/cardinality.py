"""Exactly-k cardinality constraint.

The reranker must return a list of exactly ``k`` items. As a penalty term that is::

    H_card = P * (sum_i x_i - k)^2

Expanding the square gives linear terms ``P(1 - 2k) x_i`` and quadratic terms
``2P x_i x_j``. ``dimod`` ships this as :func:`dimod.generators.combinations`, so we
use that rather than expanding it by hand -- it is the same construction, tested
upstream, and avoids a class of sign errors.
"""

from __future__ import annotations

import dimod


def build_cardinality(n: int, k: int, strength: float = 1.0) -> dimod.BinaryQuadraticModel:
    """Build a BQM whose minimum energy states are exactly those with ``k`` ones.

    Args:
        n: number of candidate items (variables ``0..n-1``).
        k: required number of selected items.
        strength: penalty weight ``P``. Must be large enough to dominate the
            objective, or the solver will happily return the wrong list size.
            :func:`qubo_rerank.formulations.builder.suggest_strength` estimates it.

    Returns:
        A BINARY BQM over variables ``0..n-1``.
    """
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, n]; got k={k}, n={n}")

    return dimod.generators.combinations(n, k, strength=strength)

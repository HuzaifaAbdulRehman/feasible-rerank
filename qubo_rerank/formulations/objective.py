"""Relevance and diversity terms of the reranking objective.

Given ``n`` candidate items with relevance scores ``r`` and a pairwise similarity
matrix ``s``, the objective rewards relevant items and penalises selecting pairs of
similar items together::

    H_obj = -sum_i r_i x_i  +  lam * sum_{i<j} s_ij x_i x_j

The first term is linear (a bonus, hence the negative sign, since samplers minimise
energy). The second is quadratic and is what makes this a *quadratic* unconstrained
binary optimisation problem rather than a simple sort.
"""

from __future__ import annotations

import dimod
import numpy as np


def build_objective(
    relevance: np.ndarray,
    similarity: np.ndarray,
    lam: float = 1.0,
    normalize_inputs: bool = True,
) -> dimod.BinaryQuadraticModel:
    """Build the relevance + diversity BQM.

    Args:
        relevance: shape ``(n,)`` relevance score per candidate item.
        similarity: shape ``(n, n)`` symmetric pairwise similarity. The diagonal is
            ignored -- an item is trivially similar to itself and ``x_i * x_i == x_i``
            for binary variables, which would silently corrupt the linear terms.
        lam: weight on the diversity penalty. ``lam=0`` gives pure relevance, which
            reduces the QUBO optimum to greedy top-k.
        normalize_inputs: scale relevance and similarity to ``[0, 1]`` so that ``lam``
            means the same thing across datasets and models. Strongly recommended;
            raw ItemKNN scores and cosine similarities live on very different scales.

    Returns:
        A BINARY :class:`dimod.BinaryQuadraticModel` over variables ``0..n-1``.
    """
    relevance = np.asarray(relevance, dtype=float).ravel()
    similarity = np.asarray(similarity, dtype=float)

    n = relevance.shape[0]
    if similarity.shape != (n, n):
        raise ValueError(
            f"similarity must be ({n}, {n}) to match relevance, got {similarity.shape}"
        )

    if normalize_inputs:
        relevance = _unit_scale(relevance)
        similarity = _unit_scale(similarity)

    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")

    # Linear: reward relevance. Negative because samplers minimise.
    for i in range(n):
        bqm.add_variable(i, -float(relevance[i]))

    if lam != 0.0:
        # Quadratic: penalise co-selecting similar items. Upper triangle only --
        # dimod stores one coupler per unordered pair, so iterating the full matrix
        # would double-count.
        for i in range(n):
            for j in range(i + 1, n):
                w = 0.5 * (similarity[i, j] + similarity[j, i])  # enforce symmetry
                if w != 0.0:
                    bqm.add_quadratic(i, j, lam * float(w))

    return bqm


def _unit_scale(a: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]; returns zeros for a constant array."""
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi - lo <= 0.0:
        return np.zeros_like(a, dtype=float)
    return (a - lo) / (hi - lo)

"""Group-exposure fairness as a QUBO penalty.

This is the contribution of this project. Prior QUBO formulations for recommender
systems optimise relevance against diversity (Ferrari Dacrema et al., RecSys 2021) or
select features (CQFS, arXiv:2110.05089). Neither encodes *fair exposure across
groups* -- categories, sellers, or brands -- directly in the QUBO.

For a partition of the candidates into groups ``C``, we want each group's share of the
``k`` slots to sit near its target ``t_c``. As a penalty::

    H_fair = mu * sum_c ( sum_{i in c} x_i  -  t_c )^2

Expanding the square for a single group ``c`` with indicator ``y_c = sum_{i in c} x_i``::

    (y_c - t_c)^2 = y_c^2 - 2 t_c y_c + t_c^2

Because the variables are binary, ``x_i^2 == x_i``, so::

    y_c^2 = sum_{i in c} x_i + 2 * sum_{i<j in c} x_i x_j

giving linear coefficients ``mu * (1 - 2 t_c)`` for every ``i`` in group ``c`` and
quadratic coefficients ``2 mu`` for every within-group pair. The constant ``mu * t_c^2``
is dropped -- it shifts all energies equally and cannot change the argmin.

Note the sign of the interaction: it is *positive*, so co-selecting two items from the
same group is penalised. That is what pushes the solver to spread selections across
groups rather than filling the list from whichever group the relevance model favours.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import dimod
import numpy as np


def build_fairness(
    groups: Sequence[int] | np.ndarray,
    k: int,
    mu: float = 1.0,
    targets: Mapping[int, float] | None = None,
) -> dimod.BinaryQuadraticModel:
    """Build a group-exposure fairness penalty BQM.

    Args:
        groups: shape ``(n,)`` group label per candidate item -- category id, seller
            id, or any partition. Labels may be any hashable ints; they need not be
            contiguous.
        k: list size, used to derive default targets.
        mu: fairness weight. ``mu=0`` disables the term entirely (returns an empty
            BQM), which is the control condition for experiments.
        targets: optional explicit target count per group. Defaults to equal
            representation, ``k / n_groups`` for every group present. Pass this to
            model proportional targets (group share of the catalogue) or contractual
            minimum-exposure quotas for sellers.

    Returns:
        A BINARY BQM over variables ``0..n-1``.
    """
    groups = np.asarray(groups).ravel()
    n = groups.shape[0]

    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    # Register every variable so the BQM composes cleanly with the other terms even
    # when mu == 0 or a group is empty.
    for i in range(n):
        bqm.add_variable(i, 0.0)

    if mu == 0.0:
        return bqm

    unique = np.unique(groups)
    if targets is None:
        equal_share = k / len(unique)
        targets = {int(g): equal_share for g in unique}

    for g in unique:
        members = np.flatnonzero(groups == g)
        t_c = float(targets[int(g)])

        # Linear part: mu * (1 - 2 t_c) per member.
        lin = mu * (1.0 - 2.0 * t_c)
        for i in members:
            bqm.add_linear(int(i), lin)

        # Quadratic part: 2 * mu for every within-group pair.
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                bqm.add_quadratic(int(members[a]), int(members[b]), 2.0 * mu)

    return bqm


def exposure_targets_proportional(
    groups: Sequence[int] | np.ndarray, k: int
) -> dict[int, float]:
    """Targets proportional to each group's share of the candidate pool.

    Equal targets punish large groups for being large. Proportional targets instead
    ask only that the *ranking* does not distort the underlying catalogue mix, which
    is usually the fairer question for e-commerce categories.
    """
    groups = np.asarray(groups).ravel()
    n = groups.shape[0]
    unique, counts = np.unique(groups, return_counts=True)
    return {int(g): k * (c / n) for g, c in zip(unique, counts)}

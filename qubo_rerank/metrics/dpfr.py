"""Individual-item fairness measures from Rampisela et al.

Ports three measures from **"Evaluating the Diversity, Perspective and Fairness of
Recommender Systems"** (Rampisela, Maistro, Ruotsalo, Lioma), whose reference
implementation is MIT-licensed and vendored at ``reference/DPFR-fairness-eval``
(``RecBole/recbole/evaluator/metrics.py``). The formulas are theirs; only the plumbing
is rewritten, because the originals are `RecBole` evaluator classes that expect a
``dataobject`` of torch tensors and this project deliberately does not depend on either.

**Why bother, given fairness.py already has Gini and exposure parity.** Those are
group-level and hand-rolled. Group exposure parity is also the quantity the QUBO's
fairness term optimises directly, so reporting it alone is self-serving -- a solver can
look fair simply because it was told to be fair on exactly that axis. These measures are
*individual*-item, carefully motivated in a peer-reviewed venue, and orthogonal to what
the penalty optimises. They are the harder test.

All three treat exposure as position-discounted attention:

* **II-F** (individual item fairness) -- mean squared gap between the exposure an item
  actually receives from a user and the exposure it "deserves" from that user. Lower is
  better.
* **AI-F** (amortized item fairness) -- the same gap, but averaged over users *before*
  squaring. An item starved by one user and over-exposed by another cancels out here
  and does not in II-F, so AI-F is the more forgiving of the two, and a large II-F with
  a small AI-F means unfairness is being spread around rather than concentrated.
* **IBO** (item better-off) -- fraction of items receiving at least 1.1x the impact they
  would get under a uniform allocation. Higher is better.

.. warning::
   Under leave-one-out evaluation each user has exactly one relevant item, which is the
   degenerate end of what these measures were designed for -- they were built for
   graded, multi-item relevance. The numbers are still comparable *between rerankers on
   this benchmark*, which is what they are used for here, but they should not be quoted
   against values published elsewhere.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Rank-biased precision patience parameter, matching the DPFR reference implementation.
GAMMA = 0.8


def _exposure_matrices(
    selections: Sequence[Sequence[int]],
    relevant: Sequence[set[int]],
    n_catalogue: int,
    gamma: float = GAMMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(n_users, n_catalogue)`` actual and target exposure matrices.

    Actual exposure of the item at rank ``r`` is ``gamma^(r-1)``. Target exposure gives
    each user a fixed budget ``(1 - gamma^R) / (1 - gamma)`` for their ``R`` relevant
    items, split evenly among them; every irrelevant item's target is zero.
    """
    n_users = len(selections)
    exposure = np.zeros((n_users, n_catalogue), dtype=float)
    target = np.zeros((n_users, n_catalogue), dtype=float)

    for u, selection in enumerate(selections):
        for rank, item in enumerate(selection, start=1):
            exposure[u, int(item)] = gamma ** (rank - 1)

        relevant_items = [int(i) for i in relevant[u]]
        if relevant_items:
            budget = (1.0 - gamma ** len(relevant_items)) / (1.0 - gamma)
            target[u, relevant_items] = budget / len(relevant_items)

    return exposure, target


def individual_item_fairness(
    selections: Sequence[Sequence[int]],
    relevant: Sequence[set[int]],
    n_catalogue: int,
    gamma: float = GAMMA,
) -> tuple[float, float]:
    """II-F and AI-F. Both lower-is-better.

    Args:
        selections: recommended item ids per user, **in rank order** and in catalogue
            index space -- not per-user candidate indices.
        relevant: ground-truth relevant item ids per user, catalogue index space.
        n_catalogue: catalogue size.
        gamma: RBP patience.

    Returns:
        ``(II-F, AI-F)``.
    """
    exposure, target = _exposure_matrices(selections, relevant, n_catalogue, gamma)
    gap = exposure - target
    return float(np.mean(gap**2)), float(np.mean(np.mean(gap, axis=0) ** 2))


def item_better_off(
    selections: Sequence[Sequence[int]],
    relevant: Sequence[set[int]],
    n_catalogue: int,
    threshold: float = 1.1,
) -> float:
    """IBO: fraction of items doing at least ``threshold`` times better than uniform.

    Higher is better. Impact uses a ``1/rank`` discount rather than RBP, and counts only
    the exposure an item receives from users it is actually relevant to.

    This is the reference implementation's ``IBO_our`` rather than ``IBO_ori``. The
    original divides by every item's uniform impact including items that are relevant to
    nobody, which makes the whole metric NaN on any realistic catalogue; the corrected
    form restricts the denominator to items with at least one relevant user. Both live
    in the DPFR source and the authors report the corrected one.
    """
    n_users = len(selections)
    if n_users == 0 or n_catalogue == 0:
        return 0.0

    k = max((len(s) for s in selections), default=0)
    if k == 0:
        return 0.0

    impact = np.zeros((n_users, n_catalogue), dtype=float)
    relevance = np.zeros((n_users, n_catalogue), dtype=float)

    for u, selection in enumerate(selections):
        for rank, item in enumerate(selection, start=1):
            impact[u, int(item)] = 1.0 / rank
        for item in relevant[u]:
            relevance[u, int(item)] = 1.0

    discount_mass = float(np.sum(1.0 / np.arange(1, k + 1)))

    actual = (impact * relevance).sum(axis=0) / n_users
    uniform = relevance.sum(axis=0) * discount_mass / n_catalogue / n_users

    scorable = uniform != 0.0
    if not scorable.any():
        return 0.0

    return float(np.mean(actual[scorable] >= threshold * uniform[scorable]))

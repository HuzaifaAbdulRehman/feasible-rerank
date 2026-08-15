"""Synthetic e-commerce candidate sets.

Real datasets arrive with RecBole in the next step. This generator exists so the
pipeline is testable and reproducible without a multi-gigabyte download, and so the
benchmark has a case where the *correct* answer is known by construction.

The generator deliberately builds in the pathology that motivates the whole project:
relevance is correlated with group membership, so a greedy reranker fills the list
from one or two dominant categories and starves the rest. That is popularity bias in
miniature, and it is exactly what Butt's project 53923 targets.
"""

from __future__ import annotations

import numpy as np

from qubo_rerank.problem import RerankInstance


def make_instance(
    n_items: int = 60,
    n_groups: int = 6,
    k: int = 10,
    dominance: float = 0.6,
    intra_group_similarity: float = 0.7,
    noise: float = 0.15,
    lam: float = 1.0,
    mu: float = 0.0,
    seed: int | None = None,
) -> RerankInstance:
    """Generate one user's candidate set.

    Args:
        n_items: number of candidates (the retrieval stage's top-N).
        n_groups: number of categories/sellers.
        k: list size to select.
        dominance: how strongly relevance favours the first group. ``0`` gives
            uniform relevance across groups; ``1`` makes group 0 dominate outright.
        intra_group_similarity: mean similarity between items of the same group.
            Items in a category really are more alike, which is what makes diversity
            and fairness pull in the same direction here.
        noise: standard deviation of the relevance noise.
        lam: diversity weight carried into the instance.
        mu: fairness weight carried into the instance.
        seed: RNG seed.

    Returns:
        A :class:`RerankInstance`.
    """
    rng = np.random.default_rng(seed)

    groups = np.arange(n_items) % n_groups

    # Relevance decays with group index, so low-numbered groups dominate the ranking.
    group_bonus = dominance * (1.0 - groups / max(n_groups - 1, 1))
    base = rng.random(n_items) * (1.0 - dominance)
    relevance = np.clip(base + group_bonus + rng.normal(0, noise, n_items), 0.0, None)

    # Similarity: high within a group, low across groups.
    same_group = groups[:, None] == groups[None, :]
    sim = np.where(
        same_group,
        intra_group_similarity + rng.normal(0, 0.05, (n_items, n_items)),
        rng.random((n_items, n_items)) * 0.2,
    )
    sim = np.clip((sim + sim.T) / 2, 0.0, 1.0)
    np.fill_diagonal(sim, 0.0)

    return RerankInstance(
        relevance=relevance,
        similarity=sim,
        k=k,
        groups=groups,
        lam=lam,
        mu=mu,
        item_ids=list(range(n_items)),
    )


def make_users(
    n_users: int = 20, seed: int | None = None, **kwargs
) -> list[RerankInstance]:
    """Generate a population of candidate sets sharing the same generator settings.

    Catalogue-level metrics -- Gini, coverage -- are only meaningful across many
    users, since a single 10-item list cannot exhibit popularity bias on its own.
    """
    rng = np.random.default_rng(seed)
    return [
        make_instance(seed=int(rng.integers(0, 2**31 - 1)), **kwargs)
        for _ in range(n_users)
    ]

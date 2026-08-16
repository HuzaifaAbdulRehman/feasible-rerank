"""Implicit-feedback matrix factorisation, as a *second* retrieval model.

Every result in this project is conditioned on ItemKNN. That is the largest remaining
threat to the fairness findings, and it is a specific one rather than a vague caveat:
ItemKNN scores an item by its similarity to what a user already holds, which makes it
structurally biased toward **popular** items -- and popularity tiers are exactly the
groups the fairness term is being scored on. A sceptic can reasonably ask whether the
QUBO wins because it optimises fairness well, or because ItemKNN's particular bias
happens to be the one the constraint targets.

Matrix factorisation biases differently. It compresses the interaction matrix into a
low-rank space, so items are scored by latent-factor agreement rather than by
co-occurrence with the profile. Popular items still dominate -- they dominate everywhere
-- but the *shape* of the bias differs, and that is what makes it a real test rather than
a second run of the same experiment.

Implements the implicit-feedback ALS of Hu, Koren and Volinsky, *Collaborative Filtering
for Implicit Feedback Datasets* (ICDM 2008). Chosen over BPR because it is deterministic
given a seed -- no negative sampling -- which matters for a project whose protocol
depends on being able to rerun anything and get the same answer.

    minimise  sum_ui c_ui (p_ui - x_u . y_i)^2  +  reg (||X||^2 + ||Y||^2)

with preference ``p_ui = 1`` where an interaction exists and confidence
``c_ui = 1 + alpha * r_ui``. The alternating step has the closed form that makes ALS
practical: precompute ``YtY`` once per iteration, then each user solves a
``factors x factors`` system rather than anything of catalogue size.

Usage::

    python benchmarks/matrix_factorisation.py --data data/amazon_lb/Luxury_Beauty.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.loader import (
    Benchmark,
    interaction_matrix,
    k_core,
    leave_one_out,
    load_ratings,
    popularity_tiers,
)
from qubo_rerank.problem import RerankInstance


def train_als(
    matrix: sparse.csr_matrix,
    factors: int = 64,
    iterations: int = 15,
    reg: float = 0.01,
    alpha: float = 40.0,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Alternating least squares for implicit feedback. Returns ``(users, items)``.

    ``alpha=40`` is Hu et al.'s own value and is not a free parameter in the usual sense:
    it sets how much more a single observed interaction is trusted than an unobserved
    one, and the paper's results are reported at 40.

    The regularisation is scaled by the number of observations per row (a "weighted-lambda"
    variant). Without it, users with two interactions are regularised as hard as users
    with two hundred, and the resulting factors are dominated by heavy users -- which
    would confound a fairness experiment specifically.
    """
    n_users, n_items = matrix.shape
    rng = np.random.default_rng(seed)
    user_factors = 0.01 * rng.standard_normal((n_users, factors))
    item_factors = 0.01 * rng.standard_normal((n_items, factors))

    confidence = matrix.copy().astype(float)
    confidence.data = alpha * confidence.data

    eye = np.eye(factors)

    def solve_side(fixed: np.ndarray, observations: sparse.csr_matrix) -> np.ndarray:
        # The ALS trick: (Y'CY + reg I) = (Y'Y) + Y'(C - I)Y, and C - I is zero except
        # on observed entries. So Y'Y is computed once for everyone and only the handful
        # of observed columns per row contribute the rest.
        gram = fixed.T @ fixed
        out = np.zeros((observations.shape[0], fixed.shape[1]))

        for row in range(observations.shape[0]):
            start, end = observations.indptr[row], observations.indptr[row + 1]
            index = observations.indices[start:end]
            if index.size == 0:
                continue
            weights = observations.data[start:end]

            sub = fixed[index]
            a = gram + (sub * weights[:, None]).T @ sub + reg * index.size * eye
            b = ((1.0 + weights)[:, None] * sub).sum(axis=0)
            out[row] = np.linalg.solve(a, b)
        return out

    confidence_t = confidence.T.tocsr()
    for _ in range(iterations):
        user_factors = solve_side(item_factors, confidence)
        item_factors = solve_side(user_factors, confidence_t)

    return user_factors, item_factors


def item_similarity(item_factors: np.ndarray) -> np.ndarray:
    """Cosine similarity between latent item vectors, self-similarity zeroed.

    Clipped at zero. Latent factors are unconstrained in sign, so cosine can be negative
    -- but the diversity term reads this as "how much does picking both cost", and a
    negative cost would *reward* redundancy. Clipping keeps the QUBO's diversity term
    meaning what it means for the ItemKNN pipeline, which is what makes the two
    retrieval models comparable at all.
    """
    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    normalised = item_factors / np.maximum(norms, 1e-12)
    similarity = np.clip(normalised @ normalised.T, 0.0, None)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def load_mf_benchmark(
    path: str | Path,
    n_users: int = 200,
    n_candidates: int = 200,
    k: int = 10,
    n_groups: int = 4,
    min_interactions: int = 5,
    factors: int = 64,
    iterations: int = 15,
    reg: float = 0.01,
    alpha: float = 40.0,
    lam: float = 1.0,
    mu: float = 0.0,
    seed: int | None = 0,
) -> Benchmark:
    """Same pipeline as ``loader.load_benchmark``, with ALS in place of ItemKNN.

    Everything else is held identical -- the same k-core, the same leave-one-out split,
    the same candidate count, the same popularity-tier groups -- so a difference in
    results is attributable to the retrieval model and not to the protocol around it.
    """
    raw = load_ratings(path)
    filtered = k_core(raw, min_interactions=min_interactions)
    train, test = leave_one_out(filtered)

    matrix, user_index, item_ids = interaction_matrix(train, binary=True)
    n_catalogue = matrix.shape[1]
    if n_candidates > n_catalogue:
        raise ValueError(f"n_candidates={n_candidates} exceeds catalogue {n_catalogue}")

    user_factors, item_factors = train_als(
        matrix, factors=factors, iterations=iterations, reg=reg, alpha=alpha, seed=seed
    )
    similarity = item_similarity(item_factors)

    popularity = np.asarray(matrix.sum(axis=0)).ravel()
    catalogue_groups = popularity_tiers(popularity, n_tiers=n_groups)

    item_position = {iid: j for j, iid in enumerate(item_ids)}
    held_out = {
        row.user_id: item_position[row.item_id]
        for row in test.itertuples()
        if row.item_id in item_position
    }

    eligible = [uid for uid in held_out if uid in user_index]
    rng = np.random.default_rng(seed)
    if len(eligible) > n_users:
        chosen = rng.choice(len(eligible), size=n_users, replace=False)
        eligible = [eligible[i] for i in sorted(chosen)]

    instances: list[RerankInstance] = []
    relevant: list[set[int]] = []

    for uid in eligible:
        row = user_index[uid]
        scores = item_factors @ user_factors[row]
        scores[matrix[row].indices] = -np.inf      # never re-recommend the profile

        candidates = np.argpartition(-scores, n_candidates)[:n_candidates]
        candidates = candidates[np.argsort(-scores[candidates])]
        candidate_scores = scores[candidates]
        if not np.isfinite(candidate_scores).all():
            continue

        span = candidate_scores.max() - candidate_scores.min()
        normalised = (
            (candidate_scores - candidate_scores.min()) / span
            if span > 0
            else np.ones_like(candidate_scores)
        )

        instances.append(
            RerankInstance(
                relevance=normalised,
                similarity=similarity[np.ix_(candidates, candidates)],
                k=k,
                groups=catalogue_groups[candidates],
                lam=lam,
                mu=mu,
                item_ids=[int(c) for c in candidates],
            )
        )
        hit = np.flatnonzero(candidates == held_out[uid])
        relevant.append({int(hit[0])} if hit.size else set())

    return Benchmark(
        instances=instances,
        relevant=relevant,
        n_catalogue=n_catalogue,
        catalogue_groups=catalogue_groups,
        name="mf",
        stats={
            "raw_interactions": len(raw),
            "core_interactions": len(filtered),
            "users": matrix.shape[0],
            "items": n_catalogue,
            "sampled_users": len(instances),
            "model": f"ALS factors={factors} iters={iterations} alpha={alpha}",
            "similarity_mean": float(similarity[similarity > 0].mean()),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=ROOT / "data/amazon_lb/Luxury_Beauty.csv")
    ap.add_argument("--n-users", type=int, default=100)
    ap.add_argument("--factors", type=int, default=64)
    args = ap.parse_args()

    bench = load_mf_benchmark(args.data, n_users=args.n_users, factors=args.factors)
    print(f"dataset: {bench.name}")
    for key, value in bench.stats.items():
        print(f"  {key:20s} {value}")
    print(f"  candidate_hit_rate   {bench.candidate_hit_rate:.4f}")


if __name__ == "__main__":
    main()

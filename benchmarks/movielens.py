"""MovieLens 100K -> :class:`Benchmark`, with **real categories** as the fairness groups.

Every benchmark up to here has been an Amazon category with groups defined as
*popularity tiers*. That partition is standard and defensible, but it leaves two
limitations stacked on top of each other:

1. Every result is from one marketplace, so cross-domain generalisation is untested.
2. More seriously, the groups are defined by *the same signal the recommender is biased
   on*. Popularity tiers make the fairness term fight ItemKNN's popularity bias directly,
   which is the sharpest test of that bias -- but it is not a test of fairness across
   groups that mean something to a person. "Spread exposure across popularity deciles"
   and "spread exposure across genres" are different requirements, and a method could
   satisfy the first structurally while doing nothing for the second.

MovieLens 100K addresses both. It is a different domain, and ``u.item`` ships 19 genre
flags per film -- actual semantic categories, assigned by the dataset's curators rather
than derived from the interaction counts being evaluated.

**Multi-label genres, resolved to one group.** A film can carry several genres (Toy Story
is Animation + Children's + Comedy) while the fairness term partitions items, so each
film is first reduced to a single *primary* genre: the **rarest** of the genres it
carries. That keeps the distinctive label rather than the generic one -- a
Film-Noir/Drama picture is Film-Noir, not Drama -- and stops Drama absorbing half the
catalogue.

The groups are then the ``n_groups - 1`` most common primary genres, plus an "other"
bucket. At ``n_groups=4`` that is Drama / Comedy / Romance / other, sized roughly
376 / 275 / 190 / 510.

An earlier version instead took the rarest genres as the groups directly, which produced
sizes of **1 / 19 / 19 / 1310** -- a partition where "equal exposure" means giving a
one-film group a tenth of every list. Balanced bin-packing was also tried and gives
near-equal groups, but they are arbitrary ("Action + Horror + Children's + Western"), and
a group nobody can name is not a fairness constraint anyone would state. The interpretable
partition is the one kept.

Usage::

    python benchmarks/movielens.py --data data/ml100k/ml-100k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.loader import (
    Benchmark,
    candidate_similarity,
    cosine_similarity,
    cosine_similarity_sparse,
    interaction_matrix,
    k_core,
    leave_one_out,
    top_k_neighbours,
    top_k_neighbours_sparse,
)
from qubo_rerank.problem import RerankInstance

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

#: Column 5 onward in u.item, in file order.
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_ratings(path: Path) -> pd.DataFrame:
    """``u.data`` is tab-separated ``user, item, rating, timestamp``."""
    return pd.read_csv(
        path / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": str, "item_id": str, "rating": float, "timestamp": np.int64},
    )


def load_genres(path: Path) -> pd.DataFrame:
    """``u.item`` is pipe-separated, latin-1, with 19 trailing binary genre flags."""
    frame = pd.read_csv(
        path / "u.item",
        sep="|",
        header=None,
        encoding="latin-1",
        names=["item_id", "title", "release", "video_release", "url", *GENRES],
        dtype={"item_id": str},
    )
    return frame[["item_id", *GENRES]]


def genre_groups(genres: pd.DataFrame, item_ids: list[str], n_groups: int) -> np.ndarray:
    """One group per item: the ``n_groups-1`` most common primary genres, plus "other".

    A film's *primary* genre is the rarest genre it carries, so multi-genre films keep
    their distinctive label instead of collapsing into Drama.

    Returns group indices aligned to ``item_ids``, with ``n_groups - 1`` as "other".
    """
    indexed = genres.set_index("item_id")
    frequency = indexed[GENRES].sum()
    rarity_rank = {name: i for i, name in enumerate(frequency.sort_values().index)}

    flags = indexed[GENRES].to_numpy()
    ranks = np.array([rarity_rank[name] for name in GENRES])

    primary: dict[str, str] = {}
    for item, row in zip(indexed.index, flags, strict=True):
        present = np.flatnonzero(row)
        primary[item] = GENRES[present[np.argmin(ranks[present])]] if present.size else "unknown"

    counts = pd.Series(list(primary.values())).value_counts()
    named = list(counts.index[: n_groups - 1])
    index_of = {name: i for i, name in enumerate(named)}

    return np.array(
        [index_of.get(primary.get(item, "unknown"), n_groups - 1) for item in item_ids],
        dtype=int,
    )


def load_movielens(
    path: str | Path,
    n_users: int = 200,
    n_candidates: int = 200,
    k: int = 10,
    n_groups: int = 4,
    min_interactions: int = 5,
    topk_neighbours: int = 100,
    shrink: float = 100.0,
    lam: float = 1.0,
    mu: float = 0.0,
    binary: bool = True,
    seed: int | None = 0,
    sparse_similarity: bool | None = None,
    dense_similarity_limit: int = 256_000_000,
) -> Benchmark:
    """Build a benchmark from MovieLens 100K with genre-based groups.

    Mirrors :func:`benchmarks.loader.load_benchmark` step for step -- same k-core, same
    leave-one-out split, same ItemKNN, same candidate retrieval -- so that a difference
    in results is attributable to the data and the groups rather than to the pipeline.
    """
    path = Path(path)
    if not (path / "u.data").exists():
        raise FileNotFoundError(
            f"{path}/u.data not found. Download and unzip:\n    curl -o ml-100k.zip {DATA_URL}"
        )

    raw = load_ratings(path)
    filtered = k_core(raw, min_interactions=min_interactions)
    train, test = leave_one_out(filtered)

    matrix, user_index, item_ids = interaction_matrix(train, binary=binary)
    n_catalogue = matrix.shape[1]
    if n_candidates > n_catalogue:
        raise ValueError(f"n_candidates={n_candidates} exceeds catalogue {n_catalogue}")

    use_sparse = (
        (n_catalogue * n_catalogue * 8) > dense_similarity_limit
        if sparse_similarity is None
        else sparse_similarity
    )
    if use_sparse:
        similarity = cosine_similarity_sparse(matrix, shrink=shrink)
        scoring_similarity = top_k_neighbours_sparse(similarity, topk=topk_neighbours)
    else:
        similarity = cosine_similarity(matrix, shrink=shrink)
        scoring_similarity = top_k_neighbours(similarity, topk=topk_neighbours)

    catalogue_groups = genre_groups(load_genres(path), item_ids, n_groups)

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
        profile = matrix[user_index[uid]]
        scored = profile @ scoring_similarity
        scores = (
            scored.toarray() if hasattr(scored, "toarray") else np.asarray(scored)
        ).ravel()
        scores[profile.indices] = -np.inf

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
                similarity=candidate_similarity(similarity, candidates),
                k=k,
                groups=catalogue_groups[candidates],
                lam=lam,
                mu=mu,
                item_ids=[int(c) for c in candidates],
            )
        )
        hit = np.flatnonzero(candidates == held_out[uid])
        relevant.append({int(hit[0])} if hit.size else set())

    sizes = np.bincount(catalogue_groups, minlength=n_groups)
    return Benchmark(
        instances=instances,
        relevant=relevant,
        n_catalogue=n_catalogue,
        catalogue_groups=catalogue_groups,
        name="movielens_100k",
        stats={
            "raw_interactions": len(raw),
            "core_interactions": len(filtered),
            "users": matrix.shape[0],
            "items": n_catalogue,
            "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
            "sampled_users": len(instances),
            "group_sizes": sizes.tolist(),
            "groups_from": "genre (rarest-first), not popularity",
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data/ml100k/ml-100k")
    ap.add_argument("--n-users", type=int, default=200)
    ap.add_argument("--n-groups", type=int, default=4)
    args = ap.parse_args()

    bench = load_movielens(args.data, n_users=args.n_users, n_groups=args.n_groups)
    print(f"dataset: {bench.name}")
    for key, value in bench.stats.items():
        print(f"  {key:20s} {value}")
    print(f"  candidate_hit_rate   {bench.candidate_hit_rate:.4f}")


if __name__ == "__main__":
    main()

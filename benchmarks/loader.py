"""Real e-commerce candidate sets: Amazon Luxury Beauty -> :class:`RerankInstance`.

The synthetic generator proves the pipeline runs. This proves it runs on data someone
else collected, where the pathology is not built in by construction.

**Why not RecBole.** The plan called for RecBole's ``ItemKNN`` to supply both the
relevance scores ``r_i`` and the item-item similarity ``s_ij``. Reading
``recbole/model/general_recommender/itemknn.py`` shows the model is a shrunk cosine
similarity over the interaction matrix followed by a top-k neighbourhood truncation --
no gradients, no GPU, nothing torch actually does. Pulling in ``torch`` (~2.5 GB) to
reach forty lines of sparse linear algebra buys a dependency, not a capability. The
same formulation is implemented here directly against ``scipy.sparse``:

    sim(i, j) = <x_i, x_j> / (||x_i|| * ||x_j|| + shrink + eps)

which is RecBole's ``ComputeSimilarity`` with ``method='item'``. The on-disk format
stays RecBole-compatible, so swapping the real thing back in later is a loader change
and nothing else.

**Groups.** The ratings file carries no category or seller metadata, so the group
partition used by the fairness term is *popularity tier*: items are rank-ordered by
training interaction count and cut into equal-sized tiers, tier 0 being the short
head. This is the standard short-head/long-tail partition from the popularity-bias
literature (Abdollahpouri et al., "Managing Popularity Bias in Recommender Systems
with Personalized Re-ranking", FLAIRS 2019). It is also the honest choice here: it
makes the fairness term fight the exact bias ItemKNN exhibits, rather than a
partition chosen to flatter it. Pass ``groups=`` explicitly to use real categories
once metadata is available.

**Evaluation protocol.** Leave-one-out on the most recent interaction per user, which
is what the Amazon timestamp column is for. Because candidates are the top-N by the
same model being reranked, the held-out item is often not in the candidate set at
all; :attr:`Benchmark.candidate_hit_rate` reports that ceiling explicitly so Recall@k
is read against what is actually achievable rather than against 1.0.

Usage::

    python benchmarks/loader.py --data data/amazon_lb/Luxury_Beauty.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# Importable both as ``benchmarks.loader`` and as a script run from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qubo_rerank.problem import RerankInstance

RATINGS_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/"
    "categoryFilesSmall/Luxury_Beauty.csv"
)

#: The McAuley ratings-only files are headerless and in this column order.
RATINGS_COLUMNS = ["item_id", "user_id", "rating", "timestamp"]


# --------------------------------------------------------------------------- data


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Read a headerless ``item,user,rating,timestamp`` ratings file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with:\n"
            f"    curl -o {path} {RATINGS_URL}"
        )
    return pd.read_csv(
        path,
        names=RATINGS_COLUMNS,
        dtype={"item_id": str, "user_id": str, "rating": float, "timestamp": np.int64},
    )


def k_core(df: pd.DataFrame, min_interactions: int = 5, max_iter: int = 30) -> pd.DataFrame:
    """Iteratively drop users and items with fewer than ``min_interactions``.

    Dropping users changes item counts and vice versa, so a single pass is not enough;
    this repeats until the frame stops shrinking. Standard preprocessing, and the
    reason published numbers on "the same dataset" are only comparable when the core
    threshold matches.
    """
    for _ in range(max_iter):
        user_counts = df.user_id.value_counts()
        item_counts = df.item_id.value_counts()
        keep = df.user_id.isin(
            user_counts[user_counts >= min_interactions].index
        ) & df.item_id.isin(item_counts[item_counts >= min_interactions].index)
        if keep.all():
            break
        df = df[keep]
    return df.reset_index(drop=True)


def leave_one_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's most recent interaction as the test item.

    ``mergesort`` is stable, so users whose last interactions share a timestamp get a
    deterministic -- if arbitrary -- tie-break rather than a run-dependent one.
    """
    ordered = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    test = ordered.groupby("user_id", sort=False).tail(1)
    train = ordered.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def interaction_matrix(
    train: pd.DataFrame, binary: bool = True
) -> tuple[sparse.csr_matrix, dict[str, int], list[str]]:
    """Build the ``(n_users, n_items)`` sparse interaction matrix.

    Args:
        train: training interactions.
        binary: treat any interaction as 1. Implicit feedback is the norm for top-N
            recommendation -- a 5-star rating does not mean the user is five times
            more likely to want the item next.

    Returns:
        ``(matrix, user_index, item_ids)`` where ``user_index`` maps raw user id to
        row, and ``item_ids[j]`` is the raw item id of column ``j``.
    """
    users = train.user_id.astype("category")
    items = train.item_id.astype("category")

    values = np.ones(len(train)) if binary else train.rating.to_numpy(dtype=float)
    matrix = sparse.csr_matrix(
        (values, (users.cat.codes.to_numpy(), items.cat.codes.to_numpy())),
        shape=(len(users.cat.categories), len(items.cat.categories)),
    )
    user_index = {uid: i for i, uid in enumerate(users.cat.categories)}
    return matrix, user_index, list(items.cat.categories)


# ---------------------------------------------------------------------- similarity


def cosine_similarity(matrix: sparse.csr_matrix, shrink: float = 100.0) -> np.ndarray:
    """Shrunk item-item cosine similarity -- RecBole's ``ItemKNN`` similarity.

    The shrink term is Cremonesi's: it divides the raw co-occurrence by the norms
    *plus* a constant, so a pair of items sharing two users out of two does not score
    as highly as a pair sharing two hundred out of two hundred. Without it the
    similarity matrix is dominated by items nobody bought.

    Returns a dense matrix -- fine at the 5-core catalogue size (~1.4k items, ~15 MB)
    and it keeps the per-candidate submatrix extraction trivial. Sparsify here if the
    catalogue grows past ~10k items.
    """
    co_occurrence = np.asarray((matrix.T @ matrix).todense(), dtype=float)
    norms = np.sqrt(np.diag(co_occurrence))
    denominator = np.outer(norms, norms) + shrink + 1e-6

    similarity = co_occurrence / denominator
    np.fill_diagonal(similarity, 0.0)
    return similarity


def cosine_similarity_sparse(
    matrix: sparse.csr_matrix, shrink: float = 100.0
) -> sparse.csr_matrix:
    """Shrunk item-item cosine similarity, without ever going dense.

    :func:`cosine_similarity` allocates ``n_items ** 2`` floats. That is 15 MB on Amazon
    Luxury Beauty and fine; it is **1,016 MB** on Amazon Digital Music, whose catalogue
    survives 5-core filtering at 11,269 items. That measurement is what this function
    exists for -- the docstring above has always said "sparsify past ~10k items", and
    this is that.

    The trick is that the dense object is never *needed*. Co-occurrence ``XᵀX`` is
    naturally sparse (two items co-occur only if some user held both), and the
    normaliser is required only at positions where co-occurrence is non-zero -- a
    similarity of ``0 / anything`` is still zero. So the norms are applied to
    ``.data`` in place rather than through an ``n x n`` outer product, which is the
    single allocation that made the dense version impossible.

    Returns the same values as :func:`cosine_similarity` to floating-point precision,
    which ``tests/test_loader.py`` asserts directly.
    """
    co_occurrence = (matrix.T @ matrix).tocsr().astype(float)
    norms = np.sqrt(co_occurrence.diagonal())

    # Row index of every stored entry, obtained from the CSR pointer array rather than
    # by building a coordinate matrix -- nnz-sized, not n²-sized.
    rows = np.repeat(np.arange(co_occurrence.shape[0]), np.diff(co_occurrence.indptr))
    cols = co_occurrence.indices

    co_occurrence.data /= norms[rows] * norms[cols] + shrink + 1e-6
    co_occurrence.setdiag(0.0)
    co_occurrence.eliminate_zeros()
    return co_occurrence


def top_k_neighbours_sparse(
    similarity: sparse.csr_matrix, topk: int = 100
) -> sparse.csr_matrix:
    """Keep each item's ``topk`` nearest neighbours; sparse counterpart.

    Works row by row over the CSR structure, so cost is proportional to the number of
    stored entries rather than to ``n_items ** 2``. Rows already holding no more than
    ``topk`` entries are passed through untouched.
    """
    indptr, indices, data = similarity.indptr, similarity.indices, similarity.data
    keep_rows, keep_cols, keep_data = [], [], []

    for row in range(similarity.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        row_data = data[start:end]
        row_cols = indices[start:end]

        if row_data.size > topk:
            # Same (-value, column) ordering as the dense path, so the two agree
            # exactly rather than merely both being valid.
            best = np.lexsort((row_cols, -row_data))[:topk]
            row_data, row_cols = row_data[best], row_cols[best]

        keep_rows.append(np.full(row_cols.size, row, dtype=np.int32))
        keep_cols.append(row_cols)
        keep_data.append(row_data)

    return sparse.csr_matrix(
        (
            np.concatenate(keep_data) if keep_data else np.array([]),
            (
                np.concatenate(keep_rows) if keep_rows else np.array([], dtype=np.int32),
                np.concatenate(keep_cols) if keep_cols else np.array([], dtype=np.int32),
            ),
        ),
        shape=similarity.shape,
    )


def candidate_similarity(similarity, candidates: np.ndarray) -> np.ndarray:
    """The dense ``len(candidates) x len(candidates)`` block the QUBO needs.

    The catalogue-wide matrix may be sparse, but the *instance* is always dense: the
    reranker sees a few hundred candidates and the formulation reads every pair. Fancy
    indexing differs between the two representations, so it is done here once rather
    than at the call site.
    """
    if sparse.issparse(similarity):
        return np.asarray(similarity[candidates][:, candidates].todense(), dtype=float)
    return similarity[np.ix_(candidates, candidates)]


def top_k_neighbours(similarity: np.ndarray, topk: int = 100) -> np.ndarray:
    """Keep only each item's ``topk`` nearest neighbours, zeroing the rest.

    ItemKNN scores with a truncated neighbourhood: distant neighbours are noise, and
    keeping them makes every user's score vector drift toward global popularity. The
    *untruncated* matrix is still what the diversity term should see, which is why
    this returns a copy rather than mutating in place.
    """
    if topk >= similarity.shape[1]:
        return similarity.copy()

    # Ties at the k-th position are common -- on Amazon Software, 98 of 727 rows have
    # several neighbours sharing the boundary value exactly, one row with ten of them.
    # argpartition picks among them by memory order, which is a valid top-k but a
    # different one depending on how the matrix was built. Sorting by (-value, column)
    # makes the choice deterministic and identical to the sparse path, so which
    # representation was used cannot change a result.
    columns = np.arange(similarity.shape[1])
    truncated = np.zeros_like(similarity)
    for row in range(similarity.shape[0]):
        keep = np.lexsort((columns, -similarity[row]))[:topk]
        truncated[row, keep] = similarity[row, keep]
    return truncated


def popularity_tiers(popularity: np.ndarray, n_tiers: int = 4) -> np.ndarray:
    """Cut items into equal-sized popularity tiers; tier 0 is the short head.

    Equal *item counts* per tier, not equal popularity mass -- so the fairness term's
    default equal-share targets are attainable rather than absurd.
    """
    n_items = popularity.shape[0]
    # Rank 0 = most popular. Negate so argsort descends; the double argsort turns the
    # ordering into a rank per item.
    rank = np.argsort(np.argsort(-popularity, kind="stable"), kind="stable")
    return (rank * n_tiers // n_items).astype(int)


# ----------------------------------------------------------------------- benchmark


@dataclass
class Benchmark:
    """A population of real candidate sets plus the ground truth to score them against.

    Attributes:
        instances: one :class:`RerankInstance` per sampled user.
        relevant: per instance, the candidate-set indices of that user's held-out
            items. Empty when the held-out item missed the candidate set.
        n_catalogue: catalogue size, for catalogue-level metrics. Note these must be
            computed over ``instance.to_catalogue_ids(selection)`` -- unlike the
            synthetic benchmark, candidate index ``3`` means a different item for
            every user.
        catalogue_groups: popularity tier per catalogue item.
        name: dataset label for results tables.
    """

    instances: list[RerankInstance]
    relevant: list[set[int]]
    n_catalogue: int
    catalogue_groups: np.ndarray
    name: str = "amazon_lb"
    stats: dict = field(default_factory=dict)

    @property
    def candidate_hit_rate(self) -> float:
        """Fraction of users whose held-out item made it into the candidate set.

        This is the ceiling on Recall@k. Reranking cannot recover an item retrieval
        never surfaced, so reporting recall without this number overstates how much
        headroom the reranker had.
        """
        if not self.relevant:
            return 0.0
        return sum(1 for r in self.relevant if r) / len(self.relevant)


def load_benchmark(
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
    """Load a ratings file and turn it into reranking instances.

    Args:
        path: ratings CSV.
        n_users: how many users to sample for the benchmark.
        n_candidates: retrieval stage top-N -- the QUBO's problem size.
        k: list size.
        n_groups: number of popularity tiers.
        min_interactions: k-core threshold.
        topk_neighbours: ItemKNN neighbourhood size used for *scoring*.
        shrink: cosine shrinkage.
        lam: diversity weight carried into each instance.
        mu: fairness weight carried into each instance.
        binary: implicit feedback rather than raw ratings.
        seed: RNG seed for the user sample.
        sparse_similarity: force the sparse or dense similarity path. ``None`` chooses
            from the catalogue size that k-core filtering actually produced.
        dense_similarity_limit: byte budget above which the sparse path is used.
            Defaults to 256 MB, which admits every catalogue here except Digital Music's
            11,269 items (~1 GB).

    Returns:
        A :class:`Benchmark`.
    """
    raw = load_ratings(path)
    filtered = k_core(raw, min_interactions=min_interactions)
    train, test = leave_one_out(filtered)

    matrix, user_index, item_ids = interaction_matrix(train, binary=binary)
    n_catalogue = matrix.shape[1]

    if n_candidates > n_catalogue:
        raise ValueError(
            f"n_candidates={n_candidates} exceeds catalogue size {n_catalogue} "
            f"after {min_interactions}-core filtering"
        )

    # A dense similarity matrix costs n_catalogue^2 * 8 bytes. That is 15 MB at 1,366
    # items and 1,016 MB at 11,269, so the choice is made from the catalogue actually
    # produced by the k-core filter rather than assumed. `sparse_similarity=True/False`
    # forces either path, which is what the equivalence test uses.
    dense_bytes = n_catalogue * n_catalogue * 8
    use_sparse = (
        dense_bytes > dense_similarity_limit
        if sparse_similarity is None
        else sparse_similarity
    )

    if use_sparse:
        similarity = cosine_similarity_sparse(matrix, shrink=shrink)
        scoring_similarity = top_k_neighbours_sparse(similarity, topk=topk_neighbours)
    else:
        similarity = cosine_similarity(matrix, shrink=shrink)
        scoring_similarity = top_k_neighbours(similarity, topk=topk_neighbours)

    popularity = np.asarray(matrix.sum(axis=0)).ravel()
    catalogue_groups = popularity_tiers(popularity, n_tiers=n_groups)

    item_position = {iid: j for j, iid in enumerate(item_ids)}
    held_out = {
        row.user_id: item_position[row.item_id]
        for row in test.itertuples()
        if row.item_id in item_position
    }

    # Only users who survived the split with a usable held-out item are worth scoring.
    eligible = [uid for uid in held_out if uid in user_index]
    rng = np.random.default_rng(seed)
    if len(eligible) > n_users:
        chosen = rng.choice(len(eligible), size=n_users, replace=False)
        eligible = [eligible[i] for i in sorted(chosen)]

    instances: list[RerankInstance] = []
    relevant: list[set[int]] = []

    for uid in eligible:
        row = user_index[uid]
        profile = matrix[row]
        # A sparse scoring matrix makes this product sparse too, and np.asarray on a
        # sparse matrix yields a 0-d object array rather than the scores -- so the
        # result is densified explicitly rather than coerced.
        scored = profile @ scoring_similarity
        scores = (
            scored.toarray() if sparse.issparse(scored) else np.asarray(scored)
        ).ravel()

        # Never recommend something already in the training profile.
        scores[profile.indices] = -np.inf

        candidates = np.argpartition(-scores, n_candidates)[:n_candidates]
        candidates = candidates[np.argsort(-scores[candidates])]

        candidate_scores = scores[candidates]
        if not np.isfinite(candidate_scores).all():
            # Too few scorable items for this user; skip rather than fabricate.
            continue

        # Rescale to [0, 1] so lam and mu mean the same thing here as on the synthetic
        # benchmark. ItemKNN scores have no natural scale, and the QUBO weights the
        # relevance term against a similarity matrix that is already bounded in [0, 1].
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

        target = held_out[uid]
        hit = np.flatnonzero(candidates == target)
        relevant.append({int(hit[0])} if hit.size else set())

    return Benchmark(
        instances=instances,
        relevant=relevant,
        n_catalogue=n_catalogue,
        catalogue_groups=catalogue_groups,
        stats={
            "raw_interactions": len(raw),
            "core_interactions": len(filtered),
            "users": matrix.shape[0],
            "items": n_catalogue,
            "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
            "sampled_users": len(instances),
            "similarity_mean": float(
                np.mean([i.similarity.mean() for i in instances]) if instances else 0.0
            ),
            "suggested_lam": suggest_lam(instances),
        },
    )


def suggest_lam(instances: list[RerankInstance]) -> float:
    """A ``lam`` at which the diversity term is roughly the size of the relevance term.

    Real similarity matrices are near-empty -- most item pairs in a candidate set share
    no users at all, so mean off-diagonal similarity here is ~0.001 against relevance
    rescaled to [0, 1]. At ``lam=1`` the diversity term is three orders of magnitude
    too small to move the argmin, and the QUBO silently degenerates into greedy top-k.

    A selection of ``k`` items contributes ``k`` relevance terms and ``k(k-1)/2``
    similarity terms, so parity is at::

        lam ~ (k * mean_relevance) / (k(k-1)/2 * mean_similarity)

    This is a starting point for the sweep, not a tuned value -- the point of the
    sweep is to trace the whole curve, and reporting a single hand-picked ``lam`` is
    how diversity results get talked into existence.
    """
    if not instances:
        return 1.0

    ratios = []
    for inst in instances:
        k = inst.k
        n = inst.n
        off_diagonal = (inst.similarity.sum()) / max(n * (n - 1), 1)
        pairs = k * (k - 1) / 2
        if off_diagonal <= 0 or pairs == 0:
            continue
        ratios.append((k * float(inst.relevance.mean())) / (pairs * off_diagonal))

    return float(np.median(ratios)) if ratios else 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/amazon_lb/Luxury_Beauty.csv"))
    ap.add_argument("--n-users", type=int, default=200)
    ap.add_argument("--n-candidates", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    bench = load_benchmark(
        args.data, n_users=args.n_users, n_candidates=args.n_candidates, k=args.k
    )

    print(f"dataset: {bench.name}")
    for key, value in bench.stats.items():
        print(f"  {key:20s} {value}")
    print(f"  {'candidate_hit_rate':20s} {bench.candidate_hit_rate:.4f}")

    inst = bench.instances[0]
    print(f"\nfirst instance: n={inst.n} k={inst.k}")
    print(f"  relevance  min={inst.relevance.min():.4f} max={inst.relevance.max():.4f}")
    print(f"  similarity mean={inst.similarity.mean():.4f} max={inst.similarity.max():.4f}")
    counts = np.bincount(inst.groups, minlength=4)
    print(f"  popularity tiers in candidate set: {counts.tolist()} (tier 0 = short head)")


if __name__ == "__main__":
    main()

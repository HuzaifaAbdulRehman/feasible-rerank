"""Tests for the real-data pipeline in ``benchmarks/loader.py``.

Everything the README reports on Amazon Luxury Beauty is downstream of this file. A
bug here does not raise -- it produces plausible numbers, slightly too good, and the
write-up then argues from them. The split is the dangerous part specifically: if a
user's held-out purchase were also left in the training matrix, ItemKNN would rank it
highly for reasons that have nothing to do with generalisation, recall would rise, and
the resulting claim ("driving parity down by 79% costs 15% of predictive accuracy, not
36%") would be measuring leakage.

So the emphasis here is on the properties that would be silently violated rather than
on line coverage: disjointness of train and test, iterative convergence of the k-core,
and the direction of the popularity ranking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.loader import (
    RATINGS_COLUMNS,
    cosine_similarity,
    cosine_similarity_sparse,
    interaction_matrix,
    k_core,
    leave_one_out,
    load_benchmark,
    load_ratings,
    popularity_tiers,
    suggest_lam,
    top_k_neighbours,
    top_k_neighbours_sparse,
)


def frame(rows: list[tuple[str, str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


@pytest.fixture
def ratings() -> pd.DataFrame:
    """Four users over five items, with distinct timestamps per user."""
    rows = []
    for u, user in enumerate(["u1", "u2", "u3", "u4"]):
        for i, item in enumerate(["a", "b", "c", "d", "e"]):
            rows.append((user, item, 5.0, 1000 + 10 * u + i))
    return frame(rows)


class TestKCore:
    def test_survivors_all_meet_the_threshold(self, ratings):
        # One extra user with a single interaction, and one item only they touched.
        extra = frame([("u5", "z", 5.0, 2000)])
        core = k_core(pd.concat([ratings, extra], ignore_index=True), min_interactions=4)

        assert core.user_id.value_counts().min() >= 4
        assert core.item_id.value_counts().min() >= 4
        assert "u5" not in set(core.user_id)
        assert "z" not in set(core.item_id)

    def test_iterates_rather_than_filtering_once(self):
        """A single pass is not enough, and this is the case that proves it.

        ``u3`` has 3 interactions and survives a first pass at threshold 3. But two of
        them are on items ``y`` and ``z``, which have only 1 and 1 users and are
        dropped -- which takes ``u3`` down to 1 interaction. A one-pass filter would
        leave ``u3`` in the output holding a single interaction.
        """
        rows = []
        for user in ["u1", "u2", "u3"]:
            for item in ["a", "b", "c"]:
                rows.append((user, item, 5.0, 1))
        rows += [("u3", "y", 5.0, 2), ("u3", "z", 5.0, 3)]

        core = k_core(frame(rows), min_interactions=3)

        assert set(core.item_id) == {"a", "b", "c"}
        assert core.user_id.value_counts().min() >= 3

    def test_empty_result_is_returned_not_crashed(self):
        core = k_core(frame([("u1", "a", 5.0, 1)]), min_interactions=5)
        assert len(core) == 0


class TestLeaveOneOut:
    def test_test_item_is_the_users_latest(self, ratings):
        _, test = leave_one_out(ratings)

        for _, row in test.iterrows():
            latest = ratings[ratings.user_id == row.user_id].timestamp.max()
            assert row.timestamp == latest

    def test_one_test_item_per_user(self, ratings):
        _, test = leave_one_out(ratings)

        assert len(test) == ratings.user_id.nunique()
        assert test.user_id.is_unique

    def test_train_and_test_are_disjoint(self, ratings):
        """The leakage check. A held-out item must not also be in training."""
        train, test = leave_one_out(ratings)

        train_pairs = set(zip(train.user_id, train.item_id, train.timestamp, strict=True))
        test_pairs = set(zip(test.user_id, test.item_id, test.timestamp, strict=True))
        assert train_pairs.isdisjoint(test_pairs)

    def test_split_loses_no_interactions(self, ratings):
        train, test = leave_one_out(ratings)
        assert len(train) + len(test) == len(ratings)

    def test_tied_timestamps_split_deterministically(self):
        """Stable sort, so a tie resolves the same way on every run.

        Arbitrary is acceptable here; run-dependent is not, because it would make the
        whole benchmark irreproducible for reasons no seed controls.
        """
        rows = [("u1", item, 5.0, 999) for item in ["a", "b", "c"]]
        first = leave_one_out(frame(rows))[1]

        for _ in range(5):
            again = leave_one_out(frame(rows))[1]
            assert again.item_id.tolist() == first.item_id.tolist()


class TestInteractionMatrix:
    def test_binary_ignores_rating_magnitude(self):
        rows = [("u1", "a", 1.0, 1), ("u1", "b", 5.0, 2), ("u2", "a", 3.0, 3)]
        matrix, _, _ = interaction_matrix(frame(rows), binary=True)

        dense = np.asarray(matrix.todense())
        assert set(dense[dense > 0].tolist()) == {1.0}
        assert dense.sum() == 3.0  # one entry per interaction, magnitude discarded

    def test_explicit_keeps_rating_magnitude(self):
        rows = [("u1", "a", 1.0, 1), ("u1", "b", 5.0, 2)]
        matrix, _, item_ids = interaction_matrix(frame(rows), binary=False)

        dense = np.asarray(matrix.todense())
        assert dense[0, item_ids.index("b")] == 5.0

    def test_indices_round_trip_to_raw_ids(self, ratings):
        matrix, user_index, item_ids = interaction_matrix(ratings)

        assert matrix.shape == (ratings.user_id.nunique(), ratings.item_id.nunique())
        for uid, row in user_index.items():
            raw_items = set(ratings[ratings.user_id == uid].item_id)
            cols = matrix[row].indices
            assert {item_ids[c] for c in cols} == raw_items


class TestCosineSimilarity:
    def test_is_symmetric_with_zero_diagonal(self, ratings):
        matrix, _, _ = interaction_matrix(ratings)
        similarity = cosine_similarity(matrix, shrink=0.0)

        assert np.allclose(similarity, similarity.T)
        assert np.allclose(np.diag(similarity), 0.0)

    def test_matches_cosine_by_hand_without_shrink(self):
        """Deliberately asymmetric supports, so the denominator has to be right.

        Two items held by exactly the same users give cosine 1.0 whatever the norms
        are, which would pass even if the denominator were wrong. Here ``a`` is held
        by three users and ``b`` by two of those three:

            a = (1,1,1), b = (1,1,0)  ->  a.b = 2, |a| = sqrt(3), |b| = sqrt(2)
            cosine = 2 / sqrt(6) = 0.8165
        """
        rows = [
            ("u1", "a", 1.0, 1), ("u1", "b", 1.0, 2),
            ("u2", "a", 1.0, 3), ("u2", "b", 1.0, 4),
            ("u3", "a", 1.0, 5),
        ]
        matrix, _, item_ids = interaction_matrix(frame(rows))
        similarity = cosine_similarity(matrix, shrink=0.0)

        a, b = item_ids.index("a"), item_ids.index("b")
        assert similarity[a, b] == pytest.approx(2.0 / np.sqrt(6.0), rel=1e-5)

    def test_identical_support_scores_one(self):
        rows = [
            ("u1", "a", 1.0, 1), ("u1", "b", 1.0, 2),
            ("u2", "a", 1.0, 3), ("u2", "b", 1.0, 4),
        ]
        matrix, _, item_ids = interaction_matrix(frame(rows))
        similarity = cosine_similarity(matrix, shrink=0.0)

        a, b = item_ids.index("a"), item_ids.index("b")
        assert similarity[a, b] == pytest.approx(1.0, rel=1e-5)

    def test_shrink_penalises_low_support_pairs_hardest(self):
        """Cremonesi's shrink term, stated as the behaviour it is there to produce.

        Two items sharing 2 users out of 2 have raw cosine 1.0, exactly like two items
        sharing 200 out of 200. Shrink is what separates them, and it must bite harder
        on the thin pair.
        """
        thin = [(f"t{u}", item, 1.0, u) for u in range(2) for item in ["a", "b"]]
        thick = [(f"k{u}", item, 1.0, u) for u in range(200) for item in ["c", "d"]]
        matrix, _, item_ids = interaction_matrix(frame(thin + thick))

        raw = cosine_similarity(matrix, shrink=0.0)
        shrunk = cosine_similarity(matrix, shrink=100.0)
        a, b = item_ids.index("a"), item_ids.index("b")
        c, d = item_ids.index("c"), item_ids.index("d")

        assert raw[a, b] == pytest.approx(raw[c, d], rel=1e-6)  # identical before
        assert shrunk[a, b] < shrunk[c, d]                       # separated after


class TestTopKNeighbours:
    def test_keeps_the_largest_k_per_row(self):
        similarity = np.array(
            [[0.0, 0.9, 0.1, 0.5],
             [0.9, 0.0, 0.7, 0.2],
             [0.1, 0.7, 0.0, 0.8],
             [0.5, 0.2, 0.8, 0.0]]
        )
        truncated = top_k_neighbours(similarity, topk=2)

        for row in range(4):
            kept = np.nonzero(truncated[row])[0]
            assert len(kept) == 2
            assert set(kept) == set(np.argsort(-similarity[row])[:2])

    def test_does_not_mutate_its_input(self):
        """The diversity term needs the *untruncated* matrix, so this must copy."""
        similarity = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.7], [0.1, 0.7, 0.0]])
        before = similarity.copy()

        top_k_neighbours(similarity, topk=1)

        assert np.array_equal(similarity, before)

    def test_topk_at_or_above_width_is_a_copy(self):
        similarity = np.array([[0.0, 0.5], [0.5, 0.0]])
        out = top_k_neighbours(similarity, topk=5)

        assert np.array_equal(out, similarity)
        assert out is not similarity


class TestPopularityTiers:
    def test_tier_zero_is_the_short_head(self):
        popularity = np.array([1, 100, 50, 3, 80, 2, 60, 4])
        tiers = popularity_tiers(popularity, n_tiers=4)

        assert tiers[np.argmax(popularity)] == 0
        assert tiers[np.argmin(popularity)] == 3

    def test_tiers_are_equal_sized_when_divisible(self):
        tiers = popularity_tiers(np.arange(100), n_tiers=4)
        counts = np.bincount(tiers, minlength=4)

        assert counts.tolist() == [25, 25, 25, 25]

    def test_every_item_lands_in_range(self):
        # 10 items into 4 tiers does not divide evenly; nothing may fall outside.
        tiers = popularity_tiers(np.arange(10), n_tiers=4)

        assert tiers.min() >= 0
        assert tiers.max() < 4
        assert len(tiers) == 10

    def test_monotone_in_popularity(self):
        popularity = np.array([5, 50, 500, 1, 10, 100])
        tiers = popularity_tiers(popularity, n_tiers=3)

        order = np.argsort(-popularity)
        assert list(tiers[order]) == sorted(tiers[order])


# ------------------------------------------------------------ end-to-end loading


@pytest.fixture
def ratings_csv(tmp_path: Path) -> Path:
    """A synthetic ratings file in the Amazon export's column order.

    Small, but structured the way the real file is: a popularity gradient so tiers are
    non-degenerate, overlapping user histories so ItemKNN has co-occurrence to work
    with, and strictly increasing timestamps so the leave-one-out split is well defined.
    Writing one here means the whole real-data path is covered in CI without a 24 MB
    download.
    """
    rng = np.random.default_rng(0)
    n_users, n_items = 60, 40
    rows = []
    stamp = 1_000_000

    # Zipf-ish: item j is chosen with weight 1/(j+1), so tier 0 is a genuine short head.
    weights = 1.0 / (np.arange(n_items) + 1.0)
    weights /= weights.sum()

    for u in range(n_users):
        picks = rng.choice(n_items, size=12, replace=False, p=weights)
        for item in picks:
            stamp += 1
            rows.append((f"item{item:03d}", f"user{u:03d}", 5.0, stamp))

    path = tmp_path / "ratings.csv"
    pd.DataFrame(rows, columns=RATINGS_COLUMNS).to_csv(path, index=False, header=False)
    return path


class TestLoadBenchmark:
    def test_builds_instances_of_the_requested_shape(self, ratings_csv):
        bench = load_benchmark(ratings_csv, n_users=10, n_candidates=15, k=4, n_groups=4)

        assert len(bench.instances) == 10
        for inst in bench.instances:
            assert inst.n == 15
            assert inst.k == 4
            assert inst.relevance.shape == (15,)
            assert inst.similarity.shape == (15, 15)
            assert set(np.unique(inst.groups)) <= {0, 1, 2, 3}

    def test_candidates_arrive_sorted_by_descending_relevance(self, ratings_csv):
        """The README states present() is a no-op on real data because of this.

        If the loader ever stopped sorting, that claim would silently become false and
        the QUBO methods would start being charged for an ordering they never chose.
        """
        bench = load_benchmark(ratings_csv, n_users=8, n_candidates=12, k=4)

        for inst in bench.instances:
            assert np.all(np.diff(inst.relevance) <= 1e-12)

    def test_relevance_is_rescaled_into_unit_range(self, ratings_csv):
        """lam and mu are only comparable across benchmarks if relevance is."""
        bench = load_benchmark(ratings_csv, n_users=8, n_candidates=12, k=4)

        for inst in bench.instances:
            assert inst.relevance.min() >= -1e-9
            assert inst.relevance.max() <= 1.0 + 1e-9

    def test_held_out_item_is_never_in_the_training_signal(self, ratings_csv):
        """Ground truth indices must point inside the candidate set, or nowhere."""
        bench = load_benchmark(ratings_csv, n_users=10, n_candidates=15, k=4)

        for inst, relevant in zip(bench.instances, bench.relevant, strict=True):
            for idx in relevant:
                assert 0 <= idx < inst.n

    def test_candidate_hit_rate_matches_the_labels(self, ratings_csv):
        bench = load_benchmark(ratings_csv, n_users=12, n_candidates=15, k=4)
        expected = sum(1 for r in bench.relevant if r) / len(bench.relevant)

        assert bench.candidate_hit_rate == pytest.approx(expected)
        assert 0.0 <= bench.candidate_hit_rate <= 1.0

    def test_weights_are_carried_onto_every_instance(self, ratings_csv):
        bench = load_benchmark(ratings_csv, n_users=5, n_candidates=12, k=4, lam=2.5, mu=1.5)

        assert all(i.lam == 2.5 and i.mu == 1.5 for i in bench.instances)

    def test_seed_is_reproducible(self, ratings_csv):
        a = load_benchmark(ratings_csv, n_users=6, n_candidates=12, k=4, seed=3)
        b = load_benchmark(ratings_csv, n_users=6, n_candidates=12, k=4, seed=3)

        assert [i.item_ids for i in a.instances] == [i.item_ids for i in b.instances]

    def test_stats_report_the_filtering(self, ratings_csv):
        bench = load_benchmark(ratings_csv, n_users=6, n_candidates=12, k=4)

        assert bench.stats["raw_interactions"] >= bench.stats["core_interactions"]
        assert bench.stats["items"] > 0
        assert 0.0 < bench.stats["density"] <= 1.0


class TestSuggestLam:
    def test_returns_a_positive_scale(self, ratings_csv):
        bench = load_benchmark(ratings_csv, n_users=8, n_candidates=12, k=4)

        assert suggest_lam(bench.instances) > 0.0

    def test_empty_input_falls_back_rather_than_dividing_by_zero(self):
        assert suggest_lam([]) == 1.0


# --------------------------------------------------------- sparse similarity path


class TestSparseSimilarity:
    """The sparse path must be a drop-in replacement, not an approximation.

    It exists because Amazon Digital Music's 11,269-item catalogue needs a ~1 GB dense
    similarity matrix. That only helps if the two paths agree: a "scalable" variant that
    quietly returns different numbers converts a memory problem into a correctness one.
    """

    def test_matches_the_dense_matrix_exactly(self, ratings_csv):
        train, _ = leave_one_out(k_core(load_ratings(ratings_csv), 5))
        matrix, _, _ = interaction_matrix(train)

        dense = cosine_similarity(matrix, shrink=100.0)
        sparse_sim = cosine_similarity_sparse(matrix, shrink=100.0)

        assert np.allclose(dense, np.asarray(sparse_sim.todense()), atol=0.0, rtol=0.0)

    def test_matches_dense_without_shrink_too(self, ratings_csv):
        train, _ = leave_one_out(k_core(load_ratings(ratings_csv), 5))
        matrix, _, _ = interaction_matrix(train)

        dense = cosine_similarity(matrix, shrink=0.0)
        sparse_sim = cosine_similarity_sparse(matrix, shrink=0.0)

        assert np.abs(dense - np.asarray(sparse_sim.todense())).max() == 0.0

    def test_stores_far_less_than_the_dense_matrix(self, ratings_csv):
        """The whole point: the saving has to be real, not notional."""
        train, _ = leave_one_out(k_core(load_ratings(ratings_csv), 5))
        matrix, _, _ = interaction_matrix(train)

        sparse_sim = cosine_similarity_sparse(matrix, shrink=100.0)
        n = sparse_sim.shape[0]

        assert sparse_sim.nnz < n * n

    def test_diagonal_is_zeroed(self, ratings_csv):
        train, _ = leave_one_out(k_core(load_ratings(ratings_csv), 5))
        matrix, _, _ = interaction_matrix(train)

        sparse_sim = cosine_similarity_sparse(matrix, shrink=100.0)

        assert np.allclose(sparse_sim.diagonal(), 0.0)

    def test_top_k_agrees_with_the_dense_path(self, ratings_csv):
        """Ties at the k-th neighbour are common, so both paths break them the same way.

        Without a shared rule each returns a valid but different top-k, and which
        representation was used would silently change downstream candidate sets.
        """
        train, _ = leave_one_out(k_core(load_ratings(ratings_csv), 5))
        matrix, _, _ = interaction_matrix(train)

        dense = top_k_neighbours(cosine_similarity(matrix, 100.0), topk=20)
        sparse_top = top_k_neighbours_sparse(cosine_similarity_sparse(matrix, 100.0), topk=20)

        assert np.array_equal(dense, np.asarray(sparse_top.todense()))

    def test_top_k_keeps_at_most_k_per_row(self, ratings_csv):
        train, _ = leave_one_out(k_core(load_ratings(ratings_csv), 5))
        matrix, _, _ = interaction_matrix(train)

        truncated = top_k_neighbours_sparse(cosine_similarity_sparse(matrix, 100.0), topk=5)

        assert np.diff(truncated.indptr).max() <= 5

    def test_benchmarks_from_either_path_are_identical(self, ratings_csv):
        """End to end: same instances, same relevance, same candidate sets."""
        dense_bench = load_benchmark(ratings_csv, n_users=8, n_candidates=12, k=4,
                                     sparse_similarity=False)
        sparse_bench = load_benchmark(ratings_csv, n_users=8, n_candidates=12, k=4,
                                      sparse_similarity=True)

        assert len(dense_bench.instances) == len(sparse_bench.instances)
        for a, b in zip(dense_bench.instances, sparse_bench.instances, strict=True):
            assert a.item_ids == b.item_ids
            assert np.allclose(a.relevance, b.relevance)
            assert np.allclose(a.similarity, b.similarity)
        assert dense_bench.relevant == sparse_bench.relevant

    def test_limit_selects_the_path_automatically(self, ratings_csv):
        """A byte budget of zero forces sparse; a huge one forces dense. Both must run
        and agree, which is what makes the automatic choice safe to leave on."""
        forced_sparse = load_benchmark(ratings_csv, n_users=5, n_candidates=10, k=3,
                                       dense_similarity_limit=0)
        forced_dense = load_benchmark(ratings_csv, n_users=5, n_candidates=10, k=3,
                                      dense_similarity_limit=10**12)

        for a, b in zip(forced_sparse.instances, forced_dense.instances, strict=True):
            assert np.allclose(a.similarity, b.similarity)

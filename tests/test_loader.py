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
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.loader import (  # noqa: E402
    cosine_similarity,
    interaction_matrix,
    k_core,
    leave_one_out,
    popularity_tiers,
    top_k_neighbours,
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
        train, test = leave_one_out(ratings)

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

        train_pairs = set(zip(train.user_id, train.item_id, train.timestamp))
        test_pairs = set(zip(test.user_id, test.item_id, test.timestamp))
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
        thin = [("t%d" % u, item, 1.0, u) for u in range(2) for item in ["a", "b"]]
        thick = [("k%d" % u, item, 1.0, u) for u in range(200) for item in ["c", "d"]]
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

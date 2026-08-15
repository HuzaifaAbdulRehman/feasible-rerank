"""Tests for the reporting metrics in ``qubo_rerank/metrics/``.

These are the units every claim in the README is denominated in, and they were the
last part of the codebase without direct tests -- which is backwards, because a solver
bug shows up as a bad number while a *metric* bug shows up as a good one.

Two of the assertions here pin values the write-up quotes by name: the arithmetic floor
of ``exposure_parity`` at 6 groups / k=10 (0.2667) and at 4 groups / k=10 (0.2). The
README calls those floors, and reads results against them -- "μ turning 0 to 1 moves
parity to 0.199, its arithmetic floor". If the metric were rescaled, those sentences
would quietly become wrong, and nothing else in the suite would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from qubo_rerank.metrics.fairness import (
    catalogue_coverage,
    category_coverage,
    exposure_parity,
    gini,
    group_counts,
    intra_list_similarity,
    item_exposure,
)
from qubo_rerank.metrics.relevance import (
    dcg,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestDCG:
    def test_log2_position_discount(self):
        # gains 3, 2 at ranks 1, 2 -> 3/log2(2) + 2/log2(3)
        assert dcg([3.0, 2.0]) == pytest.approx(3.0 + 2.0 / np.log2(3.0))

    def test_is_order_sensitive(self):
        """The whole reason ``present()`` exists in run_experiment.py."""
        assert dcg([1.0, 0.0]) > dcg([0.0, 1.0])

    def test_empty_is_zero(self):
        assert dcg([]) == 0.0


class TestNDCG:
    def test_greedy_topk_scores_exactly_one(self):
        """True by construction: the ideal ranking is drawn from the same candidates."""
        relevance = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        assert ndcg_at_k(relevance, [0, 1, 2]) == pytest.approx(1.0)

    def test_worst_selection_scores_lowest(self):
        relevance = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        assert ndcg_at_k(relevance, [3, 4]) < ndcg_at_k(relevance, [0, 1])

    def test_reordering_the_same_set_changes_the_score(self):
        """The 0.09-NDCG effect that ``present()`` corrects, in miniature."""
        relevance = np.array([1.0, 0.1])
        assert ndcg_at_k(relevance, [0, 1]) > ndcg_at_k(relevance, [1, 0])

    def test_all_zero_relevance_is_zero_not_nan(self):
        assert ndcg_at_k(np.zeros(4), [0, 1]) == 0.0


class TestRecallAndPrecision:
    def test_recall_is_over_relevant_items(self):
        assert recall_at_k([0, 1, 2], {1, 5}) == pytest.approx(0.5)

    def test_precision_is_over_the_list(self):
        assert precision_at_k([0, 1, 2], {1, 5}) == pytest.approx(1 / 3)

    def test_k_truncates_before_scoring(self):
        assert recall_at_k([9, 1], {1}, k=1) == 0.0
        assert recall_at_k([9, 1], {1}, k=2) == 1.0

    def test_no_relevant_items_is_zero_not_nan(self):
        assert recall_at_k([0, 1], set()) == 0.0


class TestGroupExposure:
    def test_counts_cover_every_group_not_just_selected_ones(self):
        """A group absent from the list must count as 0, not be missing from the dict.

        If unselected groups were dropped, the parity targets below would be computed
        against the number of groups *present*, and a list concentrated in one group
        would score as perfect parity -- inverting the metric.
        """
        groups = np.array([0, 0, 1, 1, 2, 2])
        counts = group_counts(groups, [0, 1])

        assert counts == {0: 2, 1: 0, 2: 0}

    def test_concentration_is_the_worst_parity(self):
        groups = np.repeat(np.arange(4), 5)
        concentrated = exposure_parity(groups, list(range(4)))
        spread = exposure_parity(groups, [0, 5, 10, 15])

        assert spread == pytest.approx(0.0)
        assert concentrated > spread

    def test_floor_at_six_groups_and_k_ten(self):
        """0.2667 -- the value the README quotes for the synthetic benchmark.

        k=10 over 6 groups cannot divide evenly: four groups take 2 slots and two take
        1, against a target of 10/6. That residue is the floor, and no solver can beat
        it however good it is.
        """
        groups = np.repeat(np.arange(6), 5)
        best = [0, 1, 5, 6, 10, 11, 15, 16, 20, 25]  # 2,2,2,2,1,1

        assert exposure_parity(groups, best) == pytest.approx(0.2667, abs=1e-4)

    def test_floor_at_four_groups_and_k_ten(self):
        """0.2 -- the floor the README reads the Amazon μ-sweep against."""
        groups = np.repeat(np.arange(4), 5)
        best = [0, 1, 2, 5, 6, 7, 10, 11, 15, 16]  # 3,3,2,2

        assert exposure_parity(groups, best) == pytest.approx(0.2, abs=1e-9)

    def test_category_coverage_is_a_fraction_of_all_groups(self):
        groups = np.repeat(np.arange(4), 5)

        assert category_coverage(groups, [0, 5]) == pytest.approx(0.5)
        assert category_coverage(groups, [0, 5, 10, 15]) == pytest.approx(1.0)


class TestGini:
    def test_equal_exposure_is_zero(self):
        assert gini([4.0, 4.0, 4.0, 4.0]) == pytest.approx(0.0)

    def test_concentration_approaches_one(self):
        assert gini([0.0] * 999 + [1.0]) > 0.99

    def test_ordering_of_input_does_not_matter(self):
        values = [7.0, 1.0, 3.0, 0.0, 12.0]
        assert gini(values) == pytest.approx(gini(list(reversed(values))))

    def test_all_zero_is_zero_not_nan(self):
        """A catalogue nobody was shown must not report NaN into the results CSV."""
        assert gini([0.0, 0.0, 0.0]) == 0.0

    def test_empty_is_zero(self):
        assert gini([]) == 0.0


class TestCatalogueLevel:
    def test_exposure_counts_repeat_recommendations(self):
        exposure = item_exposure([[0, 1], [0, 2]], n_items=4)
        assert exposure.tolist() == [2.0, 1.0, 1.0, 0.0]

    def test_coverage_counts_distinct_items_only(self):
        assert catalogue_coverage([[0, 1], [0, 1]], n_items=4) == pytest.approx(0.5)
        assert catalogue_coverage([[0, 1], [2, 3]], n_items=4) == pytest.approx(1.0)

    def test_coverage_of_empty_catalogue_is_zero(self):
        assert catalogue_coverage([[0]], n_items=0) == 0.0


class TestIntraListSimilarity:
    def test_mean_over_distinct_pairs(self):
        similarity = np.array([[0.0, 0.4, 0.6], [0.4, 0.0, 0.8], [0.6, 0.8, 0.0]])
        # pairs (0,1), (0,2), (1,2) -> mean of 0.4, 0.6, 0.8
        assert intra_list_similarity(similarity, [0, 1, 2]) == pytest.approx(0.6)

    def test_diverse_list_scores_lower(self):
        similarity = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.1], [0.1, 0.1, 0.0]])
        assert intra_list_similarity(similarity, [0, 2]) < intra_list_similarity(
            similarity, [0, 1]
        )

    def test_single_item_list_has_no_pairs(self):
        assert intra_list_similarity(np.zeros((3, 3)), [1]) == 0.0

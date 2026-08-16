"""Tests for the ALS retrieval model.

This model exists to answer one question: are the fairness results a property of the
*method*, or of ItemKNN's particular bias? That only works if the ALS implementation is
actually ALS. A subtly wrong factorisation still produces plausible recommendations --
popular items still rank highly, because they rank highly under almost any model -- so
the failure would be invisible in the downstream metrics and would quietly answer the
question with a second copy of the first model's bias.

So the tests check the properties that distinguish a working matrix factorisation from a
popularity predictor: that reconstruction actually improves over iterations, that the
factorisation is low-rank, that observed interactions score above unobserved ones, and
that the model is genuinely different from ItemKNN rather than a re-parameterisation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.matrix_factorisation import (
    item_similarity,
    load_mf_benchmark,
    train_als,
)

AMAZON = ROOT / "data/amazon_lb/Luxury_Beauty.csv"


@pytest.fixture(scope="module")
def toy_matrix() -> sparse.csr_matrix:
    """Two clean user groups over two disjoint item blocks.

    Deliberately separable: a working factorisation must recover the block structure,
    and a broken one that just predicts popularity cannot, because both blocks are
    equally popular by construction.
    """
    rows, cols = [], []
    for user in range(20):
        block = range(0, 10) if user < 10 else range(10, 20)
        for item in block:
            rows.append(user)
            cols.append(item)
    return sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(20, 20)
    )


class TestTraining:
    def test_returns_factors_of_the_requested_rank(self, toy_matrix):
        users, items = train_als(toy_matrix, factors=8, iterations=3, seed=0)

        assert users.shape == (20, 8)
        assert items.shape == (20, 8)

    def test_recovers_block_structure(self, toy_matrix):
        """The property a popularity predictor cannot have.

        Both blocks have identical popularity, so any model that scores by popularity
        alone ranks them equally. A real factorisation must score a user's own block
        above the other.
        """
        users, items = train_als(toy_matrix, factors=8, iterations=15, seed=0)
        scores = users @ items.T

        own = scores[:10, :10].mean()
        other = scores[:10, 10:].mean()
        assert own > other

    def test_converges_to_a_close_fit(self, toy_matrix):
        """Asserts convergence rather than monotone improvement, deliberately.

        The first version of this test required error(10) < error(1) and failed -- not
        because ALS was broken but because this matrix is trivially separable, so ALS
        reaches ~1e-6 reconstruction error after a *single* iteration and the remaining
        differences are floating-point noise. Requiring improvement where none is
        available tests the fixture, not the solver.
        """
        users, items = train_als(toy_matrix, factors=8, iterations=10, seed=0)
        dense = np.asarray(toy_matrix.todense())
        error = np.abs(dense - users @ items.T)[dense > 0].mean()

        assert error < 0.01

    def test_beats_a_rank_one_fit_on_structured_data(self):
        """The property that separates a factorisation from a popularity predictor.

        A rank-1 model can only express "some items are popular". Two disjoint user
        groups need at least rank 2, so if extra factors bought nothing the model would
        be predicting popularity under another name.
        """
        rows, cols = [], []
        for user in range(20):
            for item in (range(0, 10) if user < 10 else range(10, 20)):
                rows.append(user)
                cols.append(item)
        matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(20, 20))
        dense = np.asarray(matrix.todense())

        def error(factors: int) -> float:
            users, items = train_als(matrix, factors=factors, iterations=10, seed=0)
            return float(np.abs(dense - users @ items.T).mean())

        assert error(8) < error(1)

    def test_is_deterministic_under_a_seed(self, toy_matrix):
        a = train_als(toy_matrix, factors=8, iterations=3, seed=7)[1]
        b = train_als(toy_matrix, factors=8, iterations=3, seed=7)[1]

        assert np.allclose(a, b)

    def test_handles_users_with_no_interactions(self):
        """k-core should prevent this, but a solve that divides by zero here would
        surface as a NaN score much later and be attributed to something else."""
        matrix = sparse.csr_matrix((np.ones(3), ([0, 0, 1], [0, 1, 2])), shape=(4, 4))
        users, items = train_als(matrix, factors=4, iterations=3, seed=0)

        assert np.isfinite(users).all()
        assert np.isfinite(items).all()


class TestItemSimilarity:
    def test_is_symmetric_with_zero_diagonal(self):
        factors = np.random.default_rng(0).standard_normal((30, 8))
        similarity = item_similarity(factors)

        assert np.allclose(similarity, similarity.T)
        assert np.allclose(np.diag(similarity), 0.0)

    def test_is_never_negative(self):
        """Latent factors are unconstrained in sign, so raw cosine can be negative -- and
        a negative entry in the diversity term would *reward* redundancy rather than
        penalise it, silently inverting what lam means."""
        factors = np.random.default_rng(1).standard_normal((40, 6))

        assert (item_similarity(factors) >= 0.0).all()

    def test_identical_factors_are_maximally_similar(self):
        factors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        similarity = item_similarity(factors)

        assert similarity[0, 1] == pytest.approx(1.0)
        assert similarity[0, 2] == pytest.approx(0.0)


class TestBenchmark:
    @pytest.fixture(scope="class")
    def bench(self):
        if not AMAZON.exists():
            pytest.skip(f"{AMAZON} not downloaded; see configs/amazon_lb.yaml")
        return load_mf_benchmark(AMAZON, n_users=12, n_candidates=40, k=5,
                                 factors=16, iterations=5, seed=0)

    def test_produces_instances_of_the_requested_shape(self, bench):
        assert len(bench.instances) <= 12
        for inst in bench.instances:
            assert inst.n == 40
            assert inst.k == 5
            assert inst.similarity.shape == (40, 40)

    def test_candidates_arrive_sorted_by_relevance(self, bench):
        """present() is a no-op on this benchmark for the same reason as ItemKNN's."""
        for inst in bench.instances:
            assert np.all(np.diff(inst.relevance) <= 1e-12)

    def test_relevance_is_rescaled_to_unit_range(self, bench):
        for inst in bench.instances:
            assert inst.relevance.min() >= -1e-9
            assert inst.relevance.max() <= 1.0 + 1e-9

    def test_profile_items_are_never_recommended(self, bench):
        """Scores for held items are set to -inf; if that leaked, the model would score
        brilliantly by recommending what the user already has."""
        assert all(np.isfinite(inst.relevance).all() for inst in bench.instances)

    def test_differs_from_itemknn(self, bench):
        """The whole point: a second model, not a re-parameterisation of the first.

        ALS similarity is dense (latent factors give nearly every pair a non-zero
        cosine) where shrunk-cosine ItemKNN is ~99% zeros. If these matched, swapping
        the retrieval model would not be testing anything.
        """
        density = np.mean([(inst.similarity > 0).mean() for inst in bench.instances])

        assert density > 0.5

"""Verify our ItemKNN against RecBole's reference implementation.

Every number this project reports is downstream of ItemKNN. It supplies the relevance
scores, the similarity matrix the diversity term reads, and the candidate sets everything
is evaluated on. The component tests in ``test_loader.py`` check the pieces -- cosine is
symmetric, top-k keeps k -- but passing those is compatible with computing a *different
model* that is internally consistent. The only way to know it is ItemKNN is to compare it
against an implementation nobody here wrote.

``ComputeSimilarity`` from ``recbole/model/general_recommender/itemknn.py`` is transcribed
below verbatim, minus the block-wise loop, which is a performance detail. RecBole is the
standard library for this model and is MIT-licensed. This is the same tactic used for the
DPFR fairness metrics in ``test_dpfr.py``: bring the reference in as an oracle rather than
re-deriving the algebra in a comment and hoping.

**What the comparison found.** The similarity values agree to 6e-08 -- float32 against our
float64, i.e. exactly. The truncated neighbour sets agree for 626 of 727 items, and for
all 101 that differ, *every* disagreeing entry sits at a single exactly-tied similarity
value. So the two implementations select different members of a tied group and are
otherwise identical.

That difference is deliberate on our side. RecBole resolves ties via ``argpartition``,
which follows memory order and so depends on how the matrix was built; ours sorts by
``(-value, column)``. Ties at the k-th neighbour are common -- 101 of 727 items here --
so a non-deterministic rule means the dense and sparse code paths can return different,
equally valid answers for the same input. See ``benchmarks/loader.top_k_neighbours``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.loader import (
    cosine_similarity,
    interaction_matrix,
    k_core,
    leave_one_out,
    load_ratings,
    top_k_neighbours,
)

AMAZON = ROOT / "data/amazon_Software/Software.csv"


def recbole_compute_similarity(
    data_matrix: sp.csr_matrix, topk: int = 100, shrink: float = 0.0
) -> sp.csr_matrix:
    """RecBole's ``ComputeSimilarity(method='item').compute_similarity()``, transcribed.

    Faithful to the original including the float32 cast, the ``+ 1e-6`` in the
    denominator, zeroing self-similarity before normalisation, and the
    argpartition-then-argsort top-k. The block loop is flattened to one column at a time,
    which changes nothing but readability.

    Returns ``W`` with ``W[i, j] != 0`` iff item ``i`` is among item ``j``'s top-k
    neighbours -- note the orientation, which is the transpose of ours.
    """
    data_matrix = data_matrix.astype(np.float32)
    n_columns = data_matrix.shape[1]
    top_k = min(topk, n_columns)

    sum_of_squared = np.sqrt(np.array(data_matrix.power(2).sum(axis=0)).ravel())

    values, rows, cols = [], [], []
    for index in range(n_columns):
        line = np.asarray(data_matrix.T.dot(data_matrix[:, index].toarray())).ravel()
        line[index] = 0.0
        line = line / (sum_of_squared[index] * sum_of_squared + shrink + 1e-6)

        partition = (-line).argpartition(top_k - 1)[:top_k]
        top = partition[np.argsort(-line[partition])]

        not_zero = line[top] != 0.0
        values.extend(line[top][not_zero])
        rows.extend(top[not_zero])
        cols.extend(np.ones(int(not_zero.sum())) * index)

    return sp.csr_matrix(
        (values, (rows, cols)), shape=(n_columns, n_columns), dtype=np.float32
    )


@pytest.fixture(scope="module")
def matrix() -> sp.csr_matrix:
    if not AMAZON.exists():
        pytest.skip(f"{AMAZON} not downloaded; see configs/amazon_software.yaml")
    train, _ = leave_one_out(k_core(load_ratings(AMAZON), 5))
    return interaction_matrix(train)[0]


class TestAgainstRecBole:
    def test_similarity_values_match(self, matrix):
        """The formula itself. Tolerance is float32 resolution, not a fudge factor."""
        ours = cosine_similarity(matrix, shrink=100.0)
        reference = recbole_compute_similarity(matrix, topk=matrix.shape[1], shrink=100.0)

        stored = np.asarray(reference.todense()) != 0
        # RecBole stores W[i, j] for i in topk(j); untruncated that is the full transpose.
        assert np.abs(np.asarray(reference.todense())[stored] - ours.T[stored]).max() < 1e-6

    def test_shrink_zero_also_matches(self, matrix):
        """Shrinkage is a hyperparameter, not part of the identity being checked."""
        ours = cosine_similarity(matrix, shrink=0.0)
        reference = recbole_compute_similarity(matrix, topk=matrix.shape[1], shrink=0.0)

        stored = np.asarray(reference.todense()) != 0
        assert np.abs(np.asarray(reference.todense())[stored] - ours.T[stored]).max() < 1e-6

    def test_truncation_keeps_the_same_number_of_neighbours(self, matrix):
        ours = top_k_neighbours(cosine_similarity(matrix, shrink=100.0), topk=100)
        reference = recbole_compute_similarity(matrix, topk=100, shrink=100.0)

        assert np.count_nonzero(ours) == reference.nnz

    def test_every_neighbour_disagreement_is_a_tie(self, matrix):
        """The load-bearing test: differences must be ties, never ranking errors.

        A single item whose disagreement is *not* at a tied value would mean the two
        implementations rank neighbours differently -- a real bug, and one that would
        silently change every candidate set downstream.
        """
        full = cosine_similarity(matrix, shrink=100.0)
        ours = top_k_neighbours(full, topk=100)
        reference = np.asarray(recbole_compute_similarity(matrix, 100, 100.0).todense())

        differing = 0
        for item in range(reference.shape[0]):
            their_neighbours = set(np.flatnonzero(reference[:, item]))
            our_neighbours = set(np.flatnonzero(ours[item, :]))
            if their_neighbours == our_neighbours:
                continue

            differing += 1
            disagreement = their_neighbours ^ our_neighbours
            values = {round(float(full[item, n]), 12) for n in disagreement}
            assert len(values) == 1, (
                f"item {item}: neighbour sets differ at {len(values)} distinct "
                f"similarity values, so this is a ranking difference rather than a tie"
            )

        # Recorded rather than asserted exactly: it is a property of this dataset, and
        # pinning it would make the test fail for an uninteresting reason if the data
        # were ever refiltered.
        assert differing < reference.shape[0] // 2

    def test_candidate_sets_largely_agree(self, matrix):
        """End to end: the retrieved shortlist is what actually reaches the QUBO.

        Not asserting exact equality -- tie-breaking moves a few items in and out at the
        boundary, which is expected and harmless. Asserting the overlap stays high is the
        meaningful check: a genuine formula error would collapse it.
        """
        ours = top_k_neighbours(cosine_similarity(matrix, shrink=100.0), topk=100)
        reference = recbole_compute_similarity(matrix, 100, 100.0)

        overlaps = []
        for user in range(min(25, matrix.shape[0])):
            profile = matrix[user]
            our_scores = np.asarray(profile @ ours).ravel()
            their_scores = np.asarray((profile @ reference).todense()).ravel()
            top_ours = set(np.argsort(-our_scores)[:200])
            top_theirs = set(np.argsort(-their_scores)[:200])
            overlaps.append(len(top_ours & top_theirs) / 200)

        assert np.mean(overlaps) > 0.85

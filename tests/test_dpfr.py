"""Verify the DPFR metric port against the reference implementation.

``qubo_rerank/metrics/dpfr.py`` reimplements three measures from Rampisela et al. so
the project does not have to depend on RecBole. A reimplementation of someone else's
published metric is only worth citing if it actually computes their metric, so the
reference formulas are transcribed here -- straight from
``reference/DPFR-fairness-eval/RecBole/recbole/evaluator/metrics.py``, keeping their
1-based item indexing and their variable names -- and the two are required to agree.

If this file fails after a change to ``dpfr.py``, the fix is to ``dpfr.py``. The
transcription below is not ours to "correct".
"""

from __future__ import annotations

import numpy as np
import pytest

from qubo_rerank.metrics.dpfr import (
    GAMMA,
    individual_item_fairness,
    item_better_off,
)


def reference_iif_aif(item_matrix: np.ndarray, num_items: int, pos_items: list, gamma: float = 0.8):
    """Transcribed from ``IIF_AIF.get_metrics``. 1-based item ids, as in the original."""
    rec = item_matrix
    rel = pos_items
    m = rec.shape[0]

    user_item_rel = np.zeros((m, num_items))
    rank_matrix = np.zeros_like(user_item_rel)
    for i in range(len(rec)):
        rank_matrix[i][rec[i] - 1] = np.where(rec[i])[0] + 1

    user_item_exp_rbp = np.copy(rank_matrix)
    user_item_exp_rbp[user_item_exp_rbp.nonzero()] = gamma ** (
        user_item_exp_rbp[user_item_exp_rbp.nonzero()] - 1
    )

    for i in range(len(rel)):
        user_item_rel[i][np.asarray(rel[i]) - 1] = 1

    r_u_star = user_item_rel.sum(1)[:, np.newaxis]
    e_ui_star = user_item_rel * (1 - np.power(gamma, r_u_star)) / (1 - gamma)
    r_u_star[r_u_star == 0] = 1
    e_ui_star /= r_u_star

    diff = user_item_exp_rbp - e_ui_star
    return np.power(diff, 2).mean(), np.power(diff.mean(0), 2).mean()


def reference_ibo(item_matrix: np.ndarray, num_items: int, pos_items: list) -> float:
    """Transcribed from ``IBO.get_IBOIWO``, the ``IBO_our`` branch."""
    rec = item_matrix
    rel = pos_items
    m = rec.shape[0]
    k = rec.shape[1]
    inv = 1 / (np.arange(k, dtype="int") + 1)

    user_item_rel = np.zeros((m, num_items))
    rank_matrix = np.zeros_like(user_item_rel)
    for i in range(len(rec)):
        rank_matrix[i][rec[i] - 1] = np.where(rec[i])[0] + 1

    user_item_exp_inv = np.copy(rank_matrix)
    user_item_exp_inv[user_item_exp_inv.nonzero()] = (
        1 / user_item_exp_inv[user_item_exp_inv.nonzero()]
    )

    for i in range(len(rel)):
        user_item_rel[i][np.asarray(rel[i]) - 1] = 1

    imp_i_arr = (user_item_exp_inv * user_item_rel).sum(0) / m
    imp_unif_arr = user_item_rel.sum(0) * inv.sum() / num_items / m

    mask = imp_unif_arr != 0
    return float((imp_i_arr[mask] >= 1.1 * imp_unif_arr[mask]).mean())


@pytest.fixture
def scenario():
    """Random lists over a small catalogue, with one relevant item per user.

    Item ids are generated 1-based to suit the reference code; the port is handed the
    same data shifted to 0-based, which is the only difference between the two calls.
    """
    rng = np.random.default_rng(11)
    n_users, n_items, k = 25, 40, 8

    rec_1based = np.array(
        [rng.choice(np.arange(1, n_items + 1), size=k, replace=False) for _ in range(n_users)]
    )
    pos_1based = [np.array([rng.integers(1, n_items + 1)]) for _ in range(n_users)]

    selections = [[int(i) - 1 for i in row] for row in rec_1based]
    relevant = [{int(p[0]) - 1} for p in pos_1based]

    return rec_1based, pos_1based, selections, relevant, n_items


class TestPortMatchesReference:
    def test_iif_and_aif(self, scenario):
        rec, pos, selections, relevant, n_items = scenario
        ref_iif, ref_aif = reference_iif_aif(rec, n_items, pos, gamma=GAMMA)
        iif, aif = individual_item_fairness(selections, relevant, n_items)
        assert iif == pytest.approx(ref_iif, rel=1e-12)
        assert aif == pytest.approx(ref_aif, rel=1e-12)

    def test_ibo(self, scenario):
        rec, pos, selections, relevant, n_items = scenario
        assert item_better_off(selections, relevant, n_items) == pytest.approx(
            reference_ibo(rec, n_items, pos), rel=1e-12
        )


class TestBehaviour:
    def test_perfect_ranking_is_fairer_than_worst_ranking(self):
        """Putting each user's relevant item first must beat burying it last."""
        n_users, n_items, k = 12, 30, 5
        relevant = [{u} for u in range(n_users)]

        best = [[u] + [i for i in range(n_items) if i != u][: k - 1] for u in range(n_users)]
        worst = [[i for i in range(n_items) if i != u][: k - 1] + [u] for u in range(n_users)]

        best_iif, _ = individual_item_fairness(best, relevant, n_items)
        worst_iif, _ = individual_item_fairness(worst, relevant, n_items)
        assert best_iif < worst_iif

    def test_users_without_relevant_items_contribute_no_target(self):
        """A user with no held-out item still receives exposure but has no target.

        Half the users on the real benchmark are in exactly this position, so this
        path is not an edge case here -- it is the common case.
        """
        n_items = 20
        selections = [[0, 1, 2], [3, 4, 5]]
        iif, _ = individual_item_fairness(selections, [set(), set()], n_items)

        # Every target is zero, so II-F is just the mean squared exposure.
        expected = sum(GAMMA ** (2 * r) for r in range(3)) * 2 / (2 * n_items)
        assert iif == pytest.approx(expected)

    def test_ibo_returns_zero_when_nothing_is_relevant(self):
        assert item_better_off([[0, 1], [2, 3]], [set(), set()], 10) == 0.0

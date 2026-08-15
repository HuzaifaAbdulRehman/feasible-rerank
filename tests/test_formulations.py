"""Correctness tests for the QUBO formulation.

These are the gate described in the plan: cardinality must hold, mu=0 must reduce to
the fairness-free problem, and a hand-checkable instance must return the known
optimum. If any of these fail, every downstream metric is measuring the wrong thing.
"""

from __future__ import annotations

import dimod
import numpy as np
import pytest

from qubo_rerank.formulations import (
    build_cardinality,
    build_fairness,
    build_objective,
    build_problem,
    exposure_targets_proportional,
)


def brute_force(bqm: dimod.BinaryQuadraticModel, n: int) -> tuple[list[int], float]:
    """Exhaustively minimise a small BQM. Only tractable for n <= ~20."""
    best_sel: list[int] = []
    best_energy = float("inf")
    for mask in range(1 << n):
        sample = {i: (mask >> i) & 1 for i in range(n)}
        e = bqm.energy(sample)
        if e < best_energy:
            best_energy = e
            best_sel = sorted(i for i in range(n) if sample[i] == 1)
    return best_sel, best_energy


class TestCardinality:
    def test_exactly_k_is_the_minimum(self):
        n, k = 6, 3
        bqm = build_cardinality(n, k, strength=1.0)
        sel, _ = brute_force(bqm, n)
        assert len(sel) == k

    def test_all_k_subsets_are_degenerate(self):
        """No subset of size k should be preferred over another by the constraint."""
        n, k = 5, 2
        bqm = build_cardinality(n, k, strength=1.0)
        energies = []
        for i in range(n):
            for j in range(i + 1, n):
                sample = {v: 0 for v in range(n)}
                sample[i] = sample[j] = 1
                energies.append(bqm.energy(sample))
        assert np.allclose(energies, energies[0])

    def test_rejects_impossible_k(self):
        with pytest.raises(ValueError):
            build_cardinality(3, 5)


class TestObjective:
    def test_relevance_only_picks_highest_scores(self):
        """With lam=0 the objective alone must rank by relevance."""
        rel = np.array([0.1, 0.9, 0.5, 0.2])
        sim = np.zeros((4, 4))
        bqm = build_objective(rel, sim, lam=0.0)
        # Linear coefficients are negated relevance, so the most relevant item has
        # the most negative coefficient.
        lin = [bqm.get_linear(i) for i in range(4)]
        assert int(np.argmin(lin)) == 1

    def test_diversity_penalises_similar_pairs(self):
        rel = np.array([1.0, 1.0, 1.0])
        sim = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        bqm = build_objective(rel, sim, lam=1.0, normalize_inputs=False)
        assert bqm.get_quadratic(0, 1) > 0
        with pytest.raises(ValueError):
            bqm.get_quadratic(0, 2)  # no coupler for a zero-similarity pair

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_objective(np.zeros(4), np.zeros((3, 3)))


class TestFairness:
    def test_mu_zero_is_a_no_op(self):
        groups = np.array([0, 0, 1, 1])
        bqm = build_fairness(groups, k=2, mu=0.0)
        assert all(v == 0.0 for v in bqm.linear.values())
        assert len(bqm.quadratic) == 0

    def test_penalises_same_group_pairs(self):
        groups = np.array([0, 0, 1, 1])
        bqm = build_fairness(groups, k=2, mu=1.0)
        assert bqm.get_quadratic(0, 1) > 0  # same group -> penalised
        with pytest.raises(ValueError):
            bqm.get_quadratic(0, 2)  # different groups -> no coupler

    def test_alone_prefers_balanced_selection(self):
        """The fairness term by itself should split the list across groups."""
        groups = np.array([0, 0, 0, 1, 1, 1])
        bqm = build_fairness(groups, k=2, mu=1.0)
        balanced = {0: 1, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0}   # one per group
        lopsided = {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0}   # both from group 0
        assert bqm.energy(balanced) < bqm.energy(lopsided)

    def test_proportional_targets_sum_to_k(self):
        groups = np.array([0, 0, 0, 1])
        targets = exposure_targets_proportional(groups, k=4)
        assert pytest.approx(sum(targets.values())) == 4.0
        assert targets[0] > targets[1]  # group 0 is larger


class TestBuildProblem:
    def test_returns_exactly_k_under_brute_force(self):
        rng = np.random.default_rng(0)
        n, k = 8, 3
        rel = rng.random(n)
        sim = rng.random((n, n))
        sim = (sim + sim.T) / 2
        np.fill_diagonal(sim, 0.0)

        rp = build_problem(rel, sim, k=k, lam=0.5)
        sel, _ = brute_force(rp.bqm, n)
        assert len(sel) == k

    def test_mu_zero_matches_no_groups(self):
        """Passing mu=0 with groups must equal passing no groups at all."""
        rng = np.random.default_rng(1)
        n, k = 7, 3
        rel = rng.random(n)
        sim = rng.random((n, n))
        sim = (sim + sim.T) / 2
        np.fill_diagonal(sim, 0.0)
        groups = np.array([0, 0, 1, 1, 2, 2, 2])

        a = build_problem(rel, sim, k=k, lam=0.5, groups=groups, mu=0.0)
        b = build_problem(rel, sim, k=k, lam=0.5, groups=None)
        assert brute_force(a.bqm, n)[0] == brute_force(b.bqm, n)[0]

    def test_known_optimum_by_hand(self):
        """A six-item instance whose answer can be checked without a solver.

        Items 0-2 are highly relevant but all in group 0 and mutually identical.
        Items 3-5 are less relevant, in group 1, and mutually dissimilar. With no
        fairness pressure the optimum takes the relevant cluster; with strong
        fairness pressure it must reach into group 1.
        """
        rel = np.array([1.0, 0.95, 0.9, 0.4, 0.35, 0.3])
        sim = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                if i != j:
                    sim[i, j] = 1.0
        groups = np.array([0, 0, 0, 1, 1, 1])

        no_fair = build_problem(rel, sim, k=2, groups=groups, lam=0.0, mu=0.0)
        sel_nf, _ = brute_force(no_fair.bqm, 6)
        assert sel_nf == [0, 1], "without fairness, take the two most relevant"

        strong_fair = build_problem(rel, sim, k=2, groups=groups, lam=0.0, mu=5.0)
        sel_f, _ = brute_force(strong_fair.bqm, 6)
        assert len({int(groups[i]) for i in sel_f}) == 2, (
            "with strong fairness, the list must span both groups"
        )

    def test_energy_breakdown_sums_to_total(self):
        rng = np.random.default_rng(2)
        n, k = 6, 2
        rel = rng.random(n)
        sim = rng.random((n, n))
        sim = (sim + sim.T) / 2
        np.fill_diagonal(sim, 0.0)
        groups = np.array([0, 0, 1, 1, 2, 2])

        rp = build_problem(rel, sim, k=k, groups=groups, lam=0.5, mu=1.0, normalize=False)
        selection = [0, 3]
        parts = rp.energy_breakdown(selection)

        sample = {i: (1 if i in selection else 0) for i in range(n)}
        assert pytest.approx(sum(parts.values()), rel=1e-9) == rp.bqm.energy(sample)

    def test_requires_groups_when_mu_nonzero(self):
        with pytest.raises(ValueError):
            build_problem(np.zeros(4), np.zeros((4, 4)), k=2, mu=1.0)

    def test_rejects_k_larger_than_n(self):
        with pytest.raises(ValueError):
            build_problem(np.zeros(3), np.zeros((3, 3)), k=5)

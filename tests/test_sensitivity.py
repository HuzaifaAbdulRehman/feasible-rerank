"""Tests for the solver-budget sensitivity experiment.

This experiment exists to answer one objection to the project's central claim: *you did
not run the annealer long enough*. That makes its correctness unusually load-bearing --
if the budget ladder silently failed to vary the budget, every run would produce the same
number and the flat curve would be an artefact of the harness rather than a finding.

So the tests here check the things that would make the conclusion wrong while leaving the
output looking entirely reasonable: that budgets actually reach the solvers, that the work
axis is monotone, and that energies are computed from the same BQM the solver was handed
rather than from a rebuilt one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.sensitivity import (
    barrier_instance,
    budgets,
    energy_of,
    make_solver,
)
from qubo_rerank.solvers import GreedyTopK


class TestBudgetLadders:
    @pytest.mark.parametrize("name", ["qubo_sa", "qubo_tabu", "qubo_feasible"])
    def test_ladder_spans_at_least_two_orders_of_magnitude(self, name):
        """A ladder that barely varies cannot answer 'was the budget big enough'."""
        work = [int(np.prod(list(p.values()))) for p in budgets(name)]

        assert max(work) / min(work) >= 100

    @pytest.mark.parametrize("name", ["qubo_sa", "qubo_tabu", "qubo_feasible"])
    def test_every_rung_is_distinct(self, name):
        rungs = [tuple(sorted(p.items())) for p in budgets(name)]

        assert len(set(rungs)) == len(rungs)

    def test_unknown_solver_is_rejected(self):
        with pytest.raises(ValueError):
            budgets("qubo_imaginary")


class TestSolverConstruction:
    def test_budgets_actually_reach_the_solver(self):
        """The failure that would fake the result.

        If parameters were dropped on the way to the solver, every rung would run at the
        default budget, every energy would be identical, and the flat curve would be
        pure harness artefact -- while looking exactly like the expected finding.
        """
        solver = make_solver("qubo_sa", {"num_reads": 37, "num_sweeps": 4321}, seed=0)

        assert solver.num_reads == 37
        assert solver.num_sweeps == 4321

    def test_tabu_ladder_uses_work_based_stopping(self):
        """Tabu defaults to a 20 ms wall-clock timeout. Left on, this experiment would
        measure the clock rather than the budget, and every rung would tie."""
        solver = make_solver("qubo_tabu", {"num_restarts": 50}, seed=0)

        assert solver.timeout is None
        assert solver.num_restarts == 50

    def test_feasible_ladder_sets_both_parameters(self):
        solver = make_solver("qubo_feasible", {"num_restarts": 4, "num_sweeps": 60}, seed=0)

        assert solver.num_restarts == 4
        assert solver.num_sweeps == 60

    def test_seeds_are_honoured(self):
        a = make_solver("qubo_feasible", {"num_restarts": 2, "num_sweeps": 20}, seed=5)
        b = make_solver("qubo_feasible", {"num_restarts": 2, "num_sweeps": 20}, seed=5)
        instance = barrier_instance(40, 5, lam=4.0, seed=0)

        assert a.solve(instance).selection == b.solve(instance).selection


class TestEnergyOf:
    def test_matches_the_solver_reported_energy(self):
        """Energies must come from the same BQM the solver minimised.

        Rebuilding the problem with different weights would produce a self-consistent
        table that compares solvers on an objective none of them optimised.
        """
        instance = barrier_instance(60, 8, lam=4.0, seed=0)
        solver = make_solver("qubo_feasible", {"num_restarts": 4, "num_sweeps": 60}, seed=0)
        result = solver.solve(instance)

        assert energy_of(instance, result.selection) == pytest.approx(
            result.stats["energy"], abs=1e-9
        )

    def test_greedy_is_optimal_when_diversity_is_switched_off(self):
        """At lam=0 the QUBO reduces to relevance, so greedy top-k *is* the optimum.

        Written this way after the naive version failed: at lam=4 greedy does NOT
        minimise the QUBO energy, because it maximises relevance while ignoring the
        diversity term. That is the project's whole premise, and a test asserting
        otherwise was asserting the thing the repo exists to disprove.
        """
        instance = barrier_instance(60, 8, lam=0.0, seed=1)
        greedy = GreedyTopK().solve(instance).selection

        rng = np.random.default_rng(0)
        for _ in range(20):
            arbitrary = sorted(rng.choice(instance.n, size=instance.k, replace=False))
            assert energy_of(instance, greedy) <= energy_of(instance, list(arbitrary)) + 1e-12

    def test_diversity_makes_greedy_suboptimal(self):
        """The premise itself, pinned: with lam large, some other list beats greedy."""
        instance = barrier_instance(60, 8, lam=4.0, seed=1)
        greedy_energy = energy_of(instance, GreedyTopK().solve(instance).selection)

        rng = np.random.default_rng(0)
        beaten = any(
            energy_of(instance, sorted(rng.choice(instance.n, instance.k, replace=False)))
            < greedy_energy
            for _ in range(50)
        )
        assert beaten, "greedy was not beaten by any random list; lam may be too small"


class TestBarrierInstance:
    def test_similarity_is_symmetric_with_zero_diagonal(self):
        instance = barrier_instance(50, 6, lam=4.0, seed=0)

        assert np.allclose(instance.similarity, instance.similarity.T)
        assert np.allclose(np.diag(instance.similarity), 0.0)

    def test_similarity_is_sparse_and_spiky_like_real_data(self):
        """The x**6 shaping is what makes this resemble an Amazon matrix rather than
        the synthetic generator's dense blocks. Mean similarity on Amazon is ~0.005."""
        instance = barrier_instance(200, 10, lam=4.0, seed=0)
        off_diagonal = instance.similarity[~np.eye(200, dtype=bool)]

        assert off_diagonal.mean() < 0.15
        assert off_diagonal.max() > 0.3

    def test_is_reproducible(self):
        a = barrier_instance(40, 5, lam=4.0, seed=3)
        b = barrier_instance(40, 5, lam=4.0, seed=3)

        assert np.array_equal(a.similarity, b.similarity)
        assert np.array_equal(a.relevance, b.relevance)

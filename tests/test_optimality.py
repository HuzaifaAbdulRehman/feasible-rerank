"""Tests for the exact-optimum experiment.

This experiment is the only place in the project with **ground truth**. Everywhere else
compares solvers to each other; here they are compared to the true minimum, obtained by
enumerating every feasible subset. That makes its correctness load-bearing in a way the
relative comparisons are not: a wrong reference does not produce a wrong *ranking*, it
produces wrong *magnitudes* -- "recovers 80% of the available improvement" -- which is
exactly the kind of number that gets quoted.

The first test is a regression test for a bug that was actually made here. ``dimod``
binary quadratic models carry a constant ``offset``, and the cardinality penalty
``P(sum(x) - k)^2`` expands to include ``P*k^2``, which lands in it. The first version of
``exact_optimum`` summed only the linear and quadratic terms. That shifts every enumerated
energy by the same constant, so the *ranking* of subsets -- and therefore the returned
optimal selection -- was entirely correct, while every reported gap-to-optimal was
nonsense. It was caught only because the gaps came out uniformly positive against a
negative optimum, which looked absurd. A subtler penalty weight would have hidden it.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.optimality import (
    exact_optimum,
    random_feasible_energy,
)
from experiments.sensitivity import barrier_instance, energy_of
from qubo_rerank.solvers import FeasibleAnnealing, TabuSearch


class TestExactOptimum:
    def test_optimum_agrees_with_the_solvers_energy_function(self):
        """The regression test for the dropped offset.

        If the enumeration and the scoring function disagree by a constant, the optimum
        is still the right *subset* while every gap is wrong by that constant.
        """
        instance = barrier_instance(16, 4, lam=4.0, seed=0)
        optimum, selection = exact_optimum(instance)

        assert optimum == pytest.approx(energy_of(instance, selection), abs=1e-9)

    def test_no_feasible_subset_beats_it(self):
        """Definitional, and cheap to verify exhaustively at this size."""
        instance = barrier_instance(14, 4, lam=4.0, seed=1)
        optimum, _ = exact_optimum(instance)

        for subset in itertools.combinations(range(instance.n), instance.k):
            assert energy_of(instance, list(subset)) >= optimum - 1e-9

    def test_returns_a_feasible_selection(self):
        instance = barrier_instance(15, 5, lam=4.0, seed=2)
        _, selection = exact_optimum(instance)

        assert len(set(selection)) == instance.k
        assert all(0 <= i < instance.n for i in selection)

    def test_is_deterministic(self):
        instance = barrier_instance(14, 4, lam=4.0, seed=3)

        assert exact_optimum(instance)[0] == exact_optimum(instance)[0]

    def test_fairness_term_changes_the_optimum(self):
        """Guards against mu being silently dropped on the way into the enumeration."""
        blind = barrier_instance(16, 4, lam=1.0, seed=4)
        blind.mu = 0.0
        fair = barrier_instance(16, 4, lam=1.0, seed=4)
        fair.mu = 8.0

        assert exact_optimum(blind)[1] != exact_optimum(fair)[1]


class TestRandomBaseline:
    def test_is_worse_than_the_optimum(self):
        instance = barrier_instance(16, 4, lam=4.0, seed=0)
        optimum, _ = exact_optimum(instance)

        assert random_feasible_energy(instance, trials=50, seed=0) > optimum

    def test_is_reproducible_under_a_seed(self):
        instance = barrier_instance(14, 4, lam=4.0, seed=0)

        assert random_feasible_energy(instance, 30, seed=7) == random_feasible_energy(
            instance, 30, seed=7
        )


class TestConstraintPreservingSolversAreOptimal:
    """The headline of this experiment, pinned so a regression cannot pass silently.

    These solvers reaching the *exact* optimum -- not merely a good value -- is what
    licenses trusting them at n=200 where enumeration is impossible. If either ever stops
    doing so at n=20, that inference no longer holds and the test should fail loudly.
    """

    @pytest.mark.parametrize("seed", range(4))
    def test_tabu_finds_the_exact_optimum(self, seed):
        instance = barrier_instance(20, 5, lam=4.0, seed=seed)
        optimum, _ = exact_optimum(instance)
        result = TabuSearch(num_restarts=50, timeout=None, seed=seed).solve(instance)

        assert energy_of(instance, result.selection) == pytest.approx(optimum, abs=1e-9)

    @pytest.mark.parametrize("seed", range(4))
    def test_swap_annealing_finds_the_exact_optimum(self, seed):
        instance = barrier_instance(20, 5, lam=4.0, seed=seed)
        optimum, _ = exact_optimum(instance)
        result = FeasibleAnnealing(num_restarts=8, num_sweeps=120, seed=seed).solve(instance)

        assert energy_of(instance, result.selection) == pytest.approx(optimum, abs=1e-9)

    def test_greedy_does_not_find_it(self):
        """The premise. If greedy were optimal there would be nothing to optimise."""
        from qubo_rerank.solvers import GreedyTopK

        beaten = 0
        for seed in range(6):
            instance = barrier_instance(20, 5, lam=4.0, seed=seed)
            optimum, _ = exact_optimum(instance)
            greedy = energy_of(instance, GreedyTopK().solve(instance).selection)
            beaten += greedy > optimum + 1e-9

        assert beaten >= 5, "greedy matched the optimum too often; lam may be too small"


class TestRecoveryScalesWithN:
    def test_the_optimum_gap_is_measurable_at_every_size(self):
        """Sanity: the random baseline must be strictly worse than the optimum, or the
        '% recovered' denominator collapses and the headline percentages are undefined."""
        for n in (14, 18, 22):
            instance = barrier_instance(n, 5, lam=4.0, seed=0)
            optimum, _ = exact_optimum(instance)
            random_energy = random_feasible_energy(instance, trials=50, seed=0)

            assert random_energy - optimum > 1e-6, f"degenerate span at n={n}"

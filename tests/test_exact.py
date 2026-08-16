"""Tests for the exact MILP baseline.

This module answers the question every reviewer asks first -- "why not just use an ILP
solver?" -- so its correctness decides whether that answer means anything. It is also the
module where the two worst bugs in the project were made, both of which produced a solver
reporting ``Optimal`` on every instance while optimising the wrong function.

The first built the objective from raw relevance and similarity while the comparison
scored on the normalised BQM. The second "fixed" that by building from the
*penalty-encoded* BQM, which is backwards: an ILP enforces cardinality natively, so
carrying the penalty adds C(n,2) dense couplings purely as a handicap and made the model
100x slower. Neither was visible in the output. Both were caught by comparing against
exhaustive enumeration, which is what these tests do.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exact import objective_energy, solve_exact
from experiments.sensitivity import barrier_instance


def brute_force(instance) -> tuple[float, list[int]]:
    best, arg = np.inf, []
    for subset in itertools.combinations(range(instance.n), instance.k):
        energy = objective_energy(instance, list(subset))
        if energy < best:
            best, arg = energy, list(subset)
    return best, arg


class TestObjectiveEnergy:
    def test_excludes_the_cardinality_penalty(self):
        """The objective must be relevance and diversity only.

        Every method returns exactly k items, so the penalty is a constant that cannot
        change any ranking -- but including it would force the MILP to carry C(n,2)
        auxiliary variables for a constraint it enforces in one line.
        """
        instance = barrier_instance(12, 4, lam=1.0, seed=0)
        selection = [0, 1, 2, 3]

        relevance = -np.asarray(instance.relevance)[selection].sum()
        block = instance.similarity[np.ix_(selection, selection)]
        expected = relevance + instance.lam * 0.5 * block.sum()

        assert objective_energy(instance, selection) == pytest.approx(expected)

    def test_lam_zero_reduces_to_negative_relevance(self):
        instance = barrier_instance(12, 4, lam=0.0, seed=0)
        selection = [1, 3, 5, 7]

        assert objective_energy(instance, selection) == pytest.approx(
            -np.asarray(instance.relevance)[selection].sum()
        )

    def test_higher_lam_penalises_similar_pairs_more(self):
        low = barrier_instance(20, 5, lam=0.0, seed=1)
        high = barrier_instance(20, 5, lam=8.0, seed=1)
        selection = [0, 1, 2, 3, 4]

        assert objective_energy(high, selection) > objective_energy(low, selection)


class TestAgainstEnumeration:
    """The check that caught both historical bugs.

    A MILP can report ``Optimal`` while having been handed a different objective. The
    only way to know it solved *this* problem is to compare against every feasible
    subset at a size where that is possible.
    """

    @pytest.mark.parametrize("seed", range(3))
    def test_matches_brute_force(self, seed):
        instance = barrier_instance(16, 4, lam=4.0, seed=seed)
        instance.mu = 0.0

        expected, _ = brute_force(instance)
        selection, _, status = solve_exact(instance, time_limit=120)

        assert status == "Optimal"
        assert objective_energy(instance, selection) == pytest.approx(expected, abs=1e-8)

    def test_matches_brute_force_without_diversity(self):
        """At lam=0 the optimum is the k most relevant items, known in closed form."""
        instance = barrier_instance(18, 5, lam=0.0, seed=0)
        instance.mu = 0.0

        selection, _, status = solve_exact(instance, time_limit=120)
        greedy = sorted(np.argsort(-np.asarray(instance.relevance))[: instance.k])

        assert status == "Optimal"
        assert sorted(selection) == greedy

    def test_returns_exactly_k_items(self):
        instance = barrier_instance(16, 6, lam=4.0, seed=2)
        instance.mu = 0.0

        selection, _, _ = solve_exact(instance, time_limit=120)

        assert len(set(selection)) == instance.k


class TestSolverContract:
    def test_reports_its_status_rather_than_hiding_it(self):
        """A time-limited run returns its incumbent, which looks identical to a solved
        one. The status is the only way a caller can tell, so it must be surfaced."""
        instance = barrier_instance(14, 4, lam=4.0, seed=0)
        instance.mu = 0.0

        _, seconds, status = solve_exact(instance, time_limit=60)

        assert status in {"Optimal", "Not Solved", "Infeasible", "Undefined"}
        assert seconds >= 0.0

    def test_rejects_negative_similarity(self):
        """The one-sided McCormick relaxation is only valid for non-negative
        coefficients. A negative one would silently make the relaxation unsound, so it
        must raise rather than return a confidently wrong answer.
        """
        instance = barrier_instance(12, 4, lam=1.0, seed=0)
        instance.similarity = instance.similarity.copy()
        instance.similarity[0, 1] = instance.similarity[1, 0] = -0.5

        with pytest.raises(ValueError, match="negative similarity"):
            solve_exact(instance, time_limit=30)

    def test_is_deterministic(self):
        instance = barrier_instance(14, 4, lam=4.0, seed=3)
        instance.mu = 0.0

        first, _, _ = solve_exact(instance, time_limit=60)
        second, _, _ = solve_exact(instance, time_limit=60)

        assert first == second

"""Tests for the Simulated Bifurcation solver.

This solver is used to make a claim about the *problem* -- that penalty-encoded
cardinality defeats continuous dynamics as well as discrete search -- so the burden of
proof is higher than for a solver that merely has to work. A broken implementation would
produce exactly the same headline ("SB fails on this problem") from a bug, and that is a
much easier thing to write than a correct negative result.

So the suite is arranged in two halves:

* **The implementation is correct.** It recovers the exhaustive optimum on small random
  dense QUBOs, and its QUBO -> Ising conversion is checked numerically rather than by
  re-deriving the algebra in a comment. That conversion had a genuine factor-of-two error
  when first written -- ``J/8`` where ``J/4`` was needed -- which is invisible in the
  output because the result is still a plausible Ising problem, merely a different one.
* **The failure is a property of the encoding.** The mechanism is pinned directly: the
  cardinality penalty produces a near-uniform Ising field, and how uniform it is tracks
  the penalty strength.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.synthetic import make_instance
from qubo_rerank.formulations.builder import build_problem
from qubo_rerank.solvers import SimulatedBifurcation


def qubo_arrays(problem):
    """The dense ``(h, J)`` of a built reranking problem."""
    rp = build_problem(
        relevance=problem.relevance,
        similarity=problem.similarity,
        k=problem.k,
        groups=problem.groups,
        lam=problem.lam,
        mu=problem.mu,
        targets=problem.targets,
    )
    n = problem.n
    h = np.zeros(n)
    j = np.zeros((n, n))
    for v, bias in rp.bqm.linear.items():
        h[int(v)] = bias
    for (u, v), bias in rp.bqm.quadratic.items():
        j[int(u), int(v)] = bias
        j[int(v), int(u)] = bias
    return rp, h, j


def brute_force(h: np.ndarray, j: np.ndarray) -> float:
    best = np.inf
    for bits in itertools.product([0, 1], repeat=h.size):
        x = np.asarray(bits, dtype=float)
        best = min(best, h @ x + 0.5 * x @ (j @ x))
    return best


class TestIsingConversion:
    def test_energies_differ_only_by_a_constant(self):
        """The check that would have caught the J/8 error.

        A wrong scale factor still yields a valid Ising model, so the only way to know
        it is the *same* problem is to compare energies across many states and confirm
        the difference is constant.
        """
        rng = np.random.default_rng(0)
        n = 8
        h = rng.normal(size=n)
        j = rng.normal(size=(n, n))
        j = (j + j.T) / 2
        np.fill_diagonal(j, 0.0)

        h_ising, j_ising = SimulatedBifurcation._to_ising(h, j)

        offsets = []
        for _ in range(200):
            x = rng.integers(0, 2, n).astype(float)
            s = 2 * x - 1
            offsets.append(
                (h @ x + 0.5 * x @ (j @ x)) - (h_ising @ s + 0.5 * s @ (j_ising @ s))
            )

        assert np.ptp(offsets) < 1e-9

    def test_ordering_of_states_is_preserved(self):
        """The property that actually matters: same argmin, whatever the offset."""
        rng = np.random.default_rng(1)
        n = 10
        h = rng.normal(size=n)
        j = rng.normal(size=(n, n))
        j = (j + j.T) / 2
        np.fill_diagonal(j, 0.0)
        h_ising, j_ising = SimulatedBifurcation._to_ising(h, j)

        states = [rng.integers(0, 2, n).astype(float) for _ in range(50)]
        qubo = [h @ x + 0.5 * x @ (j @ x) for x in states]
        ising = [
            h_ising @ (2 * x - 1) + 0.5 * (2 * x - 1) @ (j_ising @ (2 * x - 1))
            for x in states
        ]

        assert np.argmin(qubo) == np.argmin(ising)


class TestImplementationIsCorrect:
    @pytest.mark.parametrize("trial", range(4))
    def test_finds_the_exhaustive_optimum_on_random_qubos(self, trial):
        """Without this, "SB fails on the reranking QUBO" is indistinguishable from a bug."""
        rng = np.random.default_rng(trial)
        n = 10
        h = rng.normal(size=n)
        j = rng.normal(size=(n, n))
        j = (j + j.T) / 2
        np.fill_diagonal(j, 0.0)

        solver = SimulatedBifurcation(num_restarts=12, num_steps=600, seed=trial)
        h_ising, j_ising = solver._to_ising(h, j)

        best = np.inf
        for restart in range(solver.num_restarts):
            spins = solver._run_once(
                h_ising.copy(), j_ising.copy(), np.random.default_rng(restart)
            )
            x = (spins + 1) / 2
            best = min(best, h @ x + 0.5 * x @ (j @ x))

        assert best == pytest.approx(brute_force(h, j), abs=1e-9)

    def test_trajectories_stay_inside_the_walls(self):
        """The inelastic walls are what keep the dynamics bounded."""
        rng = np.random.default_rng(0)
        n = 20
        h = rng.normal(size=n) * 10
        j = rng.normal(size=(n, n)) * 10
        j = (j + j.T) / 2
        np.fill_diagonal(j, 0.0)

        solver = SimulatedBifurcation(num_steps=300, seed=0)
        spins = solver._run_once(*solver._to_ising(h, j), np.random.default_rng(0))

        assert set(np.unique(spins)) <= {-1.0, 1.0}

    def test_is_reproducible_under_a_seed(self):
        instance = make_instance(n_items=30, n_groups=3, k=5, seed=0)
        a = SimulatedBifurcation(num_restarts=3, num_steps=200, seed=7).solve(instance)
        b = SimulatedBifurcation(num_restarts=3, num_steps=200, seed=7).solve(instance)

        assert a.selection == b.selection


class TestSolveInterface:
    def test_reports_cardinality_without_repairing_it(self):
        """A wrong-length result must surface, not be padded inside the wrapper.

        Whether the penalty enforces the constraint is the question this repo asks; a
        solver that quietly truncated to k would answer it by fiat.
        """
        instance = make_instance(n_items=40, n_groups=4, k=8, seed=0)
        result = SimulatedBifurcation(num_restarts=2, num_steps=200, seed=0).solve(instance)

        assert result.stats["cardinality"] == len(result.selection)
        assert result.stats["cardinality_ok"] == (len(result.selection) == instance.k)

    def test_reports_the_energy_of_the_list_it_returned(self):
        instance = make_instance(n_items=30, n_groups=3, k=5, seed=1)
        result = SimulatedBifurcation(num_restarts=3, num_steps=200, seed=0).solve(instance)
        _, h, j = qubo_arrays(instance)

        x = np.zeros(instance.n)
        x[result.selection] = 1.0

        # The BQM offset is part of the energy. Asserting the offset-free formula is
        # what previously *enforced* this solver reporting on a different scale from
        # every other one here, so the check is written against the shared definition
        # instead: whatever bqm.energy() says for the same selection.
        from qubo_rerank.formulations.builder import build_problem

        rp = build_problem(
            relevance=instance.relevance, similarity=instance.similarity, k=instance.k,
            groups=instance.groups, lam=instance.lam, mu=instance.mu,
            targets=instance.targets,
        )
        expected = rp.bqm.energy(
            {i: (1 if i in set(result.selection) else 0) for i in range(instance.n)}
        )
        assert result.stats["energy"] == pytest.approx(expected, abs=1e-9)
        assert result.stats["energy"] == pytest.approx(
            h @ x + 0.5 * x @ (j @ x) + rp.bqm.offset, abs=1e-9
        )

    def test_carries_an_energy_breakdown(self):
        instance = make_instance(n_items=30, n_groups=3, k=5, seed=2)
        result = SimulatedBifurcation(num_restarts=2, num_steps=200, seed=0).solve(instance)

        assert set(result.stats["energy_breakdown"]) >= {"objective", "cardinality"}


class TestPenaltySwampsTheObjective:
    """The mechanism behind the negative result, pinned so it cannot rot."""

    def instance(self):
        rng = np.random.default_rng(0)
        inst = make_instance(n_items=120, n_groups=4, k=10, seed=0)
        inst.similarity = (rng.random((120, 120)) ** 6) * 0.5
        inst.similarity = (inst.similarity + inst.similarity.T) / 2
        np.fill_diagonal(inst.similarity, 0.0)
        inst.lam = 4.0
        return inst

    def test_the_ising_field_is_nearly_uniform(self):
        """Every item sees essentially the same field, so every spin bifurcates alike.

        This is the whole explanation for the failure: the per-item signal the objective
        provides is a fraction of a percent of the common-mode field the penalty adds.
        """
        _, h, j = qubo_arrays(self.instance())
        h_ising, _ = SimulatedBifurcation._to_ising(h, j)

        relative_spread = np.ptp(h_ising) / abs(h_ising.mean())
        assert relative_spread < 0.01

    def test_a_weaker_penalty_restores_the_signal(self):
        """Confirms the uniformity is caused by the penalty and not by the objective.

        This is also the dilemma restated: the penalty strength that makes the field
        informative is far below the strength that enforces the constraint.
        """
        inst = self.instance()

        def relative_spread(strength):
            rp = build_problem(
                relevance=inst.relevance, similarity=inst.similarity, k=inst.k,
                groups=inst.groups, lam=inst.lam, mu=inst.mu, targets=inst.targets,
                strength=strength,
            )
            n = inst.n
            h = np.zeros(n)
            j = np.zeros((n, n))
            for v, bias in rp.bqm.linear.items():
                h[int(v)] = bias
            for (u, v), bias in rp.bqm.quadratic.items():
                j[int(u), int(v)] = bias
                j[int(v), int(u)] = bias
            h_ising, _ = SimulatedBifurcation._to_ising(h, j)
            return np.ptp(h_ising) / abs(h_ising.mean())

        assert relative_spread(1.0) > 10 * relative_spread(100.0)

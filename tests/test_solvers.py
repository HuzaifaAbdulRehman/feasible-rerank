"""Solver behaviour tests.

The formulation tests prove the QUBO encodes what we think it encodes. These prove
the solvers actually return valid lists from it, and that the classical baselines
behave the way the benchmark assumes they do.
"""

from __future__ import annotations

import numpy as np
import pytest

from qubo_rerank.formulations.builder import build_problem
from qubo_rerank.metrics.fairness import category_coverage, intra_list_similarity
from qubo_rerank.metrics.relevance import ndcg_at_k
from qubo_rerank.problem import RerankInstance
from qubo_rerank.solvers import (
    MMR,
    FeasibleAnnealing,
    GreedyTopK,
    QuotaMMR,
    SimulatedAnnealing,
    TabuSearch,
)


def qubo_energy(instance: RerankInstance, selection) -> float:
    """Energy of a selection under the instance's own composed BQM.

    Lets solvers be compared on the objective they are all nominally minimising,
    rather than only on downstream metrics where a bad optimiser can look good by
    accident (a near-random list scores well on diversity).
    """
    problem = build_problem(
        relevance=instance.relevance,
        similarity=instance.similarity,
        k=instance.k,
        groups=instance.groups,
        lam=instance.lam,
        mu=instance.mu,
        targets=instance.targets,
    )
    sample = {i: 0 for i in range(instance.n)}
    for i in selection:
        sample[int(i)] = 1
    return float(problem.bqm.energy(sample))


@pytest.fixture
def instance() -> RerankInstance:
    """Twelve candidates in three groups, with a redundant high-relevance cluster."""
    rng = np.random.default_rng(42)
    n = 12
    rel = np.linspace(1.0, 0.2, n)
    sim = rng.random((n, n)) * 0.2
    sim = (sim + sim.T) / 2
    # Items 0-3 are near-duplicates of each other and also the most relevant.
    for i in range(4):
        for j in range(4):
            if i != j:
                sim[i, j] = 0.95
    np.fill_diagonal(sim, 0.0)
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    return RerankInstance(
        relevance=rel, similarity=sim, k=4, groups=groups, lam=1.0, mu=0.0
    )


class TestGreedy:
    def test_returns_k_items(self, instance):
        r = GreedyTopK().solve(instance)
        assert len(r.selection) == instance.k

    def test_picks_the_most_relevant(self, instance):
        r = GreedyTopK().solve(instance)
        assert r.selection == [0, 1, 2, 3]

    def test_scores_perfect_ndcg(self, instance):
        """Greedy top-k is the ideal ranking by construction, so NDCG == 1."""
        r = GreedyTopK().solve(instance)
        assert ndcg_at_k(instance.relevance, r.selection) == pytest.approx(1.0)


class TestMMR:
    def test_returns_k_items(self, instance):
        r = MMR(lam=0.5).solve(instance)
        assert len(r.selection) == instance.k

    def test_is_more_diverse_than_greedy(self, instance):
        greedy = GreedyTopK().solve(instance)
        mmr = MMR(lam=0.7).solve(instance)
        assert intra_list_similarity(
            instance.similarity, mmr.selection
        ) < intra_list_similarity(instance.similarity, greedy.selection)

    def test_lam_zero_matches_greedy(self, instance):
        assert MMR(lam=0.0).solve(instance).selection == GreedyTopK().solve(instance).selection


class TestQuotaMMR:
    def test_spreads_across_groups(self, instance):
        r = QuotaMMR(lam=0.5).solve(instance)
        assert category_coverage(instance.groups, r.selection) > category_coverage(
            instance.groups, GreedyTopK().solve(instance).selection
        )

    def test_requires_groups(self):
        inst = RerankInstance(
            relevance=np.array([1.0, 0.5, 0.2]), similarity=np.zeros((3, 3)), k=2
        )
        with pytest.raises(ValueError):
            QuotaMMR().solve(inst)


class TestSimulatedAnnealing:
    def test_respects_cardinality(self, instance):
        r = SimulatedAnnealing(num_reads=50, seed=0).solve(instance)
        assert len(r.selection) == instance.k
        assert r.stats["cardinality_ok"] is True

    def test_reports_energy_breakdown(self, instance):
        r = SimulatedAnnealing(num_reads=20, seed=0).solve(instance)
        assert set(r.stats["energy_breakdown"]) == {
            "objective",
            "cardinality",
            "fairness",
        }

    def test_lam_zero_recovers_greedy_selection(self, instance):
        """With no diversity or fairness pressure, the QUBO optimum is top-k."""
        instance.lam = 0.0
        instance.mu = 0.0
        r = SimulatedAnnealing(num_reads=200, seed=0).solve(instance)
        assert sorted(r.selection) == [0, 1, 2, 3]

    def test_fairness_improves_group_coverage(self, instance):
        """The central claim: raising mu must spread the list across groups."""
        instance.lam = 0.0
        instance.mu = 0.0
        without = SimulatedAnnealing(num_reads=200, seed=0).solve(instance)

        instance.mu = 3.0
        with_fair = SimulatedAnnealing(num_reads=200, seed=0).solve(instance)

        assert category_coverage(
            instance.groups, with_fair.selection
        ) > category_coverage(instance.groups, without.selection)

    def test_fairness_costs_relevance(self, instance):
        """And that improvement must be paid for in NDCG -- that is the trade-off."""
        instance.lam = 0.0
        instance.mu = 0.0
        without = SimulatedAnnealing(num_reads=200, seed=0).solve(instance)

        instance.mu = 3.0
        with_fair = SimulatedAnnealing(num_reads=200, seed=0).solve(instance)

        assert ndcg_at_k(instance.relevance, with_fair.selection) <= ndcg_at_k(
            instance.relevance, without.selection
        )


class TestTabuSearch:
    def test_respects_cardinality(self, instance):
        r = TabuSearch(num_reads=5, seed=0).solve(instance)
        assert len(r.selection) == instance.k
        assert r.stats["cardinality_ok"] is True

    def test_lam_zero_recovers_greedy_selection(self, instance):
        instance.lam = 0.0
        instance.mu = 0.0
        r = TabuSearch(num_reads=10, seed=0).solve(instance)
        assert sorted(r.selection) == [0, 1, 2, 3]


class TestFeasibleAnnealing:
    def test_cardinality_holds_by_construction(self, instance):
        r = FeasibleAnnealing(num_restarts=4, num_sweeps=40, seed=0).solve(instance)
        assert len(r.selection) == instance.k
        assert r.stats["cardinality_ok"] is True

    def test_reported_energy_matches_the_returned_list(self, instance):
        """The reported search energy must describe the list actually returned."""
        r = FeasibleAnnealing(num_restarts=4, num_sweeps=40, seed=0).solve(instance)

        problem = build_problem(
            relevance=instance.relevance,
            similarity=instance.similarity,
            k=instance.k,
            groups=instance.groups,
            lam=instance.lam,
            mu=instance.mu,
        )
        search = problem.parts["objective"].copy()
        search.update(problem.parts["fairness"])
        sample = {i: 0 for i in range(instance.n)}
        for i in r.selection:
            sample[i] = 1

        assert float(search.energy(sample)) == pytest.approx(
            r.stats["search_energy"], abs=1e-8
        )

    def test_swap_deltas_agree_with_recomputed_energy(self, instance):
        """The real check on the incremental bookkeeping.

        Energy is accumulated from swap deltas during the anneal and recomputed from
        scratch at the end; the gap between the two is reported as ``energy_drift``.
        Float noise over ~1e4 accepted moves is expected. A large value means the
        delta formula disagrees with the BQM -- which would silently corrupt every
        accept/reject decision while still returning a plausible-looking list.
        """
        r = FeasibleAnnealing(num_restarts=4, num_sweeps=60, seed=0).solve(instance)
        assert r.stats["energy_drift"] < 1e-6

    def test_lam_zero_recovers_greedy_selection(self, instance):
        instance.lam = 0.0
        instance.mu = 0.0
        r = FeasibleAnnealing(num_restarts=4, num_sweeps=40, seed=0).solve(instance)
        assert sorted(r.selection) == [0, 1, 2, 3]

    def test_fairness_improves_group_coverage(self, instance):
        instance.lam = 0.0
        instance.mu = 0.0
        without = FeasibleAnnealing(num_restarts=4, num_sweeps=40, seed=0).solve(instance)

        instance.mu = 3.0
        with_fair = FeasibleAnnealing(num_restarts=4, num_sweeps=40, seed=0).solve(instance)

        assert category_coverage(
            instance.groups, with_fair.selection
        ) > category_coverage(instance.groups, without.selection)

    def test_k_equals_n_has_no_swap_to_make(self, instance):
        """Degenerate but reachable: selecting the whole candidate set.

        There is exactly one feasible state and no swap exists, so the sampler must
        short-circuit rather than try to draw from an empty outside-set.
        """
        instance.k = instance.n
        r = FeasibleAnnealing(num_restarts=3, num_sweeps=20, seed=0).solve(instance)
        assert sorted(r.selection) == list(range(instance.n))
        assert r.stats["accepted_moves"] == 0


@pytest.fixture
def barrier_instance() -> RerankInstance:
    """A candidate set large enough for the penalty barrier to bite.

    Mirrors the shape of real ItemKNN data: 200 candidates, k=10, and a similarity
    matrix that is mostly near-zero with a thin tail of genuinely similar pairs. At
    this size the cardinality penalty is ~700x the objective scale, which is what
    strands a single-flip sampler among feasible-but-arbitrary lists.
    """
    rng = np.random.default_rng(7)
    n = 200
    rel = np.sort(rng.random(n))[::-1]
    sim = rng.random((n, n)) ** 6 * 0.5  # heavy skew toward zero
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 0.0)
    groups = np.arange(n) % 4
    return RerankInstance(
        relevance=rel, similarity=sim, k=10, groups=groups, lam=4.0, mu=0.0
    )


class TestPenaltyBarrier:
    """The finding this project actually contributes.

    Encoding exactly-k as a penalty makes two feasible lists non-adjacent: any move
    between them passes through a state costing the full penalty weight. A
    single-flip sampler must then choose between resolving the objective and being
    able to move at all. Annealing over k-subsets sidesteps the dilemma entirely.

    These are regression tests. If a future change to the formulation, the penalty
    heuristic, or the sampler makes them fail, the headline result has moved and the
    write-up needs revisiting rather than the assertion being relaxed.
    """

    def test_feasible_annealing_beats_neal_on_its_own_objective(self, barrier_instance):
        sa = SimulatedAnnealing(num_reads=50, seed=0).solve(barrier_instance)
        feasible = FeasibleAnnealing(num_restarts=4, num_sweeps=60, seed=0).solve(
            barrier_instance
        )
        assert qubo_energy(barrier_instance, feasible.selection) < qubo_energy(
            barrier_instance, sa.selection
        )

    def test_tabu_also_beats_neal(self, barrier_instance):
        """Search memory is enough; the move set need not change.

        This is what rules out the formulation as the culprit -- a generic
        single-flip sampler *can* solve this BQM, given a strategy that escapes
        basins deliberately rather than thermally.
        """
        sa = SimulatedAnnealing(num_reads=50, seed=0).solve(barrier_instance)
        tabu = TabuSearch(num_reads=5, seed=0).solve(barrier_instance)
        assert qubo_energy(barrier_instance, tabu.selection) < qubo_energy(
            barrier_instance, sa.selection
        )

    def test_feasible_annealing_finds_the_known_optimum_at_n200(self, barrier_instance):
        """With lam=0 the objective is pure relevance, so the optimum is top-k.

        That is the one problem size here where the right answer is known outright, and
        it is 200 variables rather than the 12 of the small fixture. Recovering it
        exactly is the strongest available evidence that the swap annealer is actually
        optimising and not merely wandering somewhere better than neal.
        """
        barrier_instance.lam = 0.0
        barrier_instance.mu = 0.0
        r = FeasibleAnnealing(num_restarts=4, num_sweeps=60, seed=0).solve(barrier_instance)

        expected = sorted(np.argsort(-barrier_instance.relevance)[: barrier_instance.k])
        assert sorted(r.selection) == [int(i) for i in expected]

    def test_neal_is_feasible_but_not_optimising(self, barrier_instance):
        """neal's lists are the right length and close to arbitrary among those.

        A greedy MMR baseline reaches a lower energy on the very BQM the sampler was
        given, which is the cleanest statement of the failure.
        """
        sa = SimulatedAnnealing(num_reads=50, seed=0).solve(barrier_instance)
        assert sa.stats["cardinality_ok"] is True

        quota = QuotaMMR(lam=0.5).solve(barrier_instance)
        assert qubo_energy(barrier_instance, quota.selection) < qubo_energy(
            barrier_instance, sa.selection
        )


class TestTabuStoppingRule:
    """Tabu stops on wall-clock by default, which makes it the one solver whose quality
    depends on the machine. Both modes are pinned so the choice stays deliberate."""

    def test_default_is_the_wall_clock_rule_the_results_used(self, instance):
        result = TabuSearch(seed=0).solve(instance)

        assert result.stats["work_based_stopping"] is False

    def test_timeout_none_selects_work_based_stopping(self, instance):
        result = TabuSearch(seed=0, num_restarts=20, timeout=None).solve(instance)

        assert result.stats["work_based_stopping"] is True
        assert result.stats["num_restarts"] == 20

    def test_both_modes_respect_cardinality(self, instance):
        for solver in (
            TabuSearch(seed=0),
            TabuSearch(seed=0, num_restarts=20, timeout=None),
        ):
            assert len(solver.solve(instance).selection) == instance.k

    def test_more_restarts_never_returns_a_worse_energy(self, instance):
        """Work-based stopping should be monotone in the work done.

        Not asserting strict improvement: the plateau is reached early on this problem,
        which is the measured reason the fast default was kept.
        """
        few = TabuSearch(seed=0, num_restarts=5, timeout=None).solve(instance)
        many = TabuSearch(seed=0, num_restarts=100, timeout=None).solve(instance)

        assert many.stats["energy"] <= few.stats["energy"] + 1e-9

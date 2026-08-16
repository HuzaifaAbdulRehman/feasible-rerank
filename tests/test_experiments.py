"""Tests for the experiment drivers in ``experiments/`` and the synthetic generator.

The drivers were the last 0%-covered part of the repo, which is a bad place for a gap:
they are the only code that runs when a result is produced, and a bug in them cannot
be caught by any amount of testing of the pieces they call. `present()` in particular
carries a documented, load-bearing decision -- the README spends a paragraph on it and
puts its effect at up to 0.09 NDCG -- and until now nothing checked it did anything.

The end-to-end tests here run the real pipeline on a deliberately tiny synthetic config.
They are integration tests, not unit tests: they assert that the whole path produces a
frame with the columns the analysis reads, that the exact-k contract survives it, and
that the config switches actually switch something.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.synthetic import make_instance, make_users
from experiments.plot_pareto import pareto_front
from experiments.run_experiment import (
    build_benchmark,
    build_solvers,
    evaluate_solver,
    present,
    run,
    run_repeats,
    score_selections,
)
from experiments.tables import markdown_table, repeats_table


def tiny_cfg(**overrides) -> dict:
    """A config small enough to run the full pipeline inside a test."""
    cfg = {
        "seed": 0,
        "measure_energy": False,
        "lam": 1.0,
        "mu": 0.0,
        "data": {"n_users": 4, "n_items": 20, "n_groups": 4, "k": 5},
        "solvers": {
            "greedy": True,
            "mmr": True,
            "quota_mmr": False,
            "qubo_sa": False,
            "qubo_tabu": False,
            "qubo_feasible": True,
            "mmr_lam": 0.5,
            "num_restarts": 2,
            "num_sweeps": 20,
        },
    }
    cfg.update(overrides)
    return cfg


# ------------------------------------------------------------------ present()


class TestPresent:
    def test_orders_by_descending_relevance(self):
        instance = make_instance(n_items=10, k=3, seed=0)
        instance.relevance = np.array([0.1, 0.9, 0.5, 0.2, 0.7, 0.0, 0.3, 0.4, 0.6, 0.8])

        assert present(instance, [0, 1, 4]) == [1, 4, 0]

    def test_is_a_permutation_of_its_input(self):
        instance = make_instance(n_items=20, k=5, seed=1)
        selection = [7, 2, 19, 4, 11]

        assert sorted(present(instance, selection)) == sorted(selection)

    def test_index_order_is_not_relevance_order_in_general(self):
        """Guards the premise of the fix: if these agreed, present() would be a no-op.

        On the Amazon benchmark they *do* agree, because the loader returns candidates
        already sorted. On synthetic data they do not, which is exactly where the
        0.09 NDCG was being lost.
        """
        instance = make_instance(n_items=40, k=10, seed=3)
        selection = sorted(range(0, 40, 4))[:10]

        assert present(instance, selection) != selection

    def test_ndcg_never_falls_after_presenting(self):
        """Sorting by relevance maximises a position-discounted score over a fixed set."""
        from qubo_rerank.metrics.relevance import ndcg_at_k

        rng = np.random.default_rng(0)
        instance = make_instance(n_items=30, k=8, seed=5)

        for _ in range(20):
            selection = sorted(rng.choice(30, size=8, replace=False).tolist())
            before = ndcg_at_k(instance.relevance, selection)
            after = ndcg_at_k(instance.relevance, present(instance, selection))
            assert after >= before - 1e-12


# ------------------------------------------------------- generator and benchmark


class TestSyntheticGenerator:
    def test_instance_shapes_agree(self):
        instance = make_instance(n_items=25, n_groups=5, k=6, seed=0)

        assert instance.relevance.shape == (25,)
        assert instance.similarity.shape == (25, 25)
        assert instance.groups.shape == (25,)
        assert instance.k == 6

    def test_similarity_is_symmetric_with_zero_diagonal(self):
        instance = make_instance(n_items=20, seed=2)

        assert np.allclose(instance.similarity, instance.similarity.T)
        assert np.allclose(np.diag(instance.similarity), 0.0)

    def test_same_group_pairs_are_more_similar(self):
        """The property the diversity term relies on; without it lam does nothing."""
        instance = make_instance(n_items=60, n_groups=6, seed=4)
        same = instance.groups[:, None] == instance.groups[None, :]
        off_diagonal = ~np.eye(60, dtype=bool)

        within = instance.similarity[same & off_diagonal].mean()
        across = instance.similarity[~same].mean()
        assert within > across

    def test_seed_is_reproducible(self):
        a = make_instance(n_items=20, seed=7)
        b = make_instance(n_items=20, seed=7)

        assert np.array_equal(a.relevance, b.relevance)
        assert np.array_equal(a.similarity, b.similarity)

    def test_make_users_varies_instances_under_one_seed(self):
        users = make_users(n_users=5, n_items=20, seed=0)

        assert len(users) == 5
        assert not np.array_equal(users[0].relevance, users[1].relevance)


class TestBuildBenchmark:
    def test_synthetic_source_is_the_default(self):
        cfg = tiny_cfg()
        cfg.pop("dataset", None)
        bench = build_benchmark(cfg)

        assert bench.name == "synthetic"
        assert len(bench.instances) == 4
        assert bench.n_catalogue == 20

    def test_synthetic_has_no_ground_truth(self):
        """Recall must be blank rather than a fabricated zero on this benchmark."""
        bench = build_benchmark(tiny_cfg())

        assert all(len(r) == 0 for r in bench.relevant)

    def test_weights_reach_the_instances(self):
        bench = build_benchmark(tiny_cfg(lam=3.0, mu=2.0))

        assert bench.instances[0].lam == 3.0
        assert bench.instances[0].mu == 2.0

    def test_unknown_source_is_rejected(self):
        with pytest.raises(ValueError, match="unknown dataset source"):
            build_benchmark(tiny_cfg(dataset={"source": "netflix"}))


class TestBuildSolvers:
    def test_config_switches_select_solvers(self):
        names = {s.name for s in build_solvers(tiny_cfg())}

        # balanced_quota is on by default, like the other classical baselines. It was
        # added after an audit showed the feasibility claim rested on QuotaMMR's
        # remainder defect rather than on any property of classical reranking, so
        # leaving it off by default would reintroduce exactly that error.
        assert names == {"greedy_topk", "mmr", "balanced_quota", "qubo_feasible"}

    def test_disabling_everything_yields_nothing(self):
        cfg = tiny_cfg()
        for key in ("greedy", "mmr", "quota_mmr", "balanced_quota", "qubo_sa",
                    "qubo_tabu", "qubo_feasible"):
            cfg["solvers"][key] = False

        assert build_solvers(cfg) == []


# ----------------------------------------------------------------- end to end


class TestRunEndToEnd:
    def test_produces_the_columns_the_analysis_reads(self):
        frame = run(tiny_cfg(), build_benchmark(tiny_cfg()))

        required = {
            "method", "ndcg@k", "recall@k", "category_coverage", "exposure_parity",
            "intra_list_sim", "catalogue_coverage", "gini", "seconds",
        }
        assert required <= set(frame.columns)
        # Derived from the registry rather than hardcoded: adding a baseline is a
        # legitimate change, and a literal here turns that into a spurious failure.
        assert len(frame) == len(build_solvers(tiny_cfg()))

    def test_every_solver_returns_exactly_k(self):
        """The contract the whole formulation exists to enforce, checked end to end."""
        cfg = tiny_cfg()
        bench = build_benchmark(cfg)

        for solver in build_solvers(cfg):
            for instance in bench.instances:
                result = solver.solve(instance)
                assert len(set(result.selection)) == instance.k

    def test_greedy_scores_perfect_ndcg(self):
        """A fixed point of the whole pipeline: the ideal ranking is the greedy one."""
        frame = run(tiny_cfg(), build_benchmark(tiny_cfg()))
        greedy = frame.loc[frame.method == "greedy_topk", "ndcg@k"].iloc[0]

        assert greedy == pytest.approx(1.0)

    def test_recall_is_blank_without_ground_truth(self):
        frame = run(tiny_cfg(), build_benchmark(tiny_cfg()))

        assert frame["recall@k"].isna().all()

    def test_fairness_weight_improves_group_coverage(self):
        """mu has to do something, or the fairness term is decoration."""
        fair = run(tiny_cfg(mu=8.0), build_benchmark(tiny_cfg(mu=8.0)))
        blind = run(tiny_cfg(mu=0.0), build_benchmark(tiny_cfg(mu=0.0)))

        def parity(frame):
            return frame.loc[frame.method == "qubo_feasible", "exposure_parity"].iloc[0]

        assert parity(fair) <= parity(blind)

    def test_evaluate_solver_without_measurement_still_times(self):
        cfg = tiny_cfg()
        bench = build_benchmark(cfg)
        row = evaluate_solver(build_solvers(cfg)[0], bench, measure=False)

        assert row["seconds"] is not None and row["seconds"] >= 0.0
        assert not row["energy_measured"]


class TestRunRepeats:
    def test_varies_the_seed_and_reports_spread(self):
        per_repeat, summary = run_repeats(tiny_cfg(), repeats=3)

        assert sorted(per_repeat["seed"].unique().tolist()) == [0, 1, 2]
        assert len(per_repeat) == 3 * len(build_solvers(tiny_cfg()))
        assert ("ndcg@k", "std") in summary.columns

    def test_resampling_actually_changes_the_benchmark(self):
        """If the seed did not reach the generator, every repeat would be identical
        and the reported std would be a meaningless zero."""
        per_repeat, _ = run_repeats(tiny_cfg(), repeats=3)
        mmr = per_repeat[per_repeat.method == "mmr"]["ndcg@k"]

        assert mmr.nunique() > 1


# -------------------------------------------------------------- analysis tools


class TestParetoFront:
    def test_keeps_only_undominated_points(self):
        frame = pd.DataFrame(
            {
                "method": ["a", "b", "c"],
                "gini": [0.1, 0.2, 0.3],
                "ndcg@k": [0.9, 0.5, 0.4],  # a dominates both b and c
            }
        )
        assert pareto_front(frame, "gini")["method"].tolist() == ["a"]

    def test_keeps_genuine_trade_offs(self):
        frame = pd.DataFrame(
            {
                "method": ["cheap", "accurate"],
                "gini": [0.1, 0.4],
                "ndcg@k": [0.5, 0.9],
            }
        )
        assert len(pareto_front(frame, "gini")) == 2

    def test_drops_rows_missing_the_cost_axis(self):
        """iif is NaN on the synthetic benchmark; those rows must not become a front."""
        frame = pd.DataFrame(
            {"method": ["a", "b"], "iif": [np.nan, 0.2], "ndcg@k": [0.9, 0.5]}
        )
        assert pareto_front(frame, "iif")["method"].tolist() == ["b"]


class TestTables:
    def test_markdown_table_bolds_the_best_per_column(self):
        frame = pd.DataFrame(
            {"method": ["greedy_topk", "mmr"], "ndcg@k": [1.0, 0.5], "gini": [0.9, 0.2]}
        )
        rendered = markdown_table(frame)

        assert "**1.0000**" in rendered   # higher NDCG is better
        assert "**0.2000**" in rendered   # lower Gini is better

    def test_markdown_table_bolds_ties_together(self):
        """Two methods at the arithmetic floor of parity must both be marked; bolding
        only the first would invent a winner."""
        frame = pd.DataFrame(
            {"method": ["mmr", "quota_mmr"], "exposure_parity": [0.2667, 0.2667]}
        )
        assert markdown_table(frame).count("**0.2667**") == 2

    def test_repeats_table_names_its_methods(self):
        frame = pd.DataFrame(
            {
                "seed": [0, 0, 1, 1],
                "method": ["mmr", "quota_mmr"] * 2,
                "ndcg@k": [0.9, 0.8, 0.92, 0.82],
            }
        )
        rendered = repeats_table(frame, ["ndcg@k"])

        assert "| mmr |" in rendered
        assert "| quota_mmr |" in rendered
        assert "over 2 seeds" in rendered


class TestParityIsScoredAgainstDeclaredTargets:
    """Regression test for a bug that manufactures a fake accuracy/fairness trade-off.

    ``exposure_parity`` takes an optional target vector. If the scorer omits it while
    the benchmark declares proportional targets, every method is graded against equal
    share regardless of what it was asked to optimise -- so a method that hits its
    target exactly is recorded as unfair, and the method that ignored the target scores
    better. The first version of ``experiments/ablation.py`` shipped exactly this error,
    and it produced a plausible trade-off curve that was entirely an artefact.
    """

    def _bench(self, targets):
        from benchmarks.loader import Benchmark
        from qubo_rerank.problem import RerankInstance

        groups = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        inst = RerankInstance(
            relevance=np.linspace(1.0, 0.1, 8),
            similarity=np.eye(8),
            k=4,
            groups=groups,
            targets=targets,
        )
        return Benchmark(instances=[inst], relevant=[set()], n_catalogue=8,
                         catalogue_groups=groups, stats={})

    def test_declared_targets_are_used(self):
        """A 3/1 split is perfect under 3:1 targets and off by 1 under equal share."""
        selection = [[0, 1, 2, 6]]

        proportional = score_selections(self._bench({0: 3.0, 1: 1.0}), selection)
        equal = score_selections(self._bench(None), selection)

        assert proportional.per_user["exposure_parity"][0] == pytest.approx(0.0)
        assert equal.per_user["exposure_parity"][0] > 0.0

    def test_equal_share_reading_is_reported_alongside(self):
        """Both definitions are emitted so the choice is visible rather than implicit."""
        selection = [[0, 1, 2, 6]]

        scored = score_selections(self._bench({0: 3.0, 1: 1.0}), selection)

        assert scored.per_user["exposure_parity"][0] == pytest.approx(0.0)
        assert scored.per_user["exposure_parity_equal"][0] > 0.0

    def test_without_targets_the_two_readings_agree(self):
        """The fix must not move any previously published number, all of which come
        from benchmarks that declare no targets."""
        scored = score_selections(self._bench(None), [[0, 1, 2, 6]])

        assert (
            scored.per_user["exposure_parity"][0]
            == scored.per_user["exposure_parity_equal"][0]
        )

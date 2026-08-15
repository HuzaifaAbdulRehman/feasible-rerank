"""Tests for the tune/evaluate protocol in ``experiments/protocol.py``.

The disjointness test is the one that matters. The entire value of this module is the
claim that no user contributed to both choosing a configuration and scoring it; if the
split leaked, every number it produces would be exactly the optimistically-biased
quantity it exists to avoid, and would look completely normal.

The selection tests pin the criterion. "Maximise NDCG subject to parity <= tau" has two
failure modes that both produce plausible output: silently returning the unconstrained
argmax (which is always greedy top-k, at NDCG 1.0), and silently returning the least-bad
point when nothing meets the budget, so an infeasible method appears to have competed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.protocol import (
    Configurable,
    build_configurables,
    evaluate_grid,
    run_protocol,
    select,
    split_users,
    summarise,
)
from experiments.run_experiment import build_benchmark
from qubo_rerank.solvers import MMR


def tiny_cfg(**overrides) -> dict:
    cfg = {
        "seed": 0,
        "measure_energy": False,
        "lam": 1.0,
        "mu": 0.0,
        "data": {"n_users": 8, "n_items": 18, "n_groups": 3, "k": 4},
        "solvers": {
            "greedy": True, "mmr": True, "quota_mmr": True,
            "qubo_sa": False, "qubo_tabu": False, "qubo_feasible": True,
            "mmr_lam": 0.5, "num_restarts": 2, "num_sweeps": 10,
        },
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def bench():
    return build_benchmark(tiny_cfg())


class TestSplitUsers:
    def test_halves_are_disjoint(self, bench):
        """The claim the whole module rests on: no user tunes and is scored."""
        tune, evaluate = split_users(bench, tune_frac=0.5, seed=0)

        tune_ids = {id(i) for i in tune.instances}
        eval_ids = {id(i) for i in evaluate.instances}
        assert tune_ids.isdisjoint(eval_ids)

    def test_split_loses_no_users(self, bench):
        tune, evaluate = split_users(bench, tune_frac=0.5, seed=0)

        assert len(tune.instances) + len(evaluate.instances) == len(bench.instances)
        assert len(tune.relevant) == len(tune.instances)
        assert len(evaluate.relevant) == len(evaluate.instances)

    def test_labels_travel_with_their_users(self, bench):
        """A shuffled split must not pair user A's list with user B's ground truth."""
        original = {id(i): r for i, r in zip(bench.instances, bench.relevant, strict=True)}
        tune, evaluate = split_users(bench, tune_frac=0.5, seed=1)

        for half in (tune, evaluate):
            for inst, rel in zip(half.instances, half.relevant, strict=True):
                assert original[id(inst)] == rel

    def test_catalogue_is_shared_by_both_halves(self, bench):
        tune, evaluate = split_users(bench, tune_frac=0.5, seed=0)

        assert tune.n_catalogue == evaluate.n_catalogue == bench.n_catalogue

    def test_fraction_controls_the_cut(self, bench):
        tune, evaluate = split_users(bench, tune_frac=0.25, seed=0)

        assert len(tune.instances) == 2
        assert len(evaluate.instances) == 6

    def test_neither_half_is_ever_empty(self, bench):
        """A degenerate fraction must not produce a split that cannot be evaluated."""
        for frac in (0.01, 0.99):
            tune, evaluate = split_users(bench, tune_frac=frac, seed=0)
            assert len(tune.instances) >= 1
            assert len(evaluate.instances) >= 1

    def test_seed_changes_the_partition(self, bench):
        a, _ = split_users(bench, seed=0)
        b, _ = split_users(bench, seed=1)

        assert {id(i) for i in a.instances} != {id(i) for i in b.instances}

    def test_invalid_fraction_is_rejected(self, bench):
        for frac in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError, match="strictly between"):
                split_users(bench, tune_frac=frac)


#: (params, ndcg, parity) -- accuracy and fairness deliberately in conflict, so a
#: selection rule that ignores the budget cannot accidentally agree with one that honours it.
GRID: list[tuple[dict[str, float], float, float]] = [
    ({"lam": 0.0}, 1.00, 0.90),   # most accurate, least fair
    ({"lam": 1.0}, 0.85, 0.40),
    ({"lam": 4.0}, 0.70, 0.20),   # least accurate, most fair
]


class TestSelect:

    def test_picks_the_most_accurate_point_inside_the_budget(self):
        chosen = select(GRID, tau=0.50)

        assert chosen.params == {"lam": 1.0}
        assert chosen.feasible

    def test_a_loose_budget_admits_the_unconstrained_best(self):
        chosen = select(GRID, tau=1.0)

        assert chosen.params == {"lam": 0.0}
        assert chosen.feasible

    def test_a_tight_budget_forces_the_fair_point(self):
        chosen = select(GRID, tau=0.20)

        assert chosen.params == {"lam": 4.0}
        assert chosen.feasible

    def test_infeasible_is_flagged_not_hidden(self):
        """Nothing meets tau, so the method must be marked as not having competed."""
        chosen = select(GRID, tau=0.05)

        assert not chosen.feasible
        assert chosen.params == {"lam": 4.0}  # least-bad, but flagged

    def test_boundary_is_inclusive(self):
        chosen = select(GRID, tau=0.40)

        assert chosen.params == {"lam": 1.0}
        assert chosen.feasible

    def test_reports_how_many_configurations_were_tried(self):
        """The size of the search is part of what a selected result means."""
        assert select(GRID, tau=1.0).n_considered == 3


class TestConfigurables:
    def test_every_method_gets_a_grid(self):
        configurables = build_configurables(
            tiny_cfg(), lam_grid=[0.0, 1.0], mu_grid=[0.0, 1.0], mmr_grid=[0.3, 0.7]
        )
        grids = {c.name: len(c.grid) for c in configurables}

        assert grids["qubo_feasible"] == 4      # 2 lam x 2 mu
        assert grids["mmr"] == 2                # the baseline is tuned too
        assert grids["quota_mmr"] == 2
        assert grids["greedy_topk"] == 1        # nothing to tune

    def test_baselines_are_not_left_at_their_defaults(self):
        """The asymmetry this module exists to remove: tuning one side only."""
        configurables = build_configurables(
            tiny_cfg(), lam_grid=[0.0], mu_grid=[0.0], mmr_grid=[0.1, 0.5, 0.9]
        )
        mmr = next(c for c in configurables if c.name == "mmr")

        assert [p["mmr_lam"] for p in mmr.grid] == [0.1, 0.5, 0.9]

    def test_unknown_method_is_rejected(self):
        with pytest.raises(SystemExit, match="unknown method"):
            build_configurables(tiny_cfg(), [0.0], [0.0], [0.5], wanted={"qubo_qft"})

    def test_weights_are_applied_to_the_benchmark(self, bench):
        configurable = next(
            c for c in build_configurables(tiny_cfg(), [2.0], [3.0], [0.5])
            if c.name == "qubo_feasible"
        )
        configurable.apply(bench, {"lam": 2.0, "mu": 3.0})

        assert all(i.lam == 2.0 and i.mu == 3.0 for i in bench.instances)


class TestEvaluateGrid:
    def test_returns_one_row_per_configuration(self, bench):
        configurable = Configurable(
            name="mmr",
            grid=[{"mmr_lam": v} for v in (0.2, 0.5, 0.8)],
            make=lambda p: MMR(lam=p["mmr_lam"]),
        )
        rows = evaluate_grid(configurable, bench)

        assert len(rows) == 3
        assert all(0.0 <= ndcg <= 1.0 for _, ndcg, _ in rows)
        assert all(parity >= 0.0 for _, _, parity in rows)


class TestRunProtocol:
    def test_produces_one_row_per_seed_budget_and_method(self):
        frame = run_protocol(
            tiny_cfg(),
            build_configurables(tiny_cfg(), [0.0, 1.0], [0.0, 1.0], [0.5],
                                wanted={"greedy_topk", "quota_mmr", "qubo_feasible"}),
            taus=[0.4, 1.0],
            repeats=2,
            tune_frac=0.5,
        )

        assert len(frame) == 2 * 2 * 3
        assert set(frame["seed"]) == {0, 1}
        assert set(frame["tau"]) == {0.4, 1.0}
        assert {"feasible", "tune_ndcg", "tune_parity", "ndcg@k"} <= set(frame.columns)

    def test_greedy_never_meets_a_tight_budget(self):
        """A sanity anchor: greedy top-k optimises relevance alone, so it cannot be
        fair. If it were ever marked feasible at a tight budget, the constraint is
        not being applied."""
        frame = run_protocol(
            tiny_cfg(),
            build_configurables(tiny_cfg(), [0.0], [0.0], [0.5], wanted={"greedy_topk"}),
            taus=[0.05],
            repeats=1,
            tune_frac=0.5,
        )

        assert not frame["feasible"].any()

    def test_a_loose_budget_is_met_by_everything(self):
        frame = run_protocol(
            tiny_cfg(),
            build_configurables(tiny_cfg(), [0.0, 1.0], [0.0, 1.0], [0.5]),
            taus=[99.0],
            repeats=1,
            tune_frac=0.5,
        )

        assert frame["feasible"].all()

    def test_summary_reports_every_budget(self):
        frame = run_protocol(
            tiny_cfg(),
            build_configurables(tiny_cfg(), [0.0], [0.0], [0.5], wanted={"quota_mmr"}),
            taus=[0.3, 1.0],
            repeats=1,
            tune_frac=0.5,
        )
        text = summarise(frame)

        assert "tau = 0.3" in text
        assert "tau = 1" in text
        assert "quota_mmr" in text

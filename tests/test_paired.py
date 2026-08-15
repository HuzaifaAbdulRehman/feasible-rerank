"""Tests for the paired significance tests in ``experiments/paired.py``.

Statistical code fails quietly. A correction applied wrongly, a direction convention
inverted, or a pairing silently misaligned all produce a table that looks exactly like a
correct one -- and the whole point of this module is to decide which reported differences
are real, so a bug here corrupts the conclusions rather than crashing.

Three things are pinned specifically:

* **Holm's step-down rule.** Once a hypothesis fails to be rejected, every larger p-value
  must be held non-significant too. Skipping that step is the usual way multiple-comparison
  corrections are got wrong, and it produces *more* significant results, so nothing else
  in the suite would notice.
* **The direction convention.** ``exposure_parity`` improves downward and ``ndcg@k``
  improves upward. An inverted convention would report the losing method as the winner.
* **Pairing by user.** Differences must be taken within a user. Misaligned pairing still
  yields a plausible p-value from meaningless differences.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.paired import (
    bootstrap_ci,
    collect_per_user,
    compare,
    holm,
    run_comparisons,
    verdict,
)
from experiments.run_experiment import build_benchmark


def tiny_cfg(**overrides) -> dict:
    cfg = {
        "seed": 0,
        "measure_energy": False,
        "lam": 1.0,
        "mu": 0.0,
        "data": {"n_users": 10, "n_items": 18, "n_groups": 3, "k": 4},
        "solvers": {
            "greedy": True, "mmr": True, "quota_mmr": True,
            "qubo_sa": False, "qubo_tabu": False, "qubo_feasible": True,
            "mmr_lam": 0.5, "num_restarts": 2, "num_sweeps": 10,
        },
    }
    cfg.update(overrides)
    return cfg


class TestCompare:
    def test_detects_a_consistent_improvement(self):
        base = np.linspace(0.4, 0.9, 60)
        better = base + 0.05

        row = compare(better, base, "ndcg@k")

        assert row["median_diff"] == pytest.approx(0.05)
        assert row["better"] == 60
        assert row["worse"] == 0
        assert row["p_raw"] < 0.001

    def test_direction_is_metric_aware(self):
        """Lower parity is better; the counts must reflect that, not raw sign."""
        base = np.full(40, 0.30)
        lower = np.full(40, 0.20)

        improved = compare(lower, base, "exposure_parity")
        assert improved["better"] == 40 and improved["worse"] == 0

        # The same numbers under a higher-is-better metric mean the opposite.
        regressed = compare(lower, base, "ndcg@k")
        assert regressed["better"] == 0 and regressed["worse"] == 40

    def test_identical_inputs_do_not_raise(self):
        """Two solvers can genuinely return the same lists; p=1 is the right answer.

        scipy's wilcoxon raises when every difference is zero, so this path is handled
        explicitly rather than left to surface as a crash mid-run.
        """
        same = np.linspace(0.1, 0.9, 25)

        row = compare(same, same.copy(), "ndcg@k")

        assert row["p_raw"] == 1.0
        assert row["tied"] == 25
        assert row["better"] == 0 and row["worse"] == 0

    def test_ties_are_counted_not_dropped_from_the_report(self):
        a = np.array([0.5, 0.5, 0.5, 0.7, 0.7])
        b = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        row = compare(a, b, "ndcg@k")

        assert row["tied"] == 3
        assert row["better"] == 2
        assert row["n_users"] == 5

    def test_noise_is_not_called_significant(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.6, 0.1, 200)
        b = rng.normal(0.6, 0.1, 200)

        assert compare(a, b, "ndcg@k")["p_raw"] > 0.05


class TestBootstrapCI:
    def test_interval_brackets_the_median_difference(self):
        rng = np.random.default_rng(1)
        diff = rng.normal(0.05, 0.02, 300)

        lo, hi = bootstrap_ci(diff, seed=0)

        assert lo < np.median(diff) < hi
        assert lo > 0.0  # a real, consistently positive shift

    def test_interval_spans_zero_for_noise(self):
        rng = np.random.default_rng(2)
        diff = rng.normal(0.0, 0.05, 300)

        lo, hi = bootstrap_ci(diff, seed=0)

        assert lo < 0.0 < hi

    def test_is_reproducible_under_a_seed(self):
        diff = np.random.default_rng(3).normal(0.01, 0.02, 120)

        assert bootstrap_ci(diff, seed=7) == bootstrap_ci(diff, seed=7)

    def test_empty_input_is_nan_not_a_crash(self):
        lo, hi = bootstrap_ci(np.array([]))

        assert np.isnan(lo) and np.isnan(hi)


class TestHolm:
    def test_adjusts_by_descending_rank(self):
        rows = [{"p_raw": 0.01}, {"p_raw": 0.02}, {"p_raw": 0.03}]

        out = holm(rows, alpha=0.05)

        # Smallest p multiplied by m, next by m-1, and so on.
        assert out[0]["p_holm"] == pytest.approx(0.03)   # 0.01 * 3
        assert out[1]["p_holm"] == pytest.approx(0.04)   # 0.02 * 2
        assert out[2]["p_holm"] == pytest.approx(0.04)   # 0.03 * 1, held up by monotonicity

    def test_step_down_stops_at_the_first_failure(self):
        """The rule that controls the family-wise error rate.

        Once one hypothesis is not rejected, every larger p-value must also be held
        non-significant -- even if its own adjusted value happens to fall under alpha.
        Getting this wrong yields *more* significant results, so it never looks broken.
        """
        rows = [{"p_raw": 0.001}, {"p_raw": 0.30}, {"p_raw": 0.31}]

        out = holm(rows, alpha=0.05)

        assert out[0]["significant"]
        assert not out[1]["significant"]
        assert not out[2]["significant"]

    def test_adjusted_values_never_decrease(self):
        rows = [{"p_raw": p} for p in (0.001, 0.002, 0.004, 0.2, 0.9)]

        out = holm(rows, alpha=0.05)
        adjusted = [r["p_holm"] for r in sorted(out, key=lambda r: r["p_raw"])]

        assert adjusted == sorted(adjusted)

    def test_adjustment_is_capped_at_one(self):
        rows = [{"p_raw": 0.9}, {"p_raw": 0.95}, {"p_raw": 0.99}]

        assert all(r["p_holm"] <= 1.0 for r in holm(rows))

    def test_a_single_comparison_is_uncorrected(self):
        out = holm([{"p_raw": 0.04}], alpha=0.05)

        assert out[0]["p_holm"] == pytest.approx(0.04)
        assert out[0]["significant"]

    def test_correction_can_erase_a_marginal_result(self):
        """One p just under 0.05 among many is exactly what correction exists for."""
        rows = [{"p_raw": 0.04}] + [{"p_raw": 0.6}] * 9

        out = holm(rows, alpha=0.05)

        assert not out[0]["significant"]


class TestRunComparisons:
    def frame(self) -> pd.DataFrame:
        users = np.arange(50)
        return pd.concat([
            pd.DataFrame({"method": "ref", "user": users,
                          "ndcg@k": np.linspace(0.5, 0.9, 50),
                          "exposure_parity": np.full(50, 0.30)}),
            pd.DataFrame({"method": "better", "user": users,
                          "ndcg@k": np.linspace(0.5, 0.9, 50) + 0.08,
                          "exposure_parity": np.full(50, 0.20)}),
        ], ignore_index=True)

    def test_compares_every_method_to_the_reference(self):
        out = run_comparisons(self.frame(), "ref", ["ndcg@k", "exposure_parity"])

        assert set(out["method"]) == {"better"}
        assert set(out["metric"]) == {"ndcg@k", "exposure_parity"}
        assert (out["reference"] == "ref").all()

    def test_pairs_by_user_not_by_row_order(self):
        """Shuffling one method's rows must not change the result."""
        frame = self.frame()
        shuffled = pd.concat([
            frame[frame.method == "ref"],
            frame[frame.method == "better"].sample(frac=1.0, random_state=0),
        ], ignore_index=True)

        a = run_comparisons(frame, "ref", ["ndcg@k"])["median_diff"].iloc[0]
        b = run_comparisons(shuffled, "ref", ["ndcg@k"])["median_diff"].iloc[0]

        assert a == pytest.approx(b)

    def test_unknown_reference_is_rejected(self):
        with pytest.raises(SystemExit, match="not among"):
            run_comparisons(self.frame(), "nonexistent", ["ndcg@k"])

    def test_metrics_absent_from_the_frame_are_skipped(self):
        out = run_comparisons(self.frame(), "ref", ["ndcg@k", "recall@k"])

        assert set(out["metric"]) == {"ndcg@k"}


class TestEndToEnd:
    def test_collects_one_row_per_method_and_user(self):
        cfg = tiny_cfg()
        per_user = collect_per_user(cfg, build_benchmark(cfg))

        assert set(per_user["method"]) == {"greedy_topk", "mmr", "quota_mmr", "qubo_feasible"}
        assert len(per_user) == 4 * 10
        assert per_user.groupby("method")["user"].nunique().eq(10).all()

    def test_greedy_is_detectably_more_accurate_than_quota_mmr(self):
        """A known-direction anchor: greedy maximises relevance by construction."""
        cfg = tiny_cfg()
        per_user = collect_per_user(cfg, build_benchmark(cfg))
        out = run_comparisons(per_user, "quota_mmr", ["ndcg@k"])

        greedy = out[out.method == "greedy_topk"].iloc[0]
        assert greedy["median_diff"] > 0
        assert greedy["better"] > greedy["worse"]


class TestVerdict:
    def test_non_significant_rows_say_so_plainly(self):
        row = pd.Series({"significant": False, "better": 30, "worse": 10})

        assert verdict(row) == "no detectable difference"

    def test_significant_rows_name_the_direction_and_counts(self):
        row = pd.Series({"significant": True, "better": 30, "worse": 10})

        text = verdict(row)
        assert "better" in text and "30" in text and "10" in text

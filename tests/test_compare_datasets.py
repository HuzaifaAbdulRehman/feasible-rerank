"""Tests for the cross-dataset synthesis in ``experiments/compare_datasets.py``.

``reach`` compresses a whole protocol run into one number per method -- the tightest
fairness budget it can actually meet -- and the cross-dataset claim is read off those
numbers directly. Two mistakes would be invisible in the output: taking the *loosest*
budget instead of the tightest, and counting a budget as met when it was met on average
rather than on every seed. Both produce a plausible table.

The every-seed rule is the one with consequences. A fairness budget is a service-level
constraint; a method that satisfies it in two runs out of three has not satisfied it, and
averaging would let exactly that method claim the tighter reach.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compare_datasets import load, reach  # noqa: E402


def protocol_rows(rows: list[tuple]) -> pd.DataFrame:
    """``(dataset, method, tau, seed, feasible, ndcg)`` -> a protocol-shaped frame."""
    return pd.DataFrame(
        rows, columns=["dataset", "method", "tau", "seed", "feasible", "ndcg@k"]
    )


class TestReach:
    def test_takes_the_tightest_budget_met(self):
        frame = protocol_rows([
            ("d", "m", 0.20, 0, False, 0.90),
            ("d", "m", 0.30, 0, True, 0.85),
            ("d", "m", 1.00, 0, True, 0.95),
        ])

        assert reach(frame)["reach"].iloc[0] == pytest.approx(0.30)

    def test_a_budget_missed_on_one_seed_does_not_count(self):
        """The every-seed rule. Averaging would report 0.20 here, which is wrong."""
        frame = protocol_rows([
            ("d", "m", 0.20, 0, True, 0.90),
            ("d", "m", 0.20, 1, True, 0.90),
            ("d", "m", 0.20, 2, False, 0.90),   # one failure is a failure
            ("d", "m", 0.30, 0, True, 0.85),
            ("d", "m", 0.30, 1, True, 0.85),
            ("d", "m", 0.30, 2, True, 0.85),
        ])

        assert reach(frame)["reach"].iloc[0] == pytest.approx(0.30)

    def test_a_method_that_never_qualifies_has_no_reach(self):
        frame = protocol_rows([
            ("d", "m", 0.20, 0, False, 0.90),
            ("d", "m", 1.00, 0, False, 0.95),
        ])

        assert np.isnan(reach(frame)["reach"].iloc[0])

    def test_accuracy_is_reported_at_the_reach_not_elsewhere(self):
        """A tight reach is only interesting if the lists are still usable there."""
        frame = protocol_rows([
            ("d", "m", 0.20, 0, False, 0.99),
            ("d", "m", 0.30, 0, True, 0.85),   # the reach
            ("d", "m", 1.00, 0, True, 0.99),
        ])

        assert reach(frame)["ndcg_at_reach"].iloc[0] == pytest.approx(0.85)

    def test_accuracy_at_reach_averages_across_seeds(self):
        frame = protocol_rows([
            ("d", "m", 0.30, 0, True, 0.80),
            ("d", "m", 0.30, 1, True, 0.90),
        ])

        assert reach(frame)["ndcg_at_reach"].iloc[0] == pytest.approx(0.85)

    def test_methods_and_datasets_are_kept_separate(self):
        frame = protocol_rows([
            ("d1", "a", 0.20, 0, True, 0.90),
            ("d1", "b", 0.30, 0, True, 0.80),
            ("d2", "a", 0.40, 0, True, 0.70),
            ("d2", "b", 0.20, 0, True, 0.60),
        ])

        out = reach(frame).set_index(["dataset", "method"])["reach"]

        assert out[("d1", "a")] == pytest.approx(0.20)
        assert out[("d1", "b")] == pytest.approx(0.30)
        assert out[("d2", "a")] == pytest.approx(0.40)
        assert out[("d2", "b")] == pytest.approx(0.20)

    def test_one_row_per_dataset_and_method(self):
        frame = protocol_rows([
            ("d1", "a", 0.30, 0, True, 0.9),
            ("d1", "a", 0.40, 0, True, 0.9),
            ("d2", "a", 0.30, 0, True, 0.9),
        ])

        assert len(reach(frame)) == 2


class TestLoad:
    def test_dataset_name_comes_from_the_filename(self, tmp_path: Path):
        frame = protocol_rows([("ignored", "m", 0.3, 0, True, 0.9)]).drop(columns="dataset")
        path = tmp_path / "amazon_giftcards_protocol.csv"
        frame.to_csv(path, index=False)

        assert load([path])["dataset"].iloc[0] == "amazon_giftcards"

    def test_several_files_are_concatenated(self, tmp_path: Path):
        frame = protocol_rows([("x", "m", 0.3, 0, True, 0.9)]).drop(columns="dataset")
        paths = []
        for name in ("a_protocol.csv", "b_protocol.csv"):
            frame.to_csv(tmp_path / name, index=False)
            paths.append(tmp_path / name)

        assert set(load(paths)["dataset"]) == {"a", "b"}

    def test_no_input_is_an_explicit_error(self):
        with pytest.raises(SystemExit, match="no protocol CSVs"):
            load([])

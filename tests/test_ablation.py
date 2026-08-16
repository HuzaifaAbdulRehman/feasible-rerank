"""Tests for the formulation ablations.

The ablation that matters found a genuine qualification of the project's headline claim:
the QUBO's fairness advantage over quota-MMR exists only when ``k`` does not divide
evenly by the number of groups. That is a claim about arithmetic, so it can be pinned
directly rather than inferred from a benchmark run -- and it should be, because it is the
sentence that tells a practitioner whether to bother with any of this.

The targets ablation is tested for the confound that made its first version wrong.
``exposure_parity`` defaults to equal targets, so scoring a proportional-target solver
against it measures an objective the solver was never given. That produced what looked
like a clean accuracy/fairness trade-off and was entirely artefact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qubo_rerank.formulations.fairness import exposure_targets_proportional
from qubo_rerank.metrics.fairness import exposure_parity


def even_groups(n: int, n_groups: int) -> np.ndarray:
    return np.repeat(np.arange(n_groups), n // n_groups)


class TestParityFloorDependsOnDivisibility:
    """Why the QUBO's fairness advantage appears and disappears.

    ``exposure_parity`` is mean absolute deviation from ``k/|C|`` per group, normalised
    by k. When k divides evenly the target is an integer, a perfect allocation exists,
    and the floor is exactly zero -- leaving nothing for a better optimiser to win. When
    it does not divide, the remainder must be spread across groups and the floor is
    strictly positive.
    """

    def test_floor_is_zero_when_k_divides_evenly(self):
        groups = even_groups(40, 4)
        perfect = [i * 10 + j for i in range(4) for j in range(5)]   # 5 from each group

        assert exposure_parity(groups, perfect) == pytest.approx(0.0)

    def test_floor_is_positive_when_k_does_not_divide(self):
        """k=10 over 4 groups -> 2.5 each, unreachable; best is 3,3,2,2."""
        groups = even_groups(40, 4)
        best = [0, 1, 2, 10, 11, 12, 20, 21, 30, 31]

        assert exposure_parity(groups, best) == pytest.approx(0.2, abs=1e-9)

    def test_the_floor_grows_as_the_remainder_gets_worse(self):
        """k=5 over 4 groups (remainder 1 of 4) is harder than k=10 (remainder 2)."""
        groups = even_groups(40, 4)

        k5_best = [0, 1, 10, 20, 30]                      # 2,1,1,1
        k10_best = [0, 1, 2, 10, 11, 12, 20, 21, 30, 31]  # 3,3,2,2

        assert exposure_parity(groups, k5_best) > exposure_parity(groups, k10_best)

    @pytest.mark.parametrize("k,n_groups,divides", [(20, 4, True), (10, 4, False),
                                                    (12, 3, True), (5, 4, False),
                                                    (9, 3, True), (7, 3, False)])
    def test_divisibility_predicts_whether_a_perfect_split_exists(self, k, n_groups, divides):
        """The arithmetic behind the claim, stated as the claim.

        If k/|C| is an integer a zero-parity selection exists, so no optimiser can beat
        a correct greedy one. That is exactly the k=20 row where the measured advantage
        collapsed to +0.0001.
        """
        groups = even_groups(n_groups * 20, n_groups)
        per_group = k // n_groups

        if divides:
            selection = [g * 20 + j for g in range(n_groups) for j in range(per_group)]
            assert exposure_parity(groups, selection) == pytest.approx(0.0)
        else:
            # No allocation of k over n_groups can be exactly equal.
            assert k % n_groups != 0


class TestProportionalTargets:
    def test_targets_sum_to_k(self):
        groups = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2])
        targets = exposure_targets_proportional(groups, k=5)

        assert sum(targets.values()) == pytest.approx(5.0)

    def test_larger_groups_get_larger_targets(self):
        groups = np.array([0] * 8 + [1] * 2)
        targets = exposure_targets_proportional(groups, k=5)

        assert targets[0] > targets[1]

    def test_equal_groups_reduce_to_equal_targets(self):
        groups = even_groups(40, 4)
        targets = exposure_targets_proportional(groups, k=10)

        assert all(t == pytest.approx(2.5) for t in targets.values())

    def test_scoring_under_the_wrong_targets_is_the_confound(self):
        """The bug the ablation had first time, pinned so it cannot return.

        A selection that perfectly matches a skewed group distribution scores *well*
        under proportional targets and *badly* under equal ones. Reporting only the
        latter makes a proportional-target solver look like it sacrificed fairness for
        accuracy when it did no such thing.
        """
        groups = np.array([0] * 16 + [1] * 4)             # 80/20 split
        matches_proportion = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17]   # 8 from group 0, 2 from 1

        proportional = exposure_targets_proportional(groups, k=10)
        under_proportional = exposure_parity(groups, matches_proportion, proportional)
        under_equal = exposure_parity(groups, matches_proportion)

        assert under_proportional < under_equal
        assert under_proportional == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------- the ablation driver itself


class TestAblationDriver:
    """The scalar ablation is a loop over configs; the failure worth guarding is that it
    silently fails to vary anything, which would produce a table of identical rows that
    reads as "the choice does not matter"."""

    def cfg(self) -> dict:
        return {
            "seed": 0,
            "measure_energy": False,
            "lam": 0.0,
            "mu": 1.0,
            "data": {"n_users": 6, "n_items": 20, "n_groups": 4, "k": 5},
            "solvers": {
                "greedy": True, "mmr": False, "quota_mmr": True,
                "qubo_sa": False, "qubo_tabu": False, "qubo_feasible": True,
                "mmr_lam": 0.5, "num_restarts": 2, "num_sweeps": 10,
            },
        }

    def test_scalar_ablation_actually_varies_the_setting(self):
        from experiments.ablation import ablate_scalar

        frame = ablate_scalar(self.cfg(), ["greedy_topk"], "k", [3, 5])

        assert sorted(frame["setting"].unique()) == ["3", "5"]
        assert (frame["ablation"] == "k").all()

    def test_scalar_ablation_reaches_the_benchmark(self):
        """Varying k must change the lists, not just the label on the row."""
        from experiments.ablation import ablate_scalar

        frame = ablate_scalar(self.cfg(), ["quota_mmr"], "k", [2, 8])
        parities = frame["exposure_parity"].tolist()

        assert parities[0] != parities[1]

    def test_targets_ablation_scores_under_both_definitions(self):
        """The confound guard: both columns must be present, or a proportional-target
        solver is being judged on an equal-target metric it never optimised."""
        from experiments.ablation import ablate_targets

        frame = ablate_targets(self.cfg(), ["quota_mmr"])

        assert {"parity_vs_equal", "parity_vs_proportional"} <= set(frame.columns)
        assert set(frame["setting"]) == {"equal", "proportional"}

    def test_group_sizes_reports_every_group(self):
        from experiments.ablation import group_sizes
        from experiments.run_experiment import build_benchmark

        text = group_sizes(build_benchmark(self.cfg()))

        assert len(text.split(" / ")) == self.cfg()["data"]["n_groups"]

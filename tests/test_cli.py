"""Smoke tests for the command-line entry points.

Every result in this repo is produced by invoking these scripts, and nothing else
exercises their argument parsing, their config plumbing or their file output. That is a
gap with teeth: adding ``--solver`` to ``run_experiment.py`` meant reaching into
``cfg['solvers']`` and flipping boolean switches whose names do not all match the solver
names, and a typo there would have silently produced a run with the wrong solvers rather
than an error.

These run the real scripts in a subprocess against a tiny generated config. They are
slow relative to a unit test and deliberately few -- one per entry point, asserting the
output file exists and contains what the analysis expects, plus the failure modes a user
would actually hit.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

#: Small enough that a full run is a few seconds. qubo_sa is left out: it is the slow
#: one, and the penalty-barrier tests already cover it directly.
TINY_CONFIG = {
    "seed": 0,
    "measure_energy": False,
    "lam": 1.0,
    "mu": 0.0,
    "data": {"n_users": 3, "n_items": 16, "n_groups": 4, "k": 4},
    "solvers": {
        "greedy": True,
        "mmr": True,
        "quota_mmr": True,
        "qubo_sa": False,
        "qubo_tabu": False,
        "qubo_feasible": True,
        "mmr_lam": 0.5,
        "num_restarts": 2,
        "num_sweeps": 10,
    },
}


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(TINY_CONFIG))
    return path


def run_script(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "experiments" / script), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
    )


class TestRunExperimentCLI:
    def test_writes_a_results_csv(self, config, tmp_path):
        out = tmp_path / "out.csv"
        result = run_script("run_experiment.py", "--config", str(config), "--out", str(out))

        assert result.returncode == 0, result.stderr
        frame = pd.read_csv(out)
        assert set(frame["method"]) == {"greedy_topk", "mmr", "quota_mmr", "balanced_quota", "qubo_feasible"}
        assert {"ndcg@k", "exposure_parity", "gini", "seconds"} <= set(frame.columns)

    def test_solver_flag_restricts_the_run(self, config, tmp_path):
        """The flag whose implementation maps names onto config switches."""
        out = tmp_path / "out.csv"
        result = run_script(
            "run_experiment.py", "--config", str(config), "--out", str(out),
            "--solver", "greedy", "qubo_feasible",
        )

        assert result.returncode == 0, result.stderr
        assert set(pd.read_csv(out)["method"]) == {"greedy_topk", "qubo_feasible"}

    def test_lam_and_mu_flags_reach_the_run(self, config, tmp_path):
        """mu=0 and a large mu must not produce the same lists."""
        blind, fair = tmp_path / "a.csv", tmp_path / "b.csv"
        common = ["--config", str(config), "--solver", "qubo_feasible", "--lam", "0.0"]

        assert run_script("run_experiment.py", *common, "--mu", "0.0",
                          "--out", str(blind)).returncode == 0
        assert run_script("run_experiment.py", *common, "--mu", "16.0",
                          "--out", str(fair)).returncode == 0

        blind_parity = pd.read_csv(blind)["exposure_parity"].iloc[0]
        fair_parity = pd.read_csv(fair)["exposure_parity"].iloc[0]
        assert fair_parity < blind_parity

    def test_repeats_writes_one_row_per_seed_and_solver(self, config, tmp_path):
        out = tmp_path / "rep.csv"
        result = run_script(
            "run_experiment.py", "--config", str(config), "--out", str(out),
            "--repeats", "3", "--solver", "greedy", "mmr",
        )

        assert result.returncode == 0, result.stderr
        frame = pd.read_csv(out)
        assert sorted(frame["seed"].unique()) == [0, 1, 2]
        assert len(frame) == 6

    def test_n_users_override_is_applied(self, config, tmp_path):
        out = tmp_path / "out.csv"
        result = run_script(
            "run_experiment.py", "--config", str(config), "--out", str(out),
            "--n-users", "7", "--solver", "greedy",
        )

        assert result.returncode == 0, result.stderr
        # The user count is reported in the benchmark stats block, not the CSV, so this
        # matches on the printed value rather than on exact column alignment.
        assert re.search(r"sampled_users\s+7\b", result.stdout), result.stdout


class TestSweepCLI:
    def test_writes_a_grid_and_matched_baselines(self, config, tmp_path):
        out = tmp_path / "sweep.csv"
        result = run_script(
            "sweep.py", "--config", str(config), "--out", str(out),
            "--lam", "0.0", "4.0", "--mu", "0.0", "1.0",
        )

        assert result.returncode == 0, result.stderr

        grid = pd.read_csv(out)
        # 2 lam x 2 mu, qubo solvers only -- the baselines are run once, separately.
        assert len(grid) == 4
        assert set(grid["method"]) == {"qubo_feasible"}
        assert sorted(grid["lam"].unique()) == [0.0, 4.0]

        baselines = pd.read_csv(out.with_name(f"{out.stem}_baselines.csv"))
        assert set(baselines["method"]) == {"greedy_topk", "mmr", "quota_mmr", "balanced_quota"}

    def test_unknown_solver_is_rejected(self, config, tmp_path):
        result = run_script(
            "sweep.py", "--config", str(config), "--out", str(tmp_path / "s.csv"),
            "--solver", "qubo_quantum_supremacy",
        )

        assert result.returncode != 0
        assert "not enabled" in (result.stdout + result.stderr)


class TestPlotCLI:
    def test_writes_a_figure_from_a_sweep(self, config, tmp_path):
        sweep_out = tmp_path / "sweep.csv"
        assert run_script(
            "sweep.py", "--config", str(config), "--out", str(sweep_out),
            "--lam", "0.0", "4.0", "--mu", "0.0", "1.0",
        ).returncode == 0

        figure = tmp_path / "pareto.png"
        result = run_script(
            "plot_pareto.py", "--sweep", str(sweep_out), "--out", str(figure)
        )

        assert result.returncode == 0, result.stderr
        assert figure.exists() and figure.stat().st_size > 1000
        # It must find the baselines sweep.py wrote next to the grid, unprompted.
        assert "baselines" in result.stdout

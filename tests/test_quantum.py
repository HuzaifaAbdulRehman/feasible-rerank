"""Tests for the D-Wave QPU solver.

This is the one solver in the repo that has never been run, because it needs a Leap
account and QPU time. That makes the testable surface small but not empty, and the parts
that *are* testable are the ones that would otherwise fail confusingly at the moment
someone finally has a token: a broken import chain, or an error message that does not say
what is missing.

What cannot be tested here -- that the annealer returns good lists, what the embedding
overhead actually is -- is deliberately left untested rather than faked with a mock that
would assert only that the code calls the functions it calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.synthetic import make_instance
from qubo_rerank.solvers import DWaveAnnealer


def test_importing_the_package_does_not_need_dwave_system():
    """The optional dependency must stay optional.

    A module-level `from dwave.system import ...` would make the whole package -- and so
    the entire test suite -- unimportable on any machine without the cloud client, which
    is every machine in CI.
    """
    assert DWaveAnnealer is not None


def test_construction_needs_no_credentials():
    solver = DWaveAnnealer(num_reads=50, annealing_time=10.0)

    assert solver.name == "qubo_qpu"
    assert solver.num_reads == 50


def test_missing_dependency_explains_itself():
    """The failure a reader will actually hit, so it must name the fix.

    Skipped where dwave-system happens to be installed, since then this path is not the
    one taken -- and calling the QPU from a test would spend real quota.
    """
    try:
        import dwave.system  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("dwave-system is installed; this failure path is not reachable")

    with pytest.raises(RuntimeError, match="dwave-system"):
        DWaveAnnealer().solve(make_instance(n_items=20, k=4, seed=0))

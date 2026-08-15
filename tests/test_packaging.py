"""Checks on the package metadata itself.

Small, but each of these has a failure mode that is silent and embarrassing rather than
loud: a version that disagrees with itself, a declared Python floor the code does not
actually meet, or a dependency that is imported but never declared -- which is how this
repo previously shipped a `solvers/tabu.py` that could not import on a clean install.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

import qubo_rerank


@pytest.fixture(scope="module")
def pyproject() -> dict:
    if tomllib is None:
        pytest.skip("tomllib requires Python 3.11+")
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_version_matches_the_package():
    """Read by regex rather than via the fixture, so it runs on 3.10 too.

    This is the check most worth having everywhere: a package whose declared version
    disagrees with its own metadata is not caught by anything else, and it is the number
    that ends up in a citation.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version = "([^"]+)"', text, re.MULTILINE).group(1)

    assert declared == qubo_rerank.__version__


def test_declared_python_floor_is_not_above_the_running_interpreter(pyproject):
    """The suite passing here is evidence the floor is reachable; this pins that.

    An earlier version declared >=3.11 while being developed and tested on 3.10, which
    would have refused to install in its own development environment.
    """
    floor = pyproject["project"]["requires-python"].lstrip(">=")
    major, minor = (int(part) for part in floor.split("."))

    assert (major, minor) <= sys.version_info[:2]


def test_every_declared_package_imports(pyproject):
    import importlib

    for name in pyproject["tool"]["setuptools"]["packages"]:
        assert importlib.import_module(name) is not None


def test_solver_dependencies_are_declared(pyproject):
    """dwave-samplers was imported but undeclared once; this stops it recurring."""
    declared = " ".join(pyproject["project"]["dependencies"])

    for package in ("dimod", "dwave-neal", "dwave-samplers", "numpy", "scipy"):
        assert package in declared

"""Simulated annealing over the reranking QUBO.

Butt's project description specifies *"quantum-inspired approaches such as QUBO
formulations and simulated quantum annealing"* rather than a real quantum device, so
this runs entirely locally on CPU via ``neal``. The same BQM would go unmodified to a
D-Wave QPU through ``EmbeddingComposite`` -- swapping the sampler is the only change,
which is why the solver interface is kept this thin.
"""

from __future__ import annotations

from typing import Any

import neal

from ..formulations.builder import build_problem
from .base import SolveResult, selection_from_sample, timed


class SimulatedAnnealing:
    """Solve the reranking QUBO with ``neal.SimulatedAnnealingSampler``."""

    name = "qubo_sa"

    def __init__(
        self,
        num_reads: int = 100,
        num_sweeps: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            num_reads: independent anneals; the lowest-energy sample is kept.
            num_sweeps: sweeps per anneal. ``None`` leaves neal's default (1000).
            seed: RNG seed.

        ``num_sweeps`` is exposed so the compute budget can be varied deliberately.
        This project's central claim is that penalty-encoded cardinality defeats this
        sampler for structural reasons, and the first objection to that is always "you
        did not run it long enough" -- which is unanswerable unless the budget is a
        parameter. See ``experiments/sensitivity.py``.
        """
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.seed = seed
        self._sampler = neal.SimulatedAnnealingSampler()

    def solve(self, problem) -> SolveResult:
        stats: dict[str, Any] = {
            "num_reads": self.num_reads,
            "num_sweeps": self.num_sweeps,
        }

        with timed(stats, "build_time"):
            rp = build_problem(
                relevance=problem.relevance,
                similarity=problem.similarity,
                k=problem.k,
                groups=problem.groups,
                lam=problem.lam,
                mu=problem.mu,
                targets=problem.targets,
            )

        kwargs: dict[str, Any] = {"num_reads": self.num_reads}
        if self.num_sweeps is not None:
            kwargs["num_sweeps"] = self.num_sweeps
        if self.seed is not None:
            kwargs["seed"] = self.seed

        with timed(stats, "solve_time"):
            sampleset = self._sampler.sample(rp.bqm, **kwargs)

        best = sampleset.first
        selection = selection_from_sample(best.sample, expected_k=problem.k)

        stats["energy"] = float(best.energy)
        stats["n_selected"] = len(selection)
        # Surfaced deliberately: a mismatch means the cardinality penalty was too
        # weak relative to the objective, and every downstream metric is then being
        # computed on a list of the wrong length.
        stats["cardinality_ok"] = len(selection) == problem.k
        stats["energy_breakdown"] = rp.energy_breakdown(selection)

        return SolveResult(selection=selection, stats=stats)

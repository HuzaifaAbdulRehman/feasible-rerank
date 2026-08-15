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

    def __init__(self, num_reads: int = 100, seed: int | None = None) -> None:
        self.num_reads = num_reads
        self.seed = seed
        self._sampler = neal.SimulatedAnnealingSampler()

    def solve(self, problem) -> SolveResult:  # noqa: ANN001
        stats: dict[str, Any] = {"num_reads": self.num_reads}

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

"""Tabu search over the reranking QUBO.

The interesting control in this benchmark. :class:`FeasibleAnnealing` fixes the
penalty-barrier problem by changing the *move set*, which means giving up a generic
QUBO sampler. Tabu keeps the generic single-flip interface but adds memory: recently
flipped variables are forbidden for a few iterations, so the search is pushed out of
the basin it is sitting in rather than having to be shaken out of it thermally.

That turns out to be enough. On Amazon Luxury Beauty it lands within a few percent of
the constraint-preserving annealer while remaining a drop-in ``dimod`` sampler -- which
matters, because it is evidence that the failure of ``neal`` here is about the *search
strategy* interacting with the penalty encoding, not about anything inherent to the
QUBO formulation itself. Had tabu also failed, the formulation would have been the
suspect.

``dwave-samplers`` is Apache-2.0 and ships the same ``.sample(bqm)`` interface as
``neal`` and ``DWaveSampler``.
"""

from __future__ import annotations

from typing import Any

from dwave.samplers import TabuSampler

from ..formulations.builder import build_problem
from .base import SolveResult, selection_from_sample, timed


class TabuSearch:
    """Solve the reranking QUBO with ``dwave.samplers.TabuSampler``."""

    name = "qubo_tabu"

    def __init__(self, num_reads: int = 10, seed: int | None = None) -> None:
        self.num_reads = num_reads
        self.seed = seed
        self._sampler = TabuSampler()

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
        stats["cardinality_ok"] = len(selection) == problem.k
        stats["energy_breakdown"] = rp.energy_breakdown(selection)

        return SolveResult(selection=selection, stats=stats)

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

    def __init__(
        self,
        num_reads: int = 10,
        num_restarts: int = 50,
        timeout: float | None = 20.0,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            num_reads: independent runs; the best is kept.
            num_restarts: tabu restarts per read. **This is the stopping criterion.**
            timeout: wall-clock cap in milliseconds, or ``None`` for the work-based
                stopping this class defaults to. Set it only to reproduce the old
                behaviour deliberately.
            seed: RNG seed.

        **On the stopping rule, which is worth understanding before changing it.**

        ``TabuSampler`` stops at whichever of ``timeout`` and ``num_restarts`` fires
        first, and its own default is ``timeout=20`` -- 20 ms of *wall-clock* per read.
        A solver that stops on elapsed time does less search on a slower machine, so its
        solution quality becomes a property of the hardware. That is not hypothetical
        here: running identical seeds with this laptop on battery (CPU pinned to
        1.297 GHz) and on mains (~3.6 GHz turbo), the two fixed-work solvers returned
        bit-identical results while this one moved from 0.8801 to 0.8873 NDCG.

        The obvious fix is to stop on work instead. Measured on Amazon Luxury Beauty,
        40 users at lam=0, mu=1, that costs a great deal for almost nothing:

            timeout=20ms       NDCG 0.8983   parity 0.1992    10.8 s
            num_restarts=50    NDCG 0.8985   parity 0.1992    62.9 s
            num_restarts=200   NDCG 0.8985   parity 0.1992   232.2 s

        20 ms already reaches the quality plateau *on this machine*, and buying the last
        0.0002 NDCG costs 6-21x the wall-clock. That also explains the battery figure:
        mains-20ms sits at the plateau and battery-20ms, with 2.8x less search, sits
        below it.

        So the default keeps ``timeout=20`` -- it is Pareto-better here, and it is what
        every result in ``results/`` was produced with. ``num_restarts=50`` is the
        work-based plateau and is the setting to use for a portable comparison, by
        passing ``timeout=None``:

            TabuSearch(num_restarts=50, timeout=None)

        **Anyone reproducing these numbers on slower hardware should do exactly that**,
        or they will measure their CPU rather than the method.
        """
        self.num_reads = num_reads
        self.num_restarts = num_restarts
        self.timeout = timeout
        self.seed = seed
        self._sampler = TabuSampler()

    def solve(self, problem) -> SolveResult:
        stats: dict[str, Any] = {
            "num_reads": self.num_reads,
            "num_restarts": self.num_restarts,
            "work_based_stopping": self.timeout is None,
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

        kwargs: dict[str, Any] = {
            "num_reads": self.num_reads,
            "num_restarts": self.num_restarts,
        }
        # dwave-samplers stops at whichever criterion fires first, so a work-based run
        # needs the clock pushed far enough out that it cannot bind. Passing None is not
        # accepted; a large finite value is the supported way to disable it.
        kwargs["timeout"] = float(self.timeout) if self.timeout is not None else 1e9
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

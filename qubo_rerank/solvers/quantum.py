"""D-Wave quantum annealer, via Leap.

**Status: written, not yet run on hardware.** Everything else in this repo reports
numbers that were measured. This module does not, because running it needs a D-Wave Leap
account and the free tier's monthly minute of QPU time. It is included so the experiment
is ready to run, and it is marked clearly so nothing here is mistaken for a result.

Why it belongs in the project at all: the central finding is that penalty-encoded
cardinality defeats unaided solvers, demonstrated on a thermal sampler (``qubo_sa``) and
on continuous dynamics (``qubo_sb``). A physical annealer receives **the same
penalty-encoded model** and, on the same argument, the same barrier. Confirming that on
hardware would close the argument; failing to confirm it would be more interesting still.

There is also a second cost the classical solvers do not pay, and it is the thing worth
measuring even if the quality result is unsurprising:

* **Minor embedding.** The reranking QUBO is *dense* -- the diversity term couples every
  pair of candidates -- while a QPU's topology is sparse. Embedding a 200-variable clique
  onto Pegasus needs chains of physical qubits per logical variable, and current
  hardware's largest embeddable clique is around 180 variables. n=200 may not fit at all.
* **Chain breaks.** When a chain's qubits disagree, the reading is repaired by majority
  vote, which silently manufactures a solution the hardware did not find. The break
  fraction is reported here for exactly that reason.

Both are reported in ``stats`` so the comparison can be made on total time-to-solution
rather than on QPU access time, which is the number that flatters quantum hardware by
excluding the classical work needed to use it.

Requires ``dwave-system`` and a configured API token::

    pip install dwave-system
    dwave config create        # or: export DWAVE_API_TOKEN=...
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..formulations.builder import build_problem
from .base import SolveResult, selection_from_sample, timed


class DWaveAnnealer:
    """Solve the reranking QUBO on a D-Wave QPU through Leap.

    The import is deferred to :meth:`solve` so that the package remains importable, and
    the test suite runnable, on a machine with neither ``dwave-system`` nor a token --
    which is every machine in CI.
    """

    name = "qubo_qpu"

    def __init__(
        self,
        num_reads: int = 100,
        annealing_time: float = 20.0,
        chain_strength: float | None = None,
        label: str = "qubo-rerank",
    ) -> None:
        self.num_reads = num_reads
        self.annealing_time = annealing_time
        self.chain_strength = chain_strength
        self.label = label

    def solve(self, problem: Any) -> SolveResult:
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "qubo_qpu needs dwave-system and a Leap API token.\n"
                "    pip install dwave-system\n"
                "    dwave config create\n"
                "Every other solver in this repo runs offline; this one does not."
            ) from exc

        stats: dict[str, Any] = {
            "num_reads": self.num_reads,
            "annealing_time_us": self.annealing_time,
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

        # Embedding is timed separately and deliberately. Reporting only QPU access time
        # is how quantum results are made to look fast: the classical work of finding an
        # embedding for a dense 200-variable problem can dominate the anneal by orders of
        # magnitude, and it is not optional.
        with timed(stats, "embed_and_sample_time"):
            sampler = EmbeddingComposite(DWaveSampler())
            kwargs: dict[str, Any] = {
                "num_reads": self.num_reads,
                "annealing_time": self.annealing_time,
                "label": self.label,
                "return_embedding": True,
            }
            if self.chain_strength is not None:
                kwargs["chain_strength"] = self.chain_strength
            sampleset = sampler.sample(rp.bqm, **kwargs)

        best = sampleset.first
        selection = selection_from_sample(best.sample, expected_k=problem.k)

        info = sampleset.info or {}
        timing = info.get("timing", {}) or {}
        embedding_context = info.get("embedding_context", {}) or {}
        embedding = embedding_context.get("embedding", {}) or {}

        if embedding:
            chain_lengths = [len(v) for v in embedding.values()]
            stats["max_chain_length"] = int(np.max(chain_lengths))
            stats["mean_chain_length"] = float(np.mean(chain_lengths))
            stats["physical_qubits"] = int(np.sum(chain_lengths))
            stats["qubit_overhead"] = float(np.sum(chain_lengths) / problem.n)

        if "chain_break_fraction" in sampleset.record.dtype.names:
            breaks = sampleset.record.chain_break_fraction
            stats["chain_break_fraction"] = float(np.mean(breaks))
            # A read whose chains all held needs no majority-vote repair. Chains that
            # broke were resolved by voting, which invents a state the hardware did not
            # actually return -- so the unbroken fraction is the honest denominator.
            stats["reads_without_chain_breaks"] = float(np.mean(breaks == 0.0))

        stats["qpu_access_time_us"] = timing.get("qpu_access_time")
        stats["qpu_anneal_time_us"] = timing.get("qpu_anneal_time_per_sample")
        stats["energy"] = float(best.energy)
        stats["cardinality"] = len(selection)
        stats["cardinality_ok"] = len(selection) == problem.k
        stats["energy_breakdown"] = rp.energy_breakdown(selection)

        return SolveResult(selection=selection, stats=stats)

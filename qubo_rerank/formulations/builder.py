"""Compose the full reranking QUBO from its parts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import dimod
import numpy as np

from .cardinality import build_cardinality
from .fairness import build_fairness
from .objective import build_objective


@dataclass
class RerankProblem:
    """A fully specified reranking instance.

    Attributes:
        bqm: the composed, optionally normalised BQM handed to a solver.
        n: number of candidates.
        k: required list size.
        parts: the individual BQMs, kept for debugging and for reporting how much
            each term contributes to the energy of a given selection.
    """

    bqm: dimod.BinaryQuadraticModel
    n: int
    k: int
    parts: dict[str, dimod.BinaryQuadraticModel] = field(default_factory=dict)

    def energy_breakdown(self, selection: Sequence[int]) -> dict[str, float]:
        """Energy contribution of each term for a given selection.

        Useful for sanity-checking results: if the fairness term contributes nothing
        when ``mu > 0``, the group labels are probably all identical.
        """
        sample = {i: 0 for i in range(self.n)}
        for i in selection:
            sample[int(i)] = 1
        return {name: float(part.energy(sample)) for name, part in self.parts.items()}


def suggest_strength(objective: dimod.BinaryQuadraticModel, margin: float = 2.0) -> float:
    """Heuristic penalty weight for the cardinality constraint.

    The penalty must exceed the largest objective gain available from *violating* the
    constraint. Violating it means flipping one variable, so the relevant bound is the
    maximum single-variable swing::

        max_v ( |h_v| + sum_u |J_uv| )

    not the total absolute weight of the whole objective. Using the total is safe in
    the sense that the cardinality constraint always holds, but it is far too large:
    after ``bqm.normalize()`` the objective's own differences get compressed toward
    zero and the sampler can no longer resolve them, so it returns an arbitrary
    feasible list rather than the best one. That failure is silent -- the list is the
    right length, it is just the wrong list.

    ``margin`` adds headroom above the bound.
    """
    max_swing = 0.0
    for v in objective.variables:
        swing = abs(objective.get_linear(v))
        swing += sum(abs(bias) for bias in objective.adj[v].values())
        max_swing = max(max_swing, swing)
    return margin * max(max_swing, 1.0)


def build_problem(
    relevance: np.ndarray,
    similarity: np.ndarray,
    k: int,
    groups: Sequence[int] | np.ndarray | None = None,
    lam: float = 1.0,
    mu: float = 0.0,
    targets: Mapping[int, float] | None = None,
    strength: float | None = None,
    normalize: bool = True,
) -> RerankProblem:
    """Build the complete reranking QUBO.

    ``H = H_objective + H_cardinality + H_fairness``

    Args:
        relevance: ``(n,)`` relevance scores for the candidate set.
        similarity: ``(n, n)`` pairwise item similarity.
        k: list size.
        groups: ``(n,)`` group label per item. Required when ``mu > 0``.
        lam: diversity weight.
        mu: fairness weight. ``0`` disables fairness (the baseline condition).
        targets: per-group target counts; defaults to equal shares.
        strength: cardinality penalty. Defaults to :func:`suggest_strength`.
        normalize: rescale the composed BQM so solver parameters transfer across
            instances. Follows the practice in CQFS (arXiv:2110.05089).

    Returns:
        A :class:`RerankProblem`.
    """
    relevance = np.asarray(relevance, dtype=float).ravel()
    n = relevance.shape[0]

    if k > n:
        raise ValueError(f"cannot select k={k} items from n={n} candidates")
    if mu != 0.0 and groups is None:
        raise ValueError("groups must be provided when mu != 0")

    objective = build_objective(relevance, similarity, lam=lam)

    if strength is None:
        strength = suggest_strength(objective)
    cardinality = build_cardinality(n, k, strength=strength)

    if groups is None:
        fairness = dimod.BinaryQuadraticModel(vartype="BINARY")
        for i in range(n):
            fairness.add_variable(i, 0.0)
    else:
        fairness = build_fairness(groups, k, mu=mu, targets=targets)

    combined = dimod.BinaryQuadraticModel(vartype="BINARY")
    for part in (objective, cardinality, fairness):
        combined.update(part)

    if normalize:
        combined.normalize()

    return RerankProblem(
        bqm=combined,
        n=n,
        k=k,
        parts={"objective": objective, "cardinality": cardinality, "fairness": fairness},
    )

"""Simulated Bifurcation: a continuous-dynamics Ising solver.

Goto, Tatsumura and Dixon, *Combinatorial optimization by simulating adiabatic
bifurcations in nonlinear Hamiltonian systems*, Science Advances 2019.

Every other QUBO solver here searches **discretely** -- it holds a bit vector and flips
bits. Simulated Bifurcation does not. It gives each variable a position ``x`` and a
momentum ``y``, integrates a Hamiltonian system in which a control parameter is ramped
until each ``x`` bifurcates toward one of two attractors, and reads the answer off the
signs at the end. Spins are only recovered at the last step; in between the state is a
point in continuous space that is not any particular bit vector.

**That is exactly why this solver belongs in this repo.** The headline finding here is
that penalty-encoded cardinality defeats single-spin-flip annealing, because two valid
k-item lists are never adjacent under bit flips -- the state between them has k±1 items
and pays the full penalty. That argument is about the *move set*, and it does not apply
to a method whose trajectory passes through the interior rather than along the edges of
the hypercube. A continuous solver can cross a barrier that a discrete one has to climb.

So this is a test of the mechanism, not just another solver. If Simulated Bifurcation
handles the penalty encoding well, the barrier is a property of discrete local search
and not of the formulation -- which is a sharper claim than "tabu also works", since
tabu escapes basins by bookkeeping while this one never enters them in the same sense.

**Ballistic vs discrete SB.** ``discrete=True`` (dSB) evaluates the coupling at
``sign(x)`` rather than ``x``. Goto et al. report it as more accurate on problems with
strongly varying coefficients, which is precisely this problem: the cardinality penalty
runs to ~738 against an objective of ~1. It is the default here for that reason.

No GPU is required. The inner loop is two matrix-vector products per step, and at
``n=200`` numpy on a CPU is not the bottleneck. The algorithm is, however, the natural
place to add one -- it is embarrassingly parallel across both variables and restarts,
which is why GPU implementations of SB exist and why it appears on this project's
roadmap as the scaling path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..formulations.builder import build_problem
from .base import SolveResult, timed


class SimulatedBifurcation:
    """Solve the reranking QUBO by integrating a bifurcating Hamiltonian system."""

    name = "qubo_sb"

    def __init__(
        self,
        num_restarts: int = 8,
        num_steps: int = 600,
        dt: float = 0.5,
        discrete: bool = True,
        seed: int | None = None,
    ) -> None:
        self.num_restarts = num_restarts
        self.num_steps = num_steps
        self.dt = dt
        self.discrete = discrete
        self.seed = seed

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _to_ising(h_qubo: np.ndarray, j_qubo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """QUBO over ``x in {0,1}`` -> Ising over ``s in {-1,+1}`` via ``x = (s+1)/2``.

        With ``E = h.x + 0.5 x'Jx`` and J symmetric with a zero diagonal, substituting
        and collecting terms gives ``E = h_ising.s + 0.5 s'J_ising s + const`` with
        ``J_ising = J/4`` and ``h_ising = h/2 + (J.1)/4``.

        The ``J/4`` is easy to get wrong. Expanding the quadratic term yields
        ``(1/8) s'Js``, and writing that in the standard ``0.5 s'J_ising s`` form needs
        ``J_ising = J/4``, not ``J/8``. The factor-of-two error is invisible in the
        output -- it is still a plausible Ising problem, just a *different* one -- so
        ``tests/test_bifurcation.py`` checks numerically that QUBO and Ising energies
        differ only by a constant.

        The constant itself is dropped: it shifts every energy equally and cannot change
        which state is lowest.
        """
        j_ising = j_qubo / 4.0
        h_ising = h_qubo / 2.0 + j_qubo.sum(axis=1) / 4.0
        return h_ising, j_ising

    def _run_once(
        self, h: np.ndarray, j: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """One bifurcation trajectory; returns spins in ``{-1, +1}``."""
        n = h.size

        # Scale so the drive is comparable to the restoring force at bifurcation.
        #
        # Goto et al. set c0 from the spread of J alone, which assumes the field is of
        # similar magnitude. That assumption fails badly here. Penalty-encoded
        # cardinality couples every pair positively, so converting to Ising leaves a
        # large *uniform* field -- mean 4.74 on this benchmark -- while J itself is
        # sparse and small (std 0.0009, since most entries are zero). Taking c0 from
        # std(J) then gives 38, and c0*|h| reaches 180 against a restoring force of 1:
        # every spin saturates in the same direction on the first few steps and the
        # method returns the empty set without having searched anything.
        #
        # Normalising by the worst-case per-variable drive keeps both terms in range,
        # and leaves the *relative* weight of penalty and objective untouched -- which
        # is the thing under test and must not be quietly rescaled away.
        drive = np.abs(j).sum(axis=1) + np.abs(h)
        scale = float(drive.max()) or 1.0
        h = h / scale
        j = j / scale
        c0 = 0.5

        x = 0.02 * (rng.random(n) - 0.5)   # near the unstable equilibrium at the origin
        y = 0.02 * (rng.random(n) - 0.5)

        for step in range(self.num_steps):
            # a(t) ramps 0 -> 1; the pitchfork bifurcation happens as it passes a0=1.
            a = step / self.num_steps

            coupling = np.sign(x) if self.discrete else x
            y += (-(1.0 - a) * x - c0 * (j @ coupling + h)) * self.dt
            x += 1.0 * y * self.dt

            # Inelastic walls at +/-1. Goto et al.'s modification: a variable that runs
            # past a wall is clamped and its momentum zeroed, which keeps trajectories
            # bounded and is what makes the method robust rather than merely stable.
            outside = np.abs(x) > 1.0
            if outside.any():
                x[outside] = np.sign(x[outside])
                y[outside] = 0.0

        spins = np.sign(x)
        spins[spins == 0] = 1.0
        return spins

    # -------------------------------------------------------------------- solve

    def solve(self, problem: Any) -> SolveResult:
        stats: dict[str, Any] = {
            "num_restarts": self.num_restarts,
            "num_steps": self.num_steps,
            "discrete": self.discrete,
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

        n = problem.n
        h_qubo = np.zeros(n)
        j_qubo = np.zeros((n, n))
        for v, bias in rp.bqm.linear.items():
            h_qubo[int(v)] = bias
        for (u, v), bias in rp.bqm.quadratic.items():
            j_qubo[int(u), int(v)] = bias
            j_qubo[int(v), int(u)] = bias

        h_ising, j_ising = self._to_ising(h_qubo, j_qubo)
        rng = np.random.default_rng(self.seed)

        best_selection: list[int] = []
        best_energy = np.inf
        cardinalities = []

        with timed(stats, "solve_time"):
            for _ in range(self.num_restarts):
                spins = self._run_once(h_ising, j_ising, rng)
                x_bits = (spins + 1.0) / 2.0
                cardinalities.append(int(x_bits.sum()))

                # The BQM's constant offset is included. Without it this solver reported
                # an energy on a different scale from every other solver here -- all of
                # which surface `sampleset.first.energy` or `bqm.energy(...)`, both of
                # which carry the offset -- so `stats["energy"]` was silently
                # incomparable across methods. The penalty-encoded cardinality term
                # contributes an offset of P*k^2, which is large: 2.76 on a 30-item
                # instance whose objective spans hundredths.
                energy = float(
                    h_qubo @ x_bits + 0.5 * x_bits @ (j_qubo @ x_bits) + rp.bqm.offset
                )
                if energy < best_energy:
                    best_energy = energy
                    best_selection = [int(i) for i in np.flatnonzero(x_bits > 0.5)]

        # Cardinality is *not* repaired. The penalty is supposed to enforce it, and
        # whether it does is the question this repo is asking -- padding or truncating
        # here would hide the answer inside a solver wrapper. The caller sees the true
        # length and `cardinality_ok` says whether the constraint held.
        stats["energy"] = best_energy
        stats["cardinality"] = len(best_selection)
        stats["cardinality_ok"] = len(best_selection) == problem.k
        stats["mean_cardinality"] = float(np.mean(cardinalities))
        stats["energy_breakdown"] = rp.energy_breakdown(best_selection)

        return SolveResult(selection=best_selection, stats=stats)

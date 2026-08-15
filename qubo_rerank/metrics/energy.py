"""Energy and carbon measurement.

Butt's projects 53917 and 53926 ask for *"estimated energy usage"* alongside accuracy
and execution time. ``codecarbon`` measures process-level power draw and converts it
to CO2 equivalent using the local grid's carbon intensity.

Two honesty notes that belong in any write-up of these numbers:

* codecarbon estimates from CPU/GPU TDP and utilisation. It is not a hardware power
  meter. The Green RecSys study (Wegmeth et al., ACM TORS 2025) used a physical meter
  and is the right citation for rigorous figures.
* At the per-solve timescale used here the absolute values are tiny and dominated by
  measurement noise. They are meaningful as a *relative* comparison between solvers on
  identical hardware, and meaningless as absolute claims.

The tracker is optional: if codecarbon is not installed, measurement degrades to
wall-clock only rather than failing, so the core library stays installable without it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - import guard
    from codecarbon import EmissionsTracker

    _HAS_CODECARBON = True
except ImportError:  # pragma: no cover
    EmissionsTracker = None  # type: ignore[assignment]
    _HAS_CODECARBON = False


@dataclass
class EnergyReading:
    """One measurement window."""

    seconds: float
    kwh: float | None = None
    co2_kg: float | None = None
    measured: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seconds": self.seconds,
            "kwh": self.kwh,
            "co2_kg": self.co2_kg,
            "energy_measured": self.measured,
        }


class measure_energy:
    """Context manager recording elapsed time and, if available, energy and CO2.

    Usage::

        with measure_energy("qubo_sa") as m:
            solver.solve(instance)
        print(m.reading.as_dict())
    """

    def __init__(self, label: str = "solve", enabled: bool = True) -> None:
        self.label = label
        self.enabled = enabled and _HAS_CODECARBON
        self.reading: EnergyReading | None = None
        self._tracker: Any = None

    def __enter__(self) -> measure_energy:
        if self.enabled:
            # log_level="error" keeps codecarbon's very chatty INFO output out of
            # experiment logs; save_to_file=False avoids littering emissions.csv
            # next to every run.
            self._tracker = EmissionsTracker(
                project_name=self.label, log_level="error", save_to_file=False
            )
            self._tracker.start()

        # Started *after* the tracker: EmissionsTracker.start() probes the hardware and
        # costs a few seconds. Timing from before it charged that constant to whichever
        # solver was being measured, which made the sub-second baselines look like they
        # took five seconds while their kWh said otherwise.
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        seconds = time.perf_counter() - self._t0
        kwh = co2 = None
        measured = False

        if self._tracker is not None:
            try:
                co2 = float(self._tracker.stop())
                kwh = float(getattr(self._tracker, "_total_energy", 0.0) or 0.0)
                measured = True
            except Exception:  # pragma: no cover - codecarbon is best-effort
                measured = False

        self.reading = EnergyReading(
            seconds=seconds, kwh=kwh, co2_kg=co2, measured=measured
        )


def codecarbon_available() -> bool:
    """Whether energy measurement is active in this environment."""
    return _HAS_CODECARBON

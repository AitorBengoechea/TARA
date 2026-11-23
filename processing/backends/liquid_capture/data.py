"""Validated in-memory records for native liquid analysis."""
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LiquidKernel:
    """Completed per-atom kernel; alpha uses neutron mass and beta is signed.

    Values are unsymmetrized dimensionless S, indexed [beta, alpha].
    ``fraction`` is the original atomic fraction, never applied to S.
    """
    label: str
    alpha: np.ndarray
    beta: np.ndarray
    sab: np.ndarray
    mass_amu: float
    temperature: float
    bound_xs: float
    fraction: float
    suggested_emax: float

    def __post_init__(self):
        """Reject invalid captures before any Analyzer interpolation."""
        a, b, s = (np.array(x, dtype=float, copy=True)
                   for x in (self.alpha, self.beta, self.sab))
        if (a.ndim != 1 or b.ndim != 1 or len(a) < 2 or len(b) < 3
                or s.shape != (len(b), len(a))):
            raise ValueError("Invalid liquid kernel dimensions")
        if (not all(np.isfinite(x).all() for x in (a, b, s))
                or np.any(np.diff(a) <= 0) or np.any(np.diff(b) <= 0)
                or a[0] < 0 or not b[0] < 0 < b[-1] or np.any(s < 0)):
            raise ValueError("Invalid liquid kernel coordinates or values")
        scalars = (self.mass_amu, self.temperature, self.bound_xs, self.fraction)
        if not all(np.isfinite(x) and x > 0 for x in scalars) or self.fraction > 1:
            raise ValueError("Invalid liquid component metadata")
        if not np.isfinite(self.suggested_emax) or self.suggested_emax <= 0:
            raise ValueError("Liquid capture has no positive supported energy limit")
        for name, value in zip(("alpha", "beta", "sab"), (a, b, s)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass
class LiquidAnalysisState:
    """Loaded child results; no live C++ objects or temporary files escape."""
    target_lbl: str
    kernels: dict[str, LiquidKernel]
    energy_grid: np.ndarray
    pdos: np.ndarray
    backend_energy_grid: np.ndarray
    backend_pdos: np.ndarray
    xs_energy: np.ndarray
    xs_scattering: np.ndarray
    xs_absorption: np.ndarray
    vdos_analysis: dict[str, float]
    provenance: dict[str, Any]
    target_mass_amu: float | None = None

    @property
    def mass_amu(self):
        """Return the mass of the selected scattering component."""
        if self.target_mass_amu is not None:
            return self.target_mass_amu
        return self.kernels[self.target_lbl].mass_amu

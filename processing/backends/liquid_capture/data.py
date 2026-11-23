"""Validated in-memory records for native liquid analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LiquidKernel:
    """Completed per-atom kernel with neutron-mass alpha and signed beta.

    Parameters
    ----------
    label : str
        NCMAT element or site label (e.g., ``"H"``).
    alpha : numpy.ndarray
        Strictly increasing 1-D dimensionless momentum-transfer grid using
        NCrystal's neutron-mass convention.
    beta : numpy.ndarray
        Strictly increasing 1-D signed dimensionless energy-transfer grid
        spanning both negative and positive values.
    sab : numpy.ndarray
        2-D unsymmetrized dimensionless scattering law indexed as
        ``sab[beta_index, alpha_index]``.
    mass_amu : float
        Scattering-atom mass in atomic mass units.
    temperature : float
        Material temperature in kelvin.
    bound_xs : float
        Bound scattering cross section in barns.
    fraction : float
        Original atomic fraction in `(0, 1]`, never multiplied into ``sab``.
    suggested_emax : float
        Maximum incident neutron energy in eV supported by the kernel grid.
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

    def __post_init__(self) -> None:
        """Validate dimensions, physical bounds, and freeze kernel arrays.

        Raises
        ------
        ValueError
            If array dimensions, monotonicity, finiteness, sign coverage, or
            scalar physical metadata are invalid.
        """
        a, b, s = (
            np.array(x, dtype=float, copy=True)
            for x in (self.alpha, self.beta, self.sab)
        )
        if (
            a.ndim != 1
            or b.ndim != 1
            or len(a) < 2
            or len(b) < 3
            or s.shape != (len(b), len(a))
        ):
            raise ValueError("Invalid liquid kernel dimensions")
        if (
            not all(np.isfinite(x).all() for x in (a, b, s))
            or np.any(np.diff(a) <= 0)
            or np.any(np.diff(b) <= 0)
            or a[0] < 0
            or not (b[0] < 0 < b[-1])
            or np.any(s < 0)
        ):
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
    """Loaded child capture state; no live C++ objects or temp files escape.

    Parameters
    ----------
    target_lbl : str
        Selected target element label for differential kinematics.
    kernels : dict of str to LiquidKernel
        Captured liquid kernels keyed by element label.
    energy_grid : numpy.ndarray
        Full Core VDOS energy grid in eV.
    pdos : numpy.ndarray
        Full Core VDOS density values in 1/eV.
    backend_energy_grid : numpy.ndarray
        NCrystal-expanded positive VDOS energy grid in eV.
    backend_pdos : numpy.ndarray
        NCrystal-expanded VDOS density values.
    xs_energy : numpy.ndarray
        Incident neutron energy grid in eV for cross sections.
    xs_scattering : numpy.ndarray
        Selected active-species scattering cross section in barns per material atom.
    xs_absorption : numpy.ndarray
        Whole-material absorption cross section in barns per material atom.
    vdos_analysis : dict of str to float
        Solid-like VDOS diagnostics returned by ``analyseVDOS()``.
    provenance : dict of str to Any
        Complete capture, component, and representation metadata.
    """

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

    @property
    def mass_amu(self) -> float:
        """Return the atomic mass in amu of the selected target species.

        Returns
        -------
        float
            Target atomic mass in atomic mass units.
        """
        return self.kernels[self.target_lbl].mass_amu
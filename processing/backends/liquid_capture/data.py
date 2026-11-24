"""Validated in-memory records for native liquid analysis."""
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LiquidKernel:
    """Completed per-atom kernel; alpha uses neutron mass and beta is signed.

    Parameters
    ----------
    label : str
        NCrystal label of the scattering species.
    alpha : np.ndarray
        Dimensionless momentum-transfer coordinates scaled with the neutron
        mass.
    beta : np.ndarray
        Signed dimensionless energy-transfer coordinates.
    sab : np.ndarray
        Unsymmetrized dimensionless scattering-law values indexed by beta,
        then alpha.
    mass_amu : float
        Target mass in atomic mass units.
    temperature : float
        Evaluation temperature in kelvin.
    bound_xs : float
        Bound scattering cross section in barns.
    fraction : float
        Atomic fraction of the scattering species.
    suggested_emax : float
        Suggested upper incident-energy limit in eV.

    Notes
    -----
    Values are unsymmetrized dimensionless S, indexed [beta, alpha].
    ``fraction`` is the original atomic fraction, never applied to S.
    """
    label: str

    # Scattering-law coordinates and values.
    alpha: np.ndarray
    beta: np.ndarray
    sab: np.ndarray

    # Component metadata and supported energy range.
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
    """Loaded child results; no live C++ objects or temporary files escape.

    Parameters
    ----------
    target_lbl : str
        Label of the evaluated scattering species.
    kernels : dict[str, LiquidKernel]
        Completed liquid scattering kernels keyed by species label.
    energy_grid : np.ndarray
        Energy coordinates of the source vibrational spectrum in eV.
    pdos : np.ndarray
        Source vibrational density evaluated at the energy coordinates.
    backend_energy_grid : np.ndarray
        Energy coordinates of the VDOS passed to NCrystal in eV.
    backend_pdos : np.ndarray
        VDOS density evaluated on the NCrystal input grid.
    xs_energy : np.ndarray
        Incident-energy grid for cross-section values in eV.
    xs_scattering : np.ndarray
        Native scattering cross sections in barns per atom.
    xs_absorption : np.ndarray
        Native absorption cross sections in barns per atom.
    vdos_analysis : dict[str, float]
        Thermodynamic quantities evaluated from the native VDOS.
    provenance : dict[str, Any]
        Metadata describing the source and settings of the capture.
    target_mass_amu : float or None, optional
        Explicit target mass in atomic mass units. Defaults to the kernel
        mass when read. Default is None.
    """

    # Target component and scattering kernels.
    target_lbl: str
    kernels: dict[str, LiquidKernel]

    # Core and backend vibrational spectra.
    energy_grid: np.ndarray
    pdos: np.ndarray
    backend_energy_grid: np.ndarray
    backend_pdos: np.ndarray

    # Cross-section grids and values.
    xs_energy: np.ndarray
    xs_scattering: np.ndarray
    xs_absorption: np.ndarray

    # Derived quantities and capture metadata.
    vdos_analysis: dict[str, float]
    provenance: dict[str, Any]
    target_mass_amu: float | None = None

    @property
    def mass_amu(self):
        """Return the mass of the selected scattering component."""
        if self.target_mass_amu is not None:
            return self.target_mass_amu
        return self.kernels[self.target_lbl].mass_amu

"""Validated in-memory records for native liquid analysis."""
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LiquidKernel:
    """Completed per-atom kernel; alpha uses neutron mass and beta is signed.

    Values are unsymmetrized dimensionless S, indexed [beta, alpha].
    ``fraction`` is the original atomic fraction, never applied to S.

    Parameters
    ----------
    label : str
        Native species label.
    alpha, beta : numpy.ndarray
        Dimensionless transfer coordinates using the neutron mass and signed beta.
    sab : numpy.ndarray
        Unsymmetrized S(alpha, beta), indexed as (beta, alpha).
    mass_amu : float
        Target mass in atomic mass units.
    temperature : float
        Material temperature in kelvin.
    bound_xs : float
        Bound scattering cross section in barns.
    fraction : float
        Original atomic fraction of this species.
    suggested_emax : float
        Supported incident energy limit in eV.
    """
    label: str

    # Kernel axes and ordinates.
    alpha: np.ndarray
    beta: np.ndarray
    sab: np.ndarray

    # Species metadata and supported energy range.
    mass_amu: float
    temperature: float
    bound_xs: float
    fraction: float
    suggested_emax: float

    def __post_init__(self):
        """Reject invalid captures before any Analyzer interpolation."""
        a, b, s = (np.array(x, dtype=float, copy=True)
                   for x in (self.alpha, self.beta, self.sab))
        self._validate_kernel_arrays(a, b, s)
        self._validate_component_metadata()
        # Make the validated copies immutable before publishing them.
        for name, value in zip(("alpha", "beta", "sab"), (a, b, s)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @staticmethod
    def _validate_kernel_arrays(a, b, s):
        """Validate the (beta, alpha) kernel layout and values.

        Parameters
        ----------
        a, b : numpy.ndarray
            Dimensionless alpha and signed beta coordinates.
        s : numpy.ndarray
            Kernel values indexed by beta and alpha.
        """
        if (a.ndim != 1 or b.ndim != 1 or len(a) < 2 or len(b) < 3
                or s.shape != (len(b), len(a))):
            raise ValueError("Invalid liquid kernel dimensions")
        if (not all(np.isfinite(x).all() for x in (a, b, s))
                or np.any(np.diff(a) <= 0) or np.any(np.diff(b) <= 0)
                or a[0] < 0 or not b[0] < 0 < b[-1] or np.any(s < 0)):
            raise ValueError("Invalid liquid kernel coordinates or values")

    def _validate_component_metadata(self):
        """Check scalar properties of the captured species.
        """
        scalars = (self.mass_amu, self.temperature, self.bound_xs, self.fraction)
        if not all(np.isfinite(x) and x > 0 for x in scalars) or self.fraction > 1:
            raise ValueError("Invalid liquid component metadata")
        if not np.isfinite(self.suggested_emax) or self.suggested_emax <= 0:
            raise ValueError("Liquid capture has no positive supported energy limit")


@dataclass
class LiquidAnalysisState:
    """Loaded child results; no live C++ objects or temporary files escape.

    Parameters
    ----------
    target_lbl : str
        Species selected for kernel analysis.
    kernels : dict of str to LiquidKernel
        Captured components indexed by species label.
    energy_grid, pdos : numpy.ndarray
        Core vibrational spectrum and its energy coordinates in eV.
    backend_energy_grid, backend_pdos : numpy.ndarray
        Backend VDOS and its energy coordinates in eV.
    xs_energy : numpy.ndarray
        Incident energies in eV.
    xs_scattering, xs_absorption : numpy.ndarray
        Scattering and absorption cross sections in barns per original material atom.
    vdos_analysis : dict of str to float
        Thermodynamic quantities derived from the target VDOS.
    provenance : dict of str to Any
        Capture metadata and material identity.
    target_mass_amu : float or None, optional
        Explicit target mass in atomic mass units, used when no kernel is loaded.
    """
    target_lbl: str
    kernels: dict[str, LiquidKernel]

    # Core and backend vibrational spectra.
    energy_grid: np.ndarray
    pdos: np.ndarray
    backend_energy_grid: np.ndarray
    backend_pdos: np.ndarray

    # Cross-section energy grid and values.
    xs_energy: np.ndarray
    xs_scattering: np.ndarray
    xs_absorption: np.ndarray

    # Derived quantities and capture provenance.
    vdos_analysis: dict[str, float]
    provenance: dict[str, Any]
    target_mass_amu: float | None = None

    @property
    def mass_amu(self):
        """Return the mass of the selected scattering component.

        Returns
        -------
        float
            Target mass in atomic mass units.
        """
        if self.target_mass_amu is not None:
            return self.target_mass_amu
        return self.kernels[self.target_lbl].mass_amu

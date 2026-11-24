"""
Data structure for passing initialized NCrystal memory pointers.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class NCrystalEngine:
    """Hold loaded NCrystal state for zero-file-I/O Analyzer parsing.

    Parameters
    ----------
    filepath : str
        Path identifying the transient NCMAT resource.
    temperature : float
        Evaluation temperature in kelvin.
    target_lbl : str
        Label of the evaluated scattering species.
    mat : Any
        Loaded NCrystal material object.
    nuc_data : Any
        NCrystal atom data for the target species.
    mass_amu : float
        Target mass in atomic mass units.
    energy_grid : np.ndarray
        Energy coordinates of the source vibrational spectrum in eV.
    pdos : np.ndarray
        Source vibrational density evaluated at the energy coordinates.
    backend_energy_grid : np.ndarray or None, optional
        Energy coordinates of the VDOS passed to NCrystal in eV. Default is
        None.
    backend_pdos : np.ndarray or None, optional
        VDOS density evaluated on the NCrystal input grid. Default is None.
    scatter_inelastic : Any or None, optional
        Native inelastic scattering handle. Default is None.
    scatter_coherent_elastic : Any or None, optional
        Native coherent elastic scattering handle. Default is None.
    scatter_incoherent_elastic : Any or None, optional
        Native incoherent elastic scattering handle. Default is None.
    liquid_data : Any or None, optional
        Native liquid analysis state, when available. Default is None.

    Notes
    -----
    ``filepath`` identifies the transient NCMAT resource owned by Processing.
    ``energy_grid`` and ``pdos`` retain the full Core arrays for thermodynamics.
    ``backend_energy_grid`` and ``backend_pdos`` mirror the atom-specific VDOS
    written to NCMAT, and are used exclusively for NCrystal kernel extraction.
    The optional scattering handles provide component cross sections from
    memory. The parser performs no I/O.
    """

    # Material source and loaded target.
    filepath: str
    temperature: float
    target_lbl: str
    mat: Any
    nuc_data: Any
    mass_amu: float

    # Core and backend vibrational spectra.
    energy_grid: np.ndarray
    pdos: np.ndarray
    backend_energy_grid: Optional[np.ndarray] = None
    backend_pdos: Optional[np.ndarray] = None

    # Native scattering handles and liquid state.
    scatter_inelastic: Optional[Any] = None
    scatter_coherent_elastic: Optional[Any] = None
    scatter_incoherent_elastic: Optional[Any] = None

    # Material state information
    liquid_data: Optional[Any] = None

"""
Data structure for passing initialized NCrystal memory pointers.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class NCrystalEngine:
    """Hold loaded NCrystal state for zero-file-I/O Analyzer parsing.

    ``filepath`` identifies the transient NCMAT resource owned by Processing.
    ``energy_grid`` and ``pdos`` retain the full Core arrays for thermodynamics.
    ``backend_energy_grid`` and ``backend_pdos`` mirror the atom-specific VDOS
    written to NCMAT, and are used exclusively for NCrystal kernel extraction.
    The optional scattering handles provide component cross sections from
    memory. The parser performs no I/O.
    """

    filepath: str
    temperature: float
    target_lbl: str
    mat: Any
    nuc_data: Any
    mass_amu: float
    energy_grid: np.ndarray
    pdos: np.ndarray
    backend_energy_grid: Optional[np.ndarray] = None
    backend_pdos: Optional[np.ndarray] = None
    scatter_inelastic: Optional[Any] = None
    scatter_coherent_elastic: Optional[Any] = None
    scatter_incoherent_elastic: Optional[Any] = None

    liquid_data: Optional[Any] = None

"""
Data structure for passing initialized NJOY tape data and memory objects.
"""
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

@dataclass
class NjoyEngine:
    """Holds the fully parsed ENDF tape trees (via endf_parserpy) and essential
    metadata to prevent redundant file I/O during parsing.

    Parameters
    ----------
    base_endf_file : str
        Path to the original neutron evaluation used for THERMR.
    temperature : float
        Evaluation temperature in kelvin.
    mass_amu : float
        Target mass in atomic mass units.
    energy_grid : np.ndarray
        Energy coordinates of the source vibrational spectrum in eV.
    pdos : np.ndarray
        Source vibrational density evaluated at the energy coordinates.
    leapr_tree : Any
        Parsed LEAPR ENDF tree containing the MF7 scattering law.
    thermr_tree : Any or None, optional
        Parsed THERMR ENDF tree containing the MF3 cross sections. Default
        is None.
    njoy_output_text : str or None, optional
        Captured NJOY output used for diagnostics and text extraction.
        Default is None.
    """

    # Evaluation provenance and conditions.
    base_endf_file: str    # The original ENDF evaluation used for THERMR
    temperature: float
    mass_amu: float

    # Vibrational spectrum.
    energy_grid: np.ndarray
    pdos: np.ndarray
    
    # Parsed tapes and execution diagnostics.
    leapr_tree: Any        # Contains S(a,b) data (MF=7)
    thermr_tree: Optional[Any] = None # Contains the integrated Cross Sections (MF=3)
    
    # Raw NJOY execution output for debugging or fallback text extraction
    njoy_output_text: Optional[str] = None
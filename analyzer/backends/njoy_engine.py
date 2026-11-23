"""
Data structure for passing initialized NJOY tape data and memory objects.
"""
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

@dataclass
class NjoyEngine:
    """
    Holds the fully parsed ENDF tape trees (via endf_parserpy) and essential 
    metadata to prevent redundant file I/O during parsing.
    """
    base_endf_file: str    # The original ENDF evaluation used for THERMR
    temperature: float
    mass_amu: float
    energy_grid: np.ndarray
    pdos: np.ndarray
    
    # Parsed endf_parserpy objects
    leapr_tree: Any        # Contains S(a,b) data (MF=7)
    thermr_tree: Optional[Any] = None # Contains the integrated Cross Sections (MF=3)
    
    # Raw NJOY execution output for debugging or fallback text extraction
    njoy_output_text: Optional[str] = None
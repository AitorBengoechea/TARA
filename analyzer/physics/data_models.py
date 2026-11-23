"""
Strict data models unifying physical evaluations across different nuclear data backends.
"""
from dataclasses import dataclass
import pandas as pd
from typing import Optional, Dict


@dataclass
class ThermodynamicsData:
    """Thermodynamic scalars extracted or computed from a phonon spectrum.
    All energy units are in eV, temperature in K, and distances in Angstroms.
    """
    teff: float
    msd: float
    debye_waller: float
    b_factor: float
    cv: float
    entropy: float
    zero_point_energy: float
    free_energy: float
    internal_energy: float
    log_moment: float


@dataclass
class DifferentialData:
    """
    Double-differential scattering matrices and intermediate multi-phonon kernels.

    ``sab`` always stores Analyzer's canonical full ``S(alpha, beta)`` table:
    alpha on the row index and signed beta on the columns. Backend-native
    compact representations are normalized before assignment.
    """
    gn: Optional[pd.Series] = None
    sab: Optional[pd.DataFrame] = None
    dynamic_structure_factor: Optional[Dict[float, pd.DataFrame]] = None


@dataclass
class IntegralData:
    """Integrated cross-sections and interaction probabilities mapped over energy.
    All cross-sections are returned in barns.
    """
    # Cross Sections
    xs_total: Optional[pd.Series] = None
    xs_inelastic: Optional[pd.Series] = None
    xs_coherent_elastic: Optional[pd.Series] = None
    xs_incoherent_elastic: Optional[pd.Series] = None
    xs_absorption: Optional[pd.Series] = None

    # Kinematic Distributions (Multi-Ein support)
    scattering_kernel: Optional[Dict[float, pd.Series]] = None
    angular_distribution: Optional[Dict[float, pd.Series]] = None
    cosine_distribution: Optional[pd.DataFrame] = None
    energy_distribution: Optional[pd.DataFrame] = None
    
    # Ratios
    upscattering_ratio: Optional[pd.Series] = None
    downscattering_ratio: Optional[pd.Series] = None
    elastic_ratio: Optional[pd.Series] = None


@dataclass
class PhysicsData:
    """
    The central unifying container representing the full physical evaluation 
    of a single perturbed sample, bridging the gap between NJOY and NCrystal.
    """
    sample_index: int
    thermo: ThermodynamicsData
    diff: DifferentialData
    integ: IntegralData
    metadata: Optional[dict] = None

@dataclass
class EnsemblePhysicsData:
    """
    A unified pure-state container for aggregated ensemble matrices.
    """
    # 1D Integrals & Scalars aggregated into 2D DataFrames
    thermo: Optional[pd.DataFrame] = None
    xs_total: Optional[pd.DataFrame] = None
    xs_inelastic: Optional[pd.DataFrame] = None
    xs_coherent_elastic: Optional[pd.DataFrame] = None
    xs_incoherent_elastic: Optional[pd.DataFrame] = None
    xs_absorption: Optional[pd.DataFrame] = None
    upscattering_ratio: Optional[pd.DataFrame] = None
    downscattering_ratio: Optional[pd.DataFrame] = None
    elastic_ratio: Optional[pd.DataFrame] = None
    
    # Multi-Ein Kinematic Distributions aggregated into Dictionaries of 2D DataFrames
    scattering_kernel: Optional[Dict[float, pd.DataFrame]] = None
    angular_distribution: Optional[Dict[float, pd.DataFrame]] = None
    cosine_distribution: Optional[Dict[float, pd.DataFrame]] = None
    energy_distribution: Optional[Dict[float, pd.DataFrame]] = None

    metadata: Optional[dict] = None

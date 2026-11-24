"""
Strict data models unifying physical evaluations across different nuclear data backends.
"""
from dataclasses import dataclass
import pandas as pd
from typing import Optional, Dict


@dataclass
class ThermodynamicsData:
    """Thermodynamic scalars extracted or computed from a phonon spectrum. All
    energy units are in eV, temperature in K, and distances in Angstroms.

    Parameters
    ----------
    teff : float
        Effective temperature.
    msd : float
        Mean-square displacement.
    debye_waller : float
        Debye–Waller coefficient.
    b_factor : float
        Crystallographic temperature factor.
    cv : float
        Constant-volume heat capacity.
    entropy : float
        Entropy.
    zero_point_energy : float
        Zero-point vibrational energy.
    free_energy : float
        Vibrational free energy.
    internal_energy : float
        Vibrational internal energy.
    log_moment : float
        Logarithmic moment of the vibrational spectrum.
    """

    # Effective temperature and displacement quantities.
    teff: float
    msd: float
    debye_waller: float
    b_factor: float

    # Thermodynamic quantities and spectral moment.
    cv: float
    entropy: float
    zero_point_energy: float
    free_energy: float
    internal_energy: float
    log_moment: float


@dataclass
class DifferentialData:
    """Double-differential scattering matrices and intermediate multi-phonon
    kernels.

    Parameters
    ----------
    gn : pd.Series or None, optional
        Intermediate multiphonon kernel. Default is None.
    sab : pd.DataFrame or None, optional
        Full scattering-law table. Default is None.
    dynamic_structure_factor : dict[float, pd.DataFrame] or None, optional
        Dynamic structure factor tables keyed by incident energy. Default is
        None.

    Notes
    -----
    ``sab`` always stores Analyzer's canonical full ``S(alpha, beta)`` table:
    alpha on the row index and signed beta on the columns. Backend-native
    compact representations are normalized before assignment.
    """

    # Multiphonon kernel and differential scattering tables.
    gn: Optional[pd.Series] = None
    sab: Optional[pd.DataFrame] = None
    dynamic_structure_factor: Optional[Dict[float, pd.DataFrame]] = None


@dataclass
class IntegralData:
    """Integrated cross-sections and interaction probabilities mapped over
    energy. All cross-sections are returned in barns.

    Parameters
    ----------
    xs_total : pd.Series or None, optional
        Total scattering cross section. Default is None.
    xs_inelastic : pd.Series or None, optional
        Inelastic scattering cross section. Default is None.
    xs_coherent_elastic : pd.Series or None, optional
        Coherent elastic scattering cross section. Default is None.
    xs_incoherent_elastic : pd.Series or None, optional
        Incoherent elastic scattering cross section. Default is None.
    xs_absorption : pd.Series or None, optional
        Absorption cross section. Default is None.
    scattering_kernel : dict[float, pd.Series] or None, optional
        Double-differential scattering data keyed by incident energy.
        Default is None.
    angular_distribution : dict[float, pd.Series] or None, optional
        Angular distributions keyed by incident energy. Default is None.
    cosine_distribution : pd.DataFrame or None, optional
        Distribution in scattering-angle cosine. Default is None.
    energy_distribution : pd.DataFrame or None, optional
        Distribution in outgoing neutron energy. Default is None.
    upscattering_ratio : pd.Series or None, optional
        Fraction of scattering events classified as upscattering. Default is
        None.
    downscattering_ratio : pd.Series or None, optional
        Fraction of scattering events classified as downscattering. Default
        is None.
    elastic_ratio : pd.Series or None, optional
        Fraction of scattering events classified as elastic scattering.
        Default is None.
    """
    # Cross sections.
    xs_total: Optional[pd.Series] = None
    xs_inelastic: Optional[pd.Series] = None
    xs_coherent_elastic: Optional[pd.Series] = None
    xs_incoherent_elastic: Optional[pd.Series] = None
    xs_absorption: Optional[pd.Series] = None

    # Scattering distributions.
    scattering_kernel: Optional[Dict[float, pd.Series]] = None
    angular_distribution: Optional[Dict[float, pd.Series]] = None
    cosine_distribution: Optional[pd.DataFrame] = None
    energy_distribution: Optional[pd.DataFrame] = None
    
    # Scattering ratios.
    upscattering_ratio: Optional[pd.Series] = None
    downscattering_ratio: Optional[pd.Series] = None
    elastic_ratio: Optional[pd.Series] = None


@dataclass
class PhysicsData:
    """The central unifying container representing the full physical evaluation
    of a single perturbed sample, bridging the gap between NJOY and
    NCrystal.

    Parameters
    ----------
    sample_index : int
        Index identifying the sample. The Analyzer uses -1 for the
        reference.
    thermo : ThermodynamicsData
        Thermodynamic quantities.
    diff : DifferentialData
        Differential scattering quantities.
    integ : IntegralData
        Integrated and derived scattering quantities.
    metadata : dict or None, optional
        Additional evaluation metadata. Default is None.
    """

    # Sample identity and evaluated observables.
    sample_index: int
    thermo: ThermodynamicsData
    diff: DifferentialData
    integ: IntegralData
    metadata: Optional[dict] = None

@dataclass
class EnsemblePhysicsData:
    """A unified pure-state container for aggregated ensemble matrices.

    Parameters
    ----------
    thermo : pd.DataFrame or None, optional
        Thermodynamic values assembled across samples. Default is None.
    xs_total : pd.DataFrame or None, optional
        Total scattering cross section. Default is None.
    xs_inelastic : pd.DataFrame or None, optional
        Inelastic scattering cross section. Default is None.
    xs_coherent_elastic : pd.DataFrame or None, optional
        Coherent elastic scattering cross section. Default is None.
    xs_incoherent_elastic : pd.DataFrame or None, optional
        Incoherent elastic scattering cross section. Default is None.
    xs_absorption : pd.DataFrame or None, optional
        Absorption cross section. Default is None.
    upscattering_ratio : pd.DataFrame or None, optional
        Fraction of scattering events classified as upscattering. Default is
        None.
    downscattering_ratio : pd.DataFrame or None, optional
        Fraction of scattering events classified as downscattering. Default
        is None.
    elastic_ratio : pd.DataFrame or None, optional
        Fraction of scattering events classified as elastic scattering.
        Default is None.
    scattering_kernel : dict[float, pd.DataFrame] or None, optional
        Double-differential scattering data keyed by incident energy.
        Default is None.
    angular_distribution : dict[float, pd.DataFrame] or None, optional
        Angular distributions keyed by incident energy. Default is None.
    cosine_distribution : dict[float, pd.DataFrame] or None, optional
        Distribution in scattering-angle cosine. Default is None.
    energy_distribution : dict[float, pd.DataFrame] or None, optional
        Distribution in outgoing neutron energy. Default is None.
    metadata : dict or None, optional
        Additional evaluation metadata. Default is None.
    """
    # Thermodynamic and cross-section matrices.
    thermo: Optional[pd.DataFrame] = None
    xs_total: Optional[pd.DataFrame] = None
    xs_inelastic: Optional[pd.DataFrame] = None
    xs_coherent_elastic: Optional[pd.DataFrame] = None
    xs_incoherent_elastic: Optional[pd.DataFrame] = None
    xs_absorption: Optional[pd.DataFrame] = None

    # Scattering-ratio matrices.
    upscattering_ratio: Optional[pd.DataFrame] = None
    downscattering_ratio: Optional[pd.DataFrame] = None
    elastic_ratio: Optional[pd.DataFrame] = None
    
    # Scattering distributions across samples.
    scattering_kernel: Optional[Dict[float, pd.DataFrame]] = None
    angular_distribution: Optional[Dict[float, pd.DataFrame]] = None
    cosine_distribution: Optional[Dict[float, pd.DataFrame]] = None
    energy_distribution: Optional[Dict[float, pd.DataFrame]] = None

    # Ensemble metadata.
    metadata: Optional[dict] = None

"""
Configuration gateway for the TARA stochastic sampling engine.
"""
import warnings
import math
from collections.abc import Mapping
from numbers import Real
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


class TaraConfigError(Exception):
    """Raised when TARA configuration parameters are invalid or physically inconsistent."""
    pass


@dataclass
class VibrationSamplingConfig:
    """
    Bounds and sampling strategies for vibrational spectra.

    This class enforces mandatory perturbation limits and algorithmic defaults,
    ensuring all parameters are physically valid before initializing the mathematical
    samplers.

    Parameters
    ----------
    max_energy_shift_mev : float
        The maximum allowed energy shift for a peak in absolute meV.
    max_amplitude_shift_pct : float
        The maximum allowed amplitude variation as a relative percentage.
    n_samples : int, optional
        The total requested sample ensemble size.
        Default is 0.
    grid_partition_method : str, optional
        The strategy for partitioning the spectrum ('automatic' or 'manual').
        Default is 'automatic'.
    manual_energy_ranges : list of tuple of float, optional
        Explicit energy intervals [(E_start, E_end), ...] required if 
        grid_partition_method is 'manual'. Default is None.
    automatic_partition_prominence_pct : float, optional
        Minimum prominence of a PDOS valley, expressed as a percentage of
        the maximum density, for automatic domain detection. Smaller positive
        values retain more valleys. Ignored for manual partitioning.
        Default is 1.0, preserving the historical automatic behavior.
    domain_splitting_strategy : str, optional
        The rule for boundary variation across the ensemble ('fixed', 'deterministic', 
        or 'random'). Default is 'fixed'.
    correlation_strategy : str, optional
        The rule for structural interactions ('independent' or 'random_group'). 
        Default is 'independent'.
    variable_coupling : str, optional
        The rule for coupling spatial and amplitude variables within a domain 
        ('independent' or 'coupled'). Default is 'independent'.
    shared_grid_structure : bool, optional
        If True, amplitude and spatial operators use the exact same x-range structure. 
        Default is True.
    freeze_origin : bool, optional
        If True, explicitly freezes the absolute zero-energy boundary point across all 
        samples to preserve the Debye approximation. Default is True.
    sampling_method : {"latin_hypercube", "monte_carlo"}, optional
        Stochastic parameter-space design. Default is Latin hypercube, with
        a separate design per topology-defined sub-ensemble.
    random_seed : int, optional
        Reproducible Latin-hypercube seed. Default is 0 for direct Core callers.
    """
    # 1. Mandatory User Parameters
    max_energy_shift_mev: float
    max_amplitude_shift_pct: float

    # 2. Algorithmic Defaults
    n_samples: int = 0
    grid_partition_method: str = "automatic"
    manual_energy_ranges: Optional[List[Tuple[float, float]]] = None
    domain_splitting_strategy: str = "fixed"
    correlation_strategy: str = "independent"
    variable_coupling: str = "independent"
    shared_grid_structure: bool = True
    freeze_origin: bool = True
    fast_amplitude: bool = True
    automatic_partition_prominence_pct: float = 1.0
    sampling_method: str = "latin_hypercube"
    random_seed: int = 0
    

    def __post_init__(self):
        """
        Validates configuration boundaries and string literals immediately upon instantiation.

        Raises
        ------
        TaraConfigError
            If numerical limits are invalid, unsupported string options are provided,
            or required supplementary data is missing.
        """
        # One experiment-wide design mode; each topology block gets its own LHS.
        if self.sampling_method not in ("latin_hypercube", "monte_carlo"):
            raise TaraConfigError("sampling_method must be latin_hypercube or monte_carlo.")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32:
            raise TaraConfigError("random_seed must be an integer in [0, 2**32).")

        # Validate Mandatory Numerical Limits
        if type(self.n_samples) is not int:
            raise TaraConfigError("n_samples must be an integer.")
        if self.n_samples < 0:
            raise TaraConfigError("The sample ensemble size (n_samples) cannot be negative.")
        elif self.n_samples == 0:
            warnings.warn("n_samples is set to 0. This configuration is only valid for preview generation.")
        
        for name in ("max_energy_shift_mev", "max_amplitude_shift_pct"):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise TaraConfigError(f"{name} must be a finite real number.")

        # Check for negative values
        if self.max_energy_shift_mev < 0:
            raise TaraConfigError("Peak shifting boundary cannot be negative (meV).")
            
        if self.max_amplitude_shift_pct < 0:
            raise TaraConfigError("Amplitude shifting boundary cannot be negative (%).")
            
        # Reject configurations where ALL variance is locked
        if not self._has_active_sampling():
            raise TaraConfigError("At least one perturbation boundary must be strictly positive.")

        # Validate Grid Partitioning
        valid_partitions = ["automatic", "manual"]
        if self.grid_partition_method not in valid_partitions:
            raise TaraConfigError(f"grid_partition_method must be one of {valid_partitions}.")

        prominence = self.automatic_partition_prominence_pct
        if (
            not isinstance(prominence, Real)
            or isinstance(prominence, bool)
            or not math.isfinite(prominence)
            or not 0.0 < prominence <= 100.0
        ):
            raise TaraConfigError(
                "automatic_partition_prominence_pct must be a finite number "
                "strictly greater than 0 and at most 100 (percent)."
            )
            
        if self.grid_partition_method == "manual":
            if not self.manual_energy_ranges:
                raise TaraConfigError(
                    "manual_energy_ranges must be provided when grid_partition_method is 'manual'."
                )
            for start, end in self.manual_energy_ranges:
                if any(
                    not isinstance(value, Real)
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    for value in (start, end)
                ):
                    raise TaraConfigError("Manual energy boundaries must be finite real numbers.")
                if start >= end or start < 0:
                    raise TaraConfigError(
                        f"Invalid manual energy range [{start}, {end}]. "
                        "Start must be >= 0 and strictly less than end."
                    )

        # Validate Domain Splitting Topology
        valid_splitting = ["fixed", "deterministic", "random"]
        if self.domain_splitting_strategy not in valid_splitting:
            raise TaraConfigError(
                f"domain_splitting_strategy must be one of {valid_splitting}."
            )

        # Validate Correlation Routing
        valid_correlations = ["independent", "random_group"]
        if self.correlation_strategy not in valid_correlations:
            raise TaraConfigError(
                f"correlation_strategy must be one of {valid_correlations}."
            )
            
        # Validate Variable Coupling
        valid_coupling = ["independent", "coupled"]
        if self.variable_coupling not in valid_coupling:
            raise TaraConfigError(
                f"variable_coupling must be one of {valid_coupling}."
            )

    @property
    def vibration_enabled(self) -> bool:
        """Whether an energy or amplitude perturbation is active."""
        return self.max_energy_shift_mev > 0 or self.max_amplitude_shift_pct > 0

    def _has_active_sampling(self) -> bool:
        """Whether the configuration enables any perturbation."""
        return self.vibration_enabled

@dataclass(frozen=True)
class DiffusionSamplingConfig:
    """Optional bounded sensitivity sampling for Egelstaff--Schofield liquids.

    Parameters
    ----------
    enabled : bool, optional
        False preserves the existing translational contribution exactly.
    mode : str, optional
        ``diffusivity_only``, ``weight_fixed_d`` or ``joint_independent``.
    max_diffusivity_shift_pct : float, optional
        Maximum relative change of physical self-diffusivity in percent.
    max_weight_shift_pct : float, optional
        Maximum relative change in translational weight in percent.
    weight_compensation : str or None, optional
        Mandatory ``continuous_only`` policy for changing translational weight.
        Oscillator contributions remain fixed; no silent normalization occurs.

    Notes
    -----
    Bounds are sensitivity ranges, not calibrated uncertainty intervals.
    Model switching from free gas is deliberately unsupported.
    """

    enabled: bool = False
    mode: str = "diffusivity_only"
    max_diffusivity_shift_pct: float = 0.0
    max_weight_shift_pct: float = 0.0
    weight_compensation: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate sampling regime, finite bounds, and explicit policies."""
        if type(self.enabled) is not bool:
            raise TaraConfigError("diffusion.enabled must be a boolean.")
        allowed = {"diffusivity_only", "weight_fixed_d", "joint_independent"}
        if self.mode not in allowed:
            raise TaraConfigError(f"Unknown diffusion sampling mode: {self.mode!r}.")
        for name in ("max_diffusivity_shift_pct", "max_weight_shift_pct"):
            value = getattr(self, name)
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 100:
                raise TaraConfigError(f"{name} must be finite and in [0, 100).")
        if self.weight_compensation not in (None, "continuous_only"):
            raise TaraConfigError("Only explicit continuous_only compensation is supported.")
        if not self.enabled:
            if self.max_diffusivity_shift_pct or self.max_weight_shift_pct:
                raise TaraConfigError("Diffusion bounds require enabled=true.")
            return
        if self.mode == "diffusivity_only" and (
            self.max_diffusivity_shift_pct <= 0 or self.max_weight_shift_pct != 0
        ):
            raise TaraConfigError("diffusivity_only requires only a positive diffusivity bound.")
        if self.mode == "weight_fixed_d" and (
            self.max_weight_shift_pct <= 0 or self.max_diffusivity_shift_pct != 0
        ):
            raise TaraConfigError("weight_fixed_d requires only a positive weight bound.")
        if self.mode == "joint_independent" and (
            self.max_weight_shift_pct <= 0 or self.max_diffusivity_shift_pct <= 0
        ):
            raise TaraConfigError("joint_independent requires both positive bounds.")
        if self.max_weight_shift_pct > 0 and self.weight_compensation != "continuous_only":
            raise TaraConfigError("Weight variation requires explicit continuous_only compensation.")


@dataclass
class TaraConfig(VibrationSamplingConfig):
    """Configure vibrational, translational, or joint sampling.

    Parameters
    ----------
    max_energy_shift_mev, max_amplitude_shift_pct : float, optional
        Vibrational perturbation bounds. Both default to zero, which is
        valid for translation-only sampling.
    n_samples : int, optional
        Stochastic sample count for all active perturbations. Zero is only
        supported for a deterministic vibrational preview.
    diffusion : mapping or DiffusionSamplingConfig, optional
        Translation settings. Pass a mapping with ``enabled=True`` and the
        appropriate mode, bounds, and weight-compensation policy to activate
        diffusion sampling. Omitting this option keeps the vibration-only API.

    See Also
    --------
    VibrationSamplingConfig : Other vibrational settings and shared design options.
    DiffusionSamplingConfig : Modes, bounds, and weight-compensation policies.
    """

    max_energy_shift_mev: float = 0.0
    max_amplitude_shift_pct: float = 0.0
    diffusion: DiffusionSamplingConfig | Mapping[str, object] | None = field(
        default=None, kw_only=True,
    )

    def __post_init__(self) -> None:
        """Validate the translation options and the inherited sampling bounds."""
        if isinstance(self.diffusion, Mapping):
            try:
                self.diffusion = DiffusionSamplingConfig(**self.diffusion)
            except TypeError as error:
                raise TaraConfigError(f"Invalid diffusion settings: {error}") from error
        elif self.diffusion is not None and not isinstance(self.diffusion, DiffusionSamplingConfig):
            raise TaraConfigError("diffusion must be a mapping or DiffusionSamplingConfig.")
        if self.diffusion_enabled and type(self.n_samples) is int and self.n_samples == 0:
            raise TaraConfigError("Diffusion sampling requires a positive n_samples.")
        super().__post_init__()

    @property
    def diffusion_enabled(self) -> bool:
        """Whether diffusion sampling has been enabled explicitly."""
        return self.diffusion is not None and self.diffusion.enabled

    def _has_active_sampling(self) -> bool:
        """Allow the vibrational bounds to stay at zero for translation only."""
        return self.vibration_enabled or self.diffusion_enabled

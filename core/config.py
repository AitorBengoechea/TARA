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
    """Configure perturbations of a vibrational spectrum.

    Parameters
    ----------
    max_energy_shift_mev : float
        Maximum absolute peak-energy shift in meV.
    max_amplitude_shift_pct : float
        Maximum relative amplitude change in percent.
    n_samples : int, optional
        Number of stochastic samples. The default of 0 allows a deterministic
        preview.
    grid_partition_method : {"automatic", "manual"}, optional
        Method used to define vibrational energy domains. Default is
        "automatic".
    manual_energy_ranges : list of tuple of float or None, optional
        Domain boundaries in eV. Required for manual partitioning. Default
        is None.
    domain_splitting_strategy : {"fixed", "deterministic", "random"}, optional
        Strategy for varying domain boundaries across samples. Default is
        "fixed".
    correlation_strategy : {"independent", "random_group"}, optional
        Strategy for sharing perturbation multipliers across domains. Default
        is "independent".
    variable_coupling : {"independent", "coupled"}, optional
        Coupling between energy and amplitude multipliers. Default is
        "independent".
    shared_grid_structure : bool, optional
        Shared-grid setting recorded in the ensemble metadata. Default is True.
    freeze_origin : bool, optional
        Keep the zero-energy origin fixed during energy shifts. Default is True.
    fast_amplitude : bool, optional
        Legacy amplitude setting. The sampler selects the amplitude operator
        through its `amplitude_method` argument. Default is True.
    automatic_partition_prominence_pct : float, optional
        Minimum valley prominence as a percentage of the maximum density
        for automatic partitioning. Default is 1.0.
    sampling_method : {"latin_hypercube", "monte_carlo"}, optional
        Design used to draw perturbation multipliers. Default is
        "latin_hypercube", with a separate design for each topology block.
    random_seed : int, optional
        Design seed in the interval [0, 2**32). Default is 0.

    Attributes
    ----------
    vibration_enabled
    """
    # Perturbation bounds and sample count.
    max_energy_shift_mev: float
    max_amplitude_shift_pct: float

    # 2. Algorithmic Defaults
    n_samples: int = 0

    # Energy domains and perturbation coupling.
    grid_partition_method: str = "automatic"
    manual_energy_ranges: Optional[List[Tuple[float, float]]] = None
    domain_splitting_strategy: str = "fixed"
    correlation_strategy: str = "independent"
    variable_coupling: str = "independent"
    shared_grid_structure: bool = True
    freeze_origin: bool = True
    fast_amplitude: bool = True
    automatic_partition_prominence_pct: float = 1.0

    # Random design.
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
    """Configure translation sampling for an Egelstaff–Schofield liquid.

    Parameters
    ----------
    enabled : bool, optional
        Enable translation sampling. Default is False.
    mode : {"diffusivity_only", "weight_fixed_d", "joint_independent"}, optional
        Vary diffusivity, vary translation weight at fixed diffusivity, or
        vary both independently, respectively. Default is "diffusivity_only".
    max_diffusivity_shift_pct : float, optional
        Maximum relative diffusivity change in percent. Must be in [0, 100).
        Default is 0.0.
    max_weight_shift_pct : float, optional
        Maximum relative translation weight change in percent. Must be in
        [0, 100). Default is 0.0.
    weight_compensation : {"continuous_only"} or None, optional
        Use "continuous_only" when varying translation weight. The compensation
        is applied to the continuous vibrational weight. Default is None.

    Notes
    -----
    The bounds specify the ranges explored in a sensitivity study.
    """

    # Translation sampling mode.
    enabled: bool = False
    mode: str = "diffusivity_only"

    # Perturbation bounds and weight compensation.
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
    max_energy_shift_mev : float, optional
        Maximum absolute peak-energy shift in meV. Default is 0.0.
    max_amplitude_shift_pct : float, optional
        Maximum relative amplitude change in percent. Default is 0.0.
    n_samples : int, optional
        Number of stochastic samples, excluding the reference. The default
        of 0 allows a deterministic vibrational preview. Diffusion sampling
        requires a positive count.
    grid_partition_method : {"automatic", "manual"}, optional
        Method used to define vibrational energy domains. Default is
        "automatic".
    manual_energy_ranges : list of tuple of float or None, optional
        Domain boundaries in eV. Required for manual partitioning. Default
        is None.
    domain_splitting_strategy : {"fixed", "deterministic", "random"}, optional
        Strategy for varying domain boundaries across samples. Default is
        "fixed".
    correlation_strategy : {"independent", "random_group"}, optional
        Strategy for sharing perturbation multipliers across domains. Default
        is "independent".
    variable_coupling : {"independent", "coupled"}, optional
        Coupling between energy and amplitude multipliers. Default is
        "independent".
    shared_grid_structure : bool, optional
        Shared-grid setting recorded in the ensemble metadata. Default is True.
    freeze_origin : bool, optional
        Keep the zero-energy origin fixed during energy shifts. Default is True.
    fast_amplitude : bool, optional
        Legacy amplitude setting. The sampler selects the amplitude operator
        through its `amplitude_method` argument. Default is True.
    automatic_partition_prominence_pct : float, optional
        Minimum valley prominence as a percentage of the maximum density
        for automatic partitioning. Default is 1.0.
    sampling_method : {"latin_hypercube", "monte_carlo"}, optional
        Design used to draw perturbation multipliers. Default is
        "latin_hypercube", with a separate design for each topology block.
    random_seed : int, optional
        Design seed in the interval [0, 2**32). Default is 0.
    diffusion : DiffusionSamplingConfig or mapping or None, optional
        Keyword-only translation settings. A mapping is converted to a
        `DiffusionSamplingConfig`. Default is None.

    Attributes
    ----------
    vibration_enabled
    diffusion_enabled

    See Also
    --------
    DiffusionSamplingConfig : Translation modes and their required bounds.

    Notes
    -----
    For translation-only sampling, enable diffusion and leave both vibrational
    bounds at zero. For joint sampling, enable diffusion and set at least one
    vibrational bound above zero.
    """

    # Vibrational perturbation bounds.
    max_energy_shift_mev: float = 0.0
    max_amplitude_shift_pct: float = 0.0

    # Translation settings.
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

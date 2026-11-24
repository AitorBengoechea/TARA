"""Backend-neutral optional Egelstaff--Schofield diffusion sampling."""

from __future__ import annotations

import numpy as np

from core.config import DiffusionSamplingConfig
from core.data.material_ensemble import MaterialEnsemble
from core.data.material_scattering import MaterialScatteringModel
from core.data.spectra_ensemble import SpectraEnsemble
from core.sampler.parameter_space import ParameterSpaceSampler


class DiffusionSampler:
    """Generate bounded diffusion realizations without changing VDOS sampling.

    Parameters
    ----------
    model : MaterialScatteringModel
        Reference Egelstaff--Schofield liquid.
    config : DiffusionSamplingConfig
        Explicit opt-in sensitivity ranges and physical coupling mode.
    seed : int
        Local generator seed, independent of the legacy global NumPy RNG.
    sampling_method : {"latin_hypercube", "monte_carlo"}, optional
        Statistical design for diffusion factors. Defaults to Latin hypercube.
    """

    def __init__(
        self, model: MaterialScatteringModel, config: DiffusionSamplingConfig,
        seed: int, sampling_method: str = "latin_hypercube",
    ) -> None:
        """Initialize sampling for an enabled liquid reference.

        Parameters
        ----------
        model : MaterialScatteringModel
            Egelstaff–Schofield liquid reference.
        config : DiffusionSamplingConfig
            Translation mode and sensitivity bounds.
        seed : int
            Seed for the local random stream.
        sampling_method : {"latin_hypercube", "monte_carlo"}, optional
            Design used for the diffusion factors.

        Raises
        ------
        ValueError
            If the material, sampling design, or weight bounds are invalid.
        """
        if model.state != "liquid" or model.dynamics.translation_model != "egelstaff_schofield":
            raise ValueError("Sampling requires liquid Egelstaff--Schofield translation.")
        if not config.enabled:
            raise ValueError("DiffusionSampler requires enabled=True.")
        self.model = model
        self.config = config
        self.seed = seed
        self.parameter_space = ParameterSpaceSampler(sampling_method, int(seed))
        self._check_weight_bounds()

    def _check_weight_bounds(self) -> None:
        """Check that translation cannot exhaust the available continuum weight.

        Raises
        ------
        ValueError
            If the largest permitted translation weight exceeds the available weight.
        """
        dynamics = self.model.dynamics
        largest_weight = dynamics.translation_weight * (1 + self.config.max_weight_shift_pct / 100)
        if largest_weight > dynamics.translation_weight + dynamics.continuous_weight:
            raise ValueError("Diffusion weight bound exceeds available continuous weight.")

    def generate_ensemble(
        self, n_samples: int, spectra_ensemble: SpectraEnsemble | None = None,
    ) -> MaterialEnsemble:
        """Draw reproducible sensitivity factors and assemble a lazy ensemble.

        Parameters
        ----------
        n_samples : int
            Positive stochastic sample count.
        spectra_ensemble : SpectraEnsemble or None
            Existing vibrational samples for joint sampling, if requested.

        Returns
        -------
        MaterialEnsemble
            Validated reference and sampled material realizations.

        Raises
        ------
        ValueError
            If the requested sample count is invalid or the vibrational
            ensemble has a different count.
        """
        if type(n_samples) is not int or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        rng = np.random.default_rng(self.seed)
        ranges = self._diffusion_ranges(n_samples, spectra_ensemble)
        d_factor, w_factor = self._draw_factors(n_samples, ranges, rng)
        dynamics = self.model.dynamics
        weights = dynamics.translation_weight * w_factor

        # Scaled diffusivity is proportional to weight * c.
        c_values = dynamics.translation_parameter * d_factor / w_factor
        return MaterialEnsemble(self.model, weights, c_values, spectra_ensemble)

    def _diffusion_ranges(self, n_samples: int, spectra_ensemble):
        """Resolve the sample ranges for translation draws.

        Parameters
        ----------
        n_samples : int
            Requested stochastic sample count.
        spectra_ensemble : SpectraEnsemble or None
            Optional vibrational ensemble with topology blocks.

        Returns
        -------
        tuple of tuple of int
            Half-open ranges in stochastic sample order.

        Raises
        ------
        ValueError
            If the vibrational and translation counts differ.
        """
        # Reuse the topology-defined sub-ensemble universes of VDOS. A
        # diffusion-only experiment has exactly one universe. Rows remain
        # aligned with the VDOS ensemble and sample_000 remains untouched.
        if spectra_ensemble is None:
            ranges = ((0, n_samples),)
        else:
            if len(spectra_ensemble) != n_samples:
                raise ValueError("Diffusion and VDOS sample counts differ.")
            blocks = getattr(spectra_ensemble, "sampling_blocks", None)
            ranges = (tuple((block.start_index, block.end_index) for block in blocks)
                      if blocks is not None else ((0, n_samples),))

        # Preserve whole-ensemble draw order for legacy Monte Carlo.
        if self.parameter_space.sampling_method == "monte_carlo":
            ranges = ((0, n_samples),)
        return ranges

    def _draw_factors(self, n_samples: int, ranges, rng: np.random.Generator):
        """Draw diffusivity and weight factors in topology order.

        Parameters
        ----------
        n_samples : int
            Total stochastic sample count.
        ranges : sequence of tuple of int
            Half-open sample ranges for independent designs.
        rng : numpy.random.Generator
            Local Monte Carlo random stream.

        Returns
        -------
        tuple of numpy.ndarray
            Relative physical diffusivity and translation-weight factors.
        """
        diffusivity_factors = np.ones(n_samples, dtype=float)
        weight_factors = np.ones(n_samples, dtype=float)
        diffusivity_bound = self.config.max_diffusivity_shift_pct
        weight_bound = self.config.max_weight_shift_pct

        for subensemble_id, (start, end) in enumerate(ranges):
            rows = slice(start, end)
            size = end - start

            if diffusivity_bound:
                diffusivity_factors[rows] = self._draw_relative_factors(
                    size, diffusivity_bound, subensemble_id, "diffusivity", rng,
                )

            if weight_bound:
                weight_factors[rows] = self._draw_relative_factors(
                    size, weight_bound, subensemble_id, "translation_weight", rng,
                )

        return diffusivity_factors, weight_factors
    
    def _draw_relative_factors(
        self,
        size: int,
        bound_pct: float,
        subensemble_id: int,
        group: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw multiplicative perturbation factors for one topology block.

        Parameters
        ----------
        size : int
            Number of stochastic samples in the block.
        bound_pct : float
            Maximum relative change in percent.
        subensemble_id : int
            Index identifying the topology block.
        group : str
            Name of the parameter stream.
        rng : numpy.random.Generator
            Local random generator used for Monte Carlo draws.

        Returns
        -------
        numpy.ndarray
            One factor per sample, centered on one.
        """
        coordinates = self.parameter_space.generate(
            size,
            1,
            subensemble_id=subensemble_id,
            group=group,
            rng=rng,
        )
        multipliers = coordinates[:, 0]
        return 1.0 + (bound_pct / 100.0) * multipliers

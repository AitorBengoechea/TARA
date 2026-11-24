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
    ):
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
        """Reject configuration ranges that could make continuous weight negative."""
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
        """
        if type(n_samples) is not int or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        rng = np.random.default_rng(self.seed)
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
        d_factor = np.ones(n_samples, dtype=float)
        w_factor = np.ones(n_samples, dtype=float)
        for subensemble_id, (start, end) in enumerate(ranges):
            size = end - start
            if self.config.max_diffusivity_shift_pct:
                d_factor[start:end] = 1 + (
                    self.config.max_diffusivity_shift_pct / 100.0
                ) * self.parameter_space.generate(
                    size, 1, subensemble_id=subensemble_id,
                    group="diffusivity", rng=rng,
                )[:, 0]
            if self.config.max_weight_shift_pct:
                w_factor[start:end] = 1 + (
                    self.config.max_weight_shift_pct / 100.0
                ) * self.parameter_space.generate(
                    size, 1, subensemble_id=subensemble_id,
                    group="translation_weight", rng=rng,
                )[:, 0]
        dynamics = self.model.dynamics
        weights = dynamics.translation_weight * w_factor
        # Scaled diffusivity is proportional to wt * c. Divide by the weight factor to
        # realize the requested diffusivity draw.
        c_values = dynamics.translation_parameter * d_factor / w_factor
        return MaterialEnsemble(self.model, weights, c_values, spectra_ensemble)

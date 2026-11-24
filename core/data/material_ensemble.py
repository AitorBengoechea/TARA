"""Lazy material realizations combining spectral and diffusion samples."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np
import pandas as pd

from core.data.material_scattering import MaterialScatteringModel
from core.data.spectra_ensemble import SpectraEnsemble


@dataclass(frozen=True)
class MaterialEnsemble:
    """Stochastic material ensemble with separate immutable reference.

    Parameters
    ----------
    reference : MaterialScatteringModel
        Unperturbed material used for ``sample_000``.
    translation_weights, diffusion_parameters_c : ndarray
        Physical realization parameters, one per stochastic sample.
    spectra_ensemble : SpectraEnsemble or None
        Optional vibrational ensemble; None reuses reference spectrum.

    Notes
    -----
    Sample identifiers are zero-based for ``__getitem__``. Translation-only
    sampling does not duplicate dense vibrational spectra.
    """

    # Reference material.
    reference: MaterialScatteringModel

    # Sampled translation and vibrational data.
    translation_weights: np.ndarray
    diffusion_parameters_c: np.ndarray
    spectra_ensemble: SpectraEnsemble | None = None

    def __post_init__(self) -> None:
        """Freeze parameter arrays and validate ensemble alignment."""
        weights = np.array(self.translation_weights, dtype=float, copy=True)
        params = np.array(self.diffusion_parameters_c, dtype=float, copy=True)
        if weights.ndim != 1 or params.shape != weights.shape or not len(weights):
            raise ValueError("Diffusion arrays must have equal nonempty one-dimensional shapes.")
        if not (np.all(np.isfinite(weights)) and np.all(weights > 0)
                and np.all(np.isfinite(params)) and np.all(params > 0)):
            raise ValueError("Diffusion arrays must contain finite positive values.")
        if self.spectra_ensemble is not None and len(self.spectra_ensemble) != len(weights):
            raise ValueError("Material and vibrational ensemble lengths differ.")
        weights.setflags(write=False)
        params.setflags(write=False)
        object.__setattr__(self, "translation_weights", weights)
        object.__setattr__(self, "diffusion_parameters_c", params)

    def __len__(self) -> int:
        """Return the stochastic sample count, excluding the reference."""
        return len(self.translation_weights)

    def __getitem__(self, index: int) -> MaterialScatteringModel:
        """Reconstruct one independent physical material realization."""
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Material sample indices must be integers.")
        if index < 0 or index >= len(self):
            raise IndexError("Material sample index out of range.")
        reference = self.reference.dynamics
        new_weight = float(self.translation_weights[index])
        # Balance each translation-weight change against the continuum so the total
        # component weight is conserved.
        dynamics = replace(
            reference,
            translation_weight=new_weight,
            translation_parameter=float(self.diffusion_parameters_c[index]),
            continuous_weight=(reference.continuous_weight + reference.translation_weight - new_weight),
        )
        material = self.reference.with_dynamics(dynamics)
        # Apply the matching sample shape after assigning its physical component
        # weights.
        if self.spectra_ensemble is not None:
            material = material.with_spectrum(self.spectra_ensemble[index])
        return material

    @property
    def sample_info(self) -> pd.DataFrame:
        """Return per-sample diffusion parameters and relative perturbations."""
        original = self.reference.dynamics
        weights = self.translation_weights
        c = self.diffusion_parameters_c
        return pd.DataFrame({
            "translation_weight": weights,
            "diffusion_parameter_c": c,
            "scaled_diffusion_d": weights * c,
            "continuous_weight": original.continuous_weight + original.translation_weight - weights,
            "diffusivity_factor": weights * c / (original.translation_weight * original.translation_parameter),
            "weight_factor": weights / original.translation_weight,
        }, index=pd.Index(range(len(self)), name="Sample_Index"))

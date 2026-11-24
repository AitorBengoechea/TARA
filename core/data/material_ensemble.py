"""Lazy material realizations combining spectral and diffusion samples."""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    translation_weights : array_like
        Dimensionless translation weight for each stochastic sample.
    diffusion_parameters_c : array_like
        Egelstaff–Schofield parameter for each stochastic sample.
    spectra_ensemble : SpectraEnsemble or None, optional
        Vibrational samples in the same order as the translation arrays.
        The default of None reuses the reference spectrum.

    Attributes
    ----------
    sample_info

    Raises
    ------
    ValueError
        If the parameter arrays are empty, differ in shape, contain
        nonpositive or nonfinite values, or do not align with the
        vibrational ensemble.

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
        self._validate_sample_arrays(weights, params)

        # Own read-only copies so input arrays can be reused by the caller.
        weights.setflags(write=False)
        params.setflags(write=False)
        object.__setattr__(self, "translation_weights", weights)
        object.__setattr__(self, "diffusion_parameters_c", params)

    def _validate_sample_arrays(self, weights: np.ndarray, params: np.ndarray) -> None:
        """Validate the physical parameters and sample alignment.

        Parameters
        ----------
        weights, params : numpy.ndarray
            Translation weights and diffusion parameters, one entry per sample.

        Raises
        ------
        ValueError
            If the arrays are empty, invalid, or misaligned with the spectra.
        """
        # Each row must pair one translation weight with one diffusion parameter.
        if weights.ndim != 1 or params.shape != weights.shape or not len(weights):
            raise ValueError("Diffusion arrays must have equal nonempty one-dimensional shapes.")

        # The diffusion model requires both sampled parameters to be finite and
        # positive.
        if not (np.all(np.isfinite(weights)) and np.all(weights > 0)
                and np.all(np.isfinite(params)) and np.all(params > 0)):
            raise ValueError("Diffusion arrays must contain finite positive values.")

        # Joint realizations pair the vibrational and translation draws by index.
        if self.spectra_ensemble is not None and len(self.spectra_ensemble) != len(weights):
            raise ValueError("Material and vibrational ensemble lengths differ.")

    def __len__(self) -> int:
        """Return the stochastic sample count.

        Returns
        -------
        int
            Number of realizations, excluding the reference.
        """
        return len(self.translation_weights)

    def __getitem__(self, index: int) -> MaterialScatteringModel:
        """Reconstruct the material at a stochastic sample index.

        Parameters
        ----------
        index : int
            Zero-based stochastic sample index.

        Returns
        -------
        MaterialScatteringModel
            Sampled material using the corresponding vibrational spectrum
            when present.

        Raises
        ------
        TypeError
            If the index is neither a Python nor a NumPy integer.
        IndexError
            If the index is negative or outside the ensemble.
        """
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Material sample indices must be integers.")
        if index < 0 or index >= len(self):
            raise IndexError("Material sample index out of range.")
        reference = self.reference.dynamics
        new_weight = float(self.translation_weights[index])
        dynamics = replace(
            reference,
            translation_weight=new_weight,
            translation_parameter=float(self.diffusion_parameters_c[index]),
            continuous_weight=self._compensated_continuous_weight(new_weight),
        )
        material = self.reference.with_dynamics(dynamics)

        # Apply the matching sample shape after assigning its physical component
        # weights.
        if self.spectra_ensemble is not None:
            material = material.with_spectrum(self.spectra_ensemble[index])
        return material

    def _compensated_continuous_weight(self, weight: float | np.ndarray) -> float | np.ndarray:
        """Calculate the continuum weight after a translation-weight change.

        Parameters
        ----------
        weight : float or numpy.ndarray
            Sampled translation weight or array of weights.

        Returns
        -------
        float or numpy.ndarray
            Continuous weights that preserve the reference sum of the
            continuous and translation contributions.
        """
        reference = self.reference.dynamics
        # Exchange weight between translation and the continuum only.
        return reference.continuous_weight + reference.translation_weight - weight

    @property
    def sample_info(self) -> pd.DataFrame:
        """Return the physical parameters and relative factors for each sample.

        Returns
        -------
        pandas.DataFrame
            Table indexed by zero-based `Sample_Index`. The scaled diffusivity
            is the product of translation weight and parameter `c`.
            Diffusivity and weight factors are ratios to the reference.
        """
        original = self.reference.dynamics
        weights = self.translation_weights
        c = self.diffusion_parameters_c
        scaled_diffusion = weights * c
        reference_diffusion = original.translation_weight * original.translation_parameter
        tables_dict = {
            "translation_weight": weights,
            "diffusion_parameter_c": c,
            "scaled_diffusion_d": scaled_diffusion,
            "continuous_weight": self._compensated_continuous_weight(weights),
            "diffusivity_factor": scaled_diffusion / reference_diffusion,
            "weight_factor": weights / original.translation_weight,
        }
        return pd.DataFrame(tables_dict, index=pd.Index(range(len(self)), name="Sample_Index"))

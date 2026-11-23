"""
Internal matrix containers for batched TARA spectrum generation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PerturbationBatch:
    """
    Describe one contiguous matrix batch of stochastic perturbations.

    Parameters
    ----------
    start_index : int
        Global ensemble index of the first sample in the batch.
    end_index : int
        Exclusive global ensemble index of the final sample in the batch.
    boundaries : numpy.ndarray
        Energy-domain boundaries with shape ``(n_domains, 2)`` in eV.
    spatial_multipliers : numpy.ndarray
        Spatial stochastic state with shape ``(n_samples, n_domains)``.
    amplitude_multipliers : numpy.ndarray or None
        Amplitude stochastic state with shape ``(n_samples, n_domains)``.
        ``None`` indicates that amplitude and spatial variables are coupled.

    Notes
    -----
    This container intentionally stores NumPy views rather than copying arrays.
    It exists only for the internal generation pipeline and should therefore be
    short lived.
    """

    start_index: int
    end_index: int
    boundaries: np.ndarray
    spatial_multipliers: np.ndarray
    amplitude_multipliers: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate and normalize the shape and storage of the batch arrays.

        Raises
        ------
        ValueError
            If the input or required state violates the operation contract.
        """
        if self.end_index <= self.start_index:
            raise ValueError("PerturbationBatch end_index must exceed start_index.")

        boundaries = np.asarray(self.boundaries)
        spatial = np.asarray(self.spatial_multipliers)

        if boundaries.ndim != 2 or boundaries.shape[1] != 2:
            raise ValueError("boundaries must have shape (n_domains, 2).")
        if spatial.ndim != 2:
            raise ValueError("spatial_multipliers must be two-dimensional.")
        if spatial.shape[0] != self.n_samples:
            raise ValueError(
                "spatial_multipliers row count must match the batch size."
            )
        if spatial.shape[1] != boundaries.shape[0]:
            raise ValueError(
                "spatial_multipliers column count must match the number of domains."
            )

        if self.amplitude_multipliers is not None:
            amplitude = np.asarray(self.amplitude_multipliers)
            if amplitude.shape != spatial.shape:
                raise ValueError(
                    "amplitude_multipliers must match spatial_multipliers shape."
                )

    @property
    def n_samples(self) -> int:
        """
        Return the number of samples represented by the batch.

        Returns
        -------
        int
            Number of samples in the matrix batch.
        """
        return self.end_index - self.start_index


@dataclass(frozen=True)
class SpectrumBatch:
    """
    Store one generated matrix batch on a shared canonical energy grid.

    Parameters
    ----------
    energy_grid : numpy.ndarray
        Canonical batch energy grid with shape ``(n_energy,)`` in eV.
    density : numpy.ndarray
        Generated density matrix with shape ``(n_samples, n_energy)``.
    sample_lengths : numpy.ndarray
        Number of physical grid points used by each sample before right-side
        zero padding.
    """

    energy_grid: np.ndarray
    density: np.ndarray
    sample_lengths: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize the shape and storage of the batch arrays.

        Raises
        ------
        ValueError
            If the input or required state violates the operation contract.
        """
        energy_grid = np.asarray(self.energy_grid)
        density = np.asarray(self.density)
        sample_lengths = np.asarray(self.sample_lengths)

        if energy_grid.ndim != 1:
            raise ValueError("energy_grid must be one-dimensional.")
        if density.ndim != 2:
            raise ValueError("density must be two-dimensional.")
        if density.shape[1] != energy_grid.size:
            raise ValueError(
                "density column count must match the energy-grid length."
            )
        if sample_lengths.ndim != 1 or sample_lengths.size != density.shape[0]:
            raise ValueError(
                "sample_lengths must contain exactly one value per density row."
            )

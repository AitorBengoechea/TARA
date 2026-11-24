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

    # Sample range and energy domains.
    start_index: int
    end_index: int
    boundaries: np.ndarray

    # Perturbation multipliers.
    spatial_multipliers: np.ndarray
    amplitude_multipliers: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate the input arrays against the batch layout.

        Raises
        ------
        ValueError
            If the array shapes do not match the batch layout.
        """
        # Use a nonempty half-open interval to identify the sample rows.
        if self.end_index <= self.start_index:
            raise ValueError("PerturbationBatch end_index must exceed start_index.")

        boundaries = np.asarray(self.boundaries)
        spatial = np.asarray(self.spatial_multipliers)

        self._validate_spatial_layout(boundaries, spatial)

        # Amplitude shifts use the same sample and domain layout when they are
        # present.
        if self.amplitude_multipliers is not None:
            amplitude = np.asarray(self.amplitude_multipliers)
            if amplitude.shape != spatial.shape:
                raise ValueError(
                    "amplitude_multipliers must match spatial_multipliers shape."
                )

    def _validate_spatial_layout(self, boundaries: np.ndarray, spatial: np.ndarray) -> None:
        """Check that spatial multipliers match the sample and domain layout.

        Parameters
        ----------
        boundaries : numpy.ndarray
            Lower and upper energy bound of each domain.
        spatial : numpy.ndarray
            Matrix with samples as rows and domains as columns.

        Raises
        ------
        ValueError
            If either array has an incompatible shape.
        """
        # Each multiplier row is a sample and each column is a domain with two bounds.
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

    # Shared grid and batch storage.
    energy_grid: np.ndarray
    density: np.ndarray
    sample_lengths: np.ndarray

    def __post_init__(self) -> None:
        """Validate the input arrays against the batch layout.

        Raises
        ------
        ValueError
            If the array shapes do not match the batch layout.
        """
        energy_grid = np.asarray(self.energy_grid)
        density = np.asarray(self.density)
        sample_lengths = np.asarray(self.sample_lengths)

        self._validate_layout(energy_grid, density, sample_lengths)

    @staticmethod
    def _validate_layout(
        energy_grid: np.ndarray, density: np.ndarray, sample_lengths: np.ndarray,
    ) -> None:
        """Check that spectra align with the shared grid and length array.

        Parameters
        ----------
        energy_grid : numpy.ndarray
            Shared energy coordinates.
        density : numpy.ndarray
            Matrix with one density row per spectrum.
        sample_lengths : numpy.ndarray
            Array with one length per density row.

        Raises
        ------
        ValueError
            If the array dimensions do not describe the same batch.
        """
        # Every density row uses the same energy coordinates along its columns.
        if energy_grid.ndim != 1:
            raise ValueError("energy_grid must be one-dimensional.")
        if density.ndim != 2:
            raise ValueError("density must be two-dimensional.")
        if density.shape[1] != energy_grid.size:
            raise ValueError(
                "density column count must match the energy-grid length."
            )
        # Each density row has one length describing its support on the shared grid.
        if sample_lengths.ndim != 1 or sample_lengths.size != density.shape[0]:
            raise ValueError(
                "sample_lengths must contain exactly one value per density row."
            )

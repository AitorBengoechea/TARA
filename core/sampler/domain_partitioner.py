"""
Mathematical partitioning engine for TARA stochastic sampling.
"""
import numpy as np
from typing import List, Tuple
from scipy.signal import find_peaks

from core.data.input_pdos import PhononSpectra
from core.config import TaraConfig


class DomainPartitioner:
    """Engine for partitioning a phonon spectrum into distinct perturbation domains.

    This class determines how the baseline phonon spectrum is partitioned into
    active perturbation regions. It handles both explicit manual definitions
    and automatic natural domain extraction based on local extrema.

    Parameters
    ----------
    spectra : PhononSpectra
        Reference spectrum to partition.
    config : TaraConfig
        Manual or automatic partitioning settings.

    Attributes
    ----------
    spectra : PhononSpectra
        The validated baseline phonon spectrum.
    config : TaraConfig
        The validated configuration gateway object.
    base_partitions : list of tuple of float
        The foundational list of boundaries [(E_start, E_end), ...] before
        any dynamic Top-K unioning is applied.
    """

    def __init__(self, spectra: PhononSpectra, config: TaraConfig):
        """
        Initializes the domain partitioner and establishes the base grid.

        Parameters
        ----------
        spectra : PhononSpectra
            The standardized phonon density of states object.
        config : TaraConfig
            The configuration gateway dictating the partitioning method.
        """
        self.spectra = spectra
        self.config = config
        self.base_partitions = self._initialize_base_partitions()

    @property
    def max_partitions(self) -> int:
        """
        Returns the maximum number of detected natural domains.

        Returns
        -------
        int
            The total number of base partitions available.
        """
        return len(self.base_partitions)

    def _initialize_base_partitions(self) -> List[Tuple[float, float]]:
        """
        Routes the initialization to either manual or automatic splitting.

        Returns
        -------
        list of tuple of float
            The foundational energy boundaries.
        """
        if self.config.grid_partition_method == "manual":
            # Manual intervals also leave gaps unperturbed.
            return self.config.manual_energy_ranges
        return self._automatic_partition()

    def _automatic_partition(self) -> List[Tuple[float, float]]:
        """Identifies natural domains based on local minima and maxima to isolate
        distinct scattering peaks.

        Returns
        -------
        list of tuple of float
            A list of contiguous energy boundaries [(E_start, E_end), ...].
        """
        y = self.spectra.density_values
        x = self.spectra.energy_grid

        # Find local minima by finding peaks on the inverted spectrum.
        # The user controls the percentage; the 1% default preserves the
        # historical detection behavior.
        inverted_y = np.max(y) - y
        prominence = (
            np.max(y) * self.config.automatic_partition_prominence_pct / 100.0
        )
        minima_indices, _ = find_peaks(inverted_y, prominence=prominence)

        boundaries = [float(x[0])]
        boundaries.extend(float(x[i]) for i in minima_indices)
        boundaries.append(float(x[-1]))

        partitions = []
        for i in range(len(boundaries) - 1):
            partitions.append((boundaries[i], boundaries[i+1]))

        return partitions

    def get_top_k_partitions(self, k: int) -> List[Tuple[float, float]]:
        """Applies Top-K selection and unioning to resolve structural confinement bias.

        Automatic domains are ranked by integrated spectral density. Their
        upper boundaries determine the merged intervals. Manual domains are
        returned as supplied.

        Parameters
        ----------
        k : int
            Requested positive domain count. A count at or above the number
            of base domains returns the original partitions.

        Returns
        -------
        list of tuple of float
            The dynamically merged energy boundaries.

        Raises
        ------
        ValueError
            If the requested k is invalid for the current base partitions.
        """
        if self.config.grid_partition_method == "manual":
            # Manual domains are strictly explicitly defined and leave the rest
            # of the spectrum unperturbed; unioning is mathematically invalid here[cite: 3].
            return self.base_partitions

        if k >= self.max_partitions:
            return self.base_partitions

        if k < 1:
            raise ValueError("K must be at least 1.")

        masses = self._partition_masses()
        # Preserve source order after ranking by integrated density.
        selected = sorted(np.argsort(masses)[-k:])
        return self._merge_selected_partitions(selected)

    def _partition_masses(self) -> list[float]:
        """Integrate the reference density over each base partition.

        Returns
        -------
        list of float
            Integrated density in source partition order.
        """
        energy = self.spectra.energy_grid
        density = self.spectra.density_values
        masses = []
        for start, end in self.base_partitions:
            mask = (energy >= start) & (energy <= end)
            masses.append(np.trapezoid(density[mask], energy[mask]))
        return masses

    def _merge_selected_partitions(self, selected) -> List[Tuple[float, float]]:
        """Merge neighboring ranges at the upper edges of selected partitions.

        Parameters
        ----------
        selected : sequence of int
            Selected partition indices in ascending source order.

        Returns
        -------
        list of tuple of float
            Contiguous merged ranges covering the reference energy support.
        """
        merged = []
        current_start = self.base_partitions[0][0]
        for index in selected:
            current_end = self.base_partitions[index][1]
            merged.append((current_start, current_end))
            current_start = current_end

        # Assign the remaining high-energy tail to the last merged partition.
        last_end = self.base_partitions[-1][1]
        if merged[-1][1] != last_end:
            merged[-1] = (merged[-1][0], last_end)
        return merged

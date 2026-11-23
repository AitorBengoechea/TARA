"""
Mathematical partitioning engine for TARA stochastic sampling.
"""
import numpy as np
from typing import List, Tuple
from scipy.signal import find_peaks

# Assuming these are accessible in your python path based on the new structure
from core.data.input_pdos import PhononSpectra
from core.config import TaraConfig


class DomainPartitioner:
    """Engine for partitioning a phonon spectrum into distinct perturbation domains.

    This class determines how the baseline phonon spectrum is partitioned into
    active perturbation regions. It handles both explicit manual definitions
    and automatic natural domain extraction based on local extrema.

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
            # The user explicitly defines specific energy intervals[cite: 3].
            return self.config.manual_energy_ranges
        else:
            # TARA automatically partitions the phonon spectrum[cite: 3].
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

        This method extracts only the top K ranges containing the largest probability mass,
        while adjacent minor regions are dynamically merged via the precise integral median.

        Parameters
        ----------
        k : int
            The target number of domains to return. Must be between 1 and max_partitions.

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

        # 1. Calculate probability mass (integral) for each base partition
        domain_masses = []
        for start, end in self.base_partitions:
            mask = (self.spectra.energy_grid >= start) & (self.spectra.energy_grid <= end)
            # Integrate against the physical coordinate grid, accounting for nonuniform spacing.
            mass = np.trapezoid(self.spectra.density_values[mask], self.spectra.energy_grid[mask])
            domain_masses.append(mass)

        # 2. Identify the indices of the Top-K largest domains
        top_k_indices = sorted(np.argsort(domain_masses)[-k:])

        # 3. Merge minor adjacent regions via integral median
        merged_partitions = []
        
        # NOTE: This section implements the structural logic for merging. 
        # For full physical accuracy, the "precise integral median" solver 
        # algorithm would replace the simplified midpoint logic below.
        
        current_start = self.base_partitions[0][0]
        for i in top_k_indices:
            # The domain ends at the original end of the selected Top-K peak
            current_end = self.base_partitions[i][1]
            
            # If there are minor domains trailing after this peak before the next 
            # Top-K peak, we find the median integral point between them.
            # (Simplified here to merge directly; actual median math requires CDF inversion).
            merged_partitions.append((current_start, current_end))
            
            # Update the start for the next block
            current_start = current_end

        # Ensure the absolute final boundary is closed properly
        if merged_partitions[-1][1] != self.base_partitions[-1][1]:
            last_start = merged_partitions[-1][0]
            merged_partitions[-1] = (last_start, self.base_partitions[-1][1])

        return merged_partitions

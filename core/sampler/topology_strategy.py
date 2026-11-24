"""
Topology strategy pattern for structural domain variation across ensembles.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Iterator

from core.config import TaraConfig
from core.sampler.domain_partitioner import DomainPartitioner


class TopologyStrategy(ABC):
    """Abstract base class defining the domain splitting strategy for TARA.

    This strategy dictates how the x-range structure behaves across the
    total requested sample ensemble (N_samples).

    Parameters
    ----------
    config : TaraConfig
        Sample count, design method, and topology settings.
    partitioner : DomainPartitioner
        Reference domain partition and Top-K selections.

    Attributes
    ----------
    config : TaraConfig
        The validated configuration gateway.
    partitioner : DomainPartitioner
        The engine responsible for extracting the Top-K boundaries.
    """

    def __init__(self, config: TaraConfig, partitioner: DomainPartitioner):
        """Store the partitioner and configuration used to plan sample topologies.

        Parameters
        ----------
        config : TaraConfig
            Validated TARA configuration object.
        partitioner : DomainPartitioner
            Component providing the spectral-domain partitions.
        """
        self.config = config
        self.partitioner = partitioner

    @abstractmethod
    def generate_topology_plan(self) -> Iterator[Tuple[int, int, List[Tuple[float, float]]]]:
        """
        Generates an execution plan for the sampler.

        Yields
        ------
        tuple of int, int, and list of tuple of float
            Half-open sample range and its energy domains in eV.
        """
        pass


class FixedTopology(TopologyStrategy):
    """Implements the Fixed Configuration strategy.

    No boundary variation is permitted. The natural boundaries of the
    initially detected partitions remain strictly locked across the entire
    stochastic ensemble. Enclosed peaks can only shift within their
    localized, rigidly bounded ranges.
    """

    def generate_topology_plan(self) -> Iterator[Tuple[int, int, List[Tuple[float, float]]]]:
        """Yield a single block with the reference partition.

        Yields
        ------
        tuple of int, int, and list of tuple of float
            The complete sample range and its fixed domain bounds.
        """
        yield (
            0,
            self.config.n_samples,
            self.partitioner.base_partitions
        )


class DeterministicTopology(TopologyStrategy):
    """Implements the Deterministic Split strategy.

    The grid structure is systematically altered by uniformly iterating through
    the available Top-K domain unioning options. The total stochastic run
    is divided into deterministic sub-ensembles.
    """

    def generate_topology_plan(self) -> Iterator[Tuple[int, int, List[Tuple[float, float]]]]:
        """Allocate consecutive blocks to increasing Top-K partitions.

        Yields
        ------
        tuple of int, int, and list of tuple of float
            Half-open sample range and its deterministically selected domain bounds.
        """
        n_samples = self.config.n_samples
        n_partition = self.partitioner.max_partitions

        # With at most two domains, retain the reference partition.
        if n_partition <= 2:
            yield (0, n_samples, self.partitioner.base_partitions)
            return

        # Sweep from two domains up to one fewer than the reference partition.
        n_options = n_partition - 2
        sub_ensemble_size = max(1, n_samples // n_options)

        current_idx = 0

        # Iterate through the valid K options
        for k in range(2, n_partition):
            if current_idx >= n_samples:
                break

            end_idx = min(current_idx + sub_ensemble_size, n_samples)
            # If this is the last option, ensure it absorbs any remaining samples due to rounding
            if k == n_partition - 1:
                end_idx = n_samples

            partitions = self.partitioner.get_top_k_partitions(k)
            yield (current_idx, end_idx, partitions)

            current_idx = end_idx


class RandomTopology(TopologyStrategy):
    """Implements the Random Split strategy.

    The topological structure of the grid is varied stochastically rather than sequentially.
    Independent random variables dynamically govern the domain unioning.
    """

    def generate_topology_plan(self) -> Iterator[Tuple[int, int, List[Tuple[float, float]]]]:
        """Draw the partition size for each consecutive sample block.

        Yields
        ------
        tuple of int, int, and list of tuple of float
            Half-open sample range and its randomly selected domain bounds.
        """
        n_samples = self.config.n_samples
        n_partition = self.partitioner.max_partitions

        if n_partition <= 2:
            yield (0, n_samples, self.partitioner.base_partitions)
            return

        # Use the historical default of four when the configuration has no override.
        n_random = getattr(self.config, 'n_random', 4)
        sub_ensemble_size = max(1, n_samples // n_random)

        topology_rng = self._topology_rng()
        current_idx = 0

        while current_idx < n_samples:
            end_idx = min(current_idx + sub_ensemble_size, n_samples)

            # Draw once for each block so all its samples share a partition.
            random_k = (np.random.randint(2, n_partition + 1)
                        if topology_rng is None else
                        int(topology_rng.integers(2, n_partition + 1)))
            partitions = self.partitioner.get_top_k_partitions(random_k)

            yield (current_idx, end_idx, partitions)

            current_idx = end_idx

    def _topology_rng(self) -> np.random.Generator | None:
        """Resolve the random stream used for discrete topology choices.

        Returns
        -------
        numpy.random.Generator or None
            Independent stream for Latin hypercube designs. None selects the
            legacy global Monte Carlo stream.
        """
        # Distinguish topology choices from the named continuous streams.
        return (
            np.random.default_rng(np.random.SeedSequence(
                [self.config.random_seed, 0x70A0]
            ))
            if self.config.sampling_method == "latin_hypercube" else None
        )


def get_topology_strategy(config: TaraConfig, partitioner: DomainPartitioner) -> TopologyStrategy:
    """
    Factory function to initialize the appropriate topology strategy.

    Parameters
    ----------
    config : TaraConfig
        The validated configuration gateway.
    partitioner : DomainPartitioner
        The initialized domain partitioner.

    Returns
    -------
    TopologyStrategy
        The instantiated strategy for generating the sampler execution plan.

    Raises
    ------
    ValueError
        If the configuration specifies an unrecognized domain splitting strategy.
    """
    strategy_map = {
        "fixed": FixedTopology,
        "deterministic": DeterministicTopology,
        "random": RandomTopology
    }

    strat_class = strategy_map.get(config.domain_splitting_strategy)
    if not strat_class:
        raise ValueError(
            f"Unknown domain splitting strategy: {config.domain_splitting_strategy}"
        )

    return strat_class(config, partitioner)

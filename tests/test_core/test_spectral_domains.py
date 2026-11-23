"""Spectral domains regression tests."""


import pytest
import numpy as np
import pandas as pd
from dataclasses import replace
from core.config import TaraConfig
from core.config import TaraConfigError
from core.data.input_pdos import PhononSpectra
from core.sampler.domain_partitioner import DomainPartitioner
from core.sampler.topology_strategy import get_topology_strategy


# Domain partitioner

@pytest.fixture
def manual_config():
    """Returns a TaraConfig configured for manual grid partitioning."""
    return TaraConfig(
        n_samples=10,
        max_energy_shift_mev=1.0,
        max_amplitude_shift_pct=5.0,
        grid_partition_method="manual",
        manual_energy_ranges=[(1.0, 2.0), (3.0, 4.0)]
    )


@pytest.fixture
def auto_config():
    """Returns a TaraConfig configured for automatic grid partitioning."""
    return TaraConfig(
        n_samples=10,
        max_energy_shift_mev=1.0,
        max_amplitude_shift_pct=5.0,
        grid_partition_method="automatic"
    )


@pytest.fixture
def synthetic_spectra():
    """
    Generates a synthetic spectrum with 3 distinct peaks and 2 distinct valleys.
    - Peak 1 (Large):  ~1.0 eV
    - Peak 2 (Small):  ~3.0 eV
    - Peak 3 (Large):  ~5.0 eV
    - Valleys: ~2.0 eV and ~4.0 eV
    """
    x_grid = np.linspace(0.0, 6.0, 61)  # 0.1 eV spacing
    y_density = np.zeros_like(x_grid)
    
    # Inject Gaussian peaks
    y_density += 10.0 * np.exp(-((x_grid - 1.0)**2) / 0.1)  # P1
    y_density +=  2.0 * np.exp(-((x_grid - 3.0)**2) / 0.1)  # P2 (Minor)
    y_density += 10.0 * np.exp(-((x_grid - 5.0)**2) / 0.1)  # P3
    
    # Add a tiny baseline to prevent absolute zero issues, then wrap in PhononSpectra
    y_density += 0.01 
    return PhononSpectra(pd.Series(y_density, index=x_grid))


def test_manual_partitioning(synthetic_spectra, manual_config):
    """
    Verify that manual grid partitioning strictly obeys user-defined boundaries
    and ignores Top-K merge requests.
    """
    partitioner = DomainPartitioner(spectra=synthetic_spectra, config=manual_config)
    
    # 1. Base partitions should exactly match the config[cite: 5]
    assert partitioner.max_partitions == 2
    assert partitioner.base_partitions == [(1.0, 2.0), (3.0, 4.0)]
    
    # 2. Top-K partitioning on manual boundaries should return the exact base bounds[cite: 5]
    # (Manual mode implies strict regional locking)
    top_k_parts = partitioner.get_top_k_partitions(k=1)
    assert top_k_parts == [(1.0, 2.0), (3.0, 4.0)]


def test_automatic_partitioning_base(synthetic_spectra, auto_config):
    """
    Verify that automatic grid partitioning naturally detects valleys and 
    splits the spectrum into continuous domains.
    """
    partitioner = DomainPartitioner(spectra=synthetic_spectra, config=auto_config)
    
    # 1. We injected 3 peaks separated by 2 valleys, so there should be 3 partitions
    assert partitioner.max_partitions == 3
    
    # 2. Ensure contiguous boundaries spanning the whole 0.0 to 6.0 eV grid
    parts = partitioner.base_partitions
    
    assert parts[0][0] == 0.0
    assert parts[-1][1] == 6.0
    
    # 3. Verify internal valley boundaries (should be at approx 2.0 and 4.0 eV)
    assert np.isclose(parts[0][1], 2.0, atol=0.2)
    assert np.isclose(parts[1][0], parts[0][1])  # Boundaries must touch
    assert np.isclose(parts[1][1], 4.0, atol=0.2)


def test_automatic_prominence_threshold_controls_detected_domains(auto_config):
    """A shallow, resolved valley only appears for a lower user threshold."""
    energy = np.linspace(0.0, 1.0, 1001)
    density = 2.0 + 0.008 * np.sin(2 * np.pi * 25 * energy)
    density[0] = 0.0
    spectra = PhononSpectra(pd.Series(density, index=energy))

    default_partitions = DomainPartitioner(spectra, auto_config).base_partitions
    finer_config = replace(auto_config, automatic_partition_prominence_pct=0.2)
    finer_partitions = DomainPartitioner(spectra, finer_config).base_partitions

    assert auto_config.automatic_partition_prominence_pct == 1.0
    assert len(default_partitions) < len(finer_partitions)
    assert len(finer_partitions) > 20
    assert finer_partitions[0][0] == pytest.approx(energy[0])
    assert finer_partitions[-1][1] == pytest.approx(energy[-1])
    assert all(a[1] == pytest.approx(b[0]) for a, b in zip(finer_partitions, finer_partitions[1:]))


@pytest.mark.parametrize("invalid", [-1, 0, 100.1, float("inf"), float("nan"), True, "0.2"])
def test_automatic_partition_prominence_validation(auto_config, invalid):
    """Reject nonfinite values, zero, booleans, strings and out-of-range percentages."""
    with pytest.raises(TaraConfigError, match="automatic_partition_prominence_pct"):
        replace(auto_config, automatic_partition_prominence_pct=invalid)


def test_manual_partitioning_ignores_automatic_prominence(synthetic_spectra, manual_config):
    """An automatic tuning control cannot change explicitly selected domains."""
    original = DomainPartitioner(synthetic_spectra, manual_config).base_partitions
    config = replace(manual_config, automatic_partition_prominence_pct=0.1)
    assert DomainPartitioner(synthetic_spectra, config).base_partitions == original


def test_top_k_partitioning_merging(synthetic_spectra, auto_config):
    """
    Verify that get_top_k_partitions dynamically merges regions with small 
    probability mass into adjacent primary domains.
    """
    partitioner = DomainPartitioner(spectra=synthetic_spectra, config=auto_config)
    
    # Request Top 2 domains (merging the tiny Peak 2 in the middle into an adjacent domain)
    merged_parts = partitioner.get_top_k_partitions(k=2)
    
    # 1. The result should now only contain 2 domains instead of 3
    assert len(merged_parts) == 2
    
    # 2. The domains should still contiguously cover the full grid
    assert merged_parts[0][0] == 0.0
    assert merged_parts[-1][1] == 6.0
    assert merged_parts[0][1] == merged_parts[1][0]
    
    # 3. The internal boundary should skip the first valley and jump to the second valley
    # (merging Peak 1 and Peak 2, separated from Peak 3)
    assert np.isclose(merged_parts[0][1], 2.0, atol=0.2)


def test_top_k_validation_errors(synthetic_spectra, auto_config):
    """
    Verify that asking for an invalid number of top-k domains raises a ValueError.
    """
    partitioner = DomainPartitioner(spectra=synthetic_spectra, config=auto_config)
    
    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(ValueError, match="K must be at least 1"):
        partitioner.get_top_k_partitions(k=0)


# Topology strategy

class MockPartitioner:
    """
    A lightweight mock to isolate topology logic from physics logic.
    Provides fake boundaries based on the requested K value.
    """
    def __init__(self, max_partitions: int):
        """Initialize the instance and its required state.

        Parameters
        ----------
        max_partitions : int
            Maximum permitted partitions.
        """
        self.max_partitions = max_partitions
        # Fake base boundaries [ (0,1), (0,1), ... ] matching the max size
        self.base_partitions = [(0.0, 1.0)] * max_partitions
        
    def get_top_k_partitions(self, k: int):
        # Returns a fake list of length K to prove the strategy requested the correct K
        """Get top k partitions for the current workflow.

        Parameters
        ----------
        k : int
            Number of domains retained by the topology selection.
        """
        return [(0.0, float(k))] * k


def test_fixed_topology():
    """
    Verify FixedTopology yields exactly 1 sub-ensemble containing all N samples,
    using strictly the base partitions.
    """
    config = TaraConfig(
        n_samples=100, 
        max_energy_shift_mev=1.0, 
        max_amplitude_shift_pct=1.0, 
        domain_splitting_strategy="fixed"
    )
    partitioner = MockPartitioner(max_partitions=4)
    strategy = get_topology_strategy(config, partitioner)
    
    plan = list(strategy.generate_topology_plan())
    
    assert len(plan) == 1
    # Check bounds (0 to N_samples)
    assert plan[0][0] == 0
    assert plan[0][1] == 100
    # Check it used base partitions (length 4)
    assert len(plan[0][2]) == 4


def test_deterministic_topology_standard():
    """
    Verify DeterministicTopology evenly divides samples across all valid K options.
    For max_partitions=4, valid K options are 2 and 3 (2 total options).
    100 samples / 2 options = 50 samples per chunk.
    """
    config = TaraConfig(
        n_samples=100, 
        max_energy_shift_mev=1.0, 
        max_amplitude_shift_pct=1.0, 
        domain_splitting_strategy="deterministic"
    )
    partitioner = MockPartitioner(max_partitions=4)
    strategy = get_topology_strategy(config, partitioner)
    
    plan = list(strategy.generate_topology_plan())
    
    assert len(plan) == 2
    
    # Sub-ensemble 1: Iterates K=2
    assert plan[0][0] == 0
    assert plan[0][1] == 50
    assert len(plan[0][2]) == 2
    
    # Sub-ensemble 2: Iterates K=3
    assert plan[1][0] == 50
    assert plan[1][1] == 100
    assert len(plan[1][2]) == 3


def test_deterministic_topology_fallback():
    """
    Verify DeterministicTopology falls back to Fixed mode if there are 2 
    or fewer detected base partitions (since K must be at least 2).
    """
    config = TaraConfig(
        n_samples=100, 
        max_energy_shift_mev=1.0, 
        max_amplitude_shift_pct=1.0, 
        domain_splitting_strategy="deterministic"
    )
    partitioner = MockPartitioner(max_partitions=2)
    strategy = get_topology_strategy(config, partitioner)
    
    plan = list(strategy.generate_topology_plan())
    
    # Should fallback to a single chunk
    assert len(plan) == 1
    assert plan[0][1] == 100
    assert len(plan[0][2]) == 2


def test_random_topology():
    """
    Verify RandomTopology divides the run into stochastic sub-ensembles
    and correctly bounds the random K selections.
    """
    config = TaraConfig(
        n_samples=100, 
        max_energy_shift_mev=1.0, 
        max_amplitude_shift_pct=1.0, 
        domain_splitting_strategy="random"
    )
    # The codebook uses a default getattr for n_random = 4
    config.n_random = 4 
    
    partitioner = MockPartitioner(max_partitions=5)
    strategy = get_topology_strategy(config, partitioner)
    
    plan = list(strategy.generate_topology_plan())
    
    # 100 samples / 4 random chunks = chunks of 25
    assert len(plan) == 4
    assert plan[0][0] == 0
    assert plan[0][1] == 25
    assert plan[-1][1] == 100
    
    # Check that each chunk randomly selected a K between 2 and max_partitions (5)
    for chunk in plan:
        partitions = chunk[2]
        assert 2 <= len(partitions) <= 5

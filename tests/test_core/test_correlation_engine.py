import pytest
import numpy as np
from core.config import TaraConfig, TaraConfigError
from core.sampler.correlation_engine import (
    get_correlation_strategy,
    IndependentCorrelation,
    RandomGroupCorrelation
)

def test_independent_correlation():
    """
    Verify the Independent strategy generates a truly uncoupled matrix
    where every domain in every sample gets a unique random multiplier.
    """
    config = TaraConfig(
        n_samples=10, 
        max_energy_shift_mev=1.0, 
        max_amplitude_shift_pct=1.0, 
        correlation_strategy="independent"
    )
    engine = get_correlation_strategy(config)
    
    # Generate 10 samples, each with 4 distinct energy domains
    multipliers = engine.generate_domain_multipliers(n_samples=10, n_domains=4)
    
    # 1. Structural constraints
    assert isinstance(engine, IndependentCorrelation)
    assert multipliers.shape == (10, 4)
    
    # 2. Bounding constraints (always [-1.0, 1.0])
    assert np.all(multipliers >= -1.0)
    assert np.all(multipliers <= 1.0)
    
    # 3. Independence constraints:
    # If domains are truly independent, the variance across the domains (columns) 
    # for a single sample (row) should be strictly greater than zero.
    row_variances = np.var(multipliers, axis=1)
    assert np.all(row_variances > 0.0)


def test_random_group_correlation():
    """
    Verify the Random Group strategy generates a correlated matrix
    where a single sample's multiplier is perfectly broadcast across all domains.
    """
    config = TaraConfig(
        n_samples=10, 
        max_energy_shift_mev=1.0, 
        max_amplitude_shift_pct=1.0, 
        correlation_strategy="random_group"
    )
    engine = get_correlation_strategy(config)
    
    # Generate 10 samples, each with 4 distinct energy domains
    multipliers = engine.generate_domain_multipliers(n_samples=10, n_domains=4)
    
    # 1. Structural constraints
    assert isinstance(engine, RandomGroupCorrelation)
    assert multipliers.shape == (10, 4)
    
    # 2. Bounding constraints (always [-1.0, 1.0])
    assert np.all(multipliers >= -1.0)
    assert np.all(multipliers <= 1.0)
    
    # 3. Correlation constraints:
    # Because the same value is broadcast across all domains for a given sample,
    # the variance across a row must be exactly 0.0.
    row_variances = np.var(multipliers, axis=1)
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(row_variances, 0.0, atol=1e-12)
    
    # Explicitly check that domain 0's multipliers identically match domain 3's
    np.testing.assert_array_equal(multipliers[:, 0], multipliers[:, 3])


def test_invalid_correlation_strategy():
    """
    Verify the configuration gateway correctly blocks unrecognized correlation strategies.
    """
    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(TaraConfigError, match="correlation_strategy must be one of"):
        TaraConfig(
            n_samples=10, 
            max_energy_shift_mev=1.0, 
            max_amplitude_shift_pct=1.0, 
            correlation_strategy="non_existent_strategy"
        )

def test_tara_config_allows_isolated_modifiers():
    """
    Test that TaraConfig permits isolating mathematical modifiers by setting one boundary to zero.

    Verifies that:
    - TaraConfig initializes successfully when max_energy_shift_mev is 0.0 (Amplitude-Only mode).
    - TaraConfig initializes successfully when max_amplitude_shift_pct is 0.0 (Shift-Only mode).
    """
    # Amplitude-Only mode (0.0 energy shift)
    config_amp = TaraConfig(
        n_samples=10,
        max_energy_shift_mev=0.0,
        max_amplitude_shift_pct=5.0
    )
    assert config_amp.max_energy_shift_mev == 0.0
    
    # Shift-Only mode (0.0 amplitude shift)
    config_shift = TaraConfig(
        n_samples=10,
        max_energy_shift_mev=2.0,
        max_amplitude_shift_pct=0.0
    )
    assert config_shift.max_amplitude_shift_pct == 0.0


def test_tara_config_rejects_zero_variance():
    """
    Test that TaraConfig rejects configurations where all structural variance is locked.

    Verifies that:
    - Setting both perturbation boundaries identically to 0.0 raises a TaraConfigError.
    """
    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(TaraConfigError, match="At least one perturbation boundary must be strictly positive"):
        TaraConfig(
            n_samples=10,
            max_energy_shift_mev=0.0,
            max_amplitude_shift_pct=0.0
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), True, "10"])
@pytest.mark.parametrize("parameter", ["max_energy_shift_mev", "max_amplitude_shift_pct"])
def test_tara_config_rejects_invalid_perturbation_bounds(parameter, invalid):
    """Reject nonfinite and non-real perturbation limits at construction."""
    options = {"max_energy_shift_mev": 1.0, "max_amplitude_shift_pct": 5.0}
    options[parameter] = invalid

    with pytest.raises(TaraConfigError, match=parameter):
        TaraConfig(n_samples=10, **options)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), True, "1"])
@pytest.mark.parametrize("endpoint", [0, 1])
def test_tara_config_rejects_invalid_manual_endpoints(endpoint, invalid):
    """Validate both endpoints before comparing the interval bounds."""
    bounds = [0.0, 0.1]
    bounds[endpoint] = invalid

    with pytest.raises(TaraConfigError, match="Manual energy boundaries must be finite real"):
        TaraConfig(
            max_energy_shift_mev=1.0,
            max_amplitude_shift_pct=5.0,
            n_samples=10,
            grid_partition_method="manual",
            manual_energy_ranges=[tuple(bounds)],
        )


@pytest.mark.parametrize("count", [2.5, 2.0, True, False])
def test_tara_config_rejects_noninteger_sample_counts(count):
    """Require a Python integer for the sample count."""
    with pytest.raises(TaraConfigError, match="n_samples must be an integer"):
        TaraConfig(max_energy_shift_mev=1.0, max_amplitude_shift_pct=5.0, n_samples=count)


def test_tara_config_warns_for_zero_sample_preview():
    """Keep zero samples available for preview generation."""
    with pytest.warns(UserWarning, match="only valid for preview generation"):
        config = TaraConfig(max_energy_shift_mev=1.0, max_amplitude_shift_pct=0.0, n_samples=0)
    assert config.n_samples == 0

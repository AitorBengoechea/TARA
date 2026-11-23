"""Vdos sampling regression tests."""


from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from core.config import TaraConfig
from core.data.input_pdos import PhononSpectra
from core.data.spectra_ensemble import SpectraEnsemble
from core.sampler.concrete_samplers import TaraSampler
from core.config import TaraConfigError


# Spectrum perturbations and ensemble generation

@pytest.fixture
def synthetic_spectra():
    """Create a normalized two-peak PDOS with a physical zero-energy origin."""
    x_grid = np.linspace(0.0, 0.12, 121)
    y_density = (
        0.8 * np.exp(-((x_grid - 0.035) ** 2) / 0.00012)
        + 1.2 * np.exp(-((x_grid - 0.080) ** 2) / 0.00018)
        + 1e-6
    )
    y_density[0] = 0.0
    return PhononSpectra(pd.Series(y_density, index=x_grid))


@pytest.fixture
def default_config():
    """Create a fixed two-domain configuration for fast integration tests."""
    return TaraConfig(
        n_samples=6,
        max_energy_shift_mev=1.0,
        max_amplitude_shift_pct=5.0,
        grid_partition_method="manual",
        manual_energy_ranges=[(0.0, 0.06), (0.06, 0.12)],
        domain_splitting_strategy="fixed",
        correlation_strategy="independent",
        variable_coupling="independent",
        freeze_origin=True,
    )


def _sampler(synthetic_spectra, config, spatial_method="diffeomorphic"):
    """Build a sampler without repeating its initialization in every test."""
    return TaraSampler(
        config=config,
        spectra=synthetic_spectra,
        amplitude_method="fast",
        spatial_method=spatial_method,
    )


def test_tara_sampler_returns_spectra_ensemble_with_provenance(
    synthetic_spectra, default_config
):
    """Stochastic generation retains normalized samples and provenance."""
    sampler = _sampler(synthetic_spectra, default_config)
    ensemble = sampler.generate_ensemble(batch_size=3)

    assert isinstance(ensemble, SpectraEnsemble)
    assert ensemble.n_samples == default_config.n_samples
    assert ensemble.generation.n_samples == default_config.n_samples
    assert ensemble.generation.automatic_partition_prominence_pct == 1.0
    assert ensemble.pdos.shape[0] == default_config.n_samples
    assert ensemble.reference is synthetic_spectra
    assert len(ensemble.sampling_blocks) == 1
    assert isinstance(ensemble[0], PhononSpectra)
    assert ensemble.sample_info.index.equals(ensemble.pdos.index)

    for sample in ensemble:
        assert np.isclose(sample.integral, 1.0, rtol=1e-4)
        assert sample.energy_grid[0] == 0.0


def test_generate_ensemble_batch_size_does_not_change_random_sequence(
    synthetic_spectra, default_config
):
    """Different matrix chunk sizes produce identical stochastic ensembles."""
    np.random.seed(12345)
    sampler_small = _sampler(synthetic_spectra, default_config)
    ensemble_small = sampler_small.generate_ensemble(batch_size=2)

    np.random.seed(12345)
    sampler_large = _sampler(synthetic_spectra, default_config)
    ensemble_large = sampler_large.generate_ensemble(
        batch_size=default_config.n_samples
    )

    np.testing.assert_allclose(
        ensemble_small.pdos.to_numpy(), ensemble_large.pdos.to_numpy()
    )
    pd.testing.assert_frame_equal(
        ensemble_small.sample_info, ensemble_large.sample_info
    )


def test_preview_records_effective_auto_prominence(synthetic_spectra, default_config):
    """Preview provenance records the setting used by automatic partitioning."""
    config = replace(
        default_config,
        grid_partition_method="automatic",
        manual_energy_ranges=None,
        automatic_partition_prominence_pct=0.2,
    )
    sampler = _sampler(synthetic_spectra, config)
    preview = sampler.generate_preview()

    assert preview.generation.automatic_partition_prominence_pct == pytest.approx(0.2)
    assert len(preview.sampling_blocks[0].boundaries) == sampler.partitioner.max_partitions


@pytest.mark.parametrize(
    "amplitude_pct, shift_mev, expected_count",
    [(5.0, 1.0, 4), (5.0, 0.0, 2), (0.0, 1.0, 2)],
)
def test_preview_returns_ensemble_with_actual_extrema_count(
    synthetic_spectra, default_config, amplitude_pct, shift_mev, expected_count
):
    """The preview's sample count describes extrema, not configured samples."""
    config = replace(
        default_config,
        max_amplitude_shift_pct=amplitude_pct,
        max_energy_shift_mev=shift_mev,
    )
    preview = _sampler(synthetic_spectra, config).generate_preview()

    assert isinstance(preview, SpectraEnsemble)
    assert preview.n_samples == expected_count
    assert len(preview) == expected_count
    assert preview.generation.n_samples == expected_count
    assert preview.generation.automatic_partition_prominence_pct == 1.0
    assert preview.reference is synthetic_spectra
    assert preview.pdos.shape[0] == expected_count
    assert preview.sample_info.index.equals(preview.pdos.index)
    assert len(preview.sampling_blocks) == 1
    assert preview.sampling_blocks[0].start_index == 0
    assert preview.sampling_blocks[0].end_index == expected_count
    assert isinstance(preview[0], PhononSpectra)
    assert all(sample.integral > 0.0 for sample in preview)


@pytest.mark.parametrize("spatial_method", ["diffeomorphic", "crossfade"])
def test_preview_matches_scalar_extrema_without_normalizing(
    synthetic_spectra, default_config, spatial_method
):
    """Packing does not change scalar grids, density values, or integrals."""
    sampler = _sampler(synthetic_spectra, default_config, spatial_method)
    preview = sampler.generate_preview()
    block = preview.sampling_blocks[0]

    assert preview.n_samples == 4
    assert preview.energy_grid.size >= synthetic_spectra.energy_grid.size

    for i in range(preview.n_samples):
        reference = sampler._perturb_single_sample(
            boundaries=sampler.partitioner.base_partitions,
            spatial_multipliers=block.spatial_multipliers[i],
            amplitude_multipliers=block.amplitude_multipliers[i],
            auto_normalize=False,
        )
        reconstructed = preview[i]
        np.testing.assert_allclose(
            reconstructed.energy_grid, reference.energy_grid, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            reconstructed.density_values, reference.density_values,
            rtol=1e-10, atol=1e-12,
        )
        assert reconstructed.integral == pytest.approx(
            reference.integral, rel=1e-10, abs=1e-12
        )
        np.testing.assert_allclose(
            preview.pdos.iloc[i, :len(reference.energy_grid)].to_numpy(),
            reference.density_values, rtol=1e-10, atol=1e-12,
        )

    # At least one preview deliberately has a non-unit area. Lazy sample
    # reconstruction must not silently renormalize it to unity.
    assert any(not np.isclose(sample.integral, 1.0, atol=1e-4) for sample in preview)


@pytest.mark.parametrize(
    "amplitude_pct, shift_mev, expected_signs",
    [
        (5.0, 1.0, [(1, 1), (1, -1), (-1, 1), (-1, -1)]),
        (5.0, 0.0, [(1, 1), (1, -1)]),
        (0.0, 1.0, [(1, 1), (-1, 1)]),
    ],
)
def test_preview_multiplier_order_and_physical_provenance(
    synthetic_spectra, default_config, amplitude_pct, shift_mev, expected_signs
):
    """Metadata preserves historical endpoint order and physical units."""
    config = replace(
        default_config,
        max_amplitude_shift_pct=amplitude_pct,
        max_energy_shift_mev=shift_mev,
    )
    preview = _sampler(synthetic_spectra, config).generate_preview()
    block = preview.sampling_blocks[0]

    assert block.amplitude_multipliers is not None
    assert block.boundaries.shape == (2, 2)

    for i, (spatial_sign, amplitude_sign) in enumerate(expected_signs):
        np.testing.assert_array_equal(
            block.spatial_multipliers[i], spatial_sign
        )
        np.testing.assert_array_equal(
            block.amplitude_multipliers[i], amplitude_sign
        )
        sample_info = preview.sample_generation(i)
        np.testing.assert_allclose(
            sample_info.energy_shift_mev,
            [spatial_sign * shift_mev] * 2,
        )
        np.testing.assert_allclose(
            sample_info.amplitude_shift_pct,
            [amplitude_sign * amplitude_pct] * 2,
        )


def test_preview_extrema_remain_independent_under_coupled_config(
    synthetic_spectra, default_config
):
    """An explicit endpoint cross-product overrides stochastic coupling."""
    config = replace(default_config, variable_coupling="coupled")
    preview = _sampler(synthetic_spectra, config).generate_preview()
    block = preview.sampling_blocks[0]

    assert preview.n_samples == 4
    assert block.amplitude_multipliers is not None
    np.testing.assert_array_equal(block.spatial_multipliers[1], [1.0, 1.0])
    np.testing.assert_array_equal(block.amplitude_multipliers[1], [-1.0, -1.0])


def test_preview_with_zero_stochastic_samples(
    synthetic_spectra, default_config
):
    """A deterministic preview does not require stochastic samples."""
    with pytest.warns(UserWarning, match="n_samples is set to 0"):
        config = replace(default_config, n_samples=0)

    preview = _sampler(synthetic_spectra, config).generate_preview()
    assert preview.n_samples == 4
    assert preview.generation.n_samples == 4


def test_preview_rejects_missing_extrema(synthetic_spectra, default_config):
    """The defensive endpoint guard rejects a mutated invalid configuration."""
    sampler = _sampler(synthetic_spectra, default_config)
    sampler.config.max_amplitude_shift_pct = 0.0
    sampler.config.max_energy_shift_mev = 0.0

    with pytest.raises(ValueError, match="both maximum perturbation"):
        sampler.generate_preview()


# Latin hypercube sampling in VDOS ensembles

def _spectrum():
    x = np.linspace(0, 0.12, 65)
    y = np.sin(np.pi * x / 0.12) ** 2 + 0.1
    return PhononSpectra(pd.Series(y, index=x))


def _config(**kwargs):
    params = dict(
        n_samples=24, max_energy_shift_mev=1.0, max_amplitude_shift_pct=5.0,
        grid_partition_method='manual',
        manual_energy_ranges=[(0., .04), (.04, .08), (.08, .12)],
        domain_splitting_strategy='fixed', correlation_strategy='independent',
        variable_coupling='independent', random_seed=919,
    )
    params.update(kwargs)
    return TaraConfig(**params)


def _stratified(block, field):
    arr = np.asarray(getattr(block, field))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    for j in range(arr.shape[1]):
        np.testing.assert_array_equal(
            np.sort(np.floor((arr[:, j] + 1) * n / 2).astype(int)),
            np.arange(n),
        )


def _run(config, batch_size=8):
    return TaraSampler(config, _spectrum()).generate_ensemble(batch_size)


def test_default_is_lhs_and_both_vdos_parameter_groups_are_stratified():
    ensemble = _run(_config())
    assert ensemble.generation.sampling_method == 'latin_hypercube'
    assert len(ensemble.sampling_blocks) == 1
    block = ensemble.sampling_blocks[0]
    _stratified(block, 'spatial_multipliers')
    _stratified(block, 'amplitude_multipliers')
    assert (block.spatial_multipliers < 0).any()
    assert (block.amplitude_multipliers < 0).any()


def test_two_topology_universes_each_get_their_own_lhs(monkeypatch):
    sampler = TaraSampler(_config(n_samples=16), _spectrum())
    sampler.topology_strategy.generate_topology_plan = lambda: iter([
        (0, 6, [(0, .04), (.04, .08), (.08, .12)]),
        (6, 16, [(0, .06), (.06, .12)]),
    ])
    ensemble = sampler.generate_ensemble(batch_size=3)
    assert [(b.start_index, b.end_index) for b in ensemble.sampling_blocks] == [
        (0, 6), (6, 16),
    ]
    for block in ensemble.sampling_blocks:
        _stratified(block, 'spatial_multipliers')
        _stratified(block, 'amplitude_multipliers')


def test_batch_size_only_changes_processing_chunking():
    config = _config()
    first = _run(config, batch_size=2)
    second = _run(config, batch_size=24)
    np.testing.assert_array_equal(first.pdos.to_numpy(), second.pdos.to_numpy())
    np.testing.assert_array_equal(
        first.sampling_blocks[0].spatial_multipliers,
        second.sampling_blocks[0].spatial_multipliers,
    )


def test_coupled_group_does_not_draw_new_amplitude_dimensions():
    ensemble = _run(_config(variable_coupling='coupled'))
    block = ensemble.sampling_blocks[0]
    assert block.amplitude_multipliers is None
    _stratified(block, 'spatial_multipliers')


def test_random_group_correlation_has_single_stratified_coordinate():
    ensemble = _run(_config(correlation_strategy='random_group'))
    block = ensemble.sampling_blocks[0]
    _stratified(block, 'spatial_multipliers')
    assert np.all(block.spatial_multipliers[:, 0] == block.spatial_multipliers[:, -1])


def test_inactive_independent_spatial_parameters_do_not_consume_lhs_dimensions():
    ensemble = _run(_config(max_energy_shift_mev=0.0))
    block = ensemble.sampling_blocks[0]
    assert not np.any(block.spatial_multipliers)
    _stratified(block, 'amplitude_multipliers')


def test_explicit_mc_matches_old_global_random_draws():
    config = _config(sampling_method='monte_carlo', n_samples=9)
    np.random.seed(314)
    expected_spatial = np.random.uniform(-1., 1., size=(9, 3))
    expected_amp = np.random.uniform(-1., 1., size=(9, 3))
    np.random.seed(314)
    sample = _run(config)
    np.testing.assert_array_equal(sample.sampling_blocks[0].spatial_multipliers, expected_spatial)
    np.testing.assert_array_equal(sample.sampling_blocks[0].amplitude_multipliers, expected_amp)


def test_invalid_method_is_rejected_by_core_gateway():
    with pytest.raises(TaraConfigError, match='sampling_method'):
        _config(sampling_method='invalid')

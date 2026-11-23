"""Diffusion sampling regression tests."""


from dataclasses import FrozenInstanceError
import numpy as np
import pandas as pd
import pytest
from core.config import DiffusionSamplingConfig
from core.config import TaraConfigError
from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import DiscreteOscillator
from core.data.material_scattering import MaterialScatteringModel
from core.data.material_scattering import ScatteringSpecies
from core.data.material_scattering import SpeciesDynamics
from core.sampler.diffusion_sampler import DiffusionSampler
from core.config import TaraConfig
from core.data.spectra_ensemble import EnsembleGeneration
from core.data.spectra_ensemble import SamplingBlock
from core.data.spectra_ensemble import SpectraEnsemble
from core.sampler.concrete_samplers import TaraSampler


# Diffusion and translation weight behavior

def _model(*, continuous_weight=.45, translation_weight=.05):
    """Build an Egelstaff--Schofield water reference without a backend."""
    spectrum = PhononSpectra(pd.Series([0., 1., 2., 0.], index=[0., .01, .02, .03]))
    dynamics = SpeciesDynamics(
        continuous=spectrum, continuous_weight=continuous_weight,
        translation_weight=translation_weight, translation_model='egelstaff_schofield',
        translation_parameter=4.0,
        oscillators=(DiscreteOscillator('bend', .2, .17), DiscreteOscillator('stretch', .4, .33)),
    )
    return MaterialScatteringModel(
        'water', 'liquid', 295., ScatteringSpecies('H', 2, 1.), dynamics,
        (ScatteringSpecies('O', 1, 16.),),
    )


def _config(mode='diffusivity_only', d=10, w=0, policy=None):
    """Return one enabled bounded sensitivity sampling configuration."""
    return DiffusionSamplingConfig(
        enabled=True, mode=mode, max_diffusivity_shift_pct=d,
        max_weight_shift_pct=w, weight_compensation=policy,
    )


def test_diffusion_disabled_by_default():
    """The default configuration neither enables nor draws translation samples."""
    options = DiffusionSamplingConfig()
    assert not options.enabled
    with pytest.raises(ValueError, match='enabled'):
        DiffusionSampler(_model(), options, 1)


def test_diffusivity_only_preserves_weights_and_reference():
    """Only c and therefore diffusivity vary with fixed dynamical weights."""
    reference = _model()
    ensemble = DiffusionSampler(reference, _config(), seed=4).generate_ensemble(35)
    assert len(ensemble) == 35
    assert ensemble.reference is reference
    assert ensemble.spectra_ensemble is None
    assert all(.9 * 4 <= c <= 1.1 * 4 for c in ensemble.diffusion_parameters_c)
    assert not np.allclose(ensemble.diffusion_parameters_c, 4)
    assert all(sample == reference.dynamics.translation_weight for sample in ensemble.translation_weights)
    for idx in [0, 1, 34]:
        sample = ensemble[idx]
        assert sample.dynamics.continuous is reference.dynamics.continuous
        assert sample.dynamics.oscillators is reference.dynamics.oscillators
        assert sample.dynamics.continuous_weight == reference.dynamics.continuous_weight
        assert sample.principal is reference.principal
        assert sample.secondary is reference.secondary
    assert reference.dynamics.translation_parameter == 4.0


def test_fixed_diffusivity_weight_sampling_keeps_product_and_oscillators():
    """Changing weight preserves cw and compensates only continuous weight."""
    reference = _model()
    ensemble = DiffusionSampler(reference, _config('weight_fixed_d', 0, 20, 'continuous_only'), 3).generate_ensemble(40)
    np.testing.assert_allclose(ensemble.translation_weights * ensemble.diffusion_parameters_c, .2)
    for i in range(len(ensemble)):
        material = ensemble[i]
        assert abs(sum([material.dynamics.translation_weight, material.dynamics.continuous_weight,
                        *(o.weight for o in material.dynamics.oscillators)]) - 1) < 1e-12
        assert material.dynamics.oscillators == reference.dynamics.oscillators
    assert not np.allclose(ensemble.translation_weights, .05)


def test_joint_independent_is_reproducible_and_local():
    """Draw independent perturbations without consuming NumPy global RNG."""
    cfg = _config('joint_independent', 20, 10, 'continuous_only')
    state = np.random.get_state()
    first = DiffusionSampler(_model(), cfg, 180).generate_ensemble(100)
    other = DiffusionSampler(_model(), cfg, 180).generate_ensemble(100)
    np.testing.assert_array_equal(first.translation_weights, other.translation_weights)
    np.testing.assert_array_equal(first.diffusion_parameters_c, other.diffusion_parameters_c)
    new_state = np.random.get_state()
    assert all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
               for x, y in zip(state, new_state))
    info = first.sample_info
    assert info.index.name == 'Sample_Index'
    np.testing.assert_allclose(info['scaled_diffusion_d'],
                               first.translation_weights * first.diffusion_parameters_c)
    assert info['diffusivity_factor'].between(.8, 1.2).all()
    assert info['weight_factor'].between(.9, 1.1).all()
    with pytest.raises(ValueError):
        first.translation_weights[0] = 0
    with pytest.raises(FrozenInstanceError):
        first.reference = _model()


@pytest.mark.parametrize('kwargs', [
    {'mode': 'weight_fixed_d', 'max_weight_shift_pct': 5, 'enabled': True},
    {'mode': 'diffusivity_only', 'max_diffusivity_shift_pct': 0, 'enabled': True},
    {'mode': 'joint_independent', 'max_weight_shift_pct': 5, 'enabled': True, 'weight_compensation': 'continuous_only'},
    {'enabled': False, 'max_diffusivity_shift_pct': 5},
    {'enabled': True, 'max_diffusivity_shift_pct': 100},
    {'enabled': True, 'max_diffusivity_shift_pct': float('nan')},
])
def test_invalid_diffusion_configuration_rejected(kwargs):
    """Reject undeclared physical compensation and inconsistent bounds."""
    with pytest.raises(TaraConfigError):
        DiffusionSamplingConfig(**kwargs)


def test_impossible_weight_range_fails_before_sampling():
    """Weight bounds must not consume more than the vibrational contribution."""
    reference = _model(continuous_weight=.01, translation_weight=.49)
    with pytest.raises(ValueError, match='continuous weight'):
        DiffusionSampler(reference, _config('weight_fixed_d', 0, 20, 'continuous_only'), 1)


def test_cannot_stochastically_jump_between_free_gas_and_es():
    """The c=0 free-gas case is not a continuous Egelstaff--Schofield limit."""
    reference = _model()
    gas = reference.with_dynamics(
        SpeciesDynamics(reference.dynamics.continuous, continuous_weight=.45,
                        translation_weight=.05, translation_model='free_gas',
                        oscillators=reference.dynamics.oscillators)
    )
    with pytest.raises(ValueError, match='Egelstaff'):
        DiffusionSampler(gas, _config(), 1)


# Latin hypercube sampling in diffusion ensembles

def _reference():
    x = np.linspace(0, .12, 65)
    spectrum = PhononSpectra(pd.Series(np.sin(np.pi * x / .12) ** 2 + .1, index=x))
    dynamics = SpeciesDynamics(
        continuous=spectrum, continuous_weight=.95,
        translation_weight=.05, translation_model='egelstaff_schofield',
        translation_parameter=4.,
    )
    return MaterialScatteringModel('water', 'liquid', 295.,
                                  ScatteringSpecies('H', 2, 1.), dynamics)


def _diff_config():
    return DiffusionSamplingConfig(
        enabled=True, mode='joint_independent', max_diffusivity_shift_pct=10,
        max_weight_shift_pct=10, weight_compensation='continuous_only',
    )


def _strata(values):
    n = len(values)
    return np.sort(np.floor((values + 1) * n / 2).astype(int))


def test_diffusion_only_has_single_lhs_and_keeps_reference():
    model = _reference()
    ensemble = DiffusionSampler(model, _diff_config(), 102).generate_ensemble(17)
    assert ensemble.reference is model
    assert ensemble.spectra_ensemble is None
    factors_d = ensemble.sample_info['diffusivity_factor'].to_numpy()
    factors_w = ensemble.sample_info['weight_factor'].to_numpy()
    np.testing.assert_array_equal(_strata((factors_d - 1) / .1), np.arange(17))
    np.testing.assert_array_equal(_strata((factors_w - 1) / .1), np.arange(17))
    np.testing.assert_allclose(
        ensemble.translation_weights * ensemble.diffusion_parameters_c,
        .05 * 4 * factors_d,
    )


def test_joint_diffusion_stratifies_separately_in_each_vdos_subensemble():
    model = _reference()
    config = TaraConfig(
        n_samples=18, max_energy_shift_mev=1., max_amplitude_shift_pct=2.,
        grid_partition_method='manual',
        manual_energy_ranges=[(0., .04), (.04, .08), (.08, .12)],
        random_seed=100,
    )
    sampler = TaraSampler(config, model.dynamics.continuous)
    sampler.topology_strategy.generate_topology_plan = lambda: iter([
        (0, 8, [(0, .04), (.04, .08), (.08, .12)]),
        (8, 18, [(0, .06), (.06, .12)]),
    ])
    spectra = sampler.generate_ensemble(batch_size=5)
    ensemble = DiffusionSampler(model, _diff_config(), 100).generate_ensemble(18, spectra)
    assert ensemble.spectra_ensemble is spectra
    for start, end in [(0, 8), (8, 18)]:
        info = ensemble.sample_info.iloc[start:end]
        np.testing.assert_array_equal(_strata((info['diffusivity_factor'].to_numpy() - 1) / .1),
                                      np.arange(end - start))
        np.testing.assert_array_equal(_strata((info['weight_factor'].to_numpy() - 1) / .1),
                                      np.arange(end - start))
    np.testing.assert_array_equal(
        ensemble[8].dynamics.continuous.density_values,
        spectra[8].density_values,
    )
    assert ensemble.reference is model


def test_diffusion_stream_does_not_depend_on_vdos_dimensions():
    model = _reference()
    dcfg = _diff_config()
    a = DiffusionSampler(model, dcfg, 33).generate_ensemble(12)
    cfg = TaraConfig(
        n_samples=12, max_energy_shift_mev=1., max_amplitude_shift_pct=2.,
        grid_partition_method='manual', manual_energy_ranges=[(0., .06), (.06, .12)],
        random_seed=33,
    )
    spectra = TaraSampler(cfg, model.dynamics.continuous).generate_ensemble()
    b = DiffusionSampler(model, dcfg, 33).generate_ensemble(12, spectra)
    np.testing.assert_array_equal(a.translation_weights, b.translation_weights)
    np.testing.assert_array_equal(a.diffusion_parameters_c, b.diffusion_parameters_c)


def test_diffusion_monte_carlo_retains_v3_draws():
    """Match legacy diffusion draws regardless of the VDOS topology."""
    model = _reference()
    n_samples = 6
    seed = 12
    sampler = DiffusionSampler(model, _diff_config(), seed, sampling_method='monte_carlo')
    expected = np.random.default_rng(seed)
    diffusivity = expected.uniform(-1, 1, n_samples)
    weight = expected.uniform(-1, 1, n_samples)
    expected_weights = .05 * (1 + .1 * weight)
    expected_c = 4 * (1 + .1 * diffusivity) / (1 + .1 * weight)

    reference_spectrum = model.dynamics.continuous
    energy = reference_spectrum.energy_grid
    densities = reference_spectrum.density_values ** np.linspace(1., 2., n_samples)[:, None]
    generation = EnsembleGeneration(
        amplitude_method='fast', spatial_method='diffeomorphic',
        max_energy_shift_mev=1., max_amplitude_shift_pct=2., n_samples=n_samples,
        grid_partition_method='manual', manual_energy_ranges=((0., .12),),
        domain_splitting_strategy='deterministic', correlation_strategy='independent',
        variable_coupling='independent', shared_grid_structure=True, freeze_origin=True,
    )
    single = SamplingBlock(0, n_samples, np.array([[0., .12]]), np.zeros((n_samples, 1)))
    first = SamplingBlock(0, 2, np.array([[0., .12]]), np.zeros((2, 1)))
    second = SamplingBlock(2, n_samples, np.array([[0., .06], [.06, .12]]), np.zeros((4, 2)))

    spectra_by_topology = [
        SpectraEnsemble(energy, densities, generation, np.full(n_samples, energy.size),
                        blocks, reference=reference_spectrum)
        for blocks in ((single,), (first, second))
    ]
    for spectra in (None, *spectra_by_topology):
        actual = sampler.generate_ensemble(n_samples, spectra)
        np.testing.assert_array_equal(actual.translation_weights, expected_weights)
        np.testing.assert_array_equal(actual.diffusion_parameters_c, expected_c)
        assert actual.reference is model
        assert actual.spectra_ensemble is spectra
        if spectra is not None:
            for index in range(n_samples):
                np.testing.assert_array_equal(
                    actual[index].dynamics.continuous.density_values,
                    spectra[index].density_values,
                )

"""Mathematical and compatibility tests for opt-in liquid diffusion sampling."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from core.config import DiffusionSamplingConfig, TaraConfig, TaraConfigError
from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import (
    DiscreteOscillator, MaterialScatteringModel, ScatteringSpecies, SpeciesDynamics,
)
from core.sampler.concrete_samplers import TaraSampler, VibrationSampler
from core.sampler.diffusion_sampler import DiffusionSampler


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


@pytest.mark.parametrize('energy_bound', [0.0, 1.0])
def test_public_sampler_matches_separate_vibration_and_diffusion_calls(energy_bound):
    """The public API preserves physics and sample alignment in both liquid modes."""
    material = _model()
    diffusion = dict(enabled=True, mode='joint_independent', max_weight_shift_pct=5,
                     max_diffusivity_shift_pct=10, weight_compensation='continuous_only')
    config = TaraConfig(energy_bound, 0.0, n_samples=6, random_seed=16, diffusion=diffusion)
    spectrum = material.dynamics.continuous
    expected_spectra = None
    if energy_bound:
        expected_spectra = VibrationSampler(config, spectrum).generate_ensemble(batch_size=2)
    expected = DiffusionSampler(material, config.diffusion, seed=16).generate_ensemble(
        6, spectra_ensemble=expected_spectra,
    )
    sampler = TaraSampler(config, material=material)
    actual = sampler.generate_ensemble(batch_size=2)

    assert actual.reference is material
    assert len(actual) == 6
    np.testing.assert_array_equal(actual.translation_weights, expected.translation_weights)
    np.testing.assert_array_equal(actual.diffusion_parameters_c, expected.diffusion_parameters_c)
    if expected_spectra is None:
        assert actual.spectra_ensemble is None
        assert actual[0].dynamics.continuous is spectrum
    else:
        assert actual.spectra_ensemble.reference is spectrum
        np.testing.assert_array_equal(actual.spectra_ensemble.pdos, expected_spectra.pdos)
    assert not hasattr(sampler, 'ensemble')


def test_public_diffusion_sampler_requires_a_material():
    """A bare spectrum cannot supply the translational dynamics."""
    config = TaraConfig(n_samples=2, diffusion=_config())
    with pytest.raises(ValueError, match='reference material'):
        TaraSampler(config, spectra=_model().dynamics.continuous)


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

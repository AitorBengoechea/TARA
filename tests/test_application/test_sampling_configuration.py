"""Public TOML controls, default LHS, MC compatibility, and app handoff."""

import numpy as np
import pandas as pd
import pytest

from application.configuration import ApplicationConfigError, load_ace_config
from application.ace_workflow import _prepare_ensemble, _prepare_diffusion_ensemble
from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import (MaterialScatteringModel, SpeciesDynamics,
                                           ScatteringSpecies)


CONFIG = '''
[workflow]
backend = "njoy"
[material]
base_endf_file = "base.endf"
temperature_k = 295.0
state = "liquid"
leapr_file = "water.leapr"
[output]
directory = "out"
[ensemble]
n_samples = 13
include_reference = true
random_seed = 11
batch_size = 3
[execution]
n_jobs = 1
njoy_executable = "njoy"
[sampling]
max_energy_shift_mev = 1.0
max_amplitude_shift_pct = 2.0
grid_partition_method = "manual"
manual_energy_ranges = [[0.0, 0.06], [0.06, 0.12]]
domain_splitting_strategy = "fixed"
correlation_strategy = "independent"
variable_coupling = "independent"
shared_grid_structure = true
freeze_origin = true
fast_amplitude = true
amplitude_method = "fast"
spatial_method = "diffeomorphic"
auto_normalize = true
[ace]
max_energy_ev = 5.0
suffix = ".01"
nbin = 16
tolerance = 0.001
save_xsdir = true
mti = 222
mte = 0
'''


def _load(tmp_path, method=None, *, diffusion=False):
    text = CONFIG
    if method is not None:
        text = text.replace('random_seed = 11', f'random_seed = 11\nsampling_method = "{method}"')
    if diffusion:
        text += '\n[diffusion_sampling]\nenabled = true\nmode = "diffusivity_only"\nmax_diffusivity_shift_pct = 5.0\n'
    path = tmp_path / 'config.toml'
    path.write_text(text)
    return load_ace_config(path)


def _physical():
    x = np.linspace(0, .12, 65)
    pdos = pd.Series(np.sin(np.pi * x / .12) ** 2 + 0.1, index=x)
    spectrum = PhononSpectra(pdos)
    model = MaterialScatteringModel(
        'water', 'liquid', 295., ScatteringSpecies('H', 2, 1.),
        SpeciesDynamics(spectrum, continuous_weight=.95,
                        translation_weight=.05, translation_model='egelstaff_schofield',
                        translation_parameter=4.),
    )
    return model, pdos


def test_missing_sampling_method_is_explicitly_lhs(tmp_path):
    cfg = _load(tmp_path)
    assert cfg.section('ensemble')['sampling_method'] == 'latin_hypercube'
    _, pdos = _physical()
    sampled = _prepare_ensemble(cfg, pdos)
    assert sampled.generation.sampling_method == 'latin_hypercube'
    assert sampled.reference is not None


def test_user_can_choose_monte_carlo_and_run_is_reproducible(tmp_path):
    cfg = _load(tmp_path, 'monte_carlo')
    _, pdos = _physical()
    a = _prepare_ensemble(cfg, pdos)
    b = _prepare_ensemble(cfg, pdos)
    assert a.generation.sampling_method == 'monte_carlo'
    np.testing.assert_array_equal(a.pdos.to_numpy(), b.pdos.to_numpy())


def test_app_joint_liquid_lhs_is_aligned_and_batch_independent(tmp_path):
    cfg = _load(tmp_path, diffusion=True)
    model, pdos = _physical()
    a = _prepare_diffusion_ensemble(cfg, model, pdos)
    b = _prepare_diffusion_ensemble(cfg, model, pdos)
    assert len(a) == cfg.section('ensemble')['n_samples']
    assert a.reference is model
    np.testing.assert_array_equal(a.translation_weights, b.translation_weights)
    np.testing.assert_array_equal(a.diffusion_parameters_c, b.diffusion_parameters_c)
    np.testing.assert_array_equal(
        a.spectra_ensemble.pdos.to_numpy(), b.spectra_ensemble.pdos.to_numpy(),
    )
    # Diffusion stratification is a property of independent factors, not c.
    factors = a.sample_info['diffusivity_factor'].to_numpy()
    n = len(factors)
    np.testing.assert_array_equal(np.sort(np.floor(((factors - 1) / .05 + 1) * n / 2).astype(int)),
                                  np.arange(n))


@pytest.mark.parametrize('method', ['bad', 'LATIN_HYPERCUBE', 'random'])
def test_invalid_method_rejected_at_configuration_time(tmp_path, method):
    with pytest.raises(ApplicationConfigError, match='sampling_method'):
        _load(tmp_path, method)


def test_wrong_method_toml_type_is_rejected(tmp_path):
    path = tmp_path / 'bad.toml'
    path.write_text(CONFIG.replace('random_seed = 11', 'random_seed = 11\nsampling_method = 99'))
    with pytest.raises(ApplicationConfigError, match='sampling_method'):
        load_ace_config(path)


def test_lhs_app_does_not_advance_numpy_global_rng(tmp_path):
    cfg = _load(tmp_path)
    _, pdos = _physical()
    np.random.seed(778)
    expected = np.random.random()
    np.random.seed(778)
    _prepare_ensemble(cfg, pdos)
    assert np.random.random() == expected

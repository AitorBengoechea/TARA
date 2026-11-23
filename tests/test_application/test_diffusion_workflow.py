"""Application-level opt-in diffusion wiring, with no NJOY side effects."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from application import ace_workflow
from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import MaterialScatteringModel, ScatteringSpecies, SpeciesDynamics


def _source():
    """Create a normalized, compact physical liquid reference."""
    pdos = pd.Series([0, 1, 2, 0], index=[0., .01, .02, .03])
    spectrum = PhononSpectra(pdos)
    return MaterialScatteringModel(
        'water', 'liquid', 295., ScatteringSpecies('H', 2, 1.),
        SpeciesDynamics(spectrum, .95, .05, 'egelstaff_schofield', translation_parameter=4.),
    )


def _config(*, mode='diffusivity_only', vib=False):
    """Return a minimal material sampling configuration with explicit opt-in."""
    options = {
        'max_energy_shift_mev': 1.0 if vib else 0.0,
        'max_amplitude_shift_pct': 0.0,
    }
    diffusion = dict(enabled=True, mode=mode, max_diffusivity_shift_pct=5.0,
                     max_weight_shift_pct=0.0)
    return SimpleNamespace(
        sections={'diffusion_sampling': diffusion},
        section=lambda name: {
            'sampling': options,
            'ensemble': dict(n_samples=2, random_seed=120, batch_size=2),
            'diffusion_sampling': diffusion,
        }[name],
    )


def test_diffusion_only_does_not_invoke_vdos_sampler(monkeypatch):
    """Diffusion-only mode must not instantiate the zero-variance TaraConfig."""
    monkeypatch.setattr(ace_workflow, '_prepare_ensemble', lambda *_: pytest.fail('unexpected VDOS sampling'))
    model = _source()
    sample = ace_workflow._prepare_diffusion_ensemble(_config(), model, model.dynamics.continuous.pdos)
    assert sample.reference is model
    assert sample.spectra_ensemble is None
    assert len(sample) == 2
    assert sample[0].dynamics.continuous is model.dynamics.continuous


def test_joint_sampling_reuses_original_vdos_samples(monkeypatch):
    """Combined mode must preserve existing vibrational sampler results."""
    model = _source()
    altered = PhononSpectra(pd.Series([0, 2, 1, 0], index=[0., .01, .02, .03]))

    class Vibrational:
        """Minimal existing ensemble compatible with lazy reconstruction."""
        reference = model.dynamics.continuous
        def __len__(self):
            """Return the requested sample count."""
            return 2
        def __getitem__(self, index):
            """Return an identical precomputed perturbation for either index."""
            return altered

    monkeypatch.setattr(ace_workflow, '_prepare_ensemble', lambda *_: Vibrational())
    new = ace_workflow._prepare_diffusion_ensemble(_config(vib=True), model, model.dynamics.continuous.pdos)
    assert new.spectra_ensemble is not None
    assert new[0].dynamics.continuous is altered
    assert new.reference.dynamics.continuous is model.dynamics.continuous


def test_same_seed_reproduces_diffusion_samples():
    """Independent diffusion stream is reproducible across application calls."""
    model = _source()
    one = ace_workflow._prepare_diffusion_ensemble(_config(), model, model.dynamics.continuous.pdos)
    two = ace_workflow._prepare_diffusion_ensemble(_config(), model, model.dynamics.continuous.pdos)
    np.testing.assert_array_equal(one.diffusion_parameters_c, two.diffusion_parameters_c)


def test_diffusion_provenance_contains_physical_samples(tmp_path):
    """Successful-run metadata retains sampled c, w_t and derived cw."""
    import json
    from pathlib import Path
    from application.configuration import AceApplicationConfig

    config_file = tmp_path / 'config.toml'
    config_file.write_text('[example]\nvalue = true\n')
    config = AceApplicationConfig(
        file_path=config_file,
        sections={
            'workflow': {'backend': 'njoy'},
            'material': {'base_endf_file': 'hydrogen.endf'},
            'ensemble': {'random_seed': 7},
            'diffusion_sampling': {'enabled': True},
        },
    )
    model = _source()
    ensemble = ace_workflow._prepare_diffusion_ensemble(
        _config(), model, model.dynamics.continuous.pdos,
    )
    out = tmp_path / 'result'
    ace_workflow._write_ace_provenance(
        config, out, {'temperature_k': 295, 'base_endf_file': 'hydrogen.endf'},
        Path('/bin/true'), {'sample_000': 'reference.ace', 'sample_001': 'sample.ace'},
        samples=ensemble,
    )
    manifest = json.loads((out / 'run_manifest.json').read_text())
    assert len(manifest['diffusion_samples']) == len(ensemble)
    assert manifest['diffusion_samples'][0]['Sample_Index'] == 0
    assert manifest['diffusion_samples'][0]['diffusion_parameter_c'] > 0

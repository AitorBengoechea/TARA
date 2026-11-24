"""LEAPR liquid material conversion and translation serialization."""


import numpy as np
import pandas as pd
import pytest
from core.data.input_pdos import PhononSpectra
from input_output.leapr.core import LeaprFile
from tests.support.water import make_water_leapr_template
from input_output.leapr.material_adapter import material_model_from_leapr
from input_output.leapr.workflows import serialize_leapr_spectrum
from dataclasses import replace
from core.config import DiffusionSamplingConfig
from core.sampler.diffusion_sampler import DiffusionSampler
from input_output.leapr.workflows import serialize_leapr_material


# Liquid leapr

def test_water_model_from_leapr() -> None:
    """The adapter must keep physical weights, target species and mode identities."""
    model = material_model_from_leapr(
        make_water_leapr_template(), state='liquid', principal_label='H', secondary_label='O',
        oscillator_labels=('bending', 'stretching'),
    )
    assert model.principal.multiplicity == 2
    assert model.secondary[0].multiplicity == 1
    assert model.dynamics.translation_model == 'free_gas'
    assert model.dynamics.continuous_weight == .95
    assert model.dynamics.oscillators == ()
    assert np.isclose(model.dynamics.continuous.integral, 1.0)


def test_sampled_water_leapr_preserves_fixed_contributions(tmp_path) -> None:
    """Full-spectrum serialization removes discrete cards and retains translation."""
    template = make_water_leapr_template()
    spectrum = PhononSpectra(template.pdos)
    text = serialize_leapr_spectrum(template, spectrum, temperature=296., export_full_spectra=True)
    filename = tmp_path / 'sample.leapr'
    filename.write_text(text)
    actual = LeaprFile.from_file(str(filename))
    block = actual.temp_blocks[0]
    assert block['nd'] == 0
    assert not block.get('oscillator_energies', [])
    assert (block['twt'], block['c'], block['tbeta']) == (.05, 0., .95)
    assert actual.principal_scatterer['npr'] == 2
    assert actual.secondary_scatterer['mss'] == 1
    assert np.isclose(np.trapezoid(block['phonon_spectrum'], actual.first_positive_energy_grid), 1., atol=1e-6)
    np.testing.assert_allclose(template.temp_blocks[0]['phonon_spectrum'], [0, 1, 2, 0])


@pytest.mark.parametrize('name', ['twt', 'tbeta', 'nd'])
def test_invalid_liquid_inputs_are_rejected(name: str) -> None:
    """Reject invalid weights and oscillator length inconsistencies."""
    template = make_water_leapr_template()
    template.temp_blocks[0][name] += .02 if name != 'nd' else 1
    with pytest.raises(ValueError):
        material_model_from_leapr(template, state='liquid')


def test_diffusion_uses_the_physical_egelstaff_schofield_model() -> None:
    """Do not reinterpret the dimensionless LEAPR c as an SI diffusion constant."""
    template = make_water_leapr_template()
    template.temp_blocks[0]['c'] = .4
    model = material_model_from_leapr(template, state='liquid')
    assert model.dynamics.translation_model == 'egelstaff_schofield'
    assert model.dynamics.translation_parameter == .4


@pytest.mark.parametrize('c', [-0.1, float('nan'), float('inf')])
def test_invalid_diffusion_parameter_is_rejected(c: float) -> None:
    """Reject invalid diffusion parameters before NJOY processes the deck."""
    template = make_water_leapr_template()
    template.temp_blocks[0]['c'] = c
    with pytest.raises(ValueError, match='diffusion parameter'):
        material_model_from_leapr(template, state='liquid')


# Diffusion material leapr

def _positive_water_template():
    """Construct two distinct physical temperature blocks for leak detection."""
    template = make_water_leapr_template()
    template.temp_blocks[0]['c'] = 4.0
    later = dict(template.temp_blocks[0])
    later['temp'] = 310.
    later['c'] = 5.0
    later['twt'] = .03
    later['tbeta'] = .47
    template.temp_blocks.append(later)
    template.run_control_params['ntempr'] = 2
    return template


def test_sample_specific_leapr_roundtrip_preserves_reference_and_oscillators(tmp_path):
    """Each model writes Card 13 and leaves source temperature parameters intact."""
    template = _positive_water_template()
    model = material_model_from_leapr(template, state='liquid', temperature=310.)
    cfg = DiffusionSamplingConfig(True, 'diffusivity_only', 15.)
    samples = DiffusionSampler(model, cfg, 4).generate_ensemble(2)
    for idx, sample in enumerate((samples.reference, samples[0], samples[1])):
        path = tmp_path / f'mat_{idx}.leapr'
        path.write_text(serialize_leapr_material(template, sample, temperature=310., export_full_spectra=True))
        parsed = LeaprFile.from_file(str(path))
        block = parsed.temp_blocks[0]
        assert block['temp'] == 310.
        assert block['twt'] == pytest.approx(sample.dynamics.translation_weight)
        assert block['c'] == pytest.approx(sample.dynamics.translation_parameter)
        assert block['tbeta'] == pytest.approx(sample.dynamics.continuous_weight)
        assert block['nd'] == 0
        assert not block.get('oscillator_energies', [])
    assert template.temp_blocks[0]['c'] == 4.0
    assert template.temp_blocks[1]['c'] == 5.0


def test_default_spectrum_writer_keeps_translation_exactly():
    """Legacy V2 serialization never mutates Card 13."""
    template = _positive_water_template()
    model = material_model_from_leapr(template, state='liquid', temperature=310.)
    text = serialize_leapr_spectrum(
        template, model.dynamics.continuous, temperature=310.,
        preserve_temperature_parameters=True, export_full_spectra=True,
    )
    assert '0.03 5.0 0.97/' in text or '0.03 5. 0.97/' in text


def test_material_serializer_rejects_wrong_temperature():
    """A 310-K model must never be written into a 296-K LEAPR block."""
    template = _positive_water_template()
    model = material_model_from_leapr(template, state='liquid', temperature=310.)
    with pytest.raises(ValueError, match='temperature'):
        serialize_leapr_material(template, model, temperature=296.)

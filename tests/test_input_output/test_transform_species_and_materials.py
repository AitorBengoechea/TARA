"""Broad native-input transformation regression matrix.

Fixtures containing synthetic species/liquid substitutions test *file-format and
selection behavior*, not physical liquid/different-species kernel equivalence.
Real engine integration tests are separate and require NJOY/NCrystal runtimes.
"""
from pathlib import Path
import re

import numpy as np
import pytest

from input_output.transformations import (
    transform_inputs, available_parameters, InputTransformationError,
)
from input_output.transformations.ncrystal import NCrystalAdapter, _blocks
from input_output.transformations.leapr import LeaprAdapter, _parse_blocks
from input_output.transformations.elements import ATOMIC_MASS_AMU

ROOT = Path(__file__).resolve().parents[1] / 'fixtures' / 'input_transform'
NCMAT = ROOT / 'ncrystal_tsl_HinCH2.ncmat'
LEAPR = ROOT / 'endf81_tsl-HinCH2.leapr'


def _drop_ncmat_species(text, excluded):
    blocks = list(_blocks(text))
    for start, end, block in reversed(blocks):
        match = re.search(r'(?m)^\s*element\s+(\S+)', block)
        if match and match.group(1) in excluded:
            text = text[:start] + text[end:]
    return text


def _carbon_principal(template):
    # Synthetic format fixture, not a physical carbon LEAPR evaluation.
    assert '0.9991673 20.43634' in template
    return template.replace('0.9991673 20.43634', '11.898 20.43634', 1)


def _oxygen_principal(template):
    assert '0.9991673 20.43634' in template
    return template.replace('0.9991673 20.43634', '15.862 20.43634', 1)


def _multiatom(source):
    # Add an O VDOS block using a source C spectral shape, purely as a format test.
    cblock = next(b for _, _, b in _blocks(source) if 'element  C' in b)
    oblock = cblock.replace('element  C', 'element  O', 1).replace('fraction 1/3', 'fraction 1/4', 1)
    return source.replace('fraction 1/3', 'fraction 1/4', 1).replace('fraction 2/3', 'fraction 1/2', 1) + oblock


def _transform(tmp_path, source, template, parameters, name='new.leapr'):
    return transform_inputs(source=source, template=template, parameters=parameters, output=tmp_path / name)


def test_ncmat_discovery_based_on_element_rows():
    assert available_parameters(NCMAT) == ['C_pdos', 'H_pdos']
    assert available_parameters(LEAPR) == ['H_pdos']


def test_dynamic_carbon_spectrum_is_available_in_original_fixture():
    adapter = NCrystalAdapter()
    carbon = adapter.extract(NCMAT.read_text(), 'C_pdos')
    hydrogen = adapter.extract(NCMAT.read_text(), 'H_pdos')
    assert carbon.energy_ev.size != hydrogen.energy_ev.size
    assert carbon.energy_ev[0] > 0


@pytest.mark.parametrize('element', ['C', 'H', 'O', 'Si', 'Al', 'Fe', 'U'])
def test_arbitrary_known_element_is_not_hardcoded(element):
    # Dynamically substitute a species name, then discover the new PDOS.
    text = _drop_ncmat_species(NCMAT.read_text(), {'C'}).replace('element  H', f'element  {element}')
    assert NCrystalAdapter().available_parameters(text) == [f'{element}_pdos']
    spec = NCrystalAdapter().extract(text, f'{element}_pdos')
    assert np.isclose(np.trapezoid(spec.density_per_ev, spec.energy_ev), 1)


def test_carbon_to_carbon_principal_leapr(tmp_path):
    target = tmp_path / 'carbon.leapr'
    target.write_text(_carbon_principal(LEAPR.read_text()))
    assert available_parameters(target) == ['C_pdos']
    generated = _transform(tmp_path, NCMAT, target, ['C_pdos'])
    source = NCrystalAdapter().extract(NCMAT.read_text(), 'C_pdos')
    target_spec = LeaprAdapter().extract(generated.read_text(), 'C_pdos')
    assert len(target_spec.energy_ev) > len(source.energy_ev)
    assert target_spec.energy_ev[0] == 0
    assert target_spec.density_per_ev[0] == 0
    assert np.isclose(np.trapezoid(target_spec.density_per_ev, target_spec.energy_ev), 1)
    assert '11.898 20.43634' in generated.read_text()


def test_carbon_source_rejected_for_hydrogen_principal_leapr(tmp_path):
    with pytest.raises(InputTransformationError, match='not compatible with C'):
        _transform(tmp_path, NCMAT, LEAPR, ['C_pdos'])
    assert not (tmp_path/'new.leapr').exists()


def test_carbon_principal_source_to_original_carbon_ncmat(tmp_path):
    source = tmp_path / 'carbon.leapr'
    source.write_text(_carbon_principal(LEAPR.read_text()))
    result = _transform(tmp_path, source, NCMAT, ['C_pdos'], 'changed.ncmat')
    target = NCrystalAdapter().extract(result.read_text(), 'C_pdos')
    original_h = NCrystalAdapter().extract(NCMAT.read_text(), 'H_pdos')
    new_h = NCrystalAdapter().extract(result.read_text(), 'H_pdos')
    np.testing.assert_array_equal(original_h.density_per_ev, new_h.density_per_ev)
    assert target.energy_ev[0] > 0
    assert target.energy_ev.size == 179
    assert result.read_text() != NCMAT.read_text()


def test_monoatomic_solid_hydrogen(tmp_path):
    mono = _drop_ncmat_species(NCMAT.read_text(), {'C'}).replace('fraction 2/3', 'fraction 1', 1)
    mono_path = tmp_path / 'single.ncmat'
    mono_path.write_text(mono)
    assert available_parameters(mono_path) == ['H_pdos']
    out = _transform(tmp_path, mono_path, LEAPR, ['H_pdos'])
    assert _parse_blocks(out.read_text())[1][0][3].energy_ev.size == 389
    assert mono_path.read_text() == mono


def test_monoatomic_solid_carbon(tmp_path):
    mono = _drop_ncmat_species(NCMAT.read_text(), {'H'}).replace('fraction 1/3', 'fraction 1', 1)
    mono_path = tmp_path / 'mono_carbon.ncmat'
    mono_path.write_text(mono)
    target = tmp_path / 'carbon.leapr'
    target.write_text(_carbon_principal(LEAPR.read_text()))
    assert available_parameters(mono_path) == ['C_pdos']
    out = _transform(tmp_path, mono_path, target, ['C_pdos'])
    assert available_parameters(out) == ['C_pdos']


def test_three_species_multidyninfo_discovery_and_selective_edit(tmp_path):
    three = _multiatom(NCMAT.read_text())
    triple = tmp_path/'triple.ncmat'
    triple.write_text(three)
    assert available_parameters(triple) == ['C_pdos', 'H_pdos', 'O_pdos']
    target_o = tmp_path/'oxygen.leapr'
    target_o.write_text(_oxygen_principal(LEAPR.read_text()))
    out = _transform(tmp_path, triple, target_o, ['O_pdos'])
    assert '15.862 20.43634' in out.read_text()
    assert available_parameters(out) == ['O_pdos']
    original_o = NCrystalAdapter().extract(three, 'O_pdos')
    converted_o = LeaprAdapter().extract(out.read_text(), 'O_pdos')
    assert converted_o.energy_ev[0] == 0
    assert abs(converted_o.energy_ev[-1]-original_o.energy_ev[-1]) < 0.002
    assert three == triple.read_text()


def test_ncrystal_to_ncrystal_copy_two_species_at_once(tmp_path):
    source = tmp_path/'source.ncmat'
    base = NCMAT.read_text()
    src = base.replace('.322584 .534107', '.600000 .534107', 1).replace('.0318665 .0421859', '.090000 .0421859', 1)
    source.write_text(src)
    result = _transform(tmp_path, source, NCMAT, ['C_pdos', 'H_pdos'], name='double.ncmat')
    for sym in ('C', 'H'):
        a = NCrystalAdapter().extract(src, f'{sym}_pdos')
        b = NCrystalAdapter().extract(result.read_text(), f'{sym}_pdos')
        np.testing.assert_allclose(a.density_per_ev, b.density_per_ev, rtol=1e-12)
    assert base == NCMAT.read_text()


def test_multiple_species_to_single_leapr_rejected_atomically(tmp_path):
    # A LEAPR deck does not encode two independent continuous species PDOS.
    output = tmp_path/'not_created.leapr'
    with pytest.raises(InputTransformationError, match='not compatible with C'):
        _transform(tmp_path, NCMAT, LEAPR, ['H_pdos', 'C_pdos'], output.name)
    assert not output.exists()


def test_liquid_continuous_vdos_preserves_card13_and_oscillators(tmp_path):
    # Exercises native formatting / selected fields; not a physical water model.
    liquid = tmp_path/'liquid.ncmat'
    liquid.write_text(NCMAT.read_text().replace('  solid\n', '  liquid\n', 1))
    template = tmp_path/'liquid.leapr'
    template.write_text(LEAPR.read_text().replace('0. 0. 1. 0./\n0/\n',
        '0.3 0.45 0.8 0./\n2/\n0.15 0.35\n0.3 0.7\n', 1))
    out = _transform(tmp_path, liquid, template, ['H_pdos'])
    for line in ('0.3 0.45 0.8 0./', '2/', '0.15 0.35', '0.3 0.7'):
        assert line in out.read_text().splitlines()
    assert _parse_blocks(out.read_text())[1][0][3].energy_ev.size == 389


@pytest.mark.parametrize('unsupported_type', ['freegas', 'sterile', 'scatknl'])
def test_liquid_without_continuous_vdos_rejects_transfer(tmp_path, unsupported_type):
    source = tmp_path/'liquid.ncmat'
    src = NCMAT.read_text().replace('  solid\n', '  liquid\n', 1)
    src = src.replace('  element  H\n  fraction 2/3\n  type     vdos',
                      f'  element  H\n  fraction 2/3\n  type     {unsupported_type}', 1)
    source.write_text(src)
    assert 'H_pdos' not in available_parameters(source)
    with pytest.raises(InputTransformationError, match='not a VDOS'):
        _transform(tmp_path, source, LEAPR, ['H_pdos'])
    assert not (tmp_path/'new.leapr').exists()


def test_no_hydrogen_in_ncmat_source(tmp_path):
    source = tmp_path/'carbon_only.ncmat'
    source.write_text(_drop_ncmat_species(NCMAT.read_text(), {'H'}))
    assert available_parameters(source) == ['C_pdos']
    with pytest.raises(InputTransformationError, match='Expected exactly one'):
        _transform(tmp_path, source, LEAPR, ['H_pdos'])


def test_nonexistent_element_never_written(tmp_path):
    with pytest.raises(InputTransformationError, match='Expected exactly one'):
        _transform(tmp_path, NCMAT, LEAPR, ['O_pdos'])
    assert not (tmp_path/'new.leapr').exists()


def test_duplicate_element_vdos_blocks_not_advertised(tmp_path):
    duplicate = NCMAT.read_text()
    c = next(b for _, _, b in _blocks(duplicate) if 'element  C' in b)
    dup_path = tmp_path/'duplicates.ncmat'
    dup_path.write_text(duplicate + c)
    assert available_parameters(dup_path) == ['H_pdos']
    with pytest.raises(InputTransformationError, match='exactly one'):
        _transform(tmp_path, dup_path, LEAPR, ['C_pdos'])


def test_incompatible_oxygen_principal_is_not_hydrogen(tmp_path):
    target = tmp_path/'oxygen.leapr'
    target.write_text(_oxygen_principal(LEAPR.read_text()))
    with pytest.raises(InputTransformationError, match='not compatible with H'):
        _transform(tmp_path, NCMAT, target, ['H_pdos'])


def test_leapr_species_discovery_refuses_unidentified_awr(tmp_path):
    target = tmp_path/'non_element.leapr'
    target.write_text(LEAPR.read_text().replace('0.9991673 20.43634', '0.35 20.43634'))
    assert available_parameters(target) == []
    with pytest.raises(InputTransformationError, match='not compatible with H'):
        _transform(tmp_path, NCMAT, target, ['H_pdos'])


def test_liquid_fallback_no_implicit_diffusion_renormalization(tmp_path):
    # Transformation changes Card 11/12 only, not translational/oscillator cards.
    target = tmp_path/'liquid.leapr'
    target.write_text(LEAPR.read_text().replace('0. 0. 1. 0./', '0.45 0.34 0.64 0./'))
    old_lines, old = _parse_blocks(target.read_text())
    out = _transform(tmp_path, NCMAT, target, ['H_pdos'])
    new_lines, new = _parse_blocks(out.read_text())
    assert old_lines[old[0][2]:] == new_lines[new[0][2]:]
    assert '0.45 0.34 0.64 0./' in out.read_text()


def test_dynamic_names_use_case_sensitive_real_element_symbols(tmp_path):
    with pytest.raises(InputTransformationError, match='Unrecognized element'):
        _transform(tmp_path, NCMAT, LEAPR, ['c_pdos'])
    with pytest.raises(InputTransformationError, match='Unrecognized element'):
        _transform(tmp_path, NCMAT, LEAPR, ['Xx_pdos'])
    assert 'C' in ATOMIC_MASS_AMU and 'O' in ATOMIC_MASS_AMU

"""Native-file transformations: full polyethylene fixtures and liquid cases."""
from pathlib import Path
import numpy as np
import pytest

from input_output.transformations import (
    transform_inputs, register_adapter, ContinuousSpectrum, InputTransformationError
)
from input_output.transformations.leapr import LeaprAdapter, _parse_blocks
from input_output.transformations.ncrystal import NCrystalAdapter
from input_output.transformations.spectra import to_leapr_grid

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'input_transform'
NCMAT = FIXTURES / 'ncrystal_tsl_HinCH2.ncmat'
LEAPR = FIXTURES / 'endf81_tsl-HinCH2.leapr'


def make(tmp_path, source=NCMAT, template=LEAPR, output='result.leapr'):
    return transform_inputs(source=source, template=template,
                            parameters=['H_pdos'], output=tmp_path / output)


def _lambda(spectrum, temp=293.6):
    e, g = spectrum.energy_ev, spectrum.density_per_ev
    kbt = temp * 8.617333262145e-5
    kernel = np.empty_like(g)
    pos = e > 0
    kernel[pos] = kbt*g[pos] / (e[pos]*np.tanh(e[pos]/(2*kbt)))
    kernel[~pos] = 2*kbt*kbt*g[1]/e[1]**2
    return float(np.trapezoid(kernel, e))


def test_ncrystal_h_to_leapr_fixture_preserves_everything_else(tmp_path):
    out = make(tmp_path)
    source_text = NCMAT.read_text()
    original = LEAPR.read_text()
    new = out.read_text()
    assert source_text == NCMAT.read_text()
    assert original == LEAPR.read_text()
    old_lines, old_blocks = _parse_blocks(original)
    new_lines, new_blocks = _parse_blocks(new)
    old_ctl, old_start, old_end, _ = old_blocks[0]
    new_ctl, new_start, new_end, newspec = new_blocks[0]
    assert len(newspec.energy_ev) == 389
    assert np.isclose(newspec.energy_ev[1], 0.0009999665079365079, rtol=1e-9)
    assert np.isclose(np.trapezoid(newspec.density_per_ev, newspec.energy_ev), 1.)
    assert abs(_lambda(newspec)-0.579552168) < 2e-6
    assert old_lines[:old_ctl] == new_lines[:new_ctl]
    assert old_lines[old_end:] == new_lines[new_end:]
    assert old_lines[old_ctl].split('/')[1] == new_lines[new_ctl].split('/')[1]
    assert out == tmp_path / 'result.leapr'


def test_ncrystal_original_ordinates_preserved_on_aligned_grid():
    nc = NCrystalAdapter().extract(NCMAT.read_text(), 'H_pdos')
    converted = to_leapr_grid(nc)
    assert len(converted.energy_ev) == 389
    assert converted.density_per_ev[0] == 0
    np.testing.assert_allclose(
        converted.density_per_ev[10:] / converted.density_per_ev[10],
        nc.density_per_ev / nc.density_per_ev[0], rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(converted.density_per_ev[:10] / converted.density_per_ev[10],
                               (np.arange(10)/10)**2, rtol=1e-5, atol=1e-12)


def test_leapr_to_ncrystal_preserves_carbon_and_metadata(tmp_path):
    out = make(tmp_path, source=LEAPR, template=NCMAT, output='modified.ncmat')
    text, original = out.read_text(), NCMAT.read_text()
    assert text[:text.index('@DYNINFO\n  element  H')] == original[:original.index('@DYNINFO\n  element  H')]
    assert '  fraction 2/3' in text
    assert '0.92 g_per_cm3' in text
    original_leapr = LeaprAdapter().extract(LEAPR.read_text(), 'H_pdos')
    replaced = NCrystalAdapter().extract(text, 'H_pdos')
    assert len(replaced.energy_ev) == 179
    np.testing.assert_allclose(replaced.density_per_ev / replaced.density_per_ev.max(),
                               original_leapr.density_per_ev[1:] / original_leapr.density_per_ev[1:].max(),
                               rtol=2e-11, atol=1e-13)


def test_multi_temperature_template_all_positive_blocks_updated(tmp_path):
    # Turn the first reused negative temperature into a fully specified positive
    # second block; leave remaining negative temperatures intact.
    original = LEAPR.read_text()
    marker = '-196/ temperature\n'
    _, blocks = _parse_blocks(original)
    lines, _ = _parse_blocks(original)
    _, start, end, _ = blocks[0]
    # Card 11 through 14 are before first reused-temperature Card 10.
    additional = '196/ temperature\n' + ''.join(lines[blocks[0][0]:end+2])
    assert marker in original
    modified = original.replace(marker, additional, 1)
    template = tmp_path/'multi.leapr'
    template.write_text(modified)
    assert len(_parse_blocks(modified)[1]) == 2
    out = make(tmp_path, template=template)
    parsed = _parse_blocks(out.read_text())[1]
    assert len(parsed) == 2
    assert all(len(block[3].energy_ev) == 389 for block in parsed)
    np.testing.assert_array_equal(parsed[0][3].density_per_ev, parsed[1][3].density_per_ev)
    assert '-233/ temperature' in out.read_text()


def test_liquid_like_templates_preserve_translational_model(tmp_path):
    # Physical liquid source/target with continuous H spectrum. Only the PDOS
    # changes; translational Card 13 is not implicitly converted or erased.
    liquid_nc = tmp_path/'liquid.ncmat'
    liquid_nc.write_text(NCMAT.read_text().replace('  solid\n', '  liquid\n', 1))
    liquid_leapr = tmp_path/'liquid.leapr'
    liquid_leapr.write_text(LEAPR.read_text().replace('0. 0. 1. 0./', '0.3 0.45 0.8 0./', 1))
    out = make(tmp_path, source=liquid_nc, template=liquid_leapr)
    assert '0.3 0.45 0.8 0./' in out.read_text()
    assert '0.3 0.45 0.8 0./' in liquid_leapr.read_text()
    assert len(_parse_blocks(out.read_text())[1][0][3].energy_ev) == 389


def test_liquid_unsupported_dynamics_rejected(tmp_path):
    src = tmp_path/'diffusive.ncmat'
    src.write_text(NCMAT.read_text().replace('type     vdos', 'type     freegas', 1)
                   .replace('  element  H\n  fraction 2/3\n  type     vdos',
                            '  element  H\n  fraction 2/3\n  type     freegas', 1))
    output = tmp_path/'reject.leapr'
    with pytest.raises(InputTransformationError, match='not a VDOS'):
        make(tmp_path, source=src, output='reject.leapr')
    assert not output.exists()


def test_error_cases_are_atomic_and_do_not_overwrite(tmp_path):
    same = tmp_path/'same.leapr'
    same.write_text(LEAPR.read_text())
    with pytest.raises(InputTransformationError, match='Output must differ'):
        make(tmp_path, template=same, output='same.leapr')
    with pytest.raises(InputTransformationError, match='not a VDOS|cannot extract|Unsupported parameter'):
        transform_inputs(source=NCMAT, template=LEAPR,
                         parameters=['diffusion'], output=tmp_path/'bad.leapr')
    assert not (tmp_path/'bad.leapr').exists()
    out = make(tmp_path)
    with pytest.raises(FileExistsError):
        make(tmp_path)
    assert out.exists()
    with pytest.raises(InputTransformationError, match='output|Output'):
        make(tmp_path, output='different.ncmat')
    with pytest.raises(InputTransformationError, match='sequence'):
        transform_inputs(source=NCMAT, template=LEAPR,
                         parameters='H_pdos', output=tmp_path/'bad.leapr')


def test_non_vdos_h_dynamics_and_ambiguous_species_rejected(tmp_path):
    original = NCMAT.read_text()
    altered = tmp_path/'bad.ncmat'
    altered.write_text(original.replace('  element  H\n  fraction 2/3\n  type     vdos',
                                        '  element  H\n  fraction 2/3\n  type     freegas'))
    with pytest.raises(InputTransformationError, match='not a VDOS'):
        make(tmp_path, source=altered)
    altered.write_text(original.replace('  element  H\n', '  element  D\n'))
    with pytest.raises(InputTransformationError, match='exactly one'):
        make(tmp_path, source=altered)
    altered.write_text(original + '\n@DYNINFO\n  element H\n  fraction 0\n  type vdos\n')
    with pytest.raises(InputTransformationError, match='exactly one'):
        make(tmp_path, source=altered)


def test_leapr_wrong_principal_mass_rejected(tmp_path):
    target = tmp_path/'nonH.leapr'
    target.write_text(LEAPR.read_text().replace('0.9991673 20.43634', '11.898 20.43634'))
    with pytest.raises(InputTransformationError, match='not compatible with H'):
        make(tmp_path, template=target)


def test_extensibility_adapter_registration(tmp_path):
    # Introduce a 3rd engine without any NJOY/NCrystal pairwise logic.
    class DemoAdapter:
        engine = 'future'
        suffixes = ('.futuretara',)
        def extract(self, text, parameter):
            assert parameter == 'H_pdos'
            return ContinuousSpectrum(np.array([0., .01, .02]), np.array([0., 1., 2.]))
        def replace(self, text, parameter, spectrum):
            return text + f'\n{parameter}={len(spectrum.energy_ev)}\n'
    register_adapter(DemoAdapter())
    source = tmp_path/'third.futuretara'
    source.write_text('source')
    out = make(tmp_path, source=source, output='third.leapr')
    assert len(_parse_blocks(out.read_text())[1][0][3].energy_ev) == 3
    assert (tmp_path/'third.leapr').exists()


def test_liquid_oscillator_blocks_are_preserved(tmp_path):
    # Positive-temperature liquid-like LEAPR with two discrete oscillators.
    # Card 15/16 arrays are count-delimited with no slash, as in some CAB decks.
    original = LEAPR.read_text()
    additional = '0.4 0.01 0.02 0./\n2/\n0.15 0.35\n0.3 0.7\n'
    modified = original.replace('0. 0. 1. 0./\n0/\n', additional, 1)
    template = tmp_path/'oscillator.leapr'
    template.write_text(modified)
    _, blocks = _parse_blocks(modified)
    assert len(blocks) == 1
    out = make(tmp_path, template=template)
    for line in additional.splitlines():
        assert line in out.read_text().splitlines()


def test_ambiguous_multiple_leapr_source_spectra_rejected(tmp_path):
    original = LEAPR.read_text()
    lines, blocks = _parse_blocks(original)
    _, start, end, _ = blocks[0]
    duplicate = '196/ temperature\n' + ''.join(lines[blocks[0][0]:end+2])
    duplicate = duplicate.replace('2.185109E-01', '4.370218E-01', 1)
    modified = original.replace('-196/ temperature\n', duplicate, 1)
    source = tmp_path/'two_distinct.leapr'
    source.write_text(modified)
    with pytest.raises(InputTransformationError, match='distinct positive-temperature'):
        transform_inputs(source=source, template=NCMAT,
                         parameters=['H_pdos'], output=tmp_path/'rejected.ncmat')
    assert not (tmp_path/'rejected.ncmat').exists()


def test_source_has_nonuniform_energy_grid_not_silently_accepted():
    from input_output.transformations.spectra import to_leapr_grid
    bad = ContinuousSpectrum([0.01, 0.02, 0.034], [1, 2, 3])
    with pytest.raises(InputTransformationError, match='regular grid'):
        to_leapr_grid(bad)

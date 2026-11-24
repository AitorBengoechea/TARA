"""Pure-Python tests: no NCrystal dependency required."""
from pathlib import Path
import pytest

from processing.backends.ncmat_isolation import (
    isolate_ncmat_dyninfo,
    normalize_active_species,
)

SIMPLE = """NCMAT v5
@DENSITY
  0.92 g_per_cm3
@DYNINFO
  element C
  fraction 1/3
  type vdos
  vdos_egrid 0.01 0.04
  vdos_density 1 2 3
@DYNINFO
  element H
  fraction 2/3
  type vdos
  vdos_egrid 0.01 0.04
  vdos_density 4 5 6
@OTHER
  keep this
"""


def test_isolation_keeps_exact_fraction_and_non_dyninfo_sections():
    result = isolate_ncmat_dyninfo(SIMPLE, {"H"})
    assert "  fraction 1/3\n  type sterile\n" in result
    assert "  fraction 2/3\n  type vdos\n" in result
    assert "vdos_density 1 2 3" not in result
    assert "vdos_density 4 5 6" in result
    assert result.startswith("NCMAT v5\n@DENSITY\n  0.92 g_per_cm3\n")
    assert result.endswith("@OTHER\n  keep this\n")
    assert isolate_ncmat_dyninfo(result, {"H"}) == result
    assert SIMPLE == isolate_ncmat_dyninfo(SIMPLE, {"H", "C"})


def test_isolation_reverse_carbon():
    result = isolate_ncmat_dyninfo(SIMPLE, {"C"})
    assert "vdos_density 1 2 3" in result
    assert "vdos_density 4 5 6" not in result
    assert "  fraction 2/3\n  type sterile" in result


def test_two_h_sites_all_stay_active():
    two_h = SIMPLE.replace("@OTHER", "@DYNINFO\n  element H\n  fraction 1/6\n  type freegas\n@OTHER")
    output = isolate_ncmat_dyninfo(two_h, {"H"})
    assert "  fraction 1/6\n  type freegas" in output
    assert output.count("  type sterile") == 1


@pytest.mark.parametrize('selection,error_type', [
    ('H', TypeError), ([], ValueError), ({'Q'}, ValueError),
    ({'H', 'Missing'}, ValueError), ({'bad label'}, ValueError),
])
def test_reject_invalid_or_unknown_selection(selection,error_type):
    if error_type is TypeError or selection == [] or selection == {'bad label'}:
        with pytest.raises(error_type):
            normalize_active_species(selection)
    else:
        with pytest.raises(error_type, match='Unknown'):
            isolate_ncmat_dyninfo(SIMPLE, selection)


def test_reject_activated_sterile_species():
    output = isolate_ncmat_dyninfo(SIMPLE, {'H'})
    with pytest.raises(ValueError, match='already sterile'):
        isolate_ncmat_dyninfo(output, {'C'})


def test_reject_malformed_inactive_block():
    with pytest.raises(ValueError, match='element, fraction, and type'):
        isolate_ncmat_dyninfo(SIMPLE.replace('  fraction 1/3\n', ''), {'H'})
    with pytest.raises(ValueError, match='Repeated'):
        isolate_ncmat_dyninfo(SIMPLE.replace('  fraction 1/3\n','  fraction 1/3\n  fraction 1/3\n'), {'H'})


def test_ch2_reference_fixture_preserves_fraction_and_h_vdos():
    here = Path(__file__).resolve()
    fixtures = (
        here.parents[2] / 'tests' / 'fixtures' / 'ncrystal_tsl_HinCH2.ncmat',
        here.parents[2] / 'tests' / 'fixtures' / 'input_transform' / 'ncrystal_tsl_HinCH2.ncmat',
    )
    fixture = next((p for p in fixtures if p.exists()), None)
    if fixture is None:
        pytest.skip('Optional CH2 fixture not present')
    original = fixture.read_text(encoding='utf-8')
    result = isolate_ncmat_dyninfo(original, {'H'})
    assert 'fraction 1/3' in result
    assert 'fraction 2/3' in result
    assert result.count('type sterile') == 1

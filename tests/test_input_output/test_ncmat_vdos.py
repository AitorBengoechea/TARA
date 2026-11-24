"""NCMAT VDOS grids, liquid reconstruction, and representation diagnostics."""


from pathlib import Path
import numpy as np
from input_output.ncrystal.core import NCrystalFile
from input_output.spectrum_grid import resample_on_source_spacing
import pytest
from types import SimpleNamespace
import re
from input_output.ncrystal.liquids import replace_vdos
from input_output.ncrystal.liquids import validate_regular_vdos_grid
from input_output.ncrystal.material_adapter import fold_solidlike_vdos
from input_output.ncrystal.material_adapter import displacement_moment
from input_output.ncrystal.liquids import LiquidAtomSettings
from input_output.ncrystal.liquids import add_custom_liquids
from input_output.ncrystal.liquids import parse_custom_liquids
from input_output.leapr.core import LeaprFile
from input_output.leapr.material_adapter import material_model_from_leapr
from input_output.ncrystal.liquids import load_liquid_ncmat


# Ch2 source spacing

CH2_FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'ncrystal_tsl_HinCH2.ncmat'


def test_hydrogen_output_keeps_379_source_points_and_carbon_is_unmodified():
    """Artificial low-energy Debye points do not enter serialized H VDOS."""
    original = NCrystalFile.from_file(str(CH2_FIXTURE))
    c_before = original.pdos_for_atom('C')
    h_before = original.pdos_for_atom('H')
    h_grid = h_before.index.to_numpy(dtype=float)
    tail_grid = np.linspace(0.0, h_grid[0], 2002)[:-1]
    grid = np.concatenate((tail_grid, h_grid))
    density = np.interp(grid, h_grid, h_before.to_numpy(dtype=float), left=0, right=0)
    output_grid, _ = resample_on_source_spacing(h_grid, grid, density)
    np.testing.assert_array_equal(output_grid, h_grid)
    assert len(output_grid) == 379
    np.testing.assert_array_equal(original.pdos_for_atom('C').to_numpy(), c_before.to_numpy())


def test_hydrogen_shift_extends_grid_by_input_spacing():
    """No carbon grid change is required for a positively shifted H spectrum."""
    original = NCrystalFile.from_file(str(CH2_FIXTURE))
    h = original.pdos_for_atom('H')
    ref_grid = h.index.to_numpy(dtype=float)
    step = (ref_grid[-1] - ref_grid[0]) / (len(ref_grid) - 1)
    src_grid = np.r_[np.linspace(0, ref_grid[0], 2002)[:-1], ref_grid,
                     ref_grid[-1] + step * np.arange(1, 6)]
    src_density = np.exp(-0.5 * ((src_grid - (ref_grid[-1] + 2 * step)) / step)**2)
    output_grid, _ = resample_on_source_spacing(ref_grid, src_grid, src_density)
    np.testing.assert_array_equal(output_grid[:len(ref_grid)], ref_grid)
    assert len(output_grid) == len(ref_grid) + 5
    np.testing.assert_allclose(np.diff(output_grid)[-5:], step, atol=1e-14, rtol=0)


# Source spacing

def test_original_grid_is_identical_without_energy_extension():
    """Reconstructing a Debye tail must not create new NCMAT points."""
    original = 0.01 + 0.001 * np.arange(8)
    dense = np.concatenate((np.linspace(0, 0.01, 2001)[:-1], original))
    signal = dense**2 * np.exp(-100 * dense)
    output_energy, output_density = resample_on_source_spacing(original, dense, signal)
    np.testing.assert_array_equal(output_energy, original)
    np.testing.assert_allclose(output_density, np.interp(original, dense, signal))


def test_shifted_upper_peak_extends_only_at_original_spacing():
    """A higher-energy shifted peak gains extra *coarse* points, not Debye points."""
    original = 0.01 + 0.001 * np.arange(8)
    dense = np.linspace(0, original[-1] + 0.0024, 4200)
    signal = np.exp(-0.5 * ((dense - (original[-1] + 0.0018)) / 0.0004)**2)
    output_energy, output_density = resample_on_source_spacing(original, dense, signal)
    np.testing.assert_array_equal(output_energy[:len(original)], original)
    assert len(output_energy) == len(original) + 3
    np.testing.assert_allclose(np.diff(output_energy)[-3:], 0.001, rtol=0, atol=1e-14)
    assert output_density[-2] > 0.0
    assert output_density[-1] == 0.0  # sample beyond source support


def test_invalid_original_grid_is_rejected_without_regularizing():
    """Nonuniform source spacing is an error, not a reason to alter physics."""
    with pytest.raises(ValueError, match="uniform"):
        resample_on_source_spacing([0.01, 0.02, 0.035], [0, 0.02, 0.04], [0, 1, 0])


def test_spectral_density_is_not_implicitly_renormalized():
    """Output resampling is deterministic translation, not a physical operator."""
    original = 0.01 + 0.001 * np.arange(12)
    internal = np.linspace(0, original[-1], 1000)
    signal = 4.2 * np.ones_like(internal)
    grid, vals = resample_on_source_spacing(original, internal, signal)
    np.testing.assert_array_equal(grid, original)
    np.testing.assert_allclose(vals, 4.2)


# Ncrystal liquid production diagnostics

def test_example_water_fixture_is_valid_ncmat_v7():
    path = Path(__file__).parents[1] / 'fixtures' / 'liquid_water_295K_illustrative.ncmat'
    source = path.read_text()
    assert 'vdos_density ' in source
    assert not re.search(r'(?m)^\s*vdos\s*$', source)
    assert '@CUSTOM_LIQUIDS' in source
    chunk = source.split('@DYNINFO')[1].split('@DYNINFO')[0]
    e0, e1 = map(float, re.search(r'vdos_egrid\s+(\S+)\s+(\S+)', chunk).groups())
    parts = re.split(r'\bvdos_density\s+', chunk, maxsplit=1)[1].split()
    y = np.array([float(part) for part in parts])
    grid = np.linspace(e0, e1, y.size)
    validate_regular_vdos_grid(grid)
    assert len(grid) == 1201
    assert np.isfinite(y).all() and np.all(y >= 0)


def test_water_fixture_retains_the_low_energy_warning():
    """Reproduce the V5 sensitivity; syntax compliance is not validation."""
    reference = Path(__file__).parents[1] / 'fixtures' / 'liquid_water_295K_illustrative.ncmat'
    src = reference.read_text()
    part = src.split('@DYNINFO')[1].split('@DYNINFO')[0]
    e0, e1 = map(float, re.search(r'vdos_egrid\s+(\S+)\s+(\S+)', part).groups())
    values = np.array([float(x) for x in re.split(r'\bvdos_density\s+', part, maxsplit=1)[1].split()])
    aligned = np.linspace(e0, e1, len(values))
    # Fixture's previous, *invalid* grid, recovered solely for comparison.
    original = np.linspace(1e-5, 0.423, len(values))
    # A smooth explanatory counterexample independent of plugin installation.
    y = (original / .025) * np.exp(-original / .07)
    y /= np.trapezoid(y, original)
    new = np.interp(aligned, original, y, left=0, right=0)
    new /= np.trapezoid(new, aligned)
    change = displacement_moment(aligned, new, 295) / displacement_moment(original, y, 295) - 1
    assert abs(change) > .10


def test_folded_grid_is_regular_and_diagnostics_are_nonmutating():
    energy = np.linspace(1e-5, .423, 1201)
    original = (energy / .025) * np.exp(-energy / .07)
    original /= np.trapezoid(original, energy)
    source = SimpleNamespace(energy_grid=energy, density_values=original)
    dynamics = SimpleNamespace(continuous_weight=.8, translation_weight=.2, oscillators=())
    material = SimpleNamespace(dynamics=dynamics, temperature_k=295.)
    e0, d0 = energy.copy(), original.copy()
    grid, vals, diagnostics = fold_solidlike_vdos(material, source, return_diagnostics=True)
    assert validate_regular_vdos_grid(grid) > 0
    assert np.trapezoid(vals, grid) == pytest.approx(1, abs=1e-5)
    assert diagnostics.regular_min_ev / diagnostics.regular_step_ev == pytest.approx(1)
    assert np.isfinite(diagnostics.displacement_moment_relative_change)
    np.testing.assert_array_equal(source.energy_grid, e0)
    np.testing.assert_array_equal(source.density_values, d0)


# Ncrystal liquid vdos regressions

def _template():
    base = ("NCMAT v7\n@DENSITY\n  1 g_per_cm3\n"
            "@DYNINFO\n  element H\n  fraction 1\n  type vdos\n"
            "  vdos_egrid 0.001 0.004\n  vdos\n    1 2 2 1\n")
    return add_custom_liquids(
        base, [LiquidAtomSettings(atom='H', model='incoherent', c=1.0,
                                  wt=0.2, ws=0.8)])


def test_serializer_accepts_valid_v7_input_density_field():
    """Production parser must also accept already valid NCMAT v7 templates."""
    legacy = _template()
    valid = legacy.replace('  vdos\n    1 2 2 1', '  vdos_density 1 2 2 1')
    g, y = _grid_and_vdos()
    rendered = replace_vdos(valid, 'H', g, y)
    assert re.search(r'(?m)^\s*vdos_density\s+[0-9]', rendered)


def _grid_and_vdos():
    grid = np.array([0.001, 0.002, 0.003, 0.004])  # E_min/step = 1
    values = np.array([100., 200., 200., 100.])
    return grid, values


def test_ncmat_v7_density_keyword_and_first_values_share_line():
    """V1: 'vdos' alone on its own line was rejected by NC.createInfo."""
    g, y = _grid_and_vdos()
    rendered = replace_vdos(_template(), 'H', g, y)
    assert not re.search(r'(?m)^\s*vdos\s*$', rendered), (
        'Legacy standalone vdos keyword found; NCMAT v7 requires vdos_density.')
    match = re.search(r'(?m)^\s*vdos_density\s+([^\n]+)$', rendered)
    assert match is not None, 'Missing inline vdos_density values in generated NCMAT.'
    first_row = [float(value) for value in match.group(1).split()]
    assert len(first_row) >= 1 and first_row[0] == pytest.approx(y[0])
    assert '@CUSTOM_LIQUIDS' in rendered


def test_vdos_egrid_is_aligned_to_integer_spacing():
    """V4: reject misaligned energy grids *before* entering native C++ code.

    A correctly aligned input should serialize successfully; an unaligned one
    must fail with a Python exception, not trigger 'mm <= 20000' in NCrystal.
    """
    g, y = _grid_and_vdos()
    assert 'vdos_egrid' in replace_vdos(_template(), 'H', g, y)
    misaligned = g - 0.00099  # E_min=1e-5, spacing 0.001
    with pytest.raises(ValueError, match=r'(?i)(grid|spacing|align|regular|energy)'):
        replace_vdos(_template(), 'H', misaligned, y)


@pytest.mark.parametrize('grid,values', [
    (np.array([0.001, 0.002, 0.003]), np.array([1., np.nan, 1.])),
    (np.array([0.001, 0.002, 0.003]), np.array([1., -1., 1.])),
    (np.array([0.001, 0.002, 0.0034]), np.array([1., 2., 1.])),
    (np.array([0.001, 0.002, 0.002]), np.array([1., 2., 1.])),
])
def test_invalid_vdos_rejected_before_serialization(grid, values):
    with pytest.raises(ValueError):
        replace_vdos(_template(), 'H', grid, values)


def test_liquid_settings_and_other_sections_preserved():
    """Serialization must change target dynamics without erasing liquid physics."""
    g, y = _grid_and_vdos()
    template = _template()
    rendered = replace_vdos(template, 'H', g, y)
    assert rendered.count('@CUSTOM_LIQUIDS') == 1
    before = parse_custom_liquids(template)
    after = parse_custom_liquids(rendered)
    assert before == after
    assert rendered.count('@DENSITY') == 1
    assert rendered.count('@DYNINFO') == 1


def test_serializer_does_not_mutate_input_arrays():
    g, y = _grid_and_vdos()
    saved_g, saved_y = g.copy(), y.copy()
    replace_vdos(_template(), 'H', g, y)
    np.testing.assert_array_equal(g, saved_g)
    np.testing.assert_array_equal(y, saved_y)


# Ncrystal solidlike adapter contract

def _inputs():
    base_grid = np.linspace(.001, .4, 400)
    spectrum = np.exp(-((base_grid-.08)/.025)**2)
    spectrum /= np.trapezoid(spectrum, base_grid)
    continuous = SimpleNamespace(energy_grid=base_grid, density_values=spectrum)
    oscillators = (SimpleNamespace(energy_ev=.205, weight=.2),
                   SimpleNamespace(energy_ev=.32, weight=.1))
    dynamics = SimpleNamespace(continuous_weight=.5, translation_weight=.2,
                               oscillators=oscillators)
    material = SimpleNamespace(dynamics=dynamics)
    return material, continuous


def test_fold_constructs_regular_grid_from_start():
    """V4/V5: avoid constructing E_min=1e-5 with E_min/step≈0.03."""
    material, continuous = _inputs()
    grid, values = fold_solidlike_vdos(material, continuous, n_grid=1201)
    step = (grid[-1]-grid[0])/(len(grid)-1)
    assert step > 0 and np.allclose(np.diff(grid), step, rtol=1e-6)
    ratio = grid[0]/step
    assert ratio >= 1 and abs(ratio - round(ratio)) < 1e-6, (
        f'NCrystal regular-grid contract violated: E_min/dE={ratio:.8g}')
    assert np.isfinite(values).all() and (values >= 0).all()
    assert np.trapezoid(values, grid) == pytest.approx(1, abs=2e-5)


def test_fold_retains_input_spectrum_and_oscillator_modes():
    """Model composition must remain reversible and never mutate core arrays."""
    material, continuous = _inputs()
    e0 = continuous.energy_grid.copy()
    y0 = continuous.density_values.copy()
    grid, values = fold_solidlike_vdos(material, continuous, n_grid=1201)
    np.testing.assert_array_equal(continuous.energy_grid, e0)
    np.testing.assert_array_equal(continuous.density_values, y0)
    assert values[np.argmin(abs(grid-.205))] > 0
    assert values[np.argmin(abs(grid-.32))] > 0
    assert np.trapezoid(values, grid) == pytest.approx(1, abs=2e-5)


# CAB water VDOS conversion

FIXTURE = Path(__file__).parents[1] / 'fixtures' / 'tsl-HinH2O.leapr'
NCMAT = Path(__file__).parents[1] / 'fixtures' / 'liquid_water_295K_illustrative.ncmat'


def test_cab_water_is_not_special_in_processing():
    leapr = LeaprFile.from_file(str(FIXTURE))
    assert len(leapr.temp_blocks) == 94
    material = material_model_from_leapr(leapr, state='liquid', temperature=295,
                                         principal_label='H', secondary_label='O')
    energy, density = fold_solidlike_vdos(material, material.dynamics.continuous)
    assert np.isclose(np.trapezoid(density, energy), 1., atol=2e-6)
    assert len(material.dynamics.oscillators) == 0
    source = load_liquid_ncmat(NCMAT)
    updated = replace_vdos(source, 'H', energy, density)
    assert source.split('@CUSTOM_LIQUIDS')[1] == updated.split('@CUSTOM_LIQUIDS')[1]
    assert parse_custom_liquids(updated)[0].atom == 'H'
    assert 'element O' in updated


def test_cab_temperature_must_be_exact():
    with pytest.raises(ValueError):
        material_model_from_leapr(LeaprFile.from_file(str(FIXTURE)),
                                  state='liquid', temperature=296,
                                  principal_label='H', secondary_label='O')

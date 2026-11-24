"""Configuration regression tests."""


from __future__ import annotations


from copy import deepcopy
from pathlib import Path
import tomllib
import pytest
from application.configuration import ApplicationConfigError
from application.configuration import load_ace_config


# Configuration

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / 'application' / 'configs'


@pytest.mark.parametrize('backend', ('njoy', 'ncrystal'))
@pytest.mark.parametrize('state', ('solid', 'liquid'))
def test_templates_parse_and_paths_are_relative_to_config(backend, state):
    """All four shipped configurations load and resolve their material paths."""
    config = load_ace_config(TEMPLATES / f'ace_{backend}_{state}.toml')
    assert config.backend == backend
    assert config.section('material').get('state', 'solid') == state
    neutron_file = 'n_1-H-001g.jeff' if state == 'liquid' else 'n_6-C-012g.jeff'
    assert config.path('material', 'base_endf_file') == ROOT / 'tests' / 'fixtures' / neutron_file
    for key in ('leapr_file', 'ncmat_file'):
        if key in config.section('material'):
            assert config.path('material', key).is_file()
    assert config.section('ensemble')['n_samples'] == 3
    assert config.section('sampling')['max_energy_shift_mev'] == 1.0
    assert config.section('sampling')['automatic_partition_prominence_pct'] == 1.0
    assert config.section('ace')['save_xsdir'] is True


def _clone(tmp_path: Path, backend: str = 'njoy', replacements=()):
    text = (TEMPLATES / f'ace_{backend}_solid.toml').read_text()
    for a, b in replacements:
        assert a in text
        text = text.replace(a, b)
    path = tmp_path / 'config.toml'
    path.write_text(text)
    return path


@pytest.mark.parametrize('old,new,expected', [
    ('backend = "njoy"', 'backend = "bad"', 'backend'),
    ('n_samples = 3', 'n_samples = -1', 'n_samples'),
    ('n_samples = 3', 'n_samples = true', 'n_samples'),
    ('batch_size = 512', 'batch_size = 0', 'batch_size'),
    ('random_seed = 20261008', 'random_seed = -1', 'random_seed'),
    ('n_jobs = 1', 'n_jobs = 0', 'n_jobs'),
    ('temperature_k = 296.0', 'temperature_k = -273.6', 'temperature_k'),
    ('temperature_k = 296.0', 'temperature_k = nan', 'temperature_k'),
    ('temperature_k = 296.0', 'temperature_k = "293.6"', 'temperature_k'),
    ('max_energy_shift_mev = 1.0', 'max_energy_shift_mev = -1', 'max_energy_shift_mev'),
    ('grid_partition_method = "automatic"', 'grid_partition_method = "other"', 'grid_partition_method'),
    ('grid_partition_method = "automatic"', 'grid_partition_method = "manual"', 'manual_energy_ranges'),
    ('amplitude_method = "fast"', 'amplitude_method = "random"', 'amplitude_method'),
    ('spatial_method = "diffeomorphic"', 'spatial_method = "none"', 'spatial_method'),
    ('nbin = 16', 'nbin = 0', 'nbin'),
    ('save_xsdir = true', 'save_xsdir = "yes"', 'save_xsdir'),
    ('suffix = ".01"', 'suffix = ""', 'suffix'),
    ('include_reference = true', 'include_reference = false', 'include_reference'),
    ('leapr_file = "../../tests/fixtures/input_2.leapr"', '', 'leapr_file'),
    ('max_energy_ev = 5.0', 'max_energy_ev = inf', 'max_energy_ev'),
])
def test_reject_invalid_ace_config(tmp_path, old, new, expected):
    """Fail early with an actionable field name on incorrect settings."""
    path = _clone(tmp_path, replacements=[(old, new)])
    with pytest.raises(ApplicationConfigError, match=expected):
        load_ace_config(path)


@pytest.mark.parametrize('old,new,expected', [
    ('include_reference = true', 'include_reference = false', None),
    ('ncrystal_builtin = "C_sg194_pyrolytic_graphite.ncmat"',
     'ncmat_file = "../../tests/fixtures/custom.ncmat"', None),
    ('ncrystal_builtin = "C_sg194_pyrolytic_graphite.ncmat"',
     'ncmat_file = "example.ncmat"\nncrystal_builtin = "C_sg194_pyrolytic_graphite.ncmat"', 'exactly one'),
    ('ncrystal_builtin = "C_sg194_pyrolytic_graphite.ncmat"', '', 'exactly one'),
])
def test_ncrystal_source_and_reference_rules(tmp_path, old, new, expected):
    """NCrystal supports two mutually exclusive material-source options."""
    path = _clone(tmp_path, backend='ncrystal', replacements=[(old, new)])
    if expected:
        with pytest.raises(ApplicationConfigError, match=expected):
            load_ace_config(path)
    else:
        assert load_ace_config(path).backend == 'ncrystal'


def test_manual_ranges_are_normalized_only_at_domain_boundary(tmp_path):
    """Store simple TOML lists and pass tuples to TaraConfig later."""
    path = _clone(tmp_path, replacements=[
        ('grid_partition_method = "automatic"', 'grid_partition_method = "manual"'),
        ('manual_energy_ranges = []', 'manual_energy_ranges = [[0.0, 0.06], [0.06, 0.12]]')
    ])
    config = load_ace_config(path)
    assert config.section('sampling')['manual_energy_ranges'] == [[0.0, 0.06], [0.06, 0.12]]


def test_unknown_field_and_table_fail(tmp_path):
    """Typos in parameter names must never be silently ignored."""
    for old, new in [
        ('batch_size = 512', 'batch_size = 512\nbacth_size = 10'),
        ('[workflow]', '[not_workflow]'),
    ]:
        path = _clone(tmp_path, replacements=[(old, new)])
        with pytest.raises(ApplicationConfigError):
            load_ace_config(path)


def test_reject_invalid_ranges(tmp_path):
    """Reject invalid range shapes and overlapping invalid bounds."""
    for ranges in ('[[0.2, 0.1]]', '[[0.0]]', '[[0.1, true]]', '[[0.0, inf]]'):
        path = _clone(tmp_path, replacements=[
            ('grid_partition_method = "automatic"', 'grid_partition_method = "manual"'),
            ('manual_energy_ranges = []', f'manual_energy_ranges = {ranges}')
        ])
        with pytest.raises(ApplicationConfigError):
            load_ace_config(path)


def test_error_for_nonexistent_and_malformed_file(tmp_path):
    """Do not convert malformed syntax into an unrelated error."""
    with pytest.raises(FileNotFoundError):
        load_ace_config(tmp_path / 'missing.toml')
    path = tmp_path / 'bad.toml'
    path.write_text('[[invalid\n')
    with pytest.raises(ApplicationConfigError, match='Malformed TOML'):
        load_ace_config(path)


def test_sections_are_defensive_copies():
    """Callers cannot replace top-level stored settings by editing a section."""
    config = load_ace_config(TEMPLATES / 'ace_njoy_solid.toml')
    copy = config.section('ensemble')
    copy['n_samples'] = 999
    assert config.section('ensemble')['n_samples'] == 3


@pytest.mark.parametrize('backend', ('njoy', 'ncrystal'))
def test_automatic_partition_prominence_accepts_fractional_percentage(tmp_path, backend):
    """Every supported ACE backend exposes the same Core sampling setting."""
    config = load_ace_config(_clone(tmp_path, backend, replacements=[
        ('automatic_partition_prominence_pct = 1.0',
         'automatic_partition_prominence_pct = 0.2')
    ]))
    assert config.section('sampling')['automatic_partition_prominence_pct'] == 0.2


@pytest.mark.parametrize('value', ('0', '-0.5', '101', 'inf', 'nan', 'true', '"0.2"'))
def test_automatic_partition_prominence_rejects_invalid_values(tmp_path, value):
    """A malformed or physically invalid threshold fails at TOML loading."""
    path = _clone(tmp_path, replacements=[
        ('automatic_partition_prominence_pct = 1.0',
         f'automatic_partition_prominence_pct = {value}')
    ])
    with pytest.raises(ApplicationConfigError, match='automatic_partition_prominence_pct'):
        load_ace_config(path)


def test_old_toml_without_prominence_field_retains_one_percent_default(tmp_path):
    """Existing user configuration files remain valid without migration."""
    path = _clone(tmp_path, replacements=[
        ('automatic_partition_prominence_pct = 1.0  # Lower values detect more valleys; automatic only.', '')
    ])
    config = load_ace_config(path)
    assert config.section('sampling')['automatic_partition_prominence_pct'] == 1.0


# Diffusion config

BASE = '''
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
n_samples = 3
include_reference = true
random_seed = 7
batch_size = 2
[execution]
n_jobs = 1
njoy_executable = "njoy"
[sampling]
max_energy_shift_mev = 1.0
max_amplitude_shift_pct = 0.0
grid_partition_method = "automatic"
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


def _load(tmp_path, extra='', *, diffusion_only=False):
    """Write a minimal physical liquid experiment without external files."""
    contents = BASE
    if diffusion_only:
        contents = contents.replace('max_energy_shift_mev = 1.0', 'max_energy_shift_mev = 0.0')
    path = tmp_path / 'liquid.toml'
    path.write_text(contents + extra)
    return load_ace_config(path)


def test_legacy_config_without_section_is_accepted(tmp_path):
    """Optional diffusion settings do not require migration of existing TOML."""
    config = _load(tmp_path)
    assert 'diffusion_sampling' not in config.sections
    assert config.section('sampling')['max_energy_shift_mev'] == 1.0


def test_explicit_disabled_configuration_keeps_legacy_sampling(tmp_path):
    """Enabled=false does not activate material or diffusion sampling."""
    config = _load(tmp_path, '\n[diffusion_sampling]\nenabled = false\n')
    assert config.section('diffusion_sampling')['enabled'] is False


def test_diffusion_only_config_uses_no_vibrational_bounds(tmp_path):
    """A positive diffusion bound permits zero legacy vibrational shifts."""
    config = _load(tmp_path, '''
[diffusion_sampling]
enabled = true
mode = "diffusivity_only"
max_diffusivity_shift_pct = 5.0
''', diffusion_only=True)
    assert config.section('diffusion_sampling')['enabled']
    assert config.section('sampling')['max_energy_shift_mev'] == 0


@pytest.mark.parametrize('settings,match', [
    ('enabled = true\nmode = "weight_fixed_d"\nmax_weight_shift_pct = 5.0', 'compensation'),
    ('enabled = false\nmax_diffusivity_shift_pct = 5.0', 'enabled'),
    ('enabled = true\nmode = "diffusivity_only"\nmax_diffusivity_shift_pct = 105', 'max_diffusivity_shift_pct'),
    ('enabled = true\nmode = "something"\nmax_diffusivity_shift_pct = 5', 'mode'),
    ('enabled = true\nmode = "diffusivity_only"\nmax_diffusivity_shift_pct = 5\nextra = 1', 'extra'),
])
def test_invalid_diffusion_toml_fails_preflight(tmp_path, settings, match):
    """Invalid stochastic model requests are rejected before backend execution."""
    path = tmp_path / 'bad.toml'
    path.write_text(BASE + '\n[diffusion_sampling]\n' + settings + '\n')
    with pytest.raises(ApplicationConfigError, match=match):
        load_ace_config(path)


def test_disabled_diffusion_does_not_make_zero_vdos_valid(tmp_path):
    """Zero perturbations remain invalid without explicit diffusion activation."""
    with pytest.raises(ApplicationConfigError, match='At least one'):
        _load(tmp_path, '\n[diffusion_sampling]\nenabled = false\n', diffusion_only=True)


def test_diffusion_with_solid_is_rejected(tmp_path):
    """A solid material must not receive a liquid diffusion configuration."""
    path = tmp_path / 'solid.toml'
    path.write_text(BASE.replace('state = "liquid"', 'state = "solid"') + '''
[diffusion_sampling]
enabled = true
max_diffusivity_shift_pct = 5
''')
    with pytest.raises(ApplicationConfigError, match='NJOY liquid'):
        load_ace_config(path)


def test_joint_diffusion_vdos_shift_requires_frozen_origin(tmp_path):
    """Reject unanchored low-energy vibrational motion in a diffusion-enabled run."""
    path = tmp_path / 'bad_origin.toml'
    path.write_text(BASE.replace('freeze_origin = true', 'freeze_origin = false') + '''
[diffusion_sampling]
enabled = true
mode = "diffusivity_only"
max_diffusivity_shift_pct = 5
''')
    with pytest.raises(ApplicationConfigError, match='freeze_origin'):
        load_ace_config(path)


# Liquid config

EXAMPLE = ROOT / 'tests/fixtures/ace_njoy_water_v2.example.toml'


def _write(tmp_path, replacements=()):
    """Return an independently editable example configuration.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory.
    replacements : iterable of pairs
        Exact input replacements.

    Returns
    -------
    pathlib.Path
        Written TOML input.
    """
    text = EXAMPLE.read_text(encoding='utf8')
    for old, new in replacements:
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / 'water.toml'
    path.write_text(text)
    return path


def test_liquid_example_is_loadable() -> None:
    """Check that the sample configuration is accepted without external files."""
    config = load_ace_config(EXAMPLE)
    assert config.backend == 'njoy'
    assert config.section('material')['state'] == 'liquid'
    assert (config.section('ace')['mti'], config.section('ace')['mte']) == (222, 0)


def test_liquid_requires_explicit_reaction_identifier(tmp_path) -> None:
    """Never silently assign a water MT to another liquid material."""
    with pytest.raises(ApplicationConfigError, match='explicit'):
        load_ace_config(_write(tmp_path, [('mti = 222  # Verified for this educational H(H2O) reference, not arbitrary TSLs.', '')]))


def test_liquid_requires_unit_normalization(tmp_path) -> None:
    """The continuous spectrum must be normalized before its LEAPR weighting."""
    with pytest.raises(ApplicationConfigError, match='auto_normalize'):
        load_ace_config(_write(tmp_path, [('auto_normalize = true', 'auto_normalize = false')]))


def test_ncrystal_liquid_requires_explicit_reaction_identifier(tmp_path) -> None:
    """Require a material-specific MT when generating ACE for NCrystal liquids."""
    template = TEMPLATES / 'ace_ncrystal_liquid.toml'
    path = tmp_path / template.name
    path.write_text(template.read_text().replace(
        'mti = 222  # NJOY inelastic MT for this H(H2O) workflow.', ''
    ))
    with pytest.raises(ApplicationConfigError, match='explicit'):
        load_ace_config(path)

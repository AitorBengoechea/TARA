"""Application CLI routing and backward-compatibility tests."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from application import cli
from application import ace_workflow


ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / 'application' / 'configs'


def test_ace_requires_config_flag():
    """The new ACE route requires an explicit TOML file."""
    with pytest.raises(SystemExit) as result:
        cli.main(['ace'])
    assert result.value.code == 2


def test_ace_loads_config_and_calls_workflow(monkeypatch, capsys):
    """CLI delegates processing instead of implementing physics itself."""
    seen = []

    def fake_run(config):
        seen.append(config.backend)
        return {'sample_000': 'fake.ace'}

    monkeypatch.setattr(ace_workflow, 'run_ace_workflow', fake_run)
    cli.main(['ace', '--config', str(TEMPLATES / 'ace_njoy_solid.toml')])
    assert seen == ['njoy']
    assert '1 tables' in capsys.readouterr().out


def test_validate_only_does_not_run_sampling(monkeypatch, tmp_path, capsys):
    """Dry validation checks inputs and not Core/NJOY execution."""
    config_file = (TEMPLATES / 'ace_njoy_solid.toml').read_text()
    # Inputs relative to the copied config file must also be relative to it.
    config_file = config_file.replace(
        '../../tests/fixtures/n_6-C-012g.jeff', 'base.endf'
    ).replace('../../tests/fixtures/input_2.leapr', 'input.leapr')
    config_file = config_file.replace('njoy_executable = "njoy"',
                                      f'njoy_executable = "{sys.executable}"')
    (tmp_path / 'base.endf').write_text('test')
    (tmp_path / 'input.leapr').write_text('test')
    source = tmp_path / 'config.toml'
    source.write_text(config_file)

    def should_not_run(config):
        raise AssertionError('Actual NJOY sampling must not run in --validate-only mode')

    monkeypatch.setattr(ace_workflow, 'run_ace_workflow', should_not_run)
    cli.main(['ace', '--config', str(source), '--validate-only'])
    assert 'Valid njoy' in capsys.readouterr().out


@pytest.mark.parametrize('extra_args,expected_prominence', [([], 1.0),
    (['--automatic-partition-prominence-pct', '0.2'], 0.2)])
def test_legacy_arguments_remain_backward_compatible(
    monkeypatch, tmp_path, capsys, extra_args, expected_prominence
):
    """The old `input_file --shift-mev --shift-pct` syntax still works."""
    captured = {}

    class Config:
        n_samples = 1

        def __init__(self, **kwargs):
            self.n_samples = kwargs['n_samples']
            captured.update(kwargs)

    class Spectrum:
        def __init__(self, pdos):
            self.pdos = pdos

    class Ensemble:
        def __len__(self):
            return 1

    class Sampler:
        def __init__(self, config, spectra):
            pass

        def generate_ensemble(self):
            return Ensemble()

    mappings = {
        'core.config': {'TaraConfig': Config, 'TaraConfigError': type('TaraConfigError', (Exception,), {})},
        'core.data.input_pdos': {'PhononSpectra': Spectrum, 'PhononSpectraError': type('PhononSpectraError', (Exception,), {})},
        'core.sampler.concrete_samplers': {'TaraSampler': Sampler},
        'input_output.factory': {'load_file_object': lambda path: (object(), [1.0])},
        'input_output.exporter': {'export_ensemble': lambda *args: None},
    }
    for name, attrs in mappings.items():
        if name in sys.modules:
            monkeypatch.delitem(sys.modules, name)
        mock = types.ModuleType(name)
        mock.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mock)
    input_file = tmp_path / 'input.leapr'
    input_file.write_text('dummy')
    cli.main([str(input_file), '--shift-mev', '1.0', '--shift-pct', '5',
              '-n', '1', *extra_args])
    assert 'Generation Complete' in capsys.readouterr().out
    assert captured['automatic_partition_prominence_pct'] == expected_prominence

"""Isolated application orchestration tests (no real NJOY/NCrystal needed)."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import types

import pytest

from application.configuration import load_ace_config
from application import ace_workflow


ROOT = Path(__file__).resolve().parents[2]


def _module(monkeypatch, name, **symbols):
    mod = types.ModuleType(name)
    mod.__dict__.update(symbols)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _config(tmp_path, backend='njoy', include_reference=True, builtin=True, state='solid'):
    example = ROOT / 'application' / 'configs' / f'ace_{backend}_{state}.toml'
    text = example.read_text()
    text = text.replace('../../tests/fixtures/n_6-C-012g.jeff', 'base.endf')
    text = text.replace('../../tests/fixtures/input_2.leapr', 'input.leapr')
    text = text.replace(f'directory = "../../tara_output/ace_{backend}/cli_{state}"',
                        'directory = "output"')
    if state == 'liquid':
        text = text.replace('../../tests/fixtures/n_1-H-001g.jeff', 'base.endf')
    text = text.replace('njoy_executable = "njoy"', f'njoy_executable = "{sys.executable}"')
    if not include_reference:
        text = text.replace('include_reference = true', 'include_reference = false')
    if backend == 'ncrystal' and not builtin:
        text = text.replace('ncrystal_builtin = "C_sg194_pyrolytic_graphite.ncmat"',
                            'ncmat_file = "input.ncmat"')
        (tmp_path / 'input.ncmat').write_text('test ncmat')
    if backend == 'ncrystal' and state == 'liquid':
        text = text.replace('../../tests/fixtures/liquid_water_295K_illustrative.ncmat', 'input.ncmat')
        (tmp_path / 'input.ncmat').write_text('test liquid ncmat')
    (tmp_path / 'base.endf').write_text('test endf')
    (tmp_path / 'input.leapr').write_text('test leapr')
    path = tmp_path / 'ace.toml'
    path.write_text(text)
    return load_ace_config(path)


class Ensemble:
    reference = object()

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return object()


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """Stub each Processing boundary while preserving real ACE app orchestration."""
    calls = []

    class AceConfig:
        @classmethod
        def from_endf(cls, **kwargs):
            calls.append(('ace_config', kwargs))
            return cls()

    class Pipeline:
        def __init__(self, **kwargs):
            calls.append(('pipeline_init', kwargs))

        def run(self, **kwargs):
            calls.append(('pipeline_run', kwargs))
            output = Path(kwargs['output_dir'])
            output.mkdir(parents=True, exist_ok=True)
            if kwargs['save_xsdir']:
                (output / 'xsdir').mkdir(exist_ok=True)
            result = {}
            for sid in kwargs['sab_tapes']:
                outfile = output / f'{sid}.ace'
                outfile.write_text(f'{sid} actual ACE data')
                result[sid] = str(outfile)
                if kwargs['save_xsdir']:
                    (output / 'xsdir' / f'{sid}.xsdir').write_text('index entry')
            return result

    _module(monkeypatch, 'processing.core.config', AceGenerationConfig=AceConfig)
    _module(monkeypatch, 'processing.pipelines.ace_pipeline', UniversalAcePipeline=Pipeline)
    monkeypatch.setattr(ace_workflow, '_prepare_ensemble', lambda config, pdos: Ensemble())
    return calls


def test_njoy_end_to_end_handoff_and_provenance(monkeypatch, tmp_path, fake_pipeline):
    """CLI route passes temperature, IDs, and ACER controls to the right APIs."""
    config = _config(tmp_path)
    calls = fake_pipeline

    class Orchestrator:
        def __init__(self, **kwargs):
            calls.append(('njoy_init', kwargs))

        def generate_ensemble_sab_tapes(self, **kwargs):
            calls.append(('njoy_tapes', kwargs))
            return {f'sample_{i:03d}': f'/tmp/{i}.endf' for i in range(4)}

    _module(monkeypatch, 'input_output.factory', load_file_object=lambda filepath: (object(), ['pdos']))
    _module(monkeypatch, 'processing.orchestrators.njoy_batch', NjoyBatchOrchestrator=Orchestrator)
    result = ace_workflow.run_ace_workflow(config)
    assert set(result) == {f'sample_{i:03d}' for i in range(4)}
    assert next(x for x in calls if x[0] == 'njoy_tapes')[1]['temperature'] == 296.0
    options = next(x for x in calls if x[0] == 'ace_config')[1]
    assert options['temperature'] == 296.0
    assert options['nbin'] == 16
    assert next(x for x in calls if x[0] == 'pipeline_run')[1]['save_xsdir'] is True
    result_dir = tmp_path / 'output'
    assert (result_dir / 'run_config.toml').read_bytes() == config.file_path.read_bytes()
    metadata = json.loads((result_dir / 'run_manifest.json').read_text())
    assert metadata['sample_ids'] == sorted(result)
    assert metadata['random_seed'] == 20261008
    assert metadata['settings']['sampling']['grid_partition_method'] == 'automatic'
    assert metadata['settings']['sampling']['automatic_partition_prominence_pct'] == 1.0
    assert metadata['resolved_input_paths']['base_endf_file'] == str(tmp_path / 'base.endf')


def test_prepare_ensemble_passes_configured_prominence_to_core(monkeypatch, tmp_path):
    """ACE sampling config reaches TaraSampler without backend-specific branching."""
    import numpy as np
    import pandas as pd
    from core.sampler.concrete_samplers import TaraSampler

    config_file = _config(tmp_path).file_path
    contents = config_file.read_text().replace(
        'automatic_partition_prominence_pct = 1.0',
        'automatic_partition_prominence_pct = 0.2',
    )
    config_file.write_text(contents)
    config = load_ace_config(config_file)
    seen = {}

    def fake_generate(self, batch_size):
        seen['prominence'] = self.config.automatic_partition_prominence_pct
        seen['batch_size'] = batch_size
        return Ensemble()

    monkeypatch.setattr(TaraSampler, 'generate_ensemble', fake_generate)
    energy = np.linspace(0.0, 0.1, 101)
    density = np.sin(np.pi * energy / 0.1) ** 2
    ace_workflow._prepare_ensemble(config, pd.Series(density, index=energy))

    assert seen == {'prominence': 0.2, 'batch_size': 512}


@pytest.mark.parametrize('builtin,state', ((True, 'solid'), (False, 'solid'), (False, 'liquid')))
def test_ncrystal_handoff(monkeypatch, tmp_path, fake_pipeline, builtin, state):
    """NCrystal's built-in and user-supplied sources use the same batch API."""
    config = _config(tmp_path, backend='ncrystal', builtin=builtin, state=state)
    calls = fake_pipeline

    def pdos_for_atom(atom_label):
        """Record which species supplies the sampling spectrum."""
        calls.append(('pdos_for_atom', atom_label))
        return ['pdos']

    class NCrystalFile:
        @classmethod
        def from_file(cls, filename):
            calls.append(('parse_ncmat', str(filename)))
            return types.SimpleNamespace(pdos_for_atom=pdos_for_atom)

    class Orchestrator:
        def __init__(self, **kwargs):
            calls.append(('nc_init', kwargs))

        def generate_ensemble_sab_tapes(self, **kwargs):
            calls.append(('nc_tapes', kwargs))
            assert Path(kwargs['base_ncmat_file']).is_file()
            return {f'sample_{i:03d}': f'/tmp/{i}.endf' for i in range(4)}

    _module(monkeypatch, 'input_output.ncrystal.core', NCrystalFile=NCrystalFile)
    _module(monkeypatch, 'processing.orchestrators.ncrystal_batch', NcrystalBatchOrchestrator=Orchestrator)
    if builtin:
        _module(monkeypatch, 'NCrystal', createTextData=lambda name: types.SimpleNamespace(rawData='NCMAT content'))
    result = ace_workflow.run_ace_workflow(config)
    assert len(result) == 4
    atom_label = 'H' if state == 'liquid' else 'C'
    assert ('pdos_for_atom', atom_label) in calls
    tape_options = next(x for x in calls if x[0] == 'nc_tapes')[1]
    assert tape_options['atom_label'] == atom_label
    assert tape_options['temperature'] == (295.0 if state == 'liquid' else 293.6)
    assert tape_options['include_reference'] is True
    if state == 'liquid':
        assert tape_options['elastic_mode'] == 'none'
        assert 'liquid_options' not in tape_options
        ace_options = next(x for x in calls if x[0] == 'ace_config')[1]
        assert ace_options['allow_additional_scatterers'] is True
        assert (ace_options['mti'], ace_options['mte']) == (222, 0)


def test_ncrystal_without_reference(monkeypatch, tmp_path, fake_pipeline):
    """Excluding sample_000 is supported for the NCrystal route only."""
    config = _config(tmp_path, backend='ncrystal', include_reference=False, builtin=False)
    monkeypatch.setitem(config.sections['execution'], 'n_jobs', 2)
    calls = fake_pipeline

    def pdos_for_atom(atom_label):
        """Record which species supplies the sampling spectrum."""
        calls.append(('pdos_for_atom', atom_label))
        return ['pdos']

    class NCrystalFile:
        @classmethod
        def from_file(cls, filename):
            return types.SimpleNamespace(pdos_for_atom=pdos_for_atom)

    class Orchestrator:
        def __init__(self, *, n_jobs):
            calls.append(('nc_init', n_jobs))

        def generate_ensemble_sab_tapes(self, **kwargs):
            calls.append(('nc_tapes', kwargs))
            return {f'sample_{i:03d}': f'/tmp/{i}.endf' for i in range(1, 4)}

    _module(monkeypatch, 'input_output.ncrystal.core', NCrystalFile=NCrystalFile)
    _module(monkeypatch, 'processing.orchestrators.ncrystal_batch', NcrystalBatchOrchestrator=Orchestrator)
    output = ace_workflow.run_ace_workflow(config)
    assert ('pdos_for_atom', 'C') in calls
    assert set(output) == {'sample_001', 'sample_002', 'sample_003'}
    assert ('nc_init', 2) in calls
    assert next(x for x in calls if x[0] == 'nc_tapes')[1]['include_reference'] is False


def test_failed_rerun_invalidates_previous_manifest(monkeypatch, tmp_path, fake_pipeline):
    """Publish a success manifest only for a completed attempt."""
    config = _config(tmp_path)
    sample_ids = [f'sample_{i:03d}' for i in range(4)]
    manifest = config.path('output', 'directory') / 'run_manifest.json'

    def generate_tapes(config, endf_dir, njoy):
        """Create a complete mock ENDF batch."""
        endf_dir.mkdir(parents=True, exist_ok=True)
        tapes = {}
        for sample_id in sample_ids:
            tape = endf_dir / f'{sample_id}.endf'
            tape.write_text('complete ENDF')
            tapes[sample_id] = str(tape)
        return tapes, Ensemble()

    def fail_after_overwrite(config, endf_dir, njoy):
        """Simulate a processing failure after replacing one artifact."""
        (endf_dir / 'sample_000.endf').write_text('partial ENDF')
        raise RuntimeError('ENDF generation failed')

    monkeypatch.setattr(ace_workflow, '_process_njoy', generate_tapes)
    ace_workflow.run_ace_workflow(config)
    assert json.loads(manifest.read_text())['sample_ids'] == sample_ids

    with monkeypatch.context() as failing_run:
        failing_run.setattr(ace_workflow, '_process_njoy', fail_after_overwrite)
        with pytest.raises(RuntimeError, match='ENDF generation failed'):
            ace_workflow.run_ace_workflow(config)
    assert not manifest.exists()

    ace_workflow.run_ace_workflow(config)
    assert json.loads(manifest.read_text())['sample_ids'] == sample_ids


def test_reject_partial_thermr_batch_without_manifest(monkeypatch, tmp_path, fake_pipeline):
    """Never treat a partial reference/stochastic batch as a successful run."""
    config = _config(tmp_path)

    class Orchestrator:
        def __init__(self, **kwargs):
            pass

        def generate_ensemble_sab_tapes(self, **kwargs):
            return {'sample_000': '/tmp/ref.endf'}

    _module(monkeypatch, 'input_output.factory', load_file_object=lambda filepath: (object(), ['pdos']))
    _module(monkeypatch, 'processing.orchestrators.njoy_batch', NjoyBatchOrchestrator=Orchestrator)
    with pytest.raises(RuntimeError, match='Incomplete ENDF batch'):
        ace_workflow.run_ace_workflow(config)
    assert not (tmp_path / 'output' / 'run_config.toml').exists()


def test_missing_endf_input_checked_before_work(monkeypatch, tmp_path, fake_pipeline):
    """Inputs are preflighted before any library physics calls."""
    config = _config(tmp_path)
    (tmp_path / 'base.endf').unlink()
    with pytest.raises(FileNotFoundError, match='neutron ENDF'):
        ace_workflow.run_ace_workflow(config)


def test_xsdir_missing_prevents_run_success(monkeypatch, tmp_path, fake_pipeline):
    """Require actual XSDIR entries when explicitly requested."""
    config = _config(tmp_path)

    class Orchestrator:
        def __init__(self, **kwargs):
            pass

        def generate_ensemble_sab_tapes(self, **kwargs):
            return {f'sample_{i:03d}': f'/tmp/{i}.endf' for i in range(4)}

    _module(monkeypatch, 'input_output.factory', load_file_object=lambda filepath: (object(), ['pdos']))
    _module(monkeypatch, 'processing.orchestrators.njoy_batch', NjoyBatchOrchestrator=Orchestrator)
    # Replace the pipeline stub with a deliberately incomplete XSDIR writer.
    mod = sys.modules['processing.pipelines.ace_pipeline']
    old = mod.UniversalAcePipeline.run

    def missing_xsdir(self, **kwargs):
        result = old(self, **kwargs)
        for path in (tmp_path / 'output' / 'ace' / 'xsdir').glob('*'):
            path.unlink()
        return result

    monkeypatch.setattr(mod.UniversalAcePipeline, 'run', missing_xsdir)
    with pytest.raises(RuntimeError, match='XSDIR entry missing'):
        ace_workflow.run_ace_workflow(config)
    assert not (tmp_path / 'output' / 'run_manifest.json').exists()


def test_prepare_ensemble_maps_all_parameters_into_core(monkeypatch, tmp_path):
    """The parser does not implement mathematical Core validation itself."""
    config = _config(tmp_path)
    captured = {}

    class PhononSpectra:
        def __init__(self, **kwargs):
            captured['spectrum'] = kwargs

    class TaraConfig:
        def __init__(self, **kwargs):
            captured['config'] = kwargs

    class TaraSampler:
        def __init__(self, **kwargs):
            captured['sampler'] = kwargs

        def generate_ensemble(self, **kwargs):
            captured['generate'] = kwargs
            return Ensemble()

    _module(monkeypatch, 'core.config', TaraConfig=TaraConfig)
    _module(monkeypatch, 'core.data.input_pdos', PhononSpectra=PhononSpectra)
    _module(monkeypatch, 'core.sampler.concrete_samplers', TaraSampler=TaraSampler)
    assert isinstance(ace_workflow._prepare_ensemble(config, ['sample']), Ensemble)
    assert captured['config']['max_energy_shift_mev'] == 1
    assert captured['config']['fast_amplitude'] is True
    assert captured['sampler']['amplitude_method'] == 'fast'
    assert captured['generate']['batch_size'] == 512
    assert captured['spectrum']['auto_normalize'] is True

"""Liquid workflows regression tests."""


import json
from pathlib import Path
import sys
import types
from application import ace_workflow
from application.configuration import load_ace_config
import inspect
from application.ace_workflow import run_ncrystal_native_ensemble
from unittest.mock import patch
from input_output.ncrystal.liquids import LiquidAtomSettings
from input_output.ncrystal.liquids import add_custom_liquids


# Liquid ace handoff

EXAMPLE = Path(__file__).resolve().parents[2] / 'tests/fixtures/ace_njoy_water_v2.example.toml'


class _Ensemble:
    """Provide one reference and two stochastic samples to the mock backend."""

    reference = object()

    def __len__(self):
        """Return the sample count excluding the separate reference."""
        return 2


def test_water_application_preserves_mti_and_sample_identity(monkeypatch, tmp_path):
    """Verify the full application handoff without claiming a real ACE calculation.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replace external-only processing and physical file loading.
    tmp_path : pathlib.Path
        Temporary input and output location.
    """
    base = tmp_path / 'neutron_H1.endf'
    base.write_text('mock base')
    source = tmp_path / 'water_296.leapr'
    source.write_text('mock LEAPR')
    text = EXAMPLE.read_text()
    text = text.replace('/replace/with/neutron_H1.endf', str(base))
    text = text.replace('/replace/with/H_H2O_296K.leapr', str(source))
    text = text.replace('../../tara_output/ace_njoy_water_v2', 'output')
    file = tmp_path / 'water.toml'
    file.write_text(text)
    cfg = load_ace_config(file)

    fake_parser = types.ModuleType('endf_parserpy')

    class Parser:
        """Return model-compatible MF1 and MF7 metadata for all samples."""

        def parsefile(self, path):
            """Return only the ENDF structures read by AceGenerationConfig."""
            if str(path) == str(base):
                return {1: {451: {'ZA': 1001, 'MAT': 125}}}
            return {7: {4: {'MAT': 99, 'B': {6: 2}, 'NS': 1}}}

    fake_parser.EndfParserPy = Parser
    monkeypatch.setitem(sys.modules, 'endf_parserpy', fake_parser)
    monkeypatch.setattr(ace_workflow, '_executable', lambda command: Path('/bin/true'))
    monkeypatch.setattr(ace_workflow, '_prepare_ensemble', lambda config, pdos: _Ensemble())

    from input_output import factory
    monkeypatch.setattr(factory, 'load_file_object', lambda filepath: (object(), 'PDOS'))
    # Liquid application sampling must use the selected LEAPR source block.
    from input_output.leapr import material_adapter
    from types import SimpleNamespace
    monkeypatch.setattr(material_adapter, 'material_model_from_leapr',
                        lambda template, state, temperature: SimpleNamespace(
                            dynamics=SimpleNamespace(continuous=SimpleNamespace(pdos='PDOS'))
                        ))

    seen = {}
    fake_batch = types.ModuleType('processing.orchestrators.njoy_batch')

    class Batch:
        """Emulate the NJOY tape-generation boundary."""

        def __init__(self, **kwargs):
            """Record construction without invoking the external executable."""
            seen['njoy'] = kwargs

        def generate_ensemble_sab_tapes(self, **kwargs):
            """Emit clearly mock TSL tapes while preserving reference identity."""
            seen['leapr'] = kwargs
            destination = Path(kwargs['output_dir'])
            destination.mkdir(parents=True, exist_ok=True)
            paths = {}
            for index in range(3):
                sid = f'sample_{index:03d}'
                target = destination / f'{sid}_sab.endf'
                target.write_text('mock MF7')
                paths[sid] = str(target)
            return paths

    fake_batch.NjoyBatchOrchestrator = Batch
    monkeypatch.setitem(sys.modules, 'processing.orchestrators.njoy_batch', fake_batch)

    fake_pipeline = types.ModuleType('processing.pipelines.ace_pipeline')

    class Pipeline:
        """Emulate the universal Processing ACE boundary."""

        def __init__(self, **kwargs):
            """Record constructor options."""
            seen['pipeline_init'] = kwargs

        def run(self, **kwargs):
            """Create fake ACE/XSDIR outputs for Application result validation."""
            seen['ace'] = kwargs
            destination = Path(kwargs['output_dir'])
            (destination / 'xsdir').mkdir(parents=True, exist_ok=True)
            paths = {}
            for sid in kwargs['sab_tapes']:
                ace = destination / f'{sid}.ace'
                ace.write_text('mock ACE: never use for transport')
                (destination / 'xsdir' / f'{sid}.xsdir').write_text('mock xsdir')
                paths[sid] = str(ace)
            return paths

    fake_pipeline.UniversalAcePipeline = Pipeline
    monkeypatch.setitem(sys.modules, 'processing.pipelines.ace_pipeline', fake_pipeline)

    outputs = ace_workflow.run_ace_workflow(cfg)
    assert set(outputs) == {'sample_000', 'sample_001', 'sample_002'}
    assert seen['leapr']['material_state'] == 'liquid'
    assert seen['leapr']['temperature'] == 296.0
    assert seen['ace']['config'].mti == 222
    assert seen['ace']['config'].mte == 0
    assert seen['ace']['config'].natom == 2
    assert seen['ace']['config'].za1 == 1001
    manifest = json.loads((tmp_path / 'output/run_manifest.json').read_text())
    assert manifest['sample_ids'] == sorted(outputs)
    assert manifest['settings']['ace']['mti'] == 222


# Ncrystal liquids app

def test_generic_application_signature_has_no_water_arguments():
    params = inspect.signature(run_ncrystal_native_ensemble).parameters
    assert 'ncmat_template' in params and 'atom_label' in params
    assert not {'plugin_c','oxygen_dynamics','leapr_file'} & set(params)


# Ncrystal unified app

def test_application_native_delegation(tmp_path):
    from application.ace_workflow import run_ncrystal_native_ensemble
    src=tmp_path/'argon.ncmat'
    src.write_text(add_custom_liquids('NCMAT v7\n@DYNINFO\n  element Ar\n'
                                     '  fraction 1\n  type vdos\n'
                                     '  vdos_egrid 1e-5 .03\n  vdos\n    1 2 1\n',
                                     [LiquidAtomSettings('Ar','incoherent',1,.2,.8)]))
    with patch('processing.orchestrators.ncrystal_batch.NcrystalBatchOrchestrator') as ctor:
        ctor.return_value.generate_native_ensemble.return_value = {'sample_000': {}}
        out=run_ncrystal_native_ensemble(object(),ncmat_template=src,
            output_dir=tmp_path/'out',temperature_k=295,atom_label='Ar')
    assert 'sample_000' in out
    assert ctor.return_value.generate_native_ensemble.call_args.args[4] == 'Ar'

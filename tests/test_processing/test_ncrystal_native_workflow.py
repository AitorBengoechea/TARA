"""NCrystal native workflow contracts and liquid batch handoff."""


# Ncrystal unified native

from pathlib import Path
import importlib.util
import sys
import types

import numpy as np
import pytest

# NCrystal is a runtime plugin dependency, not required for mocked unit tests.
if importlib.util.find_spec('NCrystal') is None:
    module = types.ModuleType('NCrystal')
    module.__path__ = []
    module.ncmat2endf = types.SimpleNamespace(ncmat2endf=lambda **kw: None)
    internal = types.ModuleType('NCrystal._ncmat2endf_impl')
    internal.NuclearData = type('NuclearData', (), {})
    sys.modules.setdefault('NCrystal', module)
    sys.modules.setdefault('NCrystal._ncmat2endf_impl', internal)

if importlib.util.find_spec('parallel_execution') is None:
    parallel = types.ModuleType('parallel_execution')
    parallel.run_parallel = lambda *args, **kwargs: None
    sys.modules.setdefault('parallel_execution', parallel)

if importlib.util.find_spec('endf_parserpy') is None:
    endf = types.ModuleType('endf_parserpy')
    endf.EndfParserPy = type('EndfParserPy', (), {})
    sys.modules.setdefault('endf_parserpy', endf)

from processing.backends.ncrystal_processor import (
    NcrystalProcessor, UnsupportedLiquidAceExport,
    verify_liquids_plugin, sampled_kinematics,
)
from processing.orchestrators.ncrystal_batch import NcrystalBatchOrchestrator
from input_output.ncrystal.liquids import LiquidAtomSettings, add_custom_liquids

BASE = ('NCMAT v7\n@DYNINFO\n  element Xe\n  fraction 1\n'
        '  type vdos\n  vdos_egrid 1e-5 .03\n  vdos\n    1 2 1\n')


class Scatter:
    def crossSectionIsotropic(self, e):
        return 10. / (1 + e)

    def sampleScatterIsotropic(self, e):
        return (e * .75, .2)


class Runtime:
    def createScatter(self, cfg):
        assert 'temp=295K' in cfg
        return Scatter()


def create_template(tmp_path):
    path = tmp_path / 'other_liquid.ncmat'
    path.write_text(add_custom_liquids(BASE, [
        LiquidAtomSettings('Xe', 'incoherent', 3, .2, .8)]))
    return path


def test_native_methods_are_on_existing_processor(tmp_path):
    processor = NcrystalProcessor()
    state = processor.create_native_scattering_state(
        create_template(tmp_path), 295, verify_plugin=False, ncrystal=Runtime())
    np.testing.assert_allclose(processor.native_cross_sections(state, [.1,.2]),
                               [10/1.1, 10/1.2])
    eo, mu = processor.native_final_states(state, .025, 5)
    np.testing.assert_allclose(eo, .01875)
    np.testing.assert_allclose(mu, .2)
    with pytest.raises(UnsupportedLiquidAceExport):
        processor.export_native_liquid_ace('fake.ace')
    assert NcrystalProcessor.generate_sab_tape  # Solid export preserved.


def test_unsupported_solid_file_not_misrepresented_as_liquid(tmp_path):
    src = tmp_path / 'solid.ncmat'
    src.write_text(BASE)
    with pytest.raises(ValueError, match='CUSTOM_LIQUIDS'):
        NcrystalProcessor().create_native_scattering_state(src, 295, verify_plugin=False,
                                                           ncrystal=Runtime())


def test_plugin_diagnostic(monkeypatch):
    from processing.backends import ncrystal_processor as mod
    monkeypatch.setattr(mod.subprocess, 'run', lambda *a, **kw: types.SimpleNamespace(
        stdout='==> Liquids (dynamic)', stderr=''))
    assert 'Liquids' in verify_liquids_plugin()
    monkeypatch.setattr(mod.subprocess, 'run', lambda *a, **kw: types.SimpleNamespace(
        stdout='==> Solids (dynamic)', stderr=''))
    with pytest.raises(RuntimeError, match='not active'):
        verify_liquids_plugin()


class Spectrum:
    energy_grid = np.arange(1, 8) * (.03 / 7)
    density_values = np.ones(7) / (energy_grid[-1] - energy_grid[0])


class Ensemble:
    reference = Spectrum()

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return Spectrum()


class FakeProcessor:
    def create_native_scattering_state(self, filepath, temperature, verify_plugin=True):
        assert Path(filepath).is_file()
        assert temperature == 295
        return object()

    def native_cross_sections(self, state, energies):
        return np.full(len(energies), 5.)

    def native_final_states(self, state, energy, n):
        return np.full(n, .8*energy), np.full(n, .5)


def test_unified_batch_sample_identity(tmp_path, monkeypatch):
    from processing.orchestrators import ncrystal_batch as mod
    monkeypatch.setattr(mod, 'NcrystalProcessor', FakeProcessor)
    results = NcrystalBatchOrchestrator(n_jobs=1).generate_native_ensemble(
        Ensemble(), tmp_path/'results', create_template(tmp_path), 295,
        'Xe', [.01, .02], sample_energy_ev=.02, n_events=4, verify_plugin=False)
    assert list(results) == ['sample_000', 'sample_001', 'sample_002']
    for paths in results.values():
        assert set(paths) == {'ncmat', 'cross_sections', 'events'}
        assert all(Path(file).stat().st_size > 0 for file in paths.values())
        assert 'Xe incoherent' in Path(paths['ncmat']).read_text()
    assert not list((tmp_path/'results').glob('*.ace'))


def test_validation_before_writing(tmp_path):
    with pytest.raises(ValueError, match='principal'):
        NcrystalBatchOrchestrator().generate_native_ensemble(
            Ensemble(), tmp_path/'failed', create_template(tmp_path), 295,
            'Xe', [.01, .02], physical_model=types.SimpleNamespace(
                principal=types.SimpleNamespace(label='H'), temperature_k=295))
    assert not (tmp_path/'failed').exists()


def test_native_kinematics_are_diagnostics_only():
    x = sampled_kinematics(.025, [.02, .03], [1., -1.], temperature_k=295)
    np.testing.assert_allclose(x['energy_transfer_ev'], [.005, -.005])
    assert x['q_inv_angstrom'].min() >= 0


def test_legacy_orchestrator_signature_is_retained():
    import inspect
    names = inspect.signature(NcrystalBatchOrchestrator.generate_ensemble_sab_tapes).parameters
    assert {'ensemble','output_dir','base_ncmat_file','temperature','atom_label'} <= set(names)


def test_liquid_batch_builds_adapter_and_infers_natom(tmp_path, monkeypatch):
    """Default liquid dispatch builds once and cleans up after the workers."""
    source = create_template(tmp_path)
    builds = []

    def build(directory):
        adapter = Path(directory) / "libsabtap.so"
        assert adapter.parent.is_dir()
        adapter.touch()
        builds.append(adapter)
        return adapter

    def dispatch(worker, calls, *, n_jobs):
        results = []
        for _, kwargs in calls:
            assert kwargs["natom"] == 1
            assert Path(kwargs["capture_library"]).is_file()
            results.append((kwargs["sample_id"], kwargs["output_path"], None))
        return results

    monkeypatch.setattr(
        "processing.backends.liquid_capture.build.build_capture_library", build
    )
    monkeypatch.setattr("processing.orchestrators.ncrystal_batch.run_parallel", dispatch)
    paths = NcrystalBatchOrchestrator().generate_ensemble_sab_tapes(
        ensemble=Ensemble(), output_dir=str(tmp_path / "out"),
        base_ncmat_file=str(source), temperature=295, atom_label="Xe",
    )
    assert "sample_000" in paths
    assert len(builds) == 1 and not builds[0].parent.exists()


# Ncrystal liquid native contract

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from input_output.ncrystal.liquids import LiquidAtomSettings, add_custom_liquids


def _write_template(tmp_path):
    base = ("NCMAT v7\n@DENSITY\n  1 g_per_cm3\n@DYNINFO\n"
            "  element Ar\n  fraction 1\n  type vdos\n"
            "  vdos_egrid 0.001 0.004\n  vdos_density 1 2 2 1\n")
    material = add_custom_liquids(
        base, [LiquidAtomSettings('Ar', 'incoherent', 1., .2, .8)])
    path = tmp_path / 'input.ncmat'
    path.write_text(material)
    return path


def _ensemble():
    grid = np.array([.001, .002, .003, .004])
    y0 = np.array([100., 200., 200., 100.])
    y0 /= np.trapezoid(y0, grid)
    y1 = np.array([130., 190., 190., 130.])
    y1 /= np.trapezoid(y1, grid)
    def spectrum(y):
        return SimpleNamespace(energy_grid=grid.copy(), density_values=y.copy())
    class FakeEnsemble:
        def __init__(self):
            self.reference = spectrum(y0)
            self._samples = (spectrum(y1),)
        def __len__(self): return len(self._samples)
        def __getitem__(self, i): return self._samples[i]
    return FakeEnsemble()


def test_native_ensemble_keeps_reference_sample_and_metadata(tmp_path, monkeypatch):
    """NJOY/NCrystal workflow parity: reference=000; perturbation=001."""
    from processing.orchestrators.ncrystal_batch import NcrystalBatchOrchestrator
    import processing.orchestrators.ncrystal_batch as batch
    seen = []

    class FakeProcessor:
        def create_native_scattering_state(self, path, temp, verify_plugin):
            content = Path(path).read_text()
            assert '@CUSTOM_LIQUIDS' in content
            assert 'vdos_density ' in content
            assert not any(line.strip() == 'vdos' for line in content.splitlines())
            seen.append(Path(path).name)
            return object()
        def native_cross_sections(self, state, energies):
            return np.full_like(energies, 3.5)
        def native_final_states(self, state, energy, n_events):
            return np.full(n_events, energy), np.zeros(n_events)

    monkeypatch.setattr(batch, 'NcrystalProcessor', FakeProcessor)
    out = NcrystalBatchOrchestrator().generate_native_ensemble(
        _ensemble(), str(tmp_path/'output'), str(_write_template(tmp_path)),
        295., 'Ar', np.array([.01,.025,.1]),
        sample_energy_ev=.025, n_events=5, verify_plugin=False,
    )
    assert list(out) == ['sample_000','sample_001']
    assert seen == ['sample_000.ncmat','sample_001.ncmat']
    for sample in out:
        paths = out[sample]
        assert Path(paths['ncmat']).is_file()
        assert Path(paths['cross_sections']).is_file()
        assert Path(paths['events']).is_file()
        with np.load(paths['events']) as events:
            assert events['eout_ev'].shape == (5,)
    manifest = json.loads((tmp_path/'output'/'manifest.json').read_text())
    assert manifest['sample_ids'] == ['sample_000', 'sample_001']
    assert manifest['ace_export'] == 'unsupported'
    assert manifest['material_state'] == 'liquid'
    assert manifest['output'] == 'native'


def test_native_ensemble_rejects_unaligned_grid_before_processor(tmp_path, monkeypatch):
    """No native invocation should occur for a malformed energy grid."""
    from processing.orchestrators.ncrystal_batch import NcrystalBatchOrchestrator
    import processing.orchestrators.ncrystal_batch as batch
    calls = []
    class FakeProcessor:
        def create_native_scattering_state(self, *args, **kwargs):
            calls.append('native-called')
            raise AssertionError('Malformed grid reached compiled NCrystal!')
    monkeypatch.setattr(batch, 'NcrystalProcessor', FakeProcessor)
    ens = _ensemble()
    # The original V5 regression: E_min/ΔE = 0.01, not an integer.
    ens.reference.energy_grid[:] = np.array([1e-5, .00101, .00201, .00301])
    ens[0].energy_grid[:] = ens.reference.energy_grid
    with pytest.raises(ValueError, match=r'(?i)(grid|spacing|align|regular|energy)'):
        NcrystalBatchOrchestrator().generate_native_ensemble(
            ens, str(tmp_path/'output'), str(_write_template(tmp_path)),
            295., 'Ar', np.array([.01,.025,.1]), verify_plugin=False,
        )
    assert calls == []

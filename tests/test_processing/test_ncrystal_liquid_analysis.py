"""Isolated extraction, cleanup, component selection and batch reproducibility."""
from concurrent.futures import ProcessPoolExecutor
import gc
import multiprocessing
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('NCrystal')
from analyzer.parsers.ncrystal_parser import NCrystalParser
from analyzer.runners.ncrystal_runner import NCrystalRunner
from core.data.input_pdos import PhononSpectra
from input_output.ncrystal.core import NCrystalFile
from processing.backends.ncrystal_liquid_analysis import NCrystalLiquidAnalysisProcessor

SOURCE = Path(__file__).parents[1] / 'fixtures' / 'liquid_water_295K_illustrative.ncmat'


def spectrum(index=0):
    """Use distinct positive shape perturbations on the actual water fixture."""
    rho = NCrystalFile.from_file(str(SOURCE)).pdos_for_atom('H')
    rho = rho * (1 + .02 * index * rho.index.to_numpy() / rho.index.max())
    return PhononSpectra(rho)


def test_runner_builds_and_releases_temporary_adapter(tmp_path, monkeypatch):
    """Check automatic build ownership without needing a compiler."""
    build_paths = []

    def build_capture_library(directory):
        path = Path(directory) / "libsabtap.so"
        path.touch()
        build_paths.append(path)
        return path

    monkeypatch.setattr(
        'processing.backends.liquid_capture.build.build_capture_library',
        build_capture_library,
    )
    with patch.object(NCrystalLiquidAnalysisProcessor, '_verify_adapter', return_value={}):
        runner = NCrystalRunner(str(SOURCE), 295., temp_dir=str(tmp_path))
        assert not build_paths
        assert runner.prepare_analysis({}, {})["capture_kernel"] is False
        assert not build_paths
        runner.prepare_analysis({}, {"calc_sab": True})

    assert build_paths == [runner.processor.capture_library]
    assert runner.processor.capture_library.is_file()
    assert runner.processor.__getstate__()['_capture_build_dir'] is None

    del runner
    gc.collect()
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize('failure', ['timeout', 'exit'])
def test_failed_child_is_fatal_and_sandbox_is_cleaned(tmp_path, failure):
    with patch.object(NCrystalLiquidAnalysisProcessor, '_verify_adapter', return_value={}):
        processor = NCrystalLiquidAnalysisProcessor(SOURCE, 295., capture_library='unused', temp_dir=tmp_path)
    effect = subprocess.TimeoutExpired('child', 1) if failure == 'timeout' else None
    with patch('processing.backends.ncrystal_liquid_analysis.subprocess.run', side_effect=effect) as run:
        if failure == 'exit':
            run.return_value.returncode = 1
        with pytest.raises(RuntimeError, match='sample 7'):
            processor.create_analysis_state(7, spectrum(), atom_label='H')
    assert not list(tmp_path.iterdir())


def _live_job(args):
    """Picklable job: exercise native capture and every Analyzer output family."""
    library, scratch, index, active = args
    runner = NCrystalRunner(str(SOURCE), 295., temp_dir=scratch,
                            liquid_capture_library=library, vdoslux=0, active_species=active)
    state = runner.execute(index, spectrum(index), atom_label='H', energy_grid=np.geomspace(.0001, .1, 20))
    result = NCrystalParser().parse(index, state, calc_sab=True, calc_gn=True,
                                    calc_dsf=True, calc_scattering_kernel=True,
                                    calc_angular_distribution=True, calc_cosine_distribution=True,
                                    calc_energy_distribution=True, calc_ratios=True,
                                    calc_xs_absorption=True, ein_arr=[.01, .0253],
                                    theta=np.linspace(0, 180, 31))
    runner.cleanup(state)
    return result


@pytest.fixture
def live_library():
    """Require an explicitly configured, locally built native adapter."""
    path = os.environ.get('TARA_TEST_LIQUID_CAPTURE_LIBRARY')
    if not path:
        pytest.skip('Set TARA_TEST_LIQUID_CAPTURE_LIBRARY for real-plugin acceptance tests')
    return path


def test_live_sequential_and_spawn_parallel_equivalence(live_library, tmp_path):
    """Stable identities, independent reference, unique sandboxes and equal arrays."""
    before = SOURCE.read_bytes()
    preload = os.environ.get('LD_PRELOAD')
    jobs = [(live_library, str(tmp_path), i, None) for i in (-1, 0, 1)]
    serial = [_live_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        parallel = list(pool.map(_live_job, jobs))
    for expected, actual in zip(serial, parallel):
        assert expected.sample_index == actual.sample_index
        pd.testing.assert_frame_equal(expected.diff.sab, actual.diff.sab)
        pd.testing.assert_series_equal(expected.integ.xs_total, actual.integ.xs_total)
        pd.testing.assert_frame_equal(expected.integ.energy_distribution, actual.integ.energy_distribution)
        pd.testing.assert_frame_equal(expected.diff.dynamic_structure_factor[.0253], actual.diff.dynamic_structure_factor[.0253])
        assert actual.metadata['thermodynamics_scope'] == 'target solid-like VDOS only'
    assert not serial[1].diff.sab.equals(serial[2].diff.sab)
    assert SOURCE.read_bytes() == before
    assert os.environ.get('LD_PRELOAD') == preload
    assert not list(tmp_path.iterdir())


def test_live_active_h_preserves_kernel_and_material_fraction(live_library, tmp_path):
    full = _live_job((live_library, str(tmp_path), 0, None))
    hydrogen = _live_job((live_library, str(tmp_path), 0, {'H'}))
    pd.testing.assert_frame_equal(full.diff.sab, hydrogen.diff.sab)
    assert np.all(hydrogen.integ.xs_total < full.integ.xs_total)
    pd.testing.assert_series_equal(full.integ.xs_absorption, hydrogen.integ.xs_absorption)
    assert hydrogen.metadata['active_species'] == ['H']
    assert not list(tmp_path.iterdir())


def test_live_inactive_target_fails_without_leftover_files(live_library, tmp_path):
    with pytest.raises(RuntimeError, match='inactive analysis target'):
        _live_job((live_library, str(tmp_path), 0, {'O'}))
    assert not list(tmp_path.iterdir())


def test_live_integrated_kernel_matches_native_hydrogen(live_library, tmp_path):
    """Check normalization, alpha scaling and quadrature against native XS."""
    energies = [.001, .01, .0253, .1]
    runner = NCrystalRunner(str(SOURCE), 295., temp_dir=tmp_path,
                            liquid_capture_library=live_library, vdoslux=0,
                            active_species={'H'})
    state = runner.execute(0, spectrum(), atom_label='H', energy_grid=energies)
    result = NCrystalParser().parse(0, state, calc_scattering_kernel=True,
                                    ein_arr=energies, theta=np.linspace(0, 180, 721))
    component = state.liquid_data.kernels['H']
    awr = component.mass_amu / 1.00866491595
    free_xs = component.bound_xs * (awr / (awr + 1))**2
    for energy, kernel in result.integ.scattering_kernel.items():
        integral = np.trapezoid(kernel.values, kernel.index) * free_xs * component.fraction
        assert integral == pytest.approx(result.integ.xs_total.loc[energy], rel=2e-3)
    assert not list(tmp_path.iterdir())


def test_live_real_sampler_through_ensemble_compute(live_library, tmp_path):
    """Exercise real Core samples and Analyzer aggregation with a test scheduler.

    The uploaded archive omits parallel_execution.py. Only its scheduling
    boundary is substituted; Core, Runner, child, Parser and aggregation are
    real. This does not validate the unavailable allocation/HPC policy.
    """
    from concurrent.futures import ThreadPoolExecutor
    import importlib.util
    import sys
    import types
    from core.config import TaraConfig
    from core.sampler.concrete_samplers import TaraSampler

    calls = []
    def scheduler(function, jobs, *, n_jobs):
        calls.append(n_jobs)
        if n_jobs == 1:
            return [function(*args, **kwargs) for args, kwargs in jobs]
        assert n_jobs == -1
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(function, *args, **kwargs) for args, kwargs in jobs]
            return [future.result() for future in futures]

    placeholder = types.ModuleType('parallel_execution')
    placeholder.run_parallel = scheduler
    module_path = Path(__file__).parents[2] / 'analyzer' / 'ensemble_analyzer.py'
    spec = importlib.util.spec_from_file_location('analyzer._liquid_ensemble_under_test', module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'parallel_execution': placeholder}):
        spec.loader.exec_module(module)

    ensemble = TaraSampler(TaraConfig(max_energy_shift_mev=1., max_amplitude_shift_pct=5.,
                                      n_samples=3, random_seed=17), spectrum()).generate_ensemble()
    runner = NCrystalRunner(str(SOURCE), 295., temp_dir=tmp_path,
                            liquid_capture_library=live_library, vdoslux=0)
    common = dict(runner_kwargs={'atom_label': 'H', 'energy_grid': [.001, .01, .1]},
                  parser_kwargs={'calc_sab': True, 'calc_energy_distribution': True,
                                 'calc_cosine_distribution': True, 'calc_scattering_kernel': True,
                                 'calc_ratios': True, 'ein_arr': [.01, .0253],
                                 'theta': np.linspace(0, 180, 31)})
    serial = module.EnsembleAnalyzer(ensemble, runner, NCrystalParser())
    serial.compute(parallel=False, **common)
    concurrent = module.EnsembleAnalyzer(ensemble, runner, NCrystalParser())
    concurrent.compute(parallel=True, **common)
    assert calls == [1, -1]
    assert serial.reference_result.sample_index == concurrent.reference_result.sample_index == -1
    assert [r.sample_index for r in concurrent.raw_results] == [0, 1, 2]
    assert concurrent.xs_total_df.index.tolist() == [0, 1, 2]
    pd.testing.assert_frame_equal(serial.xs_total_df, concurrent.xs_total_df)
    pd.testing.assert_frame_equal(serial.thermo_df, concurrent.thermo_df)
    for ein in [.01, .0253]:
        pd.testing.assert_frame_equal(serial.energy_distribution_dict[ein],
                                      concurrent.energy_distribution_dict[ein])
        assert np.isfinite(concurrent.energy_distribution_dict[ein]).all().all()
    for result in concurrent.raw_results:
        assert result.metadata['components']['H']['fraction'] == pytest.approx(2/3)
    assert concurrent.thermo_df.attrs['scope'] == 'target solid-like VDOS only'
    reduced = concurrent.reduce([2, 0])
    assert calls == [1, -1]  # Reduction reuses results, never reruns the backend.
    assert [result.sample_index for result in reduced.raw_results] == [2, 0]
    assert reduced.reference_result is concurrent.reference_result
    extracted = reduced.extract(['thermo', 'xs_total', 'energy_distribution'])
    assert list(extracted.metadata['samples']) == [2, 0]
    assert extracted.metadata['reference']['sample_index'] == -1
    assert extracted.metadata['samples'][2]['components']['H']['bound_xs'] > 0
    assert extracted.thermo.attrs['scope'] == 'target solid-like VDOS only'
    assert set(extracted.energy_distribution) == {.01, .0253}
    assert not list(tmp_path.iterdir())


def test_builder_rejects_unsupported_platform_before_compilation(tmp_path):
    from processing.backends.liquid_capture.build import build_capture_library
    with patch('processing.backends.liquid_capture.build.platform.system', return_value='Darwin'), \
         patch('processing.backends.liquid_capture.build.subprocess.run') as compile_call:
        with pytest.raises(RuntimeError, match='Linux/ELF'):
            build_capture_library(tmp_path)
    compile_call.assert_not_called()


def test_builder_rejects_unverified_cpp_version_before_compilation(tmp_path):
    from processing.backends.liquid_capture.build import build_capture_library
    with patch('processing.backends.liquid_capture.build.subprocess.check_output', return_value='4.4.7\n'), \
         patch('processing.backends.liquid_capture.build.subprocess.run') as compile_call:
        with pytest.raises(RuntimeError, match='verified for NCrystal 4.4.6'):
            build_capture_library(tmp_path)
    compile_call.assert_not_called()


def test_modified_adapter_is_rejected_before_running_plugin(live_library, tmp_path):
    import shutil
    altered = tmp_path / 'libsabtap.so'
    shutil.copyfile(live_library, altered)
    shutil.copyfile(Path(live_library).with_suffix('.json'), altered.with_suffix('.json'))
    with altered.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(RuntimeError, match='installation changed'):
        NCrystalLiquidAnalysisProcessor(SOURCE, 295., capture_library=altered)


def test_material_runner_native_cross_sections_and_capture_agree(live_library, tmp_path):
    """Compare native and captured material paths at the same incident energies.

    Parameters
    ----------
    live_library : pathlib.Path
        Verified capture adapter for the installed NCrystal library.
    tmp_path : pathlib.Path
        Temporary directory for backend resources.
    """
    import NCrystal as NC
    from scipy.constants import physical_constants
    from analyzer.ensemble_analyzer import EnsembleAnalyzer
    from core.data.material_ensemble import MaterialEnsemble
    from core.data.material_scattering import MaterialScatteringModel, ScatteringSpecies, SpeciesDynamics
    from input_output.ncrystal.liquids import parse_custom_liquids

    neutron_mass = physical_constants["neutron mass in u"][0]
    hydrogen = ScatteringSpecies("H", 2, NC.atomDB("H").averageMassAMU() / neutron_mass)
    oxygen = ScatteringSpecies("O", 1, NC.atomDB("O").averageMassAMU() / neutron_mass)
    entry, = parse_custom_liquids(SOURCE.read_text())
    dynamics = SpeciesDynamics(
        spectrum(), entry.ws, entry.wt, "egelstaff_schofield", translation_parameter=entry.c,
    )
    reference = MaterialScatteringModel("water", "liquid", 295, hydrogen, dynamics, (oxygen,))
    materials = MaterialEnsemble(reference, np.array([entry.wt]), np.array([entry.c * 1.1]))
    runner = NCrystalRunner(str(SOURCE), 295, temp_dir=str(tmp_path),
                            liquid_capture_library=live_library)
    analyzer = EnsembleAnalyzer(materials, runner, NCrystalParser())
    parser_options = dict(energy_grid=np.geomspace(1e-5, 0.5, 20), calc_thermodynamics=False)
    analyzer.compute(parallel=False, parser_kwargs=parser_options)
    assert runner.processor.capture_library is None
    native = analyzer.raw_results[0].integ.xs_total.copy()
    assert not np.allclose(native, analyzer.reference_result.integ.xs_total)
    analyzer.compute(parallel=False, parser_kwargs={**parser_options, "calc_sab": True})
    assert runner.processor.capture_library.is_file()
    assert not analyzer.raw_results[0].diff.sab.empty
    np.testing.assert_allclose(analyzer.raw_results[0].integ.xs_total, native, rtol=1e-9)

"""NCrystal runner and parser contracts for solids and liquids."""


from unittest.mock import patch
import pytest
pytest.importorskip('NCrystal')
import NCrystal.misc
from analyzer.runners.ncrystal_runner import NCrystalRunner
import numpy as np
from analyzer.backends.ncrystal_engine import NCrystalEngine
from analyzer.parsers.ncrystal_parser import NCrystalParser
from analyzer.physics.data_models import DifferentialData
from analyzer.physics.data_models import IntegralData
from dataclasses import replace
import pandas as pd
from scipy.constants import physical_constants
from analyzer.physics.kinematics import Dsf
from processing.backends.liquid_capture.data import LiquidAnalysisState
from processing.backends.liquid_capture.data import LiquidKernel


# Ncrystal active species runner

def test_runner_passes_species_and_preserves_legacy_constructor():
    with patch('analyzer.runners.ncrystal_runner.NcrystalProcessor') as processor_cls:
        processor_cls.return_value.temp_dir = '/tmp'
        processor_cls.return_value.active_species = frozenset({'H'})
        runner = NCrystalRunner('base.ncmat', 296.0, active_species={'H'})
        assert runner.active_species == frozenset({'H'})
        assert processor_cls.call_args.kwargs['active_species'] == {'H'}
    with patch('analyzer.runners.ncrystal_runner.NcrystalProcessor') as processor_cls:
        processor_cls.return_value.temp_dir = '/tmp'
        processor_cls.return_value.active_species = None
        NCrystalRunner('base.ncmat', 296.0)
        assert processor_cls.call_args.kwargs['active_species'] is None


def test_runner_passes_preloaded_scattering_components_to_parser():
    """Extract the component cross sections without loading a material in Analyzer."""
    import numpy as np

    from analyzer.parsers.ncrystal_parser import NCrystalParser
    from analyzer.physics.data_models import IntegralData

    energy = np.array([0.01, 0.02])
    spectrum = object()
    with patch('analyzer.runners.ncrystal_runner.NcrystalProcessor') as processor_cls:
        processor = processor_cls.return_value
        runner = NCrystalRunner('base.ncmat', 296.0)
        state = processor.create_analysis_state.return_value
        for name, value in (('scatter_inelastic', 1.0), ('scatter_coherent_elastic', 2.0),
                            ('scatter_incoherent_elastic', 3.0)):
            getattr(state, name).xsect.return_value = np.full(energy.shape, value)

        engine = runner.execute(1, spectrum, atom_label='C')
        assert engine.backend_energy_grid is state.backend_energy_grid
        assert engine.backend_pdos is state.backend_pdos
        processor.create_analysis_state.assert_called_once_with(
            sample_index=1, spectrum=spectrum, atom_label='C',
            load_scattering_components=True,
        )
        parser = NCrystalParser()
        cross_sections = IntegralData()
        with patch('NCrystal.misc.MaterialSource', side_effect=AssertionError("Unexpected load")):
            parser._extract_cross_sections(
                engine, cross_sections, energy_grid=energy, calc_xs_total=False,
            )
            parser._extract_cross_sections(
                engine, IntegralData(), energy_grid=energy, calc_xs_total=False,
                calc_xs_inelastic=False, calc_xs_elastic=False,
            )
        for values, handle, expected in (
            (cross_sections.xs_inelastic, state.scatter_inelastic, 1.0),
            (cross_sections.xs_coherent_elastic, state.scatter_coherent_elastic, 2.0),
            (cross_sections.xs_incoherent_elastic, state.scatter_incoherent_elastic, 3.0),
        ):
            np.testing.assert_array_equal(values.index.to_numpy(), energy)
            np.testing.assert_array_equal(values.to_numpy(), np.full(energy.shape, expected))
            handle.xsect.assert_called_once_with(energy)

        runner.cleanup(engine)
        processor.cleanup_analysis_file.assert_called_once_with(state.filepath)


# Ncrystal kernel grid

def _engine(with_backend=True):
    """Provide a Debye-extended Core PDOS and an independent source grid."""
    internal_energy = np.r_[np.linspace(0.0, 0.01, 2001)[:-1],
                            0.01 + 0.001 * np.arange(15)]
    internal_density = internal_energy**2
    source_energy = 0.01 + 0.001 * np.arange(15)
    return NCrystalEngine(
        filepath="unused.ncmat", temperature=300.0, target_lbl="H",
        mat=None, nuc_data=None, mass_amu=1.0,
        energy_grid=internal_energy, pdos=internal_density,
        backend_energy_grid=source_energy if with_backend else None,
        backend_pdos=source_energy**2 if with_backend else None,
    )


def test_ncrystal_kernel_selection_retains_original_first_point():
    """NCrystal sees the source 0.01 eV grid, not the full Core E=0 tail."""
    engine = _engine()
    E, rho = NCrystalParser._kernel_vdos(engine)
    np.testing.assert_array_equal(E, engine.backend_energy_grid)
    np.testing.assert_array_equal(rho, engine.backend_pdos)
    assert E[0] == 0.01
    assert engine.energy_grid[0] == 0.0


def test_gn_and_knl_receive_same_backend_vdos():
    """Both NCrystal VDOS kernel APIs consume exactly the serialized grid."""
    engine = _engine()
    parser = NCrystalParser()
    calls = []

    def fake_gn(*, vdos, n, mass_amu, temperature):
        calls.append(("gn", vdos[0].copy(), vdos[1].copy()))
        return vdos[0], vdos[1]

    def fake_knl(*, vdos, mass_amu, temperature, target_emax):
        calls.append(("knl", vdos[0].copy(), vdos[1].copy()))
        return {
            "alpha": np.array([0.0, 1.0]),
            "beta": np.array([-1.0, 0.0, 1.0]),
            "sab": np.ones(6),
        }

    with patch("analyzer.parsers.ncrystal_parser.extractGn", fake_gn), \
         patch("analyzer.parsers.ncrystal_parser.extractKnl", fake_knl), \
         patch.object(parser, "_apply_universal_kinematics") as universal:
        parser._extract_kinematics(
            engine, DifferentialData(), IntegralData(),
            calc_gn=True, calc_sab=True,
        )
    assert universal.call_count == 1
    assert [kind for kind, *_ in calls] == ["gn", "knl"]
    for _, E, rho in calls:
        np.testing.assert_array_equal(E, engine.backend_energy_grid)
        np.testing.assert_array_equal(rho, engine.backend_pdos)


def test_missing_backend_grid_rejects_internal_debye_spacing():
    """Do not repair E=0 or fine Debye spacing inside the Analyzer parser."""
    with pytest.raises(ValueError, match="below 1e-5 eV"):
        NCrystalParser._kernel_vdos(_engine(with_backend=False))


def test_backend_fields_must_be_present_together():
    """Prevent a mismatch between energy coordinates and density samples."""
    engine = _engine()
    engine.backend_pdos = None
    with pytest.raises(ValueError, match="Incomplete NCrystal backend"):
        NCrystalParser._kernel_vdos(engine)


def test_manual_regular_grid_keeps_legacy_constructor_working():
    """Older callers with already-compatible VDOS arrays remain supported."""
    engine = _engine(with_backend=False)
    engine.energy_grid = np.array([0.01, 0.011, 0.012])
    engine.pdos = np.array([1.0, 2.0, 3.0])
    E, density = NCrystalParser._kernel_vdos(engine)
    np.testing.assert_array_equal(E, engine.energy_grid)
    np.testing.assert_array_equal(density, engine.pdos)


# Ncrystal liquid kernel

def engine():
    """Provide an oxygen-mass table so alpha-convention errors are conspicuous."""
    alpha = np.linspace(.001, 100, 80)
    beta = np.linspace(-10, 10, 101)
    sab = np.exp(-alpha[None, :] / 30 - beta[:, None]**2 / 10 - beta[:, None] / 2)
    kernel = LiquidKernel('O', alpha, beta, sab, 15.999, 295., 4.23, 1/3, 1.)
    energy = np.linspace(.001, .08, 80)
    rho = energy**2 / np.trapezoid(energy**2, energy)
    state = LiquidAnalysisState('O', {'O': kernel}, energy, rho, energy, rho,
                                np.array([.01, .1]), np.array([4., 3.]), np.array([.2, .1]),
                                {'teff': 400., 'msd': .01},
                                {'thermodynamics_scope': 'target solid-like VDOS only'})
    return NCrystalEngine('', 295., 'O', None, None, 15.999, energy, rho,
                          backend_energy_grid=energy, backend_pdos=rho, liquid_data=state)


def test_raw_signed_table_reaches_common_math_without_vdos_regeneration():
    """Correct alpha exactly once and retain every captured value and beta node."""
    e = engine()
    parser = NCrystalParser()
    with patch('analyzer.parsers.ncrystal_parser.extractKnl', side_effect=AssertionError), \
         patch.object(parser, '_apply_universal_kinematics') as apply:
        parser._extract_kinematics(e, DifferentialData(), IntegralData(), calc_sab=True)
    table = apply.call_args.args[0]
    k = e.liquid_data.kernels['O']
    np.testing.assert_array_equal(table.columns, k.beta)
    np.testing.assert_array_equal(table.to_numpy(), k.sab.T)
    np.testing.assert_allclose(table.index, k.alpha / (k.mass_amu / physical_constants['neutron mass in u'][0]))


def test_every_kinematic_output_and_native_cross_section():
    """All existing output flags operate on the same captured kernel."""
    e = engine()
    flags = dict(calc_sab=True, calc_dsf=True, calc_scattering_kernel=True,
                 calc_angular_distribution=True, calc_cosine_distribution=True,
                 calc_energy_distribution=True, calc_ratios=True,
                 calc_xs_absorption=True, ein_arr=[.01, .0253], theta=np.linspace(0, 180, 31))
    with patch('analyzer.parsers.ncrystal_parser.extractKnl', side_effect=AssertionError):
        result = NCrystalParser().parse(4, e, **flags)
    assert result.sample_index == 4
    assert result.metadata == e.liquid_data.provenance
    assert result.diff.sab.shape == (80, 101)
    assert set(result.diff.dynamic_structure_factor) == {.01, .0253}
    for attr in ('scattering_kernel', 'angular_distribution'):
        assert set(getattr(result.integ, attr)) == {.01, .0253}
    for attr in ('cosine_distribution', 'energy_distribution', 'upscattering_ratio',
                 'downscattering_ratio', 'elastic_ratio'):
        values = getattr(result.integ, attr)
        if attr == 'energy_distribution':
            assert all(np.isfinite(row.dropna()).all() and not row.dropna().empty
                       for _, row in values.iterrows())
        else:
            assert np.isfinite(values.to_numpy()).all()
    np.testing.assert_allclose(result.integ.upscattering_ratio + result.integ.downscattering_ratio + result.integ.elastic_ratio, 1.)
    np.testing.assert_array_equal(result.integ.xs_total, [4., 3.])
    np.testing.assert_array_equal(result.integ.xs_inelastic, result.integ.xs_total)
    np.testing.assert_array_equal(result.integ.xs_coherent_elastic, [0., 0.])
    np.testing.assert_array_equal(result.integ.xs_incoherent_elastic, [0., 0.])
    np.testing.assert_array_equal(result.integ.xs_absorption, [.2, .1])
    assert result.thermo.teff == 400.


def test_kinematics_agree_with_independent_coordinate_construction():
    """Catch accidental mass scaling or beta symmetry applied in the adapter."""
    e = engine()
    k = e.liquid_data.kernels['O']
    target_alpha = k.alpha * physical_constants['neutron mass in u'][0] / k.mass_amu
    expected = Dsf(pd.DataFrame(k.sab.T, index=target_alpha, columns=k.beta),
                   295., k.mass_amu).compute_data(.0253)
    result = NCrystalParser().parse(0, e, calc_dsf=True)
    np.testing.assert_allclose(result.diff.dynamic_structure_factor[.0253], expected)


def test_gn_is_explicitly_the_vibrational_component_only():
    e = engine()
    with patch('analyzer.parsers.ncrystal_parser.extractGn', return_value=([.01], [1.])) as gn, \
         patch('analyzer.parsers.ncrystal_parser.extractKnl', side_effect=AssertionError):
        result = NCrystalParser().parse(0, e, calc_gn=True)
    np.testing.assert_array_equal(gn.call_args.kwargs['vdos'][0], e.backend_energy_grid)
    assert result.diff.gn.iloc[0] == 1.
    assert result.diff.sab is None


@pytest.mark.parametrize('ein', [2., -1., np.nan])
def test_out_of_range_liquid_kinematics_fail(ein):
    with pytest.raises(ValueError, match='captured liquid kernel range'):
        NCrystalParser().parse(0, engine(), calc_dsf=True, ein_arr=[ein])


def test_cross_section_grid_mismatch_is_not_silently_interpolated():
    with pytest.raises(ValueError, match='same energy_grid'):
        NCrystalParser().parse(0, engine(), energy_grid=[.02, .1])


@pytest.mark.parametrize('field,value', [('mass_amu', -1), ('fraction', 2),
                                         ('suggested_emax', 0), ('temperature', np.nan)])
def test_invalid_component_metadata_rejected(field, value):
    with pytest.raises(ValueError):
        replace(engine().liquid_data.kernels['O'], **{field: value})


def test_capture_arrays_are_immutable_and_own_their_memory():
    k = engine().liquid_data.kernels['O']
    with pytest.raises(ValueError):
        k.sab[0, 0] = 7
    with pytest.raises(ValueError):
        replace(k, sab=np.full_like(k.sab, np.nan))
    with pytest.raises(ValueError):
        replace(k, alpha=k.alpha[::-1])


def test_multi_ein_outer_join_can_be_aggregated_without_fabricating_samples():
    """Missing beta coordinates in one incident-energy row are expected support."""
    import importlib.util
    from pathlib import Path
    import sys
    import types
    # The uploaded archive omits parallel_execution.py. This test exercises
    # aggregation only; the placeholder must never execute a worker.
    placeholder = types.ModuleType('parallel_execution')
    def forbidden(*args, **kwargs):
        raise AssertionError('Aggregation must not execute jobs')
    placeholder.run_parallel = forbidden
    module_path = Path(__file__).parents[2] / 'analyzer' / 'ensemble_analyzer.py'
    spec = importlib.util.spec_from_file_location('aggregation_under_test', module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'parallel_execution': placeholder}):
        spec.loader.exec_module(module)
    result = NCrystalParser().parse(4, engine(), calc_energy_distribution=True,
                                   ein_arr=[.01, .0253])
    analyzer = module.EnsembleAnalyzer.__new__(module.EnsembleAnalyzer)
    analyzer.raw_results = [result]
    analyzer._aggregate_dataframe_multi_ein()
    analyzer._aggregate_thermodynamics()
    assert analyzer.thermo_df.attrs['scope'] == 'target solid-like VDOS only'
    for ein, table in analyzer.energy_distribution_dict.items():
        assert list(table.index) == [4]
        expected = result.integ.energy_distribution.loc[ein].dropna()
        np.testing.assert_array_equal(table.columns, expected.index)
        np.testing.assert_array_equal(table.iloc[0], expected)

"""LEAPR full-spectrum projection and native oscillator export."""


from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import pytest
from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import DiscreteOscillator
from core.physics.vibrational_spectrum import reconstruct_vibrational_spectrum
from input_output.leapr.core import LeaprFile
from input_output.leapr.exceptions import LeaprFormatError
from input_output.leapr.material_adapter import material_model_from_leapr
from input_output.leapr.workflows import prepare_full_vibrational_template
from input_output.leapr.workflows import prepare_leapr_spectrum
from input_output.leapr.workflows import prepare_leapr_material
from input_output.leapr.workflows import inject_leapr_spectrum
from input_output.leapr.workflows import write_multiple_spectra
from unittest.mock import patch


# Full-spectrum reconstruction and LEAPR export

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/tsl-HinH2O.leapr'




def test_water_components_and_roundtrip(tmp_path):
    """The complete CAB reference survives export without duplicate modes."""
    source = LeaprFile.from_file(str(FIXTURE))
    original = pickle.dumps(source.temp_blocks)
    prepared = prepare_full_vibrational_template(source, temperature=295.)
    block = prepared.temp_blocks[0]
    assert block['nd'] == 0
    assert block['tbeta'] == pytest.approx(.52197 + .15667 + .31333)
    assert block['twt'] == .0080324
    assert block['c'] == 4.0566
    spectrum = PhononSpectra(prepared.pdos)
    grid = spectrum.energy_grid
    for lo, hi, weight in [(0., .16, .52197), (.16, .3, .15667), (.3, .5, .31333)]:
        mask = (grid >= lo) & (grid <= hi)
        assert np.trapezoid(spectrum.density_values[mask], grid[mask]) * block['tbeta'] == pytest.approx(weight, abs=1e-9)
    for template in (source, prepared, pickle.loads(pickle.dumps(prepared))):
        writer = prepare_leapr_spectrum(template, spectrum, temperature=295., export_full_spectra=True)
        path = tmp_path / 'sample.leapr'
        writer.write(str(path))
        reread = LeaprFile.from_file(str(path))
        assert reread.temp_blocks[0]['nd'] == 0
        assert reread.temp_blocks[0]['tbeta'] == pytest.approx(block['tbeta'])
        np.testing.assert_allclose(reread.pdos, spectrum.pdos, rtol=2e-6, atol=1e-8)
    assert pickle.dumps(source.temp_blocks) == original
    model = material_model_from_leapr(source, state='liquid', temperature=295.)
    assert not model.dynamics.oscillators
    writer = prepare_leapr_material(source, model, export_full_spectra=True)
    assert writer.temp_blocks[0]['nd'] == 0


def test_refined_grid_mutation_and_batch(tmp_path):
    """Explicit reconstruction spacing survives every spectrum writer path."""
    source = LeaprFile.from_file(str(FIXTURE))
    grid = np.arange(1801) * .00025
    prepared = prepare_full_vibrational_template(source, temperature=295., energy_grid=grid)
    spectrum = PhononSpectra(prepared.pdos)
    _ = prepared.pdos  # Exercise cached-property invalidation.
    inject_leapr_spectrum(prepared, spectrum, export_full_spectra=True)
    np.testing.assert_allclose(prepared.pdos.index, grid)
    frame = pd.DataFrame([spectrum.density_values] * 2, columns=grid)
    write_multiple_spectra(prepared, frame, output_base='s', directory=str(tmp_path), export_full_spectra=True)
    for path in tmp_path.glob('*.leapr'):
        reread = LeaprFile.from_file(str(path))
        assert reread.temp_blocks[0]['delta'] == pytest.approx(.00025)
        assert reread.temp_blocks[0]['nd'] == 0


def test_incomplete_spectrum_rejected():
    """Old continuum-only samples must not silently lose discrete weight."""
    source = LeaprFile.from_file(str(FIXTURE))
    low = PhononSpectra(pd.Series(source.first_positive_spectrum,
                                 index=source.first_positive_energy_grid))
    with pytest.raises(LeaprFormatError, match='full vibrational'):
        prepare_leapr_spectrum(source, low, temperature=295.)


def test_exact_temperature_required():
    """Temperature-dependent diffusion weights are never borrowed implicitly."""
    source = LeaprFile.from_file(str(FIXTURE))
    with pytest.raises(ValueError, match='not explicitly tabulated'):
        prepare_leapr_spectrum(source, PhononSpectra(source.pdos), temperature=296.)


def test_real_sampling_changes_both_high_energy_regions():
    """Three real TARA draws include bending and stretching perturbations."""
    from core.config import TaraConfig
    from core.sampler.concrete_samplers import TaraSampler
    source = LeaprFile.from_file(str(FIXTURE))
    prepared = prepare_full_vibrational_template(source, temperature=295.)
    reference = PhononSpectra(prepared.pdos)
    config = TaraConfig(max_energy_shift_mev=1., max_amplitude_shift_pct=5., n_samples=3)
    ensemble = TaraSampler(config, reference).generate_ensemble()
    assert len(ensemble) == 3
    baseline = prepare_leapr_spectrum(prepared, reference, export_full_spectra=True).temp_blocks[0]
    for lo, hi in [(.15, .3), (.3, .45)]:
        changes = []
        for sample in ensemble:
            writer = prepare_leapr_spectrum(prepared, sample, export_full_spectra=True)
            block = writer.temp_blocks[0]
            assert block['nd'] == 0
            grid = writer.first_positive_energy_grid
            mask = (grid >= lo) & (grid <= hi)
            original = np.interp(grid, prepared.first_positive_energy_grid,
                                 baseline['phonon_spectrum'], left=0., right=0.)
            changes.append(np.max(np.abs(block['phonon_spectrum'][mask] - original[mask])))
        assert max(changes) > 1e-5


def test_parallel_serialization_matches_sequential():
    """Independent process workers serialize identical prepared physics."""
    from joblib import Parallel, delayed
    from input_output.leapr.workflows import serialize_leapr_spectrum
    source = LeaprFile.from_file(str(FIXTURE))
    prepared = prepare_full_vibrational_template(source, temperature=295.)
    spectrum = PhononSpectra(prepared.pdos)
    expected = serialize_leapr_spectrum(prepared, spectrum)
    results = Parallel(n_jobs=2)(delayed(serialize_leapr_spectrum)(prepared, spectrum)
                                 for _ in range(4))
    assert results == [expected] * 4




def _peak_metrics(energy, density, center):
    """Integrate a single mode, with energies in eV and density in inverse eV."""
    region = np.abs(energy - center) < .009
    x, y = energy[region], density[region]
    area = np.trapezoid(y, x)
    centroid = np.trapezoid(x * y, x) / area
    width = np.sqrt(np.trapezoid((x - centroid)**2 * y, x) / area)
    return np.array([area, centroid, width, y.max()])


def test_cab_gaussian_peaks_match_ncmat_and_resolve_half_bin_shifts():
    """Compare fixture peak shapes and test the actual spatial warp."""
    from core.math_engine.diffeomorphic import apply_spatial_warp
    from input_output.ncrystal.core import NCrystalFile

    cab = LeaprFile.from_file(str(FIXTURE))
    full = prepare_full_vibrational_template(cab, temperature=295.)
    spectrum = PhononSpectra(full.pdos)
    ncmat = NCrystalFile.from_file(str(FIXTURE.with_name('liquid_water_295K_illustrative.ncmat')))
    reference = PhononSpectra(ncmat.pdos_for_atom('H'))
    grid, density = spectrum.energy_grid, spectrum.density_values
    assert grid[1] == pytest.approx(.001265 / 4)
    half_step = grid[1] / 2
    shifted_grid, shifted = apply_spatial_warp(
        grid, density, [(.19, .22), (.40, .4215)],
        np.array([half_step, -half_step]) * 1000, freeze_origin=True,
    )
    for center, shift in zip((.205, .415), (half_step, -half_step)):
        original = _peak_metrics(grid, density, center)
        expected = _peak_metrics(reference.energy_grid, reference.density_values, center)
        np.testing.assert_allclose(original, expected, rtol=.02)
        moved = _peak_metrics(shifted_grid, shifted, center)
        assert moved[0] == pytest.approx(original[0], rel=1e-6)
        assert moved[1] == pytest.approx(center + shift, abs=1e-7)
        assert moved[2] == pytest.approx(original[2], rel=.02)
        assert moved[3] == pytest.approx(original[3], rel=.03)


def test_gaussian_rejects_coarse_grid():
    """An explicit coarse grid must not silently change the chosen peak model."""
    source = LeaprFile.from_file(str(FIXTURE))
    with pytest.raises(ValueError, match='spacing at most sigma / 3'):
        prepare_full_vibrational_template(
            source, temperature=295., energy_grid=np.arange(451) * .001,
        )


def test_nonuniform_sample_uses_fine_leapr_grid():
    """Do not project back to the old continuum grid on the raw template path."""
    source = LeaprFile.from_file(str(FIXTURE))
    full = prepare_full_vibrational_template(source, temperature=295.)
    grid = full.first_positive_energy_grid
    density = full.first_positive_spectrum
    energy = np.sort(np.r_[grid, grid[1] / 2])
    sample = PhononSpectra(pd.Series(np.interp(energy, grid, density), index=energy))
    writer = prepare_leapr_spectrum(source, sample, temperature=295., export_full_spectra=True)
    np.testing.assert_array_equal(writer.first_positive_energy_grid, grid)
    assert writer.temp_blocks[0]['nd'] == 0


def test_cab_export_spacing_is_independent_of_sampling_grid(tmp_path):
    """Round-trip the smaller deck and reuse its spacing for shifted samples."""
    from core.sampler.concrete_samplers import TaraSampler
    from core.config import TaraConfig
    from input_output.leapr.workflows import serialize_leapr_spectrum

    source = LeaprFile.from_file(str(FIXTURE))
    prepared = prepare_full_vibrational_template(source, temperature=295.)
    reference = PhononSpectra(prepared.pdos)
    config = TaraConfig(max_energy_shift_mev=1., max_amplitude_shift_pct=3., n_samples=3)
    ensemble = TaraSampler(config, reference).generate_ensemble()
    path = tmp_path / 'cab_export.leapr'
    original_grid = reference.energy_grid.copy()
    fine_spacing = original_grid[1]
    path.write_text(serialize_leapr_spectrum(prepared, reference, output_spacing_ev=2 * fine_spacing, export_full_spectra=True))
    export_template = LeaprFile.from_file(str(path))
    block = export_template.temp_blocks[0]
    assert block['delta'] == pytest.approx(.0006325)
    # The smaller table retains the full energy range, rounded up by at
    # most one export interval. The Gaussian tail margin may change.
    expected_intervals = len(original_grid) // 2
    assert block['ni'] == expected_intervals + 1
    assert export_template.first_positive_energy_grid[-1] >= original_grid[-1] - 1e-12
    assert block['ni'] < len(original_grid)
    assert block['nd'] == 0
    assert block['tbeta'] == pytest.approx(.99197)
    assert block['twt'] == .0080324 and block['c'] == 4.0566

    for sample in (ensemble.reference, *ensemble):
        fine = prepare_leapr_spectrum(prepared, sample, export_full_spectra=True)
        coarse = prepare_leapr_spectrum(export_template, sample)
        assert coarse.temp_blocks[0]['delta'] == 2 * fine_spacing
        for center in (.205, .415):
            expected = _peak_metrics(fine.first_positive_energy_grid, fine.first_positive_spectrum, center)
            actual = _peak_metrics(coarse.first_positive_energy_grid, coarse.first_positive_spectrum, center)
            np.testing.assert_allclose(actual[[0, 2]], expected[[0, 2]], rtol=.01)
            assert actual[1] == pytest.approx(expected[1], abs=1e-5)
            assert actual[3] == pytest.approx(expected[3], rel=.05)

    np.testing.assert_array_equal(reference.energy_grid, original_grid)
    assert reference.energy_grid[1] == pytest.approx(.00031625)


def test_native_export_recovers_reference_cards(tmp_path):
    """The untouched sampled reference retains the native CAB model."""
    from input_output.leapr.temperature import select_leapr_temperature_block

    source = LeaprFile.from_file(str(FIXTURE))
    prepared = prepare_full_vibrational_template(source, temperature=295.)
    original, _ = select_leapr_temperature_block(source, 295.)
    reference = PhononSpectra(prepared.pdos)
    for template in (source, prepared):
        writer = prepare_leapr_spectrum(template, reference, temperature=295.)
        path = tmp_path / 'native.leapr'
        writer.write(str(path))
        block = LeaprFile.from_file(str(path)).temp_blocks[0]
        for key in ('twt', 'c', 'tbeta', 'delta', 'ni', 'nd'):
            assert block[key] == original[key]
        for key in ('oscillator_energies', 'oscillator_weights', 'phonon_spectrum'):
            np.testing.assert_allclose(block[key], original[key], rtol=1e-6)


def test_native_export_serializes_sampled_centroids_and_weights(tmp_path):
    """Known shifts and scaling of distinct peaks survive native LEAPR export."""
    source = LeaprFile.from_file(str(FIXTURE))
    prepared = prepare_full_vibrational_template(source, temperature=295.)
    reference = PhononSpectra(prepared.pdos)
    energy, density = reference.energy_grid, reference.density_values
    continuum = np.where(energy < .18, density, 0.)
    bending = np.where((energy >= .18) & (energy < .30), density, 0.)
    stretching = np.where(energy >= .30, density, 0.)
    shifts = np.array([2, -3]) * energy[1]
    scales = np.array([1., 1.15, .85])
    values = (
        continuum + scales[1] * np.interp(energy - shifts[0], energy, bending, left=0., right=0.)
        + scales[2] * np.interp(energy - shifts[1], energy, stretching, left=0., right=0.)
    )
    sample = PhononSpectra(pd.Series(values, index=energy))
    path = tmp_path / 'sample.leapr'
    prepare_leapr_spectrum(source, sample, temperature=295.).write(str(path))
    block = LeaprFile.from_file(str(path)).temp_blocks[0]
    assert block['nd'] == 2 and block['ni'] == 119 and block['delta'] == .001265
    assert block['twt'] == .0080324 and block['c'] == 4.0566
    expected = np.array([.52197, .15667, .31333]) * scales
    expected *= .99197 / expected.sum()
    np.testing.assert_allclose(block['oscillator_energies'], [.205, .415] + shifts, atol=1e-7)
    np.testing.assert_allclose([block['tbeta'], *block['oscillator_weights']], expected, rtol=1e-6)
    assert np.trapezoid(block['phonon_spectrum'], dx=block['delta']) == pytest.approx(1.)


def test_native_export_rejects_spectrum_without_mode_separation():
    """Broad overlapping modes require the full-table opt-in."""
    source = LeaprFile.from_file(str(FIXTURE))
    prepared = prepare_full_vibrational_template(source, temperature=295.)
    spectrum = PhononSpectra(prepared.pdos)
    energy = spectrum.energy_grid
    overlap = np.exp(-0.5 * ((energy - .177) / .015)**2)
    sample = PhononSpectra(pd.Series(spectrum.density_values + overlap, index=energy))
    with pytest.raises(LeaprFormatError, match='export_full_spectra=True'):
        prepare_leapr_spectrum(source, sample, temperature=295.)
    writer = prepare_leapr_spectrum(source, sample, temperature=295., export_full_spectra=True)
    assert writer.temp_blocks[0]['nd'] == 0


# Uniform output grid projection

def test_leapr_ni_increases_but_delta_does_not_change():
    """An extended sampled spectrum is projected onto the original LEAPR step."""
    reference_grid = np.arange(8, dtype=float) * 0.001
    source = PhononSpectra(pd.Series(
        np.array([0.0, 1, 3, 6, 8, 8, 5, 3, 2, 1, 0], dtype=float),
        index=np.arange(11, dtype=float) * 0.001,
    ))

    class Template:
        first_positive_energy_grid = reference_grid
        temp_blocks = [{'temp': 300.0, 'delta': 0.001, 'ni': 8}]

    with patch('input_output.leapr.workflows.resolve_leapr_temperature', return_value=300.0), \
         patch('input_output.leapr.workflows._create_first_writer_copy',
               side_effect=lambda template, block, arr, options, temperature:
               {'delta': block['delta'], 'ni': len(arr), 'values': arr}):
        writer = prepare_leapr_spectrum(Template(), source)

    assert writer['delta'] == 0.001
    assert writer['ni'] == 11
    assert len(writer['values']) == writer['ni']
    np.testing.assert_allclose(
        np.trapezoid(writer['values'], np.arange(writer['ni']) * writer['delta']),
        1.0,
        atol=1e-12,
    )
    assert Template.temp_blocks[0]['ni'] == 8

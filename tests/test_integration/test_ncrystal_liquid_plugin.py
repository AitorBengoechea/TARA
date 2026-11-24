"""Integration tests for the compiled NCrystal Liquids plugin."""


from pathlib import Path
import numpy as np


# Generated NCMAT acceptance

def test_live_generated_reference_ncmat_passes_create_info_and_scatter(tmp_path, liquids_plugin):
    """V1, V4 and V5: generated data, not just a hand-written valid fixture."""
    NC = liquids_plugin
    from input_output.ncrystal.liquids import load_liquid_ncmat, replace_vdos
    fixture = Path(__file__).parents[1]/'fixtures'/'liquid_water_295K_illustrative.ncmat'
    source = load_liquid_ncmat(fixture)
    # This aligned 0.423eV grid is valid without regridding or changed moments.
    energies = np.arange(1, 1202) * (0.423/1201)
    density = np.exp(-energies/0.06)*energies**2
    density /= np.trapezoid(density, energies)
    generated = replace_vdos(source, 'H', energies, density)
    path = tmp_path/'sample_000.ncmat'
    path.write_text(generated)
    info = NC.createInfo(str(path))  # Previous V4 immediate failure point.
    assert info is not None
    scatter = NC.createScatter(f'{path};temp=295K')
    xs = np.asarray([scatter.crossSectionIsotropic(e) for e in (.005, .025, .1)])
    assert np.all(np.isfinite(xs)) and np.all(xs >= 0)
    assert xs.max() > 0


# Plugin cross sections and final states

def test_actual_native_plugin_scattering(liquids_plugin):
    from processing.backends.ncrystal_processor import NcrystalProcessor
    path = Path(__file__).parents[1] / 'fixtures' / 'liquid_water_295K_illustrative.ncmat'
    processor = NcrystalProcessor()
    state = processor.create_native_scattering_state(str(path), 295, verify_plugin=True)
    xs = processor.native_cross_sections(state, np.geomspace(1e-5, .2, 12))
    assert xs.shape == (12,) and np.isfinite(xs).all() and np.any(xs > 0)
    eo, mu = processor.native_final_states(state, .025, 40)
    assert eo.shape == mu.shape == (40,)
    assert np.isfinite(eo).all() and (eo >= 0).all()
    assert (np.abs(mu) <= 1).all()

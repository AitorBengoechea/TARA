"""Conservation and real CAB format regression tests, without NJOY execution."""
import numpy as np
import pandas as pd
import pytest
from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import DiscreteOscillator
from core.physics.vibrational_spectrum import reconstruct_vibrational_spectrum



@pytest.mark.parametrize('energy', [.2, .205, .415])
def test_mode_area_and_centroid(energy):
    """A deposited line preserves its weight and first energy moment."""
    continuum = PhononSpectra(pd.Series([0., 1., 0.], index=[0., .01, .02]))
    grid = np.arange(51) * .01
    full, weight = reconstruct_vibrational_spectrum(
        continuum, 0., [DiscreteOscillator('mode', energy, .7)], grid,
        oscillator_sigma_ev=0.)
    assert weight == .7
    assert np.trapezoid(full.density_values, grid) == pytest.approx(1.)
    assert np.trapezoid(grid * full.density_values, grid) == pytest.approx(energy)
    assert full.density_values[0] == full.density_values[-1] == 0.














@pytest.mark.parametrize('node_count', [3, 6])
@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('offset_in_steps', [0., 5e-13, 2e-12])
def test_mode_at_interior_grid_boundary(node_count, direction, offset_in_steps):
    """Snap roundoff excursions, and reject modes whose centers are outside."""
    grid = np.arange(node_count) * .01
    continuum = PhononSpectra(pd.Series([0., 1., 0.], index=grid[:3]))
    node = 1 if direction < 0 else node_count - 2
    energy = grid[node] + direction * offset_in_steps * grid[1]
    mode = DiscreteOscillator('boundary', energy, .7)

    if offset_in_steps > 1e-12:
        with pytest.raises(ValueError, match='interior nodes'):
            reconstruct_vibrational_spectrum(continuum, 0., [mode], grid, oscillator_sigma_ev=0.)
        return

    full, weight = reconstruct_vibrational_spectrum(continuum, 0., [mode], grid, oscillator_sigma_ev=0.)
    np.testing.assert_array_equal(np.flatnonzero(full.density_values), [node])
    assert np.trapezoid(full.density_values, grid) * weight == pytest.approx(.7)
    assert np.trapezoid(grid * full.density_values, grid) == pytest.approx(grid[node])

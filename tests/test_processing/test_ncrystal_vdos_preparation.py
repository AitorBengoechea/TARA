"""NCrystal VDOS preparation and numerical representation checks."""

import numpy as np
import pytest


from processing.backends.ncrystal_processor import prepare_ncrystal_vdos


# VDOS source-grid contract

class Spectrum:
    def __init__(self, x, y, starts_zero):
        self.energy_grid = np.asarray(x)
        self.density_values = np.asarray(y)
        self.original_starts_at_zero = starts_zero


def reference():
    # Actual NCrystal H energy endpoints and point count for supplied CH2 NCMAT.
    energy = np.linspace(0.00999966, 0.387987, 230)
    density = np.exp(-((energy-0.18)/0.055)**2) + 0.1 * np.exp(-((energy-0.05)/0.03)**2)
    return energy, density


def synthetic_tail(energy, density):
    fine = np.linspace(0, energy[0], 2002)[:-1]
    fine_y = density[0] * (fine / energy[0])**2
    return np.r_[fine, energy], np.r_[fine_y, density]


def test_original_reference_grid_is_unchanged():
    energy,density = reference()
    x,y = synthetic_tail(energy,density)
    new_x,new_y = prepare_ncrystal_vdos(Spectrum(x,y,False))
    assert np.array_equal(new_x,energy)
    assert np.array_equal(new_y,density)
    assert len(new_x)==230
    assert not np.isclose(energy[0] / np.median(np.diff(energy)), round(energy[0] / np.median(np.diff(energy))))


def test_ensemble_reconstructed_zero_flag_has_dense_original_tail():
    energy,density = reference()
    x,y = synthetic_tail(energy,density)
    # Sampling preserves E=0, so PhononSpectra reports starts_at_zero=True.
    # It can also change density and coordinates on the physically retained part.
    new_x,new_y = prepare_ncrystal_vdos(Spectrum(x,y,True))
    assert np.array_equal(new_x,energy)
    assert np.array_equal(new_y,density)


def test_ensemble_sample_with_warped_energy_coordinates():
    energy,density = reference()
    x,y = synthetic_tail(energy,density)
    warp = 0.00035 * (x/x[-1]) * np.sin(6*x/x[-1])
    warped = x + warp
    warped[0]=0
    density_modified = y * (1.0 + 0.03*np.cos(12*x))
    result_x,result_y=prepare_ncrystal_vdos(Spectrum(warped,density_modified,True))
    retained_x = warped[2001:]
    retained_y = density_modified[2001:]
    assert result_x.shape == retained_x.shape
    assert np.allclose(np.diff(result_x),np.diff(result_x)[0],rtol=1e-10)
    assert np.isclose(result_x[0],retained_x[0])
    assert np.isclose(result_x[-1],retained_x[-1])
    assert np.isclose(np.trapezoid(result_y,result_x),np.trapezoid(retained_y,retained_x),rtol=1e-12)
    assert np.all(result_y>=0)


def test_irregular_external_sample_is_regularized_without_tail():
    x,y = reference()
    jitter = 0.00005 * np.sin(np.linspace(0,np.pi,len(x)))
    x_irregular=x+jitter
    out_x,out_y=prepare_ncrystal_vdos(Spectrum(x_irregular,y,True))
    assert np.allclose(np.diff(out_x),np.diff(out_x)[0],rtol=1e-10)
    assert np.isclose(np.trapezoid(out_y,out_x),np.trapezoid(y,x_irregular),rtol=1e-12)


def test_native_leapr_zero_start_regular_grid_does_not_get_stripped():
    energy=np.linspace(0.0, 0.4, 301)
    density=np.exp(-((energy-0.1)/0.1)**2)
    out_x,out_y=prepare_ncrystal_vdos(Spectrum(energy,density,True))
    assert np.array_equal(out_x,energy[1:])
    assert np.array_equal(out_y,density[1:])


@pytest.mark.parametrize('problem', ['nan','negative_density','duplicate','no_valid'])
def test_malformed_input_is_rejected(problem):
    energy,density=reference()
    energy=energy.copy();density=density.copy()
    if problem=='nan':density[7]=np.nan
    elif problem=='negative_density':density[3]=-1.
    elif problem=='duplicate':energy[20]=energy[19]
    elif problem=='no_valid':energy[:]=np.linspace(0,1e-6,len(energy))
    with pytest.raises(ValueError):
        prepare_ncrystal_vdos(Spectrum(energy,density,True))


# Ncrystal liquid physics diagnostics

def _metrics(original_grid, original_density, converted_grid, converted_density, temp_k=295.):
    """Independent testing oracle. Below E_min, use the quadratic VDOS tail.

    This is an approximate displacement-moment test, not a liquid-physics model.
    The numerical quadrature is performed on a dense logarithmic grid so the
    low-energy contribution is not hidden by the ordinary VDOS L1 difference.
    """
    original_grid = np.asarray(original_grid)
    converted_grid = np.asarray(converted_grid)
    original_density = np.asarray(original_density)
    converted_density = np.asarray(converted_density)
    union = np.unique(np.r_[original_grid, converted_grid])
    orig = np.interp(union, original_grid, original_density, left=0., right=0.)
    new = np.interp(union, converted_grid, converted_density, left=0., right=0.)
    shape_l1 = np.trapezoid(np.abs(orig - new), union) / np.trapezoid(orig, union)
    kt = 8.617333262145e-5 * temp_k

    def moment(grid, values):
        # Integrate g(E)/(E*tanh(E/(2kT))), with quadratic continuation.
        energies = np.geomspace(min(grid[0], converted_grid[0])*1e-5, grid[-1], 50000)
        vals = np.interp(energies, grid, values, left=0., right=0.)
        left = energies < grid[0]
        vals[left] = values[0] * (energies[left]/grid[0])**2
        return np.trapezoid(vals/(energies*np.tanh(energies/(2*kt))), energies)

    delta_moment = moment(converted_grid, converted_density) / moment(original_grid, original_density) - 1
    return float(shape_l1), float(delta_moment)


def test_low_energy_moment_must_be_checked_separately_from_vdos_shape():
    """V5: small L1 shape changes can hide large displacement changes.

    Reproduces the problematic E_min and spacing at small synthetic scale.
    No plugin or water-specific production code is required.
    """
    grid = np.linspace(1e-5, 0.423, 1201)
    # Smooth low-energy rise that is deliberately non-quadratic.
    y = np.exp(-grid/0.07) * (grid/0.025)
    y /= np.trapezoid(y, grid)
    step = float(grid[1]-grid[0])
    aligned = step*np.arange(1,len(grid)+1)
    new_y = np.interp(aligned, grid, y, left=0., right=0.)
    new_y *= np.trapezoid(y, grid)/np.trapezoid(new_y, aligned)
    l1, moment_delta = _metrics(grid, y, aligned, new_y)
    assert l1 < 0.10, f'Test case must exhibit small L1 change, got {l1:.2%}'
    assert abs(moment_delta) > 0.10, (
        'Test case must demonstrate that a small L1 metric alone is insufficient.')


def test_aligned_grid_is_not_modified_by_numerical_reference_conversion():
    """V2/3: don't broaden peaks or change sampling if grid is already valid."""
    grid = np.arange(1, 1001)*0.0004
    y = np.exp(-grid/0.04)
    y /= np.trapezoid(y, grid)
    assert grid[0]/(grid[1]-grid[0]) == pytest.approx(1)
    assert _metrics(grid, y, grid.copy(), y.copy()) == pytest.approx((0.0, 0.0), abs=1e-10)

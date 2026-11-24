import pytest
import pandas as pd
import numpy as np
from core.data.input_pdos import PhononSpectra, PhononSpectraError

def test_phononspectra_leapr_style():
    """
    Test a spectrum that naturally starts at E=0.0 (e.g., standard NJOY LEAPR input).
    It should NOT apply the Debye extrapolation.
    """
    energy_grid = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    density = np.array([0.0, 10.0, 20.0, 10.0, 0.0])
    
    raw_pdos = pd.Series(density, index=energy_grid)
    spectra = PhononSpectra(raw_pdos)
    
    # Verify origin state handling
    assert spectra.original_starts_at_zero is True
    
    # Verify no additional points were injected (length remains identical)
    assert len(spectra.pdos) == len(raw_pdos)
    assert spectra.energy_grid[0] == 0.0
    
    # Verify strict area normalization to 1.0
    assert np.isclose(spectra.integral, 1.0, rtol=1e-5)


def test_phononspectra_ncrystal_style():
    """
    Test a spectrum that starts at E > 0.0 (e.g., NCrystal @DYNINFO grid).
    It MUST apply a quadratic Debye extrapolation down to E=0.0.
    """
    # Grid starting at E=0.01 (missing the origin)
    energy_grid = np.array([0.01, 0.02, 0.03, 0.04])
    density = np.array([10.0, 20.0, 10.0, 0.0])
    
    raw_pdos = pd.Series(density, index=energy_grid)
    spectra = PhononSpectra(raw_pdos)
    
    # Verify origin state handling triggered the extrapolation
    assert spectra.original_starts_at_zero is False
    
    # Verify the Debye tail was prepended (2000 points + the origin)
    assert len(spectra.pdos) > len(raw_pdos)
    assert spectra.energy_grid[0] == 0.0
    
    # Verify strict area normalization to 1.0
    assert np.isclose(spectra.integral, 1.0, rtol=1e-5)


def test_phononspectra_debye_quadratic_behavior():
    """
    Verify that the extrapolated tail follows E^2 physics (rho = k * E^2).
    """
    energy_grid = np.array([0.01, 0.02, 0.03])
    density = np.array([10.0, 20.0, 10.0])
    
    raw_pdos = pd.Series(density, index=energy_grid)
    spectra = PhononSpectra(raw_pdos)
    
    # Check the first few extrapolated points to ensure they follow quadratic scaling
    e_tail = spectra.energy_grid[1:10]
    rho_tail = spectra.density_values[1:10]
    
    # If rho = k * E^2, then rho / E^2 should be a constant (k)
    k_values = rho_tail / (e_tail ** 2)
    assert np.allclose(k_values, k_values[0], rtol=1e-5)


def test_phononspectra_validation_failures():
    """
    Ensure mathematically or physically invalid spectra trigger the custom PhononSpectraError.
    """
    # 1. Negative energies
    with pytest.raises(PhononSpectraError, match="cannot contain negative energies"):
        PhononSpectra(pd.Series([0.0, 1.0], index=[-0.01, 0.01]))
        
    # 2. Negative densities
    with pytest.raises(PhononSpectraError, match="density values cannot be negative"):
        PhononSpectra(pd.Series([0.0, -1.0, 0.0], index=[0.0, 0.01, 0.02]))
        
    # 3. Non-monotonic grid (Auto-correction test)
    # The class automatically sorts out-of-order grids during preprocessing.
    unsorted_spectra = PhononSpectra(pd.Series([0.0, 1.0, 0.0], index=[0.0, 0.02, 0.01]))
    assert unsorted_spectra.energy_grid[1] == 0.01

    # 4. NaNs in density
    with pytest.raises(PhononSpectraError, match="NaN or Infinite"):
        PhononSpectra(pd.Series([0.0, np.nan, 0.0], index=[0.0, 0.01, 0.02]))

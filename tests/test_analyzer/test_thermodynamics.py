import numpy as np
from analyzer.physics.thermodynamics import (
    calc_cv_numba, calc_entropy_numba, calc_zero_point_energy_numba,
    calc_free_energy_numba, calc_internal_energy_numba, calc_teff_numba
)

def test_thermodynamics_zero_temperature_limits():
    """
    Test thermodynamic property evaluations at the strict zero-temperature limit.

    Verifies that:
    - Heat capacity (Cv) and entropy evaluate identically to 0.0 at T=0.
    - Free energy converges identically to the Zero Point Energy.
    - Internal energy converges identically to the Zero Point Energy.
    """
    E = np.linspace(1e-5, 0.1, 100)
    pdos = E**2  # Debye approximation tail
    temp = 0.0
    
    cv = calc_cv_numba(pdos, E, temp)
    entropy = calc_entropy_numba(pdos, E, temp)
    zpe = calc_zero_point_energy_numba(pdos, E)
    free_energy = calc_free_energy_numba(pdos, E, temp)
    internal_energy = calc_internal_energy_numba(pdos, E, temp)
    
    assert cv == 0.0
    assert entropy == 0.0
    assert free_energy == zpe
    assert internal_energy == zpe

def test_thermodynamics_thermal_evaluations():
    """
    Test robust thermodynamic property evaluations at finite temperatures.

    Verifies that:
    - Internal energy is strictly greater than the Zero Point Energy at finite T.
    - Effective temperature evaluates to a finite scalar gracefully.
    - Overflow protection handles edge-cases without NaN leakage.
    """
    E = np.linspace(1e-5, 0.1, 100)
    pdos = E**2
    temp = 300.0
    
    zpe = calc_zero_point_energy_numba(pdos, E)
    internal_energy = calc_internal_energy_numba(pdos, E, temp)
    teff = calc_teff_numba(pdos, E, temp)
    
    assert internal_energy > zpe
    assert not np.isnan(internal_energy)
    assert not np.isnan(teff)

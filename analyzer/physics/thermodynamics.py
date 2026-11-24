import numpy as np
from numba import njit
from scipy.constants import physical_constants, hbar, m_u, eV, angstrom

# ==============================================================================
#  MODULE CONSTANTS
# ==============================================================================

# Extract Boltzmann constant in eV/K at module level so Numba can use it as a 
# compile-time constant. (Numba cannot access dictionary lookups inside JIT).
KB_EV = physical_constants["Boltzmann constant in eV/K"][0]

# Dynamically calculate the MSD conversion factor: (hbar^2) / (amu * eV)
# This yields m^2. We divide by (angstrom^2) to convert to physical Angstroms^2.
HBAR2_AMU_EV_ANGSTROM2 = (hbar**2) / (m_u * eV * (angstrom**2))

# ==============================================================================
#  NUMBA KERNELS (Thermodynamics)
# ==============================================================================

def _with_debye_tail(
    pdos: np.ndarray, E: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a positive-origin VDOS to zero for thermodynamic integration.

    Parameters
    ----------
    pdos : numpy.ndarray
        One-dimensional VDOS sampled at E.
    E : numpy.ndarray
        Matching one-dimensional energy grid in eV, assumed to be strictly
        increasing and nonnegative.

    Returns
    -------
    pdos, E : tuple of numpy.ndarray
        Original arrays for a zero-origin grid, or extended arrays in the
        same order. The tabulated spectrum retains its original scale.

    Raises
    ------
    ValueError
        If the arrays are empty or incompatible, or the first energy is
        negative or nonfinite.

    Notes
    -----
    The low-energy continuation is quadratic and anchored at the first
    tabulated density. Use the same 2,000 interior points as Core.
    """
    if E.ndim != 1 or pdos.shape != E.shape or E.size == 0:
        raise ValueError("E and pdos must be matching nonempty 1-D arrays.")

    first_energy = E[0]
    if not np.isfinite(first_energy) or first_energy < 0.0:
        raise ValueError("The first energy must be finite and nonnegative.")
    if first_energy == 0.0:
        return pdos, E

    # The tabulated spectrum already contains the join point.
    tail_energy = np.linspace(0.0, first_energy, num=2002)[:-1]
    tail_pdos = pdos[0] * (tail_energy / first_energy) ** 2
    return np.concatenate((tail_pdos, pdos)), np.concatenate((tail_energy, E))


@njit(cache=True)
def calc_cv_numba(pdos: np.ndarray, E: np.ndarray, T: float) -> float:
    """
    Calculates the Heat Capacity (Cv) at constant volume using Numba optimization.

    Formula:
        Cv(T) = integral( (E/kT)^2 * exp(E/kT) / (exp(E/kT) - 1)^2 * DOS(E) ) dE

    Parameters
    ----------
    pdos : np.ndarray
        The Partial Density of States (PDOS) density values.
    E : np.ndarray
        The energy grid corresponding to the PDOS (in eV).
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        The calculated Heat Capacity (Cv) in [eV/K].
    """
    # 1. Handle T=0 Limit (Ground State)
    if T <= 1e-6:
        return 0.0

    # 2. Compute the integrand (Vectorized)
    # beta = E / (kB * T)
    beta = E / (KB_EV * T)
    
    # Initialize kernel array
    kernel = np.empty_like(beta)

    # 3. Safe Loop for Kernel Calculation
    # We iterate manually to handle numerical singularities robustly
    for i in range(len(beta)):
        b = beta[i]
        
        # Limit check: x^2 * e^x / (e^x - 1)^2 as x->0 equals 1.0
        if b < 1e-6:
            kernel[i] = 1.0
            
        # Overflow check: e^x / (e^x)^2 = e^-x. For large x, this vanishes.
        elif b > 500:
            kernel[i] = 0.0
            
        else:
            ex = np.exp(b)
            denom = (ex - 1.0)
            # Standard Formula
            kernel[i] = (b * b) * ex / (denom * denom)

    # Integrate over the actual phonon energy grid.
    y = pdos * kernel
    return np.trapezoid(y, E)


@njit(cache=True)
def calc_entropy_numba(pdos: np.ndarray, E: np.ndarray, T: float) -> float:
    """
    Calculates the Vibrational Entropy (S) using Numba optimization.

    Formula:
        S(T) = kB * integral( [ beta / (exp(beta)-1) - ln(1 - exp(-beta)) ] * DOS(E) ) dE
        where beta = E / (kB * T)

    Parameters
    ----------
    pdos : np.ndarray
        The Partial Density of States (PDOS) density values.
    E : np.ndarray
        The energy grid corresponding to the PDOS (in eV).
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        The calculated Entropy in [eV/K].
    """
    # 1. Handle T=0 Limit
    if T <= 1e-6:
        return 0.0

    beta = E / (KB_EV * T)
    kernel = np.empty_like(beta)

    # 2. Safe Loop for Kernel Calculation
    for i in range(len(beta)):
        b = beta[i]
        
        if b < 1e-6:
            # As E->0, beta->0. The term -ln(1-exp(-b)) diverges.
            # However, physical PDOS(E) ~ E^2 for crystals, so PDOS(0)=0.
            # We set kernel to 0.0 to rely on PDOS killing the divergence.
            kernel[i] = 0.0 
            
        elif b > 500:
            # Large beta limit: S -> 0
            kernel[i] = 0.0
            
        else:
            ex = np.exp(b)
            # S = kB * ( term1 + term2 )
            term1 = b / (ex - 1.0)
            term2 = -np.log(1.0 - np.exp(-b))
            kernel[i] = term1 + term2

    # 3. Integrate
    y = pdos * kernel
    return KB_EV * np.trapezoid(y, E)

@njit(cache=True)
def calc_phonon_moment_numba(pdos: np.ndarray, E: np.ndarray, n: int) -> float:
    """
    Calculates the n-th spectral moment of the phonon density of states.

    Formula:
        mu_n = integral( E^n * DOS(E) ) dE

    Parameters
    ----------
    pdos : np.ndarray
        The Partial Density of States (PDOS) density values.
    E : np.ndarray
        The energy grid corresponding to the PDOS (in eV).
    n : int
        The order of the moment (e.g., 1, 2, -1, -2).

    Returns
    -------
    float
        The value of the calculated moment.
    """
    # 1. Create the integrand term E^n
    if n == 0:
        term = np.ones_like(E)
    elif n == 1:
        term = E
    elif n == 2:
        term = E * E
    else:
        # Generic case with singularity protection
        term = np.empty_like(E)
        for i in range(len(E)):
            val = E[i]
            # Handle singularity for negative moments at E=0
            if val < 1e-12 and n < 0:
                 term[i] = 0.0 
            else:
                 term[i] = val ** n
    
    # 2. Integrate
    y = pdos * term
    return np.trapezoid(y, E)

@njit(cache=True)
def calc_zero_point_energy_numba(pdos: np.ndarray, E: np.ndarray) -> float:
    """
    Calculates the Zero Point Energy (ZPE) using Numba optimization.

    Formula:
        E_zpe = 0.5 * integral( E * DOS(E) ) dE

    Parameters
    ----------
    pdos : np.ndarray
        The Partial Density of States (PDOS) density values.
    E : np.ndarray
        The energy grid corresponding to the PDOS (in eV).

    Returns
    -------
    float
        The calculated Zero Point Energy in [eV].
    """
    # Use the 1st moment (n=1) to calculate the integral of E * DOS(E)
    mu_1 = calc_phonon_moment_numba(pdos, E, 1)
    return 0.5 * mu_1


@njit(cache=True)
def calc_free_energy_numba(pdos: np.ndarray, E: np.ndarray, T: float) -> float:
    """
    Calculates the Helmholtz Free Energy (F) using Numba optimization.

    Formula:
        F(T) = U - TS
             = kB * T * integral( ln(2 * sinh(beta/2)) * DOS(E) ) dE
             = ZPE + kB * T * integral( ln(1 - exp(-beta)) * DOS(E) ) dE

    Parameters
    ----------
    pdos : np.ndarray
        The Partial Density of States (PDOS) density values.
    E : np.ndarray
        The energy grid corresponding to the PDOS (in eV).
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        The calculated Free Energy in [eV].
    """
    # 1. Calculate Zero Point Energy:
    zpe = calc_zero_point_energy_numba(pdos, E)

    # 2. Handle T=0 Limit (Zero Point Energy)
    if T <= 1e-6:
        return zpe

    beta = E / (KB_EV * T)
    kernel = np.empty_like(beta)

    # 3. Safe Loop for Kernel Calculation
    for i in range(len(beta)):
        b = beta[i]
        
        if b < 1e-9:
            # Limit E->0
            kernel[i] = 0.0
            
        else:
            # F_thermal kernel: ln(1 - e^-beta)
            kernel[i] = np.log(1.0 - np.exp(-b))

    # 4. Combine ZPE with the thermal integral
    y = pdos * kernel
    thermal_contribution = (KB_EV * T) * np.trapezoid(y, E)

    # 5. Return the total Free Energy
    return zpe + thermal_contribution

@njit(cache=True)
def calc_internal_energy_numba(pdos: np.ndarray, E: np.ndarray, T: float) -> float:
    """
    Calculates the Internal Energy (U) using Numba optimization.

    Formula:
        U(T) = integral( E * (occupancy + 0.5) * DOS(E) ) dE
             = ZPE + integral( E * occupancy * DOS(E) ) dE
        occupancy = 1 / (exp(E/kT) - 1)

    Parameters
    ----------
    pdos : np.ndarray
        The Partial Density of States (PDOS) density values.
    E : np.ndarray
        The energy grid corresponding to the PDOS (in eV).
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        The calculated Internal Energy in [eV].
    """
    # 1. Calculate Zero Point Energy
    zpe = calc_zero_point_energy_numba(pdos, E)

    # 2. Handle T=0 Limit (Zero Point Energy)
    if T <= 1e-6:
        return zpe

    beta = E / (KB_EV * T)
    kernel = np.empty_like(beta)

    # 3. Safe Loop for Kernel Calculation
    for i in range(len(beta)):
        b = beta[i]
        if b < 1e-6:
            # E * occupancy -> kT as E -> 0
            kernel[i] = (KB_EV * T)
            
        elif b > 500:
            kernel[i] = 0.0
            
        else:
            # U_thermal kernel: E / (e^beta - 1)
            occupancy = 1.0 / (np.exp(b) - 1.0)
            kernel[i] = E[i] * occupancy

    # 4. Calculate the thermal integral
    y = pdos * kernel
    thermal_contribution = np.trapezoid(y, E)

    # 5. Return the total Internal Energy
    return zpe + thermal_contribution

@njit(cache=True)
def calc_log_moment_numba(pdos: np.ndarray, E: np.ndarray) -> float:
    """
    Calculates the logarithmic moment of the phonon density of states.
    Formula:
        mu_log = exp( integral( ln(E) * DOS(E) ) dE / integral( DOS(E) ) dE )

    Parameters
    ----------
    pdos : np.ndarray
        The Partial Density of States (PDOS) density values.
    E : np.ndarray
        The energy grid corresponding to the PDOS (in eV).

    Returns
    -------
    float
        The calculated logarithmic moment.
    """
    # Normalize area to ensure consistent probability distribution
    area = np.trapezoid(pdos, E)
    if area == 0:
        return 0.0
        
    # Mask out values near E=0 to avoid taking the log of zero or negative numbers
    mask = E > 1e-12
    e_valid = E[mask]
    pdos_valid = pdos[mask]
    
    norm_pdos = pdos_valid / area
    ln_e = np.log(e_valid)
    
    # Integrate on the actual coordinate axis rather than assuming equally spaced samples.
    return np.exp(np.trapezoid(norm_pdos * ln_e, e_valid))


@njit(cache=True)
def calc_teff_numba(pdos: np.ndarray, E: np.ndarray, T: float) -> float:
    """
    Calculates the Effective Temperature (T_eff) using Numba optimization.

    Formula:
        T_eff = 1/k_B * integral( E * (occupancy + 0.5) * DOS(E) ) dE
        (This is equivalent to the internal energy per DoF divided by k_B)

    Parameters
    ----------
    pdos : np.ndarray
        The Partial Density of States (PDOS) density values.
    E : np.ndarray
        The energy grid corresponding to the PDOS (in eV).
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        The calculated Effective Temperature in [K].
    """
    if T <= 1e-6:
        # Zero-point motion limit
        kernel = 0.5 * E / KB_EV
        return float(np.trapezoid(kernel * pdos, E))

    beta = E / (KB_EV * T)
    kernel = np.empty_like(beta)

    for i in range(len(beta)):
        b = beta[i]
        if b < 1e-6:
            # E * (kT/E) / kB = T
            kernel[i] = T
        elif b > 500:
            kernel[i] = (E[i] * 0.5) / KB_EV
        else:
            occupancy = 1.0 / (np.exp(b) - 1.0)
            kernel[i] = (E[i] * (occupancy + 0.5)) / KB_EV

    return float(np.trapezoid(kernel * pdos, E))


@njit(cache=True)
def calc_msd_numba(pdos: np.ndarray, E: np.ndarray, T: float, mass_amu: float = 1.0) -> float:
    """
    Calculates the Mean Square Displacement (MSD) in Angstroms^2.
    
    Formula:
        MSD = (hbar^2 / mass) * integral( (1/E) * (occupancy + 0.5) * DOS(E) ) dE

    Parameters
    ----------
    pdos : np.ndarray
        The Partial Density of States (PDOS) density values.
    E : np.ndarray
        The energy grid corresponding to the PDOS (in eV).
    T : float
        Temperature in Kelvin.
    mass_amu : float
        The atomic mass of the element in AMU (e.g., 1.008 for H, 12.011 for C).

    Returns
    -------
    float
        The calculated Mean Square Displacement in [Angstroms^2].
    """
    integrand = np.empty_like(E)

    if T <= 1e-6:
        # Ground State Limit (Zero Point Displacement)
        for i in range(len(E)):
            if E[i] < 1e-8:
                integrand[i] = 0.0 # g(E)/E goes to 0 as E->0
            else:
                integrand[i] = (0.5 / E[i]) * pdos[i]
    else:
        # Thermal Limit
        beta = E / (KB_EV * T)
        for i in range(len(beta)):
            b = beta[i]
            if E[i] < 1e-8:
                # CRITICAL: L'Hopital's analytical limit for 0/0 acoustic tail.
                # Integrand ~ (1/E) * (kT/E) * (A * E^2) = A * kT
                # We find the Debye coefficient 'A' from the next bin.
                if i + 1 < len(E) and E[i+1] > 1e-8:
                    A = pdos[i+1] / (E[i+1]**2)
                    integrand[i] = A * (KB_EV * T)
                else:
                    integrand[i] = 0.0
            elif b > 500:
                integrand[i] = (0.5 / E[i]) * pdos[i]
            else:
                occupancy = 1.0 / (np.exp(b) - 1.0)
                integrand[i] = ((occupancy + 0.5) / E[i]) * pdos[i]

    integral = np.trapezoid(integrand, E)
    
    # Apply the quantum mass-scaling factor
    return float((HBAR2_AMU_EV_ANGSTROM2 / mass_amu) * integral)

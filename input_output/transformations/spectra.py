"""Explicit acoustic-tail representations at native file boundaries."""
import numpy as np
from input_output.transformations.models import ContinuousSpectrum, InputTransformationError


def to_leapr_grid(spectrum: ContinuousSpectrum) -> ContinuousSpectrum:
    """Convert positive-origin NCrystal VDOS to a uniform LEAPR grid.

    Parameters
    ----------
    spectrum : ContinuousSpectrum
        Spectrum on a regular, positive-origin energy grid in eV.

    Returns
    -------
    ContinuousSpectrum
        Normalized density on a zero-origin grid with a quadratic acoustic tail.

    Raises
    ------
    InputTransformationError
        If the source grid is invalid or its first density cannot anchor the tail.

    Notes
    -----
    When the positive NCrystal starting energy is an integral number of source
    steps from zero, preserve all source ordinates exactly (up to one scalar).
    Otherwise resample linearly on the origin-anchored grid, and preserve the
    original high-energy support without extrapolation beyond its last node.
    Below the first NCrystal node use the E**2 acoustic continuation.
    """
    energy, density = spectrum.energy_ev, spectrum.density_per_ev
    step = _validate_leapr_source(energy, density)
    target_energy, target_density = _build_leapr_table(energy, density, step)
    return ContinuousSpectrum(target_energy, target_density)


def _validate_leapr_source(energy, density):
    """Validate the NCrystal grid and its acoustic-tail anchor.

    Parameters
    ----------
    energy, density : numpy.ndarray
        Source energies in eV and density in inverse eV.

    Returns
    -------
    float
        Mean grid spacing in eV.

    Raises
    ------
    InputTransformationError
        If the origin, spacing, or first density is unsuitable.
    """
    if energy[0] == 0:
        raise InputTransformationError('NCrystal VDOS must have a positive first energy.')
    steps = np.diff(energy)
    delta = float(np.mean(steps))
    if not np.allclose(steps, delta, rtol=1e-6, atol=1e-11):
        raise InputTransformationError('NCrystal source VDOS must be on a regular grid.')
    if density[0] <= 0:
        raise InputTransformationError('Cannot construct an E**2 tail from zero initial NCrystal density.')
    return delta


def _build_leapr_table(energy, density, step):
    """Build a zero-origin table with a quadratic tail below the first source node.

    Parameters
    ----------
    energy, density : numpy.ndarray
        Validated source energies in eV and density in inverse eV.
    step : float
        Validated uniform grid spacing in eV.

    Returns
    -------
    tuple of numpy.ndarray
        Target energy grid and density values before normalization.
    """
    nstart = int(round(energy[0] / step))
    aligned = nstart >= 1 and np.isclose(energy[0], nstart * step, rtol=1e-6, atol=1e-8)
    if aligned:
        # Copy the source ordinates directly when their nodes align with the target.
        target_energy = np.arange(nstart + len(energy), dtype=float) * step
        tail = _quadratic_tail(target_energy[:nstart], energy[0], density[0])
        return target_energy, np.r_[tail, density]

    # Stop at the last target node within the source's high-energy support.
    count = int(np.floor(energy[-1] / step + 1e-10)) + 1
    target_energy = np.arange(count, dtype=float) * step
    target_density = np.interp(target_energy, energy, density)
    below = target_energy < energy[0]
    target_density[below] = _quadratic_tail(target_energy[below], energy[0], density[0])
    return target_energy, target_density


def _quadratic_tail(energy, anchor_energy, anchor_density):
    """Evaluate the acoustic continuation below the first positive source node.

    Parameters
    ----------
    energy : numpy.ndarray
        Target energies in eV.
    anchor_energy : float
        First source energy in eV.
    anchor_density : float
        Density in inverse eV at the first source energy.

    Returns
    -------
    numpy.ndarray
        Tail density in inverse eV, joined continuously to the first source node.
    """
    return anchor_density * (energy / anchor_energy) ** 2


def to_ncrystal_grid(spectrum: ContinuousSpectrum) -> ContinuousSpectrum:
    """Convert a zero-origin LEAPR spectrum to a positive-origin NCrystal grid.

    Parameters
    ----------
    spectrum : ContinuousSpectrum
        Uniform source spectrum with zero density at zero energy.

    Returns
    -------
    ContinuousSpectrum
        Normalized spectrum on the original positive-energy coordinates.

    Raises
    ------
    InputTransformationError
        If the source grid or its first two ordinates cannot define the acoustic tail.

    Notes
    -----
    NCrystal extrapolates its leading vdos_egrid point with E**2. We
    preserve LEAPR positive-grid values but enforce g(0)=0 and a nonzero
    first positive point. Sub-grid behavior is necessarily represented by
    NCrystal's quadratic tail, which need not exactly reproduce LEAPR's
    native interpolation near zero.
    """
    energy, density = spectrum.energy_ev, spectrum.density_per_ev
    _validate_leapr_origin_and_spacing(energy)
    _validate_ncrystal_tail_anchor(density)

    # Keep the first positive ordinate as the anchor for NCrystal extrapolation.
    return ContinuousSpectrum(energy[1:], density[1:])


def _validate_leapr_origin_and_spacing(energy):
    """Require a uniform LEAPR grid starting at zero.

    Parameters
    ----------
    energy : numpy.ndarray
        Source energy grid in eV.

    Raises
    ------
    InputTransformationError
        If the grid starts above zero or its spacing is nonuniform.
    """
    if not np.isclose(energy[0], 0.0, atol=1e-12):
        raise InputTransformationError('LEAPR spectrum must begin at zero energy.')
    step = float(energy[1] - energy[0])
    if not np.allclose(np.diff(energy), step, rtol=1e-6, atol=1e-11):
        raise InputTransformationError('LEAPR spectrum must have a uniform grid.')


def _validate_ncrystal_tail_anchor(density):
    """Check the first two LEAPR ordinates for NCrystal's acoustic tail.

    Parameters
    ----------
    density : numpy.ndarray
        Source density in inverse eV on the zero-origin LEAPR grid.

    Raises
    ------
    InputTransformationError
        If the origin density exceeds the zero tolerance or the first
        positive-energy density cannot anchor the tail.
    """
    if not np.isclose(density[0], 0.0, atol=1e-12 * density.max()):
        raise InputTransformationError('NCrystal quadratic tail requires g(0)=0.')
    if density[1] <= 0:
        raise InputTransformationError('Cannot infer positive-energy NCrystal quadratic tail from g(delta)=0.')

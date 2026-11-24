"""Project a sampled full VDOS onto LEAPR's continuum and isolated modes."""

from copy import deepcopy

import numpy as np

from input_output.leapr.exceptions import LeaprFormatError
from input_output.leapr.vibrational import tabulate_vibrational_block


def _is_reconstructed_reference(source, spectrum, prepared):
    """Identify the unchanged source spectrum on its natural or supplied grid.

    Parameters
    ----------
    source : dict
        Original LEAPR temperature block.
    spectrum : PhononSpectra
        Candidate vibrational spectrum.
    prepared : dict or None
        Previously reconstructed source block, if available.

    Returns
    -------
    bool
        True when the candidate matches the reconstructed reference.
    """
    energy = spectrum.energy_grid
    if prepared is None:
        if not np.allclose(np.diff(energy), energy[1], rtol=1e-8, atol=1e-12):
            return False
        try:
            prepared = tabulate_vibrational_block(source, energy)
        except ValueError:
            return False
    grid = np.arange(prepared['ni']) * prepared['delta']
    return (
        len(grid) == len(energy)
        and np.allclose(grid, energy, rtol=1e-12, atol=1e-15)
        and np.allclose(prepared['phonon_spectrum'], spectrum.density_values,
                        rtol=1e-12, atol=1e-12)
    )


def _region_moments(energy, density, lower, upper):
    """Integrate the area and first energy moment on one interval.

    Parameters
    ----------
    energy, density : numpy.ndarray
        Energy coordinates in eV and their sampled density.
    lower, upper : float
        Regional integration bounds in eV.

    Returns
    -------
    tuple of float
        Area, first energy moment, and maximum density.
    """
    internal = energy[(energy > lower) & (energy < upper)]
    x = np.r_[lower, internal, upper]
    y = np.interp(x, energy, density)
    area = float(np.trapezoid(y, x))
    # E times a linear density is quadratic on each interval.
    moment = float(np.sum(np.diff(x) * (
        (2 * x[:-1] + x[1:]) * y[:-1] + (x[:-1] + 2 * x[1:]) * y[1:]
    )) / 6)
    return area, moment, float(np.max(y))


def _mode_intervals(spectrum, continuum_end, centers):
    """Choose integration intervals for the continuum and discrete modes.

    Parameters
    ----------
    spectrum : PhononSpectra
        Sampled full vibrational spectrum.
    continuum_end : float
        Upper native continuum energy in eV.
    centers : numpy.ndarray
        Active mode centers in increasing energy order.

    Returns
    -------
    tuple of float and list of tuple
        Continuum cut energy and regional moments in source energy order.

    Raises
    ------
    LeaprFormatError
        If the sampled spectrum cannot be separated at the source gaps.
    """
    energy = spectrum.energy_grid
    density = spectrum.density_values
    cuts = (np.r_[continuum_end, centers[:-1]] + centers) / 2
    if continuum_end >= centers[0] or centers[-1] >= energy[-1]:
        raise LeaprFormatError(
            'Native LEAPR export requires separated modes and a complete sampled spectrum. '
            'For overlapping modes use export_full_spectra=True.'
        )
    edges = np.r_[0., cuts, energy[-1]]
    regions = [_region_moments(energy, density, a, b) for a, b in zip(edges[:-1], edges[1:])]
    cut_density = np.interp(cuts, energy, density)
    for index, height in enumerate(cut_density):
        neighboring_peak = min(regions[index][2], regions[index + 1][2])
        if height > 1e-4 * neighboring_peak:
            raise LeaprFormatError(
                'Sampled LEAPR continuum and modes cannot be separated at their source gaps. '
                'Use export_full_spectra=True for this sample.'
            )
    return cuts[0], regions


def _recover_mode_moments(centers, weights, order, regions, total_area, vibrational_weight):
    """Recover discrete mode centroids and weights from regional moments.

    Parameters
    ----------
    centers, weights : numpy.ndarray
        Source energies and weights of all modes.
    order : numpy.ndarray
        Active mode indices sorted by energy.
    regions : sequence of tuple
        Area, first moment, and maximum density for each active region.
    total_area, vibrational_weight : float
        Sample spectrum integral and source vibrational weight.

    Returns
    -------
    tuple of numpy.ndarray
        Recovered mode energies and weights in source order.

    Raises
    ------
    LeaprFormatError
        If an active region has no positive area.
    """
    recovered_weights = np.zeros_like(weights)
    recovered_centers = centers.copy()

    # Recover each mode's weight from its area and its energy from its centroid.
    for mode, (area, moment, _) in zip(order, regions):
        if area <= 0:
            raise LeaprFormatError('A sampled LEAPR mode has no weight in its source region.')
        recovered_weights[mode] = area / total_area * vibrational_weight
        recovered_centers[mode] = moment / area

    return recovered_centers, recovered_weights


def _project_native_continuum(source, spectrum, native_grid, continuum_cut, continuum_area):
    """Project the sampled continuum to LEAPR's native grid and normalize it.

    Parameters
    ----------
    source : dict
        Original LEAPR block supplying the fallback continuum.
    spectrum : PhononSpectra
        Sampled full vibrational curve.
    native_grid : numpy.ndarray
        Original continuum grid in eV.
    continuum_cut, continuum_area : float
        Upper continuum energy in eV and its integral.

    Returns
    -------
    numpy.ndarray
        Unit-area continuum ordinates on the native grid, or source values when its weight is zero.

    Raises
    ------
    LeaprFormatError
        If a positive continuum area cannot be resolved on the native grid.
    """
    energy = spectrum.energy_grid
    continuum = np.interp(native_grid, energy, spectrum.density_values, right=0.)
    continuum[native_grid > continuum_cut] = 0.

    # LEAPR represents continuum shape and physical weight separately. Normalize the
    # grid values separately from the component weights.
    native_area = np.trapezoid(continuum, native_grid)
    if continuum_area == 0:
        continuum = deepcopy(source['phonon_spectrum'])
    elif native_area > 0:
        continuum /= native_area
    else:
        raise LeaprFormatError('The sampled LEAPR continuum has no weight on the native grid.')
    return continuum


def prepare_discrete_vibrational_block(source, spectrum, *, prepared=None):
    """Prepare a LEAPR block from a full vibrational spectrum.

    Parameters
    ----------
    source : dict
        Original positive-temperature LEAPR block containing discrete modes.
    spectrum : PhononSpectra
        Full vibrational spectrum on a zero-origin energy grid in eV.
    prepared : dict or None, optional
        Precomputed source reconstruction used to recognize the reference.

    Returns
    -------
    dict
        Independent LEAPR block on the original continuum grid. The reference
        retains its original cards. Samples use recovered continuum and
        discrete-mode contributions.

    Raises
    ------
    LeaprFormatError
        If the spectrum does not extend beyond every active source mode
        or the sampled continuum and modes cannot be separated.
    """
    centers = np.asarray(source["oscillator_energies"], dtype=float)
    weights = np.asarray(source["oscillator_weights"], dtype=float)
    active = np.flatnonzero(weights > 0)
    native_grid = np.arange(source["ni"]) * source["delta"]

    if active.size and spectrum.energy_grid[-1] <= np.max(centers[active]):
        raise LeaprFormatError(
            "Expected a full vibrational spectrum including discrete modes."
        )

    # Keep the exact source cards when the full spectrum is the reference.
    if _is_reconstructed_reference(source, spectrum, prepared):
        return deepcopy(source)

    mode_order = active[np.argsort(centers[active])]
    return _project_sampled_block(
        source, spectrum, native_grid, centers, weights, mode_order
    )


def _project_sampled_block(
    source, spectrum, native_grid, centers, weights, mode_order
):
    """Recover LEAPR continuum and oscillator cards from a sampled spectrum.

    Parameters
    ----------
    source : dict
        Original positive-temperature LEAPR block.
    spectrum : PhononSpectra
        Full sampled vibrational spectrum.
    native_grid : numpy.ndarray
        Original continuum energy grid in eV.
    centers, weights : numpy.ndarray
        Source oscillator energies in eV and dimensionless weights.
    mode_order : numpy.ndarray
        Active oscillator indices sorted by energy.

    Returns
    -------
    dict
        Independent block with updated vibrational cards. The oscillator
        count and translational contribution are preserved.

    Notes
    -----
    The integrated area fractions redistribute the original vibrational
    weight among the continuum and discrete oscillators.
    """
    continuum_cut, continuum_area, mode_regions = _partition_sampled_spectrum(
        spectrum, native_grid[-1], centers[mode_order]
    )
    total_area = spectrum.integral
    vibrational_weight = float(source["tbeta"] + weights.sum())

    recovered_centers, recovered_weights = _recover_mode_moments(
        centers, weights, mode_order, mode_regions, total_area, vibrational_weight
    )
    continuum = _project_native_continuum(
        source, spectrum, native_grid, continuum_cut, continuum_area
    )

    block = deepcopy(source)
    block["phonon_spectrum"] = continuum
    block["tbeta"] = continuum_area / total_area * vibrational_weight
    block["oscillator_energies"] = recovered_centers
    block["oscillator_weights"] = recovered_weights
    return block


def _partition_sampled_spectrum(spectrum, native_continuum_end, active_centers):
    """Identify the continuum and discrete-mode integration regions.

    Parameters
    ----------
    spectrum : PhononSpectra
        Full sampled vibrational spectrum.
    native_continuum_end : float
        Upper endpoint of the original continuum grid in eV.
    active_centers : numpy.ndarray
        Active source oscillator energies in eV, in ascending order.

    Returns
    -------
    continuum_cut : float
        Upper integration bound for the continuum in eV.
    continuum_area : float
        Integral of the sampled density over the continuum region.
    mode_regions : list of tuple
        Area, first energy moment, and peak density for each active mode,
        in ascending energy order.

    Raises
    ------
    LeaprFormatError
        If the modes cannot be separated at the source gaps.
    """
    if not active_centers.size:
        return spectrum.energy_grid[-1], spectrum.integral, []

    # The first region is the continuum, followed by the sorted active modes.
    continuum_cut, regions = _mode_intervals(
        spectrum, native_continuum_end, active_centers
    )
    return continuum_cut, regions[0][0], regions[1:]

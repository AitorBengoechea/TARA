"""Project a sampled full VDOS onto LEAPR's continuum and isolated modes."""

from copy import deepcopy

import numpy as np

from input_output.leapr.exceptions import LeaprFormatError
from input_output.leapr.vibrational import tabulate_vibrational_block


def _is_reconstructed_reference(source, spectrum, prepared):
    """Recognize an unchanged reconstruction, including supported custom grids."""
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
    """Integrate area and the first energy moment of a piecewise-linear VDOS."""
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
    """Choose fixed interval boundaries in the gaps between source components."""
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


def prepare_discrete_vibrational_block(source, spectrum, *, prepared=None):
    """Recover an isolated continuum and discrete mode moments for LEAPR.

    Parameters
    ----------
    source : dict
        Original positive-temperature LEAPR block containing discrete modes.
    spectrum : PhononSpectra
        Sampled full spectrum with a zero-origin energy grid in eV.
    prepared : dict or None
        Optional full reconstruction of the source used to recognize the
        unperturbed reference.

    Returns
    -------
    dict
        Independent LEAPR block using the original continuum grid and mode
        count. The unperturbed reconstruction recovers the original cards.
        For a sample, the continuum table is normalized on the native grid.
        Its coefficient and each mode weight come from the full sampled VDOS.

    Notes
    -----
    This projection assumes isolated modes that remain in their source
    regions. Midpoints in the gaps define integration windows. Gaussian
    shape and any tail outside the native continuum grid are not encoded.
    """
    centers = np.asarray(source['oscillator_energies'], dtype=float)
    weights = np.asarray(source['oscillator_weights'], dtype=float)
    active = np.flatnonzero(weights > 0)
    native_grid = np.arange(source['ni']) * source['delta']
    energy = spectrum.energy_grid
    if active.size and energy[-1] <= np.max(centers[active]):
        raise LeaprFormatError('Expected a full vibrational spectrum including discrete modes.')
    # Preserve the original LEAPR cards when the spectrum represents the reference.
    if _is_reconstructed_reference(source, spectrum, prepared):
        return deepcopy(source)

    order = active[np.argsort(centers[active])]
    vibrational_weight = float(source['tbeta'] + weights.sum())
    if order.size:
        continuum_cut, regions = _mode_intervals(spectrum, native_grid[-1], centers[order])
        continuum_area = regions[0][0]
    else:
        continuum_cut, continuum_area = energy[-1], spectrum.integral
        regions = []
    total_area = spectrum.integral
    recovered_weights = np.zeros_like(weights)
    recovered_centers = centers.copy()
    # Recover each mode's weight from its area and its energy from its centroid.
    for mode, (area, moment, _) in zip(order, regions[1:]):
        if area <= 0:
            raise LeaprFormatError('A sampled LEAPR mode has no weight in its source region.')
        recovered_weights[mode] = area / total_area * vibrational_weight
        recovered_centers[mode] = moment / area

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
    return {**deepcopy(source),
            'phonon_spectrum': continuum,
            'tbeta': continuum_area / total_area * vibrational_weight,
            'oscillator_energies': recovered_centers,
            'oscillator_weights': recovered_weights}

"""Generic, backend-specific physical-model to NCrystal solid-like VDOS adapter.

The continuum and unchanged oscillators are weighted and renormalised here;
the Liquids plugin independently handles translational diffusion. No material
name, species or liquid model is hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class VDOSRepresentationDiagnostics:
    """Difference from TARA's previous, NCrystal-incompatible VDOS grid.

    Parameters
    ----------
    original_min_ev : float
        First energy of the baseline grid in eV.
    original_step_ev : float
        Step size used by the baseline representation in eV.
    regular_min_ev : float
        First energy of the regularized grid in eV.
    regular_step_ev : float
        Regular grid spacing in eV.
    shape_l1_change : float
        Integral of the absolute difference between the normalized VDOS
        shapes.
    displacement_moment_relative_change : float
        Relative change in the VDOS displacement-moment diagnostic.

    Notes
    -----
    The baseline below is a *hypothetical* nonregular representation, not a
    NCrystal-valid scattering model. A large displacement-moment difference
    flags sensitivity and requires independent physical validation.
    """

    # Baseline and regularized energy grids.
    original_min_ev: float
    original_step_ev: float
    regular_min_ev: float
    regular_step_ev: float

    # Changes to the spectrum and displacement moment.
    shape_l1_change: float
    displacement_moment_relative_change: float

    def as_dict(self):
        """Return the representation diagnostics as a plain dictionary.

        Returns
        -------
        dict
            Named numerical diagnostics for serialization.
        """
        return asdict(self)


def _tabulated_displacement_integral(grid, vals, kt, nodes, weights) -> float:
    """Integrate the displacement moment over the tabulated energy intervals.

    Parameters
    ----------
    grid, vals : numpy.ndarray
        Source energy grid in eV and vibrational density.
    kt : float
        Thermal energy in eV.
    nodes, weights : numpy.ndarray
        Gauss–Legendre quadrature rule on [-1, 1].

    Returns
    -------
    float
        Unnormalized contribution of the tabulated spectrum to the displacement moment.
    """
    # Use 12-point Gauss--Legendre quadrature separately within each
    # interval, where the density is treated as piecewise linear.
    mid = .5 * (grid[:-1] + grid[1:])
    half = .5 * np.diff(grid)
    points = mid[:, None] + half[:, None] * nodes
    interpolated = (vals[:-1, None] * (1 - nodes) * .5 +
                    vals[1:, None] * (1 + nodes) * .5)

    # The coth(E / 2kT) factor appears as 1/tanh(E / 2kT).
    # The overall atomic prefactor cancels in the relative diagnostic.
    integral_high = float(np.sum(half * (interpolated /
                             (points * np.tanh(points / (2 * kt))) @ weights)))
    return integral_high


def displacement_moment(energy_ev, density, temperature_k: float) -> float:
    """Relative mean-square-displacement VDOS moment with quadratic E->0 tail.

    This uses g(E) coth(E/(2 kT))/E up to a common atom-dependent factor,
    piecewise-linear density interpolation and Gaussian quadrature. It is a
    *diagnostic*; not a quantitative liquid-scattering validation.

    Parameters
    ----------
    energy_ev : object
        VDOS energy grid in electronvolts.
    density : object
        VDOS density evaluated on the energy grid.
    temperature_k : float
        Physical temperature in kelvin.

    Returns
    -------
    float
        Relative low-energy VDOS displacement-moment diagnostic.
    """
    # Work with numerical arrays without mutating the caller-owned VDOS.
    grid = np.asarray(energy_ev, dtype=float)
    vals = np.asarray(density, dtype=float)

    # The piecewise-linear quadrature requires a finite, increasing,
    # strictly positive energy grid and a positive physical temperature.
    if (grid.ndim != 1 or vals.shape != grid.shape or grid.size < 3
            or not np.isfinite(grid).all() or not np.isfinite(vals).all()
            or np.any(np.diff(grid) <= 0) or grid[0] <= 0
            or temperature_k <= 0 or not np.isfinite(temperature_k)):
        raise ValueError("Invalid VDOS data or temperature for displacement moment.")

    # Express k_B T in eV to match the tabulated energy grid.
    kt = 8.617333262145e-5 * temperature_k
    nodes, weights = np.polynomial.legendre.leggauss(12)
    integral_high = _tabulated_displacement_integral(grid, vals, kt, nodes, weights)

    # Extend the spectrum as E^2 below its first positive point, then
    # integrate this implied low-energy tail separately.
    lowpoints = grid[0] * (nodes + 1) * .5
    integral_low = float(vals[0] / grid[0]**2 * grid[0] * .5 *
                         np.sum(weights * lowpoints /
                                np.tanh(lowpoints / (2 * kt))))

    # Include the same E^2 tail when normalizing the displacement moment.
    area = float(np.trapezoid(vals, grid) + vals[0] * grid[0] / 3)
    if area <= 0:
        raise ValueError("Cannot compute a moment for zero-area VDOS.")
    return (integral_low + integral_high) / area


def _solidlike_density(material, grid, source_grid, source_density, sigma):
    """Build a normalized continuum+oscillator mix without mutating the source.

    Parameters
    ----------
    material : object
        Liquid material model with continuous and oscillator weights.
    grid : object
        Target VDOS energy grid in electronvolts.
    source_grid : object
        Original continuous VDOS energy grid in electronvolts.
    source_density : object
        Original continuous VDOS density.
    sigma : object
        Gaussian broadening width for fixed oscillators in electronvolts.

    Returns
    -------
    object
        Normalized continuum-plus-oscillator VDOS density.
    """
    dyn = material.dynamics

    # Map the sampled continuum to NCrystal's common energy grid,
    # taking the density to vanish beyond its tabulated support.
    continuous = np.interp(grid, source_grid, source_density, left=0, right=0)

    # Normalize the discretized continuum before multiplying by its
    # independently specified physical mixture weight.
    norm = float(np.trapezoid(continuous, grid))
    if norm <= 0:
        raise ValueError("Cannot integrate the sampled continuum on the VDOS grid.")
    mixture = dyn.continuous_weight * continuous / norm

    # Approximate each fixed discrete oscillator with a Gaussian.
    # Normalize each peak on the *finite target grid* before weighting.
    for oscillator in dyn.oscillators:
        peak = np.exp(-0.5 * ((grid - oscillator.energy_ev) / sigma)**2)
        area = float(np.trapezoid(peak, grid))
        if area <= 0:
            raise ValueError("Oscillator outside NCrystal VDOS grid.")
        mixture += oscillator.weight * peak / area

    # Translation is deliberately excluded: the NCrystal Liquids
    # plugin handles its contribution outside this solid-like VDOS.
    solid_weight = dyn.continuous_weight + sum(o.weight for o in dyn.oscillators)
    return mixture / solid_weight


def _validate_solidlike_inputs(material, continuous_spectrum, oscillator_sigma_ev, n_grid):
    """Validate the sampled continuum and grid controls without mutating them.

    Parameters
    ----------
    material : object
        Physical liquid model with continuous and oscillator weights.
    continuous_spectrum : object
        Source continuous VDOS with grid and density attributes.
    oscillator_sigma_ev : float
        Gaussian broadening width for discrete oscillators in eV.
    n_grid : int
        Number of regular VDOS energy grid points.

    Returns
    -------
    tuple
        Source energy grid, source density, and target maximum energy in eV.
    """
    dyn = material.dynamics

    # A solid-like VDOS is meaningful only when the model contains
    # positive continuum/oscillator weight and valid grid controls.
    solid_weight = dyn.continuous_weight + sum(o.weight for o in dyn.oscillators)
    if (solid_weight <= 0 or dyn.continuous_weight < 0 or
            oscillator_sigma_ev <= 0 or not np.isfinite(oscillator_sigma_ev)
            or type(n_grid) is not int or n_grid < 50):
        raise ValueError("Invalid solid-like component or VDOS grid controls.")

    # Reject invalid source arrays before any interpolation or folding;
    # negative densities and duplicate energies cannot be repaired here.
    source_grid = np.asarray(continuous_spectrum.energy_grid, dtype=float)
    source_density = np.asarray(continuous_spectrum.density_values, dtype=float)
    if (source_grid.ndim != 1 or source_density.shape != source_grid.shape
            or source_grid.size < 3 or not np.isfinite(source_grid).all()
            or not np.isfinite(source_density).all() or np.any(source_density < 0)
            or np.any(np.diff(source_grid) <= 0)):
        raise ValueError("Invalid sampled continuous VDOS.")

    # The Core continuum is already unit-normalized. Preserve that
    # contract instead of silently repairing a malformed spectrum.
    if not np.isclose(np.trapezoid(source_density, source_grid), 1, atol=1e-5):
        raise ValueError("The sampled continuous VDOS must have unit area.")

    # Include an eight-sigma upper margin so Gaussian oscillator
    # tails remain represented on the regular NCrystal grid.
    emax = max(float(source_grid[-1]), max(
        (osc.energy_ev + 8 * oscillator_sigma_ev for osc in dyn.oscillators),
        default=0))
    if not np.isfinite(emax) or emax <= 0:
        raise ValueError("Invalid maximum vibrational energy.")
    return source_grid, source_density, emax


def _aligned_solidlike_grid(emax: float, n_grid: int) -> np.ndarray:
    """Build an NCrystal-aligned positive energy grid.

    Parameters
    ----------
    emax : float
        Maximum energy in eV.
    n_grid : int
        Requested number of energy grid points.

    Returns
    -------
    numpy.ndarray
        Uniform energy grid satisfying ``E_min / delta_E == 1``.
    """
    # Anchor E_min to delta_E rather than using an unrelated small
    # positive offset; this satisfies the NCrystal grid constraint.
    step = emax / n_grid
    grid = step * np.arange(1, n_grid + 1, dtype=float)

    # NCrystal rejects VDOS grids starting below this energy threshold.
    if grid[0] < 1e-5:
        raise ValueError("VDOS grid spacing below NCrystal's minimum of 1e-5 eV.")
    return grid


def _vdos_representation_diagnostics(
    material, grid, density, source_grid, source_density,
    oscillator_sigma_ev, emax, n_grid, temperature_k,
):
    """Compare aligned VDOS with the historical nonregular reference grid.

    This is a sensitivity check, not a physical validation certificate.

    Parameters
    ----------
    material : object
        Liquid physical model.
    grid, density : numpy.ndarray
        NCrystal-aligned VDOS energy grid and density.
    source_grid, source_density : numpy.ndarray
        Original continuous spectrum input arrays.
    oscillator_sigma_ev : float
        Oscillator smoothing width in eV.
    emax : float
        Maximum target energy in eV.
    n_grid : int
        Number of energy points.
    temperature_k : float or None
        Temperature override for the low-energy moment diagnostic.

    Returns
    -------
    VDOSRepresentationDiagnostics
        Shape and displacement-moment differences versus the historical grid.
    """
    # Reconstruct the *historical*, nonaligned grid only as a reference
    # for sensitivity diagnostics, never as a production NCrystal grid.
    previous_grid = np.linspace(1e-5, emax, n_grid)
    previous_density = _solidlike_density(
        material, previous_grid, source_grid, source_density, oscillator_sigma_ev,
    )

    # Compare both representations on their union grid so the shape
    # metric reflects the same energy support and interpolation rule.
    common_grid = np.union1d(previous_grid, grid)
    old = np.interp(common_grid, previous_grid, previous_density, left=0, right=0)
    new = np.interp(common_grid, grid, density, left=0, right=0)

    # Normalize the integrated absolute shape difference to the
    # historical spectrum area; this is not a physics validation test.
    old_area = float(np.trapezoid(previous_density, previous_grid))
    l1 = float(np.trapezoid(np.abs(old - new), common_grid) / old_area)

    # Use the explicitly requested temperature when provided; otherwise
    # diagnose the model at its own stored physical temperature.
    temp = temperature_k if temperature_k is not None else getattr(material, 'temperature_k', None)
    if temp is None:
        raise ValueError("temperature_k is required to diagnose low-energy moments.")

    # The relative displacement-moment change isolates sensitivity
    # to the VDOS representation, not uncertainty in liquid dynamics.
    original_moment = displacement_moment(previous_grid, previous_density, temp)
    regular_moment = displacement_moment(grid, density, temp)
    return VDOSRepresentationDiagnostics(
        original_min_ev=float(previous_grid[0]),
        original_step_ev=float(previous_grid[1] - previous_grid[0]),
        regular_min_ev=float(grid[0]),
        regular_step_ev=float(grid[1] - grid[0]),
        shape_l1_change=l1,
        displacement_moment_relative_change=float(regular_moment / original_moment - 1),
    )


def fold_solidlike_vdos(material, continuous_spectrum, *,
                        oscillator_sigma_ev: float = 0.001,
                        n_grid: int = 1201,
                        return_diagnostics: bool = False,
                        temperature_k: float | None = None):
    """Fold sampled continuum and fixed oscillators into aligned NCrystal VDOS.

    Translation is not perturbed here; the external Liquids plugin handles its
    convolution separately. The returned solid-like VDOS has unit area.

    Parameters
    ----------
    material : object
        Liquid material model with fixed continuum and oscillator weights.
    continuous_spectrum : object
        Sampled continuous spectrum with energy and density arrays.
    oscillator_sigma_ev : float, default=0.001
        Width of the fixed Gaussian oscillator representation in eV.
    n_grid : int, default=1201
        Number of points on the aligned energy grid.
    return_diagnostics : bool, default=False
        Whether to return shape and displacement-moment sensitivity metrics.
    temperature_k : float or None, optional
        Override temperature for numerical diagnostics.

    Returns
    -------
    tuple
        ``(energy_grid, density)`` or, if requested,
        ``(energy_grid, density, diagnostics)``.
    """
    # Validate inputs before constructing an NCrystal-compatible
    # positive regular grid; leave the sampled continuum unchanged.
    source_grid, source_density, emax = _validate_solidlike_inputs(
        material, continuous_spectrum, oscillator_sigma_ev, n_grid,
    )
    grid = _aligned_solidlike_grid(emax, n_grid)

    # Fold only the solid-like continuum and fixed oscillators; the
    # translational scattering term is handled independently.
    density = _solidlike_density(
        material, grid, source_grid, source_density, oscillator_sigma_ev,
    )

    # The historical-grid comparison is optional and must not affect
    # the production energy grid or density returned to the caller.
    if not return_diagnostics:
        return grid, density
    detail = _vdos_representation_diagnostics(
        material, grid, density, source_grid, source_density,
        oscillator_sigma_ev, emax, n_grid, temperature_k,
    )
    return grid, density, detail

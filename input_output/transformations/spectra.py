"""Explicit acoustic-tail representations at native file boundaries."""
import numpy as np
from input_output.transformations.models import ContinuousSpectrum, InputTransformationError


def to_leapr_grid(spectrum: ContinuousSpectrum) -> ContinuousSpectrum:
    """Convert an NCrystal-like equally spaced VDOS to LEAPR E_i=i*delta.

    When the positive NCrystal starting energy is an integral number of source
    steps from zero, preserve all source ordinates exactly (up to one scalar).
    Otherwise resample linearly on the origin-anchored grid, and preserve the
    original high-energy support without extrapolation beyond its last node.
    Below the first NCrystal node use the E**2 acoustic continuation.
    """
    e = spectrum.energy_ev
    y = spectrum.density_per_ev
    if e[0] == 0:
        raise InputTransformationError('NCrystal VDOS must have a positive first energy.')
    steps = np.diff(e)
    delta = float(np.mean(steps))
    if not np.allclose(steps, delta, rtol=1e-6, atol=1e-11):
        raise InputTransformationError('NCrystal source VDOS must be on a regular grid.')
    if y[0] <= 0:
        raise InputTransformationError('Cannot construct an E**2 tail from zero initial NCrystal density.')
    nstart = int(round(e[0] / delta))
    # Never silently interpolate a noncommensurate grid as if it were exact.
    aligned = nstart >= 1 and np.isclose(e[0], nstart * delta, rtol=1e-6, atol=1e-8)
    if aligned:
        target_e = np.arange(nstart + len(e), dtype=float) * delta
        target_g = np.r_[y[0] * (target_e[:nstart] / e[0])**2, y]
    else:
        # Keep the last LEAPR node at or below the source endpoint when the grids do not
        # align.
        n = int(np.floor(e[-1] / delta + 1e-10)) + 1
        target_e = np.arange(n, dtype=float) * delta
        target_g = np.interp(target_e, e, y)
        below = target_e < e[0]
        target_g[below] = y[0] * (target_e[below] / e[0])**2
    return ContinuousSpectrum(target_e, target_g)


def to_ncrystal_grid(spectrum: ContinuousSpectrum) -> ContinuousSpectrum:
    """Convert LEAPR E_i=i*delta to positive-origin NCrystal VDOS.

    NCrystal extrapolates its leading vdos_egrid point with E**2. We
    preserve LEAPR positive-grid values but enforce g(0)=0 and a nonzero
    first positive point. Sub-grid behavior is necessarily represented by
    NCrystal's quadratic tail, which need not exactly reproduce LEAPR's
    native interpolation near zero.
    """
    e, g = spectrum.energy_ev, spectrum.density_per_ev
    if not np.isclose(e[0], 0.0, atol=1e-12):
        raise InputTransformationError('LEAPR spectrum must begin at zero energy.')
    step = float(e[1] - e[0])
    if not np.allclose(np.diff(e), step, rtol=1e-6, atol=1e-11):
        raise InputTransformationError('LEAPR spectrum must have a uniform grid.')
    if not np.isclose(g[0], 0.0, atol=1e-12 * g.max()):
        raise InputTransformationError('NCrystal quadratic tail requires g(0)=0.')
    if g[1] <= 0:
        raise InputTransformationError('Cannot infer positive-energy NCrystal quadratic tail from g(delta)=0.')
    # No hidden acoustic approximation: after dropping the zero point, the
    # first measured point is retained as the extrapolation anchor.
    return ContinuousSpectrum(e[1:], g[1:])

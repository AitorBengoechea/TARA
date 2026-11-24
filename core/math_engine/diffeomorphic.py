"""
Diffeomorphic spatial warping mathematics.
"""
import numpy as np
from typing import List, Tuple
from scipy.signal.windows import tukey
from numba import njit, prange

def _setup_padded_grid(x_orig: np.ndarray, max_abs_pad: float) -> Tuple[np.ndarray, int]:
    """Create an extrapolated padded x-axis to safeguard against edge shifting artifacts.

    Parameters
    ----------
    x_orig : np.ndarray
        Original coordinate grid prior to padding.
    max_abs_pad : float
        Maximum permitted abs pad.

    Returns
    -------
    Tuple[np.ndarray, int]
        Extended coordinate grid and number of padding points added per side.
    """
    if max_abs_pad <= 0: 
        return x_orig, 0
        
    # Examine adjacent grid differences to enforce ordering or locate transitions.
    dx = np.mean(np.diff(x_orig))
    pad_len = int(np.ceil((max_abs_pad / dx) * 1.2))
    
    pad_left_x = x_orig[0] - np.arange(pad_len, 0, -1) * dx
    pad_right_x = x_orig[-1] + np.arange(1, pad_len + 1) * dx
    
    # Assemble the grid segments in the order required by the reconstruction.
    x_padded = np.concatenate([pad_left_x, x_orig, pad_right_x])
    return x_padded, pad_len

def _single_domain_displacement(x_padded, e_start, e_end, s_val, taper_runway, freeze_origin):
    """Construct a rigid interior shift with smooth tapers on both sides.

    Parameters
    ----------
    x_padded : numpy.ndarray
        Internal energy coordinates in eV.
    e_start, e_end : float
        Domain bounds in eV.
    s_val, taper_runway : float
        Domain shift and taper width in eV.
    freeze_origin : bool
        Anchor the acoustic origin when it lies within the domain.

    Returns
    -------
    numpy.ndarray
        Displacement at each padded coordinate.
    """
    original_start = float(e_start)
    original_end = float(e_end)
    expanded_start = original_start - taper_runway
    expanded_end = original_end + taper_runway

    delta_x = np.zeros_like(x_padded, dtype=float)

    # Left transition: 0 -> s_val outside the requested domain.
    left_taper = (x_padded >= expanded_start) & (x_padded < original_start)
    if np.any(left_taper):
        u = (x_padded[left_taper] - expanded_start) / taper_runway
        delta_x[left_taper] = s_val * 0.5 * (1.0 - np.cos(np.pi * u))

    # Exact translation inside THIS domain only.
    flat_region = (x_padded >= original_start) & (x_padded <= original_end)
    delta_x[flat_region] = s_val

    # Right transition: s_val -> 0 outside the requested domain.
    right_taper = (x_padded > original_end) & (x_padded <= expanded_end)
    if np.any(right_taper):
        u = (expanded_end - x_padded[right_taper]) / taper_runway
        delta_x[right_taper] = s_val * 0.5 * (1.0 - np.cos(np.pi * u))

    # If the acoustic origin is frozen and belongs to this domain, a smooth
    # transition from delta(0)=0 to the requested shift is mathematically
    # required. Pinning only one sample at E=0 would make the map fold for
    # sufficiently negative shifts.
    if freeze_origin and original_start <= 0.0 <= original_end:
        delta_x[x_padded <= 0.0] = 0.0

        origin_transition_end = min(original_end, taper_runway)
        if origin_transition_end > 0.0:
            origin_taper = (x_padded > 0.0) & (x_padded < origin_transition_end)
            if np.any(origin_taper):
                u = x_padded[origin_taper] / origin_transition_end
                delta_x[origin_taper] = (
                    s_val * 0.5 * (1.0 - np.cos(np.pi * u))
                )

            # Only samples beyond the origin transition get the full shift.
            origin_flat = (
                (x_padded >= origin_transition_end)
                & (x_padded <= original_end)
            )
            delta_x[origin_flat] = s_val

        origin_idx = np.isclose(x_padded, 0.0, rtol=0.0, atol=1e-14)
        delta_x[origin_idx] = 0.0

    return delta_x


def _pad_scalar_spectrum(x_grid, y_density, internal_pad):
    """Pad a single spectrum for the internal deformation calculation.

    Parameters
    ----------
    x_grid, y_density : numpy.ndarray
        Physical grid in eV and corresponding density.
    internal_pad : float
        Requested padding in eV per side.

    Returns
    -------
    tuple of numpy.ndarray
        Extended energy grid and zero-padded density.
    """
    x_padded, pad_len = _setup_padded_grid(x_grid, internal_pad)
    if not pad_len:
        return x_padded, y_density.copy()
    zeros = np.zeros(pad_len, dtype=y_density.dtype)
    return x_padded, np.concatenate([zeros, y_density, zeros])


def _resample_scalar_warp(x_grid, y_density, x_padded, y_padded, x_new, preserve_shape):
    """Interpolate a monotonic scalar deformation onto the physical energy grid.

    Parameters
    ----------
    x_grid, y_density : numpy.ndarray
        Physical grid and original density used for area correction.
    x_padded, y_padded : numpy.ndarray
        Internal grid and zero-padded density.
    x_new : numpy.ndarray
        Strictly increasing deformed grid.
    preserve_shape : bool
        Skip the Jacobian and area corrections when True.

    Returns
    -------
    numpy.ndarray
        Reconstructed density on the physical grid.
    """
    if preserve_shape:
        # No Jacobian scaling: amplitudes and local peak widths are transported
        # unchanged through the flat part of the displacement field.
        y_warped = y_padded
    else:
        jacobian = np.gradient(x_new, x_padded)
        jacobian = np.clip(jacobian, 0.5, 2.0)
        y_warped = y_padded / jacobian

    # The caller's grid is always physical (E >= 0). Internal negative padded
    # coordinates are only used to construct the deformation smoothly.
    y_resampled = np.interp(
        x_grid,
        x_new,
        y_warped,
        left=0.0,
        right=0.0,
    )

    if not preserve_shape:
        # Integrate against the physical coordinate grid, accounting for nonuniform spacing.
        area_old = np.trapezoid(y_density, x_grid)
        # Integrate against the physical coordinate grid, accounting for nonuniform spacing.
        area_new = np.trapezoid(y_resampled, x_grid)
        if area_new > 1e-12:
            y_resampled *= area_old / area_new

    return y_resampled


def _warp_single_domain(x_grid: np.ndarray, y_density: np.ndarray,
                        e_start: float, e_end: float, s_val: float,
                        freeze_origin: bool, preserve_shape: bool) -> np.ndarray:
    """Warp one energy domain while keeping its interior as a pure translation.

    The original interval [e_start, e_end] is the flat part of the displacement
    field. Smooth transition regions are placed *outside* that interval whenever
    possible. Internal padding may contain negative coordinates, but those
    coordinates are never returned to the caller.

    Parameters
    ----------
    x_grid : np.ndarray
        Input coordinate grid.
    y_density : np.ndarray
        Spectral density sampled on the input grid.
    e_start : float
        Lower energy boundary of the affected domain.
    e_end : float
        Upper energy boundary of the affected domain.
    s_val : float
        Displacement assigned to the current domain.
    freeze_origin : bool
        Whether the zero-energy boundary remains anchored.
    preserve_shape : bool
        Keep the warped density profile without Jacobian or area scaling.

    Returns
    -------
    np.ndarray
        Warped density sampled at the original energy coordinates.

    Raises
    ------
    ValueError
        If the warp produces a nonmonotonic coordinate map.
    """
    max_abs_shift = abs(float(s_val))
    if max_abs_shift < 1e-12:
        return y_density.copy()

    if e_end <= e_start:
        return y_density.copy()

    # Keep |d(delta_x)/dx| comfortably below one so x_new remains monotonic.
    max_distortion = 0.5
    taper_runway = (np.pi * max_abs_shift) / (2.0 * max_distortion)

    x_padded, y_padded = _pad_scalar_spectrum(
        x_grid, y_density, max(max_abs_shift, taper_runway),
    )
    delta_x = _single_domain_displacement(
        x_padded, e_start, e_end, s_val, taper_runway, freeze_origin,
    )
    x_new = x_padded + delta_x

    # Numerical guard: np.interp requires a strictly increasing coordinate map.
    # The taper construction should guarantee this; fail clearly if it does not.
    if np.any(np.diff(x_new) <= 0.0):
        raise ValueError(
            "Spatial warp produced a non-monotonic coordinate map. "
            "Reduce the requested shift or increase the taper runway."
        )

    return _resample_scalar_warp(x_grid, y_density, x_padded, y_padded, x_new, preserve_shape)


def _collect_displacement_events(boundaries, shifts_eV, atol):
    """Convert domain shifts to sorted jumps, merging shared endpoints.

    Parameters
    ----------
    boundaries : sequence of tuple of float
        Lower and upper domain energies in eV.
    shifts_eV : numpy.ndarray
        Rigid shift of each domain in eV.
    atol : float
        Maximum distance at which two event locations are treated as equal.

    Returns
    -------
    list of tuple of float
        Energy coordinates and nonzero changes in displacement.

    Raises
    ------
    ValueError
        If a domain's upper bound does not exceed its lower bound.
    """
    # Represent the ideal piecewise-constant displacement by jump events:
    # +s at a domain start and -s at a domain end.  Shared endpoints combine
    # automatically into the direct jump s_next - s_previous.
    raw_events = []
    for (e_start, e_end), s_val in zip(boundaries, shifts_eV):
        if e_end <= e_start:
            raise ValueError(
                f"Invalid spatial boundary ({e_start}, {e_end}): "
                "end must be greater than start."
            )
        if abs(float(s_val)) < 1e-15:
            continue
        raw_events.append((float(e_start), float(s_val)))
        raw_events.append((float(e_end), -float(s_val)))

    # Merge numerically identical event locations.
    raw_events.sort(key=lambda item: item[0])
    events = []
    for pos, jump in raw_events:
        if events and abs(pos - events[-1][0]) <= atol:
            events[-1] = (events[-1][0], events[-1][1] + jump)
        else:
            events.append((pos, jump))

    return [(position, jump) for position, jump in events if abs(jump) > 1e-15]


def _smooth_step_centered(x, center, width):
    """Raised-cosine 0->1 step centered at center over total width.

    Parameters
    ----------
    x : numpy.ndarray
        Energy coordinates in eV.
    center : float
        Central coordinate of the transition.
    width : float
        Width of the transition interval in eV.

    Returns
    -------
    numpy.ndarray
        Raised-cosine step values between zero and one.
    """
    if width <= 0.0:
        return (x >= center).astype(float)

    left = center - 0.5 * width
    right = center + 0.5 * width
    out = np.zeros_like(x, dtype=float)
    out[x >= right] = 1.0

    mask = (x > left) & (x < right)
    if np.any(mask):
        u = (x[mask] - left) / width
        out[mask] = 0.5 * (1.0 - np.cos(np.pi * u))
    return out

def _smooth_step_right_sided(x, start, width):
    """Raised-cosine 0->1 step beginning at start.

    Used at E=0 when the acoustic origin must remain fixed.

    Parameters
    ----------
    x : numpy.ndarray
        Energy coordinates in eV.
    start : float
        Beginning coordinate of the interval.
    width : float
        Width of the transition interval in eV.

    Returns
    -------
    numpy.ndarray
        Raised-cosine step values between zero and one.
    """
    if width <= 0.0:
        return (x > start).astype(float)

    out = np.zeros_like(x, dtype=float)
    right = start + width
    out[x >= right] = 1.0

    mask = (x > start) & (x < right)
    if np.any(mask):
        u = (x[mask] - start) / width
        out[mask] = 0.5 * (1.0 - np.cos(np.pi * u))
    return out

def _smooth_displacement_field(
    x_padded, events, min_widths, width_scale, freeze_origin, atol,
):
    """Evaluate one candidate displacement field at the given transition widths.

    Parameters
    ----------
    x_padded : numpy.ndarray
        Internal energy grid in eV.
    events : sequence of tuple of float
        Locations and sizes of the jumps in displacement, in eV.
    min_widths : sequence of float
        Initial widths of the raised-cosine transitions in eV.
    width_scale : float
        Common multiplier applied to the initial widths.
    freeze_origin : bool
        Keep the acoustic origin fixed.
    atol : float
        Absolute tolerance in eV used to identify the origin.

    Returns
    -------
    numpy.ndarray
        Candidate displacement on the internal energy grid.
    """
    delta_x = np.zeros_like(x_padded, dtype=float)

    for (pos, jump), base_width in zip(events, min_widths):
        width = base_width * width_scale

        if freeze_origin and abs(pos) <= atol:
            step = _smooth_step_right_sided(x_padded, 0.0, width)
        else:
            step = _smooth_step_centered(x_padded, pos, width)

        delta_x += jump * step

    if freeze_origin:
        # The acoustic origin is a hard physical anchor.
        delta_x[x_padded <= 0.0] = 0.0
        origin_idx = np.isclose(
            x_padded, 0.0, rtol=0.0, atol=max(atol, 1e-14)
        )
        delta_x[origin_idx] = 0.0
    return delta_x


def _build_global_displacement(
    x_padded: np.ndarray,
    boundaries: List[Tuple[float, float]],
    shifts_eV: np.ndarray,
    freeze_origin: bool,
    max_distortion: float = 0.5,
) -> np.ndarray:
    """Construct ONE smooth displacement field for all spatial domains.

    Each requested interval contributes a piecewise-constant displacement.
    At a shared boundary, changes are blended directly from shift_i to
    shift_{i+1}; the field does not return to zero between adjacent domains.

    The smooth transition for a displacement jump ds uses a raised-cosine
    step whose maximum slope is bounded approximately by max_distortion.

    Parameters
    ----------
    x_padded : np.ndarray
        Extended coordinate grid used to avoid edge artifacts.
    boundaries : List[Tuple[float, float]]
        Spectral-domain boundaries used by the transformation.
    shifts_eV : np.ndarray
        Spectral displacement values in eV.
    freeze_origin : bool
        Whether the zero-energy boundary remains anchored.
    max_distortion : float, optional
        Maximum permitted distortion.

    Returns
    -------
    np.ndarray
        Displacement in eV at each padded energy coordinate.

    Raises
    ------
    ValueError
        If the domain bounds are invalid or the shifts cannot be made monotonic.
    """
    if len(boundaries) == 0:
        return np.zeros_like(x_padded, dtype=float)

    # Examine adjacent grid differences to enforce ordering or locate transitions.
    dx = float(np.mean(np.diff(x_padded)))
    atol = max(abs(dx) * 1e-6, 1e-14)

    events = _collect_displacement_events(boundaries, shifts_eV, atol)
    if not events:
        return np.zeros_like(x_padded, dtype=float)

    # Width needed for a raised-cosine jump:
    # max |d(delta)/dx| = |jump| * pi / (2*width).
    min_widths = []
    for pos, jump in events:
        safe_width = (
            np.pi * abs(jump) / (2.0 * max_distortion)
            if abs(jump) > 0.0 else 0.0
        )
        min_widths.append(max(4.0 * abs(dx), 1.05 * safe_width))

    # If nearby transitions overlap, derivative contributions can accumulate.
    # Increase all transition widths together until x + delta is monotonic.
    width_scale = 1.0
    for _ in range(20):
        delta_x = _smooth_displacement_field(
            x_padded, events, min_widths, width_scale, freeze_origin, atol,
        )

        x_new = x_padded + delta_x
        # Examine adjacent grid differences to enforce ordering or locate transitions.
        if np.all(np.diff(x_new) > 0.0):
            return delta_x

        width_scale *= 1.35

    raise ValueError(
        "Could not construct a monotonic global spatial deformation. "
        "The requested neighboring shifts are too abrupt for the available "
        "domain spacing."
    )


def _validate_scalar_warp_inputs(x_grid, y_density, boundaries, shifts) -> None:
    """Validate the grid, spectrum, and shifts for a single warp.

    Parameters
    ----------
    x_grid, y_density : numpy.ndarray
        Input energies in eV and corresponding density.
    boundaries : sequence of tuple of float
        Energy bounds for each domain.
    shifts : numpy.ndarray
        Domain shifts in meV.

    Raises
    ------
    ValueError
        If input dimensions, counts, or energy coordinates are invalid.
    """
    if x_grid.ndim != 1 or y_density.ndim != 1:
        raise ValueError("x_grid and y_density must be one-dimensional.")
    if x_grid.size != y_density.size:
        raise ValueError("x_grid and y_density must have the same length.")
    if len(boundaries) != shifts.size:
        raise ValueError("The number of boundaries must match the number of shifts.")
    if x_grid.size < 2:
        raise ValueError("x_grid must contain at least two points.")
    # Examine adjacent grid differences to enforce ordering or locate transitions.
    if np.any(np.diff(x_grid) <= 0.0):
        raise ValueError("x_grid must be strictly increasing.")
    if np.any(x_grid < 0.0):
        raise ValueError("The input energy grid cannot contain negative energies.")


def _extend_warp_support(x_grid, y_density, boundaries, shifts_eV):
    """Extend the physical right edge for a positive shift of an edge domain.

    Parameters
    ----------
    x_grid, y_density : numpy.ndarray
        Input grid in eV and corresponding density.
    boundaries : sequence of tuple of float
        Sorted domain bounds in eV.
    shifts_eV : numpy.ndarray
        Domain shifts in eV.

    Returns
    -------
    tuple of numpy.ndarray
        Grid and density extended with zero density where needed.
    """
    # Keep the PhononSpectra energy axis physical. Extend only the high-energy
    # side if a positively shifted edge domain needs additional output support.
    if boundaries:
        # Examine adjacent grid differences to enforce ordering or locate transitions.
        dx = float(np.mean(np.diff(x_grid)))
        original_right = float(x_grid[-1])
        edge_tol = 2.0 * abs(dx)

        required_right_extension = 0.0
        for (e_start, e_end), s_val in zip(boundaries, shifts_eV):
            if s_val > 0.0 and e_end >= original_right - edge_tol:
                required_right_extension = max(
                    required_right_extension, float(s_val)
                )

        if required_right_extension > 0.0:
            extra_points = int(
                np.ceil(required_right_extension / abs(dx))
            ) + 2
            extra_x = x_grid[-1] + np.arange(
                1, extra_points + 1
            ) * dx
            # Assemble the grid segments in the order required by the reconstruction.
            x_grid = np.concatenate([x_grid, extra_x])
            # Assemble the grid segments in the order required by the reconstruction.
            y_density = np.concatenate([
                y_density,
                np.zeros(extra_points, dtype=y_density.dtype),
            ])
    return x_grid, y_density


def _sort_scalar_domains(x_grid, boundaries, shifts):
    """Order scalar domains while preserving their associated shifts.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Original energy grid in eV.
    boundaries : sequence of tuple of float
        Energy bounds in eV for the requested domains.
    shifts : numpy.ndarray
        Domain shifts in meV.

    Returns
    -------
    tuple
        Sorted domain bounds and corresponding shifts in eV.
    """
    shifts_eV = shifts / 1000.0
    boundaries = [(float(a), float(b)) for a, b in boundaries]

    # Validate ordering. Overlapping domains would make the requested
    # piecewise displacement ambiguous.
    ordered = sorted(
        zip(boundaries, shifts_eV),
        key=lambda item: item[0][0]
    )
    if ordered:
        boundaries = [item[0] for item in ordered]
        shifts_eV = np.asarray([item[1] for item in ordered], dtype=float)

        # Examine adjacent grid differences to enforce ordering or locate transitions.
        dx = float(np.mean(np.diff(x_grid)))
        tol = 2.0 * abs(dx)

        for i in range(len(boundaries) - 1):
            if boundaries[i][1] > boundaries[i + 1][0] + tol:
                raise ValueError(
                    "Spatial domains overlap. A single diffeomorphic "
                    "displacement is ambiguous for overlapping domains."
                )

    return boundaries, shifts_eV


def _scalar_padding_width(shifts_eV, max_distortion):
    """Determine how much internal space the scalar transitions require.

    Parameters
    ----------
    shifts_eV : numpy.ndarray
        Shifts in domain order and in eV.
    max_distortion : float
        Target upper bound on the displacement slope of an isolated transition.

    Returns
    -------
    float
        Required padding in eV.
    """
    # Internal padding is allowed to extend below zero because it is only a
    # numerical workspace for the deformation. It is never returned.
    max_abs_shift = (
        float(np.max(np.abs(shifts_eV))) if shifts_eV.size else 0.0
    )
    if shifts_eV.size > 1:
        # Examine adjacent grid differences to enforce ordering or locate transitions.
        max_jump = float(np.max(np.abs(np.diff(shifts_eV))))
    else:
        max_jump = max_abs_shift

    max_transition_width = (
        1.05 * np.pi * max(max_abs_shift, max_jump)
        / (2.0 * max_distortion)
        if max(max_abs_shift, max_jump) > 0.0
        else 0.0
    )
    internal_pad = max(max_abs_shift, max_transition_width)

    return internal_pad


def apply_spatial_warp(x_grid: np.ndarray,
                       y_density: np.ndarray,
                       boundaries: List[Tuple[float, float]],
                       shifts: np.ndarray,
                       freeze_origin: bool,
                       preserve_shape: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Apply all spatial-domain shifts through one global diffeomorphic map.

    Unlike the previous sequential implementation, this function:
      1. builds one global displacement field from every domain,
      2. constructs x_new once,
      3. resamples the ORIGINAL spectrum exactly once.

    This avoids cumulative interpolation broadening and repeated deformation
    of features near neighboring domain boundaries.

    Parameters
    ----------
    x_grid : np.ndarray
        Input coordinate grid.
    y_density : np.ndarray
        Spectral density sampled on the input grid.
    boundaries : List[Tuple[float, float]]
        Spectral-domain boundaries used by the transformation.
    shifts : np.ndarray
        Domain shifts in meV.
    freeze_origin : bool
        Whether the zero-energy boundary remains anchored.
    preserve_shape : bool, optional
        Keep the warped density profile without Jacobian or area scaling.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Physical output grid in eV and the warped density on that grid.

    Raises
    ------
    ValueError
        If the input arrays, domain bounds, or requested shifts are invalid.
    RuntimeError
        If the global deformation is nonmonotonic or the output grid is negative.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    y_density = np.asarray(y_density)
    shifts = np.asarray(shifts, dtype=float)

    _validate_scalar_warp_inputs(x_grid, y_density, boundaries, shifts)

    boundaries, shifts_eV = _sort_scalar_domains(x_grid, boundaries, shifts)
    x_grid, y_density = _extend_warp_support(x_grid, y_density, boundaries, shifts_eV)

    max_distortion = 0.5
    internal_pad = _scalar_padding_width(shifts_eV, max_distortion)
    x_padded, y_padded = _pad_scalar_spectrum(x_grid, y_density, internal_pad)

    # ONE displacement field for the entire spectrum.
    delta_x = _build_global_displacement(
        x_padded=x_padded,
        boundaries=boundaries,
        shifts_eV=shifts_eV,
        freeze_origin=freeze_origin,
        max_distortion=max_distortion,
    )

    x_new = x_padded + delta_x

    # Examine adjacent grid differences to enforce ordering or locate transitions.
    if np.any(np.diff(x_new) <= 0.0):
        raise RuntimeError(
            "Internal error: global spatial deformation is not monotonic."
        )

    y_final = _resample_scalar_warp(
        x_grid, y_density, x_padded, y_padded, x_new, preserve_shape,
    )

    if np.any(x_grid < 0.0):
        raise RuntimeError(
            "Internal error: spatial warp generated negative energies."
        )

    return x_grid, y_final



# ---------------------------------------------------------------------------
# Batched ensemble implementation
# ---------------------------------------------------------------------------

@njit(fastmath=True, inline="always")
def _interpolate_monotonic_row(x_target, x_source, y_source, output) -> None:
    """Interpolate one source row into its preallocated output row.

    Parameters
    ----------
    x_target, x_source : numpy.ndarray
        Target coordinates and increasing source coordinates in eV.
    y_source : numpy.ndarray
        Density values on the source grid.
    output : numpy.ndarray
        Zero-initialized output row. Points outside source support stay zero.
    """
    n_source = x_source.size
    n_target = x_target.size
    source_idx = 0

    for target_idx in range(n_target):
        x_value = x_target[target_idx]

        if x_value < x_source[0]:
            continue
        if x_value > x_source[n_source - 1]:
            continue

        while (
            source_idx + 1 < n_source
            and x_source[source_idx + 1] < x_value
        ):
            source_idx += 1

        if source_idx + 1 >= n_source:
            output[target_idx] = y_source[n_source - 1]
            continue

        x_left = x_source[source_idx]
        x_right = x_source[source_idx + 1]
        y_left = y_source[source_idx]
        y_right = y_source[source_idx + 1]

        if x_right <= x_left:
            output[target_idx] = y_left
            continue

        fraction = (x_value - x_left) / (x_right - x_left)
        output[target_idx] = (
            y_left + fraction * (y_right - y_left)
        )


@njit(parallel=True, fastmath=True)
def _interp_monotonic_rows(
    x_target: np.ndarray,
    x_source: np.ndarray,
    y_source: np.ndarray,
) -> np.ndarray:
    """
    Interpolate many monotonic source rows onto one shared target grid.

    Parameters
    ----------
    x_target : numpy.ndarray
        Shared target coordinates with shape ``(n_target,)``.
    x_source : numpy.ndarray
        Strictly increasing source coordinates with shape
        ``(n_samples, n_source)``.
    y_source : numpy.ndarray
        Source densities with shape ``(n_samples, n_source)``.

    Returns
    -------
    numpy.ndarray
        Interpolated density matrix with shape
        ``(n_samples, n_target)``.
    """
    n_samples = x_source.shape[0]
    n_target = x_target.size
    output = np.zeros((n_samples, n_target), dtype=np.float64)

    for sample_idx in prange(n_samples):
        _interpolate_monotonic_row(
            x_target, x_source[sample_idx], y_source[sample_idx], output[sample_idx],
        )

    return output


def _build_event_basis(
    boundaries: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build topology event positions and their linear domain coefficients.

    Parameters
    ----------
    boundaries : numpy.ndarray
        Ordered domain boundaries with shape ``(n_domains, 2)``.
    tolerance : float
        Numerical tolerance used to merge shared boundary locations.

    Returns
    -------
    positions : numpy.ndarray
        Unique event positions in eV.
    coefficients : numpy.ndarray
        Linear event map with shape ``(n_domains, n_events)``. Multiplying a
        shift matrix by this array yields every displacement jump in the batch.
    """
    n_domains = boundaries.shape[0]
    raw_events = []

    for domain_idx, (start, end) in enumerate(boundaries):
        if end <= start:
            raise ValueError(
                f"Invalid spatial boundary ({start}, {end}): "
                "end must be greater than start."
            )

        start_coeff = np.zeros(n_domains, dtype=np.float64)
        start_coeff[domain_idx] = 1.0
        raw_events.append((float(start), start_coeff))

        end_coeff = np.zeros(n_domains, dtype=np.float64)
        end_coeff[domain_idx] = -1.0
        raw_events.append((float(end), end_coeff))

    if not raw_events:
        return (
            np.empty(0, dtype=np.float64),
            np.empty((n_domains, 0), dtype=np.float64),
        )

    raw_events.sort(key=lambda item: item[0])
    positions = []
    coefficient_columns = []

    for position, coefficient in raw_events:
        if positions and abs(position - positions[-1]) <= tolerance:
            coefficient_columns[-1] += coefficient
        else:
            positions.append(position)
            coefficient_columns.append(coefficient.copy())

    return (
        np.asarray(positions, dtype=np.float64),
        np.stack(coefficient_columns, axis=1),
    )



@njit(inline="always")
def _displacement_at_energy(
    x_value, event_positions, jumps, width_scale, freeze_origin,
    atol, min_width, max_distortion,
) -> float:
    """Sum the smooth displacement contributions at one energy.

    Parameters
    ----------
    x_value : float
        Energy coordinate in eV.
    event_positions, jumps : numpy.ndarray
        Energy and displacement change in eV for each event.
    width_scale : float
        Common widening multiplier for nearby transitions.
    freeze_origin : bool
        Keep the acoustic origin fixed.
    atol : float
        Coordinate tolerance for identifying an origin event.
    min_width : float
        Minimum allowed transition width in eV.
    max_distortion : float
        Target bound on the slope of an isolated event.

    Returns
    -------
    float
        Total displacement in eV.
    """
    n_events = event_positions.size
    displacement = 0.0

    for event_idx in range(n_events):
        jump = jumps[event_idx]
        if abs(jump) <= 1e-15:
            continue

        base_width = max(
            min_width,
            1.05 * np.pi * abs(jump)
            / (2.0 * max_distortion),
        )
        width = base_width * width_scale
        position = event_positions[event_idx]

        if freeze_origin and abs(position) <= atol:
            normalized = (x_value - position) / width
        else:
            left_edge = position - 0.5 * width
            normalized = (x_value - left_edge) / width

        if normalized <= 0.0:
            step = 0.0
        elif normalized >= 1.0:
            step = 1.0
        else:
            step = 0.5 * (
                1.0 - np.cos(np.pi * normalized)
            )

        displacement += jump * step

    if freeze_origin and x_value <= 0.0:
        displacement = 0.0

    return displacement


@njit(inline="always")
def _write_candidate_map(
    x_padded, event_positions, jumps, freeze_origin, atol,
    min_width, max_distortion, width_scale, output,
) -> bool:
    """Write one candidate map and check whether its coordinates increase.

    Parameters
    ----------
    x_padded : numpy.ndarray
        Padded energy grid in eV.
    event_positions, jumps : numpy.ndarray
        Energy and displacement change in eV for each event.
    freeze_origin : bool
        Keep the acoustic origin fixed.
    atol : float
        Coordinate tolerance for identifying an origin event.
    min_width : float
        Minimum allowed transition width in eV.
    max_distortion : float
        Target bound on the slope of an isolated event.
    width_scale : float
        Common widening multiplier.
    output : numpy.ndarray
        Destination row for the candidate coordinates.

    Returns
    -------
    bool
        Whether the written coordinates are strictly increasing.
    """
    n_energy = x_padded.size
    previous_x = 0.0
    monotonic = True

    for energy_idx in range(n_energy):
        x_value = x_padded[energy_idx]
        displacement = _displacement_at_energy(
            x_value, event_positions, jumps, width_scale, freeze_origin,
            atol, min_width, max_distortion,
        )
        mapped = x_value + displacement
        output[energy_idx] = mapped

        if energy_idx > 0 and mapped <= previous_x:
            monotonic = False

        previous_x = mapped

    return monotonic


@njit(inline="always")
def _widen_until_monotonic(
    x_padded, event_positions, jumps, freeze_origin, atol,
    min_width, max_distortion, output,
) -> bool:
    """Retry a sample's coordinate map with progressively wider transitions.

    Parameters
    ----------
    x_padded : numpy.ndarray
        Padded energy grid in eV.
    event_positions, jumps : numpy.ndarray
        Energy and displacement change in eV for each event.
    freeze_origin : bool
        Keep the acoustic origin fixed.
    atol : float
        Coordinate tolerance for identifying an origin event.
    min_width : float
        Minimum allowed transition width in eV.
    max_distortion : float
        Target bound on the slope of an isolated event.
    output : numpy.ndarray
        Destination for the last candidate, including on failure.

    Returns
    -------
    bool
        Whether a monotonic map was found within 20 attempts.
    """
    width_scale = 1.0
    for _ in range(20):
        if _write_candidate_map(
            x_padded, event_positions, jumps, freeze_origin, atol,
            min_width, max_distortion, width_scale, output,
        ):
            return True
        # Overlapping transitions may need more room to remain monotonic.
        width_scale *= 1.35
    return False


@njit(parallel=True)
def _build_monotonic_displacement_rows(
    x_padded: np.ndarray,
    event_positions: np.ndarray,
    event_jumps: np.ndarray,
    freeze_origin: bool,
    atol: float,
    dx: float,
    max_distortion: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build monotonic displacement maps for a complete sample batch.

    Parameters
    ----------
    x_padded : numpy.ndarray
        Shared padded energy grid with shape ``(n_energy,)``.
    event_positions : numpy.ndarray
        Unique topology-event positions in eV with shape ``(n_events,)``.
    event_jumps : numpy.ndarray
        Per-sample event jumps with shape ``(n_samples, n_events)``.
    freeze_origin : bool
        Whether the acoustic origin must remain fixed.
    atol : float
        Numerical tolerance used to identify the origin event.
    dx : float
        Mean energy-grid spacing in eV.
    max_distortion : float
        Maximum admissible local displacement derivative.

    Returns
    -------
    x_new : numpy.ndarray
        Monotonic mapped coordinates with shape
        ``(n_samples, n_energy)``.
    valid : numpy.ndarray
        Boolean flag per sample indicating whether a monotonic mapping was
        found within the widening iterations.

    Notes
    -----
    The previous vectorized implementation allocated one or more full
    ``(n_samples, n_energy)`` temporary arrays for every topology event and
    widening iteration. This fused kernel evaluates the raised-cosine events
    directly inside compiled loops, retaining only the final coordinate matrix.
    """
    n_samples = event_jumps.shape[0]
    n_energy = x_padded.size

    x_new = np.empty((n_samples, n_energy), dtype=np.float64)
    valid = np.zeros(n_samples, dtype=np.bool_)
    min_width = 4.0 * abs(dx)

    for sample_idx in prange(n_samples):
        valid[sample_idx] = _widen_until_monotonic(
            x_padded, event_positions, event_jumps[sample_idx], freeze_origin,
            atol, min_width, max_distortion, x_new[sample_idx],
        )

    return x_new, valid

def _warped_sample_lengths(x_grid, boundaries_arr, shifts_eV, dx):
    """Calculate the physical output length of each warped row.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Input grid in eV.
    boundaries_arr : numpy.ndarray
        Domain bounds in eV, one row per domain.
    shifts_eV : numpy.ndarray
        Per-sample domain shifts in eV.
    dx : float
        Mean input grid spacing in eV.

    Returns
    -------
    numpy.ndarray
        Valid prefix length of each output row.
    """
    n_samples, n_domains = shifts_eV.shape
    edge_tolerance = 2.0 * abs(dx)
    original_right = float(x_grid[-1])
    # Determine the exact scalar-compatible physical support length for each
    # row. Samples that do not shift the high-energy edge keep the base grid.
    sample_lengths = np.full(n_samples, x_grid.size, dtype=np.int64)
    required_extension = np.zeros(n_samples, dtype=np.float64)

    if n_domains:
        edge_domains = boundaries_arr[:, 1] >= original_right - edge_tolerance
        if np.any(edge_domains):
            positive_edge_shifts = np.where(
                shifts_eV[:, edge_domains] > 0.0,
                shifts_eV[:, edge_domains],
                0.0,
            )
            required_extension = np.max(positive_edge_shifts, axis=1)

            active_extension = required_extension > 0.0
            sample_lengths[active_extension] += (
                np.ceil(required_extension[active_extension] / abs(dx)).astype(np.int64)
                + 2
            )
    return sample_lengths


def _resolve_warp_output_grid(x_grid, output_grid, sample_lengths, dx):
    """Allocate or validate the common output grid for warped spectra.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Base energy coordinates in eV.
    output_grid : numpy.ndarray or None
        Optional caller-owned capacity grid.
    sample_lengths : numpy.ndarray
        Required valid grid length of each sample.
    dx : float
        Mean grid spacing for a generated extension, in eV.

    Returns
    -------
    numpy.ndarray
        Output grid containing the base prefix and all required extensions.

    Raises
    ------
    ValueError
        If the supplied grid has an incompatible prefix or insufficient capacity.
    """
    if output_grid is None:
        max_length = int(np.max(sample_lengths))
        n_extra = max_length - x_grid.size
        if n_extra > 0:
            extra_grid = x_grid[-1] + np.arange(
                1,
                n_extra + 1,
                dtype=np.float64,
            ) * dx
            # Assemble the grid segments in the order required by the reconstruction.
            output_grid = np.concatenate((x_grid, extra_grid))
        else:
            output_grid = x_grid.copy()
    else:
        output_grid = np.asarray(output_grid, dtype=np.float64)
        if output_grid.ndim != 1 or output_grid.size < x_grid.size:
            raise ValueError(
                "output_grid must be one-dimensional and contain the base grid."
            )
        if not np.allclose(
            output_grid[: x_grid.size],
            x_grid,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("output_grid must preserve the baseline grid prefix.")
        if np.max(sample_lengths) > output_grid.size:
            raise ValueError(
                "output_grid is too short for the requested spatial shifts."
            )
    return output_grid


def _reconstruct_warped_rows(
    output_grid, x_padded, x_new, y_extended, y_padded,
    sample_lengths, monotonic, preserve_shape,
):
    """Reconstruct monotonic warped rows and remove values outside their support.

    Parameters
    ----------
    output_grid, x_padded : numpy.ndarray
        Physical output grid and padded input grid.
    x_new : numpy.ndarray
        Deformed coordinates for each sample.
    y_extended, y_padded : numpy.ndarray
        Input densities on the output grid and internal padded grid.
    sample_lengths : numpy.ndarray
        Valid prefix length of each output row.
    monotonic : numpy.ndarray
        Flags indicating which deformations are strictly increasing.
    preserve_shape : bool
        Skip Jacobian and area corrections when True.

    Returns
    -------
    numpy.ndarray
        Finite density matrix on the shared output grid.

    Raises
    ------
    ValueError
        If any row has a nonmonotonic deformation.
    RuntimeError
        If reconstruction produces nonfinite values.
    """
    if not np.all(monotonic):
        invalid = np.flatnonzero(~monotonic)
        raise ValueError(
            "Could not construct monotonic global spatial deformations for "
            f"batch rows {invalid.tolist()}."
        )

    if preserve_shape:
        y_warped = y_padded
    else:
        jacobian = np.gradient(x_new, x_padded, axis=1)
        jacobian = np.clip(jacobian, 0.5, 2.0)
        y_warped = y_padded / jacobian

    y_final = _interp_monotonic_rows(
        output_grid,
        x_new,
        y_warped,
    )

    if not preserve_shape:
        # Integrate against the physical coordinate grid, accounting for nonuniform spacing.
        old_area = np.trapezoid(y_extended, output_grid, axis=1)
        # Integrate against the physical coordinate grid, accounting for nonuniform spacing.
        new_area = np.trapezoid(y_final, output_grid, axis=1)
        valid_area = np.abs(new_area) > 1e-12
        y_final[valid_area] *= (
            old_area[valid_area] / new_area[valid_area]
        )[:, None]

    # The ensemble uses a shared capacity grid, but each scalar spectrum retains
    # its original physical right boundary. Remove numerical values beyond that
    # support so later integration reproduces the scalar semantics exactly.
    columns = np.arange(output_grid.size)[None, :]
    y_final[columns >= sample_lengths[:, None]] = 0.0

    if not np.all(np.isfinite(y_final)):
        raise RuntimeError(
            "Internal error: batched spatial warp produced non-finite values."
        )
    return y_final


def _validate_batch_warp_inputs(x_grid, y_density, shifts, boundaries):
    """Convert batch warp inputs to numeric arrays and check their layout.

    Parameters
    ----------
    x_grid, y_density : array_like
        Physical energy grid in eV and one density row per sample.
    shifts : array_like
        Matrix of sample-by-domain shifts in meV.
    boundaries : array_like
        Lower and upper bounds in eV for each domain.

    Returns
    -------
    tuple of numpy.ndarray
        Energy grid, density rows, shift rows, and domain bounds.
    """
    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_density = np.asarray(y_density, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.float64)
    boundaries_arr = np.asarray(boundaries, dtype=np.float64)

    if x_grid.ndim != 1 or x_grid.size < 2:
        raise ValueError("x_grid must be one-dimensional with at least two points.")
    if y_density.ndim != 2 or y_density.shape[1] != x_grid.size:
        raise ValueError(
            "y_density must have shape (n_samples, len(x_grid))."
        )
    if shifts.ndim != 2 or shifts.shape[0] != y_density.shape[0]:
        raise ValueError(
            "shifts must be two-dimensional with one row per density sample."
        )
    if boundaries_arr.size == 0:
        boundaries_arr = np.empty((0, 2), dtype=np.float64)
    if boundaries_arr.ndim != 2 or boundaries_arr.shape[1] != 2:
        raise ValueError("boundaries must have shape (n_domains, 2).")
    if shifts.shape[1] != boundaries_arr.shape[0]:
        raise ValueError(
            "The number of spatial-shift columns must match the number of domains."
        )
    # Examine adjacent grid differences to enforce ordering or locate transitions.
    if np.any(np.diff(x_grid) <= 0.0) or np.any(x_grid < 0.0):
        raise ValueError("x_grid must be strictly increasing and non-negative.")

    return x_grid, y_density, shifts, boundaries_arr


def _deform_batch_grid(output_grid, y_extended, boundaries_arr, shifts_eV, dx, freeze_origin):
    """Prepare the padded workspace and build the batched coordinate maps.

    Parameters
    ----------
    output_grid, y_extended : numpy.ndarray
        Shared physical grid and sample density rows extended onto it.
    boundaries_arr, shifts_eV : numpy.ndarray
        Ordered domain bounds and sample shifts in eV.
    dx : float
        Mean spacing of the original energy grid in eV.
    freeze_origin : bool
        Keep the acoustic origin fixed.

    Returns
    -------
    tuple of numpy.ndarray
        Padded grid, padded density, deformed coordinates, and monotonicity flags.
    """
    n_samples, n_domains = shifts_eV.shape
    max_abs_shift = np.max(np.abs(shifts_eV), axis=1)
    if n_domains > 1:
        # Examine adjacent grid differences to enforce ordering or locate transitions.
        max_jump = np.max(np.abs(np.diff(shifts_eV, axis=1)), axis=1)
    else:
        max_jump = max_abs_shift.copy()

    max_distortion = 0.5
    transition_scale = np.maximum(max_abs_shift, max_jump)
    max_transition_width = (
        1.05 * np.pi * transition_scale / (2.0 * max_distortion)
    )
    internal_pad = float(
        np.max(np.maximum(max_abs_shift, max_transition_width))
    )

    x_padded, pad_len = _setup_padded_grid(output_grid, internal_pad)
    y_padded = np.zeros(
        (n_samples, x_padded.size),
        dtype=np.float64,
    )
    y_padded[:, pad_len : pad_len + output_grid.size] = y_extended

    atol = max(abs(dx) * 1e-6, 1e-14)
    event_positions, event_basis = _build_event_basis(
        boundaries_arr,
        tolerance=atol,
    )
    event_jumps = shifts_eV @ event_basis

    # Build every sample deformation inside one fused Numba kernel. This avoids
    # allocating full broadcast temporaries for each topology event and keeps
    # the adaptive width-widening loop out of the Python hot path.
    x_new, monotonic = _build_monotonic_displacement_rows(
        x_padded=x_padded,
        event_positions=event_positions,
        event_jumps=event_jumps,
        freeze_origin=freeze_origin,
        atol=atol,
        dx=dx,
        max_distortion=max_distortion,
    )

    return x_padded, y_padded, x_new, monotonic


def apply_spatial_warp_batch(
    x_grid: np.ndarray,
    y_density: np.ndarray,
    boundaries: List[Tuple[float, float]],
    shifts: np.ndarray,
    freeze_origin: bool,
    preserve_shape: bool = True,
    output_grid: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply diffeomorphic spatial warping to a complete sample batch.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Baseline energy grid with shape ``(n_energy,)`` in eV.
    y_density : numpy.ndarray
        Amplitude-modified density matrix with shape
        ``(n_samples, n_energy)``.
    boundaries : list of tuple of float
        Shared topology boundaries in eV.
    shifts : numpy.ndarray
        Spatial translations with shape ``(n_samples, n_domains)`` in meV.
    freeze_origin : bool
        If True, preserve the zero-energy acoustic anchor.
    preserve_shape : bool, default=True
        If True, transport the density without Jacobian scaling.
    output_grid : numpy.ndarray, optional
        Canonical right-extended output grid. If omitted, a grid large enough
        for the current batch is constructed automatically.

    Returns
    -------
    output_grid : numpy.ndarray
        Shared output energy grid in eV.
    y_final : numpy.ndarray
        Warped density matrix with shape
        ``(n_samples, n_output_energy)``.
    sample_lengths : numpy.ndarray
        Physical support length of every sample before right-side zero padding.

    Notes
    -----
    The displacement field is evaluated for all samples simultaneously. Only
    the final monotonic-row interpolation is delegated to a Numba parallel
    kernel, avoiding a Python loop over ensemble members.
    """
    x_grid, y_density, shifts, boundaries_arr = _validate_batch_warp_inputs(
        x_grid, y_density, shifts, boundaries,
    )
    n_samples = y_density.shape[0]
    n_domains = boundaries_arr.shape[0]
    # Examine adjacent grid differences to enforce ordering or locate transitions.
    dx = float(np.mean(np.diff(x_grid)))
    edge_tolerance = 2.0 * abs(dx)

    if n_domains:
        order = np.argsort(boundaries_arr[:, 0], kind="stable")
        boundaries_arr = boundaries_arr[order]
        shifts = shifts[:, order]

        for domain_idx in range(n_domains - 1):
            if (
                boundaries_arr[domain_idx, 1]
                > boundaries_arr[domain_idx + 1, 0] + edge_tolerance
            ):
                raise ValueError(
                    "Spatial domains overlap. A single diffeomorphic "
                    "displacement is ambiguous for overlapping domains."
                )

    shifts_eV = shifts / 1000.0

    sample_lengths = _warped_sample_lengths(x_grid, boundaries_arr, shifts_eV, dx)

    output_grid = _resolve_warp_output_grid(x_grid, output_grid, sample_lengths, dx)

    # Extend the amplitude matrix once to the shared batch grid. The high-energy
    # tail is physical zero until displaced density moves into it.
    y_extended = np.zeros(
        (n_samples, output_grid.size),
        dtype=np.float64,
    )
    y_extended[:, : x_grid.size] = y_density

    if n_domains == 0 or not np.any(np.abs(shifts_eV) > 1e-15):
        return output_grid, y_extended, sample_lengths

    x_padded, y_padded, x_new, monotonic = _deform_batch_grid(
        output_grid, y_extended, boundaries_arr, shifts_eV, dx, freeze_origin,
    )

    y_final = _reconstruct_warped_rows(
        output_grid, x_padded, x_new, y_extended, y_padded,
        sample_lengths, monotonic, preserve_shape,
    )


    return output_grid, y_final, sample_lengths

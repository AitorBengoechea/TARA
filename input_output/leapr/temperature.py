"""Temperature selection policy for single-temperature LEAPR sample generation."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

from input_output.leapr.exceptions import LeaprFormatError

if TYPE_CHECKING:
    from .core import LeaprFile


TEMPERATURE_MARGIN_K = 5.0


def available_leapr_temperatures(leapr_file: LeaprFile) -> tuple[float, ...]:
    """Return the physical temperatures recorded in an entire LEAPR template.

    Negative LEAPR temperature cards reuse the preceding phonon spectrum; their
    absolute magnitudes are still physical processing temperatures.

    Parameters
    ----------
    leapr_file : LeaprFile
        Fully parsed original LEAPR input, including negative temperature cards.

    Returns
    -------
    tuple of float
        Unique, ascending physical temperatures in kelvin.

    Raises
    ------
    LeaprFormatError
        If the template has no usable first positive-temperature spectrum or
        contains an invalid temperature card.
    """
    temperatures = []
    has_spectrum = False
    for block in leapr_file.temp_blocks:
        try:
            value = float(block["temp"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise LeaprFormatError("Invalid temperature card in LEAPR template.") from exc
        if not math.isfinite(value) or value == 0.0:
            raise LeaprFormatError(f"Invalid LEAPR temperature: {value!r} K.")
        if value > 0.0:
            has_spectrum = True
        elif not has_spectrum:
            raise LeaprFormatError(
                "A negative LEAPR temperature cannot precede the first "
                "positive-temperature phonon spectrum."
            )
        temperatures.append(abs(value))
    if not has_spectrum:
        raise LeaprFormatError("LEAPR template contains no positive-temperature spectrum.")
    return tuple(sorted(set(temperatures)))


def resolve_leapr_temperature(
    leapr_file: LeaprFile,
    temperature: float | None = None,
    *,
    warn_on_reuse: bool = True,
) -> float:
    """Select one output temperature within the template's supported range.

    An interior request may reuse the first positive-temperature spectrum.
    Requests up to five kelvin beyond either boundary also reuse it with a
    warning; larger extrapolations are rejected before NJOY execution.

    Parameters
    ----------
    leapr_file : LeaprFile
        Original parsed LEAPR template, including all negative temperature cards.
    temperature : float or None, optional
        Requested output temperature in kelvin. ``None`` selects the first
        positive-temperature block, preserving the legacy behavior.
    warn_on_reuse : bool, optional
        Emit warnings when the requested temperature differs from the spectrum
        block's temperature. Disable only when a parent batch has warned once.

    Returns
    -------
    float
        Validated, unrounded temperature to encode in the single output block.

    Raises
    ------
    LeaprFormatError
        If the template contains invalid temperature records.
    ValueError
        If the request is not finite and positive, or is more than five kelvin
        beyond the original template's temperature boundaries.
    """
    available = available_leapr_temperatures(leapr_file)
    original = next(float(block["temp"]) for block in leapr_file.temp_blocks
                    if float(block["temp"]) > 0.0)
    try:
        requested = original if temperature is None else float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Requested LEAPR temperature must be positive and finite.") from exc
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("Requested LEAPR temperature must be positive and finite.")

    minimum, maximum = available[0], available[-1]
    distance_outside = max(minimum - requested, requested - maximum, 0.0)
    if distance_outside > TEMPERATURE_MARGIN_K:
        raise ValueError(
            f"Requested LEAPR temperature {requested:g} K is {distance_outside:g} K "
            f"outside the original range [{minimum:g}, {maximum:g}] K "
            f"(maximum permitted overshoot: {TEMPERATURE_MARGIN_K:g} K). "
            "Edit the input LEAPR template explicitly to use this temperature."
        )

    if warn_on_reuse and not math.isclose(requested, original, abs_tol=1e-10, rel_tol=0.0):
        message = _reuse_warning_message(requested, minimum, maximum, distance_outside)
        warnings.warn(message, UserWarning, stacklevel=2)
    return requested


def select_leapr_temperature_block(
    leapr_file: LeaprFile,
    temperature: float | None = None,
    *,
    tolerance_k: float = 1e-5,
) -> tuple[dict, float]:
    """Select the physical source block at an explicitly tabulated temperature.

    A negative Card 10 temperature references the preceding positive block;
    it does not supply its own oscillator or diffusion constants. This
    selector never silently substitutes the first template block for a
    different liquid-water temperature.

    Parameters
    ----------
    leapr_file : LeaprFile
        Parsed complete source template.
    temperature : float or None, optional
        Exact requested tabulated temperature; ``None`` uses the first block.
    tolerance_k : float, optional
        Floating-point matching tolerance in kelvin, not an interpolation band.

    Returns
    -------
    tuple of (dict, float)
        Source block containing all physical parameters and selected T in K.

    Raises
    ------
    ValueError
        If a requested temperature is not explicitly tabulated.
    LeaprFormatError
        If a negative temperature appears before any positive block.
    """
    temperatures = available_leapr_temperatures(leapr_file)
    if temperature is None:
        requested = abs(float(leapr_file.temp_blocks[0]['temp']))
    else:
        try:
            requested = float(temperature)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('Requested LEAPR temperature must be finite and positive.') from exc
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError('Requested LEAPR temperature must be finite and positive.')
    source = None
    for block in leapr_file.temp_blocks:
        value = float(block['temp'])
        if value > 0.0:
            source = block
        if math.isclose(abs(value), requested, abs_tol=tolerance_k, rel_tol=0.0):
            if source is None:
                raise LeaprFormatError('Cannot select a negative temperature before a positive block.')
            return source, abs(value)
    raise ValueError(
        f'LEAPR temperature {requested:g} K is not explicitly tabulated. '
        f'Available temperatures: {temperatures}. '
        'For liquids, select an existing block; physical interpolation is not implemented.'
    )


def _reuse_warning_message(
    requested: float, minimum: float, maximum: float, distance_outside: float
) -> str:
    """Describe the established LEAPR spectrum reuse policy in kelvin.

    Parameters
    ----------
    requested : float
        Selected target temperature in kelvin.
    minimum, maximum : float
        Original template temperature bounds in kelvin.
    distance_outside : float
        Nonnegative overshoot beyond the template bounds, in kelvin.

    Returns
    -------
    str
        Warning text with the historical wording and formatting.
    """
    if distance_outside > 0.0:
        message = (
            f"Reusing the LEAPR phonon spectrum at {requested:g} K, "
            f"which is {distance_outside:g} K outside the original "
            f"temperature boundaries [{minimum:g}, {maximum:g}] K."
        )
    else:
        message = (
            f"Reusing the LEAPR phonon spectrum at {requested:g} K "
            f"because it lies inside the original temperature range "
            f"[{minimum:g}, {maximum:g}] K."
        )
    return message

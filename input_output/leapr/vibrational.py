"""Translate LEAPR components to full vibrational tables without source mutation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import DiscreteOscillator
from core.physics.vibrational_spectrum import reconstruct_vibrational_spectrum


def _validated_discrete_modes(block: dict[str, Any]) -> tuple[int, ArrayLike, ArrayLike]:
    """Read the mode count and check that both oscillator arrays match it.

    Parameters
    ----------
    block : dict of str to Any
        LEAPR temperature block containing the discrete oscillator cards.

    Returns
    -------
    tuple of (int, array_like, array_like)
        Mode count, oscillator energies in eV, and dimensionless weights.

    Raises
    ------
    ValueError
        If the mode count is invalid or differs from either array length.
    """
    nd = int(block.get("nd", 0))
    energies = block.get("oscillator_energies", ())
    weights = block.get("oscillator_weights", ())
    if (
        nd != block.get("nd", 0)
        or nd < 0
        or len(energies) != nd
        or len(weights) != nd
    ):
        raise ValueError("Invalid LEAPR discrete-mode count or arrays.")
    return nd, energies, weights


def _validate_dynamical_weights(vibrational_weight: float, translation_weight: float) -> None:
    """Check that the reconstructed vibrational and translation weights sum to one.

    Parameters
    ----------
    vibrational_weight : float
        Combined continuum and discrete-mode weight.
    translation_weight : float
        Translational coefficient from the original LEAPR block.

    Raises
    ------
    ValueError
        If the total differs from one by more than 1e-5.
    """
    if not np.isclose(vibrational_weight + float(translation_weight), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("LEAPR dynamical weights must sum to unity.")


def _reconstruct_leapr_modes(block, energies, weights, energy_grid) -> tuple[PhononSpectra, float]:
    """Construct the continuum and line inputs for Core reconstruction.

    Parameters
    ----------
    block : dict
        Positive-temperature LEAPR data with the source continuum.
    energies, weights : array_like
        Discrete mode energies in eV and their weights.
    energy_grid : array_like or None
        Optional output grid in eV.

    Returns
    -------
    tuple of PhononSpectra and float
        Unit-area vibrational shape and its physical weight.
    """
    continuum = PhononSpectra(
        pd.Series(
            block["phonon_spectrum"],
            index=np.arange(int(block["ni"]), dtype=float) * float(block["delta"]),
        )
    )
    modes = tuple(
        DiscreteOscillator(f"mode_{i}", float(e), float(w))
        for i, (e, w) in enumerate(zip(energies, weights))
    )
    spectrum, weight = reconstruct_vibrational_spectrum(
        continuum, float(block["tbeta"]), modes, energy_grid
    )
    return spectrum, weight


def tabulate_vibrational_block(
    block: dict[str, Any],
    energy_grid: ArrayLike | None = None,
) -> dict[str, Any]:
    """Return a full-spectrum LEAPR block while retaining original components.

    Parameters
    ----------
    block : dict of str to Any
        Resolved positive-temperature LEAPR block including discrete modes.
    energy_grid : array_like or None, optional
        Optional uniform reconstruction grid in eV. For active modes, the
        default 1 meV Gaussian width requires spacing at most 1/3 meV and
        sufficient support for the tails.

    Returns
    -------
    dict of str to Any
        Independent ``nd=0`` block with total vibrational ``tbeta``. Original
        data are retained under ``vibrational_source`` (not serialized).
        An ``nd=0`` input is copied without numerical changes.

    Raises
    ------
    ValueError
        If the discrete-mode count ``nd`` or oscillator arrays are inconsistent,
        or if the total dynamical weights (``tbeta + sum(weights) + twt``) do
        not sum to unity within ``1e-5``.
    """
    result = deepcopy(block)
    nd, energies, weights = _validated_discrete_modes(block)
    if nd == 0:
        return result

    spectrum, weight = _reconstruct_leapr_modes(block, energies, weights, energy_grid)
    _validate_dynamical_weights(weight, block["twt"])

    result.update(
        vibrational_source=deepcopy(block),
        nd=0,
        oscillator_energies=np.array([]),
        oscillator_weights=np.array([]),
        tbeta=weight,
        delta=float(spectrum.energy_grid[1]),
        ni=len(spectrum.energy_grid),
        phonon_spectrum=spectrum.density_values.copy(),
    )
    return result

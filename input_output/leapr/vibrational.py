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
        Optional common uniform reconstruction grid in eV.

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
    if nd == 0:
        return result

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
    if not np.isclose(weight + float(block["twt"]), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("LEAPR dynamical weights must sum to unity.")

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

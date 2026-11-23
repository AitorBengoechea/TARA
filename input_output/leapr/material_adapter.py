"""Translate LEAPR material information to backend-independent Core data.

The original LEAPR template remains authoritative for the full NJOY deck.
Egelstaff--Schofield translation is represented as a physical model with its
own reduced parameter c, not reinterpreted as a generic SI coefficient.
"""

from __future__ import annotations

import math
import pandas as pd

from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import (
    DiscreteOscillator, MaterialScatteringModel, ScatteringSpecies, SpeciesDynamics,
)
from input_output.leapr.core import LeaprFile
from input_output.leapr.vibrational import tabulate_vibrational_block
from input_output.leapr.temperature import select_leapr_temperature_block


def _oscillators(block: dict, labels: tuple[str, ...] | None) -> tuple[DiscreteOscillator, ...]:
    """Interpret the LEAPR oscillator energy/weight arrays without reweighting.

    Parameters
    ----------
    block : dict
        Positive-temperature LEAPR block.
    labels : tuple of str or None
        Optional stable identifiers supplied by the calling experiment.

    Returns
    -------
    tuple of DiscreteOscillator
        Validated oscillators.
    """
    # LEAPR declares a fixed oscillator count; the energy and weight arrays
    # must correspond one-to-one with those declared modes.
    nd = int(block.get("nd", 0))
    energies = tuple(block.get("oscillator_energies", ()))
    weights = tuple(block.get("oscillator_weights", ()))
    if len(energies) != nd or len(weights) != nd:
        raise ValueError("LEAPR oscillator arrays must both have nd entries.")
    # Synthesize stable mode identifiers only when the caller has not
    # supplied experiment-specific oscillator labels.
    names = tuple(f"mode_{i}" for i in range(nd)) if labels is None else tuple(labels)
    if len(names) != nd:
        raise ValueError("Oscillator label count must match LEAPR nd.")
    # Preserve LEAPR oscillator energies and relative weights as provided;
    # Core validators, rather than the adapter, own physical checks.
    return tuple(DiscreteOscillator(name, float(energy), float(weight)) for name, energy, weight
                 in zip(names, energies, weights))


def material_model_from_leapr(
    template: LeaprFile,
    *,
    state: str,
    principal_label: str = "principal",
    secondary_label: str = "secondary",
    oscillator_labels: tuple[str, ...] | None = None,
    temperature: float | None = None,
) -> MaterialScatteringModel:
    """Describe the selected LEAPR dynamical block in backend-independent Core terms.

    Parameters
    ----------
    template : LeaprFile
        Validated source material input.
    state : str
        Declared physical state (``solid`` or ``liquid``).
    principal_label, secondary_label : str, optional
        User-provided physical species identifiers.
    oscillator_labels : tuple of str or None, optional
        Stable vibrational-mode identifiers.
    temperature : float or None, optional
        Exact tabulated temperature to select, defaulting to the first block.

    Returns
    -------
    MaterialScatteringModel
        Model suitable for full vibrational-spectrum sampling.

    Raises
    ------
    NotImplementedError
        If secondary scatterer dynamics require another temperature loop.
    ValueError
        If component weights, oscillators, or composition are invalid.
    """
    # Read one selected, tabulated temperature block so its spectrum,
    # translation parameters and oscillator data remain temperature-consistent.
    block, selected_temperature = select_leapr_temperature_block(template, temperature)
    _oscillators(block, oscillator_labels)  # Validate source labels before folding.
    block = tabulate_vibrational_block(block)
    principal = template.principal_scatterer
    secondary = template.secondary_scatterer
    twt = float(block["twt"])
    c = float(block["c"])
    # Egelstaff--Schofield c is the LEAPR model parameter, not an SI
    # diffusion constant; reject undefined or negative source values.
    if not math.isfinite(c) or c < 0:
        raise ValueError('LEAPR diffusion parameter c must be finite and nonnegative.')
    # A nonzero diffusion-width parameter without translational weight
    # would describe an inconsistent component decomposition.
    if twt == 0 and c != 0:
        raise ValueError('Nonzero diffusion parameter requires nonzero translation weight.')
    # The single-temperature Core model cannot encode a secondary
    # scatterer that requires its own LEAPR temperature sequence.
    if int(secondary.get("nss", 0)) > 0 and int(secondary["b7"]) != 1:
        raise NotImplementedError("Secondary LEAPR temperature loops (b7=0) are not supported.")

    # Reconstruct the uniformly spaced LEAPR energy axis directly from
    # ni and delta; do not resample the continuous spectrum here.
    continuous = PhononSpectra(pdos=pd.Series(
        block['phonon_spectrum'],
        index=pd.Index(range(int(block['ni'])), dtype=float) * float(block['delta']),
    ))
    # All vibrational modes are sampled together; translation remains separate.
    dynamics = SpeciesDynamics(
        continuous=continuous,
        continuous_weight=float(block["tbeta"]),
        translation_weight=twt,
        # Select the existing Core translation label from the LEAPR
        # component weight and model parameter without altering either.
        translation_model=(
            'egelstaff_schofield' if twt > 0 and c > 0 else
            'free_gas' if twt > 0 else 'none'
        ),
        translation_parameter=c if twt > 0 and c > 0 else None,
        oscillators=(),
    )
    # Secondary-scatterer metadata are optional; preserve the original
    # principal-only representation when LEAPR declares none.
    others = ()
    if int(secondary.get("nss", 0)) > 0:
        others = (ScatteringSpecies(
            secondary_label, int(secondary["mss"]), float(secondary["aws"])
        ),)
    # Preserve LEAPR material identity, selected temperature, species
    # multiplicities and mass ratios in the backend-independent model.
    return MaterialScatteringModel(
        identifier=template.title or "unnamed LEAPR material",
        state=state,
        temperature_k=selected_temperature,
        principal=ScatteringSpecies(
            principal_label, int(principal["npr"]), float(principal["awr"])
        ),
        dynamics=dynamics,
        secondary=others,
    )

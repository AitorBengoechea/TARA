"""NJOY deck composition for the ACE generation workflow.

Individual NJOY module syntax lives in :mod:`processing.deck_builders.njoy_modules`.
This module owns only the validated ACE workflow ordering and card values.
"""
from processing.deck_builders.njoy_modules import (
    build_acer_module,
    build_broadr_module,
    build_moder_module,
    build_reconr_module,
    build_stop_card,
    build_thermr_module,
    format_njoy_card,
)


# ACER Card 9 outgoing-energy equiprobable bins, preserved from TARA V1.
# Distinct from THERMR's ``nbin`` angular discretization (default 16).
ACER_THERMAL_NBINT = 200


def build_ace_deck(
    base_endf_path: str,
    tsl_endf_path: str,
    base_za: int,
    base_mat: int,
    tsl_mat: int,
    temp: float,
    icoh: int,
    natom: int,
    emax: float,
    suffix: str,
    za1: int,
    za2: int,
    za3: int,
    elas_mat: int,
    ielas: int,
    nbin: int = 16,
    tol: float = 0.001,
    mti: int = 237,
    mte: int | None = None,
) -> str:
    """Construct ``MODER -> RECONR -> BROADR -> THERMR -> ACER -> STOP``.

    ``base_endf_path`` and ``tsl_endf_path`` remain in the public signature for
    compatibility.  Processing must stage them as ``tape20`` and ``tape70``;
    paths are not embedded in NJOY input text.

    Parameters
    ----------
    base_endf_path : str
        Filesystem location of the base endf path.
    tsl_endf_path : str
        Filesystem location of the tsl endf path.
    base_za : int
        ZA identifier of the base evaluation.
    base_mat : int
        MAT identifier of the base neutron evaluation.
    tsl_mat : int
        MAT identifier of the thermal-scattering evaluation.
    temp : float
        Material temperature in kelvin.
    icoh : int
        THERMR flag controlling the coherent-scattering option.
    natom : int
        Number of principal atoms specified in the THERMR card.
    emax : float
        Upper energy limit for backend processing.
    suffix : str
        Identifier appended to the generated output name.
    za1 : int
        First ZA identifier in the NJOY module input.
    za2 : int
        Second ZA identifier in the NJOY module input.
    za3 : int
        Third ZA identifier in the NJOY module input.
    elas_mat : int
        Legacy elastic reaction MT identifier used by ACER.
    ielas : int
        ACER coherent/incoherent/mixed elastic option (0/1/2).
    nbin : int, optional
        THERMR angular bin count. Not ACER's outgoing-energy ``nbint``.
    tol : float, optional
        Backend numerical tolerance.
    mti : int, optional
        Thermal inelastic reaction number shared by THERMR and ACER.
    mte : int or None, optional
        Thermal elastic reaction MT; ``None`` preserves ``elas_mat``.
        Must match ``elas_mat`` when explicitly supplied.

    Returns
    -------
    str
        NJOY ACE-production deck assembled from shared modules.
    """
    del base_endf_path, tsl_endf_path
    effective_mte = _resolve_elastic_mt(mti, elas_mat, mte)

    reconstruction = build_reconr_module(
        -25,
        -21,
        label=f"Dynamically generated {base_za} at {temp}K",
        material_card=format_njoy_card(base_mat, 0, 0, terminated=False).strip(),
    )
    broadening = build_broadr_module(
        -25,
        -21,
        -22,
        material_card=format_njoy_card(base_mat, 1, 0, 0, 0, terminated=False).strip(),
        temperature_card=f"{temp}",
    )
    thermal_scattering = build_thermr_module(
        -75,
        -22,
        -23,
        control_card=(
            f"{tsl_mat} {base_mat} {nbin} 1 2 {icoh} "
            f"0 {natom} {mti} 2 /"
        ),
        temperature_card=f"{temp}",
        tolerance_card=format_njoy_card(tol, emax, terminated=False).strip(),
    )
    ace_conversion = build_acer_module(
        -25,
        -23,
        0,
        40,
        41,
        option_card=f"2 0 1 {suffix} /",
        title_card=f"{base_za} TSL at {temp}K",
        material_card=f"{base_mat} {temp} tsl /",
        za_card=f"{za1} {za2} {za3} /",
        thermal_card=(
            f"{mti} {ACER_THERMAL_NBINT} {effective_mte} {ielas} 1 {emax} 2 /"
        ),
    )

    # Thermal scattering acts on the broadened PENDF before ACER writes tapes 40 and 41.
    return "".join((
        build_moder_module(70, -75),
        build_moder_module(20, -25),
        reconstruction,
        broadening,
        thermal_scattering,
        ace_conversion,
        build_stop_card(),
    ))


def _resolve_elastic_mt(mti: int, elas_mat: int, mte: int | None) -> int:
    """Validate historical reaction IDs and resolve the ACER elastic MT.

    Parameters
    ----------
    mti : int
        Thermal inelastic reaction MT in the supported THERMR range.
    elas_mat : int
        Legacy thermal elastic reaction MT.
    mte : int or None
        Optional explicit MT, required to equal ``elas_mat`` if supplied.

    Returns
    -------
    int
        Resolved elastic MT to encode on ACER Card 9.

    Raises
    ------
    ValueError
        If the reaction values are invalid or contradictory.
    """
    if type(mti) is not int or not 221 <= mti <= 250:
        raise ValueError("THERMR/ACER mti must be an integer from 221 to 250.")
    if type(elas_mat) is not int or elas_mat < 0:
        raise ValueError("ACER elas_mat must be a nonnegative reaction MT.")
    if mte is not None and (type(mte) is not int or mte < 0):
        raise ValueError("ACER mte must be a nonnegative reaction MT.")
    if mte is not None and mte != elas_mat:
        raise ValueError("Conflicting ACER elastic reactions: mte and elas_mat must agree.")
    effective_mte = elas_mat if mte is None else mte
    return effective_mte

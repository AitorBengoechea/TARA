"""NJOY deck composition for LEAPR-based workflows.

Individual NJOY module syntax lives in :mod:`processing.deck_builders.njoy_modules`.
This module owns only LEAPR workflow composition.
"""

from processing.deck_builders.njoy_modules import (
    build_broadr_module,
    build_leapr_module,
    build_legacy_leapr_module,
    build_moder_module,
    build_reconr_module,
    build_stop_card,
    build_thermr_module,
    format_njoy_card,
    format_njoy_array,
)

# Backwards-compatible private alias for callers/tests that still import it.
_build_leapr_module = build_legacy_leapr_module


def build_leapr_deck(
    mat_id: int,
    lat: int,
    mass_amu: float,
    temperature: float,
    nalpha: int,
    nbeta: int,
    alpha_grid_str: str,
    beta_grid_str: str,
    egrid_str: str,
    rho_str: str,
) -> str:
    """Construct the legacy standalone LEAPR deck.

    Parameters
    ----------
    mat_id : int
        NJOY input value.
    lat : int
        NJOY input value.
    mass_amu : float
        NJOY input value.
    temperature : float
        NJOY input value.
    nalpha : int
        NJOY input value.
    nbeta : int
        NJOY input value.
    alpha_grid_str : str
        NJOY input value.
    beta_grid_str : str
        NJOY input value.
    egrid_str : str
        NJOY input value.
    rho_str : str
        NJOY input value.

    Returns
    -------
    str
        Serialized NJOY text.
    """
    return build_legacy_leapr_module(
        mat_id=mat_id,
        lat=lat,
        mass_amu=mass_amu,
        temperature=temperature,
        nalpha=nalpha,
        nbeta=nbeta,
        alpha_grid_str=alpha_grid_str,
        beta_grid_str=beta_grid_str,
        egrid_str=egrid_str,
        rho_str=rho_str,
    ) + build_stop_card()


def build_leapr_deck_from_module(leapr_module_text: str) -> str:
    """Construct a standalone LEAPR job from validated serialized LEAPR text.

    Parameters
    ----------
    leapr_module_text : str
        NJOY input value.

    Returns
    -------
    str
        Serialized NJOY text.
    """
    return build_leapr_module(leapr_module_text) + build_stop_card()


def build_analysis_deck(
    *,
    base_mat: int,
    leapr_mat: int,
    temperature: float,
    leapr_module_text: str,
    thermr_emax: float = 5.0,
    thermr_tol: float = 0.05,
    thermr_nbin: int = 16,
    thermr_icoh: int = 0,
    thermr_natom: int = 1,
) -> str:
    """Build the transient Analyzer NJOY sequence.

    The base ENDF evaluation must already be staged as ``tape20`` by
    :class:`processing.backends.njoy_processor.NjoyProcessor`.
    ``leapr_module_text`` is the complete sample-specific LEAPR representation
    serialized by the Input/Output layer. A terminal LEAPR ``stop`` is removed
    by :func:`build_leapr_module` before THERMR is appended.

    THERMR reads the binary BROADR PENDF and therefore also writes binary
    output. A final MODER converts that binary THERMR result to formatted ENDF
    for downstream parsing.

    Parameters
    ----------
    base_mat : int
        MAT identifier of the base neutron evaluation.
    leapr_mat : int
        MAT identifier for the LEAPR thermal-scattering output.
    temperature : float
        Material temperature in kelvin.
    leapr_module_text : str
        Complete serialized LEAPR module body.
    thermr_emax : float, optional
        Maximum energy for THERMR processing.
    thermr_tol : float, optional
        Numerical tolerance passed to THERMR.
    thermr_nbin : int, optional
        Number of THERMR energy bins.
    thermr_icoh : int, optional
        Coherent-scattering setting supplied to THERMR.
    thermr_natom : int, optional
        Number of principal atoms supplied to THERMR.

    Returns
    -------
    str
        Transient NJOY analysis-deck text assembled from shared modules.
    """
    return "".join(
        (
            build_moder_module(20, -21),
            build_reconr_module(
                -21,
                -22,
                label="Analyzer dynamic reconstruction",
                # Use the shared free-format card writer while preserving this card’s terminator
                # policy.
                material_card=format_njoy_card(base_mat, 0, 0, terminated=False).strip(),
            ),
            build_broadr_module(
                -21,
                -22,
                -23,
                # Use the shared free-format card writer while preserving this card’s terminator
                # policy.
                material_card=format_njoy_card(base_mat, 1, 0, 0, 0, terminated=False).strip(),
                temperature_card=f"{temperature}",
            ),
            build_leapr_module(leapr_module_text),
            build_thermr_module(
                24,
                -23,
                -25,
                control_card=(
                    f"{leapr_mat} {base_mat} {thermr_nbin} "
                    f"1 2 {thermr_icoh} 0 {thermr_natom} 221 2 /"
                ),
                temperature_card=f"{temperature}",
                # Use the shared free-format card writer while preserving this card’s terminator
                # policy.
                tolerance_card=format_njoy_card(thermr_tol, thermr_emax, terminated=False).strip(),
            ),
            build_moder_module(-25, 26),
            build_stop_card(),
        )
    )

"""Pure writers for individual NJOY modules and shared free-format cards.

This module owns only NJOY text syntax.  It does not decide workflow order,
perform file I/O, stage logical tapes, execute NJOY, or derive physics
parameters.  Workflow builders compose these primitives into complete jobs.
"""
from typing import Iterable


def format_njoy_card(*args: object, terminated: bool = True) -> str:
    """Format ordered fields as an NJOY free-format card.

    Parameters
    ----------
    *args : object
        Card values serialized in the given order.
    terminated : bool, optional
        Whether to append the NJOY slash terminator.

    Returns
    -------
    str
        Formatted card with its terminating newline.

    Notes
    -----
    Some NJOY cards deliberately omit the slash terminator; callers must
    retain the card-specific value of ``terminated``.
    """
    fields = " ".join(str(value) for value in args)
    return f"{fields}{' /' if terminated else ''}\n"


def format_njoy_array(
    values: Iterable[float],
    values_per_line: int = 5,
) -> str:
    """Format numeric values for NJOY free-format cards.

    Parameters
    ----------
    values : Iterable[float]
        Numerical values to process.
    values_per_line : int
        Value supplied to this NJOY module.

    Returns
    -------
    str
        Serialized NJOY text.
    """
    if values_per_line <= 0:
        raise ValueError("values_per_line must be positive.")

    formatted = [f"{float(value):.6e}" for value in values]
    lines = [
        " ".join(formatted[i:i + values_per_line])
        for i in range(0, len(formatted), values_per_line)
    ]
    return "\n".join(lines)


def build_moder_module(input_tape: int, output_tape: int) -> str:
    """Write one MODER module.

    Parameters
    ----------
    input_tape : int
        Logical NJOY tape identifier used by this module.
    output_tape : int
        Logical tape number receiving module output.

    Returns
    -------
    str
        Serialized NJOY text.
    """
    return f"moder\n{input_tape} {output_tape}\n"


def build_reconr_module(
    input_tape: int,
    output_tape: int,
    *,
    label: str,
    material_card: str,
    tolerance_card: str = "0.001 0 0.01 5e-08",
    terminator_card: str = "0 /",
) -> str:
    """Write one RECONR module from explicit NJOY card payloads.

    Parameters
    ----------
    input_tape : int
        Logical NJOY tape identifier used by this module.
    output_tape : int
        Logical tape number receiving module output.
    label : str
        Value supplied to this NJOY module.
    material_card : str
        Preformatted NJOY card payload inserted without alteration.
    tolerance_card : str
        Already formatted numerical-tolerance card text.
    terminator_card : str
        Preformatted NJOY card payload inserted without alteration.

    Returns
    -------
    str
        Serialized NJOY text.
    """
    return "".join((
        "reconr\n",
        # Use the shared free-format card writer while preserving this card’s terminator policy.
        format_njoy_card(input_tape, output_tape, terminated=False),
        f"'{label}'/\n",
        f"{material_card}\n",
        f"{tolerance_card}\n",
        f"{terminator_card}\n",
    ))


def build_broadr_module(
    input_tape: int,
    input_pendf_tape: int,
    output_tape: int,
    *,
    material_card: str,
    tolerance_card: str = "0.001 1e+06 0.01 5e-08",
    temperature_card: str,
    terminator_card: str = "0 /",
) -> str:
    """Write one BROADR module from explicit NJOY card payloads.

    Parameters
    ----------
    input_tape : int
        Logical NJOY tape identifier used by this module.
    input_pendf_tape : int
        Logical tape number containing processed ENDF input.
    output_tape : int
        Logical tape number receiving module output.
    material_card : str
        Preformatted NJOY card payload inserted without alteration.
    tolerance_card : str
        Already formatted numerical-tolerance card text.
    temperature_card : str
        Already formatted temperature card text.
    terminator_card : str
        Preformatted NJOY card payload inserted without alteration.

    Returns
    -------
    str
        Serialized NJOY text.
    """
    return "".join((
        "broadr\n",
        # Use the shared free-format card writer while preserving this card’s terminator policy.
        format_njoy_card(input_tape, input_pendf_tape, output_tape, terminated=False),
        f"{material_card}\n",
        f"{tolerance_card}\n",
        f"{temperature_card}\n",
        f"{terminator_card}\n",
    ))


def build_legacy_leapr_module(
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
    """Write the historical compact LEAPR module used by legacy callers.

    This primitive exists only to preserve the established standalone LEAPR API.
    Analyzer workflows must instead pass complete LEAPR text serialized from the
    validated LEAPR template through :func:`build_leapr_module`.

    Parameters
    ----------
    mat_id : int
        Value supplied to this NJOY module.
    lat : int
        Value supplied to this NJOY module.
    mass_amu : float
        Target atomic mass in atomic mass units.
    temperature : float
        Material temperature in kelvin.
    nalpha : int
        Value supplied to this NJOY module.
    nbeta : int
        Value supplied to this NJOY module.
    alpha_grid_str : str
        Value supplied to this NJOY module.
    beta_grid_str : str
        Value supplied to this NJOY module.
    egrid_str : str
        Value supplied to this NJOY module.
    rho_str : str
        Value supplied to this NJOY module.

    Returns
    -------
    str
        Serialized NJOY text.
    """
    return f"""leapr
-24
{mat_id} 1 {lat}
{mass_amu} 0.0253 1 {temperature}
{nalpha} {nbeta}
{alpha_grid_str}
{beta_grid_str}
{egrid_str}
{rho_str}
"""


def build_leapr_module(leapr_module_text: str) -> str:
    """Normalize one serialized LEAPR module for composition in a larger job.

    A terminal standalone ``stop`` card is removed so a following NJOY module
    can be appended.  Intermediate ``stop`` cards are rejected.

    Parameters
    ----------
    leapr_module_text : str
        Complete serialized LEAPR module body.

    Returns
    -------
    str
        Serialized NJOY text.
    """
    if not isinstance(leapr_module_text, str) or not leapr_module_text.strip():
        raise ValueError("leapr_module_text must be a non-empty string.")

    lines = leapr_module_text.strip().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()

    if lines and lines[-1].strip().lower() == "stop":
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()

    first_nonempty = next((line.strip().lower() for line in lines if line.strip()), None)
    if first_nonempty != "leapr":
        raise ValueError("leapr_module_text must begin with a LEAPR module.")

    if any(line.strip().lower() == "stop" for line in lines):
        raise ValueError(
            "leapr_module_text contains an intermediate 'stop' card. "
            "Only a terminal standalone stop may be removed."
        )

    return "\n".join(lines).rstrip() + "\n"


def build_thermr_module(
    input_tsl_tape,
    input_pendf_tape,
    output_tape,
    *,
    control_card,
    temperature_card,
    tolerance_card,
):
    """Serialize the NJOY THERMR module cards for the requested parameters.

    Parameters
    ----------
    input_tsl_tape : object
        Logical tape number containing thermal-scattering input.
    input_pendf_tape : object
        Logical tape number containing processed ENDF input.
    output_tape : object
        Logical tape number receiving module output.
    control_card : object
        Already formatted module-control card text.
    temperature_card : object
        Already formatted temperature card text.
    tolerance_card : object
        Already formatted numerical-tolerance card text.
    """
    return "".join((
        "thermr\n",
        # Use the shared free-format card writer while preserving this card’s terminator policy.
        format_njoy_card(input_tsl_tape, input_pendf_tape, output_tape),
        f"{control_card}\n",
        f"{temperature_card}\n",
        f"{tolerance_card}\n",
    ))


def build_acer_module(
    input_endf_tape: int,
    input_pendf_tape: int,
    output_dir_tape: int,
    output_ace_tape: int,
    output_xsd_tape: int,
    *,
    option_card: str,
    title_card: str,
    material_card: str,
    za_card: str,
    thermal_card: str,
) -> str:
    """Write one ACER module from explicit NJOY card payloads.

    Parameters
    ----------
    input_endf_tape : int
        Logical NJOY tape identifier used by this module.
    input_pendf_tape : int
        Logical tape number containing processed ENDF input.
    output_dir_tape : int
        Logical NJOY tape identifier used by this module.
    output_ace_tape : int
        Logical NJOY tape identifier used by this module.
    output_xsd_tape : int
        Logical NJOY tape identifier used by this module.
    option_card : str
        Preformatted NJOY card payload inserted without alteration.
    title_card : str
        Preformatted NJOY card payload inserted without alteration.
    material_card : str
        Preformatted NJOY card payload inserted without alteration.
    za_card : str
        Preformatted NJOY card payload inserted without alteration.
    thermal_card : str
        Preformatted NJOY card payload inserted without alteration.

    Returns
    -------
    str
        Serialized NJOY text.
    """
    return "".join((
        "acer\n",
        # Use the shared free-format card writer while preserving this card’s terminator policy.
        format_njoy_card(input_endf_tape, input_pendf_tape, output_dir_tape,
                         output_ace_tape, output_xsd_tape, terminated=False),
        f"{option_card}\n",
        f"'{title_card}'/\n",
        f"{material_card}\n",
        f"{za_card}\n",
        f"{thermal_card}\n",
    ))


def build_stop_card() -> str:
    """Write the terminal NJOY STOP card.

    Parameters
    ----------

    Returns
    -------
    str
        Serialized NJOY text.
    """
    return "stop\n"

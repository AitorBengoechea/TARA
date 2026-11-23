"""Element labels and approximate elemental masses for native LEAPR AWR checks.

Static standard atomic weights (amu); runtime has no chemistry-package dependency.
Mass agreement checks cannot unambiguously identify certain isotope/effective-mass
cases: refuse those instead of silently transferring a different species.
"""

from input_output.transformations.models import InputTransformationError

NEUTRON_MASS_AMU = 1.00866491595
ATOMIC_MASS_AMU = {
    'H': 1.008000,
    'He': 4.003000,
    'Li': 6.941000,
    'Be': 9.012000,
    'B': 10.812000,
    'C': 12.011000,
    'N': 14.007000,
    'O': 15.999000,
    'F': 18.998000,
    'Ne': 20.180000,
    'Na': 22.990000,
    'Mg': 24.305000,
    'Al': 26.982000,
    'Si': 28.086000,
    'P': 30.974000,
    'S': 32.067000,
    'Cl': 35.453000,
    'Ar': 39.948000,
    'K': 39.098000,
    'Ca': 40.078000,
    'Sc': 44.956000,
    'Ti': 47.867000,
    'V': 50.944000,
    'Cr': 51.996000,
    'Mn': 54.938000,
    'Fe': 55.845000,
    'Co': 58.933000,
    'Ni': 58.693000,
    'Cu': 63.546000,
    'Zn': 65.390000,
    'Ga': 69.723000,
    'Ge': 72.610000,
    'As': 74.922000,
    'Se': 78.960000,
    'Br': 79.904000,
    'Kr': 83.800000,
    'Rb': 85.468000,
    'Sr': 87.620000,
    'Y': 88.906000,
    'Zr': 91.224000,
    'Nb': 92.906000,
    'Mo': 95.940000,
    'Tc': 98.000000,
    'Ru': 101.070000,
    'Rh': 102.906000,
    'Pd': 106.420000,
    'Ag': 107.868000,
    'Cd': 112.412000,
    'In': 114.818000,
    'Sn': 118.711000,
    'Sb': 121.760000,
    'Te': 127.600000,
    'I': 126.904000,
    'Xe': 131.290000,
    'Cs': 132.905000,
    'Ba': 137.328000,
    'La': 138.906000,
    'Ce': 140.116000,
    'Pr': 140.908000,
    'Nd': 144.240000,
    'Pm': 145.000000,
    'Sm': 150.360000,
    'Eu': 151.964000,
    'Gd': 157.250000,
    'Tb': 158.925000,
    'Dy': 162.500000,
    'Ho': 164.930000,
    'Er': 167.260000,
    'Tm': 168.934000,
    'Yb': 173.040000,
    'Lu': 174.967000,
    'Hf': 178.490000,
    'Ta': 180.948000,
    'W': 183.840000,
    'Re': 186.207000,
    'Os': 190.230000,
    'Ir': 192.217000,
    'Pt': 195.078000,
    'Au': 196.967000,
    'Hg': 200.590000,
    'Tl': 204.383000,
    'Pb': 207.200000,
    'Bi': 208.980000,
    'Po': 209.000000,
    'At': 210.000000,
    'Rn': 222.000000,
    'Fr': 223.000000,
    'Ra': 226.000000,
    'Ac': 227.000000,
    'Th': 232.038000,
    'Pa': 231.036000,
    'U': 238.029000,
    'Np': 237.000000,
    'Pu': 244.000000,
    'Am': 243.000000,
    'Cm': 247.000000,
    'Bk': 247.000000,
    'Cf': 251.000000,
    'Es': 252.000000,
    'Fm': 257.000000,
    'Md': 258.000000,
    'No': 259.000000,
    'Lr': 262.000000,
    'Rf': 267.000000,
    'Db': 268.000000,
    'Sg': 269.000000,
    'Bh': 270.000000,
    'Hs': 269.000000,
    'Mt': 278.000000,
    'Ds': 281.000000,
    'Rg': 281.000000,
    'Cn': 285.000000,
    'Nh': 284.000000,
    'Fl': 289.000000,
    'Mc': 288.000000,
    'Lv': 293.000000,
    'Ts': 292.000000,
    'Og': 294.000000,
    'D': 2.014101778,
    'T': 3.016049282,
}

def parse_pdos_parameter(parameter):
    """Resolve e.g. C_pdos, O_pdos, H_pdos to their valid element label."""
    if not isinstance(parameter, str) or not parameter.endswith("_pdos"):
        raise InputTransformationError(f"Unsupported parameter {parameter!r}; expected <element>_pdos.")
    element = parameter[:-5]
    if element not in ATOMIC_MASS_AMU:
        raise InputTransformationError(f"Unrecognized element in parameter {parameter!r}.")
    return element

def principal_matches_element(awr, element, rtol=0.015):
    """Require plausible natural atomic weight AND unambiguous AWR identity.

    LEAPR gives relative atomic mass, not an element symbol. Isotope/effective
    masses or closely spaced elemental masses are not safely inferable.
    """
    expected = ATOMIC_MASS_AMU[element] / NEUTRON_MASS_AMU
    error = abs(awr - expected) / expected
    if error > rtol:
        raise InputTransformationError(
            f"LEAPR principal-scatterer AWR {awr:g} is not compatible with {element} "
            f"(expected about {expected:g}); cannot transfer {element}_pdos.")
    matches = [symbol for symbol, mass in ATOMIC_MASS_AMU.items()
               if abs(awr - mass/NEUTRON_MASS_AMU) / (mass/NEUTRON_MASS_AMU) <= rtol]
    # Restrict the disambiguation set to elements expected to be discernible.
    # If an isotope alias is almost equal to another element, reject.
    if len(matches) > 1:
        raise InputTransformationError(
            f"LEAPR AWR {awr:g} cannot distinguish {element} from {matches}; "
            "native LEAPR lacks an explicit element label.")

def element_from_awr(awr, rtol=0.015):
    """Identify the sole elemental label compatible with a mass ratio.

    Parameters
    ----------
    awr : float
        Dimensionless mass relative to the neutron mass.
    rtol : float, optional
        Relative tolerance for matching tabulated elemental mass ratios.

    Returns
    -------
    str or None
        Unique matching label, or None if no label matches or several match.
    """
    matches = [symbol for symbol, mass in ATOMIC_MASS_AMU.items()
               if abs(awr - mass/NEUTRON_MASS_AMU) / (mass/NEUTRON_MASS_AMU) <= rtol]
    return matches[0] if len(matches) == 1 else None

"""NCrystal-only NCMAT species isolation used during transient analysis.

This is *not* a LEAPR/NCMAT input transformation.  It modifies a temporary
NCrystal material just before loading it, retaining every original fraction.
"""

from collections.abc import Collection
import re
from typing import FrozenSet

_SECTION_RE = re.compile(r"^\s*@([A-Za-z][A-Za-z0-9_]*)\b")
_FIELD_RE = re.compile(r"^\s*(element|fraction|type)\s+([^#\s]+)")
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def normalize_active_species(active_species: Collection[str]) -> FrozenSet[str]:
    """Validate and normalize NCMAT ``element`` labels.

    A bare string is rejected to prevent accidentally iterating its letters.
    Multiple ``@DYNINFO`` blocks can have the same element label; all are
    selected together.  Labels are case-sensitive, as in NCMAT.

    Parameters
    ----------
    active_species : collection of str
        Requested case-sensitive NCMAT element labels.

    Returns
    -------
    frozenset of str
        Unique validated labels.

    Raises
    ------
    TypeError
        If the input is a bare string or is not a collection.
    ValueError
        If the collection is empty or contains invalid labels.
    """
    if isinstance(active_species, (str, bytes)) or not isinstance(active_species, Collection):
        raise TypeError("active_species must be a nonempty collection of NCMAT element labels")
    if not active_species or any(
        not isinstance(label, str) or not _LABEL_RE.fullmatch(label)
        for label in active_species
    ):
        raise ValueError("active_species must contain valid, nonempty NCMAT element labels")
    return frozenset(active_species)


def isolate_ncmat_dyninfo(ncmat_text: str, active_species: Collection[str]) -> str:
    """Sterilize non-selected species' complete ``@DYNINFO`` blocks.

    Preserves NCMAT headers, other sections, original element identifiers, and
    exact fraction lines.  The original source material must not be modified:
    callers should apply this to generated, transient NCMAT text only.

    Parameters
    ----------
    ncmat_text : str
        Generated NCMAT content.
    active_species : collection of str
        Element labels to retain as scattering-active.

    Returns
    -------
    str
        NCMAT content with original fractions and inactive species made sterile.

    Raises
    ------
    ValueError
        If the NCMAT has malformed dynamics sections, an unknown selected
        species, or a selected species that is already sterile.
    TypeError
        If the input text or the selected-label collection has the wrong type.
    """
    active = normalize_active_species(active_species)
    if not isinstance(ncmat_text, str):
        raise TypeError("ncmat_text must be a string")

    lines = ncmat_text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if _SECTION_RE.match(line)]
    spans = [(start, starts[pos + 1] if pos + 1 < len(starts) else len(lines))
             for pos, start in enumerate(starts)]
    seen = set()
    result = []
    cursor = 0
    n_dyninfo = 0

    for start, end in spans:
        match = _SECTION_RE.match(lines[start])
        if match.group(1).upper() != "DYNINFO":
            continue
        n_dyninfo += 1
        result.extend(lines[cursor:start])
        block = lines[start:end]
        fields, raw_fields = _read_dyninfo_fields(block)
        label = fields['element']
        seen.add(label)
        if label in active:
            if fields['type'].lower() == 'sterile':
                raise ValueError(f"Selected element {label!r} is already sterile")
            result.extend(block)
        else:
            # Remove *all* old dynamics properties, not only the type line.
            # Keep the original @DYNINFO header, element and fraction lines.
            newline = '\r\n' if lines[start].endswith('\r\n') else '\n'
            result.extend((lines[start], raw_fields['element'],
                           raw_fields['fraction'], f'  type sterile{newline}'))
        cursor = end

    if not n_dyninfo:
        raise ValueError("NCMAT has no @DYNINFO blocks to isolate")
    unknown = active - seen
    if unknown:
        raise ValueError(f"Unknown active_species in NCMAT: {sorted(unknown)!r}; available: {sorted(seen)!r}")
    result.extend(lines[cursor:])
    return ''.join(result)


def _read_dyninfo_fields(block):
    """Read the required dynamics fields and their original lines.

    Parameters
    ----------
    block : sequence of str
        NCMAT DYNINFO lines, beginning with the section header.

    Returns
    -------
    tuple of dict
        Parsed values and original lines indexed by field name.
    """
    fields = {}
    raw_fields = {}
    for line in block[1:]:
        field_match = _FIELD_RE.match(line)
        if field_match:
            key, value = field_match.groups()
            if key in fields:
                raise ValueError(f"Repeated {key!r} in NCMAT @DYNINFO")
            fields[key] = value
            raw_fields[key] = line
    if not {'element', 'fraction', 'type'} <= fields.keys():
        raise ValueError("Each NCMAT @DYNINFO needs element, fraction, and type")
    return fields, raw_fields

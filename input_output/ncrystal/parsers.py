"""
Low-level parsing utilities for NCrystal .ncmat files.
"""
import re
from typing import List, Dict, Tuple, Any

from input_output.ncrystal.exceptions import NCrystalFormatError

# NCMAT array fields permit compact repeated values, e.g. ``0r72`` denotes
# exactly 72 zeros. Do not use FLOAT_REGEX on array values: that would turn
# ``0r72`` into the two distinct numbers 0 and 72.
_NCMAT_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NCMAT_ARRAY_TOKEN = re.compile(rf"(?P<value>{_NCMAT_NUMBER})(?:r(?P<count>[1-9]\d*))?\Z")
_MAX_ARRAY_SIZE = 2_000_000

# Regex for extracting numeric values
FLOAT_REGEX = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def chunk_blocks(lines: List[str]) -> Dict[str, List[str]]:
    """
    Chunks raw file lines into NCrystal blocks identified by '@' tags.

    Comments starting with '#' are stripped, and empty lines are ignored.
    If multiple blocks have the same name (e.g., multiple @DYNINFO blocks),
    they are aggregated into a list of lists of lines, mapped to the tag.

    Parameters
    ----------
    lines : list of str
        The raw lines read from the .ncmat file.

    Returns
    -------
    dict
        A dictionary mapping block names (without '@') to their line contents.
        Values are lists of strings (for single blocks) or lists of lists of 
        strings (for repeating blocks).
    """
    blocks: Dict[str, Any] = {}
    current_tag = None
    current_lines: List[str] = []

    for raw_line in lines:
        line = raw_line.split('#')[0].strip()
        if not line:
            continue
            
        if line.startswith('@'):
            if current_tag is not None:
                _store_block(blocks, current_tag, current_lines)
                
            current_tag = line[1:].strip()
            current_lines = []
        elif current_tag is not None:
            current_lines.append(line)
            
    # Store the final block
    if current_tag is not None:
        _store_block(blocks, current_tag, current_lines)
        
    return blocks


def _store_block(blocks: Dict[str, Any], tag: str, lines: List[str]) -> None:
    """
    Helper to store block lines in the dictionary, handling repeated tags.

    Parameters
    ----------
    blocks : dict
        The dictionary storing block data.
    tag : str
        The block identifier.
    lines : list of str
        The parsed lines belonging to the block.
    """
    if tag in blocks:
        if isinstance(blocks[tag][0], str):
            blocks[tag] = [blocks[tag]]
        blocks[tag].append(lines)
    else:
        blocks[tag] = lines


def extract_floats(line: str) -> List[float]:
    """
    Extracts all floating point numbers from a given string.

    Parameters
    ----------
    line : str
        The text line to parse.

    Returns
    -------
    list of float
        A list of extracted numeric values.
    """
    return [float(x) for x in FLOAT_REGEX.findall(line)]


def expand_ncmat_array_values(line: str) -> List[float]:
    """Parse an NCMAT numeric-array line, expanding ``<value>r<count>``.

    Parameters
    ----------
    line : str
        Whitespace-separated NCMAT numeric tokens, without the field name.

    Returns
    -------
    list of float
        Expanded array values in native order.

    Raises
    ------
    NCrystalFormatError
        If a token or repetition count is invalid, or expansion exceeds the
        defensive maximum array length.

    Notes
    -----
    The ``r`` count is the *total* number of copies: ``0r3`` becomes three
    zeros, not four. Array parsing intentionally rejects unrecognised tokens
    instead of silently extracting incidental digits from them.
    """
    values: List[float] = []
    for token in line.split():
        match = _NCMAT_ARRAY_TOKEN.fullmatch(token)
        if match is None:
            raise NCrystalFormatError(f"Invalid NCMAT numeric-array token: {token!r}")
        value = float(match.group("value"))
        if not (-float("inf") < value < float("inf")):
            raise NCrystalFormatError(f"Nonfinite NCMAT array value: {token!r}")
        count = int(match.group("count") or 1)
        if len(values) + count > _MAX_ARRAY_SIZE:
            raise NCrystalFormatError("NCMAT array exceeds supported safety limit")
        values.extend([value] * count)
    return values

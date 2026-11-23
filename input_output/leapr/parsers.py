"""
Low-level parsing utilities for NJOY LEAPR input files.
"""
import re
from typing import List, TextIO

from input_output.leapr.exceptions import LeaprFormatError

# Robust regex for floats and scientific notation
FLOAT_REGEX = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def read_data_block(f: TextIO) -> List[float]:
    """
    Reads a standard LEAPR data block terminated by '/'.

    This function handles multi-line data blocks (e.g., alpha/beta grids, 
    spectra) and extracts all numerical values.

    Parameters
    ----------
    f : TextIO
        The active file object being read.

    Returns
    -------
    List[float]
        A list of float values extracted from the data block.

    Raises
    ------
    EOFError
        If the file ends before the '/' terminator is found.
    """
    data_list = []

    while True:
        line = f.readline()
        if not line:
            raise EOFError("Reached end of file while reading data block.")

        # Discard inline comments before interpreting the NJOY free-format numerical tokens.
        line_no_comment = line.split('!')[0].split('#')[0]
        terminator_found = '/' in line_no_comment

        if terminator_found:
            # Stop at the free-format slash terminator so subsequent text is not parsed as data.
            line_to_process = line_no_comment.split('/', 1)[0]
        else:
            # Stop at the free-format slash terminator so subsequent text is not parsed as data.
            line_to_process = line_no_comment

        values = FLOAT_REGEX.findall(line_to_process)
        data_list.extend(map(float, values))

        if terminator_found:
            break
    return data_list


def read_fixed_length_array(f: TextIO, count: int, *, card: str) -> List[float]:
    """Read exactly ``count`` NJOY numeric array values.

    NJOY free-format arrays may be terminated by ``/`` or simply by their
    declared element count. Do not consume the next control card if it does
    not contain a slash on the final array line. A standalone ``/`` on the
    following line is also accepted for backwards compatibility.

    Parameters
    ----------
    f : TextIO
        Seekable file object at the start of the array card.
    count : int
        Number of data values specified by the control card.
    card : str
        Name used for diagnostic errors.

    Returns
    -------
    list of float
        The exact values of the declared array.

    Raises
    ------
    LeaprFormatError
        If an early terminator or extra values occur on an array line.
    EOFError
        If the array is truncated.
    """
    if count < 0:
        raise LeaprFormatError(f"{card}: array length cannot be negative.")
    values: List[float] = []
    while len(values) < count:
        line = f.readline()
        if not line:
            raise EOFError(f"{card}: expected {count} values, got {len(values)}.")
        body = line.split('!')[0].split('#')[0]
        terminated = '/' in body
        tokens = FLOAT_REGEX.findall(body.split('/', 1)[0])
        remaining = count - len(values)
        if len(tokens) > remaining:
            raise LeaprFormatError(
                f"{card}: ni is {count}; found {len(tokens)-remaining} extra "
                "value(s) on the final array line."
            )
        values.extend(float(token) for token in tokens)
        if terminated and len(values) != count:
            raise LeaprFormatError(
                f"{card}: '/' before {count} values; got {len(values)}."
            )
        if terminated:
            return values

    # A separate slash after the last array value is optional. Peek only;
    # never consume the Card 13/16/10 that follows an unterminated array.
    position = f.tell()
    following = f.readline()
    if following:
        content = following.split('!')[0].split('#')[0].strip()
        if not content.startswith('/'):
            f.seek(position)
    return values


def read_phonon_spectrum_block(f: TextIO, ni: int) -> List[float]:
    """Read exactly ``ni`` phonon density values from LEAPR Card 12.

    NJOY's Card 12 can omit a trailing slash when the stated ``ni`` values
    have been read, as in the CAB multi-temperature H(H2O) evaluation.
    """
    return read_fixed_length_array(f, ni, card='Card 12 phonon spectrum')

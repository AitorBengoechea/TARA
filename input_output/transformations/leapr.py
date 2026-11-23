"""LEAPR Cards 11/12 VDOS adapter; preserves all unrelated cards verbatim.

Handles positive-temperature blocks and negative-temperature reuse cards.
Card-12 values can be count-terminated or slash-terminated. The reader
tracks declared array lengths rather than guessing delimiters from comments.
"""
import re
import numpy as np
from input_output.transformations.models import ContinuousSpectrum, InputTransformationError
from input_output.transformations.spectra import to_leapr_grid
from input_output.transformations.elements import parse_pdos_parameter, principal_matches_element, element_from_awr

_NUM = re.compile(r'(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?')


def _values(line):
    """Read numeric fields before comments or a slash, including D exponents."""
    return [float(s.replace('D', 'E').replace('d', 'e'))
            for s in _NUM.findall(line.split('!', 1)[0].split('#', 1)[0].split('/', 1)[0])]


class _Cursor:
    """Read LEAPR cards in order while retaining original lines and offsets."""
    def __init__(self, text):
        """Start at the first line of the LEAPR source text."""
        self.lines = text.splitlines(keepends=True)
        self.at = 0

    def line(self, card):
        """Consume one line and return its zero-based index and original text."""
        if self.at >= len(self.lines):
            raise InputTransformationError(f'LEAPR truncated while reading {card}.')
        i = self.at
        self.at += 1
        return i, self.lines[i]

    def control(self, card):
        """Consume a slash-terminated control card and return its index and values."""
        i, line = self.line(card)
        if '/' not in line.split('!', 1)[0].split('#', 1)[0]:
            raise InputTransformationError(f'{card} must be slash-terminated.')
        return i, _values(line)

    def array(self, card, count=None):
        """Consume a LEAPR array and record its span in the original text.

        Parameters
        ----------
        card : str
            Card label for diagnostics.
        count : int or None, optional
            Declared number of values. Without a count, require a slash terminator.

        Returns
        -------
        start, end : int
            Zero-based line offsets delimiting a half-open span.
        values : numpy.ndarray
            Parsed card values in source order.

        Raises
        ------
        InputTransformationError
            If the array is truncated or its length conflicts with the declared count.
        """
        start = self.at
        values = []
        terminated = False
        while count is None or len(values) < count:
            i, line = self.line(card)
            part = line.split('!', 1)[0].split('#', 1)[0]
            nums = _values(line)
            if count is not None and len(values) + len(nums) > count:
                raise InputTransformationError(f'{card} contains more than declared {count} values.')
            values.extend(nums)
            if '/' in part:
                terminated = True
                if count is not None and len(values) < count:
                    raise InputTransformationError(f'{card} ended before declared {count} values.')
                break
            if count is None and self.at >= len(self.lines):
                raise InputTransformationError(f'{card} lacks a slash terminator.')
        if count is not None and not terminated and self.at < len(self.lines):
            if self.lines[self.at].split('!', 1)[0].split('#', 1)[0].strip() == '/':
                self.at += 1
        if count is None and not terminated:
            raise InputTransformationError(f'{card} lacks a slash terminator.')
        return start, self.at, np.asarray(values, dtype=float)


def _parse_blocks(text):
    """Read normalized phonon spectra and edit locations from LEAPR text.

    Parameters
    ----------
    text : str
        LEAPR module source.

    Returns
    -------
    lines : list of str
        Original source lines with their line endings.
    blocks : list of tuple
        For each positive-temperature block, the Card 11 line index, the
        half-open Card 12 line span, and the normalized ContinuousSpectrum.
        Card 11 supplies the uniform energy spacing in eV.

    Raises
    ------
    InputTransformationError
        If the supported card layout or a spectrum is invalid, or there is
        no positive-temperature spectrum.

    Notes
    -----
    Negative-temperature reuse cards contribute no separate spectrum.
    """
    cur = _Cursor(text)
    _, module = cur.line('module')
    if module.strip().lower() != 'leapr':
        raise InputTransformationError('Not a LEAPR module input.')
    cur.line('file unit')
    cur.line('title')
    _, run = cur.control('Card 3')
    if len(run) < 3 or run[0] <= 0 or int(run[0]) != run[0]:
        raise InputTransformationError('Invalid LEAPR Card 3 temperature count.')
    ntemp = int(run[0])
    cur.control('Card 4')
    _, principal = cur.control('Card 5')
    if len(principal) < 3:
        raise InputTransformationError('Missing LEAPR principal-scatterer information.')
    nsk = int(principal[5]) if len(principal) > 5 else 0
    cur.control('Card 6')
    cur.control('Card 7')
    cur.array('Card 8 alpha grid')
    cur.array('Card 9 beta grid')
    blocks = []
    for index in range(ntemp):
        _, temp = cur.control(f'Card 10 temperature #{index+1}')
        if not temp or temp[0] == 0:
            raise InputTransformationError('LEAPR zero/empty temperature card.')
        if temp[0] < 0:
            continue
        control_idx, control = cur.control('Card 11 phonon control')
        if len(control) < 2 or control[0] <= 0 or control[1] < 3 or int(control[1]) != control[1]:
            raise InputTransformationError('Invalid LEAPR Card 11 delta/ni.')
        delta, count = control[0], int(control[1])
        begin, end, values = cur.array('Card 12 phonon density', count)
        if len(values) != count:
            raise InputTransformationError('LEAPR Card 12 wrong declared count.')
        cur.control('Card 13 translational parameters')
        _, oscillators = cur.control('Card 14 oscillator count')
        if not oscillators or oscillators[0] < 0 or int(oscillators[0]) != oscillators[0]:
            raise InputTransformationError('Invalid LEAPR Card 14 oscillator count.')
        nd = int(oscillators[0])
        if nd:
            cur.array('Card 15 oscillator energies', nd)
            cur.array('Card 16 oscillator weights', nd)
        if nsk:
            _, data = cur.control('Card 17 structure-factor control')
            if len(data) < 2 or int(data[0]) < 1:
                raise InputTransformationError('Invalid LEAPR Card 17.')
            cur.array('Card 18 static structure factor')
            if nsk == 2:
                cur.control('Card 19 coherent fraction')
        blocks.append((control_idx, begin, end, ContinuousSpectrum(
            np.arange(count, dtype=float) * delta, values)))
    if not blocks:
        raise InputTransformationError('LEAPR input has no positive-temperature phonon spectrum.')
    return cur.lines, blocks


def _principal_awr(text):
    """Read the dimensionless principal-scatterer mass ratio from Card 5."""
    cur = _Cursor(text)
    for card in ('module', 'unit', 'title'):
        cur.line(card)
    cur.control('Card 3')
    cur.control('Card 4')
    _, values = cur.control('Card 5 principal scatterer')
    if len(values) < 1 or values[0] <= 0:
        raise InputTransformationError('Invalid LEAPR principal-scatterer AWR.')
    return values[0]


class LeaprAdapter:
    """Read and replace principal-scatterer phonon spectra in LEAPR text."""
    engine = 'njoy'
    suffixes = ('.leapr',)

    def available_parameters(self, text):
        """Discover the principal-scatterer VDOS selector.

        Parameters
        ----------
        text : str
            LEAPR module source.

        Returns
        -------
        list of str
            One ``<element>_pdos`` selector if the mass ratio identifies a
            unique label, or an empty list otherwise.

        Raises
        ------
        InputTransformationError
            If the supported LEAPR cards or phonon spectrum fail validation.
        """
        _parse_blocks(text)
        element = element_from_awr(_principal_awr(text))
        return [f'{element}_pdos'] if element is not None else []

    def _check_principal(self, text, parameter):
        """Resolve a VDOS selector and check it against the principal mass ratio."""
        element = parse_pdos_parameter(parameter)
        principal_matches_element(_principal_awr(text), element)
        return element

    def extract(self, text, parameter):
        """Extract the normalized VDOS of the LEAPR principal scatterer.

        Parameters
        ----------
        text : str
            LEAPR module source.
        parameter : str
            Principal-scatterer selector of the form ``<element>_pdos``.

        Returns
        -------
        ContinuousSpectrum
            Uniform energy grid starting at zero in eV and unit-area density
            in inverse eV.

        Raises
        ------
        InputTransformationError
            If the selector is invalid, the principal species is inconsistent or
            ambiguous, or the phonon cards do not provide one unambiguous VDOS.

        Notes
        -----
        Positive-temperature blocks must have identical energy grids and
        normalized densities that agree within the comparison tolerance. The
        first spectrum represents all such blocks.
        """
        self._check_principal(text, parameter)
        _, blocks = _parse_blocks(text)
        first = blocks[0][3]
        # One source parameter cannot silently mean multiple different
        # temperature-dependent source spectra.
        for _, _, _, other in blocks[1:]:
            if (not np.array_equal(first.energy_ev, other.energy_ev)
                    or not np.allclose(first.density_per_ev, other.density_per_ev,
                                       rtol=1e-8, atol=1e-12)):
                raise InputTransformationError('LEAPR source has distinct positive-temperature H spectra; single species PDOS is ambiguous.')
        return first

    def replace(self, text, parameter, spectrum):
        """Replace the phonon cards for the LEAPR principal scatterer.

        Parameters
        ----------
        text : str
            Target LEAPR module source.
        parameter : str
            Principal-scatterer selector of the form ``<element>_pdos``.
        spectrum : ContinuousSpectrum
            Unit-area VDOS on a uniform energy grid in eV. A positive-origin
            input is converted to a zero-origin grid using the acoustic tail.

        Returns
        -------
        str
            New LEAPR text with Cards 11 and 12 replaced in every positive-
            temperature block. Other cards, including reuse cards, are preserved.

        Raises
        ------
        InputTransformationError
            If the selector, target cards, or spectrum grid fail validation.
        """
        self._check_principal(text, parameter)
        lines, blocks = _parse_blocks(text)
        converted = to_leapr_grid(spectrum) if spectrum.energy_ev[0] > 0 else spectrum
        e, g = converted.energy_ev, converted.density_per_ev
        if not np.isclose(e[0], 0, atol=1e-12):
            raise InputTransformationError('LEAPR source spectrum does not begin at zero.')
        delta = e[1] - e[0]
        if not np.allclose(np.diff(e), delta, rtol=1e-6, atol=1e-11):
            raise InputTransformationError('LEAPR requires uniform spectrum energy spacing.')
        edits = []
        for control, start, end, _ in blocks:
            original = lines[control]
            # Only Card 11's delta/ni values are changed; retain its slash,
            # free-form trailing notes and native newline convention.
            slash_pos = original.index('/')
            newline = '\r\n' if original.endswith('\r\n') else '\n'
            replacement_control = f'{delta:.16e} {len(g)}/' + original[slash_pos+1:]
            array_text = ''.join(' '.join(f'{v:.12e}' for v in g[i:i+5]) + newline
                                 for i in range(0, len(g), 5))
            # Preserve a final slash on Card 12 if present in the source.
            last = lines[end-1].rstrip('\r\n')
            content = last.split('!', 1)[0].split('#', 1)[0]
            if '/' in content:
                # Preserve Card 12's slash and any trailing / DOS annotation.
                array_text = array_text[:-len(newline)] + last[last.index('/'):] + newline
            edits.extend([(control, control+1, replacement_control), (start, end, array_text)])
        for a, b, replacement in sorted(edits, reverse=True):
            lines[a:b] = [replacement]
        return ''.join(lines)

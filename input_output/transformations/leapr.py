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
    """Read numeric fields in a LEAPR card, including Fortran exponents.

    Parameters
    ----------
    line : str
        Raw card text with optional comments and slash terminator.

    Returns
    -------
    list of float
        Values preceding the first comment or slash.
    """
    return [float(s.replace('D', 'E').replace('d', 'e'))
            for s in _NUM.findall(line.split('!', 1)[0].split('#', 1)[0].split('/', 1)[0])]


class _Cursor:
    """Read LEAPR cards in order while retaining original lines and offsets."""
    def __init__(self, text):
        """Initialize the cursor at the start of a LEAPR module.

        Parameters
        ----------
        text : str
            Full source text, including original line terminators.
        """
        self.lines = text.splitlines(keepends=True)
        self.at = 0

    def line(self, card):
        """Consume the next source line.

        Parameters
        ----------
        card : str
            Card label used in truncation diagnostics.

        Returns
        -------
        tuple of int and str
            Zero-based line index and original line text.

        Raises
        ------
        InputTransformationError
            If the cursor has reached the end of the source.
        """
        if self.at >= len(self.lines):
            raise InputTransformationError(f'LEAPR truncated while reading {card}.')
        i = self.at
        self.at += 1
        return i, self.lines[i]

    def control(self, card):
        """Consume one slash-terminated LEAPR control card.

        Parameters
        ----------
        card : str
            Card label used in format diagnostics.

        Returns
        -------
        tuple of int and list of float
            Original line index and numeric values.

        Raises
        ------
        InputTransformationError
            If the card is absent or lacks a slash terminator.
        """
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


def _read_spectrum_header(cur):
    """Read the fixed LEAPR cards before the first temperature record.

    Parameters
    ----------
    cur : _Cursor
        Cursor positioned at the module declaration.

    Returns
    -------
    tuple of int
        Temperature count and static structure-factor option.

    Raises
    ------
    InputTransformationError
        If a required header card is missing or invalid.
    """
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
    return ntemp, nsk


def _read_positive_block(cur, nsk):
    """Read a temperature block's spectrum and retain its edit positions.

    Parameters
    ----------
    cur : _Cursor
        Cursor positioned at Card 11.
    nsk : int
        Static structure-factor option.

    Returns
    -------
    tuple
        Card 11 index, Card 12 start and end indices, and normalized spectrum.

    Raises
    ------
    InputTransformationError
        If the phonon or auxiliary cards are invalid.
    """
    control_idx, begin, end, delta, count, values = _read_phonon_cards(cur)
    _consume_oscillator_cards(cur)
    if nsk:
        _consume_structure_factor_cards(cur, nsk)

    spectrum = ContinuousSpectrum(np.arange(count, dtype=float) * delta, values)
    return control_idx, begin, end, spectrum


def _read_phonon_cards(cur):
    """Read the continuum grid and density from Cards 11 and 12.

    Parameters
    ----------
    cur : _Cursor
        Cursor positioned at Card 11.

    Returns
    -------
    tuple
        Card 11 index, half-open Card 12 line span, spacing in eV, point
        count, and tabulated density values.

    Raises
    ------
    InputTransformationError
        If the grid controls or density count are invalid.
    """
    control_idx, control = cur.control('Card 11 phonon control')
    if len(control) < 2 or control[0] <= 0 or control[1] < 3 or int(control[1]) != control[1]:
        raise InputTransformationError('Invalid LEAPR Card 11 delta/ni.')
    delta, count = control[0], int(control[1])
    begin, end, values = cur.array('Card 12 phonon density', count)
    if len(values) != count:
        raise InputTransformationError('LEAPR Card 12 wrong declared count.')
    return control_idx, begin, end, delta, count, values


def _consume_oscillator_cards(cur):
    """Advance past the translation controls and any discrete oscillator data.

    Parameters
    ----------
    cur : _Cursor
        Cursor positioned at Card 13.

    Raises
    ------
    InputTransformationError
        If the oscillator count or card layout is invalid.
    """
    cur.control('Card 13 translational parameters')
    _, oscillators = cur.control('Card 14 oscillator count')
    if not oscillators or oscillators[0] < 0 or int(oscillators[0]) != oscillators[0]:
        raise InputTransformationError('Invalid LEAPR Card 14 oscillator count.')
    nd = int(oscillators[0])
    if nd:
        cur.array('Card 15 oscillator energies', nd)
        cur.array('Card 16 oscillator weights', nd)


def _consume_structure_factor_cards(cur, nsk):
    """Advance past the static structure factor and optional coherent fraction.

    Parameters
    ----------
    cur : _Cursor
        Cursor positioned at Card 17.
    nsk : int
        Static structure-factor flag. A value of two enables Card 19.

    Raises
    ------
    InputTransformationError
        If the structure-factor cards are invalid.
    """
    _, data = cur.control('Card 17 structure-factor control')
    if len(data) < 2 or int(data[0]) < 1:
        raise InputTransformationError('Invalid LEAPR Card 17.')
    cur.array('Card 18 static structure factor')
    if nsk == 2:
        cur.control('Card 19 coherent fraction')


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
    ntemp, nsk = _read_spectrum_header(cur)
    blocks = []
    for index in range(ntemp):
        _, temp = cur.control(f'Card 10 temperature #{index+1}')
        if not temp or temp[0] == 0:
            raise InputTransformationError('LEAPR zero/empty temperature card.')
        if temp[0] < 0:
            continue
        blocks.append(_read_positive_block(cur, nsk))
    if not blocks:
        raise InputTransformationError('LEAPR input has no positive-temperature phonon spectrum.')
    return cur.lines, blocks


def _principal_awr(text):
    """Read the principal scatterer mass ratio from LEAPR Card 5.

    Parameters
    ----------
    text : str
        Native LEAPR module text.

    Returns
    -------
    float
        Principal scatterer mass relative to the neutron mass.
    """
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
        """Validate a VDOS selector against the principal scatterer mass.

        Parameters
        ----------
        text : str
            Native LEAPR module text.
        parameter : str
            Selector of the form `<element>_pdos`.

        Returns
        -------
        str
            Validated principal-scatterer element label.

        Raises
        ------
        InputTransformationError
            If the selector is invalid or the mass does not identify the named element.
        """
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
        delta, density = _prepare_replacement_grid(spectrum)
        # Work backwards so a longer Card 12 array leaves earlier offsets valid.
        for control, start, end, _ in reversed(blocks):
            control_text, density_text = _format_replacement_cards(
                lines[control], lines[end - 1], delta, density,
            )
            lines[start:end] = [density_text]
            lines[control:control + 1] = [control_text]
        return ''.join(lines)


def _prepare_replacement_grid(spectrum):
    """Convert the source spectrum to a uniform LEAPR grid starting at zero.

    Parameters
    ----------
    spectrum : ContinuousSpectrum
        Source vibrational density and energy grid in eV.

    Returns
    -------
    delta : float
        Uniform energy spacing in eV.
    density : numpy.ndarray
        Density values on the converted grid.

    Raises
    ------
    InputTransformationError
        If the resulting grid has a nonzero origin or nonuniform spacing.
    """
    converted = to_leapr_grid(spectrum) if spectrum.energy_ev[0] > 0 else spectrum
    e, g = converted.energy_ev, converted.density_per_ev
    if not np.isclose(e[0], 0, atol=1e-12):
        raise InputTransformationError('LEAPR source spectrum does not begin at zero.')
    delta = e[1] - e[0]
    if not np.allclose(np.diff(e), delta, rtol=1e-6, atol=1e-11):
        raise InputTransformationError('LEAPR requires uniform spectrum energy spacing.')
    return delta, g


def _format_replacement_cards(control_line, last_density_line, delta, density):
    """Format Cards 11 and 12 while retaining their trailing annotations.

    Parameters
    ----------
    control_line, last_density_line : str
        Original Card 11 line and final line of Card 12.
    delta : float
        Replacement energy spacing in eV.
    density : numpy.ndarray
        Replacement density values on the uniform grid.

    Returns
    -------
    tuple of str
        Replacement Card 11 line and multiline Card 12 text.
    """
    slash_pos = control_line.index('/')
    newline = '\r\n' if control_line.endswith('\r\n') else '\n'
    control_text = f'{delta:.16e} {len(density)}/' + control_line[slash_pos + 1:]
    density_text = ''.join(
        ' '.join(f'{value:.12e}' for value in density[i:i + 5]) + newline
        for i in range(0, len(density), 5)
    )

    # Only a slash before the comment marker terminates Card 12.
    last = last_density_line.rstrip('\r\n')
    content = last.split('!', 1)[0].split('#', 1)[0]
    if '/' in content:
        density_text = density_text[:-len(newline)] + last[last.index('/'):] + newline
    return control_text, density_text

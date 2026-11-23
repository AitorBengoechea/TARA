"""Surgical edits of NCMAT @DYNINFO species VDOS, preserving other blocks."""
import re
import numpy as np
from input_output.transformations.models import ContinuousSpectrum, InputTransformationError
from input_output.transformations.spectra import to_ncrystal_grid
from input_output.transformations.elements import parse_pdos_parameter

_REPEAT = re.compile(r'(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)r(?P<count>\d+)$', re.I)
_GRID = re.compile(r'(?m)^(?P<leading>\s*vdos_egrid\s+)[^\n]+')
_DENS = re.compile(r'(?m)^(?P<leading>\s*vdos_density\s+)[\s\S]*?(?=^\s*(?:vdos_egrid|temperature|type|element|fraction)\b|\Z)')


def _blocks(text):
    """Yield each DYNINFO block with its half-open character span in the source."""
    marker = re.compile(r'(?m)^@DYNINFO\b')
    for match in marker.finditer(text):
        end = re.search(r'(?m)^@\w+', text[match.end():])
        stop = match.end() + end.start() if end else len(text)
        yield match.start(), stop, text[match.start():stop]


def _block(text, symbol):
    """Return the unique VDOS DYNINFO block and source span for a species."""
    candidates = []
    for start, end, block in _blocks(text):
        element = re.search(r'(?m)^\s*element\s+(\S+)', block)
        if element and element.group(1) == symbol:
            candidates.append((start, end, block))
    if len(candidates) != 1:
        raise InputTransformationError(f'Expected exactly one NCMAT @DYNINFO for {symbol}; found {len(candidates)}.')
    start, end, block = candidates[0]
    dtype = re.search(r'(?m)^\s*type\s+(\S+)', block)
    if dtype is None or dtype.group(1).lower() != 'vdos':
        raise InputTransformationError(f'NCMAT {symbol} dynamics are not a VDOS; refusing unsupported model conversion.')
    return start, end, block


def _read_vdos(block):
    """Read a VDOS block into a normalized continuous spectrum.

    Parameters
    ----------
    block : str
        NCMAT DYNINFO text with two energy bounds in eV and at least three
        density values after expanding repetition tokens.

    Returns
    -------
    ContinuousSpectrum
        Uniform energy coordinates between the bounds and unit-area density
        in inverse eV.

    Raises
    ------
    InputTransformationError
        If fields or density values are invalid or the spectrum fails validation.
    ValueError
        If either energy bound cannot be parsed as a number.
    """
    grid = _GRID.search(block)
    dens = _DENS.search(block)
    if grid is None or dens is None:
        raise InputTransformationError('NCMAT VDOS requires vdos_egrid and vdos_density.')
    bounds = grid.group().split()[1:]
    if len(bounds) != 2:
        raise InputTransformationError('Expected NCMAT vdos_egrid Emin Emax.')
    floats = []
    for token in dens.group()[len(dens.group('leading')):].split():
        repeat = _REPEAT.fullmatch(token)
        if repeat:
            floats.extend([float(repeat.group('value'))] * int(repeat.group('count')))
        else:
            try:
                floats.append(float(token))
            except ValueError as exc:
                raise InputTransformationError(f'Invalid NCMAT VDOS value {token!r}.') from exc
    if len(floats) < 3:
        raise InputTransformationError('NCMAT VDOS has too few points.')
    energies = np.linspace(float(bounds[0]), float(bounds[1]), len(floats))
    return ContinuousSpectrum(energies, np.asarray(floats))


class NCrystalAdapter:
    """Read and replace a selected species VDOS in NCMAT text."""
    engine = 'ncrystal'
    suffixes = ('.ncmat',)

    def available_parameters(self, text):
        """Discover species selectors in NCMAT VDOS blocks.

        Parameters
        ----------
        text : str
            NCMAT source text.

        Returns
        -------
        list of str
            Selectors for labels that occur once among VDOS DYNINFO blocks.

        Raises
        ------
        InputTransformationError
            If a VDOS block has an unsupported species label.
        """
        seen = {}
        for _, _, block in _blocks(text):
            element_match = re.search(r'(?m)^\s*element\s+(\S+)', block)
            type_match = re.search(r'(?m)^\s*type\s+(\S+)', block)
            if not element_match or not type_match or type_match.group(1).lower() != 'vdos':
                continue
            element = element_match.group(1)
            parse_pdos_parameter(f'{element}_pdos')
            seen[element] = seen.get(element, 0) + 1
        return [f'{symbol}_pdos' for symbol in seen if seen[symbol] == 1]

    def extract(self, text, parameter):
        """Extract the normalized VDOS of one NCMAT species.

        Parameters
        ----------
        text : str
            NCMAT source text.
        parameter : str
            Species selector of the form ``<element>_pdos``.

        Returns
        -------
        ContinuousSpectrum
            Uniform energy grid in eV and unit-area density in inverse eV.

        Raises
        ------
        InputTransformationError
            If the selector is invalid, the species is missing or ambiguous,
            or its dynamics do not provide a supported VDOS.
        ValueError
            If either energy bound cannot be parsed as a number.
        """
        element = parse_pdos_parameter(parameter)
        _, _, block = _block(text, element)
        return _read_vdos(block)

    def replace(self, text, parameter, spectrum):
        """Replace the VDOS fields for one species in NCMAT text.

        Parameters
        ----------
        text : str
            Target NCMAT source text.
        parameter : str
            Species selector of the form ``<element>_pdos``.
        spectrum : ContinuousSpectrum
            Unit-area density on a uniform grid in eV. A zero-origin spectrum
            is converted by dropping the zero-energy point and normalizing
            the retained positive-energy density.

        Returns
        -------
        str
            New NCMAT text with the selected species' VDOS bounds and density
            replaced. Other species and unrelated blocks are preserved.

        Raises
        ------
        InputTransformationError
            If species selection is invalid, the target VDOS fields are missing,
            or the input grid cannot be represented by the adapter.
        """
        element = parse_pdos_parameter(parameter)
        start, end, block = _block(text, element)
        converted = to_ncrystal_grid(spectrum) if np.isclose(spectrum.energy_ev[0], 0) else spectrum
        e, g = converted.energy_ev, converted.density_per_ev
        if not np.allclose(np.diff(e), e[1] - e[0], rtol=1e-6, atol=1e-11):
            raise InputTransformationError('NCMAT vdos_egrid requires uniform energy spacing.')
        grid = _GRID.search(block)
        dens = _DENS.search(block)
        if not grid or not dens:
            raise InputTransformationError(f'Target NCMAT {element} VDOS fields missing.')
        new_grid = f'{grid.group("leading")}{e[0]:.16g} {e[-1]:.16g}'
        # NCMAT permits free-form values and repeated zeros. We write plain
        # numeric values to avoid ambiguous repetition syntax.
        lines = [' '.join(f'{v:.15g}' for v in g[i:i+6]) for i in range(0, len(g), 6)]
        new_dens = dens.group('leading') + lines[0] + ''.join('\n    ' + line for line in lines[1:]) + '\n'
        # Replace from right to left so offsets cannot shift.
        edits = sorted([(grid.start(), grid.end(), new_grid), (dens.start(), dens.end(), new_dens)], reverse=True)
        for a, b, value in edits:
            block = block[:a] + value + block[b:]
        return text[:start] + block + text[end:]

"""
Parser module for NCrystal .ncmat files.
"""
import sys
import re
import numpy as np
from typing import Dict, Any, List, TYPE_CHECKING

from input_output.ncrystal.exceptions import NCrystalFormatError
from input_output.ncrystal.parsers import chunk_blocks, extract_floats, expand_ncmat_array_values

if TYPE_CHECKING:
    from .core import NCrystalFile


class NCrystalReader:
    """
    Parses a .ncmat text file and populates an NCrystalFile data object.

    Attributes
    ----------
    nc : NCrystalFile
        The target data structure to populate.
    """

    def __init__(self, ncrystal_obj: 'NCrystalFile'):
        """
        Initializes the reader.

        Parameters
        ----------
        ncrystal_obj : NCrystalFile
            The data structure instance to populate.
        """
        self.nc = ncrystal_obj

    def read(self) -> None:
        """
        Coordinates the parsing of the entire NCrystal file.

        Raises
        ------
        Exception
            Propagates any file reading or parsing exceptions.
        """
        try:
            # Open the external file with an explicit encoding and deterministic resource
            # cleanup.
            with open(self.nc.filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            # The NCMAT version is on the file's first nonempty, noncomment
            # line, not in an @NCMAT_VERSION block. Preserve it on round-trip.
            header = next(
                (line.split("#", 1)[0].strip() for line in lines
                 if line.split("#", 1)[0].strip()),
                "",
            )
            version_match = re.fullmatch(r"NCMAT v([1-9][0-9]*)", header)
            if version_match is None:
                raise NCrystalFormatError(f"Invalid NCMAT version header: {header!r}")
            self.nc.version = int(version_match.group(1))

            blocks = chunk_blocks(lines)
            self._dispatch_blocks(blocks)
            
        except Exception as e:
            print(f"Error reading NCrystal file {self.nc.filepath}: {e}", file=sys.stderr)
            raise

    def _dispatch_blocks(self, blocks: Dict[str, Any]) -> None:
        """
        Routes parsed text blocks to their specific extraction logic.

        Parameters
        ----------
        blocks : dict
            Dictionary mapping block tags to string arrays.
        """
        for tag, lines in blocks.items():
            if tag == "NCMAT_VERSION":
                self.nc.version = int(lines[0])
            elif tag == "DENSITY":
                self.nc.density = lines[0]
            elif tag in ("STATEOFMATTER", "STATEOFSUBSTANCE"):
                self.nc.state_of_substance = lines[0]
            elif tag == "CELL":
                self._parse_cell(lines)
            elif tag == "SPACEGROUP":
                self.nc.spacegroup = int(lines[0])
            elif tag == "ATOMPOSITIONS":
                self._parse_atom_positions(lines)
            elif tag == "DYNINFO":
                if isinstance(lines[0], str):
                    self.nc.dyninfo.append(self._parse_dyninfo(lines))
                else:
                    for sub_block in lines:
                        self.nc.dyninfo.append(self._parse_dyninfo(sub_block))
            elif tag == "CUSTOM_VDOSEBINS":
                self.nc.custom_vdosebins = self._parse_float_array(lines)
            elif tag == "CUSTOM_VDOS":
                self.nc.custom_vdos = self._parse_float_array(lines)
            else:
                self.nc.raw_blocks[tag] = lines

    def _parse_cell(self, lines: List[str]) -> None:
        """Parses the @CELL block (lengths and angles), supporting NCMAT v4.

        Parameters
        ----------
        lines : List[str]
            Input text lines to parse or serialize.
        """
        self.nc.cell['raw_lines'] = lines  # Retain verbatim text for perfect round-tripping
        if not lines:
            return
            
        for line in lines:
            # Tokenize this NCMAT record before mapping its values to a structured field.
            parts = line.split()
            if not parts:
                continue
                
            # Attempt to extract traditional single-line format fallback
            if parts[0].lower() not in ['lengths', 'angles']:
                try:
                    self.nc.cell['a'] = float(parts[0])
                except ValueError:
                    pass

    def _parse_atom_positions(self, lines: List[str]) -> None:
        """Parses the @ATOMPOSITIONS block, preserving fractions.

        Parameters
        ----------
        lines : List[str]
            Input text lines to parse or serialize.
        """
        for line in lines:
            # Tokenize this NCMAT record before mapping its values to a structured field.
            parts = line.split()
            if len(parts) >= 4:
                # Store coordinates as raw strings to safely pass through fractions like '1/4'
                self.nc.atom_positions.append({
                    'element': parts[0],
                    'x': parts[1],
                    'y': parts[2],
                    'z': parts[3]
                })

    def _parse_dyninfo(self, lines: List[str]) -> Dict[str, Any]:
        """Parses a @DYNINFO block into a structured dictionary, supporting NCMAT v4.

        Parameters
        ----------
        lines : List[str]
            Input text lines to parse or serialize.

        Returns
        -------
        Dict[str, Any]
            Parsed dynamics fields, optional energy and density arrays, and original lines.
        """
        dyn_dict, vdos_values = self._read_dyninfo_fields(lines)

        if vdos_values:
            dyn_dict['vdos'] = np.array(vdos_values, dtype=np.float64)
            if 'egrid_params' in dyn_dict:
                dyn_dict['egrid'] = self._dyninfo_energy_grid(
                    dyn_dict['egrid_params'], len(vdos_values),
                )

        return dyn_dict

    @staticmethod
    def _read_dyninfo_fields(lines: List[str]) -> tuple[Dict[str, Any], list[float]]:
        """Read DYNINFO metadata and collect wrapped VDOS values.

        Parameters
        ----------
        lines : list of str
            Original lines in one DYNINFO block.

        Returns
        -------
        tuple of dict and list of float
            Parsed fields with the original lines, and expanded density values.
        """
        # Retain the original block for lossless writing when it is unchanged.
        dyn_dict: Dict[str, Any] = {'raw_lines': lines}
        in_vdos = False
        vdos_values = []
        
        for line in lines:
            lower_line = line.lower()
            
            # Unlabelled continuation lines belong to the active VDOS block.
            if in_vdos and not lower_line.startswith(('element', 'fraction', 'type', 'temperature', 'vdos')):
                vdos_values.extend(expand_ncmat_array_values(line))
                continue
                
            if lower_line.startswith('element'):
                dyn_dict['element'] = line.split()[1]
            elif lower_line.startswith('fraction'):
                dyn_dict['fraction'] = line.split()[1]
            elif lower_line.startswith('type'):
                dyn_dict['type'] = line.split()[1]
            elif lower_line.startswith('temperature'):
                dyn_dict['temperature'] = line.split()[1]
            elif lower_line.startswith('vdos_egrid'):
                dyn_dict['egrid_params'] = extract_floats(line)
            elif lower_line.startswith('vdos_density'):
                in_vdos = True
                vdos_values.extend(expand_ncmat_array_values(line.split('vdos_density', 1)[1]))
            elif lower_line.startswith('vdos'):
                in_vdos = True
            elif 'egrid' in lower_line and 'vdos_egrid' not in lower_line:
                dyn_dict['egrid_params'] = extract_floats(line)
                
        return dyn_dict, vdos_values


    @staticmethod
    def _dyninfo_energy_grid(params, density_count: int) -> np.ndarray:
        """Expand NCMAT grid bounds using the parsed density count.

        Parameters
        ----------
        params : sequence of float
            Energy bounds in eV and optional point count.
        density_count : int
            Number of density values after expanding repetition tokens.

        Returns
        -------
        numpy.ndarray
            Uniform NCMAT energy grid in eV.

        Raises
        ------
        NCrystalFormatError
            If the grid entry has the wrong size or declares an inconsistent point count.
        """
        if len(params) == 3:
            if not float(params[2]).is_integer() or int(params[2]) != density_count:
                raise NCrystalFormatError(
                    "NCMAT VDOS grid count disagrees with expanded density length"
                )
            return np.linspace(params[0], params[1], int(params[2]))
        if len(params) == 2:
            # The two-value syntax omits the number of expanded density points.
            return np.linspace(params[0], params[1], density_count)
        raise NCrystalFormatError("NCMAT VDOS needs 2 or 3 grid parameters")


    def _parse_float_array(self, lines: List[str]) -> np.ndarray:
        """Extracts a flat numpy array from multiline text blocks.

        Parameters
        ----------
        lines : List[str]
            Input text lines to parse or serialize.

        Returns
        -------
        np.ndarray
            Flat array in source order with repetition tokens expanded.
        """
        all_floats = []
        for line in lines:
            # Extract NCMAT VDOS numeric values independently of their line wrapping.
            all_floats.extend(expand_ncmat_array_values(line))
        return np.array(all_floats, dtype=np.float64)

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
            Results associated with their sample, energy, or field identifiers.
        """
        dyn_dict: Dict[str, Any] = {'raw_lines': lines}  # Preserve exact lines for writer
        in_vdos = False
        vdos_values = []
        
        for line in lines:
            lower_line = line.lower()
            
            # Continue collecting VDOS numbers if we are inside the array
            if in_vdos and not lower_line.startswith(('element', 'fraction', 'type', 'temperature', 'vdos')):
                # Extract NCMAT VDOS numeric values independently of their line wrapping.
                vdos_values.extend(expand_ncmat_array_values(line))
                continue
                
            if lower_line.startswith('element'):
                # Tokenize this NCMAT record before mapping its values to a structured field.
                dyn_dict['element'] = line.split()[1]
            elif lower_line.startswith('fraction'):
                # Tokenize this NCMAT record before mapping its values to a structured field.
                dyn_dict['fraction'] = line.split()[1]
            elif lower_line.startswith('type'):
                # Tokenize this NCMAT record before mapping its values to a structured field.
                dyn_dict['type'] = line.split()[1]
            elif lower_line.startswith('temperature'):
                # Tokenize this NCMAT record before mapping its values to a structured field.
                dyn_dict['temperature'] = line.split()[1]
            elif lower_line.startswith('vdos_egrid'):
                # Extract NCMAT VDOS numeric values independently of their line wrapping.
                dyn_dict['egrid_params'] = extract_floats(line)
            elif lower_line.startswith('vdos_density'):
                in_vdos = True
                # Extract NCMAT VDOS numeric values independently of their line wrapping.
                vdos_values.extend(expand_ncmat_array_values(line.split('vdos_density', 1)[1]))
            elif lower_line.startswith('vdos'):
                in_vdos = True
            elif 'egrid' in lower_line and 'vdos_egrid' not in lower_line:
                # Extract NCMAT VDOS numeric values independently of their line wrapping.
                dyn_dict['egrid_params'] = extract_floats(line)
                
        if vdos_values:
            # Retain the native VDOS numeric samples in a contiguous floating-point array.
            dyn_dict['vdos'] = np.array(vdos_values, dtype=np.float64)
            if 'egrid_params' in dyn_dict:
                ep = dyn_dict['egrid_params']
                if len(ep) == 3:
                    if not float(ep[2]).is_integer() or int(ep[2]) != len(vdos_values):
                        raise NCrystalFormatError(
                            "NCMAT VDOS grid count disagrees with expanded density length"
                        )
                    dyn_dict['egrid'] = np.linspace(ep[0], ep[1], int(ep[2]))
                elif len(ep) == 2:
                    # Native 2-parameter VDOS grid uses the EXPANDED point count.
                    dyn_dict['egrid'] = np.linspace(ep[0], ep[1], len(vdos_values))
                else:
                    raise NCrystalFormatError("NCMAT VDOS needs 2 or 3 grid parameters")
                    
        return dyn_dict

    def _parse_float_array(self, lines: List[str]) -> np.ndarray:
        """Extracts a flat numpy array from multiline text blocks.

        Parameters
        ----------
        lines : List[str]
            Input text lines to parse or serialize.

        Returns
        -------
        np.ndarray
            Numerical array evaluated on the corresponding coordinate grid.
        """
        all_floats = []
        for line in lines:
            # Extract NCMAT VDOS numeric values independently of their line wrapping.
            all_floats.extend(expand_ncmat_array_values(line))
        return np.array(all_floats, dtype=np.float64)

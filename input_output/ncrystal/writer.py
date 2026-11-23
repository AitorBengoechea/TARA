"""
Serializer module for NCrystal format files.
"""
import numpy as np
from typing import TextIO, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import NCrystalFile


class NCrystalWriter:
    """
    Serializes an NCrystalFile data structure into a formatted .ncmat text file.

    Attributes
    ----------
    nc : NCrystalFile
        The source data structure to serialize.
    """

    def __init__(self, ncrystal_obj: 'NCrystalFile'):
        """
        Initializes the writer.

        Parameters
        ----------
        ncrystal_obj : NCrystalFile
            The data structure instance to serialize.
        """
        self.nc = ncrystal_obj

    def write(self, output_path: str) -> None:
        """
        Coordinates the sequential writing of the entire .ncmat file.

        Parameters
        ----------
        output_path : str
            The destination path where the file will be written.
        """
        # Open the external file with an explicit encoding and deterministic resource cleanup.
        with open(output_path, "w", encoding="utf-8") as f:
            # NCMAT's version is a required leading header, not an @-block.
            f.write(f"NCMAT v{self.nc.version}\n")
            
            if self.nc.state_of_substance:
                self._write_block(f, "STATEOFMATTER", [self.nc.state_of_substance])
                
            if self.nc.density:
                self._write_block(f, "DENSITY", [self.nc.density])
                
            if self.nc.cell:
                if 'raw_lines' in self.nc.cell:
                    self._write_block(f, "CELL", self.nc.cell['raw_lines'])
                elif 'lengths' in self.nc.cell and 'angles' in self.nc.cell:
                    l = self.nc.cell['lengths']
                    a = self.nc.cell['angles']
                    self._write_block(f, "CELL", [f"{l[0]} {l[1]} {l[2]} {a[0]} {a[1]} {a[2]}"])
                else:
                    self._write_block(f, "CELL", [str(self.nc.cell.get('a', ''))])
                    
            if self.nc.spacegroup > 0:
                self._write_block(f, "SPACEGROUP", [str(self.nc.spacegroup)])
                
            if self.nc.atom_positions:
                pos_lines = [f"{p['element']} {p['x']} {p['y']} {p['z']}" 
                             for p in self.nc.atom_positions]
                self._write_block(f, "ATOMPOSITIONS", pos_lines)
                
            for dyn_block in self.nc.dyninfo:
                self._write_dyninfo(f, dyn_block)
                
            if self.nc.custom_vdosebins is not None:
                self._write_array_block(f, "CUSTOM_VDOSEBINS", self.nc.custom_vdosebins)
                
            if self.nc.custom_vdos is not None:
                self._write_array_block(f, "CUSTOM_VDOS", self.nc.custom_vdos)
                
            for tag, lines in self.nc.raw_blocks.items():
                if isinstance(lines[0], str):
                    self._write_block(f, tag, lines)
                else:
                    for sub_lines in lines:
                        self._write_block(f, tag, sub_lines)

    def _write_block(self, f: TextIO, tag: str, lines: List[str]) -> None:
        """Writes a standard text block with an @ tag.

        Parameters
        ----------
        f : TextIO
            Open input stream or file-like object used by the parser.
        tag : str
            Format tag or section marker to locate.
        lines : List[str]
            Input text lines to parse or serialize.
        """
        f.write(f"@{tag}\n")
        for line in lines:
            f.write(f"{line}\n")
        f.write("\n")

    def _write_dyninfo(self, f: TextIO, block: Dict[str, Any]) -> None:
        """Writes a complex @DYNINFO block.

        Parameters
        ----------
        f : TextIO
            Open input stream or file-like object used by the parser.
        block : Dict[str, Any]
            Input data block associated with the format section.
        """
        if 'raw_lines' in block:
            self._write_block(f, "DYNINFO", block['raw_lines'])
        else:
            # Keep original fallback logic
            lines = []
            if 'element' in block: lines.append(f"element {block['element']}")
            if 'fraction' in block: lines.append(f"fraction {block['fraction']}")
            if 'type' in block: lines.append(f"type {block['type']}")
            if 'temperature' in block: lines.append(f"temperature {block['temperature']}")
            if 'vdos' in block:
                lines.append("vdos")
                if 'egrid_params' in block:
                    ep = block['egrid_params']
                    lines.append(f"  egrid {ep[0]} {ep[1]} {int(ep[2])}")
                lines.append("  grid")
                for val in block['vdos']:
                    lines.append(f"    {val:12.6E}")
            self._write_block(f, "DYNINFO", lines)

    def _write_array_block(self, f: TextIO, tag: str, array: np.ndarray, items_per_line: int = 5) -> None:
        """Writes a flattened numpy array block.

        Parameters
        ----------
        f : TextIO
            Open input stream or file-like object used by the parser.
        tag : str
            Format tag or section marker to locate.
        array : np.ndarray
            Numerical array to format or inspect.
        items_per_line : int, optional
            Maximum number of array values serialized per line.
        """
        f.write(f"@{tag}\n")
        flat_arr = array.flatten()
        for i, val in enumerate(flat_arr):
            f.write(f"{val:12.6E} ")
            if (i + 1) % items_per_line == 0:
                f.write("\n")
        if len(flat_arr) % items_per_line != 0:
            f.write("\n")
        f.write("\n")

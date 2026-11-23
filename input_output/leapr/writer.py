"""
Serializer module for LEAPR format files.
"""
from io import StringIO
import numpy as np
from typing import TextIO, Dict, Any, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import LeaprFile


class LeaprWriter:
    """
    Serializes a LeaprFile data structure into formatted LEAPR text.

    Attributes
    ----------
    leapr : LeaprFile
        The source data structure to serialize.
    """

    def __init__(self, leapr_obj: 'LeaprFile'):
        """
        Initializes the writer.

        Parameters
        ----------
        leapr_obj : LeaprFile
            The data structure instance to serialize.
        """
        self.leapr = leapr_obj

    def write(self, output_path: str) -> None:
        """
        Serialize the complete LEAPR file to disk.

        Parameters
        ----------
        output_path : str
            The destination path where the LEAPR file will be written.
        """
        # Open the external file with an explicit encoding and deterministic resource cleanup.
        with open(output_path, "w", encoding="utf-8") as f:
            self._write(f)

    def to_string(self) -> str:
        """Serialize the complete LEAPR file to an in-memory string.

        Returns
        -------
        str
            Serialized text or identifier produced by this operation.
        """
        buffer = StringIO()
        self._write(buffer)
        return buffer.getvalue()

    def _write(self, f: TextIO) -> None:
        """Write the complete LEAPR representation to an open text stream.

        Parameters
        ----------
        f : TextIO
            Open input stream or file-like object used by the parser.
        """
        self._write_header_and_grids(f)
        self._write_temp_blocks(f)
        self._write_footer(f)

    def _write_header_and_grids(self, f: TextIO) -> None:
        """
        Writes Cards 1-9 to the file.

        Parameters
        ----------
        f : TextIO
            The active file object.
        """
        f.write(self.leapr.module_call_line)
        f.write(self.leapr.file_units_line)
        f.write(self.leapr.title)
        
        self._write_card_3_run_control(f)
        self._write_card_4_material_info(f)
        self._write_card_5_principal_scatterer(f)
        self._write_card_6_secondary_scatterer(f)
        self._write_card_7_grid_params(f)
        
        self._write_data_block(f, self.leapr.alpha_grid, items_per_line=5)
        self._write_data_block(f, self.leapr.beta_grid, items_per_line=5)

    def _write_temp_blocks(self, f: TextIO) -> None:
        """
        Writes all temperature-dependent blocks to the file.

        Parameters
        ----------
        f : TextIO
            The active file object.
        """
        for block in self.leapr.temp_blocks:
            temp = block['temp']
            self._write_control_line(f, [temp])
            if temp > 0:
                self._write_positive_temp_block(f, block)

    def _write_positive_temp_block(self, f: TextIO, block: Dict[str, Any]) -> None:
        """
        Writes the full set of cards (11-19) for a positive temperature.

        Parameters
        ----------
        f : TextIO
            The active file object.
        block : dict
            The temperature block data mapping.
        """
        self._write_control_line(f, [block['delta'], block['ni']])
        self._write_data_block(f, block['phonon_spectrum'], items_per_line=4)
        
        card13_data = [block['twt'], block['c'], block['tbeta']]
        if '_opt4' in block:
            card13_data.append(block['_opt4'])
        self._write_control_line(f, card13_data)
        
        nd = block['nd']
        self._write_control_line(f, [nd])
        
        if nd > 0:
            self._write_data_block(f, block.get('oscillator_energies', []), items_per_line=5)
            self._write_data_block(f, block.get('oscillator_weights', []), items_per_line=5)
            
        nsk = self.leapr.principal_scatterer.get('nsk', 0)
        if nsk > 0:
            self._write_control_line(f, [block['nka'], block['dka']])
            self._write_data_block(f, block.get('static_structure_factor', []), items_per_line=5)
            if nsk == 2:
                self._write_control_line(f, [block.get('coherent_scattering_fraction', 0.0)])

    def _write_footer(self, f: TextIO) -> None:
        """
        Writes footer comments and the 'stop' card to the file.

        Parameters
        ----------
        f : TextIO
            The active file object.
        """
        for line in self.leapr.footer_comments:
            f.write(line)
        
        if self.leapr.stop_card_found:
            f.write("/\n")
            f.write("stop\n")

    def _write_control_line(self, f: TextIO, data: List[Union[int, float]]) -> None:
        """
        Writes a single control line with simple float/int formatting.

        Parameters
        ----------
        f : TextIO
            The active file object.
        data : List[Union[int, float]]
            The line values to format and write.
        """
        if not data:
            f.write("/\n")
            return
        
        line_parts = []
        for val in data:
            if isinstance(val, (int, np.integer)):
                line_parts.append(f"{val}")
            else:
                if val == float(int(val)):
                     line_parts.append(f"{int(val)}.")
                else:
                     line_parts.append(f"{val}")
        
        f.write(" ".join(line_parts))
        f.write("/\n")

    def _write_card_3_run_control(self, f: TextIO) -> None:
        """Writes Card 3 parameters.

        Parameters
        ----------
        f : TextIO
            Open input stream or file-like object used by the parser.
        """
        p = self.leapr.run_control_params
        self._write_control_line(f, [p['ntempr'], p['iprint'], p['nphon']])

    def _write_card_4_material_info(self, f: TextIO) -> None:
        """Writes Card 4 parameters.

        Parameters
        ----------
        f : TextIO
            Open input stream or file-like object used by the parser.
        """
        p = self.leapr.material_info
        values = [p['mat'], int(p['za']), p['isabt'], p['ilog']]
        if 'smin' in p:
            values.append(p['smin'])
        self._write_control_line(f, values)
        
    def _write_card_5_principal_scatterer(self, f: TextIO) -> None:
        """Writes Card 5 parameters.

        Parameters
        ----------
        f : TextIO
            Open input stream or file-like object used by the parser.
        """
        p = self.leapr.principal_scatterer
        self._write_control_line(f, [p['awr'], p['spr'], p['npr'], p['iel'], p['ncold'], p['nsk']])

    def _write_card_6_secondary_scatterer(self, f: TextIO) -> None:
        """Writes Card 6 parameters.

        Parameters
        ----------
        f : TextIO
            Open input stream or file-like object used by the parser.
        """
        p = self.leapr.secondary_scatterer
        self._write_control_line(f, [p['nss'], p['b7'], p['aws'], p['sps'], p['mss']])

    def _write_card_7_grid_params(self, f: TextIO) -> None:
        """Writes Card 7 parameters.

        Parameters
        ----------
        f : TextIO
            Open input stream or file-like object used by the parser.
        """
        p = self.leapr.grid_params
        self._write_control_line(f, [p['nalpha'], p['nbeta'], p['lat']])

    def _write_data_block(self, f: TextIO, data: Union[np.ndarray, List[float], None], items_per_line: int = 5) -> None:
        """
        Writes a general LEAPR data block with uppercase E scientific notation.

        Parameters
        ----------
        f : TextIO
            The active file object.
        data : Union[np.ndarray, List[float], None]
            The numerical data block to write.
        items_per_line : int, optional
            The maximum number of items permitted on a single line (default is 5).
        """
        if data is None:
            f.write("/\n")
            return
            
        data_arr = np.asarray(data)

        if data_arr.ndim == 0:
            data_1d = data_arr.reshape(1)
        else:
            data_1d = data_arr.flatten()
        
        num_items = len(data_1d)
        if num_items == 0:
            f.write("/\n")
            return

        for i, val in enumerate(data_1d):
            f.write(f"{val:12.6E}") 
            is_last_item = (i + 1) == num_items
            if is_last_item:
                f.write("/\n")
            elif (i + 1) % items_per_line == 0:
                f.write("\n")
            else:
                f.write(" ")

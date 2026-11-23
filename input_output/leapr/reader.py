"""
Parser module for LEAPR format files.
"""
import sys
import numpy as np
from typing import TextIO, Dict, Any, List, Tuple, TYPE_CHECKING

from input_output.leapr.exceptions import LeaprFormatError
from input_output.leapr.parsers import read_data_block, read_phonon_spectrum_block, read_fixed_length_array, FLOAT_REGEX

if TYPE_CHECKING:
    from .core import LeaprFile


class LeaprReader:
    """
    Parses a LEAPR text file and populates a LeaprFile data object.

    Attributes
    ----------
    leapr : LeaprFile
        The target data structure to populate.
    """

    def __init__(self, leapr_obj: 'LeaprFile'):
        """
        Initializes the reader.

        Parameters
        ----------
        leapr_obj : LeaprFile
            The data structure instance to populate.
        """
        self.leapr = leapr_obj

    def read(self) -> None:
        """
        Coordinates the sequential parsing of the entire LEAPR file.

        Raises
        ------
        Exception
            Propagates any file reading or parsing exceptions.
        """
        try:
            # Open the external file with an explicit encoding and deterministic resource
            # cleanup.
            with open(self.leapr.filepath, "r", encoding="utf-8") as f:
                self._read_header_and_grids(f)
                self._read_temp_blocks(f)
                self._read_footer(f)
        except Exception as e:
            print(f"Error reading LEAPR file {self.leapr.filepath}: {e}", file=sys.stderr)
            raise

    def _read_header_and_grids(self, f: TextIO) -> None:
        """
        Reads Cards 1-9 containing module info, parameters, and grids.

        Parameters
        ----------
        f : TextIO
            The active file object.
        """
        self.leapr.module_call_line = f.readline()
        self.leapr.file_units_line = f.readline()
        self.leapr.title = f.readline()

        self._parse_card_3_run_control(f)
        self._parse_card_4_material_info(f)
        self._parse_card_5_principal_scatterer(f)
        self._parse_card_6_secondary_scatterer(f)
        self._parse_card_7_grid_params(f)
        
        # Read the next LEAPR numerical block using the format-specific low-level parser.
        self.leapr.alpha_grid = np.array(read_data_block(f), dtype=np.float64)
        # Read the next LEAPR numerical block using the format-specific low-level parser.
        self.leapr.beta_grid = np.array(read_data_block(f), dtype=np.float64)

    def _read_temp_blocks(self, f: TextIO) -> None:
        """
        Reads all temperature-dependent blocks (Cards 10-19).

        Parameters
        ----------
        f : TextIO
            The active file object.

        Raises
        ------
        EOFError
            If the file ends before expected temperature blocks are read.
        """
        ntempr = self.leapr.run_control_params.get('ntempr', 0)
        if ntempr == 0:
            print("Warning: ntempr=0 or not found, no temp blocks read.", file=sys.stderr)

        first_positive_found = False

        for _ in range(ntempr):
            temp_data = self._read_control_line_floats(f)
            if not temp_data:
                 raise EOFError(f"Expected {ntempr} temp blocks, but file ended.")
            temp = temp_data[0]
            
            if temp > 0:
                block = self._read_positive_temp_block(f, temp)
                self.leapr.temp_blocks.append(block)

                if not first_positive_found:
                    self.leapr.first_positive_spectrum = block['phonon_spectrum']
                    energies = np.arange(block['ni']) * block['delta']
                    self.leapr.first_positive_energy_grid = energies
                    first_positive_found = True
                    
            elif temp < 0:
                self.leapr.temp_blocks.append({'temp': temp})

    def _read_positive_temp_block(self, f: TextIO, temp: float) -> Dict[str, Any]:
        """
        Reads the full set of cards (11-19) for a positive temperature.

        Parameters
        ----------
        f : TextIO
            The active file object.
        temp : float
            The temperature value for the block.

        Returns
        -------
        dict
            Dictionary mapping block parameters to their corresponding values.
        """
        block: Dict[str, Any] = {'temp': temp}
        nsk = self.leapr.principal_scatterer.get('nsk', 0)
        
        delta, ni = self._parse_card_11_phonon_control(f)
        block['delta'] = delta
        block['ni'] = ni
        
        block['phonon_spectrum'] = np.array(
            read_phonon_spectrum_block(f, ni), dtype=np.float64
        )
        
        block.update(self._parse_card_13_translational(f))
        
        nd = self._parse_card_14_oscillator_control(f)
        block['nd'] = nd

        if nd > 0:
            # Read the next LEAPR numerical block using the format-specific low-level parser.
            block['oscillator_energies'] = np.array(
                read_fixed_length_array(f, nd, card='Card 15 oscillator energies'),
                dtype=np.float64,
            )
            # Read the next LEAPR numerical block using the format-specific low-level parser.
            block['oscillator_weights'] = np.array(
                read_fixed_length_array(f, nd, card='Card 16 oscillator weights'),
                dtype=np.float64,
            )

        if nsk > 0:
            nka, dka = self._parse_card_17_structure_factor_control(f)
            block['nka'] = nka
            block['dka'] = dka
            # Read the next LEAPR numerical block using the format-specific low-level parser.
            block['static_structure_factor'] = np.array(read_data_block(f), dtype=np.float64)
            if nsk == 2:
                block['coherent_scattering_fraction'] = self._read_control_line_floats(f)[0]
        
        return block

    def _read_footer(self, f: TextIO) -> None:
        """
        Reads footer comments and the 'stop' card.

        Parameters
        ----------
        f : TextIO
            The active file object.

        Raises
        ------
        EOFError
            If the file ends without the 'stop' terminator.
        """
        while True:
            line = f.readline()
            if not line:
                raise EOFError("File ended without 'stop' terminator.")
            
            line_stripped = line.strip()
            if line_stripped.startswith('stop'):
                self.leapr.stop_card_found = True
                break
            
            if line_stripped.startswith("'"):
                self.leapr.footer_comments.append(line)

    def _read_control_line_floats(self, f: TextIO) -> List[float]:
        """
        Reads a single NJOY control card terminated by '/'.

        Parameters
        ----------
        f : TextIO
            The active file object.

        Returns
        -------
        List[float]
            List of parsed float values.

        Raises
        ------
        EOFError
            If file ends abruptly.
        LeaprFormatError
            If the terminator '/' is missing.
        """
        line = f.readline()
        if not line:
            raise EOFError("Reached end of file while reading control line.")
        
        # Discard inline comments before interpreting the NJOY free-format numerical tokens.
        line_no_comment = line.split('!')[0].split('#')[0]
        if '/' not in line_no_comment:
             raise LeaprFormatError(f"Control line missing terminator '/': {line.strip()}")
             
        # Stop at the free-format slash terminator so subsequent text is not parsed as data.
        line_to_process = line_no_comment.split('/', 1)[0]
        values = FLOAT_REGEX.findall(line_to_process)
        return list(map(float, values))

    def _parse_card_3_run_control(self, f: TextIO) -> None:
        """
        Reads Card 3: ntempr, iprint, nphon.

        Parameters
        ----------
        f : TextIO
            The active file object.
        """
        data = self._read_control_line_floats(f)
        if len(data) < 3:
            raise LeaprFormatError(f"Card 3 missing data. Got {len(data)}.")
        self.leapr.run_control_params = {
            "ntempr": int(data[0]),
            "iprint": int(data[1]),
            "nphon": int(data[2])
        }

    def _parse_card_4_material_info(self, f: TextIO) -> None:
        """
        Reads Card 4: mat, za, isabt, ilog.

        Parameters
        ----------
        f : TextIO
            The active file object.
        """
        data = self._read_control_line_floats(f)
        if len(data) < 2:
             raise LeaprFormatError(f"Card 4 missing data. Got {len(data)}.")
        self.leapr.material_info = {
            "mat": int(data[0]),
            "za": data[1],
            "isabt": int(data[2]) if len(data) > 2 else 0,
            "ilog": int(data[3]) if len(data) > 3 else 0
        }
        if len(data) > 4:
            self.leapr.material_info["smin"] = float(data[4])

    def _parse_card_5_principal_scatterer(self, f: TextIO) -> None:
        """
        Reads Card 5: awr, spr, npr, iel, ncold, nsk.

        Parameters
        ----------
        f : TextIO
            The active file object.
        """
        data = self._read_control_line_floats(f)
        if len(data) < 3:
             raise LeaprFormatError(f"Card 5 missing data. Got {len(data)}.")
        self.leapr.principal_scatterer = {
            "awr": data[0],
            "spr": data[1],
            "npr": int(data[2]),
            "iel": int(data[3]) if len(data) > 3 else 0,
            "ncold": int(data[4]) if len(data) > 4 else 0,
            "nsk": int(data[5]) if len(data) > 5 else 0
        }

    def _parse_card_6_secondary_scatterer(self, f: TextIO) -> None:
        """
        Reads Card 6: nss, b7, aws, sps, mss.

        Parameters
        ----------
        f : TextIO
            The active file object.
        """
        data = self._read_control_line_floats(f)
        if len(data) < 1:
             raise LeaprFormatError(f"Card 6 missing data. Got {len(data)}.")
        nss = int(data[0])
        if nss > 0 and len(data) < 5:
             raise LeaprFormatError(f"Card 6 missing data for {nss} scatterers.")
        self.leapr.secondary_scatterer = {
            "nss": nss,
            "b7": int(data[1]) if nss > 0 else 0,
            "aws": data[2] if nss > 0 else 0.0,
            "sps": data[3] if nss > 0 else 0.0,
            "mss": int(data[4]) if nss > 0 else 0
        }

    def _parse_card_7_grid_params(self, f: TextIO) -> None:
        """
        Reads Card 7: nalpha, nbeta, lat.

        Parameters
        ----------
        f : TextIO
            The active file object.
        """
        data = self._read_control_line_floats(f)
        if len(data) < 3:
             raise LeaprFormatError(f"Card 7 missing data. Got {len(data)}.")
        self.leapr.grid_params = {
            "nalpha": int(data[0]),
            "nbeta": int(data[1]),
            "lat": int(data[2])
        }

    def _parse_card_11_phonon_control(self, f: TextIO) -> Tuple[float, int]:
        """
        Reads Card 11: delta, ni.

        Parameters
        ----------
        f : TextIO
            The active file object.

        Returns
        -------
        Tuple[float, int]
            The delta and ni parameters.
        """
        data = self._read_control_line_floats(f)
        if len(data) < 2:
            raise LeaprFormatError(f"Card 11 missing data: {data}")
        return data[0], int(data[1])

    def _parse_card_13_translational(self, f: TextIO) -> Dict[str, float]:
        """
        Reads Card 13: twt, c, tbeta.

        Parameters
        ----------
        f : TextIO
            The active file object.

        Returns
        -------
        dict
            Dictionary containing Card 13 parameters.
        """
        data = self._read_control_line_floats(f)
        if len(data) < 3:
             raise LeaprFormatError(f"Card 13 missing data. Got {len(data)}.")
        block_data = {
            "twt": data[0],
            "c": data[1],
            "tbeta": data[2]
        }
        if len(data) > 3:
            block_data["_opt4"] = data[3]
        return block_data

    def _parse_card_14_oscillator_control(self, f: TextIO) -> int:
        """
        Reads Card 14: nd.

        Parameters
        ----------
        f : TextIO
            The active file object.

        Returns
        -------
        int
            The nd parameter.
        """
        data = self._read_control_line_floats(f)
        if len(data) < 1:
             raise LeaprFormatError(f"Card 14 missing data.")
        return int(data[0])

    def _parse_card_17_structure_factor_control(self, f: TextIO) -> Tuple[int, float]:
        """
        Reads Card 17: nka, dka.

        Parameters
        ----------
        f : TextIO
            The active file object.

        Returns
        -------
        Tuple[int, float]
            The nka and dka parameters.
        """
        data = self._read_control_line_floats(f)
        if len(data) < 2:
             raise LeaprFormatError(f"Card 17 missing data.")
        return int(data[0]), data[1]

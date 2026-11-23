"""
Core data structures for NJOY LEAPR input files.
"""
import copy
import numpy as np
import pandas as pd
from typing import List, Union, Dict, Any, Optional
from functools import cached_property

from input_output.base import BaseFormatFile
from input_output.leapr.exceptions import LeaprFormatError
from input_output.leapr.reader import LeaprReader
from input_output.leapr.writer import LeaprWriter


class LeaprFile(BaseFormatFile):
    """
    Object to store and manipulate NJOY LEAPR input files.

    This class serves as the central data model, delegating parsing and 
    serialization to the LeaprReader and LeaprWriter respectively.

    Attributes
    ----------
    filepath : str
        Path to the LEAPR input file.
    module_call_line : str
        NJOY module execution info (e.g., 'leapr').
    file_units_line : str
        File unit specifications (e.g., 'nin nout').
    title : str
        The descriptive title of the run (Card 2).
    run_control_params : dict
        Parameters for run control (Card 3: ntempr, iprint, nphon).
    material_info : dict
        Material information (Card 4: mat, za, isabt, ilog).
    principal_scatterer : dict
        Details of the principal scatterer (Card 5).
    secondary_scatterer : dict
        Details of the secondary scatterer (Card 6).
    grid_params : dict
        Grid control parameters (Card 7).
    alpha_grid : np.ndarray or None
        The alpha grid array (Card 8).
    beta_grid : np.ndarray or None
        The beta grid array (Card 9).
    temp_blocks : list of dict
        Temperature-dependent data blocks (Cards 10-19).
    footer_comments : list of str
        Comments found at the end of the file.
    stop_card_found : bool
        Indicates if the 'stop' card was found.
    first_positive_spectrum : np.ndarray or None
        The phonon spectrum for the first positive temperature block.
    first_positive_energy_grid : np.ndarray or None
        The calculated energy grid for the first positive temperature block.
    """

    def __init__(self, filepath: str):
        """
        Initializes the LeaprFile data object.

        Parameters
        ----------
        filepath : str
            The path to the file to be processed.
        """
        super().__init__(filepath)
        
        self.module_call_line = ""
        self.file_units_line = ""
        self.title = ""
        self.run_control_params: Dict[str, int] = {}
        self.material_info: Dict[str, Union[int, float]] = {}
        self.principal_scatterer: Dict[str, Union[int, float]] = {}
        self.secondary_scatterer: Dict[str, Union[int, float]] = {}
        self.grid_params: Dict[str, int] = {}
        self.alpha_grid: Optional[np.ndarray] = None
        self.beta_grid: Optional[np.ndarray] = None
        self.temp_blocks: List[Dict[str, Any]] = []
        self.footer_comments: List[str] = []
        self.stop_card_found = False
        
        self.first_positive_spectrum: Optional[np.ndarray] = None
        self.first_positive_energy_grid: Optional[np.ndarray] = None

    @cached_property
    def pdos(self) -> pd.Series:
        """
        Return the full vibrational spectrum for the first positive temperature.

        Raw Card 12 and discrete modes remain available in ``temp_blocks``.

        Returns
        -------
        pd.Series
            Phonon density of states indexed by the energy grid.

        Raises
        ------
        LeaprFormatError
            If the spectrum or energy grid is missing.
        """
        if self.first_positive_spectrum is None or self.first_positive_energy_grid is None:
            raise LeaprFormatError(
                "Phonon spectrum or energy grid not loaded for the first positive temperature."
            )
        return self.vibrational_pdos()

    def vibrational_pdos(self, temperature=None, energy_grid=None) -> pd.Series:
        """Return the full spectrum at an exact tabulated temperature.

        Parameters
        ----------
        temperature : float or None
            Temperature in kelvin; None selects the first block.
        energy_grid : array_like or None
            Optional uniform reconstruction grid in eV for ND>0.

        Returns
        -------
        pandas.Series
            Full vibrational density indexed by energy in eV.
        """
        from input_output.leapr.temperature import select_leapr_temperature_block
        from input_output.leapr.vibrational import tabulate_vibrational_block
        block, _ = select_leapr_temperature_block(self, temperature)
        block = tabulate_vibrational_block(block, energy_grid)
        return pd.Series(block['phonon_spectrum'],
                         index=np.arange(block['ni']) * block['delta'])

    def _read_file(self) -> None:
        """Delegates file parsing to the LeaprReader.
        """
        reader = LeaprReader(self)
        reader.read()

    def write(self, output_path: str) -> None:
        """
        Delegates file serialization to the LeaprWriter.

        Parameters
        ----------
        output_path : str
            The destination path where the LEAPR file will be written.
        """
        writer = LeaprWriter(self)
        writer.write(output_path)

    # Preserve the validated LEAPR template while preparing one sample-specific representation.
    def shallow_clone(self) -> 'LeaprFile':
        """
        Creates a lightweight clone sharing static arrays to avoid deepcopy overhead.

        Returns
        -------
        LeaprFile
            A shallow copy of the current file instance.
        """
        new_copy = copy.copy(self)
        new_copy.temp_blocks = copy.deepcopy(self.temp_blocks)
        return new_copy

    def get_writer_copy(self,
                        filepath: Optional[str] = None,
                        module_call_line: Optional[str] = None,
                        file_units_line: Optional[str] = None,
                        title: Optional[str] = None,
                        run_control_params: Optional[Dict[str, int]] = None,
                        material_info: Optional[Dict[str, Union[int, float]]] = None,
                        principal_scatterer: Optional[Dict[str, Union[int, float]]] = None,
                        secondary_scatterer: Optional[Dict[str, Union[int, float]]] = None,
                        grid_params: Optional[Dict[str, int]] = None,
                        alpha_grid: Optional[np.ndarray] = None,
                        beta_grid: Optional[np.ndarray] = None,
                        temp_blocks: Optional[List[Dict[str, Any]]] = None,
                        footer_comments: Optional[List[str]] = None,
                        stop_card_found: Optional[bool] = None
                        ) -> 'LeaprFile':
        """
        Returns a deep copy of this object, optionally replacing specific attributes.

        Parameters
        ----------
        filepath : str, optional
            Replacement path associated with the LEAPR file.
        module_call_line : str, optional
            Replacement text for the LEAPR module invocation line.
        file_units_line : str, optional
            Replacement text defining the input and output file units.
        title : str, optional
            Replacement title of the LEAPR input.
        run_control_params : dict, optional
            Replacement dictionary containing LEAPR run-control parameters.
        material_info : dict, optional
            Replacement dictionary containing material properties and
            identifiers used by LEAPR.
        principal_scatterer : dict, optional
            Replacement dictionary describing the principal scattering
            species and its associated LEAPR parameters.
        secondary_scatterer : dict, optional
            Replacement dictionary describing the secondary scattering
            species and its associated LEAPR parameters.
        grid_params : dict, optional
            Replacement dictionary defining the LEAPR grid parameters.
        alpha_grid : np.ndarray, optional
            Replacement array containing the dimensionless alpha grid.
        beta_grid : np.ndarray, optional
            Replacement array containing the dimensionless beta grid.
        temp_blocks : list of dict, optional
            Replacement temperature-dependent LEAPR data blocks,
            including temperature and phonon-spectrum information.
            When provided, the cached first positive-temperature
            spectrum and its energy grid are recalculated.
        footer_comments : list of str, optional
            Replacement comment lines following the LEAPR data cards.
        stop_card_found : bool, optional
            Replacement flag indicating whether the LEAPR STOP card
            was found in the input.

        Notes
        -----
        Parameters default to None. An argument set to None leaves the
        corresponding attribute unchanged in the copied object.

        The original instance is not modified. The method creates a
        deep copy before applying the requested replacements.

        Returns
        -------
        LeaprFile
            The updated deep copy of the file instance.
        """
        new_copy = copy.deepcopy(self)

        if filepath is not None:
            new_copy.filepath = filepath
        if module_call_line is not None:
            new_copy.module_call_line = module_call_line
        if file_units_line is not None:
            new_copy.file_units_line = file_units_line
        if title is not None:
            new_copy.title = title
        if run_control_params is not None:
            new_copy.run_control_params = run_control_params
        if material_info is not None:
            new_copy.material_info = material_info
        if principal_scatterer is not None:
            new_copy.principal_scatterer = principal_scatterer
        if secondary_scatterer is not None:
            new_copy.secondary_scatterer = secondary_scatterer
        if grid_params is not None:
            new_copy.grid_params = grid_params
        if alpha_grid is not None:
            new_copy.alpha_grid = alpha_grid
        if beta_grid is not None:
            new_copy.beta_grid = beta_grid
        if temp_blocks is not None:
            new_copy.temp_blocks = temp_blocks
        if footer_comments is not None:
            new_copy.footer_comments = footer_comments
        if stop_card_found is not None:
            new_copy.stop_card_found = stop_card_found
            
        if temp_blocks is not None:
            new_copy.__dict__.pop('pdos', None)
            new_copy.first_positive_spectrum = None
            new_copy.first_positive_energy_grid = None
            for block in new_copy.temp_blocks:
                if block['temp'] > 0:
                    new_copy.first_positive_spectrum = block['phonon_spectrum']
                    energies = np.arange(block['ni']) * block['delta']
                    new_copy.first_positive_energy_grid = energies
                    break

        return new_copy

    def __eq__(self, other: object) -> bool:
        """Compare two LEAPR files by their parsed structural content.

        Equality is determined by comparing the LEAPR input parameters,
        scattering information, numerical grids, temperature-dependent
        blocks, footer comments, and STOP-card status.

        The file path is excluded from the comparison. Alpha and beta
        grids require exact equality, while NumPy arrays within
        temperature blocks are compared using ``np.allclose`` with
        an absolute tolerance of 1e-8 and NumPy's default relative
        tolerance.

        Parameters
        ----------
        other : object
            Object to compare against the current LEAPR file.

        Returns
        -------
        bool
            True if both LEAPR objects have equivalent structural
            content; False if their compared attributes differ.

        Notes
        -----
        Returns NotImplemented when the other object is not a
        LeaprFile instance, allowing Python's equality dispatch
        to handle the comparison.
        """
        if not isinstance(other, LeaprFile):
            return NotImplemented

        def compare_temp_blocks(blocks1, blocks2):
            """Compare temperature-dependent blocks of two LEAPR inputs.

            Parameters
            ----------
            blocks1 : object
                Temperature-dependent blocks from the first format object.
            blocks2 : object
                Temperature-dependent blocks from the second format object.

            Returns
            -------
            bool
                Whether the requested condition holds.
            """
            if len(blocks1) != len(blocks2):
                return False
            for b1, b2 in zip(blocks1, blocks2):
                if set(b1.keys()) != set(b2.keys()):
                    return False
                for k in b1:
                    if isinstance(b1[k], np.ndarray):
                        if not np.allclose(b1[k], b2[k], atol=1e-8):
                            return False
                    else:
                        if b1[k] != b2[k]:
                            return False
            return True

        return (
            self.module_call_line == other.module_call_line
            and self.file_units_line == other.file_units_line
            and self.title == other.title
            and self.run_control_params == other.run_control_params
            and self.material_info == other.material_info
            and self.principal_scatterer == other.principal_scatterer
            and self.secondary_scatterer == other.secondary_scatterer
            and self.grid_params == other.grid_params
            and np.array_equal(self.alpha_grid, other.alpha_grid)
            and np.array_equal(self.beta_grid, other.beta_grid)
            and self.footer_comments == other.footer_comments
            and self.stop_card_found == other.stop_card_found
            and compare_temp_blocks(self.temp_blocks, other.temp_blocks)
        )

"""
Core data structures for NCrystal .ncmat input files.
"""
import copy
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from functools import cached_property

from input_output.base import BaseFormatFile
from input_output.ncrystal.exceptions import NCrystalFormatError
from input_output.ncrystal.reader import NCrystalReader
from input_output.ncrystal.writer import NCrystalWriter


class NCrystalFile(BaseFormatFile):
    """
    Object to store and manipulate NCrystal .ncmat input files.

    This class serves as the central data model, delegating parsing and 
    serialization to the NCrystalReader and NCrystalWriter respectively.

    Attributes
    ----------
    filepath : str
        Path to the NCrystal input file.
    version : int
        The NCMAT_VERSION of the file.
    density : str
        The string representation of the density (e.g., '0.92 gcm3').
    state_of_substance : str
        State of substance (e.g., 'Solid').
    cell : dict
        Dictionary of unit cell parameters (lengths and angles).
    spacegroup : int
        The spacegroup number.
    atom_positions : list of dict
        List of atoms and their fractional coordinates.
    dyninfo : list of dict
        List of DYNINFO block dictionaries (element, fraction, vdos, etc.).
    custom_vdosebins : np.ndarray or None
        Energy grid for custom VDOS (if provided via @CUSTOM_VDOSEBINS).
    custom_vdos : np.ndarray or None
        VDOS values corresponding to the custom energy bins.
    raw_blocks : dict
        Storage for unrecognized or unparsed blocks to ensure they are 
        written back unmodified.
    """

    def __init__(self, filepath: str):
        """Initialize an NCrystal file data container.

        Parameters
        ----------
        filepath : str
            Path to the NCMAT file associated with this instance.
        """
        super().__init__(filepath)

        # NCMAT format version and material-level metadata.
        self.version: int = 3
        self.density: str = ""
        self.state_of_substance: str = ""

        # Crystallographic structure and atomic positions.
        self.cell: Dict[str, Any] = {}
        self.spacegroup: int = 0
        self.atom_positions: List[Dict[str, Union[str, float]]] = []

        # Atomic dynamics and optional custom vibrational spectra.
        self.dyninfo: List[Dict[str, Any]] = []
        self.custom_vdosebins: Optional[np.ndarray] = None
        self.custom_vdos: Optional[np.ndarray] = None

        # Unrecognized NCMAT sections retained for round-trip serialization.
        self.raw_blocks: Dict[str, Any] = {}

    @cached_property
    def pdos(self) -> pd.Series:
        """
        Returns the primary phonon density of states (VDOS).

        Checks @CUSTOM_VDOS first. If missing, attempts to extract the VDOS
        from the first @DYNINFO block.

        Returns
        -------
        pd.Series
            Phonon density of states indexed by the energy grid in eV.

        Raises
        ------
        NCrystalFormatError
            If no VDOS data is found in the file.
        """
        if self.custom_vdos is not None and self.custom_vdosebins is not None:
            # Expose paired energy and density arrays using their original sampled coordinates.
            return pd.Series(self.custom_vdos, index=self.custom_vdosebins)
            
        for block in self.dyninfo:
            if 'vdos' in block and 'egrid' in block and block['vdos'] is not None:
                # Expose paired energy and density arrays using their original sampled
                # coordinates.
                return pd.Series(block['vdos'], index=block['egrid'])
                
        raise NCrystalFormatError("No VDOS or DYNINFO spectrum found in the NCrystal file.")

    def pdos_for_atom(self, atom_label: str) -> pd.Series:
        """Return the VDOS of a *specified* NCMAT dynamical species.

        Parameters
        ----------
        atom_label : str
            Exact ``@DYNINFO element`` label, e.g. ``"H"`` or ``"C"``.

        Returns
        -------
        pandas.Series
            Native VDOS densities indexed by the matching energy grid (eV).

        Raises
        ------
        NCrystalFormatError
            If the label has no VDOS, or identifies multiple VDOS blocks.

        Notes
        -----
        Unlike the legacy :attr:`pdos` property, this selector is explicit and
        cannot silently return the wrong species in multi-element NCMAT files.
        It does not mutate any component or aggregate C and H spectra.
        """
        matches = [
            block for block in self.dyninfo
            if block.get("element") == atom_label
            and block.get("vdos") is not None
            and block.get("egrid") is not None
        ]
        if len(matches) != 1:
            available = [
                block.get("element") for block in self.dyninfo
                if block.get("vdos") is not None and block.get("egrid") is not None
            ]
            raise NCrystalFormatError(
                f"Expected exactly one VDOS for atom {atom_label!r}; "
                f"found {len(matches)}. Available VDOS labels: {available}"
            )
        block = matches[0]
        return pd.Series(block["vdos"], index=block["egrid"])

    def _read_file(self) -> None:
        """Delegates file parsing to the NCrystalReader.
        """
        reader = NCrystalReader(self)
        reader.read()

    def write(self, output_path: str) -> None:
        """
        Delegates file serialization to the NCrystalWriter.

        Parameters
        ----------
        output_path : str
            The destination path where the .ncmat file will be written.
        """
        writer = NCrystalWriter(self)
        writer.write(output_path)

    def shallow_clone(self) -> 'NCrystalFile':
        """
        Creates a lightweight clone sharing static arrays to avoid deepcopy overhead.

        Returns
        -------
        NCrystalFile
            A shallow copy of the current file instance.
        """
        new_copy = copy.copy(self)
        new_copy.dyninfo = copy.deepcopy(self.dyninfo)
        new_copy.raw_blocks = copy.deepcopy(self.raw_blocks)
        return new_copy

    def __eq__(self, other: object) -> bool:
        """Compare two NCMAT files by their parsed structural content.

        Equality is determined by comparing material metadata,
        crystallographic information, dynamic-information blocks,
        custom VDOS arrays, and preserved raw blocks.

        The file path is excluded from the comparison. Arrays within
        dynamic-information blocks are compared using ``np.allclose``
        with default tolerances. Custom VDOS arrays require exact
        equality.

        Parameters
        ----------
        other : object
            Object to compare against the current NCMAT file.

        Returns
        -------
        bool
            True if both objects have equivalent structural content;
            False if their compared attributes differ.

        Notes
        -----
        Returns NotImplemented for objects that are not NCrystalFile
        instances, allowing Python's equality dispatch to continue.
        """
        if not isinstance(other, NCrystalFile):
            return NotImplemented
            
        # Helper to compare lists of dictionaries with NumPy arrays
        def _compare_dyninfo(
            first: List[Dict[str, Any]],
            second: List[Dict[str, Any]],
        ) -> bool:
            """Compare NCMAT dynamic-information blocks.

            NumPy arrays are compared using ``np.allclose`` with its
            default tolerances. Other values require exact equality.

            Parameters
            ----------
            first : list of dict
                Dynamic-information blocks from the first NCMAT object.
            second : list of dict
                Dynamic-information blocks from the second NCMAT object.

            Returns
            -------
            bool
                True if the blocks have matching keys and equivalent values.
            """
            if len(first) != len(second):
                return False

            for block_a, block_b in zip(first, second):
                if set(block_a) != set(block_b):
                    return False

                for key in block_a:
                    value_a = block_a[key]
                    value_b = block_b[key]

                    if isinstance(value_a, np.ndarray):
                        if not np.allclose(value_a, value_b):
                            return False
                    elif value_a != value_b:
                        return False

            return True

        return (
            self.version == other.version
            and self.density == other.density
            and self.state_of_substance == other.state_of_substance
            and self.cell == other.cell
            and self.spacegroup == other.spacegroup
            and self.atom_positions == other.atom_positions
            and _compare_dyninfo(self.dyninfo, other.dyninfo)
            and np.array_equal(self.custom_vdosebins, other.custom_vdosebins)
            and np.array_equal(self.custom_vdos, other.custom_vdos)
            and self.raw_blocks == other.raw_blocks
        )

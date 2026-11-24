from pathlib import Path
from endf_parserpy import EndfParserPy

from processing.validators.tsl_temperature import validate_tsl_temperature as _validate_tsl_temperature


class EndfValidator:
    """Validate ENDF thermal-scattering files before NJOY processing."""

    @staticmethod
    def validate_tsl_temperature(
        file_path: str, requested_temperature: float, *, sample_id: str = "sample_000"
    ) -> tuple[float, ...]:
        """Validate ACE temperature against one representative MF7/MT4 tape.

        Parameters
        ----------
        file_path : str
            ENDF TSL evaluation to inspect.
        requested_temperature : float
            NJOY processing temperature in kelvin.
        sample_id : str, optional
            Sample identifier used for actionable diagnostics.

        Returns
        -------
        tuple of float
            Temperatures encoded in the TSL evaluation.
        """
        return _validate_tsl_temperature(
            file_path, requested_temperature, sample_id=sample_id
        )

    def __init__(self):
        # Initialize the parser once to save overhead on multiple validations
        """Initialize the ENDF parser used for subsequent validation.
        """
        self.parser = EndfParserPy()

    def validate_sab_tape(self, file_path: str) -> bool:
        """
        Check an ENDF file for the MF7/MT4 thermal-inelastic section.

        The parser returns a single-material dictionary keyed first by MF and
        then by MT. This check verifies section presence after parsing, not
        the physical consistency of the thermal-scattering data.

        Parameters
        ----------
        file_path : str
            Path to the ENDF file to validate.

        Returns
        -------
        bool
            ``True`` if the file is nonempty, can be parsed, and contains
            MF7/MT4; ``False`` otherwise.
        """
        path = Path(file_path)
        if not path.is_file() or path.stat().st_size == 0:
            return False

        try:
            # Parse the tape into a Python dictionary structure
            tape_data = self.parser.parsefile(str(path))

            # For a single-material file, parsefile() maps MF numbers directly
            # to dictionaries of MT sections (there is no outer MAT level).
            return 7 in tape_data and 4 in tape_data[7]

        except Exception:
            # If endf_parserpy throws ANY parsing error (corrupted file, missing headers),
            # it is strictly not a valid tape.
            return False

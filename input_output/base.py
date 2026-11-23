"""
Abstract base classes for nuclear data input/output formats.
"""
from abc import ABC, abstractmethod


class BaseFormatFile(ABC):
    """
    Abstract base class for all nuclear data format handlers.

    Enforces a consistent API for reading, storing, and writing
    different format types (e.g., LEAPR, NCrystal) to facilitate
    seamless translation between them.

    Attributes
    ----------
    filepath : str
        The path to the input file.
    """

    def __init__(self, filepath: str):
        """
        Initializes the base format file object.

        Parameters
        ----------
        filepath : str
            The path to the file to be processed.
        """
        # Retain the original resource path for subsequent format-specific I/O operations.
        self.filepath = filepath

    @classmethod
    def from_file(cls, filepath: str) -> 'BaseFormatFile':
        """
        Constructs the file object by reading from the specified path.

        Parameters
        ----------
        filepath : str
            The path to the input file.

        Returns
        -------
        BaseFormatFile
            The populated format file instance.
        """
        obj = cls(filepath)
        obj._read_file()
        return obj

    @abstractmethod
    def _read_file(self) -> None:
        """Coordinates the sequential reading of the file format.

        Must be implemented by all child classes.
        """
        pass

    @abstractmethod
    def write(self, output_path: str) -> None:
        """
        Serializes the object's attributes back into a formatted text file.
        Must be implemented by all child classes.

        Parameters
        ----------
        output_path : str
            The destination path where the file will be written.
        """
        pass

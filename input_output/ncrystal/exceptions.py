"""
Custom exceptions for NCrystal format parsing.
"""

class NCrystalFormatError(Exception):
    """
    Raised when the NCrystal input deck violates expected block structures.
    """
    pass
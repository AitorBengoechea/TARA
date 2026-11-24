"""Small extension registry: adapters can be registered without pairwise converters."""
from pathlib import Path
from input_output.transformations.models import InputTransformationError

_ADAPTERS = {}


def register_adapter(adapter):
    """Register a native input adapter for its supported extensions.

    Parameters
    ----------
    adapter : object
        Adapter providing the engine, suffixes, and extraction and replacement methods.

    Raises
    ------
    InputTransformationError
        If an extension already belongs to a different adapter.
    """
    for suffix in adapter.suffixes:
        key = suffix.lower()
        if key in _ADAPTERS and _ADAPTERS[key] is not adapter:
            raise InputTransformationError(f'Input extension {key!r} already registered.')
        _ADAPTERS[key] = adapter


def adapter_for(path):
    """Look up the native input adapter for a path suffix.

    Parameters
    ----------
    path : str or pathlib.Path
        Input path with a registered filename extension.

    Returns
    -------
    object
        Registered adapter instance.

    Raises
    ------
    InputTransformationError
        If the case-insensitive suffix has no registered adapter.
    """
    suffix = Path(path).suffix.lower()
    try:
        return _ADAPTERS[suffix]
    except KeyError as exc:
        raise InputTransformationError(f'No input transformation adapter registered for extension {suffix!r}.') from exc

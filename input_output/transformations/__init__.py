"""Engine-agnostic input transformation API."""

from input_output.transformations.api import (
    transform_inputs,
    available_parameters,
)
from input_output.transformations.models import (
    ContinuousSpectrum,
    InputTransformationError,
)
from input_output.transformations.registry import register_adapter
from input_output.transformations.leapr import LeaprAdapter
from input_output.transformations.ncrystal import NCrystalAdapter

register_adapter(LeaprAdapter())
register_adapter(NCrystalAdapter())

__all__ = [
    "transform_inputs",
    "available_parameters",
    "register_adapter",
    "InputTransformationError",
    "ContinuousSpectrum",
]
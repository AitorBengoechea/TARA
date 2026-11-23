"""Engine-neutral physical parameters handled by optional native-file transforms."""
from dataclasses import dataclass
import numpy as np


class InputTransformationError(ValueError):
    """Unsupported or unsafe native-input transformation."""


@dataclass(frozen=True)
class ContinuousSpectrum:
    """A continuous, unweighted density as a function of energy (eV).

    The stored density represents the continuous vibrational contribution.
    Construction copies both arrays, normalizes the density to unit
    trapezoidal integral, and marks the arrays read-only.

    Parameters
    ----------
    energy_ev : numpy.ndarray
        One-dimensional, strictly increasing, nonnegative energy grid in
        eV with at least three points.
    density_per_ev : numpy.ndarray
        Matching finite, nonnegative density values with positive area.

    Raises
    ------
    InputTransformationError
        If the grid or density fails validation.
    """
    energy_ev: np.ndarray
    density_per_ev: np.ndarray

    def __post_init__(self):
        """Validate paired samples and store read-only arrays with unit-area density."""
        energy = np.asarray(self.energy_ev, dtype=float).copy()
        density = np.asarray(self.density_per_ev, dtype=float).copy()
        if energy.ndim != 1 or density.ndim != 1 or len(energy) != len(density) or len(energy) < 3:
            raise InputTransformationError('Spectrum must contain at least three paired points.')
        if (not np.all(np.isfinite(energy)) or not np.all(np.isfinite(density))
                or np.any(energy < 0) or np.any(np.diff(energy) <= 0)
                or np.any(density < 0)):
            raise InputTransformationError('Spectrum energies must increase; density must be finite and nonnegative.')
        area = float(np.trapezoid(density, energy))
        if not np.isfinite(area) or area <= 0:
            raise InputTransformationError('Spectrum must have positive integrated density.')
        energy.setflags(write=False)
        density = density / area
        density.setflags(write=False)
        object.__setattr__(self, 'energy_ev', energy)
        object.__setattr__(self, 'density_per_ev', density)

"""
Core structures for standardizing and validating Phonon Density of States (PDOS).
"""
import numpy as np
import pandas as pd


class PhononSpectraError(Exception):
    """
    Raised when a provided phonon spectrum fails physical or computational validation.
    """
    pass


class PhononSpectra:
    """
    A validated, standardized wrapper for Phonon Density of States (PDOS) data.

    This class guarantees that any spectrum passed to the core analysis modules
    (e.g., TARA) is mathematically well-formed. It automatically corrects missing
    zero-energy points using a Debye quadratic extrapolation, enforces non-negative
    densities, and auto-normalizes the final spectrum to an area of 1.0.

    Attributes
    ----------
    pdos : pd.Series
        The validated and normalized phonon density of states. The index represents 
        the energy grid (eV), and the values represent the density $\\rho(E)$.
    original_starts_at_zero : bool
        True if the original provided spectrum started at E=0 (e.g., LEAPR).
        False if it was artificially extrapolated to E=0 (e.g., NCrystal).
    """

    def __init__(self, pdos: pd.Series, auto_normalize: bool = True):
        """
        Initializes, corrects, validates, and normalizes the phonon spectrum.

        Parameters
        ----------
        pdos : pd.Series
            The raw phonon density of states to be processed.
        auto_normalize : bool, optional
            If True, automatically normalizes the area to 1.0. If False, skips 
            normalization and issues a warning. Default is True.

        Raises
        ------
        PhononSpectraError
            If the input fails structural or physical checks after extrapolation.
        """
        import warnings
        self.original_starts_at_zero = False
        
        # 1. Coerce to clean numeric types
        _pdos = self._preprocess_types(pdos.copy())
        
        # 2. Apply quadratic low-energy extrapolation if needed
        self._pdos = self._enforce_zero_start(_pdos)
        
        # 3. Execute strict physical and structural checks
        self._validate()

        # 4. Handle normalization
        if auto_normalize:
            self.normalize(target_area=1.0)
        else:
            warnings.warn(
                "PhononSpectra auto_normalize=False: This spectrum is NOT normalized to unity. "
                "It is physically invalid and should ONLY be used for visual previewing."
            )

    @property
    def pdos(self) -> pd.Series:
        """
        Returns the strictly validated and normalized phonon spectrum.

        Returns
        -------
        pd.Series
            The density of states data.
        """
        return self._pdos

    @property
    def energy_grid(self) -> np.ndarray:
        """
        Returns the energy grid axis of the spectrum.

        Returns
        -------
        np.ndarray
            The energy values in eV.
        """
        return self._pdos.index.to_numpy(dtype=np.float64)

    @property
    def density_values(self) -> np.ndarray:
        """
        Returns the density values of the spectrum.

        Returns
        -------
        np.ndarray
            The $\\rho(E)$ values.
        """
        return self._pdos.to_numpy(dtype=np.float64)

    @property
    def integral(self) -> float:
        """
        Calculates the area under the phonon spectrum using the trapezoidal rule.

        Returns
        -------
        float
            The integral of $\\rho(E)$ with respect to $E$.
        """
        # Integrate against the physical coordinate grid, accounting for nonuniform spacing.
        return float(np.trapezoid(self.density_values, self.energy_grid))

    def _preprocess_types(self, pdos: pd.Series) -> pd.Series:
        """Coerces the raw input into strictly numeric, sorted types.

        Parameters
        ----------
        pdos : pd.Series
            Phonon density-of-states values.

        Returns
        -------
        pd.Series
            Computed values indexed by the relevant physical coordinates.

        Raises
        ------
        PhononSpectraError
            If the input or required state violates the operation contract.
        """
        if not isinstance(pdos, pd.Series):
            raise PhononSpectraError(
                f"Expected a pandas Series, but received {type(pdos).__name__}."
            )

        if pdos.empty:
            raise PhononSpectraError("The provided phonon spectrum is empty.")

        try:
            pdos.index = pd.to_numeric(pdos.index)
            pdos = pd.to_numeric(pdos)
        except ValueError as e:
            raise PhononSpectraError(f"Spectrum contains non-numeric data: {e}")

        return pdos.sort_index()

    def _enforce_zero_start(self, pdos: pd.Series) -> pd.Series:
        """
        Checks if the spectrum starts at E=0. If not, prepends a Debye 
        quadratic tail to link the spectrum to the origin.

        Parameters
        ----------
        pdos : pd.Series
            The numeric, sorted phonon spectrum.

        Returns
        -------
        pd.Series
            The fully reconstructed spectrum starting at E=0.
        """
        e_first = pdos.index[0]

        if e_first <= 0.0:
            self.original_starts_at_zero = True
            return pdos

        self.original_starts_at_zero = False
        vdos_first = pdos.iloc[0]

        _k = vdos_first / (e_first ** 2)
        _x = np.linspace(0.0, e_first, 2000 + 2)[1:-1]
        unit_factor = 1.0
        
        tail = pd.Series(_k * (_x ** 2), index=_x / unit_factor)
        zero_point = pd.Series([0.0], index=[0.0])

        return pd.concat([zero_point, tail, pdos])

    def _validate(self) -> None:
        """
        Executes a comprehensive suite of checks on the reconstructed PDOS.

        Raises
        ------
        PhononSpectraError
            If the data contains NaN/Inf, possesses negative energies/densities, 
            has a non-monotonic grid, or lacks a positive integral.
        """
        if self._pdos.index.isna().any() or np.isinf(self._pdos.index).any():
            raise PhononSpectraError("Energy grid contains NaN or Infinite values.")
            
        if self._pdos.isna().any() or np.isinf(self._pdos).any():
            raise PhononSpectraError("Density values contain NaN or Infinite values.")

        if not self._pdos.index.is_monotonic_increasing:
            raise PhononSpectraError("The energy grid must be monotonically increasing.")

        if not self._pdos.index.is_unique:
            raise PhononSpectraError("The energy grid contains duplicate values.")

        if (self._pdos.index < 0).any():
            raise PhononSpectraError("The energy grid cannot contain negative energies.")

        if (self._pdos < 0).any():
            raise PhononSpectraError("Phonon density values cannot be negative.")

        # Ensure the integral of the spectrum is positive before normalization.
        if self.integral <= 0:
            raise PhononSpectraError("Cannot normalize a spectrum with a non-positive integral.")

    def normalize(self, target_area: float = 1.0) -> None:
        """
        Normalizes the spectrum in-place to a specific target area.

        Parameters
        ----------
        target_area : float, optional
            The desired integral of the spectrum (default is 1.0).
            
        Raises
        ------
        ValueError
            If the target area is less than or equal to zero.
        """
        if target_area <= 0:
            raise ValueError("Target normalization area must be strictly positive.")
            
        current_integral = self.integral
        self._pdos = self._pdos * (target_area / current_integral)

import functools
import numpy as np
import pandas as pd
from numba import njit
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear
from typing import Dict, Union, List, Optional, Tuple

# --- Define the Decorator ---
def requires_fit(func):
    """Decorator to check if the instance has been fitted before running.

    Parameters
    ----------
    func : object
        Function to wrap.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        """Check fitted-model state before forwarding the decorated call.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to the wrapped function.
        **kwargs : Any
            Additional options accepted by the current interface.
        """
        self.check_fitted()
        return func(self, *args, **kwargs)
    return wrapper

class BsplineFitter:
    """
    A strictly constrained, non-negative P-Spline fitter.
    
    Guarantees that coefficients are >= 0 to prevent artificial oscillations 
    (Gibbs phenomenon) in flat regions. Math engine strictly mandates cubic 
    splines (degree=3).
    """

    def __init__(self, pdf_series: pd.Series, n_pred: Optional[int] = None):
        """Prepare source and prediction grids for a constrained spline fit.

        Parameters
        ----------
        pdf_series : pd.Series
            Spectrum represented as a pandas Series.
        n_pred : Optional[int], optional
            Requested number of prediction coordinates.
        """
        x_raw = pdf_series.index.values.astype(np.float64)
        y_raw = pdf_series.values.astype(np.float64)
        
        # 1. Identify trailing zeros to avoid anchoring the spline artificially
        nonzero_indices = np.nonzero(y_raw)[0]
        cut_idx = 0 if len(nonzero_indices) == 0 else nonzero_indices[-1] + 1
            
        if cut_idx < len(y_raw):
            self.x_data = x_raw[:cut_idx]
            self.y_data = y_raw[:cut_idx]
            self.tail_point = x_raw[cut_idx:]
        else:
            self.x_data = x_raw
            self.y_data = y_raw
            self.tail_point = None
        
        self.x_min = self.x_data.min()
        self.x_max = self.x_data.max()
        
        n_pred = len(self.x_data) if n_pred is None else n_pred
        self.x_pred = np.linspace(self.x_min, self.x_max, n_pred)
        
        self.bspline = None
        self.knots = None
        self.coeffs = None
        self.degree = None

    @staticmethod
    def _build_clamped_knots(x_min: float, x_max: float, n_splines: int, degree: int) -> np.ndarray:
        """Build an endpoint-clamped knot vector for a spline basis.

        Parameters
        ----------
        x_min : float
            Lower endpoint of the coordinate interval.
        x_max : float
            Upper endpoint of the coordinate interval.
        n_splines : int
            Number of spline basis functions to fit.
        degree : int
            Polynomial degree of the spline basis.

        Returns
        -------
        np.ndarray
            Endpoint-clamped knot sequence for the B-spline basis.

        Raises
        ------
        ValueError
            If the input or required state violates the operation contract.
        """
        k = degree
        n_inner = n_splines - (k + 1)
        
        if n_inner < 0:
            raise ValueError(f"n_splines ({n_splines}) must be >= degree + 1 ({k+1})")

        interior_knots = np.linspace(x_min, x_max, n_inner + 2)[1:-1] if n_inner > 0 else np.array([])

        # Assemble the grid segments in the order required by the reconstruction.
        return np.concatenate((
            np.repeat(x_min, k + 1),
            interior_knots,
            np.repeat(x_max, k + 1)
        ))

    def fit(self, n_splines: int = 1000, degree: int = 3, alpha: float = 1e-6) -> None:
        """Fit nonnegative B-spline coefficients using penalized least squares.

        The least-squares penalty discourages curvature while the coefficient bounds keep the
        fit nonnegative.

        Parameters
        ----------
        n_splines : int, optional
            Number of spline basis functions to fit.
        degree : int, optional
            Polynomial degree of the spline basis.
        alpha : float, optional
            Strength of the spline curvature regularization.

        Raises
        ------
        ValueError
            If the input or required state violates the operation contract.
        """
        if degree != 3:
            raise ValueError(f"Math engine strictly mandates cubic splines. Got {degree}.")

        t = self._build_clamped_knots(self.x_min, self.x_max, n_splines, degree)
        n_coefs = len(t) - degree - 1
        
        # --- CRITICAL OPTIMIZATION ---
        # Replace the slow Python 'for i in range(n_coefs):' loop with SciPy's native C matrix builder
        A = BSpline.design_matrix(self.x_data, t, degree).toarray()
        
        if n_coefs > 2:
            D2 = np.zeros((n_coefs - 2, n_coefs))
            for i in range(n_coefs - 2):
                D2[i, i] = 1.0
                D2[i, i+1] = -2.0
                D2[i, i+2] = 1.0
            
            A_aug = np.vstack([A, alpha * D2])
            # Assemble the grid segments in the order required by the reconstruction.
            y_aug = np.concatenate([self.y_data, np.zeros(n_coefs - 2)])
        else:
            A_aug = A
            y_aug = self.y_data
            
        # Solve the constrained least-squares problem with nonnegative fitted coefficients.
        res = lsq_linear(A_aug, y_aug, bounds=(0, np.inf))
        
        self.coeffs = res.x
        self.knots = t
        self.degree = degree
        self.bspline = BSpline(t, self.coeffs, degree)

    def check_fitted(self) -> None:
        """Reject operations that require a fitted spline model.

        Raises
        ------
        ValueError
            If the input or required state violates the operation contract.
        """
        if self.bspline is None:
            raise ValueError("Model not fitted yet. Run 'fit()' first.")

    def find_best_n_splines(self, n_spline_start: int = 1000, jump: int = 100, 
                            degree: int = 3, alpha: float = 1e-6, 
                            rel_tolerance: float = 5e-4, fit: bool = True,
                            verbose: bool = True) -> int:
        """Increase the spline basis until consecutive fits converge.

        Parameters
        ----------
        n_spline_start : int, optional
            Initial number of spline basis functions in the search.
        jump : int, optional
            Number of additional basis functions per search iteration.
        degree : int, optional
            Polynomial degree of the spline basis.
        alpha : float, optional
            Strength of the spline curvature regularization.
        rel_tolerance : float, optional
            Relative convergence tolerance for the fit comparison.
        fit : bool, optional
            Whether to retain the selected fitted model.
        verbose : bool, optional
            Whether to print progress messages.

        Returns
        -------
        int
            The number of spline basis functions selected by the convergence search.

        Raises
        ------
        ValueError
            If the input or required state violates the operation contract.
        """
        if degree != 3:
            raise ValueError("Math engine strictly mandates cubic splines (degree=3).")
            
        if verbose:
            print(f"Searching optimal n_splines from {n_spline_start} (jump={jump})...")
            
        # Scale tolerance to data magnitude to prevent early exit on small peak values
        max_y = np.max(self.y_data) if len(self.y_data) > 0 else 1.0
        abs_tolerance = rel_tolerance * max_y if max_y > 0 else rel_tolerance
        
        current_k = n_spline_start
        self.fit(n_splines=current_k, degree=degree, alpha=alpha)
        y_prev = self.bspline(self.x_pred) 
        prev_mae = float('inf')

        while True:
            next_k = current_k + jump
            self.fit(n_splines=next_k, degree=degree, alpha=alpha)
            y_next = self.bspline(self.x_pred)
            mae = np.mean(np.abs(y_next - y_prev))
            
            if mae > prev_mae:
                if verbose: print("Warning: MAE increased. Stopping to prevent overfitting.")
                optimal_k = current_k
                break
            
            if mae <= abs_tolerance:
                optimal_k = current_k
                break
                
            prev_mae = mae
            y_prev = y_next
            current_k = next_k
    
            if current_k > 3000:
                if verbose: print("Warning: Reached 3000 splines without convergence.")
                optimal_k = current_k
                break

        if fit and (len(self.coeffs) != optimal_k):
            self.fit(n_splines=optimal_k, degree=degree, alpha=alpha)
                
        return optimal_k

    @requires_fit
    def get_knot_centers(self) -> np.ndarray:
        """Calculates Greville abscissae (spatial center) of each basis function.

        Returns
        -------
        np.ndarray
            Greville abscissae identifying the fitted basis centers.
        """
        return np.array([
            np.mean(self.knots[i+1 : i+1+self.degree]) 
            for i in range(len(self.coeffs))
        ])

    @requires_fit
    def get_coefficients(self) -> np.ndarray:
        """Return a copy of the fitted spline coefficients.

        Returns
        -------
        np.ndarray
            Independent copy of the fitted coefficient array.
        """
        return self.coeffs.copy()

    # (Retained for backward compatibility with legacy analysis scripts)
    @requires_fit
    def get_basis_functions(self) -> pd.DataFrame:
        """Return the fitted B-spline basis functions on the prediction grid.

        Returns
        -------
        pd.DataFrame
            Spline basis values evaluated on the prediction coordinates.
        """
        # Construct the spline basis matrix directly rather than evaluating each basis in a
        # Python loop.
        design_mat = BSpline.design_matrix(self.x_pred, self.knots, self.degree)
        return pd.DataFrame(design_mat.T.toarray(), columns=self.x_pred)


# --- Isolated Helper Functions ---

@njit(fastmath=True)
def _calculate_coefficient_shifts_numba(centers: np.ndarray, 
                                        boundaries: np.ndarray, 
                                        shifts: np.ndarray, 
                                        degree: int) -> np.ndarray:
    """JIT-compiled generation of linearly transitioned coefficient multipliers.

    Runs at raw C-speed by unrolling array masks into explicit compiled loops.

    Parameters
    ----------
    centers : np.ndarray
        Coordinates representing spline-basis centers.
    boundaries : np.ndarray
        Spectral-domain boundaries used by the transformation.
    shifts : np.ndarray
        Perturbation values assigned to the spectral domains.
    degree : int
        Polynomial degree of the spline basis.

    Returns
    -------
    np.ndarray
        Numerical array evaluated on the corresponding coordinate grid.
    """
    coeff_shifts = np.zeros_like(centers)
    n_boundaries = boundaries.shape[0]
    n_centers = len(centers)
    
    # 1. Assign bulk shifts exclusively to internal domain coefficients
    for i in range(n_boundaries):
        e_start = boundaries[i, 0]
        e_end = boundaries[i, 1]
        shift = shifts[i]
        
        for j in range(n_centers):
            if centers[j] >= e_start and centers[j] < e_end:
                coeff_shifts[j] = shift
                
    # Bind the absolute terminal boundary point
    if n_boundaries > 0:
        term_val = boundaries[n_boundaries - 1, 1]
        term_shift = shifts[n_boundaries - 1]
        for j in range(n_centers):
            if centers[j] == term_val:
                coeff_shifts[j] = term_shift
                
    # 2. Modulate internal boundaries with linear functional transitions
    N_affected = degree + 1 
    for i in range(n_boundaries - 1):
        e_boundary = boundaries[i, 1]
        left_shift = shifts[i]
        right_shift = shifts[i+1]
        
        # Locate closest center manually (faster in Numba than np.argmin)
        min_dist = np.inf
        b_idx = 0
        for j in range(n_centers):
            dist = abs(centers[j] - e_boundary)
            if dist < min_dist:
                min_dist = dist
                b_idx = j
                
        start_idx = max(0, b_idx - (N_affected // 2))
        end_idx = min(n_centers, start_idx + N_affected)
        actual_N = end_idx - start_idx
        
        if actual_N > 1:
            # Inline linear interpolation (faster in Numba than np.linspace)
            step = (right_shift - left_shift) / (actual_N - 1)
            for k in range(actual_N):
                coeff_shifts[start_idx + k] = left_shift + k * step
                
    return coeff_shifts


@njit(fastmath=True)
def _build_coefficient_transform_numba(
    centers: np.ndarray,
    boundaries: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Build the complete domain-to-coefficient transformation in compiled code.

    Parameters
    ----------
    centers : numpy.ndarray
        Greville abscissae of the fitted spline basis.
    boundaries : numpy.ndarray
        Domain boundaries with shape ``(n_domains, 2)``.
    degree : int
        Spline degree.

    Returns
    -------
    numpy.ndarray
        Linear transformation with shape
        ``(n_domains, n_coefficients)``.

    Notes
    -----
    The transformation is topology dependent and cached by the caller. Moving
    the unit-vector construction into one Numba invocation removes repeated
    Python-to-JIT transitions when a new topology is encountered.
    """
    n_domains = boundaries.shape[0]
    transform = np.zeros((n_domains, centers.size), dtype=np.float64)
    unit_shift = np.zeros(n_domains, dtype=np.float64)

    for domain_idx in range(n_domains):
        for shift_idx in range(n_domains):
            unit_shift[shift_idx] = 0.0
        unit_shift[domain_idx] = 1.0

        transform[domain_idx] = _calculate_coefficient_shifts_numba(
            centers,
            boundaries,
            unit_shift,
            degree,
        )

    return transform


class BsplineAmplitudePerturber:
    """
    Evaluate constrained B-spline amplitude perturbations in matrix batches.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Baseline energy grid on which perturbed spectra are evaluated.
    base_fitter : BsplineFitter
        Already fitted constrained B-spline model.

    Notes
    -----
    For a fixed topology, the existing scalar coefficient-shift rule is linear
    in the domain shifts. A topology matrix is therefore built once by applying
    that trusted rule to the unit vectors of the domain space. Subsequent sample
    batches are evaluated with matrix multiplication, preserving the scalar
    transition mathematics without repeatedly entering Python or Numba.
    """

    def __init__(
        self,
        x_grid: np.ndarray,
        base_fitter: BsplineFitter,
    ) -> None:
        """Prepare source and prediction grids for a constrained spline fit.

        Parameters
        ----------
        x_grid : np.ndarray
            Input coordinate grid.
        base_fitter : BsplineFitter
            Previously initialized spline-fitting model.

        Raises
        ------
        ValueError
            If the input or required state violates the operation contract.
        """
        base_fitter.check_fitted()

        self.x_grid = np.asarray(x_grid, dtype=np.float64)
        if self.x_grid.ndim != 1:
            raise ValueError("x_grid must be one-dimensional.")

        self.base_fitter = base_fitter
        self.coeffs = base_fitter.get_coefficients()
        self.centers = base_fitter.get_knot_centers()
        self.degree = int(base_fitter.degree)

        # Evaluate only inside the fitted spline support. The original scalar
        # implementation uses extrapolate=False and converts out-of-range NaNs
        # to zero, so explicit zero tails are mathematically equivalent.
        x_min = float(base_fitter.knots[self.degree])
        x_max = float(base_fitter.knots[-self.degree - 1])
        self._basis_mask = (
            (self.x_grid >= x_min)
            & (self.x_grid <= x_max)
        )
        # Construct the spline basis matrix directly rather than evaluating each basis in a
        # Python loop.
        self._basis = BSpline.design_matrix(
            self.x_grid[self._basis_mask],
            base_fitter.knots,
            self.degree,
            extrapolate=False,
        )

        self._transform_cache: Dict[
            Tuple[Tuple[float, float], ...],
            np.ndarray,
        ] = {}

    @staticmethod
    def _cache_key(
        boundaries: List[Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], ...]:
        """Return an immutable cache key for one boundary topology.

        Parameters
        ----------
        boundaries : List[Tuple[float, float]]
            Spectral-domain boundaries used by the transformation.

        Returns
        -------
        Tuple[Tuple[float, float], ...]
            Calculated arrays or records in the order described by the operation.
        """
        return tuple((float(start), float(end)) for start, end in boundaries)

    def _build_topology_transform(
        self,
        boundaries: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Build the linear domain-to-spline-coefficient transformation.

        Parameters
        ----------
        boundaries : list of tuple of float
            Energy-domain boundaries in eV.

        Returns
        -------
        numpy.ndarray
            Transformation matrix with shape
            ``(n_domains, n_coefficients)``.
        """
        n_domains = len(boundaries)
        if n_domains == 0:
            return np.zeros(
                (0, self.coeffs.size),
                dtype=np.float64,
            )

        boundaries_arr = np.asarray(boundaries, dtype=np.float64)

        # Build the exact trusted linear map inside one JIT invocation. The
        # result is cached, so subsequent batches remain pure matrix algebra.
        return _build_coefficient_transform_numba(
            self.centers,
            boundaries_arr,
            self.degree,
        )

    def _get_topology_transform(
        self,
        boundaries: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Return the cached coefficient transformation for one topology.

        Parameters
        ----------
        boundaries : List[Tuple[float, float]]
            Spectral-domain boundaries used by the transformation.

        Returns
        -------
        np.ndarray
            Numerical array evaluated on the corresponding coordinate grid.
        """
        key = self._cache_key(boundaries)
        transform = self._transform_cache.get(key)
        if transform is None:
            transform = self._build_topology_transform(boundaries)
            self._transform_cache[key] = transform
        return transform

    def apply_batch(
        self,
        boundaries: List[Tuple[float, float]],
        shifts: np.ndarray,
    ) -> np.ndarray:
        """
        Apply constrained B-spline amplitude shifts to a sample batch.

        Parameters
        ----------
        boundaries : list of tuple of float
            Energy-domain boundaries in eV.
        shifts : numpy.ndarray
            Fractional amplitude shifts with shape
            ``(n_samples, n_domains)``.

        Returns
        -------
        numpy.ndarray
            Perturbed density matrix with shape ``(n_samples, n_energy)``.

        Raises
        ------
        ValueError
            If ``shifts`` is not two-dimensional or its domain dimension does
            not match ``boundaries``.
        """
        shifts = np.asarray(shifts, dtype=np.float64)
        if shifts.ndim != 2:
            raise ValueError("Batch amplitude shifts must be two-dimensional.")
        if shifts.shape[1] != len(boundaries):
            raise ValueError(
                "The number of amplitude-shift columns must match the number "
                "of domains."
            )

        n_samples = shifts.shape[0]
        if len(boundaries) == 0 or not np.any(shifts):
            coefficient_shifts = np.zeros(
                (n_samples, self.coeffs.size),
                dtype=np.float64,
            )
        else:
            transform = self._get_topology_transform(boundaries)
            coefficient_shifts = shifts @ transform

        perturbed_coeffs = (
            self.coeffs[None, :]
            * (1.0 + coefficient_shifts)
        )

        y_batch = np.zeros(
            (n_samples, self.x_grid.size),
            dtype=np.float64,
        )

        # The cubic design matrix is sparse: each energy point touches only a
        # handful of coefficients. Sparse matrix evaluation avoids materializing
        # a potentially very large dense basis matrix.
        evaluated = self._basis @ perturbed_coeffs.T
        y_batch[:, self._basis_mask] = np.asarray(evaluated).T
        return np.nan_to_num(y_batch, nan=0.0)

    def apply(
        self,
        boundaries: List[Tuple[float, float]],
        shifts: np.ndarray,
    ) -> np.ndarray:
        """
        Apply constrained B-spline amplitude shifts to one sample.

        Parameters
        ----------
        boundaries : list of tuple of float
            Energy-domain boundaries in eV.
        shifts : numpy.ndarray
            One fractional amplitude shift per domain.

        Returns
        -------
        numpy.ndarray
            Perturbed one-dimensional density array.
        """
        shifts = np.asarray(shifts, dtype=np.float64)
        if shifts.ndim != 1:
            raise ValueError("Single-sample amplitude shifts must be one-dimensional.")
        return self.apply_batch(
            boundaries=boundaries,
            shifts=shifts[None, :],
        )[0]


# --- Main Application Orchestrator ---

def apply_amplitude_perturbation(x_grid: np.ndarray, 
                                 base_fitter: BsplineFitter, 
                                 boundaries: List[Tuple[float, float]], 
                                 shifts: np.ndarray) -> np.ndarray:
    """Applies amplitude scaling by parametrically modifying the cached B-spline.

    coefficients and evaluating the continuous analytic function directly.

    Parameters
    ----------
    x_grid : np.ndarray
        Input coordinate grid.
    base_fitter : BsplineFitter
        Previously initialized spline-fitting model.
    boundaries : List[Tuple[float, float]]
        Spectral-domain boundaries used by the transformation.
    shifts : np.ndarray
        Perturbation values assigned to the spectral domains.

    Returns
    -------
    np.ndarray
        Numerical array evaluated on the corresponding coordinate grid.
    """
    # 1. Handle empty active logic gracefully
    if len(boundaries) == 0 or len(shifts) == 0 or np.allclose(shifts, 0.0):
        # Return the analytic baseline if no shifts exist
        y_base = BSpline(base_fitter.knots, base_fitter.coeffs, base_fitter.degree, extrapolate=False)(x_grid)
        return np.nan_to_num(y_base, nan=0.0)

    # 2. Extract abstract parameter states natively from the CACHED fitter
    coeffs = base_fitter.get_coefficients()
    centers = base_fitter.get_knot_centers()
    
    # 3. Generate smooth parametric transitions
    # --- NUMBA INTEGRATION ---
    # Convert list of tuples into a strict 2D float64 array for the JIT compiler
    boundaries_arr = np.array(boundaries, dtype=np.float64)
    
    coeff_shifts = _calculate_coefficient_shifts_numba(
        centers, 
        boundaries_arr, 
        shifts, 
        base_fitter.degree
    )
    # -------------------------

    perturbed_coeffs = coeffs * (1.0 + coeff_shifts)
    
    # 4. Evaluate analytically in compiled C directly onto the provided axis
    y_modified = BSpline(base_fitter.knots, perturbed_coeffs, base_fitter.degree, extrapolate=False)(x_grid)
    
    # 5. Convert out-of-bounds NaN values back into the physical zero tail
    return np.nan_to_num(y_modified, nan=0.0)

"""
Experimental peak broadening mathematics.
"""
import numpy as np
from typing import List, Tuple
from scipy.ndimage import gaussian_filter1d


def apply_gaussian_broadening(x_grid: np.ndarray, 
                              y_density: np.ndarray, 
                              boundaries: List[Tuple[float, float]], 
                              sigmas: np.ndarray) -> np.ndarray:
    """
    Applies a localized Gaussian filter to broaden peaks within specific domains.

    Parameters
    ----------
    x_grid : np.ndarray
        The energy grid array.
    y_density : np.ndarray
        The phonon density array.
    boundaries : list of tuple of float
        The [(E_start, E_end), ...] domains bounding the broadening regions.
    sigmas : np.ndarray
        The standard deviation values (sigma) for the Gaussian kernel 
        mapped to each boundary domain.

    Returns
    -------
    np.ndarray
        The broadened phonon density array.
    """
    y_final = y_density.copy()
    
    for i, (e_start, e_end) in enumerate(boundaries):
        sigma = sigmas[i]
        
        # Bypass if no broadening is requested
        if sigma <= 0:
            continue
            
        mask = (x_grid >= e_start) & (x_grid <= e_end)
        
        if not np.any(mask):
            continue
            
        # Extract the segment and apply the filter
        segment = y_final[mask]
        # Smooth local density fluctuations before identifying robust domain boundaries.
        filtered_segment = gaussian_filter1d(segment, sigma=sigma, mode='reflect')
        
        # Secure exact original magnitudes at the domain borders[cite: 6]
        filtered_segment[0] = segment[0]
        filtered_segment[-1] = segment[-1]
        
        y_final[mask] = filtered_segment
        
    # Safeguard: Ensure no negative artifacts are generated
    if np.any(y_final < 0):
        y_final = np.clip(y_final, 0.0, None)
        
    return y_final

"""
Workflows and batch processing utilities for NCrystal files.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from input_output.ncrystal.exceptions import NCrystalFormatError
from input_output.ncrystal.core import NCrystalFile


def plot_spectrum(ncrystal_file: NCrystalFile, save_path: Optional[str] = None) -> None:
    """
    Plots the primary Phonon Density of States (VDOS) of an NCrystal instance.

    Parameters
    ----------
    ncrystal_file : NCrystalFile
        The NCrystal file instance containing the data to plot.
    save_path : str, optional
        The path where the figure should be saved. If None, the plot is shown.
    """
    try:
        pdos = ncrystal_file.pdos
    except NCrystalFormatError as e:
        print(e)
        return

    plt.plot(pdos.index, pdos.values, color='green')
    plt.title(f"NCrystal VDOS - {ncrystal_file.state_of_substance or 'Material'}")
    plt.xlabel("Energy (eV)")
    plt.ylabel("VDOS(E)")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def interpolate_spectrum(ncrystal_file: NCrystalFile, 
                         input_energy_axis: np.ndarray, 
                         input_spectrum_values: np.ndarray) -> np.ndarray:
    """
    Interpolates a new spectrum onto the NCrystalFile's original energy grid.

    Parameters
    ----------
    ncrystal_file : NCrystalFile
        The NCrystal file instance establishing the target energy grid.
    input_energy_axis : np.ndarray
        The energy axis of the incoming spectrum.
    input_spectrum_values : np.ndarray
        The amplitude values of the incoming spectrum.

    Returns
    -------
    np.ndarray
        The interpolated phonon spectrum.

    Raises
    ------
    NCrystalFormatError
        If the primary energy grid is missing from the file structure.
    """
    try:
        target_energy_axis = ncrystal_file.pdos.index.to_numpy()
    except NCrystalFormatError:
        raise NCrystalFormatError(
            "NCrystalFile energy grid is not loaded. Cannot interpolate."
        )
    
    interpolated_spectrum = np.interp(
        target_energy_axis,
        input_energy_axis,
        input_spectrum_values
    )
    return interpolated_spectrum
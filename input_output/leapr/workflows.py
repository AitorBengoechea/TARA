"""
Workflows and batch processing utilities for LEAPR files.
"""
import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

from input_output.leapr.exceptions import LeaprFormatError
from input_output.leapr.discrete_export import prepare_discrete_vibrational_block
from input_output.leapr.vibrational import tabulate_vibrational_block
from core.physics.vibrational_spectrum import DEFAULT_OSCILLATOR_SIGMA_EV
from input_output.spectrum_grid import resample_on_source_spacing
from input_output.leapr.core import LeaprFile
from input_output.leapr.writer import LeaprWriter
from input_output.leapr.temperature import resolve_leapr_temperature, select_leapr_temperature_block
from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import MaterialScatteringModel


def plot_spectrum(leapr_file: LeaprFile, save_path: Optional[str] = None) -> None:
    """
    Plot the full vibrational spectrum at the first positive temperature.

    Parameters
    ----------
    leapr_file : LeaprFile
        The LEAPR file instance containing the data to plot.
    save_path : str, optional
        The path where the figure should be saved. If None, the plot is shown.
    """
    if leapr_file.first_positive_spectrum is None:
        print("No spectrum to plot.")
        return

    spectrum = leapr_file.pdos
    plt.plot(spectrum.index, spectrum.values)
    temp = '?'
    for block in leapr_file.temp_blocks:
        if block['temp'] > 0:
            temp = block['temp']
            break
    
    plt.title(f"Phonon Density of States (T={temp} K)")
    plt.xlabel("Energy (eV)")
    plt.ylabel("ρ(E)")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def interpolate_spectrum(leapr_file: LeaprFile, 
                         input_energy_axis: np.ndarray, 
                         input_spectrum_values: np.ndarray) -> np.ndarray:
    """
    Interpolates a new spectrum onto the LeaprFile's original energy grid.

    Parameters
    ----------
    leapr_file : LeaprFile
        The LEAPR file instance establishing the target energy grid.
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
    LeaprFormatError
        If the `first_positive_energy_grid` of the LeaprFile is not loaded.
    """
    if leapr_file.first_positive_energy_grid is None:
        raise LeaprFormatError(
            "LeaprFile `first_positive_energy_grid` is not loaded."
        )
    
    target_energy_axis = leapr_file.first_positive_energy_grid
    interpolated_spectrum = np.interp(
        target_energy_axis,
        input_energy_axis,
        input_spectrum_values
    )
    return interpolated_spectrum


def _create_first_writer_copy(leapr_file: LeaprFile,
                              template_block: Dict[str, Any],
                              spectrum: np.ndarray,
                              kwargs: Dict[str, Any],
                              temperature: Optional[float] = None) -> LeaprFile:
    """
    Handles creating the first writer copy during batch processing.

    Parameters
    ----------
    leapr_file : LeaprFile
        The base instance to copy from.
    template_block : dict
        The template data block for the temperature run.
    spectrum : np.ndarray
        The first phonon spectrum to insert.
    kwargs : dict
        Additional parameters for initializing the copy.
    temperature : float or None, optional
        Validated single output temperature, replacing the template temperature.

    Returns
    -------
    LeaprFile
        The newly initialized writer copy instance.
    """
    new_temp_block = copy.deepcopy(template_block)
    new_temp_block['phonon_spectrum'] = spectrum
    new_temp_block['ni'] = len(spectrum)
    if temperature is not None:
        new_temp_block['temp'] = temperature
    
    kwargs_for_copy = kwargs.copy()
    if 'temp_blocks' in kwargs_for_copy:
         print("Warning: 'temp_blocks' in kwargs is being overridden.", file=sys.stderr)
    if 'run_control_params' in kwargs_for_copy:
         print("Warning: 'run_control_params' partially overridden.", file=sys.stderr)

    base_run_control = kwargs.get('run_control_params', leapr_file.run_control_params)
    kwargs_for_copy['temp_blocks'] = [new_temp_block]
    kwargs_for_copy['run_control_params'] = {**base_run_control, 'ntempr': 1}
    
    # Preserve the validated LEAPR template while preparing one sample-specific representation.
    new_base = leapr_file.shallow_clone()
    return new_base.get_writer_copy(**kwargs_for_copy)


def _modify_writer_copy_inplace(writer_copy: LeaprFile, spectrum: np.ndarray) -> None:
    """
    Modifies an active writer copy in-place for fast batch serialization.

    Parameters
    ----------
    writer_copy : LeaprFile
        The active file object being mutated.
    spectrum : np.ndarray
        The new phonon spectrum to inject.
    """
    writer_copy.temp_blocks[0]['phonon_spectrum'] = spectrum
    writer_copy.temp_blocks[0]['ni'] = len(spectrum)
    writer_copy.first_positive_spectrum = spectrum
    writer_copy.first_positive_energy_grid = (
        np.arange(len(spectrum), dtype=np.float64)
        * float(writer_copy.temp_blocks[0]['delta'])
    )



def _interpolate_and_normalize(
    leapr_file: LeaprFile,
    spectrum: PhononSpectra,
    target_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Map a TARA spectrum onto the selected LEAPR grid and normalize it.

    Parameters
    ----------
    leapr_file : LeaprFile
        Parsed LEAPR object providing the default target grid.
    spectrum : PhononSpectra
        Source phonon spectrum with energy coordinates in eV.
    target_grid : numpy.ndarray or None, optional
        Target energy grid in eV. Defaults to the grid of the first
        positive-temperature block in ``leapr_file``.

    Returns
    -------
    numpy.ndarray
        Density in inverse eV on the target grid, normalized to unit
        trapezoidal integral. Interpolation is linear with zero density
        outside the source grid.

    Raises
    ------
    LeaprFormatError
        If the interpolated area is nonpositive or nonfinite.
    """
    target_grid = np.asarray(
        leapr_file.first_positive_energy_grid if target_grid is None else target_grid,
        dtype=np.float64,
    )
    interpolated = np.interp(
        target_grid,
        np.asarray(spectrum.energy_grid, dtype=np.float64),
        np.asarray(spectrum.density_values, dtype=np.float64),
        left=0.0, right=0.0,
    )
    area = np.trapezoid(interpolated, target_grid)
    if not np.isfinite(area) or area <= 0.0:
        raise LeaprFormatError('Interpolated phonon spectrum must have positive finite area.')
    return interpolated / area


def inject_leapr_spectrum(
    leapr_obj: LeaprFile,
    spectrum: PhononSpectra,
    *,
    export_full_spectra: bool = False,
) -> None:
    """Inject one TARA spectrum into the first positive LEAPR block in place.

    Parameters
    ----------
    leapr_obj : LeaprFile
        LEAPR format object to update.
    spectrum : PhononSpectra
        Full sampled phonon spectrum.
    export_full_spectra : bool, optional
        Export the entire continuous VDOS with ND=0.

    Raises
    ------
    LeaprFormatError
        If the input or required state violates the operation contract.
    """
    writer = prepare_leapr_spectrum(
        leapr_obj, spectrum, export_full_spectra=export_full_spectra,
    )
    for index, block in enumerate(leapr_obj.temp_blocks):
        if block['temp'] > 0:
            leapr_obj.temp_blocks[index] = writer.temp_blocks[0]
            leapr_obj.first_positive_spectrum = writer.first_positive_spectrum
            leapr_obj.first_positive_energy_grid = writer.first_positive_energy_grid
            leapr_obj.__dict__.pop('pdos', None)
            return


def prepare_leapr_spectrum(
    leapr_file: LeaprFile,
    spectrum: PhononSpectra,
    *,
    temperature: Optional[float] = None,
    warn_on_temperature_reuse: bool = True,
    preserve_temperature_parameters: bool = False,
    output_spacing_ev: float | None = None,
    export_full_spectra: bool = False,
    **kwargs,
) -> LeaprFile:
    """Create a single-temperature writer copy containing one TARA spectrum.

    Parameters
    ----------
    leapr_file : LeaprFile
        Filesystem location of the leapr file.
    spectrum : PhononSpectra
        Complete vibrational spectrum, including any originally discrete modes.
    temperature : float or None, optional
        Requested temperature in kelvin, bounded by the original template.
    warn_on_temperature_reuse : bool, optional
        Emit a warning if the first template spectrum is reused at another temperature.
    preserve_temperature_parameters : bool, optional
        Use exactly the tabulated source block (diffusion, weights, oscillators)
        at ``temperature``. Mode-bearing inputs always require exact selection.
        ND=0 solid inputs retain the original temperature-reuse policy.
    output_spacing_ev : float or None, optional
        Uniform spacing in eV for the exported LEAPR table. Defaults to
        the prepared grid. The input spectrum and its sampling grid are
        unaffected. Use a convergence check before coarsening the table.
    export_full_spectra : bool, optional
        Export the entire continuous VDOS with ND=0. By default, recover
        any discrete modes from the sampled spectrum on the original grid.
    **kwargs : Any
        Additional writer-copy overrides.

    Returns
    -------
    LeaprFile
        Calculated result in the declared return type.

    Raises
    ------
    LeaprFormatError
        If the input or required state violates the operation contract.
    """
    if leapr_file.first_positive_energy_grid is None:
        raise LeaprFormatError("Energy grid not loaded. Cannot interpolate.")

    has_source_modes = any(
        int(b.get('vibrational_source', b).get('nd', 0)) > 0
        for b in leapr_file.temp_blocks
    )
    if preserve_temperature_parameters or has_source_modes:
        source_block, output_temperature = select_leapr_temperature_block(
            leapr_file, temperature,
        )
    else:
        source_block = next(
            (block for block in leapr_file.temp_blocks if block['temp'] > 0), None,
        )
        if source_block is None:
            raise LeaprFormatError('No positive temperature block found in LEAPR template.')
        output_temperature = resolve_leapr_temperature(
            leapr_file, temperature, warn_on_reuse=warn_on_temperature_reuse,
        )
    native_block = source_block.get('vibrational_source', source_block)
    if not export_full_spectra and int(native_block.get('nd', 0)) > 0:
        if output_spacing_ev is not None:
            raise ValueError('output_spacing_ev requires export_full_spectra=True for discrete-mode inputs.')
        prepared = source_block if native_block is not source_block else None
        result = prepare_discrete_vibrational_block(native_block, spectrum, prepared=prepared)
        return _create_first_writer_copy(
            leapr_file, result, result['phonon_spectrum'], kwargs,
            temperature=output_temperature,
        )
    # A sampled curve already contains the modes: only change the output
    # representation here, never add the original modes a second time.
    if int(source_block.get('nd', 0)) > 0:
        if spectrum.energy_grid[-1] < max(source_block['oscillator_energies']):
            raise LeaprFormatError('Expected a full vibrational spectrum including discrete modes.')
        # Prefer the incoming grid when it already resolves the modes.
        grid = spectrum.energy_grid
        spacing = float(grid[1])
        if (spacing <= DEFAULT_OSCILLATOR_SIGMA_EV / 3.0 * (1.0 + 1e-12)
                and np.allclose(np.diff(grid), spacing, rtol=1e-8, atol=1e-12)):
            source_block = copy.deepcopy(source_block)
            source_block.update(
                nd=0, oscillator_energies=np.array([]), oscillator_weights=np.array([]),
                tbeta=float(source_block['tbeta']) + sum(source_block['oscillator_weights']),
                vibrational_source=copy.deepcopy(source_block),
                delta=spacing, ni=len(grid),
            )
        else:
            # Reconstruct only to obtain output metadata and the fine grid.
            # The sampled density replaces the prepared reference below.
            source_block = tabulate_vibrational_block(source_block)
    target_grid = np.arange(int(source_block['ni'])) * float(source_block['delta'])
    if output_spacing_ev is not None:
        spacing = float(output_spacing_ev)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("LEAPR output spacing must be positive and finite.")
        upper = max(target_grid[-1], spectrum.energy_grid[-1])
        last_index = max(1, int(np.ceil(upper / spacing - 1e-12)))
        target_grid = np.arange(last_index + 1, dtype=float) * spacing
        source_block = {**source_block, 'delta': spacing}
    target_grid, _ = resample_on_source_spacing(
        target_grid, spectrum.energy_grid, spectrum.density_values,
    )
    interpolated = _interpolate_and_normalize(leapr_file, spectrum, target_grid)
    first_pos_template = copy.deepcopy(source_block)
    return _create_first_writer_copy(
        leapr_file, first_pos_template, interpolated, kwargs,
        temperature=output_temperature,
    )


def prepare_leapr_material(
    leapr_file: LeaprFile, model: MaterialScatteringModel, *,
    temperature: Optional[float] = None,
    export_full_spectra: bool = False,
) -> LeaprFile:
    """Prepare an exact-temperature liquid LEAPR writer with sampled dynamics.

    Parameters
    ----------
    leapr_file : LeaprFile
        Untouched source material template.
    model : MaterialScatteringModel
        Validated physical realization for the requested temperature.
    temperature : float or None
        Exact tabulated source temperature, defaulting to model temperature.
    export_full_spectra : bool, optional
        Export the entire continuous VDOS with ND=0 instead of discrete modes.

    Returns
    -------
    LeaprFile
        Sample-specific independent writer copy.
    """
    if model.state != "liquid":
        raise ValueError("Material serializer requires liquid state.")
    target_temperature = model.temperature_k if temperature is None else temperature
    if not np.isclose(target_temperature, model.temperature_k, rtol=0, atol=1e-8):
        raise ValueError("Material temperature must match LEAPR output temperature.")
    if not np.isclose(model.principal.mass_ratio, float(leapr_file.principal_scatterer['awr']), rtol=1e-9):
        raise ValueError("Sampled material principal mass differs from LEAPR template.")
    if model.dynamics.oscillators:
        raise ValueError('Tabulate discrete modes before serializing sampled LEAPR material.')
    writer = prepare_leapr_spectrum(
        leapr_file, model.dynamics.continuous, temperature=target_temperature,
        warn_on_temperature_reuse=False, preserve_temperature_parameters=True,
        export_full_spectra=export_full_spectra,
    )
    block = writer.temp_blocks[0]
    block['twt'] = model.dynamics.translation_weight
    block['c'] = model.dynamics.translation_parameter or 0.0
    if block['nd']:
        vibrational_weight = block['tbeta'] + sum(block['oscillator_weights'])
        scale = model.dynamics.continuous_weight / vibrational_weight
        block['tbeta'] *= scale
        block['oscillator_weights'] = np.asarray(block['oscillator_weights']) * scale
    else:
        block['tbeta'] = model.dynamics.continuous_weight
    return writer


def serialize_leapr_material(
    leapr_file: LeaprFile, model: MaterialScatteringModel, *,
    temperature: Optional[float] = None,
    export_full_spectra: bool = False,
) -> str:
    """Serialize sampled liquid dynamics without changing the source template."""
    writer = prepare_leapr_material(
        leapr_file, model, temperature=temperature, export_full_spectra=export_full_spectra,
    )
    return LeaprWriter(writer).to_string()


def serialize_leapr_spectrum(
    leapr_file: LeaprFile,
    spectrum: PhononSpectra,
    *,
    temperature: Optional[float] = None,
    warn_on_temperature_reuse: bool = True,
    preserve_temperature_parameters: bool = False,
    output_spacing_ev: float | None = None,
    export_full_spectra: bool = False,
    **kwargs,
) -> str:
    """Serialize a template-derived, sample-specific LEAPR module in memory.

    Parameters
    ----------
    leapr_file : LeaprFile
        Filesystem location of the leapr file.
    spectrum : PhononSpectra
        Complete vibrational spectrum, including any originally discrete modes.
    temperature : float or None, optional
        Target LEAPR temperature in kelvin. Defaults to the source spectrum's
        temperature when omitted.
    warn_on_temperature_reuse : bool, optional
        Warn when the temperature differs from the original source spectrum.
    preserve_temperature_parameters : bool, optional
        Require an exact tabulated temperature and retain its source parameters.
    output_spacing_ev : float or None, optional
        Uniform spacing in eV for the exported LEAPR table. Defaults to
        the prepared grid. The input spectrum and its sampling grid are
        unaffected. Use a convergence check before coarsening the table.
    export_full_spectra : bool, optional
        Export the entire continuous VDOS with ND=0. By default, recover
        any discrete modes from the sampled spectrum on the original grid.
    **kwargs : Any
        Additional writer-copy options.

    Returns
    -------
    str
        Serialized single-temperature LEAPR module.
    """
    writer_copy = prepare_leapr_spectrum(
        leapr_file, spectrum, temperature=temperature,
        warn_on_temperature_reuse=warn_on_temperature_reuse,
        preserve_temperature_parameters=preserve_temperature_parameters,
        output_spacing_ev=output_spacing_ev, export_full_spectra=export_full_spectra, **kwargs,
    )
    return LeaprWriter(writer_copy).to_string()

def write_multiple_spectra(leapr_file: LeaprFile,
                           spectra_matrix: pd.DataFrame,
                           output_base: Optional[str] = None,
                           directory: str = ".",
                           *,
                           temperature: Optional[float] = None,
                           **kwargs) -> None:
    """Writes a series of LEAPR files, one per spectrum (row) in the DataFrame.

    Every row uses the same full-spectrum serialization contract.

    Parameters
    ----------
    leapr_file : LeaprFile
        The original LEAPR object supplying grid templates and base metadata.
    spectra_matrix : pd.DataFrame
        DataFrame where each row is a phonon spectrum, and the columns
        are the energy grid (in eV) for that spectrum.
    output_base : str, optional
        The base name for output files (e.g., "gr_sample").
        If None, defaults to the base name of the original file.
    directory : str, optional
        The directory to write the new files into. Defaults to ".".
    temperature : float or None, optional
        Requested output temperature, subject to the original template range.
    **kwargs :
        Optional keyword arguments passed to `get_writer_copy()`
        to modify the *base properties* of all output files.

    Raises
    ------
    LeaprFormatError
        If the necessary temperature blocks or energy grids are missing.
    """
    if leapr_file.first_positive_energy_grid is None:
        raise LeaprFormatError("Energy grid not loaded. Cannot interpolate.")
    if not leapr_file.temp_blocks:
        raise LeaprFormatError("Temperature blocks not loaded. Cannot write file.")
    if spectra_matrix.empty:
        print("Warning: Empty spectra_matrix provided. No files written.", file=sys.stderr)
        return

    output_temperature = resolve_leapr_temperature(leapr_file, temperature)

    # Create the destination before writing independently named ensemble outputs.
    os.makedirs(directory, exist_ok=True)
    
    if output_base is None:
        base = os.path.basename(leapr_file.filepath)
        base_no_ext = os.path.splitext(base)[0]
        output_base = f"{base_no_ext}_sample"

    input_energy_axis = spectra_matrix.columns.to_numpy(dtype=np.float64)

    print(f"Preparing to write {len(spectra_matrix)} spectra to '{directory}'...")
    
    for index, row in spectra_matrix.iterrows():
        spectrum = PhononSpectra(pd.Series(row.to_numpy(dtype=float), index=input_energy_axis))
        writer = prepare_leapr_spectrum(
            leapr_file, spectrum, temperature=output_temperature,
            warn_on_temperature_reuse=False, **kwargs,
        )
        out_path = os.path.join(directory, f"{output_base}{index + 1}.leapr")
        writer.write(out_path)
        print(f"Wrote spectrum {index + 1} -> {out_path}")


def prepare_full_vibrational_template(leapr_file, *, temperature=None, energy_grid=None):
    """Prepare one temperature and common grid before ensemble sampling.

    Parameters
    ----------
    leapr_file : LeaprFile
        Original parsed input, retained unchanged as provenance.
    temperature : float or None
        Exact tabulated temperature in kelvin; None selects the first block.
    energy_grid : array_like or None
        Optional uniform full-spectrum grid in eV. For ND>0, choose its spacing
        through convergence studies and include margin for sampled shifts.

    Returns
    -------
    LeaprFile
        Independent single-temperature sampling template. Its ``pdos`` is the
        sampling reference. Pass the original LEAPR file to processing workers
        so they can recover the native discrete modes during export.
        ``vibrational_source`` in its block retains the original components.
    """
    block, selected = select_leapr_temperature_block(leapr_file, temperature)
    block = tabulate_vibrational_block(block, energy_grid)
    block['temp'] = selected
    return leapr_file.get_writer_copy(
        temp_blocks=[block],
        run_control_params={**leapr_file.run_control_params, 'ntempr': 1},
    )

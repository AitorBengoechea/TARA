"""Reusable visualizations for TARA sampling and physical analysis notebooks.

Phonon spectra are stored on an eV grid with densities in 1/eV. The PDOS
axis displays meV and 1/meV through tick formatters without changing the
underlying coordinates. Consequently, region annotations use eV.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

from core.data.material_scattering import MaterialScatteringModel


_XS_LABELS = {
    "xs_total": "Total scattering cross section",
    "xs_inelastic": "Inelastic scattering cross section",
    "xs_coherent_elastic": "Coherent-elastic scattering cross section",
    "xs_incoherent_elastic": "Incoherent-elastic scattering cross section",
    "xs_absorption": "Absorption cross section",
}


def _spectrum_xy(spectrum):
    """Return numeric PDOS coordinates from a spectrum or material.

    Parameters
    ----------
    spectrum : PhononSpectra, MaterialScatteringModel, or pandas.Series
        Vibrational input, or material whose continuous spectrum is displayed.

    Returns
    -------
    tuple of numpy.ndarray
        Energy in eV and density in inverse eV.
    """
    if isinstance(spectrum, MaterialScatteringModel):
        spectrum = spectrum.dynamics.continuous
    if isinstance(spectrum, pd.Series):
        return (spectrum.index.to_numpy(dtype=float),
                spectrum.to_numpy(dtype=float))
    return (np.asarray(spectrum.energy_grid, dtype=float),
            np.asarray(spectrum.density_values, dtype=float))


def plot_PDOS(
    ax,
    data,
    pdos_ref,
    title="PDOS Analysis",
    ref_label="Reference",
    title_fontsize=20,
    xrange=None,
    xlimMax=None,
):
    """Draw stochastic PDOS curves and a distinguished reference spectrum.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Destination axes.
    data : sequence of PhononSpectra or pandas.DataFrame
        Stochastic spectra. A DataFrame uses sample rows and energy columns.
    pdos_ref : PhononSpectra, pandas.Series, or pandas.DataFrame
        The reference spectrum. For legacy positional calls, this argument may
        instead be the title, with the actual reference passed as ``title``.
    title : str, default='PDOS Analysis'
        Panel heading.
    ref_label : str, default='Reference'
        Legend label for the baseline.
    title_fontsize : int, default=20
        Title font size.
    xrange : tuple of float, optional
        Boundary markers in eV.
    xlimMax : float, optional
        Optional right axis bound in eV.

    Returns
    -------
    matplotlib.axes.Axes
        The updated axes.
    """
    # Support the historical call: plot_PDOS(ax, df, title, reference, label).
    if isinstance(pdos_ref, str) and not isinstance(title, str):
        title, pdos_ref = pdos_ref, title

    colors = plt.cm.tab10.colors
    if isinstance(data, pd.DataFrame):
        for sample_index, (_, row) in enumerate(data.iterrows()):
            ax.plot(data.columns.to_numpy(dtype=float), row.to_numpy(dtype=float),
                    color=colors[sample_index % len(colors)], linewidth=1.0, zorder=2)
    else:
        for sample_index, spectrum in enumerate(data):
            energy, density = _spectrum_xy(spectrum)
            ax.plot(energy, density,
                    color=colors[sample_index % len(colors)], linewidth=1.0,
                    zorder=2)

    references = (
        [pdos_ref[column] for column in pdos_ref.columns]
        if isinstance(pdos_ref, pd.DataFrame)
        else [pdos_ref]
    )
    for index, spectrum in enumerate(references):
        energy, density = _spectrum_xy(spectrum)
        label = ref_label if len(references) == 1 else f"{ref_label} {index + 1}"
        ax.plot(energy, density, color="black" if index == 0 else f"C{index + 1}",
                linewidth=2.0, label=label, zorder=3)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Energy [meV]", fontsize=14)
    ax.set_ylabel("Phonon spectrum [1/meV]", fontsize=14)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 1000:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1000:g}"))
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    if xlimMax is not None:
        ax.set_xlim(right=xlimMax)
    if xrange is not None:
        for edge in xrange:
            ax.axvline(edge, color="gray", linestyle="--", linewidth=1.5)
    ax.legend(loc="best", fontsize=11)
    return ax


def _clean_xs(series):
    """Convert a backend XS Series to sorted, finite numeric coordinates."""
    if not isinstance(series, pd.Series):
        raise TypeError("Cross sections must be pandas.Series objects.")
    coordinates = np.asarray(series.index, dtype=float)
    values = np.asarray(series, dtype=float)
    if coordinates.size != values.size or not np.isfinite(coordinates).all():
        raise ValueError("Cross-section energy coordinates must be finite.")
    if np.any(coordinates <= 0):
        raise ValueError("Incident neutron energies must be positive for log axes.")
    if not np.isfinite(values).all():
        raise ValueError("Cross-section values must be finite.")
    if np.any(values < 0):
        raise ValueError("Cross-section values must be nonnegative.")
    clean = pd.Series(values, index=coordinates)
    clean = clean.groupby(level=0).mean().sort_index()
    return clean


def _aligned_xs(reference, sample):
    """Interpolate a sample onto the reference grid without extrapolation."""
    reference = _clean_xs(reference)
    sample = _clean_xs(sample)
    x = reference.index.to_numpy(dtype=float)
    if sample.empty:
        return pd.Series(np.full(x.shape, np.nan), index=reference.index)
    sampled = np.interp(
        x,
        sample.index.to_numpy(dtype=float),
        sample.to_numpy(dtype=float),
        left=np.nan, right=np.nan,
    )
    return pd.Series(sampled, index=reference.index)


def _collect_xs(analyzer, xs_type):
    """Get one physical channel from actual Analyzer results.

    Never manufacture a missing channel. Keep reference and stochastic samples
    distinct, using the independently evaluated reference's energy grid.
    """
    if xs_type not in _XS_LABELS:
        raise ValueError(f"Unknown cross-section channel: {xs_type!r}")
    baseline = getattr(analyzer, "reference_result", None)
    if baseline is None or getattr(baseline.integ, xs_type) is None:
        raise ValueError(f"Reference {xs_type} is unavailable.")
    reference = _clean_xs(getattr(baseline.integ, xs_type))
    rows = []
    valid_indices = []
    for result in analyzer.raw_results:
        series = getattr(result.integ, xs_type)
        if series is None:
            continue
        aligned = _aligned_xs(reference, series)
        rows.append(aligned.to_numpy(dtype=float))
        valid_indices.append(int(result.sample_index))
    if not rows:
        raise ValueError(f"No stochastic {xs_type} cross sections are available.")
    samples = pd.DataFrame(
        np.asarray(rows, dtype=float).T,
        index=reference.index,
        columns=[f"Sample {index + 1}" for index in valid_indices],
    )
    return reference, samples, valid_indices


def _relative_xs_error(reference, samples):
    """Return 100*(sample-reference)/reference, undefined at zero reference."""
    baseline = reference.to_numpy(dtype=float)
    sample_values = samples.to_numpy(dtype=float)
    valid = np.isfinite(baseline) & (baseline != 0.0)
    errors = np.full(sample_values.shape, np.nan, dtype=float)
    np.divide(
        100.0 * (sample_values - baseline[:, None]),
        baseline[:, None],
        out=errors,
        where=valid[:, None] & np.isfinite(sample_values),
    )
    return pd.DataFrame(errors, index=samples.index, columns=samples.columns)


def plot_XS(ax, data, comp_label="Reference", *, reference=None, xs_type="xs_total"):
    """Plot sampled and reference cross sections on logarithmic axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Destination axes.
    data : pandas.DataFrame
        Sample curves, energy-indexed. For a legacy call with no ``reference``,
        the first column is treated as the reference and the rest as samples.
    comp_label : str, default='Reference'
        Label for the baseline.
    reference : pandas.Series, optional
        Separately evaluated baseline.
    xs_type : str, default='xs_total'
        Name of the integral channel, used in the axis title.

    Returns
    -------
    matplotlib.axes.Axes
        The updated axes.
    """
    if xs_type not in _XS_LABELS:
        raise ValueError(f"Unknown cross-section channel: {xs_type!r}")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Cross-section samples must be a pandas.DataFrame.")
    if reference is None:
        if data.shape[1] == 0:
            raise ValueError("Cross-section data has no columns.")
        reference = data.iloc[:, 0]
        data = data.iloc[:, 1:]
    colors = plt.cm.tab10.colors
    reference = _clean_xs(reference)
    for sample_index, (name, sample) in enumerate(data.items()):
        xs = _clean_xs(sample.dropna()) if sample.notna().any() else sample
        if len(xs):
            ax.plot(xs.index, np.where(xs.values > 0, xs.values, np.nan),
                    color=colors[sample_index % len(colors)],
                    linewidth=1.25)
    ax.plot(reference.index,
            np.where(reference.values > 0, reference.values, np.nan),
            color="black", linewidth=2.2, label=comp_label, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(f"{_XS_LABELS[xs_type]} [barn]", fontsize=13)
    ax.set_xlabel("Incident neutron energy [eV]", fontsize=13)
    ax.legend(loc="best")
    return ax


def plot_relative_diff(
    ax,
    data,
    xrange_zoom=(0.002, 5.0),
    inset_pos=(0.5, 0.05, 0.45, 0.42),
    *,
    show_inset=False,
):
    """Plot relative cross-section differences as percentages.

    ``NaN`` means that the reference was zero or the sample lacked coverage.
    The optional inset is omitted by default in the compact dashboard.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Relative errors must be a pandas.DataFrame.")
    colors = plt.cm.tab10.colors
    for sample_index, (_, sample) in enumerate(data.items()):
        ax.plot(data.index, sample.to_numpy(dtype=float),
                color=colors[sample_index % len(colors)],
                linewidth=1.25)
    ax.set_xscale("log")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.3)
    ax.set_ylabel("Relative difference [%]", fontsize=13)
    ax.set_xlabel("Incident neutron energy [eV]", fontsize=13)
    if show_inset:
        visible = data.loc[(data.index >= xrange_zoom[0]) &
                           (data.index <= xrange_zoom[1])]
        finite = visible.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            inset = ax.inset_axes(inset_pos)
            for sample_index, (_, sample) in enumerate(visible.items()):
                inset.plot(visible.index, sample.to_numpy(dtype=float),
                           color=colors[sample_index % len(colors)],
                           linewidth=1.25)
            inset.set_xscale("log")
            inset.axhline(0, color="black", linestyle="--", linewidth=1)
            minimum, maximum = float(finite.min()), float(finite.max())
            padding = max(0.05 * (maximum - minimum), 0.01)
            inset.set_xlim(*xrange_zoom)
            inset.set_ylim(minimum - padding, maximum + padding)
            inset.tick_params(labelsize=8)
            inset.set_xlabel("")
            inset.set_ylabel("")
    return ax


def _available_thermo_properties(analyzer, properties):
    """Return requested scalar thermodynamic fields present on reference and samples."""
    supported = (
        "msd", "teff", "cv", "entropy", "zero_point_energy",
        "free_energy", "internal_energy", "log_moment",
    )
    requested = supported if properties == "all" else tuple(properties)
    baseline = getattr(analyzer, "reference_result", None)
    if baseline is None or baseline.thermo is None:
        raise ValueError("Reference thermodynamics are required for property plots.")
    valid = [name for name in requested if name in supported
             and getattr(baseline.thermo, name, None) is not None]
    if not valid:
        raise ValueError("None of the requested thermodynamic properties are available.")
    return valid


def _thermo_plot_data(analyzer, properties):
    """Adapt Analyzer thermodynamics to the existing statistical plotters."""
    data = {}
    for prop in properties:
        values = [getattr(analyzer.reference_result.thermo, prop)]
        indices = [-1]
        for result in analyzer.raw_results:
            value = getattr(result.thermo, prop, None)
            if value is not None and np.isfinite(value):
                values.append(value)
                indices.append(int(result.sample_index))
        data[prop] = pd.Series(values, index=indices, dtype=float)
    return SimpleNamespace(**data)


def plot_master_dashboard(
    data_obj,
    ref_label="Reference",
    properties=None,
    pdos_title="PDOS Analysis",
    main_title=None,
    plot_pdos=True,
    plot_scatter=False,
    plot_histo=False,
    plot_xs=False,
    plot_error=False,
    savepath=None,
    *,
    xs_type="xs_total",
    show_error_inset=False,
):
    """Plot sample PDOS, thermodynamic distributions, cross sections, and errors.

    Parameters
    ----------
    data_obj : EnsembleAnalyzer
        Computed analyzer, including ``ensemble``, ``raw_results`` and a
        separately evaluated ``reference_result``.
    ref_label : str, default='Reference'
        Label applied to both baseline curves.
    properties : sequence of str or 'all', optional
        Thermodynamic fields displayed in optional scatter and histogram rows.
    pdos_title : str, default='PDOS Analysis'
        Title for the top PDOS panel.
    main_title : str, optional
        Figure heading.
    plot_pdos, plot_scatter, plot_histo, plot_xs, plot_error : bool
        Enable individual panels. Relative errors require cross sections.
    savepath : str or pathlib.Path, optional
        Destination for the completed figure.
    xs_type : str, default='xs_total'
        One of the five integrated cross-section channels.
    show_error_inset : bool, default=False
        Add an inset zoom on the relative-error panel.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, dict)
        Figure and axes keyed by panel name.

    Raises
    ------
    ValueError
        If panel requests are incompatible, or physical data are unavailable.
    """
    if plot_error and not plot_xs:
        raise ValueError("plot_error=True requires plot_xs=True.")
    if (plot_scatter or plot_histo) and properties is None:
        raise ValueError("properties must be provided for scatter/histograms.")
    if not any((plot_pdos, plot_scatter, plot_histo, plot_xs)):
        raise ValueError("At least one plot type must be enabled.")

    selected = (
        _available_thermo_properties(data_obj, properties)
        if (plot_scatter or plot_histo) else []
    )
    thermo = _thermo_plot_data(data_obj, selected) if selected else None

    reference = samples = None
    valid_indices = []
    if plot_xs:
        reference, samples, valid_indices = _collect_xs(data_obj, xs_type)

    row_names = [
        name for name, enabled in (
            ("pdos", plot_pdos),
            ("scatter", plot_scatter),
            ("histo", plot_histo),
            ("xs", plot_xs),
            ("error", plot_error),
        ) if enabled
    ]
    ratios = {"pdos": 1.2, "scatter": 1.2, "histo": 1.2,
              "xs": 1.3, "error": 1.0}
    columns = max(1, len(selected))
    width = max(12, 4 * columns)
    fig = plt.figure(figsize=(width, 4.0 * len(row_names) + 1.5),
                     layout="constrained")
    grid = GridSpec(
        len(row_names), columns, figure=fig,
        height_ratios=[ratios[name] for name in row_names],
    )
    axes = {}
    for index, name in enumerate(row_names):
        if name in ("scatter", "histo"):
            axes[name] = np.asarray(
                [fig.add_subplot(grid[index, col]) for col in range(columns)],
                dtype=object,
            )
        else:
            axes[name] = fig.add_subplot(grid[index, :])

    if plot_pdos:
        ensemble = data_obj.ensemble
        if ensemble.reference is None:
            raise ValueError("The ensemble has no PDOS reference.")
        spectra = ([ensemble[index] for index in valid_indices]
                   if plot_xs else list(ensemble))
        plot_PDOS(axes["pdos"], spectra, ensemble.reference,
                  title=pdos_title, ref_label=ref_label)
    if plot_scatter:
        plot_sampling_results(thermo, properties=selected, ax=axes["scatter"])
    if plot_histo:
        plot_sampling_histograms(thermo, properties=selected, ax=axes["histo"])
    if plot_xs:
        plot_XS(axes["xs"], samples, ref_label,
                reference=reference, xs_type=xs_type)
    if plot_error:
        errors = _relative_xs_error(reference, samples)
        plot_relative_diff(axes["error"], errors,
                           show_inset=show_error_inset)
    if main_title:
        fig.suptitle(main_title, fontsize=19)
    if savepath is not None:
        fig.savefig(Path(savepath), dpi=300, bbox_inches="tight")
    return fig, axes


# Existing sampling-statistics plots retained from the supplied plotting code.

def plot_sampling_results(data_obj, properties=('msd', 'teff', 'cv'), ax=None, multiplier=1.96, title=None):

    """

    Plots sampling results. If properties='all', creates a 2x3 grid with shared x-axes.

    """

    labels_map = {

        'teff': r'T$_{eff}$ [K]', 'msd': r'MSD [$\AA^2$]', 'cv': r'$C_V$ [eV/K]',

        'entropy': r'Entropy [eV/K]', 'free_energy': r'Free Energy [eV]',

        'internal_energy': r'Internal Energy [eV]', 'log_moment': r'Log Moment [a.u.]',

        'lambda_': r'$\lambda$ [a.u.]', 'zero_point_energy': r'Zero Point Energy [eV]'

    }



    # 1. Determine Property List and Grid Dimensions

    is_all = (properties == 'all')

    if is_all:

        plot_list = ['msd', 'cv', 'entropy', 'teff', 'zero_point_energy', 'free_energy']

        n_rows, n_cols = 2, 3

    else:

        plot_list = [p for p in properties if hasattr(data_obj, p) and p in labels_map]

        n_rows, n_cols = 1, len(plot_list)



    if not plot_list:

        print("No valid properties found to plot.")

        return



    # 2. Handle Axis Layout

    if ax is None:

        # Use sharex=True only for the 2x3 'all' layout

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows),

                                 sharex=is_all)

        plt.subplots_adjust(wspace=0.35, hspace=0.1 if is_all else 0.3)

    else:

        axes = ax

        fig = axes.flat[0].figure if isinstance(axes, np.ndarray) else axes[0].figure



    # Flatten axes for consistent indexing

    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]



    if title:

        if is_all:

            fig.suptitle(title, fontsize=24, y=0.92)

        else:

            fig.suptitle(title, fontsize=24, y=0.95)



    # 3. Plotting Loop

    for i, prop_name in enumerate(plot_list):

        current_ax = axes_flat[i]



        if not hasattr(data_obj, prop_name):

            current_ax.axis('off')

            continue



        data = getattr(data_obj, prop_name)

        ylabel = labels_map[prop_name]



        # Calculations

        nominal_value = data.iloc[0]

#        std_dev = data.std()

#        uncertainty = multiplier * std_dev



        # Confidence Interval & Nominal Line

#        current_ax.axhspan(ymin=nominal_value - uncertainty, ymax=nominal_value + uncertainty,

#                           color='lightgrey', label=f'95% CI (±{uncertainty:.4f})')

        current_ax.axhline(y=nominal_value, color='black', linewidth=2)



        # Scatter Plot

        sns.scatterplot(x=data.index[1:], y=data.values[1:],

                        color='darkred',

                        s=50,

                        edgecolor=None,

                        linewidth=0,

                        ax=current_ax,

                        zorder=2)



        # Formatting

        current_ax.set_title(ylabel, fontsize=20)



        # Only set X-label for the bottom row if sharing axes, or for all if not

        if not is_all or i >= n_cols:

            current_ax.set_xlabel('Samples', fontsize=20)



        current_ax.legend(fontsize=15, loc='best')



        # Scientific notation logic

        current_ax.yaxis.set_major_formatter(FormatStrFormatter('%.3e'))



    # Clean up empty subplots if the grid is larger than the data

    for j in range(i + 1, len(axes_flat)):

        axes_flat[j].axis('off')



    return fig, axes



def plot_sampling_histograms(data_obj, properties=('msd', 'teff', 'cv'), ax=None, multiplier=1.96, title=None):

    """

    Plots histograms of sampling results with dynamic statistical summaries.

    """

    labels_map = {

        'teff': r'T$_{eff}$ [K]', 'msd': r'MSD [$\AA^2$]', 'cv': r'$C_V$ [eV/K]',

        'entropy': r'Entropy [eV/K]', 'free_energy': r'Free Energy [eV]',

        'internal_energy': r'Internal Energy [eV]', 'log_moment': r'Log Moment [a.u.]',

        'lambda_': r'$\lambda$ [a.u.]', 'zero_point_energy': r'Zero Point Energy [eV]'

    }



    # 1. Determine Property List and Grid Dimensions

    is_all = (properties == 'all')

    if is_all:

        plot_list = ['msd', 'cv', 'entropy', 'teff', 'zero_point_energy', 'free_energy']

        n_rows, n_cols = 2, 3

    else:

        plot_list = [p for p in properties if hasattr(data_obj, p) and p in labels_map]

        n_rows, n_cols = (len(plot_list) + 2) // 3, min(len(plot_list), 3)



    if not plot_list:

        print("No valid properties found to plot.")

        return None, None



    # 2. Handle Axis Layout

    if ax is None:

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

        plt.subplots_adjust(wspace=0.3, hspace=0.4)

    else:

        axes = ax

        fig = axes.flat[0].figure if isinstance(axes, np.ndarray) else axes.figure



    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]



    if title:

        fig.suptitle(title, fontsize=24, y=0.98)



    # 3. Plotting Loop

    for i, prop_name in enumerate(plot_list):

        current_ax = axes_flat[i]



        if not hasattr(data_obj, prop_name):

            current_ax.axis('off')

            continue



        data = getattr(data_obj, prop_name)

        vals = data.values[1:]  # Exclude first row (usually the reference/nominal)

        nominal_value = data.iloc[0]

        xlabel = labels_map[prop_name]



        # --- Statistics Calculations ---

        mean_val = np.mean(vals)

        std_dev = np.std(vals)

        # Relative metrics (as percentages)

        std_dev_pct = (std_dev / abs(mean_val)) * 100 if mean_val != 0 else 0

        ci_95_pct = (multiplier * std_dev / abs(mean_val)) * 100 if mean_val != 0 else 0

        delta_k_pct = ((np.max(vals) - np.min(vals)) / abs(mean_val)) * 100 if mean_val != 0 else 0



        # --- Build Dynamic Stats Text ---

        # .4g handles the scale from 1e-5 to 1e3 automatically

        stats_text = (

            f"Median: {np.median(vals):.4g}\n"

            f"Std Dev: {std_dev_pct:.2f}%\n"

            f"95% CI: $\\pm${ci_95_pct:.2f}%\n"

            f"Skew: {stats.skew(vals):.2f}\n"

            f"Kurt: {stats.kurtosis(vals):.2f}"

        )



        # --- Plotting ---

        current_ax.axvline(x=nominal_value, color='black', linewidth=2, linestyle='--', label='Nominal')



        sns.histplot(data=vals, bins='auto', kde=True, color='darkred', ax=current_ax,

                     edgecolor='white', zorder=2)



        # --- Formatting & Text Box ---

        current_ax.set_title(xlabel, fontsize=20)

        current_ax.set_ylabel('Frequency', fontsize=16)



        # Adding the stats box (anchored top-right)

        props = dict(boxstyle='round', facecolor='white', edgecolor='lightgrey')

        current_ax.text(0.95, 0.95, stats_text, transform=current_ax.transAxes,

                        fontsize=13, verticalalignment='top', horizontalalignment='right',

                        bbox=props, family='monospace')



        # Only plot the "Last" legend (Nominal) using the slice method you liked

        handles, labels = current_ax.get_legend_handles_labels()

        if handles:

            current_ax.legend(handles[-1:], labels[-1:], fontsize=10, loc='upper left')



        # Scientific notation for X-axis

        current_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))

        current_ax.tick_params(axis='x', rotation=90)



    # Clean up empty subplots

    for j in range(i + 1, len(axes_flat)):

        axes_flat[j].axis('off')



    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, axes



def plot_sampling_combined_view(data_obj, properties=('teff', 'cv'), title=None):

    """

    Creates a 2-row layout using the user's existing plotting functions.

    Row 1: Scatter/Trace (via plot_sampling_results)

    Row 2: Histogram (via plot_sampling_histograms)

    """

    # 1. Determine valid properties to plot

    labels_map = ['cv', 'teff', 'zero_point_energy', 'free_energy', 'internal_energy']



    if properties == 'all':

        plot_list = [p for p in labels_map if hasattr(data_obj, p)]

    else:

        plot_list = [p for p in properties if hasattr(data_obj, p)]



    if not plot_list:

        print("No valid properties found to plot.")

        return



    n_cols = len(plot_list)



    # 2. Create the grid: 2 rows (Scatter top, Histogram bottom)

    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 12))



    # Ensure axes is 2D even if only one property is requested

    if n_cols == 1:

        axes = axes.reshape(2, 1)



    # 3. Use your functions by passing the specific row axes

    # Row 0 (Top): Scatter plots

    plot_sampling_results(data_obj, properties=plot_list, ax=axes[0, :])



    # Row 1 (Bottom): Histograms

    plot_sampling_histograms(data_obj, properties=plot_list, ax=axes[1, :])



    # 4. Final Formatting

    if title:

        fig.suptitle(title, fontsize=26, y=1.02)



    plt.tight_layout()

    return fig, axes

def plot_sab_beta_slice(
    ax,
    analyzer,
    alpha_selected,
    *,
    ref_label="NCrystal reference",
    label_fontsize=14,
    beta_threshold=1e-5,
):
    """Plot S(alpha, beta) over the active beta range at a selected alpha.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the scattering-law profiles are plotted.
    analyzer : EnsembleAnalyzer
        Analyzer containing the reference and stochastic results.
    alpha_selected : float
        Selected dimensionless momentum-transfer coordinate.
    ref_label : str, default='NCrystal reference'
        Legend label for the reference curve.
    label_fontsize : int, default=14
        Font size for the axis labels.
    beta_threshold : float, default=1e-5
        Relative intensity threshold used to determine the displayed
        beta range. The threshold is applied to the maximum intensity
        across the reference and all stochastic samples.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the reference and stochastic beta profiles.

    Notes
    -----
    The scattering-law tables are expected to have alpha values as
    their index and beta values as their columns.

    Profiles are linearly interpolated along alpha when the selected
    coordinate does not coincide with an existing grid point.

    Each profile retains its native beta grid. The relative threshold
    affects only the displayed beta limits; scattering-law values
    are never truncated or modified.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if not np.isfinite(alpha_selected):
        raise ValueError("alpha_selected must be finite.")

    if not np.isfinite(beta_threshold) or not 0.0 <= beta_threshold < 1.0:
        raise ValueError("beta_threshold must be between 0 and 1.")

    def profile_at_alpha(sab, selected):
        """Interpolate an alpha profile while preserving the beta grid."""
        if not isinstance(sab, pd.DataFrame):
            raise TypeError("S(alpha, beta) must be a pandas DataFrame.")

        alpha = sab.index.to_numpy(dtype=float)
        beta = sab.columns.to_numpy(dtype=float)
        values = sab.to_numpy(dtype=float)

        if (
            alpha.size < 2
            or beta.size < 2
            or not np.isfinite(alpha).all()
            or not np.isfinite(beta).all()
            or np.any(np.diff(alpha) <= 0.0)
            or np.any(np.diff(beta) <= 0.0)
        ):
            raise ValueError(
                "S(alpha, beta) must have strictly increasing, "
                "finite alpha and beta coordinates."
            )

        if not alpha[0] <= selected <= alpha[-1]:
            raise ValueError(
                f"Selected alpha={selected:g} is outside the available "
                f"range [{alpha[0]:g}, {alpha[-1]:g}]."
            )

        upper = int(np.searchsorted(alpha, selected))

        if upper == alpha.size:
            profile = values[-1]
        elif alpha[upper] == selected:
            profile = values[upper]
        elif upper == 0:
            profile = values[0]
        else:
            lower = upper - 1

            weight = (
                (selected - alpha[lower])
                / (alpha[upper] - alpha[lower])
            )

            profile = (
                (1.0 - weight) * values[lower]
                + weight * values[upper]
            )

        return beta, profile

    reference_result = analyzer.reference_result

    if reference_result is None:
        raise ValueError("Reference result is unavailable.")

    reference_sab = reference_result.diff.sab

    if reference_sab is None:
        raise ValueError("Reference S(alpha, beta) is unavailable.")

    # Collect profiles without modifying the underlying results.
    beta_ref, profile_ref = profile_at_alpha(
        reference_sab,
        alpha_selected,
    )

    sample_profiles = []

    for sample_index, result in enumerate(analyzer.raw_results):
        sab = result.diff.sab

        if sab is None:
            raise ValueError(
                f"S(alpha, beta) is unavailable for sample "
                f"{sample_index + 1}."
            )

        beta, profile = profile_at_alpha(sab, alpha_selected)

        sample_profiles.append((sample_index, beta, profile))

    # Draw all stochastic samples using the same color convention
    # as plot_PDOS.
    colors = plt.cm.tab10.colors

    for sample_index, beta, profile in sample_profiles:
        ax.plot(
            beta,
            profile,
            color=colors[sample_index % len(colors)],
            linewidth=1.0,
            alpha=0.5,
            zorder=2,
        )

    # Draw the reference above the stochastic curves.
    ax.plot(
        beta_ref,
        profile_ref,
        color="black",
        linewidth=2.0,
        label=ref_label,
        zorder=3,
    )

    # Determine the maximum intensity across all plotted profiles.
    profiles = [(beta_ref, profile_ref)] + [
        (beta, profile)
        for _, beta, profile in sample_profiles
    ]

    max_intensity = max(
        (
            float(np.max(values[np.isfinite(values)]))
            for _, values in profiles
            if np.any(np.isfinite(values))
        ),
        default=0.0,
    )

    # Restrict the beta-axis to regions exceeding the relative threshold.
    if max_intensity > 0.0:
        cutoff = beta_threshold * max_intensity
        active_beta = []

        for beta, values in profiles:
            mask = (
                np.isfinite(beta)
                & np.isfinite(values)
                & (values > cutoff)
            )

            if np.any(mask):
                active_beta.append(beta[mask])

        if active_beta:
            active_beta = np.concatenate(active_beta)

            beta_min = float(active_beta.min())
            beta_max = float(active_beta.max())

            margin = max(
                0.03 * (beta_max - beta_min),
                0.01,
            )

            ax.set_xlim(
                beta_min - margin,
                beta_max + margin,
            )

    # Match the presentation style of plot_PDOS.
    ax.set_title(
        rf"$S(\alpha={alpha_selected:.4g}, \beta)$",
        fontsize=20,
    )

    ax.set_xlabel(
        r"Energy transfer $\beta$",
        fontsize=label_fontsize,
    )

    ax.set_ylabel(
        r"$S(\alpha,\beta)$",
        fontsize=label_fontsize,
    )

    ax.set_ylim(bottom=0.0)
    ax.tick_params(axis="both", labelsize=label_fontsize - 2)
    ax.legend(loc="best", fontsize=11)

    return ax

def _extract_dsf_profile(table, theta_selected):
    """Extract a DSF outgoing-energy profile at a scattering angle.

    Parameters
    ----------
    table : pandas.DataFrame
        Dynamic structure factor indexed by mu, with outgoing
        neutron energies in eV as columns.
    theta_selected : float
        Scattering angle in degrees.

    Returns
    -------
    tuple of np.ndarray
        Outgoing energies in eV and corresponding DSF values.
    """
    mu = table.index.to_numpy(dtype=float)
    eout = table.columns.to_numpy(dtype=float)
    values = table.to_numpy(dtype=float)

    if (
        mu.size < 2
        or eout.size < 2
        or not np.isfinite(mu).all()
        or not np.isfinite(eout).all()
    ):
        raise ValueError("Invalid DSF coordinate grids.")

    mu_order = np.argsort(mu)
    mu = mu[mu_order]
    values = values[mu_order, :]

    if np.any(np.diff(mu) <= 0.0):
        raise ValueError("DSF angular coordinates must be unique.")

    selected_mu = float(np.cos(np.deg2rad(theta_selected)))

    if not 0.0 <= theta_selected <= 180.0:
        raise ValueError("Scattering angle must be between 0 and 180 degrees.")

    if np.isclose(selected_mu, mu[0], atol=1e-12):
        selected_mu = mu[0]
    elif np.isclose(selected_mu, mu[-1], atol=1e-12):
        selected_mu = mu[-1]

    if not mu[0] <= selected_mu <= mu[-1]:
        raise ValueError("Selected angle is outside the DSF angular grid.")

    upper = int(np.searchsorted(mu, selected_mu))

    if upper == mu.size:
        profile = values[-1, :]

    elif mu[upper] == selected_mu:
        profile = values[upper, :]

    elif upper == 0:
        profile = values[0, :]

    else:
        lower = upper - 1
        weight = (
            (selected_mu - mu[lower])
            / (mu[upper] - mu[lower])
        )
        profile = (
            (1.0 - weight) * values[lower, :]
            + weight * values[upper, :]
        )

    # Sort the outgoing-energy axis without changing its values.
    order = np.argsort(eout, kind="stable")

    return eout[order], profile[order]


def plot_dsf_energy_slice(
    ax,
    analyzer,
    incident_energy,
    theta_selected,
    *,
    ref_label="NCrystal reference",
    energy_threshold=1e-5,
    label_fontsize=14,
):
    """Plot dynamic structure factor profiles at fixed energy and angle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the dynamic structure factor is plotted.
    analyzer : EnsembleAnalyzer
        Analyzer containing the reference and stochastic results.
    incident_energy : float
        Incident neutron energy in eV.
    theta_selected : float
        Scattering angle in degrees.
    ref_label : str, default='NCrystal reference'
        Legend label for the reference curve.
    energy_threshold : float, default=1e-5
        Relative intensity threshold used to restrict the displayed
        outgoing-energy range.
    label_fontsize : int, default=14
        Font size for the axis labels.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the reference and stochastic DSF profiles.

    Notes
    -----
    The scattering profiles retain their native outgoing-energy grids.

    The relative intensity threshold determines only the displayed
    outgoing-energy limits. It does not modify the DSF values.

    A vertical dashed line indicates the incident neutron energy.
    This marker is always included in the displayed energy range.
    """
    if not np.isfinite(incident_energy) or incident_energy < 0.0:
        raise ValueError(
            "incident_energy must be finite and nonnegative."
        )

    if not np.isfinite(theta_selected) or not 0.0 <= theta_selected <= 180.0:
        raise ValueError(
            "theta_selected must be between 0 and 180 degrees."
        )

    if (
        not np.isfinite(energy_threshold)
        or not 0.0 <= energy_threshold < 1.0
    ):
        raise ValueError(
            "energy_threshold must be finite and in [0, 1)."
        )

    colors = plt.cm.tab10.colors
    profiles = []

    # ---------------------------------------------------------
    # Plot stochastic samples.
    # ---------------------------------------------------------

    for sample_index, result in enumerate(analyzer.raw_results):
        tables = result.diff.dynamic_structure_factor

        if tables is None or incident_energy not in tables:
            raise ValueError(
                f"Sample {sample_index + 1} has no dynamic "
                f"structure factor at Ein={incident_energy:g} eV."
            )

        eout, profile = _extract_dsf_profile(
            tables[incident_energy],
            theta_selected,
        )

        profiles.append((eout, profile))

        ax.plot(
            1000.0 * eout,
            profile,
            color=colors[sample_index % len(colors)],
            linewidth=1.0,
            alpha=0.45,
            zorder=2,
        )

    # ---------------------------------------------------------
    # Plot reference.
    # ---------------------------------------------------------

    reference_result = analyzer.reference_result

    if reference_result is None:
        raise ValueError(
            "Reference dynamic structure factor is unavailable."
        )

    reference_tables = (
        reference_result.diff.dynamic_structure_factor
    )

    if reference_tables is None or incident_energy not in reference_tables:
        raise ValueError(
            f"Reference has no dynamic structure factor "
            f"at Ein={incident_energy:g} eV."
        )

    eout_ref, profile_ref = _extract_dsf_profile(
        reference_tables[incident_energy],
        theta_selected,
    )

    profiles.append((eout_ref, profile_ref))

    ax.plot(
        1000.0 * eout_ref,
        profile_ref,
        color="black",
        linewidth=2.0,
        label=ref_label,
        zorder=3,
    )

    # ---------------------------------------------------------
    # Determine the active outgoing-energy interval.
    # ---------------------------------------------------------

    finite_profiles = [
        values[np.isfinite(values)]
        for _, values in profiles
        if np.any(np.isfinite(values))
    ]

    max_intensity = max(
        (
            float(np.max(values))
            for values in finite_profiles
        ),
        default=0.0,
    )

    incident_energy_mev = 1000.0 * incident_energy

    if max_intensity > 0.0:
        cutoff = energy_threshold * max_intensity
        active_energies = []

        for eout, values in profiles:
            mask = (
                np.isfinite(eout)
                & np.isfinite(values)
                & (values > cutoff)
            )

            if np.any(mask):
                active_energies.append(
                    1000.0 * eout[mask]
                )

        if active_energies:
            active_energies = np.concatenate(active_energies)

            energy_min = float(active_energies.min())
            energy_max = float(active_energies.max())

            margin = max(
                0.03 * (energy_max - energy_min),
                0.01,
            )

            # Always include the incident-energy marker.
            lower_limit = max(
                0.0,
                min(
                    energy_min - margin,
                    incident_energy_mev,
                ),
            )

            upper_limit = max(
                energy_max + margin,
                incident_energy_mev,
            )

            # Avoid identical limits for degenerate profiles.
            if upper_limit <= lower_limit:
                upper_limit = lower_limit + 0.1

            ax.set_xlim(
                lower_limit,
                upper_limit,
            )

    # ---------------------------------------------------------
    # Mark the incident neutron energy.
    # ---------------------------------------------------------

    ax.axvline(
        x=incident_energy_mev,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=r"$E_{\mathrm{in}}$",
        zorder=4,
    )

    # ---------------------------------------------------------
    # Styling consistent with plot_PDOS.
    # ---------------------------------------------------------

    ax.set_title(
        rf"Dynamic structure factor at "
        rf"$E_{{\mathrm{{in}}}}={incident_energy:g}$ eV, "
        rf"$\theta={theta_selected:g}^\circ$",
        fontsize=20,
    )

    ax.set_xlabel(
        "Outgoing neutron energy [meV]",
        fontsize=label_fontsize,
    )

    ax.set_ylabel(
        "Dynamic structure factor",
        fontsize=label_fontsize,
    )

    ax.set_ylim(bottom=0.0)

    ax.tick_params(
        axis="both",
        labelsize=label_fontsize - 2,
    )

    ax.legend(
        loc="best",
        fontsize=11,
    )

    return ax

def _distribution_row(frame, incident_energy):
    """Extract a distribution at a selected incident neutron energy.

    Parameters
    ----------
    frame : pandas.DataFrame or None
        Distribution table indexed by incident neutron energy in eV.
        Columns contain the distribution coordinate.
    incident_energy : float
        Requested incident neutron energy in eV.

    Returns
    -------
    pandas.Series or None
        Distribution indexed by its native coordinate, or None when
        the requested incident energy is unavailable.

    Raises
    ------
    TypeError
        If the input is not a DataFrame.
    ValueError
        If coordinates are invalid or duplicated.
    """
    if frame is None:
        return None

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Distributions must be pandas.DataFrame objects.")

    energies = frame.index.to_numpy(dtype=float)

    matches = np.flatnonzero(
        np.isclose(
            energies,
            incident_energy,
            rtol=1e-8,
            atol=1e-12,
        )
    )

    if len(matches) != 1:
        return None

    row = frame.iloc[int(matches[0])]

    x = row.index.to_numpy(dtype=float)
    y = row.to_numpy(dtype=float)

    if not np.isfinite(x).all():
        raise ValueError("Distribution coordinates must be finite.")

    valid = np.isfinite(y)

    series = pd.Series(
        y[valid],
        index=x[valid],
        dtype=float,
    ).sort_index()

    if series.index.has_duplicates:
        raise ValueError("Distribution coordinates must be unique.")

    return series if len(series) >= 2 else None


def _collect_distribution(
    analyzer,
    distribution_type,
    incident_energy,
):
    """Collect reference and sampled distributions on a common grid.

    Samples are interpolated onto the reference distribution coordinates.
    Values outside each sample's coordinate range remain NaN.

    Parameters
    ----------
    analyzer : EnsembleAnalyzer
        Analyzer containing reference and sampled physical results.
    distribution_type : {'cosine_distribution', 'energy_distribution'}
        Distribution to extract.
    incident_energy : float
        Incident neutron energy in eV.

    Returns
    -------
    reference : pandas.Series
        Reference distribution.
    samples : pandas.DataFrame
        Sample distributions aligned with the reference coordinates.
    sample_indices : list of int
        Original ensemble indices of the retained samples.
    """
    supported = (
        "cosine_distribution",
        "energy_distribution",
    )

    if distribution_type not in supported:
        raise ValueError(
            f"Unsupported distribution type: {distribution_type!r}"
        )

    if analyzer.reference_result is None:
        raise ValueError(
            "Reference physical results are unavailable."
        )

    reference_frame = getattr(
        analyzer.reference_result.integ,
        distribution_type,
        None,
    )

    reference = _distribution_row(
        reference_frame,
        incident_energy,
    )

    if reference is None:
        raise ValueError(
            f"Reference {distribution_type} is unavailable "
            f"at {incident_energy:g} eV."
        )

    reference_x = reference.index.to_numpy(dtype=float)

    sample_curves = {}
    sample_indices = []

    for result in analyzer.raw_results:
        frame = getattr(
            result.integ,
            distribution_type,
            None,
        )

        sample = _distribution_row(
            frame,
            incident_energy,
        )

        if sample is None:
            continue

        sample_x = sample.index.to_numpy(dtype=float)
        sample_y = sample.to_numpy(dtype=float)

        aligned = np.interp(
            reference_x,
            sample_x,
            sample_y,
            left=np.nan,
            right=np.nan,
        )

        if not np.isfinite(aligned).any():
            continue

        sample_index = int(result.sample_index)

        label = f"Sample {sample_index + 1}"

        if label in sample_curves:
            raise ValueError(
                f"Duplicate distribution result for {label}."
            )

        sample_curves[label] = aligned
        sample_indices.append(sample_index)

    if not sample_curves:
        raise ValueError(
            f"No sampled {distribution_type} is available "
            f"at {incident_energy:g} eV."
        )

    samples = pd.DataFrame(
        sample_curves,
        index=reference.index,
    )

    return reference, samples, sample_indices


def _distribution_x_values(
    coordinates,
    distribution_type,
    incident_energy,
    temperature,
):
    """Convert native distribution coordinates into display coordinates.

    Parameters
    ----------
    coordinates : array_like
        Native coordinates: scattering cosine or beta.
    distribution_type : {'cosine_distribution', 'energy_distribution'}
        Distribution type.
    incident_energy : float
        Incident neutron energy in eV.
    temperature : float or None
        Material temperature in K. Required for energy distributions.

    Returns
    -------
    numpy.ndarray
        Scattering cosine or outgoing neutron energy in meV.

    Notes
    -----
    The energy-distribution conversion assumes

        beta = (E_out - E_in) / (k_B * T)

    and hence

        E_out = E_in + beta * k_B * T.

    The transformation changes only the horizontal display coordinate.
    Distribution values are not renormalized or converted between
    probability-density measures.
    """
    coordinates = np.asarray(
        coordinates,
        dtype=float,
    )

    if distribution_type == "cosine_distribution":
        return coordinates

    if distribution_type != "energy_distribution":
        raise ValueError(
            f"Unsupported distribution type: {distribution_type!r}"
        )

    if temperature is None:
        raise ValueError(
            "The analysis temperature is required to convert "
            "beta into outgoing neutron energy."
        )

    temperature = float(temperature)
    incident_energy = float(incident_energy)

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Temperature must be finite and positive."
        )

    if not np.isfinite(incident_energy) or incident_energy < 0:
        raise ValueError(
            "Incident neutron energy must be finite and nonnegative."
        )

    k_b_ev_per_k = 8.617333262145e-5

    outgoing_energy_ev = (
        incident_energy
        + coordinates * k_b_ev_per_k * temperature
    )

    return 1000.0 * outgoing_energy_ev


def plot_distribution(
    ax,
    reference,
    samples,
    distribution_type,
    *,
    x_values=None,
    sample_indices=None,
    label_fontsize=14,
):
    """Plot a reference distribution and its stochastic samples.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Destination axes.
    reference : pandas.Series
        Reference distribution.
    samples : pandas.DataFrame
        Aligned sample distributions, one per column.
    distribution_type : {'cosine_distribution', 'energy_distribution'}
        Distribution type.
    x_values : array_like, optional
        Display coordinates. Defaults to the native reference coordinates.
    sample_indices : sequence of int, optional
        Original sample indices used for consistent tab10 colors.
    label_fontsize : int, default=14
        Font size for both axis labels.

    Returns
    -------
    matplotlib.axes.Axes
        Updated axes.
    """
    labels = {
        "cosine_distribution": (
            r"Scattering cosine $\mu$",
            "Cosine distribution",
        ),
        "energy_distribution": (
            "Outgoing neutron energy [meV]",
            "Energy distribution",
        ),
    }

    if distribution_type not in labels:
        raise ValueError(
            f"Unsupported distribution type: {distribution_type!r}"
        )

    xlabel, ylabel = labels[distribution_type]

    if x_values is None:
        if distribution_type == "energy_distribution":
            raise ValueError(
                "Outgoing-energy coordinates must be supplied "
                "for energy distributions."
            )
        x_values = reference.index.to_numpy(dtype=float)

    x_values = np.asarray(x_values, dtype=float)

    if len(x_values) != len(reference):
        raise ValueError(
            "Display coordinates must match the reference length."
        )

    if sample_indices is None:
        sample_indices = list(range(samples.shape[1]))

    if len(sample_indices) != samples.shape[1]:
        raise ValueError(
            "sample_indices must match the number of sample columns."
        )

    colors = plt.cm.tab10.colors

    for sample_index, (label, sample) in zip(
        sample_indices,
        samples.items(),
    ):
        ax.plot(
            x_values,
            sample.to_numpy(dtype=float),
            color=colors[int(sample_index) % len(colors)],
            linewidth=1.25,
            alpha=0.65,
            label=label,
        )

    ax.plot(
        x_values,
        reference.to_numpy(dtype=float),
        color="black",
        linewidth=2.2,
        label="Reference",
        zorder=3,
    )

    ax.set_xlabel("")

    ax.set_ylabel(
        ylabel,
        fontsize=label_fontsize,
    )

    ax.set_title("")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    return ax


def plot_distribution_dashboard(
    analyzer,
    distribution_type,
    incident_energy,
    *,
    ref_label="Reference",
    temperature=None,
    label_fontsize=14,
):
    """Plot PDOS, distribution, and relative difference in three panels.

    Parameters
    ----------
    analyzer : EnsembleAnalyzer
        Analyzer with reference and sampled physical results.
    distribution_type : {'cosine_distribution', 'energy_distribution'}
        Distribution to display.
    incident_energy : float
        Selected incident neutron energy in eV.
    ref_label : str, default='Reference'
        Reference label for the PDOS panel.
    temperature : float, optional
        Analysis temperature in K. If omitted, the function attempts
        to read ``analyzer.temperature``. Required when plotting
        an energy distribution stored in beta coordinates.
    label_fontsize : int, default=14
        Common font size for x-axis and y-axis labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Dashboard figure.
    axes : dict
        Axes named 'pdos', 'distribution', and 'error'.

    Notes
    -----
    Relative differences are calculated as

        100 * (sample - reference) / reference.

    Values are undefined where the reference is zero or sample coverage
    is missing; these positions remain NaN.

    The energy-distribution x-axis assumes beta coordinates and displays
    outgoing neutron energy in meV. Distribution ordinates retain their
    original units.
    """
    distribution_names = {
        "cosine_distribution": "Cosine distribution",
        "energy_distribution": "Energy distribution",
    }

    if distribution_type not in distribution_names:
        raise ValueError(
            f"Unsupported distribution type: {distribution_type!r}"
        )

    reference, samples, sample_indices = _collect_distribution(
        analyzer,
        distribution_type,
        incident_energy,
    )

    if temperature is None:
        temperature = getattr(
            analyzer,
            "temperature",
            None,
        )

    x_values = _distribution_x_values(
        reference.index,
        distribution_type,
        incident_energy,
        temperature,
    )

    figure_title = (
        f"{distribution_names[distribution_type]} "
        f"at {incident_energy:g} eV"
    )

    fig, (ax_pdos, ax_dist, ax_error) = plt.subplots(
        3,
        1,
        figsize=(12, 13),
        sharex=False,
        gridspec_kw={"height_ratios": [1.2, 1.3, 1.0]},
        layout="constrained",
    )

    # Only the distribution and error panels share the x-axis.
    ax_dist.sharex(ax_error)

    # ---------------------------------------------------------------
    # Top panel: PDOS.
    # ---------------------------------------------------------------
    spectra = [
        analyzer.ensemble[index]
        for index in sample_indices
    ]

    plot_PDOS(
        ax_pdos,
        spectra,
        analyzer.ensemble.reference,
        title=figure_title,
        ref_label=ref_label,
    )

    ax_pdos.xaxis.label.set_size(label_fontsize)
    ax_pdos.yaxis.label.set_size(label_fontsize)

    # ---------------------------------------------------------------
    # Middle panel: distribution at the selected incident energy.
    # ---------------------------------------------------------------
    plot_distribution(
        ax_dist,
        reference,
        samples,
        distribution_type,
        x_values=x_values,
        sample_indices=sample_indices,
        label_fontsize=label_fontsize,
    )

    ax_dist.tick_params(axis="x", labelbottom=False)
    ax_dist.set_xlabel("")

    # ---------------------------------------------------------------
    # Bottom panel: relative difference from the reference.
    # ---------------------------------------------------------------
    reference_values = reference.to_numpy(dtype=float)
    sample_values = samples.to_numpy(dtype=float)

    reference_column = reference_values[:, None]

    errors = np.full(
        sample_values.shape,
        np.nan,
        dtype=float,
    )

    np.divide(
        100.0 * (sample_values - reference_column),
        reference_column,
        out=errors,
        where=(
            np.isfinite(sample_values)
            & np.isfinite(reference_column)
            & (reference_column != 0.0)
        ),
    )

    colors = plt.cm.tab10.colors

    for column_number, sample_index in enumerate(sample_indices):
        ax_error.plot(
            x_values,
            errors[:, column_number],
            color=colors[int(sample_index) % len(colors)],
            linewidth=1.25,
            alpha=0.65,
            label=samples.columns[column_number],
        )

    ax_error.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.3,
    )

    # Incident neutron energy marker (eV -> meV).
    if distribution_type == "energy_distribution":
        incident_energy_mev = 1000.0 * incident_energy

        for ax in (ax_dist, ax_error):
            ax.axvline(
                x=incident_energy_mev,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label="_nolegend_",
            )

    xlabels = {
        "cosine_distribution": r"Scattering cosine $\mu$",
        "energy_distribution": "Outgoing neutron energy [meV]",
    }

    ax_error.set_xlabel(
        xlabels[distribution_type],
        fontsize=label_fontsize,
    )

    ax_error.set_ylabel(
        "Relative difference",
        fontsize=label_fontsize,
    )

    ax_error.set_title("")
    ax_error.grid(alpha=0.2)

    axes = {
        "pdos": ax_pdos,
        "distribution": ax_dist,
        "error": ax_error,
    }

    return fig, axes


def _native_series(series, observable):
    """
    Return a native 1D curve for visualization.

    Preserve repeated coordinates and original ordering.
    Do not merge, average, interpolate, or otherwise alter the data.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"{observable} must be a pandas Series."
        )

    x = series.index.to_numpy(dtype=float)
    y = series.to_numpy(dtype=float)

    if not np.isfinite(x).all():
        raise ValueError(
            f"{observable} contains non-finite coordinates."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"{observable} contains non-finite values."
        )

    return x, y


def _kernel_display_window(curves, relative_threshold=1e-2, padding=0.05):
    """
    Determine a shared x-axis window for scattering-kernel curves.

    Only values exceeding a relative threshold contribute to the
    display limits. The original curves are not modified.

    Parameters
    ----------
    curves : list[tuple[np.ndarray, np.ndarray]]
        Native (outgoing_energy, kernel_value) arrays.
    relative_threshold : float
        Fraction of the maximum absolute kernel value used to
        identify the relevant region.
    padding : float
        Fractional padding added to the resulting x-axis span.
    """
    if not 0 <= relative_threshold < 1:
        raise ValueError("relative_threshold must be in [0, 1).")

    max_value = max(
        float(np.max(np.abs(y))) for _, y in curves if len(y)
    )

    significant_energies = []

    for x, y in curves:
        mask = np.abs(y) >= relative_threshold * max_value
        if np.any(mask):
            significant_energies.append(x[mask])

    if not significant_energies:
        return None

    x_relevant = np.concatenate(significant_energies)
    xmin = float(np.min(x_relevant))
    xmax = float(np.max(x_relevant))

    span = xmax - xmin

    if span == 0:
        span = max(abs(xmin) * 0.1, 1e-8)

    return (
        xmin - padding * span,
        xmax + padding * span,
    )


def plot_scattering_kernel(
    ax,
    analyzer,
    incident_energy,
    *,
    ref_label="Reference",
    relative_threshold=1e-4,
):
    """
    Plot scattering kernels at a selected incident energy.

    A common display window is determined from significant
    contributions across the reference and stochastic samples.
    Native kernel values and coordinates remain unchanged.
    """
    if analyzer.reference_result is None:
        raise ValueError("Reference scattering kernel is unavailable.")

    reference_tables = analyzer.reference_result.integ.scattering_kernel

    if not reference_tables or incident_energy not in reference_tables:
        raise ValueError(
            f"No reference kernel at Ein={incident_energy:g} eV."
        )

    curves = []

    for result in analyzer.raw_results:
        tables = result.integ.scattering_kernel

        if not tables or incident_energy not in tables:
            continue

        eout, values = _native_series(
            tables[incident_energy],
            "scattering kernel",
        )

        index = int(result.sample_index)

        curves.append((eout, values))

        ax.plot(
            1000.0 * eout,
            values,
            color=plt.cm.tab10.colors[index % 10],
            alpha=0.65,
            linewidth=1.3,
            label=f"Sample {index + 1}",
        )

    if not curves:
        raise ValueError(
            f"No stochastic kernels at Ein={incident_energy:g} eV."
        )

    eout_ref, values_ref = _native_series(
        reference_tables[incident_energy],
        "reference kernel",
    )

    curves.append((eout_ref, values_ref))

    ax.plot(
        1000.0 * eout_ref,
        values_ref,
        color="black",
        linewidth=2.2,
        label=ref_label,
        zorder=3,
    )

    ax.axvline(
        1000.0 * incident_energy,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=r"$E_{\mathrm{in}}$",
    )

    # Zoom based on significant contributions.
    window = _kernel_display_window(
        curves,
        relative_threshold=relative_threshold,
    )

    if window is not None:
        ax.set_xlim(
            1000.0 * window[0],
            1000.0 * window[1],
        )

    ax.set_title(
        rf"Scattering kernel at "
        rf"$E_{{\mathrm{{in}}}}={incident_energy:g}$ eV"
    )
    ax.set_xlabel("Outgoing neutron energy [meV]")
    ax.set_ylabel("Scattering kernel [native units]")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    return ax



_RATIO_FIELDS = {
    "upscattering_ratio": "Upscattering fraction",
    "downscattering_ratio": "Downscattering fraction",
    "elastic_ratio": "Elastic residual fraction (1 − up − down)",
}


def plot_scattering_ratio(ax, analyzer, ratio_type, *, ref_label="Reference"):
    """Display computed incident-energy ratios without redefining elastic physics."""
    if ratio_type not in _RATIO_FIELDS:
        raise ValueError(f"Unknown ratio channel: {ratio_type!r}")
    if analyzer.reference_result is None:
        raise ValueError("Reference scattering ratios are unavailable.")
    reference = getattr(analyzer.reference_result.integ, ratio_type)
    if reference is None:
        raise ValueError(f"Reference {ratio_type} is unavailable.")
    plotted = 0
    for result in analyzer.raw_results:
        series = getattr(result.integ, ratio_type)
        if series is None:
            continue
        energy, ratio = _native_series(series, ratio_type)
        index = int(result.sample_index)
        ax.plot(energy, ratio, marker="o", color=plt.cm.tab10.colors[index % 10],
                label=f"Sample {index + 1}", linewidth=1.3)
        plotted += 1
    if not plotted:
        raise ValueError(f"No stochastic {ratio_type} values are available.")
    energy, ratio = _native_series(reference, f"reference {ratio_type}")
    ax.plot(energy, ratio, marker="o", color="black", linewidth=2.2,
            label=ref_label, zorder=3)
    ax.set_title(_RATIO_FIELDS[ratio_type])
    ax.set_xlabel("Incident neutron energy [eV]")
    ax.set_ylabel("Fraction")
    ax.set_xscale("log")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    return ax


def plot_analysis_pair(
    analyzer,
    kind,
    *,
    ref_label="Reference",
    incident_energy=None,
    distribution_type=None,
    xs_type=None,
    ratio_type=None,
    alpha_selected=None,
    theta_selected=None,
    temperature=None,
    beta_threshold=1e-4,
    energy_threshold=1e-5,
):
    """Render the common two-row analysis figure.

    Row 1 always contains the unchanged reference and every stochastic PDOS.
    Row 2 contains the selected physical result, controlled by notebook widgets.
    Interactive selections change only the lower observable, not the PDOS.
    Existing domain-specific plotters handle physics data without notebook math.
    """
    if analyzer.ensemble.reference is None:
        raise ValueError("A separate reference PDOS is required.")
    fig, (ax_pdos, ax_result) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.3]},
        layout="constrained",
    )
    try:
        plot_PDOS(
            ax_pdos, [analyzer.ensemble[i] for i in range(len(analyzer.ensemble))],
            analyzer.ensemble.reference,
            title="Reference PDOS and stochastic spectra", ref_label=ref_label,
            title_fontsize=16,
        )
        if kind == "sab":
            plot_sab_beta_slice(
                ax_result, analyzer, alpha_selected,
                ref_label=ref_label, beta_threshold=beta_threshold,
            )
        elif kind == "dsf":
            plot_dsf_energy_slice(
                ax_result, analyzer, incident_energy, theta_selected,
                ref_label=ref_label, energy_threshold=energy_threshold,
            )
        elif kind == "distribution":
            reference, samples, indices = _collect_distribution(
                analyzer, distribution_type, incident_energy,
            )
            coordinates = _distribution_x_values(
                reference.index, distribution_type, incident_energy, temperature,
            )
            plot_distribution(
                ax_result, reference, samples, distribution_type,
                x_values=coordinates, sample_indices=indices,
            )
            ax_result.set_xlabel(
                r"Scattering cosine $\mu$" if distribution_type == "cosine_distribution"
                else "Outgoing neutron energy [meV]"
            )
            ax_result.set_title(
                f"{distribution_type.replace('_', ' ').title()} at Ein={incident_energy:g} eV"
            )
        elif kind == "cross_section":
            reference, samples, _ = _collect_xs(analyzer, xs_type)
            plot_XS(ax_result, samples, ref_label,
                    reference=reference, xs_type=xs_type)
            ax_result.set_title(_XS_LABELS[xs_type])
        elif kind == "scattering_ratio":
            plot_scattering_ratio(ax_result, analyzer, ratio_type,
                                  ref_label=ref_label)
        elif kind == "scattering_kernel":
            plot_scattering_kernel(ax_result, analyzer, incident_energy,
                                   ref_label=ref_label)
        else:
            raise ValueError(f"Unknown analysis kind: {kind!r}")
    except Exception:
        plt.close(fig)
        raise
    return fig, {"pdos": ax_pdos, "result": ax_result}

def plot_water_PDOS(
    axes,
    data,
    pdos_ref,
    *,
    oscillators=None,
    ref_label="Reference",
):
    """Plot the water spectrum in three panels with independent y-axes.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
        Three axes for the low-energy, bending, and stretching regions.
        Create them without sharing the y-axis.
    data : sequence or pandas.DataFrame
        Stochastic spectra. A DataFrame uses sample rows and energy columns.
    pdos_ref : PhononSpectra or pandas.Series
        Independent reference spectrum.
    oscillators : sequence of DiscreteOscillator, optional
        Pass ``material.dynamics.oscillators`` for NJOY LEAPR water.
        The mode panels then show the fixed oscillator weights. Leave this
        unset for NCrystal water, whose modes are already in the VDOS.
    ref_label : str, default="Reference"
        Label for the reference curve.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        The three updated axes.
    """
    axes = tuple(axes)
    if len(axes) != 3:
        raise ValueError("Water PDOS requires exactly three axes.")

    if isinstance(data, pd.DataFrame):
        data = (row for _, row in data.iterrows())

    curves = []
    for spectrum in [*data, pdos_ref]:
        energy, density = _spectrum_xy(spectrum)
        curves.append((energy * 1e3, density / 1e3))

    upper_mev = max(450.0, max(energy[-1] for energy, _ in curves))
    bands = (
        (0.0, 150.0, "Low-energy region"),
        (195.0, 215.0, "Bending region"),
        (405.0, 420.0, "Stretching region"),
    )
    modes = (
        None if oscillators is None
        else sorted(oscillators, key=lambda mode: mode.energy_ev)
    )
    if modes is not None and len(modes) != 2:
        raise ValueError("The LEAPR water plot requires two discrete oscillators.")

    colors = plt.cm.tab10.colors
    for panel, (ax, (lower, upper, title)) in enumerate(zip(axes, bands)):
        if modes is not None and panel > 0:
            mode = modes[panel - 1]
            energy_mev = mode.energy_ev * 1e3
            ax.stem(
                [energy_mev], [mode.weight],
                linefmt="k-", markerfmt="ko", basefmt=" ",
            )
            ax.annotate(
                f"{energy_mev:g} meV",
                xy=(energy_mev, mode.weight),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
            )
            ax.set(
                title=("Bending mode", "Stretching mode")[panel - 1],
                xlim=(min(lower, energy_mev - 10), max(upper, energy_mev + 10)),
                ylim=(0, 1.25 * mode.weight if mode.weight > 0 else 1),
                ylabel="Oscillator weight",
            )
        else:
            # Slice the plotted data so each panel autos-scales to its region.
            if modes is not None:
                upper = max(upper, max(energy[-1] for energy, _ in curves))
                title = "Low-energy continuum"

            for index, (energy, density) in enumerate(curves):
                visible = (energy >= lower) & (energy <= upper)
                is_reference = index == len(curves) - 1
                ax.plot(
                    energy[visible],
                    density[visible],
                    color="black" if is_reference else colors[index % len(colors)],
                    linewidth=2.0 if is_reference else 1.0,
                    label=ref_label if is_reference else None,
                    zorder=3 if is_reference else 2,
                )

            ax.set(
                title=title,
                xlim=(lower, upper),
                ylabel="Normalized continuum [1/meV]" if modes is not None
                else "Phonon spectrum [1/meV]",
            )
            ax.set_ylim(bottom=0)

        ax.set_xlabel("Energy [meV]")
        ax.grid(alpha=0.25)

    axes[0].legend(loc="best")
    return axes

def plot_water_master_dashboard(
    data_obj,
    ref_label="Reference",
    pdos_title="Water vibrational spectrum",
    main_title=None,
    plot_pdos=True,
    plot_xs=False,
    plot_error=False,
    savepath=None,
    *,
    xs_type="xs_total",
    oscillators=None,
    show_error_inset=False,
):
    """Plot a three-panel water spectrum and optional cross-section rows.

    Parameters
    ----------
    data_obj : EnsembleAnalyzer
        Analyzer holding the ensemble and, for cross-section plots, the
        computed reference and sample results.
    ref_label, pdos_title : str
        Reference label and title for the spectrum row.
    main_title : str, optional
        Title for the whole dashboard.
    plot_pdos, plot_xs, plot_error : bool
        Enable the spectrum row, cross sections, and relative differences.
        Relative differences require cross sections.
    savepath : str or pathlib.Path, optional
        Destination for the completed figure.
    xs_type : str, default="xs_total"
        Cross-section channel to display.
    oscillators : sequence of DiscreteOscillator, optional
        Pass ``material.dynamics.oscillators`` for NJOY LEAPR water.
        Leave unset for NCrystal water.
    show_error_inset : bool, default=False
        Add the shared relative-difference zoom inset.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, dict)
        Figure and axes. The ``pdos`` entry contains the three spectrum axes.
    """
    if plot_error and not plot_xs:
        raise ValueError("plot_error=True requires plot_xs=True.")
    if not (plot_pdos or plot_xs):
        raise ValueError("Enable at least one plot type.")

    if plot_xs:
        reference, samples, sample_indices = _collect_xs(data_obj, xs_type)

    if plot_pdos:
        ensemble = data_obj.ensemble
        if ensemble.reference is None:
            raise ValueError("The ensemble has no PDOS reference.")
        spectra = (
            [ensemble[index] for index in sample_indices]
            if plot_xs else list(ensemble)
        )

    rows = [
        name for name, enabled in (
            ("pdos", plot_pdos),
            ("xs", plot_xs),
            ("error", plot_error),
        )
        if enabled
    ]
    ratios = {"pdos": 1.2, "xs": 1.3, "error": 1.0}
    fig = plt.figure(figsize=(14, 3.8 * len(rows)), layout="constrained")
    grid = GridSpec(
        len(rows), 3, figure=fig,
        height_ratios=[ratios[name] for name in rows],
    )
    axes = {}

    for row, name in enumerate(rows):
        if name == "pdos":
            axes[name] = plot_water_PDOS(
                [fig.add_subplot(grid[row, col]) for col in range(3)],
                spectra,
                ensemble.reference,
                oscillators=oscillators,
                ref_label=ref_label,
            )
        elif name == "xs":
            axes[name] = fig.add_subplot(grid[row, :])
            plot_XS(
                axes[name], samples, ref_label,
                reference=reference, xs_type=xs_type,
            )
        else:
            axes[name] = fig.add_subplot(grid[row, :], sharex=axes["xs"])
            plot_relative_diff(
                axes[name],
                _relative_xs_error(reference, samples),
                show_inset=show_error_inset,
            )

    title = main_title or (pdos_title if plot_pdos else None)
    if title:
        fig.suptitle(title, fontsize=19)
    if savepath is not None:
        fig.savefig(Path(savepath), dpi=300, bbox_inches="tight")
    return fig, axes


import numpy as np
import pandas as pd

from core.math_engine.bspline_fit import BsplineAmplitudePerturber, BsplineFitter
from core.math_engine.diffeomorphic import (
    apply_spatial_warp as apply_diffeomorphic_warp,
    apply_spatial_warp_batch as apply_diffeomorphic_warp_batch,
)
from core.math_engine.fast_amplitude import FastAmplitudePerturber
from core.math_engine.segment_translation import (
    apply_spatial_warp as apply_segment_translation,
    apply_spatial_warp_batch as apply_segment_translation_batch,
)


def _spectrum():
    """Spectrum for the current workflow.

    Returns
    -------
    tuple
        Calculated arrays or records in the order described by the operation.
    """
    x_grid = np.linspace(0.0, 0.12, 121)
    y_density = (
        np.exp(-((x_grid - 0.035) ** 2) / 0.00008)
        + 0.8 * np.exp(-((x_grid - 0.085) ** 2) / 0.00012)
    )
    y_density[0] = 0.0
    return x_grid, y_density


def test_fast_amplitude_scalar_and_batch_paths_are_identical():
    """Test scalar/batch parity for the cached direct amplitude operator."""
    x_grid, y_density = _spectrum()
    boundaries = [(0.0, 0.06), (0.06, 0.12)]
    shifts = np.array([[0.1, -0.2], [-0.05, 0.15]])
    engine = FastAmplitudePerturber(x_grid=x_grid, y_density=y_density)

    batch = engine.apply_batch(boundaries=boundaries, shifts=shifts)

    for row, row_shifts in zip(batch, shifts):
        # Check the expected numerical agreement within the declared tolerance.
        np.testing.assert_allclose(row, engine.apply(boundaries, row_shifts), rtol=1e-12, atol=1e-12)

    zero = engine.apply_batch(boundaries, np.zeros_like(shifts))
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(zero, np.vstack([y_density, y_density]))


def test_bspline_amplitude_scalar_and_batch_paths_are_identical():
    """Test scalar/batch parity for the constrained B-spline amplitude operator."""
    x_grid, y_density = _spectrum()
    fitter = BsplineFitter(pd.Series(y_density, index=x_grid))
    fitter.fit(n_splines=24, degree=3, alpha=1e-6)
    engine = BsplineAmplitudePerturber(x_grid=x_grid, base_fitter=fitter)
    boundaries = [(0.0, 0.06), (0.06, 0.12)]
    shifts = np.array([[0.08, -0.05], [-0.04, 0.1]])

    batch = engine.apply_batch(boundaries=boundaries, shifts=shifts)

    for row, row_shifts in zip(batch, shifts):
        # Check the expected numerical agreement within the declared tolerance.
        np.testing.assert_allclose(row, engine.apply(boundaries, row_shifts), rtol=1e-12, atol=1e-12)


def test_diffeomorphic_batch_matches_scalar_reference():
    """
    Test the optimized diffeomorphic matrix path against the scalar reference.
    """
    x_grid, y_density = _spectrum()
    boundaries = [(0.0, 0.06), (0.06, 0.12)]
    shifts = np.array([[0.5, 0.8], [-0.4, 0.0]])
    density_matrix = np.vstack([y_density, y_density])

    output_grid, batch, lengths = apply_diffeomorphic_warp_batch(
        x_grid=x_grid,
        y_density=density_matrix,
        boundaries=boundaries,
        shifts=shifts,
        freeze_origin=True,
    )

    for sample_index, row_shifts in enumerate(shifts):
        scalar_grid, scalar_density = apply_diffeomorphic_warp(
            x_grid=x_grid,
            y_density=y_density,
            boundaries=boundaries,
            shifts=row_shifts,
            freeze_origin=True,
        )
        n_points = int(lengths[sample_index])
        # Check the expected numerical agreement within the declared tolerance.
        np.testing.assert_allclose(output_grid[:n_points], scalar_grid, rtol=1e-12, atol=1e-12)
        # Check the expected numerical agreement within the declared tolerance.
        np.testing.assert_allclose(batch[sample_index, :n_points], scalar_density, rtol=1e-10, atol=1e-12)
        assert output_grid[0] == 0.0


def test_segment_translation_numba_batch_matches_scipy_scalar_reference():
    """
    Test valley-aware Numba/PCHIP batching against the retained SciPy scalar path.
    """
    x_grid, y_density = _spectrum()
    boundaries = [(0.0, 0.06), (0.06, 0.12)]
    shifts = np.array([[0.4, 0.8], [-0.3, 0.2]])
    density_matrix = np.vstack([y_density, y_density])

    output_grid, batch, lengths = apply_segment_translation_batch(
        x_grid=x_grid,
        y_density=density_matrix,
        boundaries=boundaries,
        shifts=shifts,
        freeze_origin=True,
        preserve_area=True,
    )

    for sample_index, row_shifts in enumerate(shifts):
        scalar_grid, scalar_density = apply_segment_translation(
            x_grid=x_grid,
            y_density=y_density,
            boundaries=boundaries,
            shifts=row_shifts,
            freeze_origin=True,
            preserve_area=True,
        )
        n_points = int(lengths[sample_index])
        # Check the expected numerical agreement within the declared tolerance.
        np.testing.assert_allclose(output_grid[:n_points], scalar_grid, rtol=1e-12, atol=1e-12)
        # Check the expected numerical agreement within the declared tolerance.
        np.testing.assert_allclose(batch[sample_index, :n_points], scalar_density, rtol=1e-8, atol=1e-10)

        source_area = np.trapezoid(y_density, x_grid)
        result_area = np.trapezoid(batch[sample_index, :n_points], output_grid[:n_points])
        assert np.isclose(source_area, result_area, rtol=1e-6)


def test_spatial_batch_zero_shift_is_exact_identity():
    """Test zero-shift identity for both current spatial implementations."""
    x_grid, y_density = _spectrum()
    boundaries = [(0.0, 0.06), (0.06, 0.12)]
    density_matrix = np.vstack([y_density, y_density])
    shifts = np.zeros((2, 2))

    for operator in (apply_diffeomorphic_warp_batch, apply_segment_translation_batch):
        output_grid, batch, lengths = operator(
            x_grid=x_grid,
            y_density=density_matrix,
            boundaries=boundaries,
            shifts=shifts,
            freeze_origin=True,
        )
        # Require exact array identity here; numerical closeness would be insufficient.
        np.testing.assert_array_equal(output_grid[: x_grid.size], x_grid)
        # Check the expected numerical agreement within the declared tolerance.
        np.testing.assert_allclose(batch[:, : x_grid.size], density_matrix, rtol=0.0, atol=0.0)
        # Require exact array identity here; numerical closeness would be insufficient.
        np.testing.assert_array_equal(lengths, np.full(2, x_grid.size))

import numpy as np
import pandas as pd
import pytest

from core.data.input_pdos import PhononSpectra
from core.data.spectra_ensemble import (
    EnsembleGeneration,
    SamplingBlock,
    SpectraEnsemble,
)


def _generation(n_samples: int = 3, *, variable_coupling: str = "coupled") -> EnsembleGeneration:
    """Generation for the current workflow.

    Parameters
    ----------
    n_samples : int, optional
        Number of stochastic samples to generate or select.
    variable_coupling : str, optional
        Configuration of spatial and amplitude multiplier coupling.

    Returns
    -------
    EnsembleGeneration
        Calculated result in the declared return type.
    """
    return EnsembleGeneration(
        amplitude_method="fast",
        spatial_method="diffeomorphic",
        max_energy_shift_mev=4.0,
        max_amplitude_shift_pct=10.0,
        n_samples=n_samples,
        grid_partition_method="manual",
        manual_energy_ranges=((0.0, 0.02), (0.02, 0.04)),
        domain_splitting_strategy="fixed",
        correlation_strategy="random_group",
        variable_coupling=variable_coupling,
        shared_grid_structure=True,
        freeze_origin=True,
    )


def _reference() -> PhononSpectra:
    """Reference for the current workflow.

    Returns
    -------
    PhononSpectra
        Calculated result in the declared return type.
    """
    grid = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    density = np.array([0.0, 10.0, 20.0, 10.0, 0.0])
    return PhononSpectra(pd.Series(density, index=grid))


def _ensemble() -> SpectraEnsemble:
    """Ensemble for the current workflow.

    Returns
    -------
    SpectraEnsemble
        Calculated result in the declared return type.
    """
    reference = _reference()
    energy_grid = reference.energy_grid
    base = reference.density_values
    density_matrix = np.vstack(
        [
            base,
            base * np.array([1.0, 0.9, 1.1, 1.0, 1.0]),
            base,
        ]
    )
    density_matrix[0, -1] = 0.0
    density_matrix[2, -1] = 0.0

    block = SamplingBlock(
        start_index=0,
        end_index=3,
        boundaries=np.array([[0.0, 0.02], [0.02, 0.04]]),
        spatial_multipliers=np.array([[-0.5], [0.0], [0.75]]),
        amplitude_multipliers=None,
    )

    return SpectraEnsemble(
        energy_grid=energy_grid,
        density_matrix=density_matrix,
        generation=_generation(),
        sample_lengths=np.array([4, 5, 4]),
        sampling_blocks=[block],
        reference=reference,
    )


def test_spectra_ensemble_uses_dense_sample_by_energy_storage():
    """
    Test the canonical dense ensemble representation and right-side padding.

    Verifies that SpectraEnsemble:
    - Exposes samples as DataFrame rows and energy values as columns.
    - Retains the longest canonical energy grid without interpolation.
    - Stores shorter physical supports as right-side zero padding.
    """
    ensemble = _ensemble()

    assert ensemble.n_samples == 3
    assert ensemble.pdos.shape == (3, 5)
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(ensemble.pdos.columns.to_numpy(), ensemble.energy_grid)
    assert ensemble.pdos.index.tolist() == [0, 1, 2]
    assert ensemble.pdos.iloc[0, -1] == 0.0
    assert ensemble.pdos.iloc[2, -1] == 0.0


def test_spectra_ensemble_reconstructs_only_physical_sample_support():
    """
    Test lazy PhononSpectra reconstruction from a padded ensemble row.

    Verifies that:
    - ``ensemble[i]`` returns a PhononSpectra instance.
    - Right-side ensemble padding is removed using the stored sample length.
    - The baseline reference remains separate from stochastic sample indexing.
    """
    ensemble = _ensemble()

    sample = ensemble[0]

    assert isinstance(sample, PhononSpectra)
    assert len(sample.energy_grid) == 4
    assert len(ensemble[-1].energy_grid) == 4
    assert ensemble.reference is not None
    assert ensemble.reference is not sample
    assert len(ensemble) == ensemble.generation.n_samples


def test_sampling_block_preserves_compact_immutable_provenance():
    """
    Test compact stochastic storage for perfectly correlated sampling blocks.

    Verifies that SamplingBlock:
    - Keeps random-group state in the compact ``(N, 1)`` representation.
    - Uses ``amplitude_multipliers=None`` to represent variable coupling.
    - Protects persisted provenance arrays from mutation.
    """
    ensemble = _ensemble()
    block = ensemble.sampling_blocks[0]

    assert block.spatial_multipliers.shape == (3, 1)
    assert block.amplitude_multipliers is None
    assert block.boundaries.flags.writeable is False
    assert block.spatial_multipliers.flags.writeable is False

    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(ValueError):
        block.spatial_multipliers[0, 0] = 0.0


def test_sample_info_is_derived_from_normalized_multiplier_state():
    """
    Test lazy conversion of compact multipliers into physical provenance.

    Verifies that SpectraEnsemble.sample_info:
    - Broadcasts a compact random-group multiplier across all domains.
    - Converts spatial multipliers to meV using EnsembleGeneration.
    - Reuses the same state for amplitude perturbations when variables are coupled.
    """
    ensemble = _ensemble()
    info = ensemble.sample_info

    assert info.index.tolist() == [0, 1, 2]
    assert info.loc[0, "energy_shift_mev"] == (-2.0, -2.0)
    assert info.loc[0, "amplitude_shift_pct"] == (-5.0, -5.0)
    assert info.loc[2, "energy_shift_mev"] == (3.0, 3.0)
    assert info.loc[2, "amplitude_shift_pct"] == (7.5, 7.5)


def test_spectra_ensemble_rejects_incomplete_sampling_block_coverage():
    """
    Test that every stochastic sample must be covered exactly once by provenance.
    """
    reference = _reference()
    block = SamplingBlock(
        start_index=0,
        end_index=2,
        boundaries=np.array([[0.0, 0.04]]),
        spatial_multipliers=np.zeros((2, 1)),
    )

    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(ValueError, match="cover every ensemble sample"):
        SpectraEnsemble(
            energy_grid=reference.energy_grid,
            density_matrix=np.vstack([reference.density_values] * 3),
            generation=_generation(),
            sample_lengths=np.array([5, 5, 5]),
            sampling_blocks=[block],
            reference=reference,
        )

import numpy as np
import pandas as pd
import pytest

from analyzer.ensemble_analyzer import EnsembleAnalyzer
from analyzer.physics.data_models import (
    DifferentialData,
    EnsemblePhysicsData,
    IntegralData,
    PhysicsData,
    ThermodynamicsData,
)
from core.data.input_pdos import PhononSpectra
from core.data.spectra_ensemble import EnsembleGeneration, SamplingBlock, SpectraEnsemble


class MockRunner:
    """Record single-spectrum backend executions and cleanup calls."""

    def __init__(self):
        """Initialize the instance and its required state.
        """
        self.executed_indices = []
        self.cleaned_indices = []

    def execute(self, sample_index, spectrum, **kwargs):
        """Execute for the current workflow.

        Parameters
        ----------
        sample_index : object
            Identifier of the stochastic sample being evaluated.
        spectrum : object
            Source phonon spectrum to evaluate or perturb.
        **kwargs : Any
            Additional options accepted by the current interface.

        Returns
        -------
        dict
            Results associated with their sample, energy, or field identifiers.
        """
        assert isinstance(spectrum, PhononSpectra)
        self.executed_indices.append(sample_index)
        return {"sample_index": sample_index, "runner_kwargs": kwargs}

    def cleanup(self, output):
        """Cleanup for the current workflow.

        Parameters
        ----------
        output : object
            Generated backend or serialization output.
        """
        self.cleaned_indices.append(output["sample_index"])


class MockParser:
    """Return deterministic sample-dependent physics for aggregation tests."""

    def parse(self, sample_index, runner_output, **kwargs):
        """Parse for the current workflow.

        Parameters
        ----------
        sample_index : object
            Identifier of the stochastic sample being evaluated.
        runner_output : object
            Backend state returned by the runner for a single evaluation.
        **kwargs : Any
            Additional options accepted by the current interface.
        """
        thermo = ThermodynamicsData(
            teff=300.0 + sample_index,
            msd=0.01,
            debye_waller=0.5,
            b_factor=0.5,
            cv=10.0 + (sample_index * 2.0),
            entropy=5.0,
            zero_point_energy=0.1,
            free_energy=0.05,
            internal_energy=0.15,
            log_moment=0.0,
        )

        kernel_grids = {
            -1: [0.01, 0.02],
            0: [0.01, 0.02],
            1: [0.01, 0.02, 0.03],
            2: [0.01, 0.02, 0.02, 0.03],
            3: [0.01, 0.03],
        }
        kernel_grid = kernel_grids[sample_index]
        kernel_values = np.linspace(0.2, 0.8, len(kernel_grid))

        cosine_columns = [0.0, 0.5, 1.0] if sample_index in {1, 2} else [0.0, 1.0]
        cosine = pd.DataFrame(
            [np.linspace(0.0, 1.0, len(cosine_columns))],
            index=pd.Index([0.0253], name="Ein"),
            columns=cosine_columns,
        )

        integ = IntegralData(
            xs_total=pd.Series([1.0 + sample_index, 2.0 + sample_index], index=[0.01, 0.1]),
            scattering_kernel={
                0.0253: pd.Series(kernel_values, index=kernel_grid),
            },
            cosine_distribution=cosine,
        )
        return PhysicsData(
            sample_index=sample_index,
            thermo=thermo,
            diff=DifferentialData(),
            integ=integ,
        )


def _ensemble(n_samples: int = 4) -> SpectraEnsemble:
    """Ensemble for the current workflow.

    Parameters
    ----------
    n_samples : int, optional
        Number of stochastic samples to generate or select.

    Returns
    -------
    SpectraEnsemble
        Calculated result in the declared return type.
    """
    grid = np.array([0.0, 0.01, 0.02, 0.03])
    reference = PhononSpectra(pd.Series([0.0, 10.0, 10.0, 0.0], index=grid))
    density = np.vstack([reference.density_values] * n_samples)
    block = SamplingBlock(
        start_index=0,
        end_index=n_samples,
        boundaries=np.array([[0.0, 0.03]]),
        spatial_multipliers=np.zeros((n_samples, 1)),
        amplitude_multipliers=None,
    )
    generation = EnsembleGeneration(
        amplitude_method="fast",
        spatial_method="diffeomorphic",
        max_energy_shift_mev=1.0,
        max_amplitude_shift_pct=5.0,
        n_samples=n_samples,
        grid_partition_method="manual",
        manual_energy_ranges=((0.0, 0.03),),
        domain_splitting_strategy="fixed",
        correlation_strategy="random_group",
        variable_coupling="coupled",
        shared_grid_structure=True,
        freeze_origin=True,
    )
    return SpectraEnsemble(
        energy_grid=grid,
        density_matrix=density,
        generation=generation,
        sample_lengths=np.full(n_samples, grid.size),
        sampling_blocks=[block],
        reference=reference,
    )


def test_ensemble_analyzer_requires_spectra_ensemble():
    """
    Test that obsolete list-based Analyzer construction is rejected explicitly.
    """
    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(TypeError, match="SpectraEnsemble"):
        EnsembleAnalyzer([], MockRunner(), MockParser())


def test_ensemble_analyzer_compute_preserves_stochastic_identity_and_reference():
    """
    Test stochastic aggregation with a reference evaluated outside sample tables.

    Verifies that EnsembleAnalyzer.compute:
    - Uses the original SpectraEnsemble sample indices for stochastic results.
    - Evaluates ``ensemble.reference`` separately with the internal reference identifier.
    - Keeps stochastic DataFrames aligned with ``ensemble.pdos`` and ``sample_info``.
    - Cleans every backend result after parsing.
    """
    ensemble = _ensemble()
    runner = MockRunner()
    analyzer = EnsembleAnalyzer(ensemble, runner, MockParser())

    analyzer.compute(parallel=False)

    assert [result.sample_index for result in analyzer.raw_results] == [0, 1, 2, 3]
    assert analyzer.reference_result is not None
    assert analyzer.reference_result.sample_index == -1
    assert analyzer.thermo_df.index.tolist() == ensemble.pdos.index.tolist()
    # Confirm that aggregated results retain the stochastic provenance contract.
    assert analyzer.thermo_df.index.tolist() == ensemble.sample_info.index.tolist()
    assert runner.executed_indices == [0, 1, 2, 3, -1]
    assert runner.cleaned_indices == [0, 1, 2, 3, -1]
    assert -1 not in analyzer.thermo_df.index


def test_ensemble_analyzer_aligns_variable_and_duplicate_kinematic_grids():
    """
    Test coordinate-aware aggregation of sample-dependent kinematic outputs.

    Verifies that:
    - Dict[Ein, Series] results use the longest existing cleaned grid.
    - Duplicate source coordinates are collapsed before interpolation.
    - DataFrame-shaped multi-Ein results follow their dedicated aggregation path.
    - The resulting coordinate columns are unique for HDF5 serialization.
    """
    analyzer = EnsembleAnalyzer(_ensemble(), MockRunner(), MockParser())
    analyzer.compute(parallel=False, compute_reference=False)

    kernel = analyzer.scattering_kernel_dict[0.0253]
    cosine = analyzer.cosine_distribution_dict[0.0253]

    assert kernel.shape == (4, 3)
    assert kernel.columns.is_unique
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(kernel.columns.to_numpy(), [0.01, 0.02, 0.03])
    assert cosine.shape == (4, 3)
    assert cosine.columns.is_unique


def test_ensemble_analyzer_extract_returns_requested_aggregates_only():
    """
    Test extraction of stochastic aggregate state without mixing in the reference.
    """
    analyzer = EnsembleAnalyzer(_ensemble(), MockRunner(), MockParser())
    analyzer.compute(parallel=False)

    data = analyzer.extract(["xs_total", "scattering_kernel", "xs_inelastic"])

    assert isinstance(data, EnsemblePhysicsData)
    assert data.xs_total is not None
    assert data.xs_total.index.tolist() == [0, 1, 2, 3]
    assert data.scattering_kernel is not None
    assert data.xs_inelastic is None
    assert -1 not in data.xs_total.index

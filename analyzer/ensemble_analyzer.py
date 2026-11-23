"""
Backend-agnostic ensemble orchestrator and statistical aggregator.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from parallel_execution import run_parallel

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from core.data.input_pdos import PhononSpectra
from core.data.material_ensemble import MaterialEnsemble
from core.data.material_scattering import MaterialScatteringModel
from core.data.spectra_ensemble import SpectraEnsemble

from analyzer.core_interfaces import BaseParser, BaseRunner
from analyzer.physics.data_models import PhysicsData


# The reference spectrum is not a stochastic sample. A negative identifier is
# used only while passing it through the existing single-sample backend API.
_REFERENCE_SAMPLE_INDEX = -1


def _isolated_worker(
    index: int,
    spectrum: PhononSpectra | MaterialScatteringModel,
    runner: BaseRunner,
    parser: BaseParser,
    runner_kwargs: Dict[str, Any],
    parser_kwargs: Dict[str, Any],
) -> PhysicsData:
    """
    Execute and parse one spectrum with guaranteed backend cleanup.

    Parameters
    ----------
    index : int
        Stable ensemble sample index used by the runner and parser.
    spectrum : PhononSpectra or MaterialScatteringModel
        Spectrum or full material realization to evaluate.
    runner : BaseRunner
        Backend execution adapter.
    parser : BaseParser
        Backend extraction adapter.
    runner_kwargs : dict
        Keyword arguments forwarded to ``runner.execute``.
    parser_kwargs : dict
        Keyword arguments forwarded to ``parser.parse``.

    Returns
    -------
    PhysicsData
        Parsed physical evaluation for the requested sample.
    """
    runner_output = None
    try:
        runner_output = runner.execute(index, spectrum, **runner_kwargs)
        return parser.parse(index, runner_output, **parser_kwargs)
    finally:
        if runner_output is not None:
            runner.cleanup(runner_output)


class EnsembleAnalyzer:
    """
    Analyze vibrational or material realizations through one backend pipeline.

    The analyzer preserves the original stochastic sample indices throughout
    execution, aggregation, and reduction. The baseline spectrum stored in
    ``ensemble.reference`` is evaluated separately and is never inserted into
    the stochastic result tables.

    Parameters
    ----------
    ensemble : SpectraEnsemble or MaterialEnsemble
        Stochastic spectra or complete liquid materials produced by TARA.
    runner : BaseRunner
        Backend execution adapter.
    parser : BaseParser
        Backend extraction adapter.
    sample_indices : sequence of int, optional
        Original ensemble indices to analyze. If omitted, all stochastic
        samples are selected. This argument is primarily used when creating a
        reduced analyzer while retaining the original ensemble.

    Raises
    ------
    TypeError
        If the supplied ensemble has an unsupported type.
    ValueError
        If ``sample_indices`` contains duplicates or indices outside the
        ensemble range.
    """

    def __init__(
        self,
        ensemble: SpectraEnsemble | MaterialEnsemble,
        runner: BaseRunner,
        parser: BaseParser,
        sample_indices: Optional[Sequence[int]] = None,
    ) -> None:
        """Initialize the analyzer with an ensemble and backend adapters.

        Parameters
        ----------
        ensemble : SpectraEnsemble or MaterialEnsemble
            Stochastic spectra or material realizations with a separate reference.
        runner : BaseRunner
            Backend adapter used to evaluate one spectrum.
        parser : BaseParser
            Adapter that extracts physics results from an in-memory backend state.
        sample_indices : Optional[Sequence[int]], optional
            Original stochastic sample indices to include in the analysis.

        Raises
        ------
        TypeError
            If the input or required state violates the operation contract.
        """
        if not isinstance(ensemble, (SpectraEnsemble, MaterialEnsemble)):
            raise TypeError("ensemble must be a SpectraEnsemble instance or a MaterialEnsemble.")

        self.ensemble = ensemble
        self.runner = runner
        self.parser = parser
        # Retain the original sample identifiers when selecting or reducing the ensemble.
        self.sample_indices = self._validate_sample_indices(sample_indices)

        # Stochastic results and baseline results are intentionally separated so
        # Sample_Index remains aligned with ensemble.pdos and sample_info.
        self.raw_results: List[PhysicsData] = []
        self.reference_result: Optional[PhysicsData] = None

        self.thermo_df: Optional[pd.DataFrame] = None
        self.xs_total_df: Optional[pd.DataFrame] = None
        self.xs_inelastic_df: Optional[pd.DataFrame] = None

    def _validate_sample_indices(
        self,
        sample_indices: Optional[Sequence[int]],
    ) -> np.ndarray:
        """
        Validate and normalize the selected stochastic sample indices.

        Parameters
        ----------
        sample_indices : sequence of int or None
            Requested original ensemble indices. ``None`` selects every sample.

        Returns
        -------
        numpy.ndarray
            One-dimensional integer array of stable ensemble indices.

        Raises
        ------
        ValueError
            If indices are empty, duplicated, or outside the ensemble range.
        """
        if sample_indices is None:
            return np.arange(len(self.ensemble), dtype=np.int64)

        indices = np.asarray(sample_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("sample_indices must be a non-empty one-dimensional sequence.")
        if np.any(indices < 0) or np.any(indices >= len(self.ensemble)):
            raise ValueError("sample_indices contains an index outside the ensemble range.")
        if np.unique(indices).size != indices.size:
            raise ValueError("sample_indices must not contain duplicate indices.")

        return indices.copy()

    def compute(
        self,
        parallel: bool = True,
        runner_kwargs: Optional[Dict[str, Any]] = None,
        parser_kwargs: Optional[Dict[str, Any]] = None,
        compute_reference: bool = True,
    ) -> None:
        """
        Execute backend calculations for the selected stochastic samples.

        Parameters
        ----------
        parallel : bool, default=True
            If True, evaluate stochastic samples with joblib using the available
            worker allocation determined by :mod:`parallel_execution`.
        runner_kwargs : dict, optional
            Keyword arguments forwarded to every runner execution.
        parser_kwargs : dict, optional
            Keyword arguments forwarded to every parser call.
        compute_reference : bool, default=True
            If True and ``ensemble.reference`` is available, evaluate the
            baseline separately and store it in ``reference_result``.

        Notes
        -----
        The reference spectrum never appears in ``raw_results`` or aggregated
        stochastic DataFrames. This preserves one-to-one alignment between
        analyzer ``Sample_Index`` values and the original ensemble indices.
        """
        runner_kwargs = dict(runner_kwargs or {})
        parser_kwargs = parser_kwargs or {}
        if isinstance(self.runner, BaseRunner):
            runner_kwargs = self.runner.prepare_analysis(runner_kwargs, parser_kwargs)

        self.raw_results = self._compute_stochastic_results(
            parallel=parallel,
            runner_kwargs=runner_kwargs,
            parser_kwargs=parser_kwargs,
        )

        # The baseline is a distinct physical reference, not stochastic sample 0.
        self.reference_result = None
        if compute_reference and self.ensemble.reference is not None:
            # Store the baseline result separately so it never occupies a stochastic sample
            # index.
            self.reference_result = _isolated_worker(
                _REFERENCE_SAMPLE_INDEX,
                self.ensemble.reference,
                self.runner,
                self.parser,
                runner_kwargs,
                parser_kwargs,
            )

        self._aggregate_dataframes()

    @staticmethod
    def _prepare_series_for_alignment(series: pd.Series) -> pd.Series:
        """
        Normalize one kinematic Series for ensemble-level alignment.

        Duplicate coordinates are collapsed by averaging their values. The
        coordinate order of first occurrence is retained so the selected
        canonical grid preserves the orientation produced by the physics
        calculation.

        Parameters
        ----------
        series : pandas.Series
            One-dimensional kinematic result indexed by numeric coordinates.

        Returns
        -------
        pandas.Series
            Numeric, finite Series with a unique coordinate index.

        Raises
        ------
        ValueError
            If coordinates or values are non-numeric, non-finite, or have
            inconsistent lengths.
        """
        try:
            coordinates = np.asarray(series.index, dtype=np.float64)
            values = np.asarray(series.values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Kinematic result coordinates and values must be numeric."
            ) from exc

        if coordinates.size != values.size:
            raise ValueError(
                "Kinematic result coordinates and values must have matching lengths."
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Kinematic result coordinates must be finite.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Kinematic result values must be finite.")

        prepared = pd.Series(values, index=pd.Index(coordinates))
        if not prepared.index.is_unique:
            prepared = prepared.groupby(level=0, sort=False).mean()

        return prepared

    @staticmethod
    def _align_series_collection(
        series_by_sample: Sequence[tuple[int, pd.Series]],
    ) -> pd.DataFrame:
        """
        Align sample-dependent one-dimensional results onto one existing grid.

        Each input Series is first normalized to a unique coordinate axis by
        averaging exact duplicate coordinates. The longest cleaned grid already
        present in the collection is then used as the canonical coordinate
        system. Other series are linearly interpolated onto that grid, with zero
        outside their original support.

        Parameters
        ----------
        series_by_sample : sequence of tuple of int and pandas.Series
            Pairs of original ensemble sample index and one-dimensional
            kinematic result.

        Returns
        -------
        pandas.DataFrame
            Sample-indexed matrix with a unique canonical coordinate axis.

        Raises
        ------
        ValueError
            If no series are supplied or a series cannot be normalized to a
            finite numeric coordinate/value representation.
        """
        if not series_by_sample:
            raise ValueError("At least one series is required for alignment.")

        prepared_series = [
            (
                int(sample_index),
                EnsembleAnalyzer._prepare_series_for_alignment(series),
            )
            for sample_index, series in series_by_sample
        ]

        canonical_series = max(
            (series for _, series in prepared_series),
            key=lambda series: series.size,
        )
        canonical_index = canonical_series.index.copy()
        target_grid = canonical_index.to_numpy(dtype=np.float64, copy=False)

        matrix = np.empty(
            (len(prepared_series), target_grid.size),
            dtype=np.float64,
        )
        sample_indices = []

        for row_idx, (sample_index, series) in enumerate(prepared_series):
            matrix[row_idx] = EnsembleAnalyzer._interpolate_clean_series(
                series, target_grid,
            )

            sample_indices.append(sample_index)

        aligned = pd.DataFrame(
            matrix,
            index=sample_indices,
            columns=canonical_index,
        )
        aligned.index.name = "Sample_Index"
        return aligned

    @staticmethod
    def _interpolate_clean_series(series: pd.Series, target_grid: np.ndarray) -> np.ndarray:
        """Interpolate one cleaned series without extrapolating its support.

        Parameters
        ----------
        series : pandas.Series
            Series with unique coordinates and numerical values.
        target_grid : numpy.ndarray
            Destination numeric coordinate grid.

        Returns
        -------
        numpy.ndarray
            Interpolated values, zero outside the original support.
        """
        source_grid = series.index.to_numpy(dtype=np.float64, copy=False)
        if source_grid.size == 0:
            return np.zeros(target_grid.size, dtype=np.float64)
        order = np.argsort(source_grid, kind="stable")
        return np.interp(
            target_grid,
            source_grid[order],
            series.to_numpy(dtype=np.float64, copy=False)[order],
            left=0.0, right=0.0,
        )

    def _compute_stochastic_results(
        self,
        *,
        parallel: bool,
        runner_kwargs: Dict[str, Any],
        parser_kwargs: Dict[str, Any],
    ) -> List[PhysicsData]:
        """Evaluate selected stochastic samples through the shared runner.

        Parameters
        ----------
        parallel : bool
            Whether to use the reserved, HPC-aware worker count.
        runner_kwargs : dict[str, Any]
            Arguments for the backend runner.
        parser_kwargs : dict[str, Any]
            Arguments for the backend parser.

        Returns
        -------
        list[PhysicsData]
            Results ordered by the original sample selection.
        """
        return run_parallel(
            _isolated_worker,
            (
                ((int(index), self.ensemble[int(index)], self.runner,
                  self.parser, runner_kwargs, parser_kwargs), {})
                for index in self.sample_indices
            ),
            n_jobs=-1 if parallel else 1,
        )

    def _aggregate_dataframes(self) -> None:
        """Aggregate computed stochastic results into ensemble DataFrames.
        """
        if not self.raw_results:
            return

        self._aggregate_thermodynamics()
        self._aggregate_integral_1d()
        self._aggregate_dict_multi_ein()
        self._aggregate_dataframe_multi_ein()

    def _aggregate_thermodynamics(self) -> None:
        """Aggregate thermodynamic scalar values by stochastic sample index.
        """
        thermo_list = []
        for result in self.raw_results:
            thermo = result.thermo
            thermo_list.append(
                {
                    "Sample_Index": result.sample_index,
                    "teff": thermo.teff,
                    "msd": thermo.msd,
                    "cv": thermo.cv,
                    "entropy": thermo.entropy,
                    "zero_point_energy": thermo.zero_point_energy,
                    "free_energy": thermo.free_energy,
                }
            )

        self.thermo_df = pd.DataFrame(thermo_list).set_index("Sample_Index")
        scopes = {(result.metadata or {}).get("thermodynamics_scope")
                  for result in self.raw_results}
        if len(scopes) == 1 and None not in scopes:
            self.thermo_df.attrs["scope"] = scopes.pop()

    def _aggregate_integral_1d(self) -> None:
        """Aggregate one-dimensional integral outputs on their shared grids.
        """
        integral_1d_attributes = [
            "xs_total",
            "xs_inelastic",
            "xs_coherent_elastic",
            "xs_incoherent_elastic",
            "xs_absorption",
            "upscattering_ratio",
            "downscattering_ratio",
            "elastic_ratio",
        ]

        for attr in integral_1d_attributes:
            available = [
                result
                for result in self.raw_results
                if getattr(result.integ, attr) is not None
            ]
            if not available:
                continue

            shared_grid = getattr(available[0].integ, attr).index
            matrix = np.array(
                [getattr(result.integ, attr).values for result in available]
            )
            valid_indices = [result.sample_index for result in available]

            df = pd.DataFrame(
                matrix,
                index=valid_indices,
                columns=shared_grid,
            )
            df.index.name = "Sample_Index"
            setattr(self, f"{attr}_df", df)

    def _aggregate_dict_multi_ein(self) -> None:
        """Aggregate ``Dict[Ein, Series]`` kinematic outputs by incident energy.
        """
        attributes = [
            "scattering_kernel",
            "angular_distribution",
        ]

        for attr in attributes:
            available = [
                result
                for result in self.raw_results
                if getattr(result.integ, attr) is not None
            ]
            if not available:
                continue

            eins = sorted(
                {
                    ein
                    for result in available
                    for ein in getattr(result.integ, attr).keys()
                }
            )
            aggregated_dict = {}

            for ein in eins:
                series_by_sample = [
                    (result.sample_index, getattr(result.integ, attr)[ein])
                    for result in available
                    if ein in getattr(result.integ, attr)
                ]
                if series_by_sample:
                    aggregated_dict[ein] = self._align_series_collection(
                        series_by_sample
                    )

            setattr(self, f"{attr}_dict", aggregated_dict)

    def _aggregate_dataframe_multi_ein(self) -> None:
        """Aggregate DataFrame-shaped kinematic outputs by incident energy.
        """
        attributes = [
            "cosine_distribution",
            "energy_distribution",
        ]

        for attr in attributes:
            available = [
                result
                for result in self.raw_results
                if getattr(result.integ, attr) is not None
            ]
            if not available:
                continue

            eins = sorted(
                {
                    ein
                    for result in available
                    for ein in getattr(result.integ, attr).index
                }
            )
            aggregated_dict = {}

            for ein in eins:
                series_by_sample = [
                    # Multi-Ein frames use an outer join of native coordinates.
                    # Missing cells are absent support, not nonfinite physics.
                    (result.sample_index, getattr(result.integ, attr).loc[ein].dropna())
                    for result in available
                    if ein in getattr(result.integ, attr).index
                ]
                if series_by_sample:
                    aggregated_dict[ein] = self._align_series_collection(
                        series_by_sample
                    )

            setattr(self, f"{attr}_dict", aggregated_dict)

    def extract(self, keys: List[str]) -> Any:
        """
        Extract explicitly requested aggregated properties.

        Parameters
        ----------
        keys : list of str
            Aggregated analyzer attributes to place in the returned container.

        Returns
        -------
        EnsemblePhysicsData
            Pure-state container containing the requested computed properties.
        """
        from .physics.data_models import EnsemblePhysicsData

        ensemble_data = EnsemblePhysicsData()
        sample_metadata = {result.sample_index: result.metadata
                           for result in self.raw_results if result.metadata is not None}
        reference_metadata = (None if self.reference_result is None
                              else self.reference_result.metadata)
        if sample_metadata or reference_metadata is not None:
            ensemble_data.metadata = {"samples": sample_metadata,
                                      "reference": reference_metadata}

        for key in keys:
            target_df = f"{key}_df"
            target_dict = f"{key}_dict"

            if hasattr(self, target_df):
                df = getattr(self, target_df)
                if df is not None and not df.empty:
                    setattr(ensemble_data, key, df)
                else:
                    logging.warning("Key '%s' was requested but is empty.", key)
            elif hasattr(self, target_dict):
                data = getattr(self, target_dict)
                if data:
                    setattr(ensemble_data, key, data)
                else:
                    logging.warning("Key '%s' was requested but is empty.", key)
            else:
                logging.warning(
                    "Key '%s' is not a recognized computed attribute.",
                    key,
                )

        return ensemble_data

    def reduce(
        self,
        sample_indices: Sequence[int],
    ) -> "EnsembleAnalyzer":
        """
        Return an analyzer restricted to selected stochastic samples.

        Original ensemble indices and provenance are preserved.
        Previously computed results are reused when available.
        """
        indices = self._validate_sample_indices(sample_indices)

        if not np.all(np.isin(indices, self.sample_indices)):
            raise ValueError(
                "Selected indices must belong to the current analyzer."
            )

        reduced = EnsembleAnalyzer(
            ensemble=self.ensemble,
            runner=self.runner,
            parser=self.parser,
            sample_indices=indices,
        )

        # Reuse computed results without running the backend again.
        if self.raw_results:
            results_by_index = {
                result.sample_index: result
                for result in self.raw_results
            }

            reduced.raw_results = [
                results_by_index[int(index)]
                for index in indices
                if int(index) in results_by_index
            ]

            reduced._aggregate_dataframes()

        # Reference is independent of stochastic sample selection.
        reduced.reference_result = self.reference_result

        return reduced
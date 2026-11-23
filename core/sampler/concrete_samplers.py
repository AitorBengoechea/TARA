"""
Main orchestration engine for TARA stochastic sampling.
"""

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from core.config import TaraConfig, VibrationSamplingConfig
from core.data.input_pdos import PhononSpectra
from core.data.material_ensemble import MaterialEnsemble
from core.data.material_scattering import MaterialScatteringModel
from core.data.spectra_batch import PerturbationBatch, SpectrumBatch
from core.data.spectra_ensemble import (
    EnsembleGeneration,
    SamplingBlock,
    SpectraEnsemble,
)
from core.sampler.correlation_engine import get_correlation_strategy
from core.sampler.diffusion_sampler import DiffusionSampler
from core.sampler.parameter_space import ParameterSpaceSampler
from core.sampler.domain_partitioner import DomainPartitioner
from core.sampler.topology_strategy import get_topology_strategy

# Import the mathematical operators
from core.math_engine.bspline_fit import (
    BsplineAmplitudePerturber,
    BsplineFitter,
    apply_amplitude_perturbation,
)
from core.math_engine.fast_amplitude import FastAmplitudePerturber
from core.math_engine.diffeomorphic import (
    apply_spatial_warp as apply_diffeomorphic_warp,
    apply_spatial_warp_batch as apply_diffeomorphic_warp_batch,
)
from core.math_engine.segment_translation import (
    apply_spatial_warp as apply_segment_translation,
    apply_spatial_warp_batch as apply_segment_translation_batch,
)


class VibrationSampler:
    """
    Execute the stochastic sampling pipeline for phonon spectra.

    This orchestrator binds the configuration gateway, validated data model,
    domain topology, correlation strategy, amplitude perturbation method, and
    spatial perturbation method used to generate the final ensemble.

    Parameters
    ----------
    config : VibrationSamplingConfig
        Validated vibrational configuration.
    spectra : PhononSpectra
        Baseline phonon density of states.
    amplitude_method : {"precise", "fast"}, default="fast"
        Amplitude perturbation method. ``"precise"`` uses the constrained
        B-spline implementation, while ``"fast"`` uses direct domain scaling.
    spatial_method : {"diffeomorphic", "crossfade"}, default="diffeomorphic"
        Spatial perturbation method. ``"diffeomorphic"`` uses continuous
        coordinate warping. ``"crossfade"`` uses the valley-aware segment-translation
        implementation.

    Attributes
    ----------
    config : VibrationSamplingConfig
        Validated user configuration constraints.
    base_spectra : PhononSpectra
        Standardized baseline phonon spectrum.
    partitioner : DomainPartitioner
        Engine managing the energy-grid partitioning.
    topology_strategy
        Strategy controlling boundary variation across generated samples.
    correlation_engine
        Generator for orthogonal or correlated random domain multipliers.
    amplitude_method : str
        Normalized amplitude perturbation method.
    spatial_method : str
        Normalized spatial perturbation method.
    _amplitude_operator : callable
        Bound amplitude perturbation implementation.
    _spatial_operator : callable
        Bound spatial perturbation implementation.
    """

    def __init__(
        self,
        config: VibrationSamplingConfig,
        spectra: PhononSpectra,
        amplitude_method: str = "fast",
        spatial_method: str = "diffeomorphic",
    ) -> None:
        """
        Initialize the vibrational sampler.

        Parameters
        ----------
        config : VibrationSamplingConfig
            Validated vibrational configuration.
        spectra : PhononSpectra
            Baseline phonon density of states.
        amplitude_method : {"precise", "fast"}, default="fast"
            Amplitude perturbation method.
        spatial_method : {"diffeomorphic", "crossfade"}, default="diffeomorphic"
            Spatial perturbation method.

        Raises
        ------
        ValueError
            If an unsupported amplitude or spatial method is requested.
        """
        self.config = config
        self.base_spectra = spectra

        # Initialize the stochastic rule engines
        self.partitioner = DomainPartitioner(self.base_spectra, self.config)
        self.topology_strategy = get_topology_strategy(
            self.config,
            self.partitioner,
        )
        self.correlation_engine = get_correlation_strategy(self.config)
        self.parameter_space = ParameterSpaceSampler(
            sampling_method=self.config.sampling_method, seed=self.config.random_seed,
        )

        # Initialize only the mathematical engines requested by the caller.
        self._initialize_amplitude_engine(amplitude_method)
        self._initialize_spatial_engine(spatial_method)

    def _initialize_amplitude_engine(self, amplitude_method: str) -> None:
        """
        Initialize the selected amplitude perturbation operator.

        Parameters
        ----------
        amplitude_method : {"precise", "fast"}
            Amplitude perturbation method. ``"precise"`` preserves the
            constrained B-spline methodology, while ``"fast"`` applies direct
            domain scaling with local smooth transitions.

        Raises
        ------
        ValueError
            If ``amplitude_method`` is not supported.
        """
        method = str(amplitude_method).strip().lower()
        if method not in {"precise", "fast"}:
            raise ValueError(
                "amplitude_method must be either 'precise' or 'fast'. "
                f"Got {amplitude_method!r}."
            )

        self.amplitude_method = method
        self.base_fitter = None
        self.precise_amplitude_engine = None
        self.fast_amplitude_engine = None

        if method == "precise":
            # Fit the constrained baseline once, then cache the sparse spline
            # basis and topology transforms used by the batched hot path.
            base_series = pd.Series(
                self.base_spectra.density_values,
                index=self.base_spectra.energy_grid,
            )
            self.base_fitter = BsplineFitter(base_series)
            self.base_fitter.find_best_n_splines(verbose=False)
            self.precise_amplitude_engine = BsplineAmplitudePerturber(
                x_grid=self.base_spectra.energy_grid,
                base_fitter=self.base_fitter,
            )
            self._amplitude_operator = self._apply_precise_amplitude
            self._amplitude_batch_operator = (
                self.precise_amplitude_engine.apply_batch
            )
        else:
            # Fast setup stores the baseline arrays and topology weights once.
            self.fast_amplitude_engine = FastAmplitudePerturber(
                x_grid=self.base_spectra.energy_grid,
                y_density=self.base_spectra.density_values,
            )
            self._amplitude_operator = self._apply_fast_amplitude
            self._amplitude_batch_operator = (
                self.fast_amplitude_engine.apply_batch
            )

    def _apply_precise_amplitude(
        self,
        x_grid: np.ndarray,
        boundaries: List[Tuple[float, float]],
        shifts: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the constrained B-spline amplitude perturbation.

        Parameters
        ----------
        x_grid : numpy.ndarray
            Energy grid of the spectrum.
        boundaries : list of tuple of float
            Energy-domain boundaries.
        shifts : numpy.ndarray
            Per-domain relative amplitude shifts.

        Returns
        -------
        numpy.ndarray
            Perturbed density values.
        """
        return apply_amplitude_perturbation(
            x_grid=x_grid,
            base_fitter=self.base_fitter,
            boundaries=boundaries,
            shifts=shifts,
        )

    def _apply_fast_amplitude(
        self,
        x_grid: np.ndarray,
        boundaries: List[Tuple[float, float]],
        shifts: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the fast domain-wise amplitude perturbation.

        Parameters
        ----------
        x_grid : numpy.ndarray
            Energy grid of the spectrum. The fast engine already stores the
            baseline grid, so this argument is accepted for operator symmetry.
        boundaries : list of tuple of float
            Energy-domain boundaries.
        shifts : numpy.ndarray
            Per-domain relative amplitude shifts.

        Returns
        -------
        numpy.ndarray
            Perturbed density values.
        """
        del x_grid
        return self.fast_amplitude_engine.apply(
            boundaries=boundaries,
            shifts=shifts,
        )

    def _initialize_spatial_engine(self, spatial_method: str) -> None:
        """
        Initialize the selected spatial perturbation operator.

        Parameters
        ----------
        spatial_method : {"diffeomorphic", "crossfade"}
            Spatial perturbation method. ``"diffeomorphic"`` uses the
            topology-preserving coordinate warp, while ``"crossfade"`` uses
            valley-aware segment translation.

        Raises
        ------
        ValueError
            If ``spatial_method`` is not supported.
        """
        method = str(spatial_method).strip().lower()

        aliases = {
            "diffeomorphic": "diffeomorphic",
            "warp": "diffeomorphic",
            "crossfade": "crossfade",
            "cross_fade": "crossfade",
            "cross-fade": "crossfade",
            "segment": "crossfade",
            "segment_translation": "crossfade",
        }

        if method not in aliases:
            raise ValueError(
                "spatial_method must be either 'diffeomorphic' or 'crossfade'. "
                f"Got {spatial_method!r}."
            )

        self.spatial_method = aliases[method]

        if self.spatial_method == "diffeomorphic":
            # Scalar execution is retained for previews and regression checks.
            # Ensemble generation uses the matrix implementation directly.
            self._spatial_operator = apply_diffeomorphic_warp
            self._spatial_batch_operator = apply_diffeomorphic_warp_batch
        else:
            # Valley-aware PCHIP translation exposes the same batch contract.
            # Its trusted reconstruction remains internally sample dependent.
            self._spatial_operator = apply_segment_translation
            self._spatial_batch_operator = apply_segment_translation_batch

    def generate_preview(self) -> SpectraEnsemble:
        """
        Generate deterministic extrema as a compact, unnormalized ensemble.

        The preview explores both signs of each active perturbation, keeping
        the established two- or four-state ordering. Its reference spectrum
        is stored separately and not counted among the extrema.

        Returns
        -------
        SpectraEnsemble
            Deterministic preview spectra and explicit perturbation metadata.

        Raises
        ------
        ValueError
            If neither perturbation type is active.
        RuntimeError
            If preview spectra cannot share a canonical energy-grid prefix.
        """
        boundaries = np.asarray(self.partitioner.base_partitions, dtype=np.float64)
        spatial, amplitude = self._preview_multiplier_states(len(boundaries))
        batch = self._preview_spectrum_batch(boundaries, spatial, amplitude)

        block = SamplingBlock(
            start_index=0,
            end_index=len(spatial),
            boundaries=boundaries,
            spatial_multipliers=spatial,
            amplitude_multipliers=amplitude,
        )
        return self._assemble_ensemble(
            batch.energy_grid,
            batch.density,
            batch.sample_lengths,
            (block,),
            auto_normalize_samples=False,
        )

    def _preview_multiplier_states(
        self,
        n_domains: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct ordered, full-domain endpoint multipliers for the preview.

        Parameters
        ----------
        n_domains : int
            Number of physical energy domains in the fixed preview topology.

        Returns
        -------
        tuple of numpy.ndarray
            Spatial and amplitude multiplier matrices, each containing one
            row per deterministic extreme state.

        Raises
        ------
        ValueError
            If both maximum perturbation magnitudes are zero.
        """
        shift_active = self.config.max_energy_shift_mev > 0.0
        amp_active = self.config.max_amplitude_shift_pct > 0.0

        if shift_active and amp_active:
            signs = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
        elif amp_active:
            signs = ((1.0, 1.0), (1.0, -1.0))
        elif shift_active:
            signs = ((1.0, 1.0), (-1.0, 1.0))
        else:
            raise ValueError(
                "Cannot generate preview extrema when both maximum "
                "perturbation magnitudes are zero."
            )

        endpoints = np.asarray(signs, dtype=np.float64)
        spatial = np.repeat(endpoints[:, 0:1], n_domains, axis=1)
        amplitude = np.repeat(endpoints[:, 1:2], n_domains, axis=1)
        return spatial, amplitude

    def _preview_spectrum_batch(
        self,
        boundaries: np.ndarray,
        spatial: np.ndarray,
        amplitude: np.ndarray,
    ) -> SpectrumBatch:
        """
        Pack scalar preview extrema without modifying their original areas.

        Scalar spatial operators are used to preserve historical previews,
        including the extension lengths produced by the crossfade method.

        Parameters
        ----------
        boundaries : numpy.ndarray
            Shared preview energy domains in eV.
        spatial : numpy.ndarray
            Spatial multipliers, one row per preview spectrum.
        amplitude : numpy.ndarray
            Amplitude multipliers, one row per preview spectrum.

        Returns
        -------
        SpectrumBatch
            Right-padded preview densities and original physical lengths.

        Raises
        ------
        RuntimeError
            If a generated spectrum does not share the canonical grid prefix.
        """
        spectra = [
            self._perturb_single_sample_arrays(
                boundaries=boundaries,
                spatial_multipliers=spatial[i],
                amplitude_multipliers=amplitude[i],
                normalize=False,
            )
            for i in range(len(spatial))
        ]
        energy_grid = max(spectra, key=lambda sample: sample[0].size)[0]
        density = np.zeros((len(spectra), energy_grid.size), dtype=np.float64)
        sample_lengths = np.empty(len(spectra), dtype=np.int64)

        for i, (sample_grid, sample_density) in enumerate(spectra):
            n_points = sample_grid.size
            if not np.allclose(
                sample_grid, energy_grid[:n_points], rtol=1e-10, atol=1e-12
            ):
                raise RuntimeError(
                    f"Preview {i} does not share the canonical energy-grid prefix."
                )
            density[i, :n_points] = sample_density
            sample_lengths[i] = n_points

        return SpectrumBatch(
            energy_grid=energy_grid,
            density=density,
            sample_lengths=sample_lengths,
        )

    def generate_ensemble(
        self,
        batch_size: int = 512,
    ) -> SpectraEnsemble:
        """
        Generate the stochastic ensemble through matrix-batched operators.

        The topology strategy still defines scientifically meaningful blocks,
        while ``batch_size`` controls only the transient numerical workload.
        Each chunk is processed as a matrix through amplitude perturbation,
        spatial perturbation, validation, and normalization before being written
        directly into the final ensemble storage.

        Parameters
        ----------
        batch_size : int, default=512
            Maximum number of spectra evaluated simultaneously. This parameter
            affects computational memory and throughput only; it does not alter
            the stochastic model or generated random sequence.

        Returns
        -------
        SpectraEnsemble
            Generated spectra together with compact block-level provenance.

        Raises
        ------
        ValueError
            If ``config.n_samples`` is zero or negative, or if ``batch_size`` is
            not a positive integer.
        RuntimeError
            If a batch operator returns invalid arrays or a grid incompatible
            with the canonical TARA extension rule.
        """
        if self.config.n_samples <= 0:
            raise ValueError(
                "Cannot generate a stochastic ensemble with n_samples=0. "
                "Set n_samples > 0 or use generate_preview() instead."
            )
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        capacity_grid = self._build_ensemble_capacity_grid()
        density_matrix = np.zeros(
            (self.config.n_samples, capacity_grid.size),
            dtype=np.float64,
        )
        sample_lengths = np.empty(self.config.n_samples, dtype=np.int64)
        sampling_blocks: List[SamplingBlock] = []

        # Topology remains the outer scientific unit. Random states are drawn for
        # the complete topology block before numerical chunking so changing
        # ``batch_size`` does not change the random-number sequence.
        for subensemble_id, (
            start_idx,
            end_idx,
            boundaries,
        ) in enumerate(self.topology_strategy.generate_topology_plan()):
            n_sub_samples = end_idx - start_idx
            n_domains = len(boundaries)
            boundaries_arr = np.asarray(boundaries, dtype=np.float64)

            # Each topology block is its own LHS universe. Parameter groups
            # have independent, stable named streams; coupling deliberately
            # reuses one group's coordinates rather than allocating new ones.
            coupled = self.config.variable_coupling == "coupled"
            spatial_active = (self.config.max_energy_shift_mev > 0 or coupled
                              or self.config.sampling_method == "monte_carlo")
            if spatial_active:
                spatial_multipliers = self.parameter_space.generate_domains(
                    n_sub_samples, n_domains,
                    correlation_strategy=self.config.correlation_strategy,
                    subensemble_id=subensemble_id, group="vdos_spatial",
                )
            else:
                spatial_multipliers = np.zeros((n_sub_samples, n_domains))

            if coupled:
                amplitude_multipliers = spatial_multipliers
            elif (self.config.max_amplitude_shift_pct > 0
                  or self.config.sampling_method == "monte_carlo"):
                amplitude_multipliers = self.parameter_space.generate_domains(
                    n_sub_samples, n_domains,
                    correlation_strategy=self.config.correlation_strategy,
                    subensemble_id=subensemble_id, group="vdos_amplitude",
                )
            else:
                amplitude_multipliers = np.zeros((n_sub_samples, n_domains))

            # Process cache-friendly matrix chunks rather than dispatching one
            # Python function call per spectrum.
            for local_start in range(0, n_sub_samples, int(batch_size)):
                local_end = min(local_start + int(batch_size), n_sub_samples)
                global_start = start_idx + local_start
                global_end = start_idx + local_end

                perturbation_batch = PerturbationBatch(
                    start_index=global_start,
                    end_index=global_end,
                    boundaries=boundaries_arr,
                    spatial_multipliers=spatial_multipliers[
                        local_start:local_end
                    ],
                    amplitude_multipliers=(
                        None
                        if self.config.variable_coupling == "coupled"
                        else amplitude_multipliers[local_start:local_end]
                    ),
                )

                spectrum_batch = self._perturb_batch_arrays(
                    batch=perturbation_batch,
                    capacity_grid=capacity_grid,
                    normalize=True,
                )

                if not np.allclose(
                    spectrum_batch.energy_grid,
                    capacity_grid,
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "Batch spatial operator changed the canonical TARA "
                        "energy grid. Ensemble aggregation does not interpolate."
                    )

                density_matrix[global_start:global_end, :] = (
                    spectrum_batch.density
                )
                sample_lengths[global_start:global_end] = (
                    spectrum_batch.sample_lengths
                )

            # Persist the stochastic state once per scientific topology block,
            # independent of the transient computational batch size.
            sampling_blocks.append(
                SamplingBlock(
                    start_index=start_idx,
                    end_index=end_idx,
                    boundaries=boundaries_arr,
                    spatial_multipliers=self._compact_multiplier_state(
                        spatial_multipliers
                    ),
                    amplitude_multipliers=(
                        None
                        if self.config.variable_coupling == "coupled"
                        else self._compact_multiplier_state(
                            amplitude_multipliers
                        )
                    ),
                )
            )

        return self._assemble_ensemble(
            capacity_grid,
            density_matrix,
            sample_lengths,
            sampling_blocks,
        )

    def _generation_record(self, n_samples: int) -> EnsembleGeneration:
        """
        Build a metadata snapshot for a generated collection of spectra.

        Parameters
        ----------
        n_samples : int
            Actual number of spectra in the collection.

        Returns
        -------
        EnsembleGeneration
            Reusable snapshot shared by stochastic and preview ensembles.
        """
        manual_ranges = (
            None
            if self.config.manual_energy_ranges is None
            else tuple(
                (float(start), float(end))
                for start, end in self.config.manual_energy_ranges
            )
        )
        return EnsembleGeneration(
            amplitude_method=self.amplitude_method,
            spatial_method=self.spatial_method,
            max_energy_shift_mev=float(self.config.max_energy_shift_mev),
            max_amplitude_shift_pct=float(self.config.max_amplitude_shift_pct),
            n_samples=int(n_samples),
            grid_partition_method=str(self.config.grid_partition_method),
            automatic_partition_prominence_pct=float(
                self.config.automatic_partition_prominence_pct
            ),
            manual_energy_ranges=manual_ranges,
            domain_splitting_strategy=str(self.config.domain_splitting_strategy),
            correlation_strategy=str(self.config.correlation_strategy),
            variable_coupling=str(self.config.variable_coupling),
            shared_grid_structure=bool(self.config.shared_grid_structure),
            freeze_origin=bool(self.config.freeze_origin),
            sampling_method=self.config.sampling_method,
        )

    def _assemble_ensemble(
        self,
        energy_grid: np.ndarray,
        density_matrix: np.ndarray,
        sample_lengths: np.ndarray,
        sampling_blocks: Sequence[SamplingBlock],
        *,
        auto_normalize_samples: bool = True,
    ) -> SpectraEnsemble:
        """
        Assemble compact ensemble storage and shared generation metadata.

        Parameters
        ----------
        energy_grid : numpy.ndarray
            Canonical grid large enough for every sample.
        density_matrix : numpy.ndarray
            Right-padded density rows.
        sample_lengths : numpy.ndarray
            Unpadded physical support length of each row.
        sampling_blocks : sequence of SamplingBlock
            Provenance covering the complete ensemble.
        auto_normalize_samples : bool, default=True
            Whether lazy ``PhononSpectra`` reconstruction normalizes rows.

        Returns
        -------
        SpectraEnsemble
            Trimmed ensemble with reference spectrum and provenance.
        """
        max_points = int(np.max(sample_lengths))
        return SpectraEnsemble(
            energy_grid=energy_grid[:max_points],
            density_matrix=density_matrix[:, :max_points],
            generation=self._generation_record(len(sample_lengths)),
            sample_lengths=sample_lengths,
            sampling_blocks=sampling_blocks,
            reference=self.base_spectra,
            auto_normalize_samples=auto_normalize_samples,
        )

    def _build_ensemble_capacity_grid(self) -> np.ndarray:
        """
        Build a conservative shared grid for direct ensemble allocation.

        TARA spatial operators preserve the existing grid and may only extend
        the high-energy side. New points use the mean base-grid spacing. The
        configured maximum spatial shift therefore provides an upper bound on
        the number of extra points required by any sample.

        Returns
        -------
        numpy.ndarray
            Preallocation grid large enough for every configured sample.

        Raises
        ------
        ValueError
            If the baseline spectrum has fewer than two energy points or an
            invalid mean grid spacing.
        """
        base_grid = np.asarray(
            self.base_spectra.energy_grid,
            dtype=np.float64,
        )
        if base_grid.size < 2:
            raise ValueError(
                "TaraSampler requires at least two baseline energy points."
            )

        # Examine adjacent grid differences to enforce ordering or locate transitions.
        dx = float(np.mean(np.diff(base_grid)))
        if not np.isfinite(dx) or dx <= 0.0:
            raise ValueError(
                "The baseline spectrum must have a positive mean grid spacing."
            )

        max_extension_eV = self.config.max_energy_shift_mev / 1000.0
        if max_extension_eV <= 0.0:
            return base_grid.copy()

        # Both spatial implementations append two numerical guard points after
        # the minimum extension required to contain a positively shifted edge.
        n_extra = int(np.ceil(max_extension_eV / dx)) + 2
        extra_grid = base_grid[-1] + np.arange(
            1,
            n_extra + 1,
            dtype=np.float64,
        ) * dx
        # Assemble the grid segments in the order required by the reconstruction.
        return np.concatenate((base_grid, extra_grid))

    def _compact_multiplier_state(self, multipliers: np.ndarray) -> np.ndarray:
        """
        Compress a block multiplier matrix without losing stochastic information.

        Random-group correlation contains one independent value per sample even
        though its public array shape spans every domain. Only that first column
        is persisted; independent correlation retains the full matrix.

        Parameters
        ----------
        multipliers : numpy.ndarray
            Multiplier matrix returned by the configured correlation engine.

        Returns
        -------
        numpy.ndarray
            Full independent state or a one-column random-group state.
        """
        multipliers = np.asarray(multipliers, dtype=np.float64)
        if self.config.correlation_strategy == "random_group":
            return multipliers[:, :1]
        return multipliers

    def _perturb_batch_arrays(
        self,
        batch: PerturbationBatch,
        capacity_grid: np.ndarray,
        normalize: bool,
    ) -> SpectrumBatch:
        """
        Apply amplitude and spatial perturbations to one matrix batch.

        Parameters
        ----------
        batch : PerturbationBatch
            Shared topology and stochastic state for a contiguous sample chunk.
        capacity_grid : numpy.ndarray
            Canonical ensemble grid large enough for every configured shift.
        normalize : bool
            If True, normalize every generated spectrum to unit area.

        Returns
        -------
        SpectrumBatch
            Validated generated matrix and per-row physical support lengths.
        """
        boundaries = [
            (float(start), float(end))
            for start, end in np.asarray(batch.boundaries, dtype=np.float64)
        ]

        amplitude_multipliers = (
            batch.spatial_multipliers
            if batch.amplitude_multipliers is None
            else batch.amplitude_multipliers
        )

        amp_shifts = np.asarray(
            amplitude_multipliers,
            dtype=np.float64,
        ) * (self.config.max_amplitude_shift_pct / 100.0)
        peak_shifts = np.asarray(
            batch.spatial_multipliers,
            dtype=np.float64,
        ) * self.config.max_energy_shift_mev

        # 1. Evaluate amplitude perturbations for every row simultaneously.
        y_amplitude = self._amplitude_batch_operator(
            boundaries=boundaries,
            shifts=amp_shifts,
        )

        # 2. Apply spatial deformation through the selected batch engine. The
        # output always uses the preallocated canonical capacity grid.
        x_shifted, y_final, sample_lengths = self._spatial_batch_operator(
            x_grid=self.base_spectra.energy_grid,
            y_density=y_amplitude,
            boundaries=boundaries,
            shifts=peak_shifts,
            freeze_origin=self.config.freeze_origin,
            output_grid=capacity_grid,
        )

        spectrum_batch = SpectrumBatch(
            energy_grid=np.asarray(x_shifted, dtype=np.float64),
            density=np.asarray(y_final, dtype=np.float64),
            sample_lengths=np.asarray(sample_lengths, dtype=np.int64),
        )
        return self._validate_generated_batch(
            batch=spectrum_batch,
            normalize=normalize,
        )

    @staticmethod
    def _validate_generated_batch(
        batch: SpectrumBatch,
        normalize: bool,
    ) -> SpectrumBatch:
        """
        Validate and optionally normalize a generated spectrum matrix.

        Parameters
        ----------
        batch : SpectrumBatch
            Generated matrix on one canonical energy grid.
        normalize : bool
            Whether each physical sample support should integrate to unity.

        Returns
        -------
        SpectrumBatch
            The same batch with density values normalized in-place when
            requested.

        Raises
        ------
        RuntimeError
            If the grid, support lengths, density values, or integrated areas
            are physically invalid.
        """
        x_grid = np.asarray(batch.energy_grid, dtype=np.float64)
        density = np.asarray(batch.density, dtype=np.float64)
        sample_lengths = np.asarray(batch.sample_lengths, dtype=np.int64)

        if x_grid.ndim != 1 or x_grid.size < 2:
            raise RuntimeError(
                "Generated batch energy grid must contain at least two points."
            )
        # Examine adjacent grid differences to enforce ordering or locate transitions.
        if np.any(x_grid < 0.0) or np.any(np.diff(x_grid) <= 0.0):
            raise RuntimeError(
                "Generated batch energy grid must be strictly increasing and "
                "non-negative."
            )
        if not np.isclose(x_grid[0], 0.0, rtol=0.0, atol=1e-14):
            raise RuntimeError(
                "Generated TARA spectra must retain the zero-energy origin."
            )
        if density.ndim != 2 or density.shape[1] != x_grid.size:
            raise RuntimeError(
                "Generated density matrix does not match the batch energy grid."
            )
        if sample_lengths.size != density.shape[0]:
            raise RuntimeError(
                "Generated sample_lengths does not match the density row count."
            )
        if np.any(sample_lengths < 2) or np.any(sample_lengths > x_grid.size):
            raise RuntimeError(
                "Generated sample support lengths lie outside the canonical grid."
            )
        if not np.all(np.isfinite(density)):
            raise RuntimeError(
                "Generated phonon-density batch contains NaN or infinite values."
            )
        if np.any(density < 0.0):
            raise RuntimeError(
                "Generated phonon-density batch contains negative values."
            )

        # Enforce the storage invariant explicitly: columns outside each row's
        # physical support are zero padding and must never affect normalization.
        columns = np.arange(x_grid.size)[None, :]
        density[columns >= sample_lengths[:, None]] = 0.0

        # Vectorized trapezoidal integration with a per-row support mask avoids
        # calling ``np.trapezoid`` once per sample while preserving exact
        # physical integration limits for shorter right-padded spectra.
        dx = np.diff(x_grid)
        interval_areas = (
            0.5
            * (density[:, :-1] + density[:, 1:])
            * dx[None, :]
        )
        interval_mask = (
            np.arange(dx.size)[None, :]
            < (sample_lengths - 1)[:, None]
        )
        areas = np.sum(
            interval_areas * interval_mask,
            axis=1,
        )

        if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
            invalid = np.flatnonzero((~np.isfinite(areas)) | (areas <= 0.0))
            raise RuntimeError(
                "Generated phonon-density batch contains non-positive or "
                f"non-finite integrals at rows {invalid.tolist()}."
            )

        if normalize:
            density /= areas[:, None]

        return SpectrumBatch(
            energy_grid=x_grid,
            density=density,
            sample_lengths=sample_lengths,
        )

    def _perturb_single_sample_arrays(
        self,
        boundaries: List[Tuple[float, float]],
        spatial_multipliers: np.ndarray,
        amplitude_multipliers: np.ndarray,
        normalize: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply both perturbation stages and return NumPy arrays directly.

        Parameters
        ----------
        boundaries : list of tuple of float
            Energy-domain boundaries used for the perturbations.
        spatial_multipliers : numpy.ndarray
            Per-domain spatial multipliers bounded in ``[-1.0, 1.0]``.
        amplitude_multipliers : numpy.ndarray
            Per-domain amplitude multipliers bounded in ``[-1.0, 1.0]``.
        normalize : bool
            If True, normalize the resulting density to unit area in-place.

        Returns
        -------
        x_shifted : numpy.ndarray
            Generated energy grid in eV.
        y_final : numpy.ndarray
            Validated density values evaluated on ``x_shifted``.
        """
        x_grid = self.base_spectra.energy_grid.copy()

        amp_shifts = amplitude_multipliers * (
            self.config.max_amplitude_shift_pct / 100.0
        )
        peak_shifts = (
            spatial_multipliers * self.config.max_energy_shift_mev
        )

        # 1. Apply amplitude variation using the selected numerical engine.
        y_amp_shifted = self._amplitude_operator(
            x_grid=x_grid,
            boundaries=boundaries,
            shifts=amp_shifts,
        )

        # 2. Apply spatial variation to the amplitude-modified spectrum.
        x_shifted, y_final = self._spatial_operator(
            x_grid=x_grid,
            y_density=y_amp_shifted,
            boundaries=boundaries,
            shifts=peak_shifts,
            freeze_origin=self.config.freeze_origin,
        )

        return self._validate_generated_arrays(
            x_shifted=x_shifted,
            y_final=y_final,
            normalize=normalize,
        )

    @staticmethod
    def _validate_generated_arrays(
        x_shifted: np.ndarray,
        y_final: np.ndarray,
        normalize: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and optionally normalize one generated spectrum in NumPy.

        This reproduces the structural checks needed by ensemble generation
        without constructing a pandas ``Series`` and ``PhononSpectra`` object
        for every stochastic sample.

        Parameters
        ----------
        x_shifted : numpy.ndarray
            Generated energy grid in eV.
        y_final : numpy.ndarray
            Generated density values.
        normalize : bool
            Whether to normalize the density to unit integrated area.

        Returns
        -------
        x_shifted : numpy.ndarray
            Validated one-dimensional energy grid.
        y_final : numpy.ndarray
            Validated and optionally normalized density values.

        Raises
        ------
        RuntimeError
            If arrays are malformed, non-finite, negative, non-monotonic, do
            not start at the TARA origin, or have non-positive integrated area.
        """
        x_shifted = np.asarray(x_shifted, dtype=np.float64)
        y_final = np.asarray(y_final, dtype=np.float64)

        if x_shifted.ndim != 1 or y_final.ndim != 1:
            raise RuntimeError(
                "Generated spectrum arrays must be one-dimensional."
            )
        if x_shifted.size != y_final.size or x_shifted.size < 2:
            raise RuntimeError(
                "Generated energy and density arrays must have equal length "
                "and contain at least two points."
            )
        if not np.all(np.isfinite(x_shifted)) or not np.all(np.isfinite(y_final)):
            raise RuntimeError(
                "Generated spectrum contains NaN or infinite values."
            )
        # Examine adjacent grid differences to enforce ordering or locate transitions.
        if np.any(x_shifted < 0.0) or np.any(np.diff(x_shifted) <= 0.0):
            raise RuntimeError(
                "Generated energy grid must be strictly increasing and non-negative."
            )
        if not np.isclose(x_shifted[0], 0.0, rtol=0.0, atol=1e-14):
            raise RuntimeError(
                "Generated TARA spectra must retain the zero-energy origin."
            )
        if np.any(y_final < 0.0):
            raise RuntimeError(
                "Generated phonon density contains negative values."
            )

        # Integrate against the physical coordinate grid, accounting for nonuniform spacing.
        area = float(np.trapezoid(y_final, x_shifted))
        if not np.isfinite(area) or area <= 0.0:
            raise RuntimeError(
                "Generated phonon density must have a positive finite integral."
            )

        if normalize:
            # The mathematical operators already return owned output arrays in
            # the normal path, so in-place scaling avoids another large allocation.
            y_final /= area

        return x_shifted, y_final

    def _perturb_single_sample(
        self,
        boundaries: List[Tuple[float, float]],
        spatial_multipliers: np.ndarray,
        amplitude_multipliers: np.ndarray,
        auto_normalize: bool = True,
    ) -> PhononSpectra:
        """
        Apply perturbations and return one user-facing ``PhononSpectra`` object.

        This wrapper is retained for deterministic previews and explicit
        single-spectrum use. Full ensemble generation uses the NumPy-first
        ``_perturb_single_sample_arrays`` path instead.

        Parameters
        ----------
        boundaries : list of tuple of float
            Energy-domain boundaries used for the perturbations.
        spatial_multipliers : numpy.ndarray
            Per-domain spatial multipliers bounded in ``[-1.0, 1.0]``.
        amplitude_multipliers : numpy.ndarray
            Per-domain amplitude multipliers bounded in ``[-1.0, 1.0]``.
        auto_normalize : bool, default=True
            Whether the returned ``PhononSpectra`` normalizes itself.

        Returns
        -------
        PhononSpectra
            Perturbed phonon spectrum.
        """
        x_shifted, y_final = self._perturb_single_sample_arrays(
            boundaries=boundaries,
            spatial_multipliers=spatial_multipliers,
            amplitude_multipliers=amplitude_multipliers,
            normalize=False,
        )

        new_series = pd.Series(y_final, index=x_shifted)
        return PhononSpectra(
            new_series,
            auto_normalize=auto_normalize,
        )


class TaraSampler:
    """Generate a vibrational, translational, or joint sampling ensemble.

    Parameters
    ----------
    config : TaraConfig
        Vibrational bounds and optional diffusion settings.
    spectra : PhononSpectra, optional
        Explicit vibrational reference. When omitted, use the continuous
        spectrum of ``material``.
    amplitude_method : {"precise", "fast"}, default="fast"
        Amplitude perturbation method for vibration sampling.
    spatial_method : {"diffeomorphic", "crossfade"}, default="diffeomorphic"
        Spatial perturbation method for vibration sampling.
    material : MaterialScatteringModel, optional
        Reference material. An Egelstaff--Schofield liquid is required when
        diffusion sampling is enabled.
    """

    def __init__(
        self,
        config: TaraConfig,
        spectra: PhononSpectra | None = None,
        amplitude_method: str = "fast",
        spatial_method: str = "diffeomorphic",
        *,
        material: MaterialScatteringModel | None = None,
    ) -> None:
        self.config = config
        self._diffusion_sampler = None
        self._vibration_sampler = None

        if config.diffusion_enabled:
            if material is None:
                raise ValueError("Diffusion sampling requires a reference material.")
            seed = config.random_seed
            if config.sampling_method == "monte_carlo":
                # Use the historical ACE diffusion stream independently of VDOS draws.
                seed = int(np.random.SeedSequence([seed, 0xD1FF]).generate_state(1)[0])
            self._diffusion_sampler = DiffusionSampler(
                material, config.diffusion, seed=seed, sampling_method=config.sampling_method,
            )

        if config.vibration_enabled:
            if spectra is None:
                if material is None:
                    raise ValueError("Vibration sampling requires a reference spectrum or material.")
                spectra = material.dynamics.continuous
            self._vibration_sampler = VibrationSampler(
                config, spectra, amplitude_method, spatial_method,
            )

    def __getattr__(self, name: str):
        """Preserve access to vibration diagnostics on existing sampler instances."""
        vibration = object.__getattribute__(self, "_vibration_sampler")
        if vibration is None:
            raise AttributeError(name)
        return getattr(vibration, name)

    def generate_preview(self) -> SpectraEnsemble:
        """Return deterministic extrema for the configured vibration bounds.

        Returns
        -------
        SpectraEnsemble
            The vibrational reference and preview spectra.

        Raises
        ------
        ValueError
            If vibrational perturbations are inactive.
        """
        if self._vibration_sampler is None:
            raise ValueError("A preview requires active vibrational perturbations.")
        return self._vibration_sampler.generate_preview()

    def generate_ensemble(self, batch_size: int = 512) -> SpectraEnsemble | MaterialEnsemble:
        """Generate and return all stochastic samples for this experiment.

        Parameters
        ----------
        batch_size : int, default=512
            Maximum number of spectra evaluated together during vibration sampling.

        Returns
        -------
        SpectraEnsemble or MaterialEnsemble
            The reference and samples. With diffusion enabled, each material
            realization uses the matching vibrational sample when available.
        """
        spectra_ensemble = None
        if self._vibration_sampler is not None:
            spectra_ensemble = self._vibration_sampler.generate_ensemble(batch_size=batch_size)
        if self._diffusion_sampler is not None:
            return self._diffusion_sampler.generate_ensemble(
                self.config.n_samples, spectra_ensemble=spectra_ensemble,
            )
        return spectra_ensemble

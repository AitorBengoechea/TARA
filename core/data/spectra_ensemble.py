"""
Core data structures for phonon-spectrum ensembles and sampling provenance.
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from core.data.input_pdos import PhononSpectra


@dataclass(frozen=True)
class EnsembleGeneration:
    """
    Store an immutable snapshot of the settings used to generate an ensemble.

    Parameters
    ----------
    amplitude_method : str
        Amplitude perturbation implementation selected by ``TaraSampler``.
    spatial_method : str
        Spatial perturbation implementation selected by ``TaraSampler``.
    max_energy_shift_mev : float
        Maximum configured spatial perturbation in meV.
    max_amplitude_shift_pct : float
        Maximum configured amplitude perturbation in percent.
    n_samples : int
        Number of stochastic samples requested for the ensemble.
    grid_partition_method : str
        Energy-domain partitioning strategy.
    manual_energy_ranges : tuple of tuple of float or None
        User-specified energy domains in eV when manual partitioning is used.
    domain_splitting_strategy : str
        Strategy controlling domain-boundary variation across the ensemble.
    correlation_strategy : str
        Strategy controlling correlations between energy domains.
    variable_coupling : str
        Strategy controlling coupling between amplitude and spatial variables.
    shared_grid_structure : bool
        Whether amplitude and spatial perturbations share the same domain layout.
    freeze_origin : bool
        Whether the acoustic origin was constrained during spatial perturbation.
    automatic_partition_prominence_pct : float, optional
        Automatic valley-prominence threshold as a percentage of maximum
        density. Default is 1.0 for backward-compatible metadata records.
    sampling_method : str, optional
        Parameter-space design used for the stochastic ensemble. Defaults to
        monte_carlo for compatibility with older manually assembled records.
    """

    # Perturbation methods, bounds, and sample count.
    amplitude_method: str
    spatial_method: str
    max_energy_shift_mev: float
    max_amplitude_shift_pct: float
    n_samples: int

    # Domain topology and coupling.
    grid_partition_method: str
    manual_energy_ranges: Optional[Tuple[Tuple[float, float], ...]]
    domain_splitting_strategy: str
    correlation_strategy: str
    variable_coupling: str
    shared_grid_structure: bool
    freeze_origin: bool
    automatic_partition_prominence_pct: float = 1.0

    # Random design.
    sampling_method: str = "monte_carlo"


@dataclass(frozen=True)
class SampleGeneration:
    """
    Describe the physical perturbations used to create one sample.

    ``SampleGeneration`` is a lightweight user-facing view. The ensemble does
    not store one instance per sample; records are reconstructed lazily from the
    compact NumPy sampling blocks only when requested.

    Parameters
    ----------
    boundaries : tuple of tuple of float
        Energy-domain boundaries in eV used for this sample.
    amplitude_shift_pct : tuple of float
        Actual amplitude perturbation applied to each domain, in percent.
    energy_shift_mev : tuple of float
        Actual spatial perturbation applied to each domain, in meV.
    """

    # Energy domains and their perturbations.
    boundaries: Tuple[Tuple[float, float], ...]
    amplitude_shift_pct: Tuple[float, ...]
    energy_shift_mev: Tuple[float, ...]


@dataclass(frozen=True)
class SamplingBlock:
    """
    Store the compact stochastic state for one topology sub-ensemble.

    A topology strategy assigns one set of domain boundaries to a contiguous
    sample range. Those boundaries are therefore stored once per block rather
    than once per sample. Random multipliers remain as NumPy arrays in the same
    vectorized form used during generation.

    Parameters
    ----------
    start_index : int
        Inclusive global sample index for the block.
    end_index : int
        Exclusive global sample index for the block.
    boundaries : numpy.ndarray
        Domain boundaries with shape ``(n_domains, 2)`` in eV.
    spatial_multipliers : numpy.ndarray
        Spatial stochastic state with shape ``(n_samples, n_domains)`` or
        ``(n_samples, 1)`` for broadcast-correlated sampling.
    amplitude_multipliers : numpy.ndarray or None, optional
        Independent amplitude stochastic state. ``None`` means amplitude and
        spatial perturbations use the same underlying stochastic state.

    Raises
    ------
    ValueError
        If indices, boundaries, or multiplier-array dimensions are invalid.
    """

    # Sample range and energy domains.
    start_index: int
    end_index: int
    boundaries: np.ndarray

    # Perturbation multipliers.
    spatial_multipliers: np.ndarray
    amplitude_multipliers: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate ensemble generation metadata and its sample coverage.

        Raises
        ------
        ValueError
            If the block range, domain bounds, or multiplier arrays are invalid.
        """
        # Track each topology block with a nonempty half-open range of sample indices.
        if self.start_index < 0 or self.end_index <= self.start_index:
            raise ValueError(
                "SamplingBlock requires 0 <= start_index < end_index."
            )

        boundaries = np.asarray(self.boundaries, dtype=np.float64)
        spatial = np.asarray(self.spatial_multipliers, dtype=np.float64)

        self._validate_boundaries(boundaries)

        # Use the block range and domain count to check the multiplier layout.
        n_samples = self.end_index - self.start_index
        n_domains = boundaries.shape[0]

        self._validate_state_array(
            spatial,
            n_samples=n_samples,
            n_domains=n_domains,
            name="spatial_multipliers",
        )

        # Validate a separate amplitude design only when the block stores one.
        amplitude = self.amplitude_multipliers
        if amplitude is not None:
            amplitude = np.asarray(amplitude, dtype=np.float64)
            self._validate_state_array(
                amplitude,
                n_samples=n_samples,
                n_domains=n_domains,
                name="amplitude_multipliers",
            )

        object.__setattr__(self, "boundaries", self._freeze_array(boundaries))
        object.__setattr__(self, "spatial_multipliers", self._freeze_array(spatial))
        if amplitude is not None:
            amplitude = self._freeze_array(amplitude)
        object.__setattr__(self, "amplitude_multipliers", amplitude)

    @staticmethod
    def _freeze_array(values: np.ndarray) -> np.ndarray:
        """Take ownership of immutable sampling provenance.

        Parameters
        ----------
        values : numpy.ndarray
            Validated boundaries or perturbation multipliers.

        Returns
        -------
        numpy.ndarray
            Read-only, contiguous float64 copy.
        """
        # The sampler may reuse its working arrays after construction.
        stored = np.array(values, dtype=np.float64, order="C", copy=True)
        stored.setflags(write=False)
        return stored

    @staticmethod
    def _validate_boundaries(boundaries: np.ndarray) -> None:
        """Validate the bounds of each recorded energy domain.

        Parameters
        ----------
        boundaries : numpy.ndarray
            Lower and upper energy bound of each domain.

        Raises
        ------
        ValueError
            If a bound is nonfinite, unordered within a pair, or has an invalid shape.
        """
        # Each domain needs finite lower and upper energy bounds in increasing order.
        if boundaries.ndim != 2 or boundaries.shape[1] != 2:
            raise ValueError("boundaries must have shape (n_domains, 2).")
        if boundaries.shape[0] == 0:
            raise ValueError("SamplingBlock requires at least one domain.")
        if not np.all(np.isfinite(boundaries)):
            raise ValueError("SamplingBlock boundaries must be finite.")
        if np.any(boundaries[:, 0] >= boundaries[:, 1]):
            raise ValueError("Every SamplingBlock boundary must satisfy start < end.")

    @staticmethod
    def _validate_state_array(
        state: np.ndarray,
        n_samples: int,
        n_domains: int,
        name: str,
    ) -> None:
        """
        Validate a compact multiplier matrix.

        Parameters
        ----------
        state : numpy.ndarray
            Multiplier state to validate.
        n_samples : int
            Number of samples represented by the block.
        n_domains : int
            Number of topology domains represented by the block.
        name : str
            Field name used in validation errors.

        Raises
        ------
        ValueError
            If the state cannot represent the block samples and domains.
        """
        if state.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array.")
        if state.shape[0] != n_samples:
            raise ValueError(
                f"{name} row count must match the SamplingBlock sample count."
            )
        if state.shape[1] not in {1, n_domains}:
            raise ValueError(
                f"{name} must have either 1 or n_domains columns."
            )
        if not np.all(np.isfinite(state)):
            raise ValueError(f"{name} must contain only finite values.")
        if np.any(np.abs(state) > 1.0 + 1e-12):
            raise ValueError(f"{name} values must remain inside [-1.0, 1.0].")


class SpectraEnsemble:
    """
    Store a NumPy-first phonon ensemble and its compact sampling provenance.

    The main storage is a dense two-dimensional density matrix with samples on
    rows and energy values on columns. All samples share the same left boundary
    and canonical grid prefix. Samples that originally terminate earlier are
    represented by right-side zeros on the shared longest grid.

    Generation provenance is stored as compact ``SamplingBlock`` objects.
    Human-readable per-sample metadata is created lazily only when requested,
    avoiding Python-object expansion during the normal sampling/analyzer path.

    Parameters
    ----------
    energy_grid : numpy.ndarray
        Canonical ensemble energy grid in eV.
    density_matrix : numpy.ndarray
        Density matrix with shape ``(n_samples, n_energy)``. Preview rows
        may deliberately retain non-unit integrals.
    generation : EnsembleGeneration
        Immutable snapshot of the sampler settings.
    sample_lengths : numpy.ndarray
        Number of physical grid points belonging to each sample before right-side
        zero padding.
    sampling_blocks : sequence of SamplingBlock
        Compact stochastic provenance covering every sample exactly once.
    reference : PhononSpectra, optional
        Baseline spectrum. It is stored separately and is not counted as a
        stochastic sample.
    auto_normalize_samples : bool, default=True
        Whether lazy PhononSpectra reconstruction normalizes individual rows.
        Use False for deterministic extrema whose physical integrals must remain
        unchanged.

    Attributes
    ----------
    reference : PhononSpectra or None
        Baseline spectrum associated with the ensemble.
    generation : EnsembleGeneration
        Immutable ensemble-level generation settings.

    Raises
    ------
    ValueError
        If the grid, matrix dimensions, sample count, or block coverage are
        inconsistent.
    TypeError
        If generation or sampling-block objects have invalid types.
    """

    def __init__(
        self,
        energy_grid: np.ndarray,
        density_matrix: np.ndarray,
        generation: EnsembleGeneration,
        sample_lengths: np.ndarray,
        sampling_blocks: Sequence[SamplingBlock],
        reference: Optional[PhononSpectra] = None,
        auto_normalize_samples: bool = True,
    ) -> None:
        """Initialize a stochastic spectrum ensemble on a shared energy grid.

        Parameters
        ----------
        energy_grid : numpy.ndarray
            Shared energy coordinates in eV.
        density_matrix : numpy.ndarray
            Spectral densities with samples as rows and energies as columns.
        generation : EnsembleGeneration
            Sampling configuration and expected stochastic sample count.
        sample_lengths : numpy.ndarray
            Number of grid points belonging to each sample before padding.
        sampling_blocks : sequence of SamplingBlock
            Topology and perturbation records covering the stochastic rows.
        reference : PhononSpectra or None, optional
            Unperturbed spectrum stored separately from the stochastic rows.
        auto_normalize_samples : bool, optional
            Normalize spectra when reconstructing individual samples.
            Default is True.

        Raises
        ------
        TypeError
            If the generation record or any sampling block has the wrong type.
        ValueError
            If the grid, matrix, sample lengths, or block coverage are invalid.
        """
        blocks = tuple(sampling_blocks)
        self._validate_generation_metadata(generation, blocks)

        energy_grid = np.asarray(energy_grid, dtype=np.float64)
        density_matrix = np.asarray(density_matrix, dtype=np.float64)
        sample_lengths = np.asarray(sample_lengths, dtype=np.int64)

        n_samples = generation.n_samples
        self._validate_sample_arrays(
            energy_grid, density_matrix, sample_lengths, n_samples,
        )
        self._validate_block_coverage(blocks, n_samples)

        # Own the coordinate and length arrays while retaining the sampler's matrix.
        self._energy_grid = np.ascontiguousarray(energy_grid).copy()
        self._density_matrix = density_matrix
        self._sample_lengths = np.ascontiguousarray(sample_lengths).copy()
        self._sampling_blocks = blocks

        self.reference = reference
        self.generation = generation
        self._auto_normalize_samples = bool(auto_normalize_samples)

        # Build pandas views only when a caller requests them.
        self._pdos_cache: Optional[pd.DataFrame] = None
        self._sample_info_cache: Optional[pd.DataFrame] = None

    @staticmethod
    def _validate_generation_metadata(
        generation: EnsembleGeneration,
        blocks: Sequence[SamplingBlock],
    ) -> None:
        """Validate the types of the ensemble provenance records.

        Parameters
        ----------
        generation : EnsembleGeneration
            Record describing the stochastic ensemble.
        blocks : sequence of SamplingBlock
            Topology and perturbation records for the sample rows.

        Raises
        ------
        TypeError
            If the generation record or any sampling block has the wrong type.
        """
        if not isinstance(generation, EnsembleGeneration):
            raise TypeError("generation must be an EnsembleGeneration instance.")

        if any(not isinstance(block, SamplingBlock) for block in blocks):
            raise TypeError(
                "Every sampling_blocks entry must be a SamplingBlock instance."
            )

    @staticmethod
    def _validate_sample_arrays(
        energy_grid: np.ndarray,
        density_matrix: np.ndarray,
        sample_lengths: np.ndarray,
        n_samples: int,
    ) -> None:
        """Validate the shared grid and its alignment with the sample arrays.

        Parameters
        ----------
        energy_grid : numpy.ndarray
            Shared energy coordinates in eV.
        density_matrix : numpy.ndarray
            Spectral densities with samples as rows and energies as columns.
        sample_lengths : numpy.ndarray
            Number of valid grid points in each sample row before padding.
        n_samples : int
            Expected number of stochastic samples.

        Raises
        ------
        ValueError
            If the grid is invalid or the arrays do not match the grid and
            expected sample count.
        """
        # Every spectrum occupies a prefix of the same energy grid.
        if energy_grid.ndim != 1 or energy_grid.size < 2:
            raise ValueError("energy_grid must be one-dimensional with at least 2 points.")
        if not np.all(np.isfinite(energy_grid)):
            raise ValueError("energy_grid must contain only finite values.")
        if np.any(energy_grid < 0.0) or np.any(np.diff(energy_grid) <= 0.0):
            raise ValueError("energy_grid must be strictly increasing and non-negative.")

        # The matrix needs one row per sample and one column per grid point.
        if density_matrix.ndim != 2:
            raise ValueError("density_matrix must be two-dimensional.")
        if density_matrix.shape[1] != energy_grid.size:
            raise ValueError("density_matrix column count must match energy_grid length.")
        if density_matrix.shape[0] != n_samples:
            raise ValueError("EnsembleGeneration.n_samples must match density_matrix rows.")

        # Each length marks the end of the valid data in its padded row.
        if sample_lengths.ndim != 1 or sample_lengths.size != n_samples:
            raise ValueError(
                "sample_lengths must contain exactly one entry per ensemble sample."
            )
        if np.any(sample_lengths < 2) or np.any(sample_lengths > energy_grid.size):
            raise ValueError(
                "Every sample length must be between 2 and the canonical grid length."
            )

    @staticmethod
    def _validate_block_coverage(
        blocks: Sequence[SamplingBlock],
        n_samples: int,
    ) -> None:
        """
        Validate contiguous one-to-one sample coverage by sampling blocks.

        Parameters
        ----------
        blocks : sequence of SamplingBlock
            Blocks ordered by global sample index.
        n_samples : int
            Total number of ensemble samples.

        Raises
        ------
        ValueError
            If blocks overlap, leave gaps, or do not cover the full ensemble.
        """
        if not blocks:
            raise ValueError("SpectraEnsemble requires at least one SamplingBlock.")

        expected_start = 0
        for block in blocks:
            if block.start_index != expected_start:
                raise ValueError(
                    "SamplingBlock ranges must be contiguous, ordered, and non-overlapping."
                )
            expected_start = block.end_index

        if expected_start != n_samples:
            raise ValueError(
                "SamplingBlock ranges must cover every ensemble sample exactly once."
            )

    @property
    def n_samples(self) -> int:
        """
        Return the number of stochastic spectra.

        Returns
        -------
        int
            Number of ensemble samples.
        """
        return self._density_matrix.shape[0]

    @property
    def energy_grid(self) -> np.ndarray:
        """
        Return the canonical ensemble energy grid.

        Returns
        -------
        numpy.ndarray
            Shared energy grid in eV.
        """
        return self._energy_grid.copy()

    @property
    def sampling_blocks(self) -> Tuple[SamplingBlock, ...]:
        """
        Return compact topology and stochastic provenance blocks.

        Returns
        -------
        tuple of SamplingBlock
            Immutable block-level sampling trace.
        """
        return self._sampling_blocks

    @property
    def pdos(self) -> pd.DataFrame:
        """
        Return the ensemble PDOS table.

        Rows correspond to sample numbers and columns correspond to energy-grid
        values in eV. The DataFrame is created only on first access.

        Returns
        -------
        pandas.DataFrame
            Ensemble density matrix using samples as rows and energies as columns.
        """
        if self._pdos_cache is None:
            sample_index = pd.Index(range(self.n_samples), name="Sample_Index")
            self._pdos_cache = pd.DataFrame(
                self._density_matrix,
                index=sample_index,
                columns=pd.Index(self._energy_grid, name="Energy_eV"),
                copy=False,
            )
        return self._pdos_cache

    @property
    def sample_info(self) -> pd.DataFrame:
        """
        Return human-readable per-sample generation provenance.

        The table is generated lazily from the compact block state. Boundaries
        are reported in eV, amplitude perturbations in percent, and spatial
        perturbations in meV.

        Returns
        -------
        pandas.DataFrame
            Per-sample physical construction metadata.
        """
        if self._sample_info_cache is None:
            records = [
                self.sample_generation(sample_index)
                for sample_index in range(self.n_samples)
            ]
            self._sample_info_cache = pd.DataFrame(
                {
                    "boundaries": [record.boundaries for record in records],
                    "amplitude_shift_pct": [
                        record.amplitude_shift_pct for record in records
                    ],
                    "energy_shift_mev": [
                        record.energy_shift_mev for record in records
                    ],
                },
                index=pd.Index(range(self.n_samples), name="Sample_Index"),
            )
        return self._sample_info_cache.copy(deep=True)

    def sample_generation(self, index: int) -> SampleGeneration:
        """
        Reconstruct the physical perturbations used for one sample.

        Parameters
        ----------
        index : int
            Positional sample index. Negative indices follow normal Python
            sequence semantics.

        Returns
        -------
        SampleGeneration
            Boundaries and physical perturbations for the requested sample.

        Raises
        ------
        IndexError
            If ``index`` lies outside the ensemble.
        """
        sample_index = self._normalize_index(index)
        block = self._find_block(sample_index)
        local_index = sample_index - block.start_index
        n_domains = block.boundaries.shape[0]

        spatial = self._expand_state_row(
            block.spatial_multipliers,
            local_index=local_index,
            n_domains=n_domains,
        )

        # ``None`` denotes coupled variables, so the exact same stochastic
        # state drives both amplitude and spatial perturbations.
        amplitude_state = (
            block.spatial_multipliers
            if block.amplitude_multipliers is None
            else block.amplitude_multipliers
        )
        amplitude = self._expand_state_row(
            amplitude_state,
            local_index=local_index,
            n_domains=n_domains,
        )

        amplitude_pct = amplitude * self.generation.max_amplitude_shift_pct
        energy_mev = spatial * self.generation.max_energy_shift_mev

        return SampleGeneration(
            boundaries=tuple(
                (float(start), float(end))
                for start, end in block.boundaries
            ),
            amplitude_shift_pct=tuple(float(value) for value in amplitude_pct),
            energy_shift_mev=tuple(float(value) for value in energy_mev),
        )

    @staticmethod
    def _expand_state_row(
        state: np.ndarray,
        local_index: int,
        n_domains: int,
    ) -> np.ndarray:
        """
        Return one block row expanded to the full domain count.

        Parameters
        ----------
        state : numpy.ndarray
            Compact multiplier matrix with one or ``n_domains`` columns.
        local_index : int
            Row index within the sampling block.
        n_domains : int
            Number of domains represented by the block.

        Returns
        -------
        numpy.ndarray
            One-dimensional multiplier view with length ``n_domains``.
        """
        row = state[local_index]
        if row.size == n_domains:
            return row
        return np.broadcast_to(row, (n_domains,))

    def _find_block(self, sample_index: int) -> SamplingBlock:
        """
        Locate the compact provenance block containing one sample.

        Parameters
        ----------
        sample_index : int
            Normalized global sample index.

        Returns
        -------
        SamplingBlock
            Block containing ``sample_index``.
        """
        for block in self._sampling_blocks:
            if block.start_index <= sample_index < block.end_index:
                return block

        raise RuntimeError(
            "Internal error: sampling-block coverage is inconsistent."
        )

    def _normalize_index(self, index: int) -> int:
        """
        Normalize one positional sample index.

        Parameters
        ----------
        index : int
            Requested positional index.

        Returns
        -------
        int
            Non-negative normalized index.

        Raises
        ------
        IndexError
            If the index is outside the ensemble.
        """
        normalized_index = int(index)
        if normalized_index < 0:
            normalized_index += self.n_samples
        if normalized_index < 0 or normalized_index >= self.n_samples:
            raise IndexError("SpectraEnsemble sample index out of range.")
        return normalized_index

    def sample(self, index: int) -> PhononSpectra:
        """
        Reconstruct one ensemble row as a ``PhononSpectra`` object.

        ``PhononSpectra`` construction is intentionally lazy. The stored sample
        length removes right-side ensemble padding before reconstruction, so the
        original physical support and, when requested, unnormalized preview
        integrals are preserved exactly.

        Parameters
        ----------
        index : int
            Positional sample index.

        Returns
        -------
        PhononSpectra
            Requested sample on the canonical ensemble grid.
        """
        normalized_index = self._normalize_index(index)
        n_points = int(self._sample_lengths[normalized_index])
        series = pd.Series(
            self._density_matrix[normalized_index, :n_points].copy(),
            index=self._energy_grid[:n_points].copy(),
        )
        return PhononSpectra(
            series, auto_normalize=self._auto_normalize_samples
        )

    def __len__(self) -> int:
        """Return the number of stochastic spectra in the ensemble.

        Returns
        -------
        int
            Integer count or identifier determined by this operation.
        """
        return self.n_samples

    def __iter__(self) -> Iterator[PhononSpectra]:
        """Iterate lazily over samples as ``PhononSpectra`` objects.

        Returns
        -------
        Iterator[PhononSpectra]
            Calculated result in the declared return type.
        """
        for sample_index in range(self.n_samples):
            yield self.sample(sample_index)

    def __getitem__(
        self,
        index: Union[int, slice],
    ) -> Union[PhononSpectra, List[PhononSpectra]]:
        """
        Return one sample or a list of samples by positional index.

        Parameters
        ----------
        index : int or slice
            Positional sample selector.

        Returns
        -------
        PhononSpectra or list of PhononSpectra
            Selected sample or samples.
        """
        if isinstance(index, slice):
            indices = range(*index.indices(self.n_samples))
            return [self.sample(i) for i in indices]

        if isinstance(index, (int, np.integer)):
            return self.sample(int(index))

        raise TypeError("SpectraEnsemble indices must be integers or slices.")

    def __repr__(self) -> str:
        """Return a compact developer-facing ensemble summary.

        Returns
        -------
        str
            Serialized text or identifier produced by this operation.
        """
        return (
            "SpectraEnsemble("
            f"n_samples={self.n_samples}, "
            f"n_energy={self._density_matrix.shape[1]}, "
            f"energy_range=({self._energy_grid[0]:.6g}, "
            f"{self._energy_grid[-1]:.6g}) eV, "
            f"n_sampling_blocks={len(self._sampling_blocks)}, "
            f"amplitude_method={self.generation.amplitude_method!r}, "
            f"spatial_method={self.generation.spatial_method!r}"
            ")"
        )

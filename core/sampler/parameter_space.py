"""Centralized, backend-neutral stochastic parameter-space designs.

Each topology sub-ensemble is a separate statistical universe. Randomized
Latin hypercubes stratify every *independent* active dimension once per
sub-ensemble. Named streams make VDOS and diffusion coordinates align by row
without coupling their random variables or depending on generation order.
"""

from __future__ import annotations

from dataclasses import dataclass
import zlib

import numpy as np


SAMPLING_METHODS = frozenset({'latin_hypercube', 'monte_carlo'})


@dataclass(frozen=True)
class ParameterSpaceSampler:
    """Produce signed uniform multipliers for named parameter groups.

    ``monte_carlo`` with no supplied ``rng`` uses NumPy's legacy global
    generator to preserve VDOS sequences. Supply a local generator for
    diffusion Monte Carlo. Latin hypercube sampling uses group-specific
    streams independently of the global generator.

    Parameters
    ----------
    sampling_method : {"latin_hypercube", "monte_carlo"}, optional
        Design used to generate multipliers. Default is "latin_hypercube".
    seed : int, optional
        Seed for the named Latin-hypercube streams. Default is 0.
    """

    sampling_method: str = 'latin_hypercube'
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the design method and seed.

        Raises
        ------
        ValueError
            If the design is unsupported or the seed is outside [0, 2**32).
        """
        if self.sampling_method not in SAMPLING_METHODS:
            raise ValueError(f'sampling_method must be one of {sorted(SAMPLING_METHODS)}.')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32).')

    def generate(
        self,
        n_samples: int,
        n_dimensions: int,
        *,
        subensemble_id: int = 0,
        group: str,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw signed multipliers on one named parameter stream.

        Parameters
        ----------
        n_samples, n_dimensions : int
            Number of sample rows and independent design columns.
        subensemble_id : int, optional
            Topology block index within the ensemble.
        group : str
            Stable name of the parameter stream.
        rng : numpy.random.Generator or None, optional
            Local Monte Carlo stream. None uses the legacy NumPy generator.

        Returns
        -------
        numpy.ndarray
            Matrix of shape (n_samples, n_dimensions) with values in [-1, 1].

        Raises
        ------
        ValueError
            If the sample count, dimension count, topology index, or name is invalid.
        """
        self._validate_design_request(n_samples, n_dimensions, subensemble_id, group)
        if n_dimensions == 0:
            return np.empty((n_samples, 0), dtype=np.float64)

        if self.sampling_method == 'monte_carlo':
            if rng is None:
                return np.random.uniform(-1., 1., size=(n_samples, n_dimensions))
            return rng.uniform(-1., 1., size=(n_samples, n_dimensions))

        return self._latin_hypercube(n_samples, n_dimensions, subensemble_id, group)

    @staticmethod
    def _validate_design_request(n_samples, n_dimensions, subensemble_id, group) -> None:
        """Validate the dimensions and identity of a parameter-space design.

        Parameters
        ----------
        n_samples, n_dimensions : int
            Number of sample rows and independent design columns.
        subensemble_id : int
            Topology block index within the ensemble.
        group : str
            Stable name of the parameter stream.

        Raises
        ------
        ValueError
            If the matrix dimensions, topology index, or stream name are invalid.
        """
        if type(n_samples) is not int or n_samples <= 0:
            raise ValueError('n_samples must be a positive integer.')
        if type(n_dimensions) is not int or n_dimensions < 0:
            raise ValueError('n_dimensions must be a nonnegative integer.')
        if type(subensemble_id) is not int or subensemble_id < 0:
            raise ValueError('subensemble_id must be a nonnegative integer.')
        if not isinstance(group, str) or not group:
            raise ValueError('group must be a nonempty string.')

    def _latin_hypercube(self, n_samples, n_dimensions, subensemble_id, group) -> np.ndarray:
        """Draw independent stratified coordinates for each design column.

        Parameters
        ----------
        n_samples, n_dimensions : int
            Number of sample rows and independent design columns.
        subensemble_id : int
            Topology block index within the ensemble.
        group : str
            Stable name of the parameter stream.

        Returns
        -------
        numpy.ndarray
            Signed Latin-hypercube coordinates with one row per sample.
        """
        # CRC32 is intentionally stable across processes, unlike Python hash().
        group_code = zlib.crc32(group.encode('utf-8'))
        local_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, subensemble_id, group_code])
        )
        output = np.empty((n_samples, n_dimensions), dtype=np.float64)
        for column in range(n_dimensions):
            # Visit each stratum once, then jitter within it.
            permutation = local_rng.permutation(n_samples)
            jitter = local_rng.random(n_samples)
            output[:, column] = 2.0 * (permutation + jitter) / n_samples - 1.0
        return output

    def generate_domains(
        self,
        n_samples: int,
        n_domains: int,
        *,
        correlation_strategy: str,
        subensemble_id: int,
        group: str,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw full or broadcast multipliers for the configured correlation.

        Parameters
        ----------
        n_samples, n_domains : int
            Number of sample rows and domain columns.
        correlation_strategy : {"independent", "random_group"}
            Whether each domain gets a separate coordinate or shares one per row.
        subensemble_id : int
            Index of the topology block.
        group : str
            Stable name of the parameter stream.
        rng : numpy.random.Generator or None, optional
            Local Monte Carlo stream. None uses the legacy NumPy generator.

        Returns
        -------
        numpy.ndarray
            Multiplier matrix with one row per sample and one column per domain.

        Raises
        ------
        ValueError
            If the domain count or correlation strategy is invalid.
        """
        if type(n_domains) is not int or n_domains <= 0:
            raise ValueError('n_domains must be a positive integer.')
        if correlation_strategy not in ('independent', 'random_group'):
            raise ValueError('Unsupported correlation_strategy.')
        dimensions = n_domains if correlation_strategy == 'independent' else 1
        coordinates = self.generate(
            n_samples, dimensions, subensemble_id=subensemble_id,
            group=group, rng=rng,
        )
        if correlation_strategy == 'random_group':
            return np.broadcast_to(coordinates, (n_samples, n_domains))
        return coordinates

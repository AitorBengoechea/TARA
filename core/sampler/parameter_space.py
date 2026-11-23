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
    generator to preserve VDOS sequences. Supply a local Generator for the
    original diffusion Monte Carlo behavior. ``latin_hypercube`` uses
    group-specific SeedSequence streams, independently of the global RNG.
    """

    sampling_method: str = 'latin_hypercube'
    seed: int = 0

    def __post_init__(self) -> None:
        """Require a supported design and an integer seed in [0, 2**32)."""
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
        """Generate one signed matrix with shape ``(n_samples, n_dimensions)``.

        In LHS mode, each column occupies every equal-probability stratum
        exactly once; different named groups and sub-ensembles get separate
        deterministic streams. Zero-dimensional requests return empty arrays.
        """
        if type(n_samples) is not int or n_samples <= 0:
            raise ValueError('n_samples must be a positive integer.')
        if type(n_dimensions) is not int or n_dimensions < 0:
            raise ValueError('n_dimensions must be a nonnegative integer.')
        if type(subensemble_id) is not int or subensemble_id < 0:
            raise ValueError('subensemble_id must be a nonnegative integer.')
        if not isinstance(group, str) or not group:
            raise ValueError('group must be a nonempty string.')
        if n_dimensions == 0:
            return np.empty((n_samples, 0), dtype=np.float64)

        if self.sampling_method == 'monte_carlo':
            if rng is None:
                return np.random.uniform(-1., 1., size=(n_samples, n_dimensions))
            return rng.uniform(-1., 1., size=(n_samples, n_dimensions))

        # CRC32 is intentionally stable across processes, unlike Python hash().
        group_code = zlib.crc32(group.encode('utf-8'))
        local_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, subensemble_id, group_code])
        )
        output = np.empty((n_samples, n_dimensions), dtype=np.float64)
        for column in range(n_dimensions):
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
        """Generate a full or broadcast domain matrix from active dimensions."""
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

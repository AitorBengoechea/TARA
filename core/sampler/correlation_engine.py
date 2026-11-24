"""
Correlation routing engine for TARA stochastic sampling.
"""

from abc import ABC, abstractmethod

import numpy as np

from core.config import TaraConfig


class CorrelationStrategy(ABC):
    """
    Define how legacy uniform draws are shared across energy domains.

    The public samplers use ParameterSpaceSampler to apply the same
    correlation policies to either supported statistical design.

    Parameters
    ----------
    config : TaraConfig
        Validated TARA configuration.
    """

    def __init__(self, config: TaraConfig) -> None:
        """Store the configuration for generating correlated domain multipliers.

        Parameters
        ----------
        config : TaraConfig
            Validated TARA configuration object.
        """
        self.config = config

    @abstractmethod
    def generate_domain_multipliers(
        self,
        n_samples: int,
        n_domains: int,
    ) -> np.ndarray:
        """
        Generate stochastic multipliers for one topology sub-ensemble.

        Parameters
        ----------
        n_samples : int
            Number of samples in the current sub-ensemble.
        n_domains : int
            Number of energy domains in the current topology.

        Returns
        -------
        numpy.ndarray
            Multiplier matrix with shape ``(n_samples, n_domains)`` and values
            bounded in ``[-1.0, 1.0]``.
        """
        pass


class IndependentCorrelation(CorrelationStrategy):
    """
    Generate independent stochastic values for every sample and domain.
    """

    def generate_domain_multipliers(
        self,
        n_samples: int,
        n_domains: int,
    ) -> np.ndarray:
        """
        Generate independent uniform multipliers.

        Parameters
        ----------
        n_samples : int
            Number of samples in the current sub-ensemble.
        n_domains : int
            Number of energy domains in the current topology.

        Returns
        -------
        numpy.ndarray
            Independent multipliers with shape ``(n_samples, n_domains)``.
        """
        # One independent random value is required for every sample-domain pair.
        return np.random.uniform(
            -1.0,
            1.0,
            size=(n_samples, n_domains),
        )


class RandomGroupCorrelation(CorrelationStrategy):
    """
    Broadcast one stochastic value per sample across all energy domains.
    """

    def generate_domain_multipliers(
        self,
        n_samples: int,
        n_domains: int,
    ) -> np.ndarray:
        """
        Generate perfectly correlated domain multipliers.

        Parameters
        ----------
        n_samples : int
            Number of samples in the current sub-ensemble.
        n_domains : int
            Number of energy domains in the current topology.

        Returns
        -------
        numpy.ndarray
            Broadcast view with shape ``(n_samples, n_domains)``. Each row
            contains one stochastic value repeated mathematically across all
            domains without allocating repeated storage.
        """
        # Generate only the independent information: one random value per sample.
        base_random_vector = np.random.uniform(
            -1.0,
            1.0,
            size=(n_samples, 1),
        )

        # ``broadcast_to`` preserves the public matrix shape while avoiding the
        # allocation and memory traffic produced by ``np.repeat``.
        return np.broadcast_to(
            base_random_vector,
            (n_samples, n_domains),
        )


def get_correlation_strategy(config: TaraConfig) -> CorrelationStrategy:
    """
    Select the legacy Monte Carlo correlation strategy.

    Parameters
    ----------
    config : TaraConfig
        Validated configuration gateway.

    Returns
    -------
    CorrelationStrategy
        Correlated or independent multiplier generator.

    Raises
    ------
    ValueError
        If ``config.correlation_strategy`` is unsupported.
    """
    strategy_map = {
        "independent": IndependentCorrelation,
        "random_group": RandomGroupCorrelation,
    }

    strat_class = strategy_map.get(config.correlation_strategy)
    if strat_class is None:
        raise ValueError(
            f"Unknown correlation strategy: {config.correlation_strategy}"
        )

    return strat_class(config)

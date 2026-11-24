"""Backend-independent invariants for the first stochastic liquid material model."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import (
    DiscreteOscillator, MaterialScatteringModel, ScatteringSpecies, SpeciesDynamics,
)


def _spectrum(height: float = 1.0) -> PhononSpectra:
    """Create a unit-normalized spectrum with a controllable shape."""
    return PhononSpectra(pd.Series([0.0, height, 1.0, 0.0], index=[0.0, .01, .02, .03]))


def _water() -> MaterialScatteringModel:
    """Construct educational water weights without backend dependencies."""
    dynamics = SpeciesDynamics(
        continuous=_spectrum(),
        continuous_weight=.45,
        translation_weight=.05,
        translation_model='free_gas',
        oscillators=(
            DiscreteOscillator('bending', .205, .167),
            DiscreteOscillator('stretching', .480, .333),
        ),
    )
    return MaterialScatteringModel(
        identifier='water', state='liquid', temperature_k=296.0,
        principal=ScatteringSpecies('H', 2, 1.0),
        secondary=(ScatteringSpecies('O', 1, 16.0),),
        dynamics=dynamics,
    )


def test_liquid_material_retains_independent_contributions() -> None:
    """A sampled VDOS must not change translation, oscillators or composition."""
    model = _water()
    sample = _spectrum(4.0)
    sampled = model.with_spectrum(sample)
    assert sampled is not model
    assert sampled.principal is model.principal
    assert sampled.secondary == model.secondary
    assert sampled.dynamics.oscillators == model.dynamics.oscillators
    assert sampled.dynamics.continuous_weight == .45
    assert sampled.dynamics.translation_weight == .05
    assert sampled.dynamics.continuous is sample
    assert np.isclose(sample.integral, 1)
    assert model.dynamics.continuous is not sample
    with pytest.raises(FrozenInstanceError):
        sampled.temperature_k = 300.0


@pytest.mark.parametrize('weight', [.44, -.01, float('nan')])
def test_invalid_weight_rejected(weight: float) -> None:
    """Reject incomplete, negative and nonfinite dynamical weights."""
    with pytest.raises(ValueError):
        SpeciesDynamics(_spectrum(), continuous_weight=weight)


def test_nonunit_spectrum_rejected() -> None:
    """Reject intentionally unnormalized input, rather than weighting it twice."""
    with pytest.warns(UserWarning):
        spectrum = PhononSpectra(pd.Series([0, 2, 0], index=[0., 1., 2.]), auto_normalize=False)
    with pytest.raises(ValueError, match='unit area'):
        SpeciesDynamics(spectrum)


def test_duplicate_oscillator_labels_rejected() -> None:
    """Identifiable modes must remain unique across future perturbation scenarios."""
    with pytest.raises(ValueError, match='unique'):
        SpeciesDynamics(
            _spectrum(), continuous_weight=.5,
            oscillators=(DiscreteOscillator('bend', .2, .2),
                         DiscreteOscillator('bend', .4, .3)),
        )


def test_no_translation_is_valid_for_a_solid() -> None:
    """A harmonic solid can use just a normalized continuous spectrum."""
    dynamics = SpeciesDynamics(_spectrum())
    assert dynamics.translation_model == 'none'
    assert dynamics.oscillators == ()

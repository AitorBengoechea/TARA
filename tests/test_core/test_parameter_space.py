"""Statistical and reproducibility contracts of the centralized design engine."""

import numpy as np
import pytest

from core.sampler.parameter_space import ParameterSpaceSampler


def strata(matrix):
    """Map signed coordinates to marginal equal-probability stratum IDs."""
    n = matrix.shape[0]
    return np.floor((matrix + 1) * n / 2).astype(int)


@pytest.mark.parametrize('n,d', [(1, 1), (2, 13), (7, 1), (9, 8), (37, 5)])
def test_every_independent_column_visits_each_stratum_once(n, d):
    plan = ParameterSpaceSampler(seed=123)
    values = plan.generate(n, d, subensemble_id=0, group='vdos_spatial')
    assert values.shape == (n, d)
    assert np.all((-1 <= values) & (values < 1))
    for j in range(d):
        np.testing.assert_array_equal(np.sort(strata(values)[:, j]), np.arange(n))


def test_both_signs_are_covered_with_two_or_more_samples():
    values = ParameterSpaceSampler(seed=2).generate(20, 3, group='vdos_amplitude')
    assert (values.min(axis=0) < 0).all()
    assert (values.max(axis=0) > 0).all()


def test_named_groups_are_stable_and_independent_of_call_order():
    design = ParameterSpaceSampler(seed=717)
    first = design.generate(21, 2, group='diffusivity', subensemble_id=3)
    _ = design.generate(21, 7, group='vdos_spatial', subensemble_id=3)
    again = design.generate(21, 2, group='diffusivity', subensemble_id=3)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, ParameterSpaceSampler(seed=718).generate(
        21, 2, group='diffusivity', subensemble_id=3,
    ))
    assert not np.array_equal(first, design.generate(21, 2, group='diffusivity', subensemble_id=4))
    assert not np.array_equal(first, design.generate(21, 2, group='translation_weight', subensemble_id=3))


def test_group_columns_have_stable_prefix_if_other_groups_are_added():
    design = ParameterSpaceSampler(seed=80)
    a = design.generate(12, 2, group='vdos_spatial')
    _ = design.generate(12, 3, group='new_physical_parameter')
    b = design.generate(12, 2, group='vdos_spatial')
    np.testing.assert_array_equal(a, b)


def test_random_group_broadcasts_single_lhs_dimension():
    matrix = ParameterSpaceSampler(seed=22).generate_domains(
        13, 6, correlation_strategy='random_group', subensemble_id=0,
        group='vdos_amplitude',
    )
    assert matrix.shape == (13, 6)
    for j in range(1, 6):
        np.testing.assert_array_equal(matrix[:, 0], matrix[:, j])
    np.testing.assert_array_equal(np.sort(strata(matrix[:, :1])[:, 0]), np.arange(13))


def test_legacy_monte_carlo_uses_exact_global_numpy_sequence():
    np.random.seed(123)
    expected = np.random.uniform(-1, 1, size=(6, 4))
    np.random.seed(123)
    actual = ParameterSpaceSampler('monte_carlo', 0).generate(6, 4, group='anything')
    np.testing.assert_array_equal(actual, expected)


def test_local_monte_carlo_generator_uses_its_own_sequence():
    expected = np.random.default_rng(99).uniform(-1, 1, size=(6, 2))
    actual = ParameterSpaceSampler('monte_carlo', 0).generate(
        6, 2, group='anything', rng=np.random.default_rng(99),
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('bad', ['lhs', 'montecarlo', '', None])
def test_unknown_sampling_method_rejected(bad):
    with pytest.raises(ValueError, match='sampling_method'):
        ParameterSpaceSampler(bad)


@pytest.mark.parametrize('kwargs', [
    {'n_samples': 0, 'n_dimensions': 3, 'group': 'x'},
    {'n_samples': 3, 'n_dimensions': -1, 'group': 'x'},
    {'n_samples': 3, 'n_dimensions': 2, 'group': ''},
    {'n_samples': 3, 'n_dimensions': 2, 'group': 'x', 'subensemble_id': -1},
])
def test_invalid_sampling_shape_rejected(kwargs):
    with pytest.raises(ValueError):
        ParameterSpaceSampler().generate(**kwargs)


def test_does_not_advance_global_rng_state():
    np.random.seed(74)
    expected = np.random.random()
    np.random.seed(74)
    ParameterSpaceSampler(seed=74).generate(7, 3, group='vdos_spatial')
    assert np.random.random() == expected

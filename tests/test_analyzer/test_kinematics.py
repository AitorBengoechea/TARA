"""Scattering-law normalization and Analyzer kinematics."""


from dataclasses import fields
import numpy as np
import pandas as pd
import pytest
from scipy.constants import physical_constants
from analyzer.physics.kinematics import Dsf
from analyzer.physics.kinematics import DsfResult
from analyzer.physics.kinematics import canonicalize_sab
from analyzer.physics.kinematics import compute_dsf
from analyzer.physics.kinematics import compute_ratios
from analyzer.parsers.base_tsl_parser import BaseTslParser
from analyzer.physics.data_models import DifferentialData
from analyzer.physics.data_models import IntegralData


# Scattering law normalization and dynamic structure factors

def _compact_sab() -> pd.DataFrame:
    """NJOY-style S(alpha,-|beta|) stored on non-negative |beta|."""
    alpha = np.linspace(0.0, 100.0, 201)
    beta_abs = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    return pd.DataFrame(
        np.ones((alpha.size, beta_abs.size)),
        index=pd.Index(alpha, name="alpha"),
        columns=pd.Index(beta_abs, name="beta"),
    )


def _full_sab() -> pd.DataFrame:
    """Canonical full signed-beta version of the compact fixture."""
    return canonicalize_sab(_compact_sab())


def test_canonicalize_compact_sab_builds_full_signed_beta_table():
    """Compact NJOY input is expanded once using detailed balance."""
    full = canonicalize_sab(_compact_sab())
    beta = full.columns.to_numpy(dtype=float)

    assert np.all(np.diff(beta) > 0.0)
    assert np.count_nonzero(np.isclose(beta, 0.0)) == 1
    assert beta.min() < 0.0
    assert beta.max() > 0.0

    for beta_abs in (0.5, 1.0, 1.5, 2.0):
        negative = full.loc[:, np.isclose(beta, -beta_abs)].iloc[:, 0]
        positive = full.loc[:, np.isclose(beta, beta_abs)].iloc[:, 0]
        # Check the expected numerical agreement within the declared tolerance.
        np.testing.assert_allclose(negative.to_numpy(), 1.0)
        # Check the expected numerical agreement within the declared tolerance.
        np.testing.assert_allclose(
            positive.to_numpy(),
            np.exp(-beta_abs),
        )


def test_canonicalize_full_signed_sab_preserves_backend_grid_exactly():
    """Already-full signed input is returned without changing beta coordinates."""
    alpha = np.array([0.0, 1.0, 2.0])
    beta = np.array([-1.0e-13, 0.0, 1.0e-13])
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    sab = pd.DataFrame(
        values,
        index=pd.Index(alpha, name="alpha"),
        columns=pd.Index(beta, name="beta"),
    )

    canonical = canonicalize_sab(sab)

    pd.testing.assert_frame_equal(canonical, sab)
    # Require exact array identity here; numerical closeness would be insufficient.
    np.testing.assert_array_equal(
        canonical.columns.to_numpy(dtype=float),
        beta,
    )


def test_canonicalize_compact_sab_mirrors_original_beta_values_exactly():
    """Compact expansion mirrors the native beta grid without snapping it."""
    alpha = np.array([0.0, 1.0])
    beta = np.array([0.0, 1.0e-13, 0.5])
    values = np.array(
        [
            [2.0, 4.0, 6.0],
            [3.0, 5.0, 7.0],
        ]
    )
    compact = pd.DataFrame(
        values,
        index=pd.Index(alpha, name="alpha"),
        columns=pd.Index(beta, name="beta"),
    )

    canonical = canonicalize_sab(compact)

    expected_beta = np.array([-0.5, -1.0e-13, 0.0, 1.0e-13, 0.5])
    # Require exact array identity here; numerical closeness would be insufficient.
    np.testing.assert_array_equal(
        canonical.columns.to_numpy(dtype=float),
        expected_beta,
    )
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(canonical.loc[:, -1.0e-13], values[:, 1])
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(
        canonical.loc[:, 1.0e-13],
        values[:, 1] * np.exp(-1.0e-13),
    )


def test_canonicalize_rejects_actual_duplicate_beta_coordinates():
    """Exact duplicate backend coordinates are invalid and are never merged."""
    alpha = np.array([0.0, 1.0])
    beta = np.array([-1.0, 0.0, 0.0, 1.0])
    sab = pd.DataFrame(
        np.ones((alpha.size, beta.size)),
        index=pd.Index(alpha, name="alpha"),
        columns=pd.Index(beta, name="beta"),
    )

    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(ValueError, match="strictly increasing"):
        canonicalize_sab(sab)



def test_canonicalize_rejects_negative_only_beta_table():
    """Negative-only beta cannot be inferred as compact or full representation."""
    alpha = np.array([0.0, 1.0])
    beta = np.array([-2.0, -1.0])
    sab = pd.DataFrame(
        np.ones((2, 2)),
        index=pd.Index(alpha, name="alpha"),
        columns=pd.Index(beta, name="beta"),
    )

    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(ValueError, match="negative-only"):
        canonicalize_sab(sab)


def test_dsf_result_is_data_only_container():
    """DsfResult stores one Ein result and owns no fixed physics state."""
    assert [field.name for field in fields(DsfResult)] == [
        "Ein",
        "data",
        "kernel",
        "angular_distribution",
        "energy_distribution",
        "ratios",
    ]


def test_dsf_reuses_canonical_fixed_state_and_one_container_per_ein():
    """One Dsf owns full signed S(alpha,beta) while results store only Ein data."""
    model = Dsf(_compact_sab(), temperature=300.0, mass_amu=12.0)

    first = model.result(0.1)
    second = model.result(0.1)
    other = model.result(0.2)

    assert first is second
    assert first is not other
    assert set(model.results) == {0.1, 0.2}
    assert model.beta.min() < 0.0
    assert model.beta.max() > 0.0
    assert not hasattr(model, "balance_factor")
    assert not hasattr(first, "temperature")
    assert not hasattr(first, "mass_amu")
    assert not hasattr(first, "sab")


def test_dsf_methods_populate_container_without_duplicate_calculation():
    """Derived methods populate fields on the unique cached DsfResult."""
    model = Dsf(_compact_sab(), temperature=300.0, mass_amu=12.0)
    result = model.evaluate(
        0.1,
        calc_data=True,
        calc_kernel=True,
        calc_angular=True,
        calc_energy=True,
        calc_ratios=True,
    )

    assert result is model.result(0.1)
    assert result.data is not None
    assert result.kernel is not None
    assert result.angular_distribution is not None
    assert result.energy_distribution is not None
    assert result.ratios is not None
    assert model.compute_data(0.1) is result.data
    assert model.compute_kernel(0.1) is result.kernel
    assert model.compute_ratios(0.1) is result.ratios


def test_dsf_uses_full_signed_sab_without_second_detailed_balance_step():
    """Down/up branches read canonical S directly; detailed balance is not repeated."""
    model = Dsf(
        _compact_sab(),
        temperature=300.0,
        mass_amu=12.0,
        theta=np.array([90.0]),
    )
    data = model.compute_data(1.0)

    kb = physical_constants["Boltzmann constant in eV/K"][0]
    neutron_mass = physical_constants["neutron mass in u"][0]
    kbT = kb * 300.0
    A = 12.0 / neutron_mass
    prefactor = ((A + 1.0) / A) ** 2 / (2.0 * kbT)

    e_down = 1.0 - kbT
    e_up = 1.0 + kbT

    down = data.loc[:, np.isclose(data.columns, e_down)].iloc[0, 0]
    up = data.loc[:, np.isclose(data.columns, e_up)].iloc[0, 0]

    assert np.isclose(down, prefactor * np.sqrt(e_down))
    assert np.isclose(up, np.exp(-1.0) * prefactor * np.sqrt(e_up))


def test_compact_and_explicit_full_sab_produce_identical_dsf():
    """Equivalent NJOY compact and NCrystal full representations share one math path."""
    compact = _compact_sab()
    full = _full_sab()

    from_compact = compute_dsf(compact, Ein=0.1, T=300.0, mass_amu=12.0)
    from_full = compute_dsf(full, Ein=0.1, T=300.0, mass_amu=12.0)

    pd.testing.assert_frame_equal(from_compact, from_full)


def test_multiple_ein_share_model_and_ratios_conserve():
    """Multiple incident energies reuse fixed state and preserve ratio conservation."""
    ein = np.array([0.05, 0.1, 0.2])
    model = Dsf(_compact_sab(), temperature=300.0, mass_amu=12.0)

    results = model.evaluate_many(ein, calc_ratios=True)

    assert set(results) == set(ein)
    assert all(result.ratios is not None for result in results.values())
    assert all(result.ratios["upscattering"] > 0.0 for result in results.values())
    assert all(result.ratios["downscattering"] > 0.0 for result in results.values())
    assert all(np.isclose(result.ratios.sum(), 1.0) for result in results.values())
    for result in results.values():
        assert result.Ein in model.compute_kernel(result.Ein).index
        assert result.ratios["elastic"] == 0.0

    compatibility = compute_ratios(_compact_sab(), ein, 300.0, 12.0)
    assert np.allclose(compatibility.sum(axis=1), 1.0)


def test_constant_kernel_ratios_are_independent_of_grid_resolution():
    """Split a constant kernel at Ein on both coarse and refined grids."""
    for size in (3, 21, 201):
        model = Dsf(_compact_sab(), temperature=300.0, mass_amu=12.0)
        model.result(1.0).kernel = pd.Series(1.0, index=np.linspace(0.0, 2.0, size))

        ratios = model.compute_ratios(1.0)
        np.testing.assert_allclose(ratios[["downscattering", "upscattering"]], [0.5, 0.5])
        assert ratios["elastic"] == 0.0


def test_ratios_integrate_linear_kernel_on_nonuniform_grid():
    """Recover the exact branch areas of K(Eout) = 1 + Eout."""
    model = Dsf(_compact_sab(), temperature=300.0, mass_amu=12.0)
    eout = np.array([0.0, 0.2, 1.0, 1.7, 2.5, 3.0])
    model.result(1.0).kernel = pd.Series(1.0 + eout, index=eout)

    ratios = model.compute_ratios(1.0)
    np.testing.assert_allclose(ratios[["downscattering", "upscattering"]], [0.2, 0.8])
    assert ratios["elastic"] == 0.0


def test_zero_kernel_returns_undefined_ratios():
    """Report all fractions as undefined when the integrated kernel is zero."""
    ratios = compute_ratios(_compact_sab() * 0.0, np.array([0.1]), 300.0, 12.0)

    assert set(ratios.columns) == {"upscattering", "downscattering", "elastic"}
    assert ratios.loc[0.1].isna().all()


def test_low_incident_energy_resolves_both_physical_transfer_branches():
    """Low Ein keeps Eout >= 0 and resolves both sides of Ein."""
    alpha = np.linspace(0.0, 100.0, 201)
    beta_abs = np.array([0.0, 0.0395257, 0.0790514, 0.1581028, 0.3162056])
    sab = pd.DataFrame(
        np.ones((alpha.size, beta_abs.size)),
        index=pd.Index(alpha, name="alpha"),
        columns=pd.Index(beta_abs, name="beta"),
    )

    ein = 0.001
    model = Dsf(sab, temperature=293.6, mass_amu=12.0)
    result = model.evaluate(ein, calc_data=True, calc_ratios=True)

    eout = result.data.columns.to_numpy(dtype=float)

    assert np.isclose(eout.min(), 0.0)
    assert np.all(eout >= 0.0)
    assert np.any((eout > 0.0) & (eout < ein))
    assert np.any(eout > ein)

    assert result.ratios["downscattering"] > 0.0
    assert result.ratios["upscattering"] > 0.0


def test_eout_grid_is_not_limited_to_original_beta_nodes():
    """Physical Eout samples include interpolated beta points between source nodes."""
    model = Dsf(_compact_sab(), temperature=300.0, mass_amu=12.0)
    data = model.compute_data(0.1)

    native_positive_beta = model.beta[model.beta > 0.0]
    native_up = 0.1 + native_positive_beta * model.kbT
    eout = data.columns.to_numpy(dtype=float)

    assert len(eout) > len(native_up)
    assert np.any(
        [not np.any(np.isclose(value, native_up)) for value in eout if value > 0.1]
    )


# Signed energy-transfer grids

def _synthetic_dsf():
    """Construct a smooth, full signed-beta scattering law."""
    alpha = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    beta = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    values = np.exp(-0.15 * alpha[:, None]) * np.exp(
        -0.08 * beta[None, :] ** 2 - 0.5 * beta[None, :]
    )
    sab = pd.DataFrame(values, index=alpha, columns=beta)
    return Dsf(sab, temperature=293.6, mass_amu=12.011,
               theta=np.linspace(0.0, 180.0, 81))


def test_default_energy_distribution_keeps_signed_beta_and_kernel_values():
    model = _synthetic_dsf()
    incident_energy = 0.05
    kernel = model.compute_kernel(incident_energy)
    distribution = model.compute_energy_distribution(incident_energy)

    beta = distribution.index.to_numpy(dtype=float)
    expected_beta = (
        kernel.index.to_numpy(dtype=float) - incident_energy
    ) / model.kbT

    np.testing.assert_allclose(beta, expected_beta, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(distribution.to_numpy(dtype=float), kernel.to_numpy(dtype=float))
    assert np.any(beta < 0.0)
    assert np.any(beta > 0.0)
    assert np.all(np.diff(beta) > 0.0)
    assert np.isclose(kernel.index.min(), 0.0, atol=1e-12)
    assert model.mu.min() == pytest.approx(-1.0)
    assert model.mu.max() == pytest.approx(1.0)


def test_explicit_beta_grid_interpolates_without_extrapolation():
    model = _synthetic_dsf()
    incident_energy = 0.05
    kernel = model.compute_kernel(incident_energy)
    native_beta = (kernel.index.to_numpy(dtype=float) - incident_energy) / model.kbT
    requested_beta = np.array([native_beta.min() - 1.0, -1.0, 0.0, 1.0, native_beta.max() + 1.0])
    distribution = model.compute_energy_distribution(incident_energy, requested_beta)

    assert np.isnan(distribution.iloc[0])
    assert np.isnan(distribution.iloc[-1])
    np.testing.assert_allclose(
        distribution.iloc[1:-1].to_numpy(dtype=float),
        np.interp(requested_beta[1:-1], native_beta, kernel.to_numpy(dtype=float)),
    )


@pytest.mark.parametrize("beta_grid", [[0.0, 0.0], [1.0, -1.0], [np.nan, 1.0], [1.0]])
def test_explicit_beta_grid_rejects_invalid_values(beta_grid):
    model = _synthetic_dsf()
    with pytest.raises(ValueError, match="beta_grid"):
        model.compute_energy_distribution(0.05, beta_grid)


def test_alpha_grid_uses_physical_recoil_for_all_angles_and_energies():
    model = _synthetic_dsf()
    incident_energy = 0.05
    outgoing = np.array([0.0, 0.02, 0.05, 0.10])
    actual = model._alpha_grid(incident_energy, outgoing)
    expected = (
        outgoing[None, :] + incident_energy
        - 2.0 * model.mu[:, None] * np.sqrt(outgoing[None, :] * incident_energy)
    ) / (model.A * model.kbT)

    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (len(model.mu), len(outgoing))
    assert np.min(actual) >= -1e-12
    forward_index = int(np.argmax(model.mu))
    backward_index = int(np.argmin(model.mu))
    assert actual[forward_index, 2] == pytest.approx(0.0, abs=1e-12)
    assert actual[backward_index, 2] == pytest.approx(
        4.0 * incident_energy / (model.A * model.kbT)
    )


# Analyzer extraction across incident energies

class _Parser(BaseTslParser):
    """Represent parser state in the TARA workflow.
    """
    def parse(self, *args, **kwargs):
        """Parse for the current workflow.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to the wrapped function.
        **kwargs : Any
            Additional options accepted by the current interface.
        """
        raise NotImplementedError


def _flat_sab() -> pd.DataFrame:
    """Flat sab for the current workflow.

    Returns
    -------
    pd.DataFrame
        Tabulated values with the coordinates established by this operation.
    """
    alpha = np.linspace(0.0, 100.0, 201)
    beta = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    return pd.DataFrame(
        np.ones((alpha.size, beta.size)),
        index=pd.Index(alpha, name="alpha"),
        columns=pd.Index(beta, name="beta"),
    )


def test_base_tsl_parser_packs_multiple_ein_without_repeating_fixed_state():
    """One fixed DSF state yields the established multi-Ein Analyzer schema."""
    diff = DifferentialData()
    integ = IntegralData()
    ein = np.array([0.05, 0.1, 0.2])

    _Parser()._apply_universal_kinematics(
        _flat_sab(),
        diff,
        integ,
        temp=300.0,
        mass_amu=12.0,
        ein_arr=ein,
        calc_dsf=True,
        calc_scattering_kernel=True,
        calc_angular_distribution=True,
        calc_cosine_distribution=True,
        calc_energy_distribution=True,
        calc_ratios=True,
    )

    assert set(diff.dynamic_structure_factor) == set(ein)
    assert set(integ.scattering_kernel) == set(ein)
    assert set(integ.angular_distribution) == set(ein)
    assert np.allclose(integ.cosine_distribution.index, ein)
    assert np.allclose(integ.energy_distribution.index, ein)
    assert np.allclose(integ.upscattering_ratio.index, ein)
    assert np.allclose(
        integ.upscattering_ratio
        + integ.downscattering_ratio
        + integ.elastic_ratio,
        1.0,
    )

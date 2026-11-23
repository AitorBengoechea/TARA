"""MF7 representation regression tests: ENDF flags, not notebook workarounds."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analyzer.physics.kinematics import canonicalize_sab, decode_njoy_mf7_sab
from analyzer.parsers.njoy_parser import NjoyParser
from analyzer.physics.data_models import DifferentialData, IntegralData


def _tabulated(values, beta=(0.0, 1.0, 2.0)):
    """Build a small valid MF7 alpha-by-beta table."""
    return pd.DataFrame(
        np.asarray(values, dtype=float),
        index=pd.Index([0.0, 2.0], name="alpha"),
        columns=pd.Index(np.asarray(beta, dtype=float), name="beta"),
    )


def _engine(table, *, lasym=0, lln=0):
    """Construct an ENDF-shaped synthetic tape with actual output flags."""
    mf7 = {
        "LASYM": lasym,
        "LLN": lln,
        "beta": dict(enumerate(table.columns.to_numpy(dtype=float))),
        "S_table": {
            i: {
                "alpha": table.index.to_numpy(dtype=float),
                "S": table.iloc[:, i].to_numpy(dtype=float),
            }
            for i in range(table.shape[1])
        },
    }
    return SimpleNamespace(leapr_tree={7: {4: mf7}}, temperature=300.0, mass_amu=1.0)


def _parsed_table(engine):
    """Intercept the shared handoff so the test checks only the MF7 decoder."""
    parser = NjoyParser()
    received = []
    parser._apply_universal_kinematics = lambda sab, *args, **kwargs: received.append(sab)
    parser._extract_kinematics(
        engine, DifferentialData(), IntegralData(), calc_sab=True
    )
    assert len(received) == 1
    return received[0]


def test_symmetric_linear_mf7_restores_both_physical_branches():
    """LASYM=0, LLN=0 produces signed beta with half-beta factors."""
    raw = _tabulated([[4., 2., 1.], [8., 6., 3.]])
    result = _parsed_table(_engine(raw))
    np.testing.assert_array_equal(result.columns.to_numpy(), [-2., -1., 0., 1., 2.])
    np.testing.assert_allclose(result.loc[:, -1.].to_numpy(), raw.loc[:, 1.].to_numpy() * np.exp(.5))
    np.testing.assert_allclose(result.loc[:, 1.].to_numpy(), raw.loc[:, 1.].to_numpy() * np.exp(-.5))
    np.testing.assert_allclose(result.loc[:, -2.].to_numpy(), raw.loc[:, 2.].to_numpy() * np.exp(1.))
    np.testing.assert_allclose(result.loc[:, 2.].to_numpy(), raw.loc[:, 2.].to_numpy() * np.exp(-1.))
    np.testing.assert_array_equal(result.loc[:, 0.].to_numpy(), raw.loc[:, 0.].to_numpy())
    # Canonicalize and DSF must not apply a second detailed-balance transform.
    pd.testing.assert_frame_equal(canonicalize_sab(result), result)
    np.testing.assert_allclose(result.loc[:, 2.], result.loc[:, -2.] * np.exp(-2.))


def test_logarithmic_symmetric_mf7_decodes_natural_log_before_balance():
    """LLN=1 contains ln(S), not linear S or logarithm base ten."""
    linear = _tabulated([[4., 2., 1.], [8., 6., 3.]])
    logged = _tabulated(np.log(linear.to_numpy()))
    actual = _parsed_table(_engine(logged, lasym=0, lln=1))
    expected = decode_njoy_mf7_sab(linear, lasym=0, lln=0)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-13, atol=1e-13)


def test_logarithmic_symmetric_avoids_intermediate_underflow():
    """Combine log(S) and beta/2 before exponentiation for weak loss tails."""
    log_table = _tabulated([[-1., -800.], [-2., -801.]], beta=(0., 200.))
    converted = decode_njoy_mf7_sab(log_table, lasym=0, lln=1)
    np.testing.assert_allclose(converted.loc[:, -200.].to_numpy(), np.exp([-700., -701.]))
    assert (converted.loc[:, -200.].to_numpy() > 0.).all()


def test_asymmetric_mf7_preserves_signed_beta_and_values():
    """LASYM=1 must not mirror a native signed-beta grid."""
    raw = _tabulated([[2., 4., 6.], [3., 5., 7.]], beta=(-2., 0., 3.))
    pd.testing.assert_frame_equal(_parsed_table(_engine(raw, lasym=1)), raw)
    logged = _tabulated(np.log(raw.to_numpy()), beta=(-2., 0., 3.))
    result = _parsed_table(_engine(logged, lasym=1, lln=1))
    pd.testing.assert_frame_equal(result, raw, check_exact=False, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("flags", [{"LASYM": None}, {"LLN": None}, {"LASYM": 2}, {"LLN": 3}])
def test_mf7_invalid_or_missing_flags_fail_instead_of_guessing(flags):
    """No implicit asymmetric interpretation when ENDF metadata is unknown."""
    engine = _engine(_tabulated([[1., 2., 3.], [4., 5., 6.]]))
    mf7 = engine.leapr_tree[7][4]
    for flag, value in flags.items():
        if value is None:
            mf7.pop(flag)
        else:
            mf7[flag] = value
    with pytest.raises(ValueError, match="LASYM|LLN"):
        _parsed_table(engine)


@pytest.mark.parametrize("lasym,beta", [(0, (-1., 0., 1.)), (1, (0., 1., 2.))])
def test_mf7_flag_and_beta_grid_must_agree(lasym, beta):
    """Symmetric is compact; asymmetric supplies both signs explicitly."""
    with pytest.raises(ValueError, match="LASYM"):
        _parsed_table(_engine(_tabulated([[1., 2., 3.], [2., 3., 4.]], beta), lasym=lasym))


def test_mf7_invalid_decoded_values_are_rejected():
    """Negative linear data and overflowing logarithms cannot be normalized."""
    linear = _tabulated([[1., -2., 3.], [4., 5., 6.]])
    with pytest.raises(ValueError, match="nonnegative"):
        _parsed_table(_engine(linear))
    huge_log = _tabulated([[1., 9999., 3.], [4., 5., 6.]])
    with pytest.raises(ValueError, match="finite|overflowed"):
        _parsed_table(_engine(huge_log, lln=1))


def test_no_kinematics_requested_preserves_laziness():
    """No MF7 decoding is attempted if no kinematic product is requested."""
    engine = _engine(_tabulated([[1., 2., 3.], [4., 5., 6.]]))
    engine.leapr_tree[7][4].pop('LLN')
    parser = NjoyParser()
    parser._extract_kinematics(engine, DifferentialData(), IntegralData())

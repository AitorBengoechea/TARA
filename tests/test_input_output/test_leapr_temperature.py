"""Single-temperature serialization from a complete multi-temperature LEAPR input."""

import warnings

import numpy as np
import pandas as pd
import pytest

from core.data.input_pdos import PhononSpectra
from input_output.leapr.core import LeaprFile
from input_output.leapr.temperature import (
    available_leapr_temperatures,
    resolve_leapr_temperature,
)
from input_output.leapr.workflows import prepare_leapr_spectrum, serialize_leapr_spectrum


def _leapr_template(tmp_path, extra_temperatures=(-400.0, -500.0, -2000.0)):
    """Write and reparse a minimal complete LEAPR input including negative cards."""
    path = tmp_path / "template.leapr"
    template = LeaprFile(str(path))
    template.module_call_line = "leapr\n"
    template.file_units_line = "20 21 /\n"
    template.title = "'Test TSL'\n"
    template.run_control_params = {"ntempr": len(extra_temperatures) + 1, "iprint": 0, "nphon": 1}
    template.material_info = {"mat": 31, "za": 6012, "isabt": 0, "ilog": 0}
    template.principal_scatterer = {
        "awr": 11.89692, "spr": 5.0, "npr": 1, "iel": 0, "ncold": 0, "nsk": 0,
    }
    template.secondary_scatterer = {"nss": 0, "b7": 0.0, "aws": 0.0, "sps": 0.0, "mss": 0}
    template.grid_params = {"nalpha": 2, "nbeta": 2, "lat": 0}
    template.alpha_grid = np.array([0.0, 1.0])
    template.beta_grid = np.array([0.0, 1.0])
    template.temp_blocks = [{
        "temp": 296.0, "delta": 0.01, "ni": 4,
        "phonon_spectrum": np.array([0.0, 10.0, 20.0, 0.0]),
        "twt": 0.0, "c": 0.0, "tbeta": 1.0, "nd": 0,
    }] + [{"temp": value} for value in extra_temperatures]
    template.stop_card_found = True
    template.write(str(path))
    return LeaprFile.from_file(str(path))


def _spectrum():
    """Return a normalized source spectrum suitable for LEAPR interpolation."""
    return PhononSpectra(pd.Series(
        [0.0, 10.0, 20.0, 0.0], index=[0.0, 0.01, 0.02, 0.03],
    ))


def test_full_input_includes_negative_temperature_magnitudes(tmp_path):
    """Read every physical temperature without interpreting negatives literally."""
    original = _leapr_template(tmp_path)
    assert [block["temp"] for block in original.temp_blocks] == [296.0, -400.0, -500.0, -2000.0]
    assert available_leapr_temperatures(original) == (296.0, 400.0, 500.0, 2000.0)


@pytest.mark.parametrize("temperature", [296.0, 300.0, 400.0, 800.0, 2000.0])
def test_internal_requests_preserve_exact_single_temperature(tmp_path, temperature):
    """Emit one positive block at the exact request; never retain negative cards."""
    original = _leapr_template(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = serialize_leapr_spectrum(original, _spectrum(), temperature=temperature)
    output = tmp_path / "single.leapr"
    output.write_text(text)
    parsed = LeaprFile.from_file(str(output))
    assert parsed.run_control_params["ntempr"] == 1
    assert len(parsed.temp_blocks) == 1
    assert parsed.temp_blocks[0]["temp"] == temperature
    assert len(caught) == (0 if temperature == 296.0 else 1)
    if caught:
        assert "inside the original temperature range" in str(caught[0].message)
    assert len(original.temp_blocks) == 4
    assert original.temp_blocks[0]["temp"] == 296.0


@pytest.mark.parametrize("temperature,overshoot", [(291.0, 5.0), (2005.0, 5.0), (295.5, 0.5), (293.6, 2.4)])
def test_boundary_margin_is_inclusive(tmp_path, temperature, overshoot):
    """Permit a boundary overshoot up to and including five kelvin."""
    template = _leapr_template(tmp_path)
    with pytest.warns(UserWarning, match="outside the original temperature boundaries"):
        result = resolve_leapr_temperature(template, temperature)
    assert result == temperature


@pytest.mark.parametrize("temperature", [290.999, 2005.001, -50.0, 0.0, float('inf'), float('nan')])
def test_unsupported_temperature_rejected_before_serialization(tmp_path, temperature):
    """Reject extrapolations beyond five kelvin and invalid temperatures."""
    template = _leapr_template(tmp_path)
    with pytest.raises(ValueError):
        prepare_leapr_spectrum(template, _spectrum(), temperature=temperature)
    assert template.temp_blocks[0]["temp"] == 296.0


def test_default_remains_first_positive_block(tmp_path):
    """Existing callers without a temperature remain compatible."""
    template = _leapr_template(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        copy = prepare_leapr_spectrum(template, _spectrum())
    assert not caught
    assert copy.temp_blocks[0]["temp"] == 296.0
    assert copy.run_control_params["ntempr"] == 1


def test_single_original_temperature_also_respects_five_kelvin_margin(tmp_path):
    """One-temperature templates have identical lower and upper bounds."""
    template = _leapr_template(tmp_path, extra_temperatures=())
    with pytest.warns(UserWarning, match="outside"):
        assert resolve_leapr_temperature(template, 301.0) == 301.0
    with pytest.raises(ValueError, match="outside the original range"):
        resolve_leapr_temperature(template, 301.001)


def test_missing_positive_leapr_block_is_invalid(tmp_path):
    """Negative temperature records cannot define a source phonon spectrum."""
    template = LeaprFile(str(tmp_path / "invalid.leapr"))
    template.temp_blocks = [{"temp": -400.0}]
    with pytest.raises(Exception, match="cannot precede"):
        available_leapr_temperatures(template)


def test_write_multiple_spectra_uses_requested_temperature(tmp_path):
    """The disk-writing helper also emits one block at the requested temperature."""
    from input_output.leapr.workflows import write_multiple_spectra

    template = _leapr_template(tmp_path)
    samples = pd.DataFrame(
        [[0.0, 10.0, 20.0, 0.0], [0.0, 20.0, 10.0, 0.0]],
        columns=[0.0, 0.01, 0.02, 0.03],
    )
    out_dir = tmp_path / "outputs"
    with pytest.warns(UserWarning, match="inside the original temperature range") as caught:
        write_multiple_spectra(template, samples, directory=str(out_dir), temperature=600.0)
    assert len(caught) == 1
    paths = sorted(out_dir.glob("*.leapr"))
    assert len(paths) == 2
    for path in paths:
        parsed = LeaprFile.from_file(str(path))
        assert parsed.run_control_params["ntempr"] == 1
        assert [block["temp"] for block in parsed.temp_blocks] == [600.0]

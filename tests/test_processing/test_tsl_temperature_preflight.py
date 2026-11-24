"""Regression checks for NJOY's single-temperature ENDF-to-ACE contract."""

from pathlib import Path
from unittest.mock import patch

import pytest

from processing.orchestrators.njoy_batch import NjoyBatchOrchestrator
from processing.validators.tsl_temperature import (
    read_tsl_temperatures,
    validate_tsl_temperature,
)


def _record(values, *, mf=7, mt=4, sequence=1):
    """Format six ENDF fields and the MAT/MF/MT/NS control suffix."""
    fields = "".join(
        f"{v:11d}" if isinstance(v, int) else f"{float(v):11.4E}"
        for v in values
    )
    return f"{fields}{31:4d}{mf:2d}{mt:3d}{sequence:5d}\n"


def _tsl_file(tmp_path: Path, temperatures=(296.0,)) -> Path:
    """Construct the beginning of a valid MF7/MT4 first-beta section."""
    extra = len(temperatures) - 1
    records = [
        _record([6012.0, 12.0, 0, 0, 0, 0]),  # HEAD
        _record([0.0, 0.0, 0, 0, 6, 0]),     # B LIST
        _record([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        _record([0.0, 0.0, 0, 0, 1, 1]),     # TAB2 with one beta
        _record([1, 2, 0, 0, 0, 0]),         # NR interpolation payload
        _record([temperatures[0], 0.0, extra, 0, 1, 2]),  # TAB1
    ]
    if extra:
        records.append(_record([2, 2, 0, 0, 0, 0]))     # alpha interp + 2 pairs
        for temperature in temperatures[1:]:
            records.extend([
                _record([temperature, 0.0, 2, 0, 2, 0]),
                _record([0.2, 0.3, 0.0, 0.0, 0.0, 0.0]),
            ])
    records.append(_record([0.0] * 6, mt=0))
    path = tmp_path / "sample_000_sab.endf"
    path.write_text("".join(records), encoding="ascii")
    return path


def _config():
    """Return the minimal config needed for the preflight boundary."""
    return {"temperature": 293.6}


def test_reads_actual_tsl_temperature(tmp_path):
    """The reference temperature is read from TSL MF7, not the LEAPR file."""
    path = _tsl_file(tmp_path, (293.6,))
    assert read_tsl_temperatures(str(path)) == (293.6,)
    assert validate_tsl_temperature(str(path), 293.6) == (293.6,)


def test_reads_additional_mf7_temperature_tables(tmp_path):
    """LT LIST tables represent subsequent evaluation temperatures."""
    path = _tsl_file(tmp_path, (296.0, 400.0, 600.0))
    assert read_tsl_temperatures(str(path)) == (296.0, 400.0, 600.0)
    assert validate_tsl_temperature(str(path), 400.0) == (296.0, 400.0, 600.0)


def test_mismatch_explains_both_temperatures_and_the_source_fix(tmp_path):
    """A valid LEAPR range is not a substitute for the generated ENDF value."""
    path = _tsl_file(tmp_path, (296.0,))
    with pytest.raises(ValueError, match="requests 293.6 K.*contains \\[296.0\\] K") as exc:
        validate_tsl_temperature(str(path), 293.6)
    assert "generate_ensemble_sab_tapes" in str(exc.value)


def test_outside_available_list_even_if_between_values(tmp_path):
    """An ENDF library must contain the exact selected temperature."""
    path = _tsl_file(tmp_path, (296.0, 400.0))
    with pytest.raises(ValueError, match="TSL temperature mismatch"):
        validate_tsl_temperature(str(path), 350.0)


def test_preflight_rejects_mismatch_before_dispatch(tmp_path):
    """One reference mismatch aborts ACE processing without starting workers."""
    path = _tsl_file(tmp_path, (296.0,))
    batch = NjoyBatchOrchestrator(njoy_exec="unused", n_jobs=2)
    with patch("processing.orchestrators.njoy_batch.run_parallel") as parallel:
        with pytest.raises(ValueError, match="TSL temperature mismatch"):
            batch.process_ace_ensemble(
                {"sample_000": str(path), "sample_001": "unread.endf"},
                str(tmp_path / "ace"), _config(),
            )
    parallel.assert_not_called()
    assert not (tmp_path / "ace").exists()


def test_preflight_reads_only_reference_before_dispatch(tmp_path):
    """A consistent reference file is sufficient for the shared batch policy."""
    path = _tsl_file(tmp_path, (293.6,))
    batch = NjoyBatchOrchestrator(njoy_exec="unused", n_jobs=1)
    with patch("processing.orchestrators.njoy_batch.run_parallel", return_value=[]) as parallel:
        batch.process_ace_ensemble(
            {"sample_000": str(path), "sample_001": "/not/read.endf"},
            str(tmp_path / "ace"), _config(),
        )
    parallel.assert_called_once()


def test_preflight_uses_first_input_without_reference(tmp_path):
    """External batches without sample_000 still get exactly one preflight."""
    path = _tsl_file(tmp_path, (293.6,))
    batch = NjoyBatchOrchestrator(njoy_exec="unused", n_jobs=1)
    with patch("processing.orchestrators.njoy_batch.run_parallel", return_value=[]) as parallel:
        batch.process_ace_ensemble(
            {"sample_001": str(path), "sample_002": "/not/read.endf"},
            str(tmp_path / "ace"), _config(),
        )
    parallel.assert_called_once()


def test_malformed_mf7_rejected_instead_of_skipped(tmp_path):
    """Broken or absent temperature records must not silently pass validation."""
    path = tmp_path / "broken.endf"
    path.write_text("not an ENDF evaluation", encoding="ascii")
    with pytest.raises(ValueError, match="MF7/MT4"):
        read_tsl_temperatures(str(path))

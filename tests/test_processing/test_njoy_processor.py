from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from processing.backends.njoy_processor import NjoyProcessor


EXPECTED_NJOY_PATH = "/usr/local/bin/njoy"
HAS_NJOY = Path(EXPECTED_NJOY_PATH).is_file()


def _processor(tmp_path):
    """Processor for the current workflow.

    Parameters
    ----------
    tmp_path : object
        Filesystem location of the tmp path.
    """
    dummy_exec = tmp_path / "dummy_njoy"
    dummy_exec.touch()
    return NjoyProcessor(njoy_exec=str(dummy_exec), temp_dir=str(tmp_path))


def _mock_successful_popen(mock_popen, *, tapes=("tape24",)):
    """Mock successful popen for the current workflow.

    Parameters
    ----------
    mock_popen : object
        Mock subprocess factory used by the test.
    tapes : object, optional
        Named tape paths available to the current NJOY job.
    """
    process = MagicMock()
    process.communicate.return_value = ("Success output", None)
    process.returncode = 0

    def fake_popen(*args, **kwargs):
        """Provide a deterministic test substitute for popen.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to the wrapped function.
        **kwargs : Any
            Additional options accepted by the current interface.
        """
        cwd = Path(kwargs["cwd"])
        for tape in tapes:
            (cwd / tape).write_text(f"dummy {tape}", encoding="utf-8")
        return process

    mock_popen.side_effect = fake_popen
    return process


def test_njoy_processor_initialization_rejects_missing_executable():
    """Test executable validation at the reusable Processing boundary."""
    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(FileNotFoundError, match="NJOY executable not found"):
        NjoyProcessor(njoy_exec="/fake/path/to/njoy")


@patch("processing.backends.njoy_processor.subprocess.Popen")
def test_njoy_processor_transient_execution_lifetime(mock_popen, tmp_path):
    """
    Test context-managed transient NJOY execution used by Analyzer.

    Verifies that:
    - Generated tapes are accessible while the Processing context is alive.
    - ``has_tape`` and ``tape_path`` expose transient outputs without persistence.
    - The complete sandbox is removed immediately after context exit.
    """
    _mock_successful_popen(mock_popen, tapes=("tape24", "tape25"))
    processor = _processor(tmp_path)

    with processor.execute("dummy deck") as execution:
        run_dir = execution.run_dir
        assert run_dir.is_dir()
        assert execution.has_tape("tape24")
        assert execution.has_tape("tape25")
        assert execution.tape_path("tape24").read_text(encoding="utf-8") == "dummy tape24"
        assert execution.stdout_text == "Success output"

    assert not run_dir.exists()


@patch("processing.backends.njoy_processor.subprocess.Popen")
def test_njoy_processor_cleans_sandbox_when_consumer_raises(mock_popen, tmp_path):
    """Test sandbox cleanup when parsing/consumption fails inside the context."""
    _mock_successful_popen(mock_popen, tapes=("tape24",))
    processor = _processor(tmp_path)
    run_dir = None

    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(ValueError, match="parser failed"):
        with processor.execute("dummy deck") as execution:
            run_dir = execution.run_dir
            raise ValueError("parser failed")

    assert run_dir is not None
    assert not run_dir.exists()


@patch("processing.backends.njoy_processor.subprocess.Popen")
def test_njoy_processor_persistent_leapr_wrapper(mock_popen, tmp_path):
    """
    Test that the persistent Processing API delegates through transient execution.
    """
    _mock_successful_popen(mock_popen, tapes=("tape24",))
    processor = _processor(tmp_path)
    output_file = tmp_path / "final_output.endf"

    processor.execute_leapr("dummy deck", str(output_file))

    assert output_file.read_text(encoding="utf-8") == "dummy tape24"
    assert not list(tmp_path.glob("njoy_*"))


@patch("processing.backends.njoy_processor.subprocess.Popen")
def test_njoy_processor_failure_handling(mock_popen, tmp_path):
    """
    Test error translation and cleanup for failed NJOY subprocess execution.
    """
    processor = _processor(tmp_path)
    process = MagicMock()
    process.communicate.return_value = ("error in LEAPR module", None)
    process.returncode = 1
    mock_popen.return_value = process

    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(RuntimeError, match="NJOY execution failed"):
        with processor.execute("dummy deck"):
            pass

    assert not list(tmp_path.glob("njoy_*"))


@pytest.mark.integration
@pytest.mark.skipif(not HAS_NJOY, reason="Real NJOY executable not found (run inside Docker).")
def test_njoy_processor_real_integration(tmp_path):
    """
    Integration test verifying genuine persistent LEAPR execution through NjoyProcessor.
    """
    processor = NjoyProcessor(njoy_exec=EXPECTED_NJOY_PATH, temp_dir=str(tmp_path))
    minimal_deck = """leapr
-24
1 1 0
1.0 0.0253 1 296.0
1 1
0.1
0.1
0.1
0.1
stop
"""
    out_file = tmp_path / "real_sab.endf"

    processor.execute_leapr(leapr_deck=minimal_deck, output_endf_path=str(out_file))

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def _write_mock_ace_inputs(tmp_path):
    """Write placeholder input evaluations for sandbox staging."""
    (tmp_path / "base.endf").write_text("base ENDF", encoding="utf-8")
    (tmp_path / "tsl.endf").write_text("TSL ENDF", encoding="utf-8")


@pytest.mark.parametrize("save_xsdir", [True, False])
@patch("processing.backends.njoy_processor.subprocess.Popen")
def test_execute_ace_extracts_correct_njoy_tapes(mock_popen, tmp_path, save_xsdir):
    """The ACER ACE output is tape40; tape41 is a small XSDIR index."""
    ace_bytes = b"tsl.01t 11.8969\nTEST ACE LIBRARY RECORDS\n" + b"1.000000E+00 " * 40
    xsdir_line = b"tsl.01t   11.896920 filename route 1 1   1544386     0     0 2.530E-08\n"
    process = MagicMock()
    process.returncode = 0
    process.communicate.return_value = ("NJOY completed", None)

    def fake_popen(*args, **kwargs):
        sandbox = Path(kwargs["cwd"])
        (sandbox / "tape40").write_bytes(ace_bytes)
        (sandbox / "tape41").write_bytes(xsdir_line)
        return process

    mock_popen.side_effect = fake_popen
    processor = _processor(tmp_path)
    _write_mock_ace_inputs(tmp_path)
    output = tmp_path / "ace" / "sample_000.ace"
    processor.execute_ace(
        "acer deck", str(output), base_endf_path=str(tmp_path / "base.endf"),
        tsl_endf_path=str(tmp_path / "tsl.endf"), save_xsdir=save_xsdir,
    )
    assert output.read_bytes() == ace_bytes
    index = output.parent / "xsdir" / "sample_000.xsdir"
    if save_xsdir:
        assert index.read_bytes() == xsdir_line
    else:
        assert not index.exists()
    assert not list(tmp_path.glob("njoy_*"))


@patch("processing.backends.njoy_processor.subprocess.Popen")
def test_execute_ace_rejects_missing_ace_without_returning_xsdir(mock_popen, tmp_path):
    """A successful XSDIR output alone is not a successful ACE conversion."""
    process = MagicMock()
    process.returncode = 0
    process.communicate.return_value = ("NJOY completed", None)

    def fake_popen(*args, **kwargs):
        (Path(kwargs["cwd"]) / "tape41").write_text("tsl.01t filename route 1 1")
        return process

    mock_popen.side_effect = fake_popen
    _write_mock_ace_inputs(tmp_path)
    output = tmp_path / "ace" / "sample_001.ace"
    output.parent.mkdir()
    output.write_text("stale ACE")
    with pytest.raises(FileNotFoundError, match="tape40"):
        _processor(tmp_path).execute_ace("deck", str(output), base_endf_path=str(tmp_path / "base.endf"), tsl_endf_path=str(tmp_path / "tsl.endf"))
    assert not output.exists()
    assert not (output.parent / "xsdir" / "sample_001.xsdir").exists()


@patch("processing.backends.njoy_processor.subprocess.Popen")
def test_execute_ace_missing_xsdir_is_optional_only_when_disabled(mock_popen, tmp_path):
    """An absent index is an error only if XSDIR persistence was requested."""
    process = MagicMock(returncode=0)
    process.communicate.return_value = ("NJOY completed", None)

    def fake_popen(*args, **kwargs):
        (Path(kwargs["cwd"]) / "tape40").write_text("ACE data\nABC records\n")
        return process

    mock_popen.side_effect = fake_popen
    _write_mock_ace_inputs(tmp_path)
    output = tmp_path / "library.ace"
    processor = _processor(tmp_path)
    with pytest.raises(FileNotFoundError, match="tape41"):
        processor.execute_ace("deck", str(output), base_endf_path=str(tmp_path / "base.endf"), tsl_endf_path=str(tmp_path / "tsl.endf"))
    assert not output.exists()
    processor.execute_ace("deck", str(output), base_endf_path=str(tmp_path / "base.endf"), tsl_endf_path=str(tmp_path / "tsl.endf"),
                          save_xsdir=False)
    assert output.read_text().startswith("ACE data")


@patch("processing.backends.njoy_processor.subprocess.Popen")
def test_execute_ace_rejects_xsdir_masquerading_as_ace(mock_popen, tmp_path):
    """Regression guard against saving the XSDIR entry under an ACE filename."""
    process = MagicMock(returncode=0)
    process.communicate.return_value = ("NJOY completed", None)

    def fake_popen(*args, **kwargs):
        sandbox = Path(kwargs["cwd"])
        (sandbox / "tape40").write_text("tsl.01t 11.896920 filename route 1 1 1544386")
        (sandbox / "tape41").write_text("tsl.01t 11.896920 filename route 1 1 1544386")
        return process

    mock_popen.side_effect = fake_popen
    _write_mock_ace_inputs(tmp_path)
    output = tmp_path / "sample_000.ace"
    with pytest.raises(ValueError, match="XSDIR entry"):
        _processor(tmp_path).execute_ace("deck", str(output), base_endf_path=str(tmp_path / "base.endf"), tsl_endf_path=str(tmp_path / "tsl.endf"))
    assert not output.exists()


@pytest.mark.parametrize("full", [False, True])
def test_analysis_serialization_uses_requested_temperature(tmp_path, full):
    """Analyzer's LEAPR sample must have the same temperature as its THERMR deck."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from processing.backends import njoy_processor

    processor = _processor(tmp_path)
    processor._leapr_template = SimpleNamespace(
        material_info={"mat": 31},
        principal_scatterer={"npr": 1, "iel": 0},
    )
    processor.base_endf_file = "base.endf"
    processor.base_mat = 625

    @contextmanager
    def fake_execute(deck, input_tapes):
        yield SimpleNamespace(stdout_text="ok", run_dir=tmp_path)

    with patch.object(njoy_processor, "serialize_leapr_spectrum", return_value="leapr\n") as serialize, \
         patch.object(njoy_processor, "build_analysis_deck", return_value="deck") as build, \
         patch.object(processor, "execute", side_effect=fake_execute):
        with processor.execute_analysis(object(), temperature=293.6, export_full_spectra=full):
            pass

    assert serialize.call_args.kwargs["temperature"] == 293.6
    assert serialize.call_args.kwargs["export_full_spectra"] is full
    assert build.call_args.kwargs["temperature"] == 293.6

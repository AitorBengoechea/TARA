import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from joblib import parallel_backend

from processing.orchestrators.njoy_batch import NjoyBatchOrchestrator

class DummyPhononSpectra:
    def __init__(self, name):
        self.name = name
        self.energy_grid = [0.0, 1.0]
        self.density_values = [0.0, 1.0]


# ==========================================
# NCRYSTAL ORCHESTRATOR TESTS
# ==========================================



# ==========================================
# NJOY ORCHESTRATOR TESTS (PHASE 1 & 2)
# ==========================================

@patch("processing.backends.njoy_processor.NjoyProcessor._resolve_njoy_path", return_value="/dummy/njoy")
@patch("processing.orchestrators.njoy_batch.build_leapr_deck")
@patch("processing.orchestrators.njoy_batch.NjoyProcessor.execute_leapr")
@patch("processing.orchestrators.njoy_batch.EndfValidator.validate_sab_tape")
def test_njoy_leapr_ensemble_fault_tolerance(mock_validate, mock_execute, mock_build, mock_resolve, tmp_path):
    """
    Test fault tolerance and error trapping during NJOY LEAPR batch processing (Phase 1).

    Verifies that NjoyBatchOrchestrator.process_leapr_ensemble:
    - Builds LEAPR input decks and dispatches jobs across an ensemble of phonon spectra.
    - Traps individual Fortran subprocess crashes or runtime failures gracefully.
    - Filters out failed runs and returns only verified ENDF MF7 output paths.
    """
    orchestrator = NjoyBatchOrchestrator(njoy_exec="dummy_exec", n_jobs=2)
    ensemble = {
        "sample_A": DummyPhononSpectra("A"),
        "sample_B": DummyPhononSpectra("B") 
    }
    
    config = {
        "mat_id": 125, "mass_amu": 12.0, "temperature": 296.0, 
        "nalpha": 1, "nbeta": 1, "alpha_grid_str": "1", "beta_grid_str": "1"
    }

    mock_build.return_value = "dummy deck string"
    
    def fake_execute(deck, out_path):
        if "sample_B" in out_path:
            raise RuntimeError("NJOY Segfault")
    mock_execute.side_effect = fake_execute
    mock_validate.return_value = True 
    
    out_dir = tmp_path / "leapr_outputs"
    
    with parallel_backend("threading"):
        results = orchestrator.process_leapr_ensemble(ensemble, str(out_dir), config)
    
    assert len(results) == 1
    assert "sample_A" in results


@patch("processing.backends.njoy_processor.NjoyProcessor._resolve_njoy_path", return_value="/dummy/njoy")
@patch("processing.orchestrators.njoy_batch.EndfValidator.validate_tsl_temperature", return_value=(296.0,))
@patch("processing.orchestrators.njoy_batch.build_ace_deck")
@patch("processing.orchestrators.njoy_batch.NjoyProcessor.execute_ace")
def test_njoy_ace_ensemble_fault_tolerance(mock_execute, mock_build, mock_validate_temp, mock_resolve, tmp_path):
    """
    Test fault tolerance during parallel NJOY ACE library generation (Phase 2).

    Verifies that NjoyBatchOrchestrator.process_ace_ensemble:
    - Constructs ACER/BROADR/THERMR execution decks from intermediate ENDF tapes.
    - Isolates individual backend execution failures to prevent batch job termination.
    - Returns a dictionary mapped exclusively to successfully compiled ACE library paths.
    """
    orchestrator = NjoyBatchOrchestrator(njoy_exec="dummy_exec", n_jobs=2)
    
    sab_files = {
        "sample_1": "/fake/path/1.endf",
        "sample_2": "/fake/path/2.endf" 
    }
    
    config = {
        "base_endf_path": "base.endf", "base_za": 1001, "base_mat": 125, 
        "tsl_mat": 125, "temperature": 296.0, "icoh": 0, "natom": 1, 
        "emax": 5.0, "suffix": ".00t", "za1": 1001, "za2": 0, "za3": 0, 
        "elas_mat": 125, "ielas": 1
    }
    
    mock_build.return_value = "dummy ace deck"
    
    def fake_execute(deck, out_path, **kwargs):
        if "sample_2" in out_path:
            raise RuntimeError("NJOY ACE Formatting Error")
    mock_execute.side_effect = fake_execute

    out_dir = tmp_path / "ace_outputs"
    
    with parallel_backend("threading"):
        results = orchestrator.process_ace_ensemble(sab_files, str(out_dir), config)
    
    assert len(results) == 1
    assert "sample_1" in results
    mock_validate_temp.assert_called_once()


# ==========================================
# NJOY TEMPLATE-BASED ENSEMBLE API
# ==========================================

class DummyTaraEnsemble:
    """Represent TARA's separate reference and zero-based stochastic samples."""

    def __init__(self, sample_count=2, reference=True):
        self.reference = DummyPhononSpectra("reference") if reference else None
        self.samples = [DummyPhononSpectra(f"stochastic_{i}") for i in range(sample_count)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


@pytest.fixture
def njoy_template_mocks(monkeypatch):
    """Mock NJOY while preserving the real template-to-deck call boundary."""
    from processing.orchestrators import njoy_batch

    # One distinct template object per worker, as required for mutable LEAPR state.
    from types import SimpleNamespace
    read_template = MagicMock(side_effect=lambda _: SimpleNamespace(temp_blocks=[{"temp": 296.0}]))
    serializer = MagicMock(side_effect=lambda leapr_file, spectrum, **kwargs: f"module:{spectrum.name}")
    builder = MagicMock(side_effect=lambda leapr_module_text: f"deck:{leapr_module_text}")
    execute = MagicMock(
        side_effect=lambda leapr_deck, output_endf_path: Path(output_endf_path).write_text(leapr_deck)
    )
    validator = MagicMock(return_value=True)

    monkeypatch.setattr(njoy_batch.LeaprFile, "from_file", read_template)
    monkeypatch.setattr(njoy_batch, "serialize_leapr_spectrum", serializer)
    monkeypatch.setattr(njoy_batch, "build_leapr_deck_from_module", builder)
    processor = MagicMock()
    processor.execute_leapr = execute
    monkeypatch.setattr(njoy_batch, "NjoyProcessor", MagicMock(return_value=processor))
    monkeypatch.setattr(
        njoy_batch, "EndfValidator", MagicMock(return_value=MagicMock(validate_sab_tape=validator))
    )
    return read_template, serializer, builder, execute, validator


def _make_leapr_template(tmp_path):
    path = tmp_path / "complete_template.leapr"
    path.write_text("complete LEAPR input preserved by serializer")
    return path


def test_njoy_generate_ensemble_tapes_sequential_reference_and_template(
    tmp_path, njoy_template_mocks
):
    """A real ensemble's first output is its reference, not stochastic sample zero.

    Every sample uses the Input/Output template serializer and the full LEAPR
    module builder; all tapes are validated and returned in stable ID order.
    """
    read_template, serializer, builder, execute, validator = njoy_template_mocks
    base = _make_leapr_template(tmp_path)
    output_dir = tmp_path / "out"
    result = NjoyBatchOrchestrator("unused", n_jobs=1).generate_ensemble_sab_tapes(
        ensemble=DummyTaraEnsemble(sample_count=2),
        output_dir=str(output_dir),
        base_leapr_file=str(base),
    )

    assert list(result) == ["sample_000", "sample_001", "sample_002"]
    assert [Path(p).name for p in result.values()] == [
        "sample_000_sab.endf", "sample_001_sab.endf", "sample_002_sab.endf"
    ]
    assert [Path(p).read_text() for p in result.values()] == [
        "deck:module:reference", "deck:module:stochastic_0", "deck:module:stochastic_1"
    ]
    assert read_template.call_count == 4  # One preflight plus one independent load per worker.
    assert serializer.call_count == builder.call_count == execute.call_count == 3
    assert validator.call_count == 3
    # No template is reused across jobs, even for sequential execution.
    templates = [call.kwargs["leapr_file"] for call in serializer.call_args_list]
    assert len({id(template) for template in templates}) == 3


def test_njoy_generate_ensemble_tapes_parallel(tmp_path, njoy_template_mocks):
    """Two-worker joblib execution produces all reference and sample outputs."""
    base = _make_leapr_template(tmp_path)
    with parallel_backend("threading"):
        result = NjoyBatchOrchestrator("unused", n_jobs=2).generate_ensemble_sab_tapes(
            DummyTaraEnsemble(sample_count=5), str(tmp_path / "out"), str(base)
        )
    assert list(result) == [f"sample_{i:03d}" for i in range(6)]
    assert all(Path(path).is_file() for path in result.values())


def test_njoy_generate_ensemble_tapes_reference_only(tmp_path, njoy_template_mocks):
    """An ensemble with zero stochastic samples still produces the reference."""
    base = _make_leapr_template(tmp_path)
    result = NjoyBatchOrchestrator("unused", n_jobs=1).generate_ensemble_sab_tapes(
        DummyTaraEnsemble(sample_count=0), str(tmp_path / "out"), str(base)
    )
    assert list(result) == ["sample_000"]


def test_njoy_generate_ensemble_tapes_requires_reference(tmp_path, njoy_template_mocks):
    """A missing reference cannot silently shift stochastic output numbering."""
    base = _make_leapr_template(tmp_path)
    with pytest.raises(ValueError, match="reference spectrum"):
        NjoyBatchOrchestrator("unused", n_jobs=1).generate_ensemble_sab_tapes(
            DummyTaraEnsemble(reference=False), str(tmp_path / "out"), str(base)
        )
    assert not (tmp_path / "out").exists()


def test_njoy_generate_ensemble_tapes_requires_template(tmp_path, njoy_template_mocks):
    """The complete LEAPR template is mandatory for the template-based path."""
    with pytest.raises(FileNotFoundError, match="LEAPR template not found"):
        NjoyBatchOrchestrator("unused", n_jobs=1).generate_ensemble_sab_tapes(
            DummyTaraEnsemble(), str(tmp_path / "out"), str(tmp_path / "missing.leapr")
        )
    assert not (tmp_path / "out").exists()


def test_njoy_generate_ensemble_tapes_rejects_invalid_endf_and_cleans_up(
    tmp_path, njoy_template_mocks
):
    """MF7/MT4 validation failure raises and removes the corresponding tape."""
    _, _, _, _, validator = njoy_template_mocks
    validator.side_effect = lambda file_path: "sample_001" not in file_path
    base = _make_leapr_template(tmp_path)
    output = tmp_path / "out"

    with pytest.raises(RuntimeError, match="sample_001.*validation"):
        NjoyBatchOrchestrator("unused", n_jobs=1).generate_ensemble_sab_tapes(
            DummyTaraEnsemble(), str(output), str(base)
        )
    assert not (output / "sample_001_sab.endf").exists()
    assert (output / "sample_000_sab.endf").is_file()


def test_njoy_generate_ensemble_tapes_cleans_partial_file_on_njoy_error(
    tmp_path, njoy_template_mocks
):
    """Failed NJOY execution cannot leave a partial or stale ENDF tape."""
    _, _, _, execute, _ = njoy_template_mocks

    def failing_execute(leapr_deck, output_endf_path):
        target = Path(output_endf_path)
        target.write_text("partial ENDF")
        if "sample_002" in target.name:
            raise RuntimeError("NJOY subprocess crashed")

    execute.side_effect = failing_execute
    base = _make_leapr_template(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "sample_002_sab.endf").write_text("old valid ENDF")

    with pytest.raises(RuntimeError, match="sample_002.*NJOY subprocess crashed"):
        NjoyBatchOrchestrator("unused", n_jobs=1).generate_ensemble_sab_tapes(
            DummyTaraEnsemble(), str(output), str(base)
        )
    assert not (output / "sample_002_sab.endf").exists()


def test_njoy_generate_ensemble_tapes_reports_serialization_failure(
    tmp_path, njoy_template_mocks
):
    """LEAPR serialization failures are attributed to the right sample."""
    _, serializer, _, _, _ = njoy_template_mocks

    def bad_spectrum(leapr_file, spectrum, **kwargs):
        if spectrum.name == "stochastic_0":
            raise ValueError("Unsupported template grid")
        return f"module:{spectrum.name}"

    serializer.side_effect = bad_spectrum
    base = _make_leapr_template(tmp_path)
    with pytest.raises(RuntimeError, match="sample_001.*Unsupported template grid"):
        NjoyBatchOrchestrator("unused", n_jobs=1).generate_ensemble_sab_tapes(
            DummyTaraEnsemble(), str(tmp_path / "out"), str(base)
        )


def test_njoy_generate_ensemble_tapes_requires_actual_output(
    tmp_path, njoy_template_mocks
):
    """A no-op NJOY backend is not interpreted as a successful tape."""
    _, _, _, execute, _ = njoy_template_mocks
    execute.side_effect = lambda leapr_deck, output_endf_path: None
    base = _make_leapr_template(tmp_path)
    with pytest.raises(RuntimeError, match="sample_000.*did not produce"):
        NjoyBatchOrchestrator("unused", n_jobs=1).generate_ensemble_sab_tapes(
            DummyTaraEnsemble(sample_count=0), str(tmp_path / "out"), str(base)
        )


def test_njoy_temperature_is_validated_once_and_forwarded(tmp_path, njoy_template_mocks):
    """Preflight uses the entire source range and issues one batch-level warning."""
    from types import SimpleNamespace
    read_template, serializer, _, _, _ = njoy_template_mocks
    read_template.side_effect = lambda _: SimpleNamespace(
        temp_blocks=[{"temp": 296.0}, {"temp": -400.0}, {"temp": -2000.0}],
    )
    base = _make_leapr_template(tmp_path)
    with pytest.warns(UserWarning, match="inside the original temperature range") as caught:
        outputs = NjoyBatchOrchestrator("unused", n_jobs=1).generate_ensemble_sab_tapes(
            DummyTaraEnsemble(sample_count=2), str(tmp_path / "out"), str(base),
            temperature=1500.125,
        )
    assert len(caught) == 1
    assert list(outputs) == ["sample_000", "sample_001", "sample_002"]
    assert all(call.kwargs["temperature"] == 1500.125 for call in serializer.call_args_list)
    assert all(call.kwargs["warn_on_temperature_reuse"] is False
               for call in serializer.call_args_list)


def test_njoy_temperature_outside_margin_aborts_before_workers(tmp_path, njoy_template_mocks):
    """Unacceptable requested temperatures must not create outputs or run NJOY."""
    from types import SimpleNamespace
    read_template, serializer, _, execute, _ = njoy_template_mocks
    read_template.side_effect = lambda _: SimpleNamespace(
        temp_blocks=[{"temp": 296.0}, {"temp": -400.0}],
    )
    base = _make_leapr_template(tmp_path)
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="outside the original range"):
        NjoyBatchOrchestrator("unused", n_jobs=1).generate_ensemble_sab_tapes(
            DummyTaraEnsemble(), str(destination), str(base), temperature=405.0001,
        )
    assert not destination.exists()
    assert read_template.call_count == 1
    serializer.assert_not_called()
    execute.assert_not_called()


def test_njoy_temperature_five_kelvin_outside_is_allowed(tmp_path, njoy_template_mocks):
    """Inclusive tolerance is honored before emitting the sample decks."""
    from types import SimpleNamespace
    read_template, serializer, _, _, _ = njoy_template_mocks
    read_template.side_effect = lambda _: SimpleNamespace(
        temp_blocks=[{"temp": 296.0}, {"temp": -400.0}],
    )
    base = _make_leapr_template(tmp_path)
    with pytest.warns(UserWarning, match="outside the original temperature boundaries") as caught:
        outputs = NjoyBatchOrchestrator("unused", n_jobs=1).generate_ensemble_sab_tapes(
            DummyTaraEnsemble(sample_count=0), str(tmp_path / "out"), str(base),
            temperature=405.0,
        )
    assert len(caught) == 1
    assert list(outputs) == ["sample_000"]
    assert serializer.call_args.kwargs["temperature"] == 405.0

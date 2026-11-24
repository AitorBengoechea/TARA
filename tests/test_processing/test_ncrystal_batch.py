"""Ncrystal batch regression tests."""


from types import SimpleNamespace
from unittest.mock import patch
import pytest
from processing.orchestrators import ncrystal_batch
from processing.orchestrators.ncrystal_batch import NcrystalBatchOrchestrator
from pathlib import Path
from unittest.mock import MagicMock
from joblib import parallel_backend


# Ncrystal ensemble batch

class DummyEnsemble:
    reference = "unperturbed"

    def __init__(self, count):
        self.samples = [f"stochastic_{i}" for i in range(count)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_constructor_does_not_instantiate_writer():
    """The batch constructor must not instantiate an unconfigured NCrystalWriter."""
    with patch("processing.orchestrators.ncrystal_batch.NCrystalWriter") as writer_cls:
        writer_cls.side_effect = TypeError("missing required ncrystal_obj")
        batch = NcrystalBatchOrchestrator(n_jobs=2)
    writer_cls.assert_not_called()
    assert batch.n_jobs == 2


def test_batch_preserves_reference_and_sample_numbering(tmp_path):
    template = tmp_path / "graphite.ncmat"
    template.write_text("NCMAT", encoding="utf-8")
    seen = []

    def fake_worker(**kwargs):
        seen.append((kwargs["sample_id"], kwargs["spectrum"]))
        return kwargs["sample_id"], kwargs["output_path"], None

    batch = NcrystalBatchOrchestrator(n_jobs=1)
    with patch.object(batch, "_generate_tape_from_template", side_effect=fake_worker):
        outputs = batch.generate_ensemble_sab_tapes(
            ensemble=DummyEnsemble(3),
            output_dir=str(tmp_path / "endf"),
            base_ncmat_file=str(template),
            temperature=293.6,
            atom_label="C",
        )

    assert list(outputs) == ["sample_000", "sample_001", "sample_002", "sample_003"]
    assert seen[0] == ("sample_000", "unperturbed")
    assert seen[1] == ("sample_001", "stochastic_0")
    assert all(path.endswith(f"{sample_id}_sab.endf") for sample_id, path in outputs.items())



@pytest.mark.parametrize("include_reference", [True, False])
def test_parallel_batch_preserves_identifiers(tmp_path, include_reference):
    """Use the configured worker count with or without the reference."""
    from joblib import parallel_backend

    template = tmp_path / "graphite.ncmat"
    template.write_text("NCMAT", encoding="utf-8")
    ensemble = DummyEnsemble(12)
    if not include_reference:
        ensemble.reference = None

    def fake_worker(**kwargs):
        """Return the scheduled sample and its output path."""
        return kwargs["sample_id"], kwargs["output_path"], None

    batch = NcrystalBatchOrchestrator(n_jobs=2)
    with patch.object(batch, "_generate_tape_from_template", side_effect=fake_worker) as worker, \
         patch.object(ncrystal_batch, "run_parallel", wraps=ncrystal_batch.run_parallel) as run, \
         parallel_backend("threading"):
        outputs = batch.generate_ensemble_sab_tapes(
            ensemble=ensemble, output_dir=str(tmp_path / "endf"),
            base_ncmat_file=str(template), temperature=293.6, atom_label="C",
            include_reference=include_reference,
        )

    assert run.call_args.kwargs["n_jobs"] == 2
    assert list(outputs) == [f"sample_{index:03d}" for index in range(int(not include_reference), 13)]
    spectra = {call.kwargs["sample_id"]: call.kwargs["spectrum"] for call in worker.call_args_list}
    assert spectra["sample_001"] == "stochastic_0"


@pytest.mark.parametrize("include_reference, expected_count", [(True, 3), (False, 2)])
def test_batch_reports_failures_without_silent_omission(tmp_path, include_reference, expected_count):
    """Report a failed sample and the number of scheduled jobs."""
    template = tmp_path / "graphite.ncmat"
    template.write_text("NCMAT", encoding="utf-8")

    def fake_worker(**kwargs):
        """Fail the second stochastic sample."""
        sid = kwargs["sample_id"]
        if sid == "sample_002":
            return sid, None, "bad physics"
        return sid, kwargs["output_path"], None

    batch = NcrystalBatchOrchestrator(n_jobs=1)
    with patch.object(batch, "_generate_tape_from_template", side_effect=fake_worker):
        with pytest.raises(RuntimeError, match=f"1 of {expected_count} spectra: sample_002: bad physics"):
            batch.generate_ensemble_sab_tapes(
                ensemble=DummyEnsemble(2),
                output_dir=str(tmp_path / "endf"),
                base_ncmat_file=str(template),
                temperature=293.6,
                atom_label="C",
                include_reference=include_reference,
            )


def test_worker_validates_and_always_cleans_temporary_ncmat(tmp_path):
    state = SimpleNamespace(filepath=str(tmp_path / "temporary.ncmat"))
    with patch("processing.orchestrators.ncrystal_batch.NcrystalProcessor") as processor_cls, \
         patch("processing.orchestrators.ncrystal_batch.EndfValidator") as validator_cls:
        processor = processor_cls.return_value
        processor.create_analysis_state.return_value = state
        validator_cls.return_value.validate_sab_tape.return_value = True
        result = NcrystalBatchOrchestrator._generate_tape_from_template(
            sample_id="sample_000", sample_index=0, spectrum="spectrum",
            output_path=str(tmp_path / "sample_000_sab.endf"),
            base_ncmat_file="graphite.ncmat", temperature=293.6,
            atom_label="C", elastic_mode="mixed",
        )
        assert result[0] == "sample_000" and result[2] is None
        processor.generate_sab_tape.assert_called_once()
        processor.cleanup_analysis_file.assert_called_once_with(filepath=state.filepath)


def test_worker_rejects_invalid_endf_and_removes_output(tmp_path):
    output = tmp_path / "sample_000_sab.endf"
    output.write_text("invalid", encoding="utf-8")
    state = SimpleNamespace(filepath=str(tmp_path / "temporary.ncmat"))
    with patch("processing.orchestrators.ncrystal_batch.NcrystalProcessor") as processor_cls, \
         patch("processing.orchestrators.ncrystal_batch.EndfValidator") as validator_cls:
        processor_cls.return_value.create_analysis_state.return_value = state
        validator_cls.return_value.validate_sab_tape.return_value = False
        result = NcrystalBatchOrchestrator._generate_tape_from_template(
            sample_id="sample_000", sample_index=0, spectrum="spectrum",
            output_path=str(output), base_ncmat_file="graphite.ncmat",
            temperature=293.6, atom_label="C", elastic_mode="mixed",
        )
    assert result[1] is None
    assert not output.exists()
    processor_cls.return_value.cleanup_analysis_file.assert_called_once_with(filepath=state.filepath)


def test_batch_requires_reference(tmp_path):
    template = tmp_path / "graphite.ncmat"
    template.write_text("NCMAT", encoding="utf-8")
    ensemble = DummyEnsemble(3)
    ensemble.reference = None
    with pytest.raises(ValueError, match="no reference"):
        NcrystalBatchOrchestrator(n_jobs=1).generate_ensemble_sab_tapes(
            ensemble=ensemble, output_dir=str(tmp_path),
            base_ncmat_file=str(template), temperature=293.6, atom_label="C",
        )


# NCrystal sample failure handling

class DummyPhononSpectra:
    def __init__(self, name):
        self.name = name
        self.energy_grid = [0.0, 1.0]
        self.density_values = [0.0, 1.0]


# ==========================================
# NCRYSTAL ORCHESTRATOR TESTS
# ==========================================

@patch("processing.orchestrators.ncrystal_batch.NcrystalProcessor.generate_sab_tape")
@patch("processing.orchestrators.ncrystal_batch.NCrystalWriter")  
@patch("processing.orchestrators.ncrystal_batch.EndfValidator.validate_sab_tape")
def test_ncrystal_batch_fault_tolerance(mock_validate, mock_writer_class, mock_generate, tmp_path):
    """
    Test fault tolerance and validation filtering in NCrystal batch orchestration.

    Verifies that NcrystalBatchOrchestrator.process_ensemble:
    - Traps and logs backend execution errors during tape generation without aborting the batch.
    - Discards samples that fail post-generation ENDF MF7 validation.
    - Successfully aggregates and returns only valid generated tape paths.
    """
    orchestrator = NcrystalBatchOrchestrator(n_jobs=2)
    
    ensemble = {
        "sample_good": DummyPhononSpectra("good"),
        "sample_crash": DummyPhononSpectra("crash"), 
        "sample_invalid": DummyPhononSpectra("bad") 
    }
    config = {"temperature": 296.0, "material_prefix": "Test", "elastic_mode": "mixed"}

    mock_writer_instance = MagicMock()
    mock_writer_class.return_value = mock_writer_instance
    
    def fake_write(out_path):
        Path(out_path).write_text("dummy ncmat data")
    mock_writer_instance.write.side_effect = fake_write
    
    def fake_generate(*args, **kwargs):
        if "sample_crash" in kwargs.get("material_name", ""):
            raise RuntimeError("NCrystal physics exception")
    mock_generate.side_effect = fake_generate
    
    def fake_validate(out_path):
        return "sample_invalid" not in out_path
    mock_validate.side_effect = fake_validate
    
    out_dir = tmp_path / "ncrystal_outputs"
    
    # Use threading backend to avoid multiprocessing pickle errors with MagicMocks
    with parallel_backend("threading"):
        results = orchestrator.process_ensemble(ensemble, str(out_dir), config)
    
    assert len(results) == 1
    assert "sample_good" in results
    assert "sample_crash" not in results
    assert "sample_invalid" not in results
    assert mock_validate.call_count == 2


# ==========================================
# NJOY ORCHESTRATOR TESTS (PHASE 1 & 2)
# ==========================================





# ==========================================
# NJOY TEMPLATE-BASED ENSEMBLE API
# ==========================================

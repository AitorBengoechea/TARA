"""Check that all TARA domains delegate concurrency to one root module."""

import ast
from pathlib import Path
from unittest.mock import patch

from joblib import parallel_backend

from processing.orchestrators.ncrystal_batch import NcrystalBatchOrchestrator
from processing.orchestrators.njoy_batch import NjoyBatchOrchestrator


ROOT = Path(__file__).resolve().parents[2]


def test_parallel_domains_have_no_direct_joblib_execution():
    """No domain should silently bypass centralized CPU limits."""
    paths = [
        "processing/orchestrators/ncrystal_batch.py",
        "processing/orchestrators/njoy_batch.py",
        "analyzer/ensemble_analyzer.py",
    ]
    for relative in paths:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.ImportFrom) and node.module == "joblib"
            for node in ast.walk(tree)
        ), relative
        assert any(
            isinstance(node, ast.ImportFrom) and node.module == "parallel_execution"
            for node in ast.walk(tree)
        ), relative


def test_ncrystal_orchestrator_caps_excessive_requests(tmp_path, monkeypatch):
    """NCrystal honors an oversized n_jobs request without raising an error."""
    import parallel_execution

    monkeypatch.setattr(parallel_execution, "_allocated_cpus", lambda: 2)
    template = tmp_path / "dummy.ncmat"
    template.write_text("NCMAT", encoding="utf-8")
    batch = NcrystalBatchOrchestrator(n_jobs=1000)

    class Ensemble:
        reference = "original"

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return f"random_{index}"

    def fake_worker(**kwargs):
        return kwargs["sample_id"], kwargs["output_path"], None

    from types import SimpleNamespace

    # This test isolates CPU allocation; a separate suite checks LEAPR parsing.
    template_data = SimpleNamespace(temp_blocks=[{"temp": 296.0}])
    with patch("processing.orchestrators.njoy_batch.LeaprFile.from_file", return_value=template_data), \
         patch.object(batch, "_generate_tape_from_template", side_effect=fake_worker), \
         parallel_backend("threading"):
        result = batch.generate_ensemble_sab_tapes(
            ensemble=Ensemble(), output_dir=str(tmp_path / "out"),
            base_ncmat_file=str(template), temperature=300.0, atom_label="C",
        )
    assert list(result) == ["sample_000", "sample_001", "sample_002"]


def test_njoy_orchestrator_caps_excessive_requests(tmp_path, monkeypatch):
    """NJOY keeps reference identity while limiting worker allocation."""
    import parallel_execution

    monkeypatch.setattr(parallel_execution, "_allocated_cpus", lambda: 2)
    template = tmp_path / "dummy.leapr"
    template.write_text("template", encoding="utf-8")
    batch = NjoyBatchOrchestrator(njoy_exec="unused", n_jobs=999)

    class Ensemble:
        reference = "original"

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return f"random_{index}"

    def fake_worker(**kwargs):
        return kwargs["sample_id"], kwargs["output_path"], None

    from types import SimpleNamespace

    # This test isolates CPU allocation; a separate suite checks LEAPR parsing.
    template_data = SimpleNamespace(temp_blocks=[{"temp": 296.0}])
    with patch("processing.orchestrators.njoy_batch.LeaprFile.from_file", return_value=template_data), \
         patch.object(batch, "_generate_tape_from_template", side_effect=fake_worker), \
         parallel_backend("threading"):
        result = batch.generate_ensemble_sab_tapes(
            ensemble=Ensemble(), output_dir=str(tmp_path / "out"),
            base_leapr_file=str(template),
        )
    assert list(result) == ["sample_000", "sample_001", "sample_002"]

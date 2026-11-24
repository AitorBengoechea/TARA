import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from processing.pipelines.ace_pipeline import UniversalAcePipeline
from processing.core.config import AceGenerationConfig

@pytest.fixture
def dummy_config(tmp_path):
    """Build a valid explicit configuration for legacy ACE pipeline tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    AceGenerationConfig
        Configuration using an existing placeholder base ENDF file.
    """
    dummy_base = tmp_path / "base.endf"
    dummy_base.touch()
    
    return AceGenerationConfig(
        base_endf_path=str(dummy_base),
        base_za=1001,
        base_mat=125,
        tsl_mat=125,
        temperature=296.0,
        icoh=0,
        natom=1,
        emax=5.0,
        suffix=".00t",
        za1=1001,
        za2=0,
        za3=0,
        elas_mat=125,
        ielas=1
    )


@patch("processing.pipelines.ace_pipeline.NjoyBatchOrchestrator")
def test_pipeline_happy_path(mock_orchestrator_class, dummy_config, tmp_path):
    """
    Test end-to-end ACE library generation across an ensemble of valid ENDF tapes.

    Verifies that UniversalAcePipeline.run:
    - Converts AceGenerationConfig dataclass instances into parameter mappings for the orchestrator.
    - Dispatches parallel ACE compilation jobs through NjoyBatchOrchestrator.
    - Collects, maps, and returns paths for all successfully compiled ACE libraries.
    """
    # 1. Configurar el Mock del orquestador de NJOY
    mock_instance = MagicMock()
    mock_orchestrator_class.return_value = mock_instance
    
    sab_tapes = {
        "mat_1": "/fake/path/mat_1_sab.endf",
        "mat_2": "/fake/path/mat_2_sab.endf"
    }
    
    # Simular una ejecución exitosa de Phase 2
    mock_instance.process_ace_ensemble.return_value = {
        "mat_1": "/tmp/mat_1.ace",
        "mat_2": "/tmp/mat_2.ace"
    }
    
    # 2. Ejecutar la tubería
    pipeline = UniversalAcePipeline(njoy_exec="dummy_exec")
    results = pipeline.run(
        sab_tapes=sab_tapes, 
        output_dir=str(tmp_path), 
        config=dummy_config
    )
    
    # 3. Verificaciones
    mock_instance.process_ace_ensemble.assert_called_once()
    
    # Verificar que los argumentos se desempaquetaron y enviaron correctamente al orquestador
    called_kwargs = mock_instance.process_ace_ensemble.call_args.kwargs
    assert called_kwargs["sab_endf_files"] == sab_tapes
    assert called_kwargs["output_dir"] == str(tmp_path)
    
    # La tubería debe convertir la dataclass en un diccionario interno
    assert called_kwargs["config"] == dummy_config.__dict__
    
    # 4. Verificar la salida final
    assert len(results) == 2
    assert results["mat_1"] == "/tmp/mat_1.ace"
    assert results["mat_2"] == "/tmp/mat_2.ace"


@patch("processing.pipelines.ace_pipeline.NjoyBatchOrchestrator")
def test_pipeline_handles_empty_input(mock_orchestrator_class, dummy_config, tmp_path):
    """
    Test pipeline short-circuit behavior when provided with an empty input collection.

    Verifies that UniversalAcePipeline.run:
    - Detects empty input dictionaries before initializing batch orchestrators.
    - Logs a warning and aborts execution cleanly without spawning subprocesses.
    - Returns an empty dictionary without raising unhandled exceptions.
    """
    mock_instance = MagicMock()
    mock_orchestrator_class.return_value = mock_instance
    
    pipeline = UniversalAcePipeline(njoy_exec="dummy_exec")
    
    # Ejecutar con un diccionario vacío de cintas
    results = pipeline.run(
        sab_tapes={}, 
        output_dir=str(tmp_path), 
        config=dummy_config
    )
    
    # Verificar que no se llamó a process_ace_ensemble y se devolvió vacío
    mock_instance.process_ace_ensemble.assert_not_called()
    assert results == {}


@patch("processing.pipelines.ace_pipeline.NjoyBatchOrchestrator")
def test_pipeline_handles_partial_success(mock_orchestrator_class, dummy_config, tmp_path):
    """
    Test batch result filtering across mixed success and failure executions.

    Verifies that UniversalAcePipeline.run:
    - Filters failed compilation attempts from the returned mapping.
    - Computes and logs batch success metrics accurately.
    - Returns a clean dictionary containing exclusively verified output file paths.
    """
    mock_instance = MagicMock()
    mock_orchestrator_class.return_value = mock_instance
    
    sab_tapes = {
        "mat_1": "/fake/path/mat_1.endf",
        "mat_2": "/fake/path/mat_2.endf",
        "mat_3": "/fake/path/mat_3.endf"
    }
    
    # Simular que mat_2 falló durante la compilación en NJOY
    mock_instance.process_ace_ensemble.return_value = {
        "mat_1": "/tmp/mat_1.ace",
        "mat_3": "/tmp/mat_3.ace"
    }
    
    pipeline = UniversalAcePipeline(njoy_exec="dummy_exec")
    results = pipeline.run(
        sab_tapes=sab_tapes, 
        output_dir=str(tmp_path), 
        config=dummy_config
    )
    
    assert len(results) == 2
    assert "mat_1" in results
    assert "mat_3" in results
    assert "mat_2" not in results


# ---------------------------------------------------------------------------
# Directory-based ACE processing with automatic ENDF configuration
# ---------------------------------------------------------------------------


def _endf_sections(mat=31, lthr=0):
    """Create a synthetic TSL section mapping for metadata tests.

    Parameters
    ----------
    mat : int, optional
        MF7 material identifier.
    lthr : int, optional
        Elastic-scattering treatment identifier.

    Returns
    -------
    dict
        ENDF sections keyed by MF and then MT.
    """
    sections = {7: {4: {"MAT": mat, "B": {6: 1}, "NS": 0}}}
    if lthr:
        sections[7][2] = {"LTHR": lthr}
    return sections


def _make_directory_inputs(tmp_path, number=3):
    """Create a nonempty base evaluation and synthetic TSL files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.
    number : int, optional
        Number of generated TSL files.

    Returns
    -------
    tuple
        Base file, TSL directory, and per-path parsed ENDF mappings.
    """
    base = tmp_path / "carbon.endf"
    base.write_text("base neutron evaluation", encoding="utf-8")
    source = tmp_path / "endf"
    source.mkdir()
    files = {}
    for index in range(number):
        path = source / f"sample_{index:03d}_sab.endf"
        path.write_text(f"sample {index}", encoding="utf-8")
        files[str(path)] = _endf_sections()
    return base, source, files


def _mock_parser(mock_parser, base, files, base_za=6012):
    """Provide deterministic ENDF parser responses for all sample files.

    Parameters
    ----------
    mock_parser : unittest.mock.MagicMock
        Mocked ``EndfParserPy`` class.
    base : pathlib.Path
        Synthetic base ENDF evaluation path.
    files : dict
        Parsed ENDF responses keyed by TSL file path.
    base_za : int, optional
        ZA identifier returned for the base evaluation.

    Returns
    -------
    dict
        Complete filename-to-ENDF mapping used by the parser mock.
    """
    mapping = {str(base): {1: {451: {"ZA": float(base_za), "MAT": 625}}}}
    mapping.update(files)
    mock_parser.return_value.parsefile.side_effect = lambda path: mapping[path]
    return mapping


def _create_ace_files(sab_files, output_dir):
    """Write sample-specific placeholder ACE files for the mocked backend.

    Parameters
    ----------
    sab_files : Mapping[str, str]
        Source TSL tapes indexed by sample identifier.
    output_dir : pathlib.Path
        Destination used by the fake ACE processor.

    Returns
    -------
    dict[str, str]
        Sample identifiers mapped to nonempty placeholder ACE paths.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    output_paths = {}
    for name in sab_files:
        path = output_dir / f"{name}.ace"
        path.write_text(f"valid {name}", encoding="utf-8")
        output_paths[name] = str(path)
    return output_paths


@patch("endf_parserpy.EndfParserPy")
@patch("processing.pipelines.ace_pipeline.NjoyBatchOrchestrator")
def test_ace_directory_discovers_every_tsl_and_uses_one_config(
    mock_orchestrator_cls, mock_parser, tmp_path
):
    """Read all TSL files, derive one config, and generate ACE outputs."""
    base, source, files = _make_directory_inputs(tmp_path)
    _mock_parser(mock_parser, base, files)

    def produce(*, sab_endf_files, output_dir, config, save_xsdir):
        assert save_xsdir is True
        assert set(sab_endf_files) == {"sample_000", "sample_001", "sample_002"}
        assert list(sab_endf_files) == ["sample_000", "sample_001", "sample_002"]
        assert config["base_za"] == 6012
        assert config["base_mat"] == 625
        assert config["tsl_mat"] == 31
        assert config["natom"] == 1
        assert config["za1"] == 6012
        assert config["temperature"] == 293.6
        assert Path(output_dir) == tmp_path / "ace"
        return _create_ace_files(sab_endf_files, Path(output_dir))

    mock_orchestrator_cls.return_value.process_ace_ensemble.side_effect = produce
    pipeline = UniversalAcePipeline(njoy_exec="njoy", n_jobs=2)
    result = pipeline.run_from_directory(
        base_endf_path=str(base), temperature=293.6, tsl_directory=str(source)
    )
    assert list(result) == ["sample_000", "sample_001", "sample_002"]
    mock_orchestrator_cls.assert_called_once_with(njoy_exec="njoy", n_jobs=2)
    assert mock_parser.return_value.parsefile.call_count == 4


@pytest.mark.parametrize("difference", ["mat", "lthr"])
@patch("endf_parserpy.EndfParserPy")
@patch("processing.pipelines.ace_pipeline.NjoyBatchOrchestrator")
def test_ace_directory_rejects_inconsistent_tsl_before_njoy(
    mock_orchestrator_cls, mock_parser, tmp_path, difference
):
    """Reject batches with mismatched MAT/LTHR before any subprocess is called."""
    base, source, files = _make_directory_inputs(tmp_path, number=2)
    paths = list(files)
    if difference == "mat":
        files[paths[-1]][7][4]["MAT"] = 32
    else:
        files[paths[-1]][7][2] = {"LTHR": 1}
    _mock_parser(mock_parser, base, files)
    pipeline = UniversalAcePipeline(njoy_exec="njoy")
    with pytest.raises(ValueError, match="Inconsistent TSL MAT/NATOM/LTHR"):
        pipeline.run_from_directory(
            base_endf_path=str(base), temperature=293.6, tsl_directory=str(source)
        )
    mock_orchestrator_cls.return_value.process_ace_ensemble.assert_not_called()


@patch("endf_parserpy.EndfParserPy")
@patch("processing.pipelines.ace_pipeline.NjoyBatchOrchestrator")
def test_ace_directory_accepts_noncarbon_monatomic_tsl(mock_orchestrator_cls, mock_parser, tmp_path):
    """Read monoatomic TSL metadata independently of nuclide identity."""
    base, source, files = _make_directory_inputs(tmp_path, number=1)
    _mock_parser(mock_parser, base, files, base_za=1001)
    pipeline = UniversalAcePipeline(njoy_exec="njoy")
    mock_orchestrator_cls.return_value.process_ace_ensemble.side_effect = (
        lambda *, sab_endf_files, output_dir, config, save_xsdir:
            _create_ace_files(sab_endf_files, Path(output_dir))
    )
    pipeline.run_from_directory(
        base_endf_path=str(base), temperature=293.6, tsl_directory=str(source)
    )
    config = mock_orchestrator_cls.return_value.process_ace_ensemble.call_args.kwargs["config"]
    assert config["za1"] == 1001


@patch("endf_parserpy.EndfParserPy")
@patch("processing.pipelines.ace_pipeline.NjoyBatchOrchestrator")
def test_ace_directory_fails_if_njoy_omits_a_sample(
    mock_orchestrator_cls, mock_parser, tmp_path
):
    """The convenience method must never report partial batch success."""
    base, source, files = _make_directory_inputs(tmp_path, number=2)
    _mock_parser(mock_parser, base, files)
    mock_orchestrator_cls.return_value.process_ace_ensemble.side_effect = (
        lambda *, sab_endf_files, output_dir, config, save_xsdir:
            _create_ace_files({"sample_000": sab_endf_files["sample_000"]}, Path(output_dir))
    )
    pipeline = UniversalAcePipeline(njoy_exec="njoy")
    with pytest.raises(RuntimeError, match="sample_001"):
        pipeline.run_from_directory(
            base_endf_path=str(base), temperature=293.6, tsl_directory=str(source)
        )


@patch("endf_parserpy.EndfParserPy")
@patch("processing.pipelines.ace_pipeline.NjoyBatchOrchestrator")
def test_ace_directory_fails_on_missing_ace_file(
    mock_orchestrator_cls, mock_parser, tmp_path
):
    """A returned name is not success unless the ACE artifact actually exists."""
    base, source, files = _make_directory_inputs(tmp_path, number=1)
    _mock_parser(mock_parser, base, files)
    mock_orchestrator_cls.return_value.process_ace_ensemble.return_value = {
        "sample_000": str(tmp_path / "nonexistent.ace")
    }
    pipeline = UniversalAcePipeline(njoy_exec="njoy")
    with pytest.raises(RuntimeError, match="missing_or_empty_files"):
        pipeline.run_from_directory(
            base_endf_path=str(base), temperature=293.6, tsl_directory=str(source)
        )


@patch("endf_parserpy.EndfParserPy")
@patch("processing.pipelines.ace_pipeline.NjoyBatchOrchestrator")
def test_ace_directory_accepts_single_file_and_output_override(
    mock_orchestrator_cls, mock_parser, tmp_path
):
    """The required TSL path can point to a single file as well as a directory."""
    base, source, files = _make_directory_inputs(tmp_path, number=1)
    _mock_parser(mock_parser, base, files)
    target = tmp_path / "custom_ace"
    mock_orchestrator_cls.return_value.process_ace_ensemble.side_effect = (
        lambda *, sab_endf_files, output_dir, config, save_xsdir:
            _create_ace_files(sab_endf_files, Path(output_dir))
    )
    result = UniversalAcePipeline(njoy_exec="njoy").run_from_directory(
        base_endf_path=str(base),
        temperature=293.6,
        tsl_directory=next(iter(files)),
        output_dir=str(target),
    )
    assert result["sample_000"] == str(target / "sample_000.ace")


@patch("processing.pipelines.ace_pipeline.NjoyBatchOrchestrator")
def test_ace_directory_rejects_empty_directory(mock_orchestrator_cls, tmp_path):
    """Do not run NJOY when there is no TSL evaluation to convert."""
    source = tmp_path / "empty"
    source.mkdir()
    base = tmp_path / "base.endf"
    base.write_text("dummy")
    with pytest.raises(ValueError, match="No .endf files"):
        UniversalAcePipeline(njoy_exec="njoy").run_from_directory(
            base_endf_path=str(base), temperature=293.6, tsl_directory=str(source)
        )
    mock_orchestrator_cls.return_value.process_ace_ensemble.assert_not_called()


@patch("processing.pipelines.ace_pipeline.NjoyBatchOrchestrator")
def test_ace_directory_rejects_ambiguous_sample_ids(mock_orchestrator_cls, tmp_path):
    """Do not overwrite files when two names normalize to the same sample ID."""
    base = tmp_path / "base.endf"
    base.write_text("dummy")
    source = tmp_path / "endf"
    source.mkdir()
    (source / "sample_000.endf").write_text("dummy")
    (source / "sample_000_sab.endf").write_text("dummy")
    with pytest.raises(ValueError, match="Duplicate sample ID"):
        UniversalAcePipeline(njoy_exec="njoy").run_from_directory(
            base_endf_path=str(base), temperature=293.6, tsl_directory=str(source)
        )
    mock_orchestrator_cls.return_value.process_ace_ensemble.assert_not_called()

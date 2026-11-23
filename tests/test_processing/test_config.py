import pytest
from pathlib import Path
from unittest.mock import patch
from processing.core.config import AceGenerationConfig


def test_config_valid_initialization(tmp_path):
    """
    Test instantiation of AceGenerationConfig under valid parameter sets.

    Verifies that AceGenerationConfig:
    - Initializes all required nuclear identifiers (MAT, ZA) and calculation boundaries.
    - Preserves provided physical properties and output suffix definitions.
    - Applies default numerical convergence tolerances and angular bin settings.
    """
    dummy_endf = tmp_path / "base.endf"
    dummy_endf.touch()

    config = AceGenerationConfig(
        base_endf_path=str(dummy_endf),
        base_za=1001,
        base_mat=125,
        tsl_mat=1001,
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
    
    assert config.base_za == 1001
    assert config.temperature == 296.0
    assert config.nbin == 16  # Verify default value
    assert config.tol == 0.001  # Verify default value


def test_config_missing_base_endf_file(tmp_path):
    """
    Test filesystem validation during AceGenerationConfig initialization.

    Verifies that AceGenerationConfig.__post_init__:
    - Confirms the existence of the base ENDF cross-section evaluation on disk.
    - Raises a descriptive FileNotFoundError when the specified base tape cannot be found.
    """
    ghost_path = tmp_path / "nonexistent.endf"

    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(FileNotFoundError, match="Base ENDF evaluation not found"):
        AceGenerationConfig(
            base_endf_path=str(ghost_path),
            base_za=1001,
            base_mat=125,
            tsl_mat=1001,
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


def test_config_missing_required_fields():
    """
    Test parameter constraint enforcement on mandatory nuclear data fields.

    Verifies that AceGenerationConfig:
    - Enforces explicit definition of mandatory physics inputs.
    - Raises a TypeError when essential nuclear data parameters are omitted.
    """
    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(TypeError):
        AceGenerationConfig(
            base_endf_path="dummy.endf"
            # Missing all other required physics parameters
        )


# ---------------------------------------------------------------------------
# ENDF-derived ACE configurations: parsing, controls and consistency
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


@pytest.mark.parametrize("lthr,expected_ielas,expected_elas_mat", [
    (0, 0, 0), (1, 0, 238), (2, 1, 238), (3, 2, 238)
])
@patch("endf_parserpy.EndfParserPy")
def test_ace_endf_factory_elastic_modes(
    mock_parser, tmp_path, lthr, expected_ielas, expected_elas_mat
):
    """Select the supported elastic options without handwritten notebook metadata."""
    base, source, files = _make_directory_inputs(tmp_path, number=2)
    for data in files.values():
        data[7][2] = {"LTHR": lthr}
    _mock_parser(mock_parser, base, files)
    cfg = AceGenerationConfig.from_endf(
        base_endf_path=str(base),
        tsl_endf_files={f"sample_{i}": path for i, path in enumerate(files)},
        temperature=296.0,
    )
    assert cfg.icoh == int(lthr != 0)
    assert cfg.ielas == expected_ielas
    assert cfg.elas_mat == expected_elas_mat
    assert cfg.suffix == ".01"
    assert cfg.nbin == 16
    assert cfg.tol == 0.001


@patch("endf_parserpy.EndfParserPy")
def test_ace_config_rejects_missing_mf7_mt4(mock_parser, tmp_path):
    """Every discovered ENDF must be a thermal-scattering tape with MF7/MT4."""
    base, source, files = _make_directory_inputs(tmp_path, number=1)
    files[next(iter(files))] = {7: {2: {"LTHR": 1}}}
    _mock_parser(mock_parser, base, files)
    cfg_paths = {"sample_000": next(iter(files))}
    with pytest.raises(ValueError, match="MF7/MT4 MAT"):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base), tsl_endf_files=cfg_paths, temperature=293.6
        )


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan")])
@patch("endf_parserpy.EndfParserPy")
def test_ace_config_rejects_invalid_temperature(mock_parser, tmp_path, temperature):
    """Reject nonphysical temperatures before parsing or generating ACE data."""
    base, _, files = _make_directory_inputs(tmp_path, number=1)
    with pytest.raises(ValueError, match="finite positive"):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base),
            tsl_endf_files={"sample_000": next(iter(files))},
            temperature=temperature,
        )
    mock_parser.assert_not_called()


@pytest.mark.parametrize("natom", [1, 2, 4])
@patch("endf_parserpy.EndfParserPy")
def test_ace_from_endf_reads_principal_atom_multiplicity(mock_parser, tmp_path, natom):
    """Take THERMR NATOM from ENDF MF7/MT4 B(6), never from an element preset."""
    base, _source, files = _make_directory_inputs(tmp_path, number=2)
    for tape in files.values():
        tape[7][4]["B"][6] = float(natom)
    _mock_parser(mock_parser, base, files)
    config = AceGenerationConfig.from_endf(
        base_endf_path=str(base), tsl_endf_files={path: path for path in files}, temperature=293.6
    )
    assert config.natom == natom
    assert config.za1 == 6012


@patch("endf_parserpy.EndfParserPy")
def test_ace_from_endf_requires_natom_in_tsl(mock_parser, tmp_path):
    """Never substitute a guessed atom count for missing ENDF B(6)."""
    base, _source, files = _make_directory_inputs(tmp_path, number=1)
    next(iter(files.values()))[7][4].pop("B")
    _mock_parser(mock_parser, base, files)
    with pytest.raises(ValueError, match=r"B\(6\)"):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base), tsl_endf_files={path: path for path in files}, temperature=293.6
        )


@patch("endf_parserpy.EndfParserPy")
def test_ace_from_endf_rejects_extra_analytical_atom_types(mock_parser, tmp_path):
    """Multi-species TSLs need additional moderator ZA inputs not in one base file."""
    base, _source, files = _make_directory_inputs(tmp_path, number=1)
    next(iter(files.values()))[7][4]["NS"] = 1
    _mock_parser(mock_parser, base, files)
    with pytest.raises(ValueError, match="additional atom types"):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base), tsl_endf_files={path: path for path in files}, temperature=293.6
        )


@patch("endf_parserpy.EndfParserPy")
def test_ace_from_endf_rejects_inconsistent_natom(mock_parser, tmp_path):
    """A single ACE configuration cannot process tapes with different NATOM."""
    base, _source, files = _make_directory_inputs(tmp_path, number=2)
    list(files.values())[-1][7][4]["B"][6] = 2
    _mock_parser(mock_parser, base, files)
    with pytest.raises(ValueError, match="Inconsistent TSL MAT/NATOM/LTHR"):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base), tsl_endf_files={path: path for path in files}, temperature=293.6
        )


@patch("endf_parserpy.EndfParserPy")
def test_ace_from_endf_preserves_advanced_numeric_options(mock_parser, tmp_path):
    """Advanced numerical configuration must remain independent of ENDF metadata."""
    base, _source, files = _make_directory_inputs(tmp_path, number=1)
    _mock_parser(mock_parser, base, files)
    config = AceGenerationConfig.from_endf(
        base_endf_path=str(base), tsl_endf_files={path: path for path in files}, temperature=293.6,
        emax=3.2, suffix=".02", nbin=24, tol=0.0005,
    )
    assert (config.emax, config.suffix, config.nbin, config.tol) == (
        3.2, ".02", 24, 0.0005
    )


@patch("endf_parserpy.EndfParserPy")
def test_ace_from_endf_requires_secondary_atom_count(mock_parser, tmp_path):
    """Never silently interpret an unknown ENDF atom-type count as NS=0."""
    base, _source, files = _make_directory_inputs(tmp_path, number=1)
    next(iter(files.values()))[7][4].pop("NS")
    _mock_parser(mock_parser, base, files)
    with pytest.raises(ValueError, match="NS atom-type count"):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base),
            tsl_endf_files={path: path for path in files},
            temperature=293.6,
        )


def test_ace_graphite_factory_removed():
    """The graphite-specific alias is intentionally no longer part of the API."""
    assert not hasattr(AceGenerationConfig, "from_graphite_endf_files")


@patch("endf_parserpy.EndfParserPy")
def test_ace_from_endf_rejects_missing_base_header(mock_parser, tmp_path):
    """Missing MF1/MT451 data must not produce guessed base identifiers."""
    base, _source, files = _make_directory_inputs(tmp_path, number=1)
    mapping = _mock_parser(mock_parser, base, files)
    mapping[str(base)] = {3: {1: {"XS": 1.0}}}

    with pytest.raises(ValueError, match="MF1/MT451"):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base),
            tsl_endf_files={"sample_000": next(iter(files))},
            temperature=293.6,
        )


@patch("endf_parserpy.EndfParserPy")
def test_ace_from_endf_rejects_invalid_enumerated_identifiers(mock_parser, tmp_path):
    """Integer-valued positive ZA and MAT identifiers are mandatory."""
    base, _source, files = _make_directory_inputs(tmp_path, number=1)
    mapping = _mock_parser(mock_parser, base, files)
    mapping[str(base)][1][451]["MAT"] = -17

    with pytest.raises(ValueError, match="base MAT"):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base),
            tsl_endf_files={"sample_000": next(iter(files))},
            temperature=293.6,
        )


@patch("endf_parserpy.EndfParserPy")
def test_ace_from_endf_wraps_parser_errors_with_file_context(mock_parser, tmp_path):
    """The ENDF parser failure must identify the affected input file."""
    base, _source, files = _make_directory_inputs(tmp_path, number=1)
    mock_parser.return_value.parsefile.side_effect = RuntimeError("bad ENDF card")

    with pytest.raises(ValueError, match="base neutron ENDF evaluation.*bad ENDF card"):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base),
            tsl_endf_files={"sample_000": next(iter(files))},
            temperature=293.6,
        )


@pytest.mark.parametrize("field, invalid", [("emax", 0.0), ("nbin", 0), ("tol", float("nan"))])
@patch("endf_parserpy.EndfParserPy")
def test_ace_from_endf_rejects_invalid_advanced_options(mock_parser, tmp_path, field, invalid):
    """Validate optional processing controls before reading any ENDF file."""
    base, _source, files = _make_directory_inputs(tmp_path, number=1)
    with pytest.raises(ValueError, match="ACE"):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base),
            tsl_endf_files={"sample_000": next(iter(files))},
            temperature=293.6,
            **{field: invalid},
        )
    mock_parser.assert_not_called()


@patch("endf_parserpy.EndfParserPy")
def test_ace_from_endf_requires_nonempty_input_mapping(mock_parser, tmp_path):
    """The factory must reject empty sample batches before parsing."""
    base = tmp_path / "base.endf"
    base.write_text("dummy", encoding="utf-8")
    with pytest.raises(ValueError, match="No thermal-scattering"):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base), tsl_endf_files={}, temperature=293.6
        )
    mock_parser.assert_not_called()


@patch("endf_parserpy.EndfParserPy")
def test_ace_from_endf_requires_nonempty_tsl_file(mock_parser, tmp_path):
    """An empty generated TSL tape cannot contribute metadata to an ACE batch."""
    base, _source, files = _make_directory_inputs(tmp_path, number=1)
    _mock_parser(mock_parser, base, files)
    Path(next(iter(files))).write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing or empty.*sample_000"):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base),
            tsl_endf_files={"sample_000": next(iter(files))},
            temperature=293.6,
        )

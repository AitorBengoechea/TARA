"""Ncrystal solid export regression tests."""


from pathlib import Path
from unittest.mock import patch
import pytest
from processing.backends.ncrystal_processor import NcrystalProcessor
from processing.backends.ncrystal_processor import UnsupportedLiquidAceExport


# Ncrystal export guard regression

@patch("processing.backends.ncrystal_processor.ncmat2endf.ncmat2endf")
def test_legacy_mocked_converter_accepts_nonexistent_template(mock_converter, tmp_path):
    """Original V1 persistent ENDF API delegates the input path to NCrystal."""
    def write_endf(*args, **kwargs):
        output = Path(kwargs["outdir"]) / f"{kwargs['material_name']}.endf"
        output.write_text("dummy thermal scattering data", encoding="utf-8")

    mock_converter.side_effect = write_endf
    result = tmp_path / "final_sab.endf"
    NcrystalProcessor().generate_sab_tape(
        ncmat_filepath="dummy.ncmat",
        material_name="dummy_mat",
        temperature=296.0,
        output_endf_path=str(result),
    )
    assert result.read_text(encoding="utf-8") == "dummy thermal scattering data"
    mock_converter.assert_called_once()


@patch("processing.backends.ncrystal_processor.ncmat2endf.ncmat2endf")
def test_legacy_mocked_converter_error_translation(mock_converter, tmp_path):
    """Exceptions originating in the converter retain V1 RuntimeError form."""
    mock_converter.side_effect = ValueError("Simulated NCrystal physics error")
    with pytest.raises(RuntimeError, match="NCrystal ENDF generation failed"):
        NcrystalProcessor().generate_sab_tape(
            ncmat_filepath="dummy.ncmat",
            material_name="dummy_mat",
            temperature=296.0,
            output_endf_path=str(tmp_path / "out.endf"),
        )
    mock_converter.assert_called_once()


@patch("processing.backends.ncrystal_processor.ncmat2endf.ncmat2endf")
def test_existing_plugin_liquid_is_not_sent_to_solid_converter(mock_converter, tmp_path):
    """A real liquid NCMAT must never lose its convolution during export."""
    material = tmp_path / "liquid.ncmat"
    material.write_text(
        "NCMAT v7\n@DYNINFO\n  element Xe\n  fraction 1\n"
        "  type freegas\n@CUSTOM_LIQUIDS\n"
        "  Xe incoherent 3 0.2 0.8\n",
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedLiquidAceExport):
        NcrystalProcessor().generate_sab_tape(
            ncmat_filepath=str(material),
            material_name="liquid",
            temperature=295.0,
            output_endf_path=str(tmp_path / "out.endf"),
        )
    mock_converter.assert_not_called()
    assert not (tmp_path / "out.endf").exists()


@patch("processing.backends.ncrystal_processor.ncmat2endf.ncmat2endf")
def test_existing_solid_template_still_works(mock_converter, tmp_path):
    """Real, existing solid NCMAT files retain the persistent ENDF pathway."""
    material = tmp_path / "solid.ncmat"
    material.write_text("NCMAT v7\n@DYNINFO\n", encoding="utf-8")

    def write_endf(*args, **kwargs):
        (Path(kwargs["outdir"]) / "solid.endf").write_text("solid data", encoding="utf-8")

    mock_converter.side_effect = write_endf
    output = tmp_path / "solid_result.endf"
    NcrystalProcessor().generate_sab_tape(
        ncmat_filepath=str(material),
        material_name="solid",
        temperature=295.0,
        output_endf_path=str(output),
    )
    assert output.read_text(encoding="utf-8") == "solid data"
    mock_converter.assert_called_once()


# Ncrystal liquids native

def test_solid_converter_refuses_liquid_plugin_fixture(tmp_path):
    from processing.backends.ncrystal_processor import (
        NcrystalProcessor, UnsupportedLiquidAceExport,
    )
    source = Path(__file__).parents[1]/'fixtures'/'liquid_water_295K_illustrative.ncmat'
    with pytest.raises(UnsupportedLiquidAceExport):
        NcrystalProcessor(base_ncmat_file=str(source), temperature=295)
    with pytest.raises(UnsupportedLiquidAceExport):
        NcrystalProcessor().generate_sab_tape(str(source),295,'bad',str(tmp_path/'bad.endf'))
    assert not (tmp_path/'bad.endf').exists()

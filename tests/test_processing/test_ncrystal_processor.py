"""Ncrystal processor regression tests."""


from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
pytest.importorskip("NCrystal")
import NCrystal.misc
from core.data.input_pdos import PhononSpectra
from processing.backends.ncrystal_processor import MIN_NCRYSTAL_VDOS_ENERGY_EV
from processing.backends.ncrystal_processor import NcrystalProcessor
from processing.backends.ncrystal_processor import prepare_ncrystal_vdos
from processing.backends.ncrystal_processor import reconstruct_debye_tail


# Ncrystal processor

def _spectrum() -> PhononSpectra:
    """Spectrum for the current workflow.

    Returns
    -------
    PhononSpectra
        Calculated result in the declared return type.
    """
    energy = np.array([0.0, 5e-6, 1e-5, 2e-5, 3e-5, 4e-5])
    density = np.array([0.0, 0.2, 0.8, 1.0, 0.5, 0.0])
    return PhononSpectra(pd.Series(density, index=energy))


def test_prepare_ncrystal_vdos_removes_only_unsupported_low_energy_tail():
    """
    Test the shared TARA-to-NCrystal VDOS boundary conversion.
    """
    spectrum = _spectrum()

    energy, density = prepare_ncrystal_vdos(spectrum)

    assert energy[0] >= MIN_NCRYSTAL_VDOS_ENERGY_EV
    # Require exact array identity here; numerical closeness would be insufficient.
    np.testing.assert_array_equal(energy, spectrum.energy_grid[spectrum.energy_grid >= MIN_NCRYSTAL_VDOS_ENERGY_EV])
    # Require exact array identity here; numerical closeness would be insufficient.
    np.testing.assert_array_equal(density, spectrum.density_values[spectrum.energy_grid >= MIN_NCRYSTAL_VDOS_ENERGY_EV])


def test_reconstruct_debye_tail_connects_quadratically_to_first_ncrystal_point():
    """Test quadratic low-energy reconstruction from the retained NCrystal grid."""
    energy = np.array([1e-5, 2e-5, 3e-5])
    density = np.array([2.0, 3.0, 1.0])

    full_energy, full_density = reconstruct_debye_tail(energy, density, n_points=8)

    assert full_energy[0] == 0.0
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(full_energy[-3:], energy)
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(full_density[-3:], density)
    tail_mask = (full_energy > 0.0) & (full_energy < energy[0])
    ratios = full_density[tail_mask] / full_energy[tail_mask] ** 2
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(ratios, ratios[0])


def test_ncrystal_processor_requires_complete_analysis_template_configuration():
    """Test paired base-NCMAT/temperature requirements for transient analysis."""
    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(ValueError, match="provided together"):
        NcrystalProcessor(base_ncmat_file="base.ncmat")


def test_ncrystal_processor_create_and_cleanup_analysis_state(tmp_path):
    """
    Test Processing ownership of transient NCMAT creation, loading, and cleanup.
    """
    base_composer = MagicMock()
    composer = MagicMock()
    base_composer.clone.return_value = composer
    composer.find_label.return_value = "C-internal"

    def fake_write(filepath):
        """Provide a deterministic test substitute for write.

        Parameters
        ----------
        filepath : object
            Path identifying the input or temporary resource.
        """
        Path(filepath).write_text("dummy ncmat", encoding="utf-8")

    composer.write.side_effect = fake_write
    nuclear_data = SimpleNamespace(elements={"C": SimpleNamespace(_awr=12.0)})

    with patch("processing.backends.ncrystal_processor.NC.NCMATComposer.from_cfgstr", return_value=base_composer), \
         patch("processing.backends.ncrystal_processor.NC.clearCaches"), \
         patch("processing.backends.ncrystal_processor.NC.load", return_value=object()), \
         patch("processing.backends.ncrystal_processor.NuclearData", return_value=nuclear_data):
        processor = NcrystalProcessor(
            base_ncmat_file="base.ncmat",
            temperature=300.0,
            temp_dir=str(tmp_path),
        )
        state = processor.create_analysis_state(7, _spectrum(), atom_label="C")

    state_path = Path(state.filepath)
    assert state_path.is_file()
    assert state.target_lbl == "C"
    np.testing.assert_array_equal(state.energy_grid, _spectrum().energy_grid)
    # Preserve two distinct representations: original full Core arrays and the
    # NCrystal-compatible arrays actually written by Processing.
    output_kwargs = composer.set_dyninfo_vdos.call_args.kwargs
    np.testing.assert_array_equal(state.backend_energy_grid, output_kwargs["vdos_egrid"])
    np.testing.assert_array_equal(state.backend_pdos, output_kwargs["vdos"])
    assert state.backend_energy_grid[0] >= MIN_NCRYSTAL_VDOS_ENERGY_EV
    composer.set_dyninfo_vdos.assert_called_once()

    processor.cleanup_analysis_file(state.filepath)
    assert not state_path.exists()


def test_ncrystal_processor_cleans_analysis_file_when_loading_fails(tmp_path):
    """Test transient NCMAT cleanup when NCrystal state loading raises."""
    base_composer = MagicMock()
    composer = MagicMock()
    base_composer.clone.return_value = composer
    composer.find_label.return_value = "C-internal"

    def fake_write(filepath):
        """Provide a deterministic test substitute for write.

        Parameters
        ----------
        filepath : object
            Path identifying the input or temporary resource.
        """
        Path(filepath).write_text("dummy ncmat", encoding="utf-8")

    composer.write.side_effect = fake_write

    with patch("processing.backends.ncrystal_processor.NC.NCMATComposer.from_cfgstr", return_value=base_composer), \
         patch("processing.backends.ncrystal_processor.NC.clearCaches"), \
         patch("processing.backends.ncrystal_processor.NC.load", side_effect=RuntimeError("load failed")):
        processor = NcrystalProcessor(
            base_ncmat_file="base.ncmat",
            temperature=300.0,
            temp_dir=str(tmp_path),
        )
        # Confirm that invalid inputs trigger the documented failure path.
        with pytest.raises(RuntimeError, match="load failed"):
            processor.create_analysis_state(2, _spectrum(), atom_label="C")

    assert not list(tmp_path.glob("pdos_2_*.ncmat"))


@patch("processing.backends.ncrystal_processor.ncmat2endf.ncmat2endf")
def test_ncrystal_processor_persistent_sab_generation(mock_ncmat2endf, tmp_path):
    """Test that the existing persistent Processing ENDF API remains supported."""
    def fake_ncmat2endf(*args, **kwargs):
        """Provide a deterministic test substitute for ncmat2endf.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to the wrapped function.
        **kwargs : Any
            Additional options accepted by the current interface.
        """
        outdir = Path(kwargs["outdir"])
        (outdir / f"{kwargs['material_name']}.endf").write_text("dummy thermal scattering data")

    mock_ncmat2endf.side_effect = fake_ncmat2endf
    out_file = tmp_path / "final_sab.endf"
    processor = NcrystalProcessor()

    processor.generate_sab_tape(
        ncmat_filepath="dummy.ncmat",
        material_name="dummy_mat",
        temperature=296.0,
        output_endf_path=str(out_file),
    )

    assert out_file.read_text() == "dummy thermal scattering data"


@patch("processing.backends.ncrystal_processor.ncmat2endf.ncmat2endf")
def test_ncrystal_processor_persistent_failure_translation(mock_ncmat2endf, tmp_path):
    """Test exception translation for persistent NCrystal ENDF generation."""
    mock_ncmat2endf.side_effect = ValueError("Simulated NCrystal physics error")

    # Confirm that invalid inputs trigger the documented failure path.
    with pytest.raises(RuntimeError, match="NCrystal ENDF generation failed"):
        NcrystalProcessor().generate_sab_tape(
            ncmat_filepath="dummy.ncmat",
            material_name="dummy_mat",
            temperature=296.0,
            output_endf_path=str(tmp_path / "out.endf"),
        )


def test_analysis_component_loading_uses_processing_and_cleans_up_on_failure(tmp_path):
    """Load component handles under the Processing temporary-file lifecycle."""
    processor = NcrystalProcessor(temp_dir=str(tmp_path))
    processor.temperature = 300.0
    processor._base_composer = MagicMock()
    composer = processor._base_composer.clone.return_value
    composer.write.side_effect = lambda path: Path(path).write_text("dummy ncmat")
    nuclear_data = SimpleNamespace(elements={"C": SimpleNamespace(_awr=12.0)})
    handles = [object(), object(), object()]

    with patch("processing.backends.ncrystal_processor.NC.clearCaches"), \
         patch("processing.backends.ncrystal_processor.NC.load"), \
         patch("processing.backends.ncrystal_processor.NuclearData", return_value=nuclear_data), \
         patch("processing.backends.ncrystal_processor.NC.misc.MaterialSource") as material_source:
        loader = material_source.return_value.load
        loader.side_effect = [SimpleNamespace(scatter=handle) for handle in handles]
        state = processor.create_analysis_state(1, _spectrum(), atom_label="C")
        material_source.assert_not_called()
        processor.cleanup_analysis_file(state.filepath)

        state = processor.create_analysis_state(
            2, _spectrum(), atom_label="C", load_scattering_components=True,
        )
        material_source.assert_called_once_with(f"{state.filepath};temp=300.0K")
        assert [args.args[0] for args in loader.call_args_list] == [
            "comp=inelas", "comp=coh_elas", "comp=incoh_elas",
        ]
        assert all(args.kwargs == {"doInfo": False, "doAbsorption": False}
                   for args in loader.call_args_list)
        assert (state.scatter_inelastic, state.scatter_coherent_elastic,
                state.scatter_incoherent_elastic) == tuple(handles)
        processor.cleanup_analysis_file(state.filepath)

        loader.side_effect = RuntimeError("component load failed")
        with pytest.raises(RuntimeError, match="component load failed"):
            processor.create_analysis_state(
                3, _spectrum(), atom_label="C", load_scattering_components=True,
            )

    assert not list(tmp_path.glob("pdos_*.ncmat"))


# Ncrystal active species

def _fake_processor(tmp_path, *, active_species):
    """Create a transient processor without touching the source template."""
    processor = NcrystalProcessor()
    processor.base_ncmat_file = 'untouched.ncmat'
    processor.temperature = 296.0
    processor.temp_dir = tmp_path
    from processing.backends.ncmat_isolation import normalize_active_species
    processor.active_species = (None if active_species is None else normalize_active_species(active_species))
    original = Path(__file__).resolve().parents[2] / 'tests/fixtures/ncrystal_tsl_HinCH2.ncmat'
    if not original.exists():
        pytest.skip('CH2 fixture missing')
    baseline = original.read_text(encoding='utf-8')
    composer = MagicMock()
    composer.write.side_effect = lambda filepath: Path(filepath).write_text(baseline, encoding='utf-8')
    base_composer = MagicMock()
    base_composer.clone.return_value = composer
    processor._base_composer = base_composer
    return processor, composer, original


def test_transient_isolation_is_applied_after_injection(tmp_path, monkeypatch):
    processor, composer, original = _fake_processor(tmp_path, active_species={'H'})
    initial = original.read_bytes()
    monkeypatch.setattr(processor, '_inject_spectra', lambda *args: ('H', np.array([.01,.02]), np.array([1.,1.])))
    files_at_load = []
    def fake_load(cfg):
        path = Path(cfg.split(';',1)[0]); content = path.read_text()
        files_at_load.append(content)
        assert 'type sterile' in content
        return object()
    nuc = SimpleNamespace(elements={'H': SimpleNamespace(_awr=1.0)})
    with patch('processing.backends.ncrystal_processor.NC.load', side_effect=fake_load), \
         patch('processing.backends.ncrystal_processor.NC.clearCaches'), \
         patch('processing.backends.ncrystal_processor.NuclearData', return_value=nuc):
        state = processor.create_analysis_state(10, _spectrum())
    assert state.target_lbl == 'H'
    assert len(files_at_load) == 1
    assert original.read_bytes() == initial
    assert 'vdos_density' in files_at_load[0]  # hydrogen still has dynamics
    processor.cleanup_analysis_file(state.filepath)
    assert not Path(state.filepath).exists()


def test_default_full_material_file_is_unchanged(tmp_path, monkeypatch):
    processor, _, original = _fake_processor(tmp_path, active_species=None)
    monkeypatch.setattr(processor, '_inject_spectra', lambda *args: ('H', np.array([.01,.02]), np.array([1.,1.])))
    nuc = SimpleNamespace(elements={'H': SimpleNamespace(_awr=1.0)})
    with patch('processing.backends.ncrystal_processor.NC.load', return_value=object()), \
         patch('processing.backends.ncrystal_processor.NC.clearCaches'), \
         patch('processing.backends.ncrystal_processor.NuclearData', return_value=nuc):
        state = processor.create_analysis_state(11, _spectrum())
    assert Path(state.filepath).read_text() == original.read_text()
    processor.cleanup_analysis_file(state.filepath)


def test_inactive_target_is_rejected_and_temp_cleaned(tmp_path, monkeypatch):
    processor, _, _ = _fake_processor(tmp_path, active_species={'H'})
    monkeypatch.setattr(processor, '_inject_spectra', lambda *args: ('C', np.array([.01,.02]), np.array([1.,1.])))
    with pytest.raises(ValueError, match='Analysis target'):
        processor.create_analysis_state(12, object())
    assert not list(tmp_path.glob('pdos_12_*.ncmat'))

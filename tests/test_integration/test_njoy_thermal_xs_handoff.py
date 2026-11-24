"""Processing -> Analyzer regression checks for incoherent-elastic thermal XS.

The first test exercises the REAL TARA processor/runner/parser code while
substituting synthetic parsed ENDF sections for NJOY's external process.
The second test runs real NJOY when an incoherent-elastic input fixture is
explicitly supplied through environment variables.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analyzer.parsers.njoy_parser import NjoyParser
from analyzer.physics.data_models import IntegralData
from analyzer.runners.njoy_runner import NjoyRunner
from core.data.input_pdos import PhononSpectra
from processing.backends.njoy_processor import NjoyProcessor
import processing.backends.njoy_processor as processor_module


def _assert_incoherent_thermal_channels(engine):
    """Check the raw THERMR output before asserting Analyzer results.

    Parameters
    ----------
    engine : NjoyEngine
        Parsed LEAPR/THERMR tape state from the Analyzer runner.

    Returns
    -------
    IntegralData
        Thermal scattering cross sections calculated by Analyzer.
    """
    leapr_elastic = (engine.leapr_tree or {}).get(7, {}).get(2)
    assert leapr_elastic is not None, "LEAPR tape24 lacks MF7/MT2 elastic metadata."
    assert int(leapr_elastic["LTHR"]) == 2, (
        "Fixture is not incoherent-elastic: expected MF7/MT2 LTHR=2, "
        f"got {leapr_elastic.get('LTHR')!r}."
    )

    mf3 = (engine.thermr_tree or {}).get(3, {})
    assert 221 in mf3, "Processing/NJOY failed to produce THERMR MF3/MT221."
    assert 222 in mf3, "Processing/NJOY failed to produce incoherent-elastic MF3/MT222."

    # Use the common supported incident-neutron energy interval; do not test
    # outside the ENDF grid, where interpolation would insert artificial zeroes.
    e_in = np.asarray(mf3[221]["xstable"]["E"], dtype=float)
    e_el = np.asarray(mf3[222]["xstable"]["E"], dtype=float)
    lower = max(float(np.min(e_in)), float(np.min(e_el)))
    upper = min(float(np.max(e_in)), float(np.max(e_el)))
    assert 0 < lower < upper, "MT221 and MT222 have no common positive energy range."
    grid = np.geomspace(lower, upper, num=64)

    result = IntegralData()
    NjoyParser()._extract_cross_sections(engine, result, energy_grid=grid)

    assert result.xs_inelastic is not None, "Analyzer discarded MT221."
    assert result.xs_incoherent_elastic is not None, "Analyzer discarded MT222 (LTHR=2)."
    assert result.xs_total is not None, "Analyzer did not construct the thermal scattering sum."
    assert result.xs_coherent_elastic is None, "Incoherent MT222 was misclassified as coherent."

    for channel in (result.xs_inelastic, result.xs_incoherent_elastic, result.xs_total):
        assert isinstance(channel, pd.Series)
        np.testing.assert_array_equal(channel.index.to_numpy(), grid)
        assert np.isfinite(channel.to_numpy()).all()
        assert (channel.to_numpy() >= 0).all()

    np.testing.assert_allclose(
        result.xs_total.to_numpy(),
        result.xs_inelastic.to_numpy() + result.xs_incoherent_elastic.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.any(result.xs_incoherent_elastic.to_numpy() > 0), (
        "The elastic section exists but its cross section is identically zero."
    )
    return result


@pytest.mark.parametrize("full", [False, True])
def test_processing_runner_parser_incoherent_elastic_handoff(monkeypatch, tmp_path, full):
    """Exercise real TARA handoff logic without requiring an NJOY executable.

    The external NJOY execution and ENDF deserialization are replaced by a
    deterministic synthetic fixture. This is a contract test, NOT a real NJOY
    physics validation.
    """
    processor = object.__new__(NjoyProcessor)
    processor.base_endf_file = str(tmp_path / "base.endf")
    processor.base_mat = 125
    processor._leapr_template = SimpleNamespace(
        material_info={"mat": 40},
        principal_scatterer={"npr": 1, "iel": 1},
    )
    Path(processor.base_endf_file).write_text("synthetic base fixture\n", encoding="utf-8")

    def serialize(template, spectrum, *, temperature=None, export_full_spectra=False):
        assert export_full_spectra is full
        return "leapr\n20 24 /\nstop\n"

    monkeypatch.setattr(processor_module, "serialize_leapr_spectrum", serialize)

    captured_decks = []

    @contextmanager
    def fake_execute(deck, input_tapes=None):
        captured_decks.append(deck)
        assert input_tapes == {"tape20": processor.base_endf_file}
        for name in ("tape24", "tape26"):
            (tmp_path / name).touch()
        yield SimpleNamespace(
            tape_path=lambda name: tmp_path / name,
            has_tape=lambda name: (tmp_path / name).is_file(),
            stdout_text="simulated NJOY run",
        )

    monkeypatch.setattr(processor, "execute", fake_execute)

    energy = [1e-4, 1e-2, 1.0]
    parsed_leapr = {7: {2: {"LTHR": 2}}}
    parsed_thermr = {3: {
        221: {"xstable": {"E": energy, "xs": [2.0, 3.0, 4.0]}},
        222: {"xstable": {"E": energy, "xs": [0.5, 1.0, 1.5]}},
    }}

    class FakeEndfParser:
        def parsefile(self, filename, **kwargs):
            if Path(filename).name == "tape24":
                return parsed_leapr
            assert Path(filename).name == "tape26"
            assert kwargs == {"include": [3]}
            return parsed_thermr

    runner = object.__new__(NjoyRunner)
    runner.processor = processor
    runner.endf_parser = FakeEndfParser()
    runner.base_endf_file = processor.base_endf_file
    runner.temperature = 293.6
    runner.mass_amu = 1.0

    spectrum = SimpleNamespace(
        energy_grid=np.array([0.0, 0.01, 0.02, 0.03]),
        density_values=np.array([0.0, 20.0, 20.0, 0.0]),
    )
    engine = runner.execute(0, spectrum, export_full_spectra=full)

    assert len(captured_decks) == 1
    assert "thermr" in captured_decks[0].lower()
    # TARA's deck uses MTREF=221 and passes the elastic flag derived from IEL.
    assert "40 125 16 1 2 1 0 1 221 2 /" in captured_decks[0]
    _assert_incoherent_thermal_channels(engine)


@pytest.mark.parametrize("fixture", ["incoherent_elastic"])
def test_real_njoy_incoherent_elastic_processing_to_analyzer(fixture, tmp_path):
    """Run NJOY + ENDF parsing + XS extraction with a known elastic fixture.

    Required environment variables for this opt-in test:
      TARA_NJOY_TEST_EXEC       Path to real NJOY executable
      TARA_NJOY_TEST_BASE_ENDF  Base neutron ENDF evaluation
      TARA_NJOY_TEST_LEAPR      Compatible incoherent-elastic LEAPR input
      TARA_NJOY_TEST_MASS_AMU   Principal scatterer's mass in amu
      TARA_NJOY_TEST_TEMP_K     Temperature represented by the LEAPR input

    The LEAPR fixture must describe incoherent elastic scattering (LTHR=2).
    The test explicitly checks raw NJOY sections before Analyzer fields.
    """
    del fixture
    env = {
        "executable": os.environ.get("TARA_NJOY_TEST_EXEC"),
        "base_endf": os.environ.get("TARA_NJOY_TEST_BASE_ENDF"),
        "leapr": os.environ.get("TARA_NJOY_TEST_LEAPR"),
        "mass": os.environ.get("TARA_NJOY_TEST_MASS_AMU"),
        "temperature": os.environ.get("TARA_NJOY_TEST_TEMP_K"),
    }
    if not any(env.values()):
        pytest.skip("Real NJOY fixture not configured; see integration test README.")
    missing = [name for name, value in env.items() if not value]
    if missing:
        pytest.fail(f"Incomplete real NJOY fixture configuration: {', '.join(missing)}")

    for key in ("executable", "base_endf", "leapr"):
        assert Path(env[key]).is_file(), f"Missing {key}: {env[key]}"
    temperature = float(env["temperature"])
    mass_amu = float(env["mass"])
    assert temperature > 0 and mass_amu > 0

    from input_output.leapr.core import LeaprFile

    source = LeaprFile.from_file(str(Path(env["leapr"]).resolve()))
    spectrum = PhononSpectra(source.pdos)

    runner = NjoyRunner(
        njoy_exec=str(Path(env["executable"]).resolve()),
        base_endf_file=str(Path(env["base_endf"]).resolve()),
        base_leapr_file=str(Path(env["leapr"]).resolve()),
        temperature=temperature,
        mass_amu=mass_amu,
        temp_dir=str(tmp_path),
    )
    engine = runner.execute(0, spectrum)
    _assert_incoherent_thermal_channels(engine)

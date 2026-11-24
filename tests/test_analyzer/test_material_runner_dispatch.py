"""Regression tests for Analyzer's material-ensemble entry point."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import physical_constants

from analyzer.ensemble_analyzer import EnsembleAnalyzer
from analyzer.parsers.njoy_parser import NjoyParser
from analyzer.runners.njoy_runner import NjoyRunner
from core.data.material_ensemble import MaterialEnsemble
from input_output.leapr.core import LeaprFile
from input_output.leapr.material_adapter import material_model_from_leapr

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_njoy_material_ensemble_uses_sampled_decks_and_distinct_reference(tmp_path, monkeypatch):
    """Preserve translation coefficients in the real serializer before NJOY execution."""
    import sys
    from endf_parserpy import EndfParserPy
    from processing.backends.njoy_processor import NjoyProcessor

    path = FIXTURES / "tsl-HinH2O.leapr"
    template = LeaprFile.from_file(str(path))
    reference = material_model_from_leapr(
        template, state="liquid", temperature=295, principal_label="H", secondary_label="O",
    )
    dynamics = reference.dynamics
    ensemble = MaterialEnsemble(
        reference, [dynamics.translation_weight, dynamics.translation_weight * 1.05],
        [dynamics.translation_parameter * 0.9, dynamics.translation_parameter * 1.1],
    )
    read_original = EndfParserPy.parsefile
    decks = []

    @contextmanager
    def mock_execute(self, deck, input_tapes):
        """Record sample decks while providing output paths to the normal runner."""
        assert Path(input_tapes["tape20"]).is_file()
        decks.append(deck)
        yield SimpleNamespace(
            tape_path=lambda tape: tmp_path / tape,
            has_tape=lambda tape: tape == "tape26", stdout_text="",
        )

    def mock_parse(self, path, **kwargs):
        """Supply synthetic NJOY output while reading the real base evaluation."""
        if Path(path).name == "tape24":
            return {7: {}}
        if Path(path).name == "tape26":
            return {3: {221: {"xstable": {"E": [0.01, 0.1], "xs": [5.0, 2.0]}}}}
        return read_original(self, path, **kwargs)

    monkeypatch.setattr(NjoyProcessor, "execute", mock_execute)
    monkeypatch.setattr(EndfParserPy, "parsefile", mock_parse)
    runner = NjoyRunner(
        sys.executable, str(FIXTURES / "n_1-H-001g.jeff"), str(path), 295,
        reference.principal.mass_ratio * physical_constants["neutron mass in u"][0],
    )
    analyzer = EnsembleAnalyzer(ensemble, runner, NjoyParser())
    analyzer.compute(parallel=False, parser_kwargs={
        "energy_grid": np.array([0.01, 0.1]), "calc_thermodynamics": False,
    })

    assert [result.sample_index for result in analyzer.raw_results] == [0, 1]
    assert analyzer.reference_result.sample_index == -1
    assert len(set(decks)) == 3
    for index, deck in enumerate(decks):
        # LEAPR is followed by THERMR; parse just its input to inspect Card 13.
        deck_path = tmp_path / f"material-{index}.leapr"
        lines = deck.splitlines()
        start = next(i for i, line in enumerate(lines) if line.lower() == "leapr")
        end = next(i for i in range(start + 1, len(lines)) if lines[i].lower() == "thermr")
        deck_path.write_text("\n".join(lines[start:end]) + "\nstop\n")
        block = LeaprFile.from_file(str(deck_path)).temp_blocks[0]
        expected = (ensemble[index] if index < len(ensemble) else reference).dynamics
        assert block["twt"] == pytest.approx(expected.translation_weight, rel=1e-6)
        assert block["c"] == pytest.approx(expected.translation_parameter, rel=1e-6)
    np.testing.assert_array_equal(analyzer.reference_result.integ.xs_total, [5.0, 2.0])

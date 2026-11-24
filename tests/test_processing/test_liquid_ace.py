"""Explicit liquid-water reaction IDs and legacy solid deck compatibility."""

from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest

from processing.core.config import AceGenerationConfig
from processing.deck_builders.ace_builder import build_ace_deck


def _kwargs(tmp_path) -> dict:
    """Provide the existing valid solid ACE-deck parameters."""
    endf = tmp_path / 'h.endf'
    endf.write_text('test')
    return dict(base_endf_path=str(endf), tsl_endf_path='tsl.endf', base_za=1001,
                base_mat=125, tsl_mat=7, temp=296., icoh=0, natom=2,
                emax=5., suffix='.01', za1=1001, za2=0, za3=0,
                elas_mat=0, ielas=0)


def test_liquid_water_thermr_acer_reaction_consistency(tmp_path) -> None:
    """NJOY THERMR/ACER must agree on mti=222 and omit a water elastic reaction."""
    deck = build_ace_deck(**_kwargs(tmp_path), mti=222, mte=0)
    assert '0 2 222 2 /' in deck
    assert '222 200 0 0 1 5.0 2 /' in deck
    assert '237 200' not in deck
    assert deck.endswith('stop\n')


def test_legacy_solid_ace_deck_is_unchanged(tmp_path) -> None:
    """The original hard-coded solid MTs remain the default when not overridden."""
    deck = build_ace_deck(**_kwargs(tmp_path))
    assert '0 2 237 2 /' in deck
    assert '237 200 0 0 1 5.0 2 /' in deck


def _fake_parser(monkeypatch, base: Path, tapes: dict[str, str], ns: int):
    """Install a fake ENDF parser without requiring NJOY or endf_parserpy."""
    mapping = {str(base): {1: {451: {'ZA': 1001., 'MAT': 125}}}}
    mapping.update({f: {7: {4: {'MAT': 7, 'B': {6: 2}, 'NS': ns}}} for f in tapes.values()})
    fake = types.ModuleType('endf_parserpy')
    fake.EndfParserPy = lambda: types.SimpleNamespace(parsefile=lambda path: mapping[path])
    monkeypatch.setitem(sys.modules, 'endf_parserpy', fake)


def test_explicit_liquid_config_allows_secondary_species(tmp_path, monkeypatch) -> None:
    """H(H2O) secondary oxygen no longer blocks an explicit, principal-only ACE model."""
    base = tmp_path / 'base.endf'
    base.write_text('base')
    f = tmp_path / 'water.endf'
    f.write_text('tsl')
    tapes = {'sample_000': str(f)}
    _fake_parser(monkeypatch, base, tapes, ns=1)
    with pytest.raises(ValueError, match='additional atom types'):
        AceGenerationConfig.from_endf(base_endf_path=str(base), tsl_endf_files=tapes, temperature=296.)
    with pytest.raises(ValueError, match='Explicit ACER mti'):
        AceGenerationConfig.from_endf(base_endf_path=str(base), tsl_endf_files=tapes,
                                      temperature=296., allow_additional_scatterers=True)
    config = AceGenerationConfig.from_endf(
        base_endf_path=str(base), tsl_endf_files=tapes, temperature=296.,
        allow_additional_scatterers=True, mti=222, mte=0,
    )
    assert config.natom == 2
    assert config.mti == 222 and config.mte == 0
    assert config.za1 == 1001 and config.za2 == 0
    assert config.elas_mat == 0 and config.ielas == 0


def test_reject_liquid_elastic_mismatch(tmp_path, monkeypatch) -> None:
    """Without MF7/MT2, ACER must not be assigned an elastic reaction."""
    base = tmp_path / 'base.endf'
    base.write_text('base')
    f = tmp_path / 'water.endf'
    f.write_text('tsl')
    tapes = {'sample_000': str(f)}
    _fake_parser(monkeypatch, base, tapes, ns=1)
    with pytest.raises(ValueError, match='requires an MF7/MT2'):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base), tsl_endf_files=tapes, temperature=296.,
            allow_additional_scatterers=True, mti=222, mte=223,
        )


def test_liquid_samples_must_agree_on_secondary_species_count(tmp_path, monkeypatch) -> None:
    """Reject inconsistent MF7 NS metadata across a stochastic ACE batch."""
    base = tmp_path / 'base.endf'
    base.write_text('base')
    a, b = tmp_path / 'a.endf', tmp_path / 'b.endf'
    a.write_text('mock A')
    b.write_text('mock B')
    tapes = {'sample_000': str(a), 'sample_001': str(b)}
    mapping = {
        str(base): {1: {451: {'ZA': 1001, 'MAT': 125}}},
        str(a): {7: {4: {'MAT': 99, 'B': {6: 2}, 'NS': 1}}},
        str(b): {7: {4: {'MAT': 99, 'B': {6: 2}, 'NS': 2}}},
    }
    fake = types.ModuleType('endf_parserpy')
    fake.EndfParserPy = lambda: types.SimpleNamespace(parsefile=lambda path: mapping[path])
    monkeypatch.setitem(sys.modules, 'endf_parserpy', fake)
    with pytest.raises(ValueError, match='Inconsistent TSL'):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base), tsl_endf_files=tapes, temperature=296.,
            allow_additional_scatterers=True, mti=222, mte=0,
        )


def test_reaction_numbers_outside_thermr_range_rejected(tmp_path) -> None:
    """THERMR only permits the documented range 221 through 250."""
    for reaction in (0, 220, 251):
        with pytest.raises(ValueError):
            build_ace_deck(**_kwargs(tmp_path), mti=reaction, mte=0)

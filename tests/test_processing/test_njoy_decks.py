"""Njoy decks regression tests."""


from processing.deck_builders.leapr_builder import build_analysis_deck
from processing.deck_builders.leapr_builder import build_leapr_deck
from processing.deck_builders.leapr_builder import format_njoy_array
from hashlib import sha256
from pathlib import Path
import sys
import types
import pytest
from processing.core.config import AceGenerationConfig
from processing.deck_builders.ace_builder import build_ace_deck


# Deck builders

def _leapr_kwargs():
    """Leapr kwargs for the current workflow.
    """
    return dict(
        mat_id=125,
        lat=0,
        mass_amu=12.0,
        temperature=296.0,
        nalpha=2,
        nbeta=2,
        alpha_grid_str="0.1 1.0",
        beta_grid_str="0.0 1.0",
        egrid_str="1.0e-05 2.0e-05",
        rho_str="1.0 2.0",
    )


def test_format_njoy_array_groups_scientific_values():
    """Test shared NJOY free-format array serialization."""
    rendered = format_njoy_array([1.0, 2.0, 3.0], values_per_line=2)
    lines = rendered.splitlines()

    assert lines == ["1.000000e+00 2.000000e+00", "3.000000e+00"]


def test_standalone_leapr_deck_has_one_terminal_stop():
    """
    Test that the persistent LEAPR deck is standalone and terminates exactly once.
    """
    deck = build_leapr_deck(**_leapr_kwargs())

    assert deck.endswith("stop\n")
    assert deck.count("\nstop\n") == 1


def test_analysis_deck_composes_leapr_before_thermr_without_intermediate_stop():
    """
    Test the shared transient Analyzer NJOY sequence.

    Verifies the exact module ordering ``MODER -> RECONR -> BROADR -> LEAPR ->
    THERMR -> STOP`` and guards against the historical ``stop\nthermr`` bug.
    """
    leapr_module_text = """leapr
-24
125 0 0
"""

    deck = build_analysis_deck(
        base_mat=625,
        leapr_mat=125,
        temperature=296.0,
        leapr_module_text=leapr_module_text,
    )
    lower = deck.lower()

    positions = [
        lower.index(name)
        for name in ("moder", "reconr", "broadr", "leapr", "thermr", "stop")
    ]

    assert positions == sorted(positions)
    assert "\nthermr\n" in lower
    assert "stop\nthermr" not in lower
    assert lower.rstrip().endswith("stop")
    assert lower.count("\nstop") == 1


# Acer card9

def _deck_kwargs(elas_mat: int = 0, ielas: int = 0) -> dict:
    """Return fixed V1 ACE parameters for byte-identical deck comparisons.

    Parameters
    ----------
    elas_mat : int, optional
        Historical name for elastic reaction MT.
    ielas : int, optional
        Coherent/incoherent/mixed elastic selection for ACER.

    Returns
    -------
    dict
        Complete NJOY deck configuration.
    """
    return dict(
        base_endf_path='base.endf', tsl_endf_path='tsl.endf',
        base_za=1001, base_mat=125, tsl_mat=7, temp=296.0,
        icoh=0, natom=2, emax=5.0, suffix='.01', za1=1001,
        za2=0, za3=0, elas_mat=elas_mat, ielas=ielas,
    )


@pytest.mark.parametrize('elastic_mt,ielas,expected_sha256', [
    (0, 0, '385c0320ba2a9ebbd15e61ff0cbbdaff24301bf9439629d7a6013712a184749f'),
    (238, 0, '2335f9676a66c295432cd96068631768904a806b224a450e181c604d5fb02df8'),
    (238, 1, '965efa2f9ba8b5ffeda19922ad21d87352aa7302e68ef9ce2eed5a13fbb5e1ee'),
    (238, 2, 'f7a3d0907ed11a21860f4f7b43ce3623c09f735b71d7324d7ce831677005c699'),
    (125, 1, 'e710c586dfb4de22b97d953a54ccf2945c0cd662489e20f8f92da5af61d1ae79'),
])
def test_v1_solids_byte_identical(elastic_mt, ielas, expected_sha256):
    """Maintain byte-for-byte decks captured from the supplied V1 codebook."""
    deck = build_ace_deck(**_deck_kwargs(elastic_mt, ielas))
    assert sha256(deck.encode('utf-8')).hexdigest() == expected_sha256


def test_card9_water_preserves_nbint_200():
    """Keep ACER's energy bin count independent from elastic reaction MT."""
    deck = build_ace_deck(**_deck_kwargs(), mti=222, mte=0)
    assert '222 200 0 0 1 5.0 2 /\n' in deck
    assert '7 125 16 1 2 0 0 2 222 2 /\n' in deck


def test_card9_distinguishes_thermr_nbin_from_acer_nbint():
    """Custom THERMR angular bins must not overwrite ACER Card 9 ``nbint``."""
    deck = build_ace_deck(**_deck_kwargs(), nbin=24, mti=222, mte=0)
    assert '7 125 24 1 2 0 0 2 222 2 /\n' in deck
    assert '222 200 0 0 1 5.0 2 /\n' in deck


@pytest.mark.parametrize('mte', [223, 0])
def test_builder_rejects_explicitly_conflicting_elastic_mt(mte):
    """Reject mismatch between the legacy and explicit ACER elastic MT."""
    with pytest.raises(ValueError, match='Conflicting ACER elastic'):
        build_ace_deck(**_deck_kwargs(238), mti=222, mte=mte)


def _direct_config(tmp_path: Path, elas_mat: int, mte=None) -> AceGenerationConfig:
    """Build a direct legacy-style ACE config without ENDF parsing."""
    base = tmp_path / 'base.endf'
    base.write_text('fixture')
    return AceGenerationConfig(
        str(base), 1001, 125, 7, 296., 1, 2, 5., '.01',
        1001, 0, 0, elas_mat, 0, mte=mte,
    )


def test_legacy_config_sets_mte_to_elas_mat(tmp_path):
    """Preserve positional V1 constructors and their elastic MT."""
    config = _direct_config(tmp_path, elas_mat=238)
    assert config.mte == config.elas_mat == 238


def test_config_rejects_explicitly_conflicting_elastic_mt(tmp_path):
    """Avoid silently preferring one elastic MT over another."""
    with pytest.raises(ValueError, match='Conflicting ACER elastic'):
        _direct_config(tmp_path, elas_mat=238, mte=223)


def _batch_type(monkeypatch):
    """Load the orchestrator with a test-only shared parallel helper.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Insert only the absent root helper in the reconstructed codebook tree.

    Returns
    -------
    type
        Real NJOY batch orchestrator class, without external execution.
    """
    parallel = types.ModuleType('parallel_execution')
    parallel.run_parallel = lambda fn, calls, n_jobs: [
        fn(*args, **kwargs) for args, kwargs in calls
    ]
    monkeypatch.setitem(sys.modules, 'parallel_execution', parallel)
    from processing.orchestrators.njoy_batch import NjoyBatchOrchestrator
    return NjoyBatchOrchestrator


def _mock_endf(monkeypatch, tmp_path: Path, lthr: int):
    """Provide MF7 metadata for config tests with no external parser/backend."""
    base = tmp_path / 'base.endf'
    tsl = tmp_path / 'tsl.endf'
    base.write_text('mock base ENDF')
    tsl.write_text('mock TSL ENDF')
    sections = {
        str(base): {1: {451: {'ZA': 1001, 'MAT': 125}}},
        str(tsl): {7: {
            4: {'MAT': 7, 'B': {6: 2}, 'NS': 0},
            **({2: {'LTHR': lthr}} if lthr else {}),
        }},
    }
    fake = types.ModuleType('endf_parserpy')
    fake.EndfParserPy = lambda: types.SimpleNamespace(parsefile=lambda path: sections[path])
    monkeypatch.setitem(sys.modules, 'endf_parserpy', fake)
    return base, {'sample_000': str(tsl)}


@pytest.mark.parametrize('lthr,ielas', [(1, 0), (2, 1), (3, 2)])
def test_from_endf_resolves_mti_plus_one_and_mixed_elastic(monkeypatch, tmp_path, lthr, ielas):
    """Map THERMR MTs into ACER, including LTHR=3 mixed-elastic mode."""
    base, tapes = _mock_endf(monkeypatch, tmp_path, lthr)
    config = AceGenerationConfig.from_endf(
        base_endf_path=str(base), tsl_endf_files=tapes,
        temperature=296., mti=222,
    )
    assert config.mti == 222
    assert config.mte == config.elas_mat == 223
    assert config.ielas == ielas
    deck = _batch_type(monkeypatch)._build_sample_ace_deck(
        vars(config), tapes['sample_000'],
    )
    assert f'222 200 223 {ielas} 1 5.0 2 /' in deck
    assert '0 2 222 2 /' in deck


def test_from_endf_rejects_elastic_reaction_mismatch(monkeypatch, tmp_path):
    """Explicit elastic reactions must agree with THERMR's MT assignment."""
    base, tapes = _mock_endf(monkeypatch, tmp_path, lthr=1)
    with pytest.raises(ValueError, match='mti \\+ 1'):
        AceGenerationConfig.from_endf(
            base_endf_path=str(base), tsl_endf_files=tapes,
            temperature=296., mti=222, mte=238,
        )


def test_njoy_batch_legacy_config_uses_elas_mat(monkeypatch):
    """Keep old dictionary orchestrator callers without an explicit mte."""
    kwargs = _deck_kwargs(238, 1)
    config = {
        'base_endf_path': kwargs['base_endf_path'],
        'base_za': kwargs['base_za'], 'base_mat': kwargs['base_mat'],
        'tsl_mat': kwargs['tsl_mat'], 'temperature': kwargs['temp'],
        'icoh': kwargs['icoh'], 'natom': kwargs['natom'],
        'emax': kwargs['emax'], 'suffix': kwargs['suffix'],
        'za1': kwargs['za1'], 'za2': kwargs['za2'], 'za3': kwargs['za3'],
        'elas_mat': kwargs['elas_mat'], 'ielas': kwargs['ielas'],
    }
    deck = _batch_type(monkeypatch)._build_sample_ace_deck(config, 'tsl.endf')
    assert '237 200 238 1 1 5.0 2 /' in deck

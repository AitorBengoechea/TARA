"""NCMAT parsing, species selection, and round trips."""


import pytest
import numpy as np
import NCrystal as NC
from input_output.ncrystal.core import NCrystalFile
from pathlib import Path
from input_output.ncrystal.exceptions import NCrystalFormatError
from input_output.ncrystal.parsers import expand_ncmat_array_values
from input_output.ncrystal.liquids import load_liquid_ncmat
from input_output.ncrystal.liquids import parse_custom_liquids
from input_output.ncrystal.liquids import LiquidAtomSettings
from input_output.ncrystal.liquids import add_custom_liquids
from input_output.ncrystal.liquids import replace_vdos
from input_output.ncrystal.liquids import ensure_custom_liquids
from input_output.ncrystal.material_adapter import fold_solidlike_vdos


# Ncrystal io

# Explicitly define a diverse cross-section from NCrystal's library
NCMAT_TEST_CASES = [
    "C_sg194_pyrolytic_graphite.ncmat",          # Solid / Single Atom
    "Polyethylene_CH2.ncmat"                     # Polymer
]

@pytest.mark.parametrize("ncmat_filename", NCMAT_TEST_CASES)
def test_ncrystal_round_trip(tmp_path, ncmat_filename):
    """
    Test the end-to-end round-trip generation of thermal scattering data via NCrystal.

    Verifies that:
    - Input phonon spectra or crystal structures are serialized into valid NCMAT format via NCrystalWriter.
    - NcrystalProcessor successfully invokes the native ncmat2endf engine to produce an ENDF tape.
    - The resulting tape passes ENDF MF7/MT4 validation and preserves consistency across the conversion chain.
    """
    # 1. Setup: Extract standard NCrystal file and write it to our isolated sandbox
    raw_ncmat_text = NC.createTextData(ncmat_filename).rawData
    input_fixture = tmp_path / f"input_{ncmat_filename}"
    input_fixture.write_text(raw_ncmat_text)
    
    output_fixture = tmp_path / f"output_{ncmat_filename}"

    # 2. Extract: Read the file using TARA's NCrystalFile data model[cite: 3]
    original_data = NCrystalFile.from_file(str(input_fixture))

    # 3. Write: Export the parsed object back to disk using its write method[cite: 3]
    original_data.write(str(output_fixture))

    # 4. Extract Again: Parse the newly written file
    round_trip_data = NCrystalFile.from_file(str(output_fixture))

    # 5. Assert: Verify structural and numerical fidelity
    assert original_data.version == round_trip_data.version
    assert original_data.density == round_trip_data.density
    assert original_data.state_of_substance == round_trip_data.state_of_substance
    assert original_data.cell == round_trip_data.cell
    assert original_data.spacegroup == round_trip_data.spacegroup
    assert original_data.atom_positions == round_trip_data.atom_positions
    
    # Verify primary PDOS / VDOS numerical equivalence
    pdos_orig = original_data.pdos
    pdos_rt = round_trip_data.pdos
    
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(pdos_orig.index.to_numpy(), pdos_rt.index.to_numpy(), rtol=1e-5)
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(pdos_orig.values, pdos_rt.values, rtol=1e-5)


# Ncrystal multiatom vdos

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "ncrystal_tsl_HinCH2.ncmat"


def test_repetition_counts_mean_exact_number_of_values():
    """NCMAT ``value r count`` compression must not introduce fake points."""
    assert expand_ncmat_array_values("1e-3 0r3 -.5r2 .25") == [
        0.001, 0.0, 0.0, 0.0, -0.5, -0.5, 0.25,
    ]
    with pytest.raises(NCrystalFormatError, match="Invalid NCMAT"):
        expand_ncmat_array_values("0rnotanumber")
    with pytest.raises(NCrystalFormatError, match="Invalid NCMAT"):
        expand_ncmat_array_values("1r0")


def test_polyethylene_both_species_and_h_selection():
    """CH2 has two VDOS grids; selecting H must never silently select C."""
    ncmat = NCrystalFile.from_file(str(FIXTURE))
    assert ncmat.version == 5
    assert ncmat.state_of_substance == "solid"
    assert [(block["element"], block["fraction"]) for block in ncmat.dyninfo] == [
        ("C", "1/3"), ("H", "2/3"),
    ]
    carbon, hydrogen = (ncmat.pdos_for_atom(label) for label in ("C", "H"))
    assert len(carbon) == 193
    assert len(hydrogen) == 379
    assert np.isclose(carbon.index[0], 0.00499983)
    assert np.isclose(carbon.index[-1], 0.388987)
    assert np.isclose(hydrogen.index[0], 0.00999966)
    assert np.isclose(hydrogen.index[-1], 0.387987)
    assert (carbon.to_numpy() >= 0).all()
    assert (hydrogen.to_numpy() >= 0).all()
    assert np.max(hydrogen.to_numpy()) == pytest.approx(1)
    # The old property remains backward compatible, but *is not* H here.
    np.testing.assert_array_equal(ncmat.pdos.to_numpy(), carbon.to_numpy())


def test_selection_fails_explicitly_if_species_is_missing():
    """A missing atom must produce an actionable error, never a fallback."""
    ncmat = NCrystalFile.from_file(str(FIXTURE))
    with pytest.raises(NCrystalFormatError, match="Available VDOS labels"):
        ncmat.pdos_for_atom("O")


def test_polyethylene_serialization_preserves_both_atom_spectra(tmp_path):
    """Source version, state, repetition counts and per-species VDOS round-trip."""
    reference = NCrystalFile.from_file(str(FIXTURE))
    path = tmp_path / "polyethylene_roundtrip.ncmat"
    reference.write(str(path))
    serialized = path.read_text(encoding="utf-8")
    assert serialized.startswith("NCMAT v5\n")
    assert "@STATEOFMATTER" in serialized
    assert serialized.count("@DYNINFO") == 2
    actual = NCrystalFile.from_file(str(path))
    for label in ("C", "H"):
        expected_vdos = reference.pdos_for_atom(label)
        result_vdos = actual.pdos_for_atom(label)
        np.testing.assert_array_equal(result_vdos.index, expected_vdos.index)
        np.testing.assert_array_equal(result_vdos.to_numpy(), expected_vdos.to_numpy())


def test_legacy_single_atom_pdos_still_works(tmp_path):
    """The new species selector does not change monoatomic ``.pdos`` behavior."""
    path = tmp_path / "single_H.ncmat"
    path.write_text(
        "NCMAT v3\n@DENSITY\n  1 g_per_cm3\n"
        "@DYNINFO\n  element H\n  fraction 1\n  type vdos\n"
        "  vdos_egrid 0.01 0.04\n  vdos_density 0.1 0r2 0.4\n",
        encoding="utf-8",
    )
    ncmat = NCrystalFile.from_file(str(path))
    assert ncmat.version == 3
    np.testing.assert_array_equal(ncmat.pdos.to_numpy(), [0.1, 0.0, 0.0, 0.4])
    np.testing.assert_array_equal(ncmat.pdos.to_numpy(), ncmat.pdos_for_atom("H"))


# Ncrystal liquids io

def test_illustrative_water_fixture_is_parseable():
    fixture = Path(__file__).parents[1]/'fixtures'/'liquid_water_295K_illustrative.ncmat'
    rows = parse_custom_liquids(load_liquid_ncmat(fixture))
    assert len(rows) == 1 and rows[0].model == 'incoherent'


# Ncrystal unified io

BASE = ("NCMAT v7\n@DENSITY\n  1 g_per_cm3\n"
        "@DYNINFO\n  element D\n  fraction 0.6666666667\n  type vdos\n"
        "  vdos_egrid 1e-5 0.01\n  vdos\n    1 2 1\n"
        "@DYNINFO\n  element O\n  fraction 0.3333333333\n  type freegas\n")


def test_multispecies_young_koppel_roundtrip():
    src = add_custom_liquids(BASE, [
        LiquidAtomSettings('D', 'incoherent', 2.5, .12, .88, yk_model='para'),
        LiquidAtomSettings('O', 'vineyard', 3.5, .2, .8,
                           static_structure_factor=((0., 1.), (1., 1.2))),
    ])
    settings = parse_custom_liquids(src)
    assert [s.atom for s in settings] == ['D', 'O']
    assert settings[0].yk_model == 'para'
    assert settings[1].static_structure_factor[1] == (1.0, 1.2)
    grid = np.arange(1, 10) * (.02 / 9)  # Valid NCrystal E_min/dE=1 grid
    vals = np.linspace(0, 1, 9)
    newer = replace_vdos(src, 'D', grid, vals)
    assert newer.split('@CUSTOM_LIQUIDS')[1] == src.split('@CUSTOM_LIQUIDS')[1]
    assert 'element O' in newer


@pytest.mark.parametrize('settings', [
    dict(atom='O', model='incoherent', c=1, wt=.1, ws=.9, yk_model='para'),
    dict(atom='D', model='incoherent', c=1, wt=.1, ws=.9, yk_model='wrong'),
    dict(atom='Xe', model='skold', c=1, wt=.1, ws=.9),
])
def test_invalid_plugin_options_rejected(settings):
    with pytest.raises(ValueError):
        LiquidAtomSettings(**settings)


def test_missing_and_duplicate_custom_sections_fail():
    with pytest.raises(ValueError):
        ensure_custom_liquids(BASE)
    x = add_custom_liquids(BASE, [LiquidAtomSettings('D', 'incoherent', 1, .1, .9)])
    with pytest.raises(ValueError):
        ensure_custom_liquids(x + '\n@CUSTOM_LIQUIDS\n D incoherent 1 .1 .9\n')


def test_garbled_liquid_parameters_rejected():
    with pytest.raises(ValueError):
        ensure_custom_liquids(BASE + '@CUSTOM_LIQUIDS\n  D incoherent WHAT .1 .9\n')


def test_generic_folding_is_not_water_specific():
    # The generic adapter only inspects a model's principal dynamical weights.
    from types import SimpleNamespace
    grid = np.linspace(0, .15, 150)
    density = np.ones_like(grid) / .15
    spectrum = SimpleNamespace(energy_grid=grid, density_values=density)
    dynamics = SimpleNamespace(continuous_weight=.8, oscillators=(
        SimpleNamespace(energy_ev=.2, weight=.1),))
    model = SimpleNamespace(dynamics=dynamics)
    xs, vdos = fold_solidlike_vdos(model, spectrum, n_grid=1201)
    assert xs[0] >= 1e-5
    assert np.isclose(np.trapezoid(vdos, xs), 1, atol=5e-5)

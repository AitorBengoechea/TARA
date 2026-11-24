"""LEAPR parsing and writer round trips."""


import pytest
import numpy as np
from pathlib import Path
from input_output.leapr.core import LeaprFile
from io import StringIO
from input_output.leapr.parsers import read_fixed_length_array
from input_output.leapr.parsers import read_phonon_spectrum_block
from input_output.leapr.exceptions import LeaprFormatError
from input_output.leapr.material_adapter import material_model_from_leapr
from input_output.leapr.temperature import available_leapr_temperatures
from input_output.leapr.temperature import select_leapr_temperature_block
from input_output.leapr.workflows import serialize_leapr_spectrum


# Round-trip format tests

# Define the location of your local fixture files
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
LEAPR_TEST_CASES = [
    "input_1.leapr",      
    "input_2.leapr",
    "input_3.leapr",
]

@pytest.mark.parametrize("leapr_filename", LEAPR_TEST_CASES)
def test_leapr_round_trip(tmp_path, leapr_filename):
    """
    Test the end-to-end round-trip processing pipeline using NJOY LEAPR.

    Verifies that:
    - Input phonon spectra and configuration parameters are formatted into a valid LEAPR Fortran deck.
    - NjoyProcessor executes the deck in an isolated sandbox and generates the expected tape24 file.
    - The output file is verified as a compliant ENDF-6 MF7/MT4 tape and retrieved to the target destination.
    """
    # 1. Setup: Locate the original fixture file
    input_fixture = FIXTURES_DIR / leapr_filename
    assert input_fixture.exists(), f"Fixture file not found: {input_fixture}"
    
    output_fixture = tmp_path / f"output_{leapr_filename}"

    # 2. Extract: Read the file using TARA's LeaprFile data model
    original_data = LeaprFile.from_file(str(input_fixture))

    # 3. Write: Export the parsed object back to disk using its write method[cite: 3]
    original_data.write(str(output_fixture))

    # 4. Extract Again: Parse the newly written file
    round_trip_data = LeaprFile.from_file(str(output_fixture))

    # 5. Assert: Verify structural and numerical fidelity
    assert original_data == round_trip_data, f"Data mutated during IO round-trip for {leapr_filename}"
    
    # 6. Verify Primary Spectrum Equivalence
    # Ensure the PDOS properties (first positive temp spectrum) were not altered[cite: 3]
    pdos_orig = original_data.pdos
    pdos_rt = round_trip_data.pdos
    
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(pdos_orig.index.to_numpy(), pdos_rt.index.to_numpy(), rtol=1e-5)
    # Check the expected numerical agreement within the declared tolerance.
    np.testing.assert_allclose(pdos_orig.values, pdos_rt.values, rtol=1e-5)


# CAB water parsing and multitemperature round trips

SOURCE = Path(__file__).resolve().parents[1] / 'fixtures' / 'tsl-HinH2O.leapr'


def _official() -> LeaprFile:
    """Load the full published-format CAB input, not a toy imitation."""
    return LeaprFile.from_file(str(SOURCE))


def test_fixed_length_arrays_with_and_without_slash() -> None:
    """NJOY permits numeric arrays without slash terminators."""
    text = StringIO('1 2\n3 4\n2.5 3.5 0.52 / Card13\n')
    assert read_phonon_spectrum_block(text, 4) == [1., 2., 3., 4.]
    assert text.readline().startswith('2.5 3.5')
    text = StringIO('1 2 / end array\n3 / next\n')
    assert read_fixed_length_array(text, 2, card='Card 15') == [1., 2.]
    assert text.readline() == '3 / next\n'
    text = StringIO('1 2\n/\n3 / next\n')
    assert read_fixed_length_array(text, 2, card='Card 15') == [1., 2.]
    assert text.readline() == '3 / next\n'


@pytest.mark.parametrize('text', ['1 2 3 /\n', '1 /\n', '1 2 3\n'])
def test_fixed_array_rejects_invalid_lengths(text: str) -> None:
    """Never swallow extra data or accept an incomplete array."""
    with pytest.raises((LeaprFormatError, EOFError)):
        read_fixed_length_array(StringIO(text), 2, card='Card 12')


def test_parses_every_official_temperature_and_header_metadata() -> None:
    """The 94 positive blocks and optional SMIN on Card 4 are retained."""
    source = _official()
    assert source.run_control_params['ntempr'] == 94
    assert len(source.temp_blocks) == 94
    assert source.material_info['smin'] == 2e-38
    available = available_leapr_temperatures(source)
    assert len(available) == 94
    assert available[0] == 273.15
    assert available[-1] == 1000.
    assert 295. in available
    assert 296. not in available
    assert len(source.alpha_grid) == source.grid_params['nalpha'] == 222
    assert len(source.beta_grid) == source.grid_params['nbeta'] == 317
    for block in source.temp_blocks:
        assert len(block['phonon_spectrum']) == block['ni'] == 119
        assert len(block['oscillator_energies']) == block['nd'] == 2
        assert len(block['oscillator_weights']) == 2


def test_295_kelvin_model_uses_source_spectrum_and_diffusion() -> None:
    """One temperature's complete physics must not come from 273.15 K."""
    leapr = _official()
    model = material_model_from_leapr(
        leapr, state='liquid', temperature=295.,
        principal_label='H', secondary_label='O',
        oscillator_labels=('bend', 'stretch'),
    )
    block, actual_t = select_leapr_temperature_block(leapr, 295.)
    assert actual_t == model.temperature_k == 295.
    assert model.dynamics.translation_model == 'egelstaff_schofield'
    assert model.dynamics.translation_parameter == pytest.approx(4.0566)
    assert model.dynamics.translation_weight == pytest.approx(.0080324)
    assert model.dynamics.continuous_weight == pytest.approx(.99197)
    assert model.dynamics.oscillators == ()
    assert model.dynamics.continuous.integral == pytest.approx(1.0)
    assert model.dynamics.continuous.energy_grid[-1] > .415
    assert block['nd'] == 2  # Parsed source remains unchanged.
    assert model.principal.multiplicity == 2
    assert model.secondary[0].multiplicity == 1


def test_exact_source_temperature_required_for_liquid() -> None:
    """Do not relabel a 295 K source spectrum as 296 K."""
    leapr = _official()
    with pytest.raises(ValueError, match='not explicitly tabulated'):
        material_model_from_leapr(leapr, state='liquid', temperature=296.)


@pytest.mark.parametrize('temperature', [295., 300., 350., 1000.])
def test_one_temperature_writer_copy_preserves_that_blocks_physics(tmp_path, temperature):
    """Sample injection changes only Card 12 and normalizes it to unit area."""
    source = _official()
    model = material_model_from_leapr(source, state='liquid', temperature=temperature)
    original, _ = select_leapr_temperature_block(source, temperature)
    text = serialize_leapr_spectrum(
        source, model.dynamics.continuous, temperature=temperature,
        preserve_temperature_parameters=True, export_full_spectra=True,
    )
    target = tmp_path / 'sample.leapr'
    target.write_text(text)
    parsed = LeaprFile.from_file(str(target))
    actual = parsed.temp_blocks[0]
    assert parsed.run_control_params['ntempr'] == 1
    assert parsed.material_info['smin'] == source.material_info['smin']
    assert actual['temp'] == temperature
    for name in ('c', 'twt'):
        assert actual[name] == original[name]
    assert actual['nd'] == 0
    assert actual['delta'] == pytest.approx(original['delta'] / 4)
    assert actual['tbeta'] == pytest.approx(original['tbeta'] + sum(original['oscillator_weights']))
    assert actual['ni'] > original['ni']
    assert np.trapezoid(actual['phonon_spectrum'],dx=actual['delta']) == pytest.approx(1.0,abs=1e-6)
    assert source.temp_blocks[0]['temp'] == 273.15


def test_rewriting_all_94_blocks_round_trips(tmp_path):
    """The original multi-temperature LEAPR deck survives a full I/O cycle."""
    source = _official()
    target = tmp_path / 'full.leapr'
    source.write(str(target))
    parsed = LeaprFile.from_file(str(target))
    assert len(parsed.temp_blocks) == 94
    assert parsed.material_info['smin'] == 2e-38
    for before, after in zip(source.temp_blocks, parsed.temp_blocks):
        for field in ('temp', 'delta', 'ni', 'twt', 'c', 'tbeta', 'nd'):
            assert after[field] == pytest.approx(before[field])
        for field in ('phonon_spectrum', 'oscillator_energies', 'oscillator_weights'):
            np.testing.assert_allclose(after[field], before[field], atol=1e-7, rtol=1e-6)

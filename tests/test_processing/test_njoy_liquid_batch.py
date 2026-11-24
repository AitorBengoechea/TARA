"""NJOY liquid LEAPR batch representation and sample identity."""


import sys
import types
from pathlib import Path
import pytest
from core.config import DiffusionSamplingConfig
from core.sampler.diffusion_sampler import DiffusionSampler
from input_output.leapr.material_adapter import material_model_from_leapr
from input_output.leapr.core import LeaprFile
from tests.support.water import make_water_leapr_template
import numpy as np
from core.data.input_pdos import PhononSpectra


# Sample-specific translation through NJOY

def test_material_ensemble_tapes_preserve_identity_and_new_translation(monkeypatch, tmp_path):
    """Each sample is serialized with its own c and unchanged reference weight."""
    parallel = types.ModuleType('parallel_execution')
    parallel.run_parallel = lambda fn, calls, n_jobs: [fn(*a, **kw) for a, kw in calls]
    monkeypatch.setitem(sys.modules, 'parallel_execution', parallel)
    parser = types.ModuleType('endf_parserpy')
    parser.EndfParserPy = object
    parser.EndfParser = object
    monkeypatch.setitem(sys.modules, 'endf_parserpy', parser)

    from processing.orchestrators import njoy_batch
    template = make_water_leapr_template()
    template.temp_blocks[0]['c'] = 4.0
    path = tmp_path / 'water.leapr'
    template.write(str(path))
    material = material_model_from_leapr(LeaprFile.from_file(str(path)), state='liquid', temperature=296.)
    samples = DiffusionSampler(material, DiffusionSamplingConfig(True, 'diffusivity_only', 5), 11).generate_ensemble(2)
    captured = []

    class Processor:
        """Fake backend that records decks, without claiming physical validity."""
        def __init__(self, njoy_exec):
            """Use an artificial executable name."""
            assert njoy_exec == 'fake-njoy'

        def execute_leapr(self, *, leapr_deck, output_endf_path):
            """Record one deck and create a fake ENDF artifact."""
            captured.append(leapr_deck)
            Path(output_endf_path).write_text('fake endf')

    class Validator:
        """Accept test-only fake tapes."""
        def validate_sab_tape(self, file_path):
            """Check fake tape generation only."""
            return Path(file_path).is_file()

    monkeypatch.setattr(njoy_batch, 'NjoyProcessor', Processor)
    monkeypatch.setattr(njoy_batch, 'EndfValidator', Validator)
    batch = njoy_batch.NjoyBatchOrchestrator(njoy_exec='fake-njoy', n_jobs=1)
    result = batch.generate_ensemble_sab_tapes(
        samples, str(tmp_path / 'out'), str(path), temperature=296., material_state='liquid',
    )
    assert list(result) == ['sample_000', 'sample_001', 'sample_002']
    assert len(captured) == 3
    assert captured[0] != captured[1]
    assert captured[1] != captured[2]
    assert '0.05 4. 0.45/' in captured[0]
    for deck in captured:
        assert deck.count('\nstop\n') == 1
    assert LeaprFile.from_file(str(path)).temp_blocks[0]['c'] == 4.0


def test_material_ensemble_is_not_allowed_in_solid_batch(monkeypatch, tmp_path):
    """The material-level route cannot silently change an existing solid run."""
    parallel = types.ModuleType('parallel_execution')
    parallel.run_parallel = lambda fn, calls, n_jobs: []
    monkeypatch.setitem(sys.modules, 'parallel_execution', parallel)
    parser = types.ModuleType('endf_parserpy')
    parser.EndfParserPy = object
    parser.EndfParser = object
    monkeypatch.setitem(sys.modules, 'endf_parserpy', parser)
    from processing.orchestrators.njoy_batch import NjoyBatchOrchestrator
    source = make_water_leapr_template()
    source.temp_blocks[0]['c'] = 4
    path = tmp_path / 'water.leapr'
    source.write(str(path))
    material = material_model_from_leapr(source, state='liquid', temperature=296.)
    samples = DiffusionSampler(material, DiffusionSamplingConfig(True, 'diffusivity_only', 5), 2).generate_ensemble(1)
    with pytest.raises(ValueError, match='liquid'):
        NjoyBatchOrchestrator('fake').generate_ensemble_sab_tapes(
            samples, str(tmp_path), str(path), material_state='solid',
        )


def test_reference_diffusion_must_match_selected_temperature_block(monkeypatch, tmp_path):
    """Preflight rejects ensemble references originating from another LEAPR input."""
    parallel = types.ModuleType('parallel_execution')
    parallel.run_parallel = lambda fn, calls, n_jobs: []
    monkeypatch.setitem(sys.modules, 'parallel_execution', parallel)
    parser = types.ModuleType('endf_parserpy')
    parser.EndfParserPy = object
    parser.EndfParser = object
    monkeypatch.setitem(sys.modules, 'endf_parserpy', parser)
    from processing.orchestrators.njoy_batch import NjoyBatchOrchestrator
    source = make_water_leapr_template()
    source.temp_blocks[0]['c'] = 4
    path = tmp_path / 'water.leapr'
    source.write(str(path))
    material = material_model_from_leapr(source, state='liquid', temperature=296.)
    samples = DiffusionSampler(material, DiffusionSamplingConfig(True, 'diffusivity_only', 5), 2).generate_ensemble(1)
    source.temp_blocks[0]['c'] = 5.0
    source.write(str(path))
    with pytest.raises(ValueError, match='reference translation'):
        NjoyBatchOrchestrator('fake').generate_ensemble_sab_tapes(
            samples, str(tmp_path), str(path), material_state='liquid', temperature=296.,
        )


# CAB temperature and export representations

SOURCE = Path(__file__).resolve().parents[1] / 'fixtures' / 'tsl-HinH2O.leapr'


class _CabReferenceEnsemble:
    """Keep one reference and one independent sample in a minimal test harness."""

    def __init__(self, reference):
        self.reference = reference
        self.sample = reference

    def __len__(self):
        """Count only the stochastic draws."""
        return 1

    def __getitem__(self, index):
        """Provide one sample, independent of the reference identifier."""
        assert index == 0
        return self.sample


@pytest.mark.parametrize("full", [False, True])
def test_liquid_batch_serializes_the_selected_295_kelvin_block(monkeypatch, tmp_path, full):
    """All physical cards except sampled Card 12 come from the 295 K block."""
    from processing.orchestrators import njoy_batch
    original = LeaprFile.from_file(str(SOURCE))
    reference = material_model_from_leapr(original, state='liquid', temperature=295.)
    captures = []

    class Processor:
        """Mock external NJOY execution while inspecting the real generated decks."""

        def __init__(self, njoy_exec):
            """Accept the stand-in executable only."""
            assert njoy_exec == 'fake-njoy'

        def execute_leapr(self, *, leapr_deck, output_endf_path):
            """Store a generated deck for physical-card inspection."""
            captures.append(leapr_deck)
            Path(output_endf_path).write_text('FAKE ENDF - DO NOT USE')

    class Validator:
        """Mock only the unavailable external ENDF validation."""

        def validate_sab_tape(self, file_path):
            """Confirm a mock output was written."""
            return Path(file_path).is_file()

    monkeypatch.setattr(njoy_batch, 'NjoyProcessor', Processor)
    monkeypatch.setattr(njoy_batch, 'EndfValidator', Validator)
    monkeypatch.setattr(njoy_batch, 'run_parallel', lambda fn, calls, n_jobs: [
        fn(*args, **kwargs) for args, kwargs in calls
    ])
    batch = njoy_batch.NjoyBatchOrchestrator('fake-njoy', n_jobs=1)
    results = batch.generate_ensemble_sab_tapes(
        _CabReferenceEnsemble(reference.dynamics.continuous), str(tmp_path), str(SOURCE),
        temperature=295., material_state='liquid', export_full_spectra=full,
    )
    assert list(results) == ['sample_000', 'sample_001']
    assert len(captures) == 2
    for deck in captures:
        assert '4.0566' in deck
        assert '0.0080324' in deck
        saved = tmp_path / 'captured.leapr'
        saved.write_text(deck)
        parsed = LeaprFile.from_file(str(saved))
        block = parsed.temp_blocks[0]
        if full:
            assert block['nd'] == 0 and block['tbeta'] == pytest.approx(.99197)
            assert parsed.first_positive_energy_grid[-1] > .415
        else:
            assert block['nd'] == 2 and block['ni'] == 119
            assert block['tbeta'] == pytest.approx(.52197)
            assert block['oscillator_energies'] == pytest.approx([.205, .415])
        assert '295./' in deck or '295.0/' in deck or '295/' in deck
        assert '2e-38' in deck
        assert deck.count('\nstop\n') == 1


def test_liquid_batch_refuses_296_kelvin_before_any_jobs(monkeypatch, tmp_path):
    """Never borrow another temperature's diffusion/VDOS to fake a 296 K ACE."""
    from processing.orchestrators import njoy_batch
    original = LeaprFile.from_file(str(SOURCE))
    reference = material_model_from_leapr(original, state='liquid', temperature=295.)

    def bad_run(*args, **kwargs):
        raise AssertionError('NJOY workers should not be scheduled')
    monkeypatch.setattr(njoy_batch, 'run_parallel', bad_run)
    batch = njoy_batch.NjoyBatchOrchestrator('fake-njoy', n_jobs=1)
    with pytest.raises(ValueError, match='not explicitly tabulated'):
        batch.generate_ensemble_sab_tapes(
            _CabReferenceEnsemble(reference.dynamics.continuous), str(tmp_path / 'no_output'),
            str(SOURCE), temperature=296., material_state='liquid',
        )
    assert not (tmp_path / 'no_output').exists()


# Sampled spectra and independent reference

class _SampledWaterEnsemble:
    """Minimal separate-reference ensemble respecting TARA's indexing contract."""

    def __init__(self, template: LeaprFile):
        # Change only the continuum. Both inputs must still include the modes.
        block = {**template.temp_blocks[0], 'phonon_spectrum': np.array([0., 3., 1., 0.])}
        sample_template = template.get_writer_copy(temp_blocks=[block])
        self.reference = PhononSpectra(template.vibrational_pdos(296.))
        self.samples = [PhononSpectra(sample_template.vibrational_pdos(296.))]

    def __len__(self):
        """Return stochastic sample count excluding the reference."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return stochastic sample indexed from zero."""
        return self.samples[index]


def test_sampled_liquid_tape_generator_keeps_reference_and_modes(monkeypatch, tmp_path):
    """Preflight and NJOY deck serialization include both reference and sample.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patch external-only execution resources.
    tmp_path : pathlib.Path
        Isolated fixture directory.
    """
    # The generated context archive does not include the root shared parallel
    # helper; supply a test-only deterministic sequential replacement.
    parallel = types.ModuleType('parallel_execution')
    parallel.run_parallel = lambda fn, calls, n_jobs: [fn(*args, **kwargs) for args, kwargs in calls]
    monkeypatch.setitem(sys.modules, 'parallel_execution', parallel)
    parser = types.ModuleType('endf_parserpy')
    parser.EndfParserPy = object
    parser.EndfParser = object
    monkeypatch.setitem(sys.modules, 'endf_parserpy', parser)

    from processing.orchestrators import njoy_batch
    from input_output.leapr.core import LeaprFile

    original = make_water_leapr_template()
    source = tmp_path / 'source.leapr'
    original.write(str(source))
    captured = []

    class Processor:
        """Emulate the Processor boundary, not NJOY physics."""

        def __init__(self, njoy_exec):
            """Record the fake executable selection."""
            assert njoy_exec == 'not-an-njoy-executable'

        def execute_leapr(self, *, leapr_deck, output_endf_path):
            """Capture a syntactically built deck and emit a fake TSL file."""
            captured.append(leapr_deck)
            from pathlib import Path
            Path(output_endf_path).write_text('mock MF7: not a real ENDF evaluation')

    class Validator:
        """Accept fake output so the test isolates serialization logic."""

        def validate_sab_tape(self, file_path):
            """Confirm the generated fake tape exists."""
            from pathlib import Path
            return Path(file_path).is_file()

    monkeypatch.setattr(njoy_batch, 'NjoyProcessor', Processor)
    monkeypatch.setattr(njoy_batch, 'EndfValidator', Validator)
    worker = njoy_batch.NjoyBatchOrchestrator(njoy_exec='not-an-njoy-executable', n_jobs=1)
    output = worker.generate_ensemble_sab_tapes(
        _SampledWaterEnsemble(original), str(tmp_path / 'endf'), str(source), temperature=296.,
        material_state='liquid',
    )
    assert list(output) == ['sample_000', 'sample_001']
    assert len(captured) == 2
    assert captured[0] != captured[1]
    exported = []
    for index, deck in enumerate(captured):
        assert deck.count('\nstop\n') == 1
        path = tmp_path / f'captured_{index}.leapr'
        path.write_text(deck)
        block = LeaprFile.from_file(str(path)).temp_blocks[0]
        assert block['nd'] == 2
        assert (block['twt'], block['c'], block['tbeta']) == pytest.approx((.05, 0., .45))
        np.testing.assert_allclose(block['oscillator_energies'], [.205, .480])
        np.testing.assert_allclose(block['oscillator_weights'], [.167, .333])
        exported.append(block['phonon_spectrum'])
    np.testing.assert_allclose(exported[0], original.temp_blocks[0]['phonon_spectrum'])
    assert not np.allclose(exported[0], exported[1])
    # The source template must not be mutated by either stochastic worker.
    restored = LeaprFile.from_file(str(source))
    assert restored.temp_blocks[0]['oscillator_weights'].tolist() == [.167, .333]

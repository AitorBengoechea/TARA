"""Application-level orchestration of the existing TARA ACE production APIs.

This module does not implement sampling mathematics, ENDF processing, or ACE
serialization; those responsibilities remain in Core, Input/Output and Processing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from application.configuration import AceApplicationConfig, ApplicationConfigError


def _executable(name: str) -> Path:
    """Resolve an NJOY executable by explicit path or by executable name.

    Parameters
    ----------
    name : str
        Name or path of the required NJOY executable.

    Returns
    -------
    Path
        Resolved executable path.
    """
    candidate = Path(name).expanduser()
    if candidate.is_absolute() or candidate.parent != Path('.'):
        result = candidate.resolve()
    else:
        resolved = shutil.which(name)
        if resolved is None:
            raise FileNotFoundError(f"NJOY executable '{name}' not found on PATH.")
        result = Path(resolved).resolve()
    if not result.is_file() or not os.access(result, os.X_OK):
        raise FileNotFoundError(f"NJOY executable does not exist or is not executable: {result}")
    return result


def _existing_input(path: Path, label: str) -> Path:
    """Require a nonempty source material file before starting calculations."""
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")
    return path


def _prepare_ensemble(config: AceApplicationConfig, pdos: Any, *, material: Any = None):
    """Create a reproducibly seeded ensemble using the public Core APIs.

    Parameters
    ----------
    config : AceApplicationConfig
        Validated workflow or backend configuration.
    pdos : Any
        Source phonon density of states.
    material : MaterialScatteringModel, optional
        Physical reference for workflows with diffusion sampling.

    Returns
    -------
    SpectraEnsemble or MaterialEnsemble
        Reference and stochastic realizations for the active perturbations.
    """
    import numpy as np
    from core.config import TaraConfig
    from core.data.input_pdos import PhononSpectra
    from core.sampler.concrete_samplers import TaraSampler

    sampling = config.section('sampling')
    ensemble = config.section('ensemble')
    diffusion = config.section('diffusion_sampling') if material is not None else None
    tara_config = TaraConfig(
        n_samples=ensemble['n_samples'],
        max_energy_shift_mev=sampling['max_energy_shift_mev'],
        max_amplitude_shift_pct=sampling['max_amplitude_shift_pct'],
        grid_partition_method=sampling['grid_partition_method'],
        automatic_partition_prominence_pct=sampling['automatic_partition_prominence_pct'],
        manual_energy_ranges=[tuple(v) for v in sampling.get('manual_energy_ranges', [])] or None,
        domain_splitting_strategy=sampling['domain_splitting_strategy'],
        correlation_strategy=sampling['correlation_strategy'],
        variable_coupling=sampling['variable_coupling'],
        shared_grid_structure=sampling['shared_grid_structure'],
        freeze_origin=sampling['freeze_origin'],
        fast_amplitude=sampling['fast_amplitude'],
        sampling_method=ensemble.get('sampling_method', 'latin_hypercube'),
        random_seed=ensemble['random_seed'],
        diffusion=diffusion,
    )
    spectrum = None
    if sampling['max_energy_shift_mev'] > 0 or sampling['max_amplitude_shift_pct'] > 0:
        spectrum = PhononSpectra(pdos=pdos, auto_normalize=sampling['auto_normalize'])
        # Legacy vibrational Monte Carlo consumes the global NumPy RNG.
        if ensemble.get('sampling_method', 'latin_hypercube') == 'monte_carlo':
            np.random.seed(ensemble['random_seed'])
    sampler = TaraSampler(
        config=tara_config, spectra=spectrum, material=material,
        amplitude_method=sampling['amplitude_method'],
        spatial_method=sampling['spatial_method'],
    )
    samples = sampler.generate_ensemble(batch_size=ensemble['batch_size'])
    if ensemble['include_reference'] and samples.reference is None:
        raise RuntimeError('The generated ensemble is missing the sample_000 reference.')
    if len(samples) != ensemble['n_samples']:
        raise RuntimeError(
            f"Sample count mismatch: requested {ensemble['n_samples']}, got {len(samples)}."
        )
    return samples


def _ncrystal_tapes(
    config: AceApplicationConfig, ensemble: Any, template: Path, endf_dir: Path
) -> dict[str, str]:
    """Produce NCrystal TSL tapes, optionally omitting the reference spectrum.

    Parameters
    ----------
    config : AceApplicationConfig
        Validated workflow or backend configuration.
    ensemble : Any
        Reference spectrum and stochastic samples.
    template : Path
        NCMAT template path.
    endf_dir : Path
        Directory for generated ENDF thermal-scattering tapes.

    Returns
    -------
    dict[str, str]
        Sample identifiers mapped to ENDF file paths.
    """
    from processing.orchestrators.ncrystal_batch import NcrystalBatchOrchestrator

    material = config.section('material')
    temperature = material['temperature_k']
    n_jobs = config.section('execution')['n_jobs']
    return NcrystalBatchOrchestrator(n_jobs=n_jobs).generate_ensemble_sab_tapes(
        ensemble=ensemble, output_dir=str(endf_dir),
        base_ncmat_file=str(template), temperature=temperature,
        atom_label=material['atom_label'], elastic_mode=material['elastic_mode'],
        include_reference=config.section('ensemble')['include_reference'],
    )


def _prepare_diffusion_ensemble(config: AceApplicationConfig, model: Any, pdos: Any):
    """Prepare translation-only or joint realizations for a physical liquid."""
    if model is None:
        raise ValueError("Diffusion sampling requires a liquid physical model.")
    return _prepare_ensemble(config, pdos, material=model)


def _process_njoy(config: AceApplicationConfig, endf_dir: Path, njoy: Path):
    """Load the LEAPR material, sample it, and generate NJOY MF7 data.

    Parameters
    ----------
    config : AceApplicationConfig
        Validated workflow or backend configuration.
    endf_dir : Path
        Directory for generated ENDF thermal-scattering tapes.
    njoy : Path
        Resolved NJOY executable path.

    Returns
    -------
    object
        Generated ENDF path mapping and sampled ensemble.
    """
    from input_output.factory import load_file_object
    from processing.orchestrators.njoy_batch import NjoyBatchOrchestrator

    source = _existing_input(config.path('material', 'leapr_file'), 'LEAPR material')
    template, pdos = load_file_object(filepath=source)
    if template is None:
        raise TypeError(f'LEAPR source did not supply a material template: {source}')
    material = config.section('material')
    physical_model = None
    if material.get('state') == 'liquid':
        # The original temperature block owns the liquid diffusion parameters.
        from input_output.leapr.material_adapter import material_model_from_leapr
        physical_model = material_model_from_leapr(
            template, state='liquid', temperature=material['temperature_k'],
        )
        pdos = physical_model.dynamics.continuous.pdos
    diffusion_options = config.sections.get('diffusion_sampling', {})
    if diffusion_options.get('enabled', False):
        samples = _prepare_diffusion_ensemble(config, physical_model, pdos)
    else:
        samples = _prepare_ensemble(config, pdos)
    execution = config.section('execution')
    # Preserve the original call signature for legacy solid configurations.
    state_options = {'material_state': material['state']} if 'state' in material else {}
    tapes = NjoyBatchOrchestrator(
        njoy_exec=str(njoy), n_jobs=execution['n_jobs'],
    ).generate_ensemble_sab_tapes(
        ensemble=samples, output_dir=str(endf_dir),
        base_leapr_file=str(source),
        temperature=material['temperature_k'],
        **state_options,
    )
    return tapes, samples


def _process_ncrystal(config: AceApplicationConfig, endf_dir: Path):
    """Load the NCrystal material and generate its MF7 evaluation ensemble.

    Parameters
    ----------
    config : AceApplicationConfig
        Validated workflow or backend configuration.
    endf_dir : Path
        Directory for generated ENDF thermal-scattering tapes.

    Returns
    -------
    object
        Generated ENDF path mapping and sampled ensemble.
    """
    from input_output.ncrystal.core import NCrystalFile

    material = config.section('material')
    atom_label = material['atom_label']
    if 'ncmat_file' in material:
        source = _existing_input(config.path('material', 'ncmat_file'), 'NCMAT material')
        ncmat = NCrystalFile.from_file(str(source))
        samples = _prepare_ensemble(config, ncmat.pdos_for_atom(atom_label))
        return _ncrystal_tapes(config, samples, source, endf_dir), samples

    import NCrystal as NC

    with tempfile.TemporaryDirectory(prefix='tara_ace_ncmat_') as folder:
        name = material['ncrystal_builtin']
        source = Path(folder) / Path(name).name
        source.write_text(NC.createTextData(name).rawData, encoding='utf-8')
        ncmat = NCrystalFile.from_file(str(source))
        samples = _prepare_ensemble(config, ncmat.pdos_for_atom(atom_label))
        tapes = _ncrystal_tapes(config, samples, source, endf_dir)
    return tapes, samples


def validate_ace_inputs(config: AceApplicationConfig) -> None:
    """Check required material files and NJOY executable before a run.

    Parameters
    ----------
    config : AceApplicationConfig
        Parsed, validated ACE application settings.
    """
    _existing_input(config.path('material', 'base_endf_file'), 'neutron ENDF')
    _executable(config.section('execution')['njoy_executable'])
    if config.backend == 'njoy':
        _existing_input(config.path('material', 'leapr_file'), 'LEAPR material')
    elif 'ncmat_file' in config.section('material'):
        _existing_input(config.path('material', 'ncmat_file'), 'NCMAT material')


def _expected_sample_ids(config: AceApplicationConfig, samples: Any) -> set[str]:
    """Collect required sample identifiers, including an optional reference.

    Parameters
    ----------
    config : AceApplicationConfig
        User-supplied workflow settings.
    samples : SpectraEnsemble
        Generated source spectra.

    Returns
    -------
    set of str
        Expected sample identifiers for both the ENDF and ACE stages.
    """
    requested_ids = {f'sample_{i + 1:03d}' for i in range(len(samples))}
    if config.section('ensemble')['include_reference']:
        requested_ids.add('sample_000')
    return requested_ids


def _require_complete_batch(
    actual: dict[str, Any], expected: set[str], stage: str,
) -> None:
    """Reject incomplete ENDF or ACE batches before reporting success.

    Parameters
    ----------
    actual : dict
        Generated sample-ID-to-output mapping.
    expected : set of str
        Required sample identifiers.
    stage : str
        Stage name to include in error messages.
    """
    if set(actual) != expected:
        raise RuntimeError(
            f'Incomplete {stage} batch: expected {sorted(expected)}, '
            f'got {sorted(actual)}.'
        )


def _ace_reaction_overrides(material: dict, opts: dict) -> dict[str, Any]:
    """Derive optional NJOY ACE reaction settings without changing defaults.

    Parameters
    ----------
    material : dict
        Material configuration, including optional liquid state.
    opts : dict
        ACE configuration options.

    Returns
    -------
    dict of str to object
        Only explicitly requested or required additional options.
    """
    liquid = material.get('state') == 'liquid'
    ace_reactions = {}
    if 'mti' in opts:
        ace_reactions['mti'] = opts['mti']
    if 'mte' in opts or liquid:
        ace_reactions['mte'] = opts.get('mte', 0)
    if liquid:
        ace_reactions['allow_additional_scatterers'] = True
    return ace_reactions


def _validate_ace_artifacts(result: dict[str, str], output: Path, save_xsdir: bool) -> None:
    """Reject missing, empty, or incorrectly serialized ACE output files.

    Parameters
    ----------
    result : dict of str to str
        Per-sample ACE output paths.
    output : pathlib.Path
        Workflow output directory.
    save_xsdir : bool
        Whether each generated sample needs an XSDIR entry.
    """
    for sid, raw_path in result.items():
        path = Path(raw_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'ACE table missing or empty for {sid}: {path}')
        with path.open('rb') as file:
            if b'filename route' in file.read(1024).lower():
                raise RuntimeError(f'XSDIR text appeared in ACE file for {sid}: {path}')
        if save_xsdir:
            xsdir = output / 'ace' / 'xsdir' / f'{sid}.xsdir'
            if not xsdir.is_file() or xsdir.stat().st_size == 0:
                raise RuntimeError(f'XSDIR entry missing for {sid}: {xsdir}')


def _write_ace_provenance(
    config: AceApplicationConfig, output: Path, material: dict,
    njoy: Path, result: dict[str, str], samples: Any = None,
) -> None:
    """Retain run configuration and input/output provenance after success.

    Parameters
    ----------
    config : AceApplicationConfig
        Resolved user settings.
    output : pathlib.Path
        Workflow output directory.
    material : dict
        Material settings and input paths.
    njoy : pathlib.Path
        Resolved NJOY executable.
    result : dict of str to str
        Validated ACE output paths.
    """
    # Never save a successful-run manifest for an incomplete batch.
    output.mkdir(parents=True, exist_ok=True)
    retained_config = output / 'run_config.toml'
    if config.file_path != retained_config.resolve():
        shutil.copyfile(config.file_path, retained_config)
    manifest = {
        'backend': config.backend,
        'config_sha256': hashlib.sha256(config.file_path.read_bytes()).hexdigest(),
        'resolved_input_paths': {
            k: str(config.path('material', k))
            for k in ('base_endf_file', 'leapr_file', 'ncmat_file') if k in material
        },
        'resolved_output_dir': str(output),
        'njoy_executable': str(njoy),
        'temperature_k': material['temperature_k'],
        'random_seed': config.section('ensemble')['random_seed'],
        'sampling_method': config.section('ensemble').get('sampling_method', 'latin_hypercube'),
        'sample_ids': sorted(result),
        'settings': config.sections,
    }
    if hasattr(samples, 'generation'):
        manifest['sampling_topologies'] = [
            {'start_index': b.start_index, 'end_index': b.end_index,
             'n_domains': len(b.boundaries)} for b in samples.sampling_blocks
        ]
    if getattr(samples, 'spectra_ensemble', None) is not None:
        manifest['sampling_topologies'] = [
            {'start_index': b.start_index, 'end_index': b.end_index,
             'n_domains': len(b.boundaries)}
            for b in samples.spectra_ensemble.sampling_blocks
        ]
    if hasattr(samples, 'sample_info') and hasattr(samples, 'diffusion_parameters_c'):
        manifest['diffusion_samples'] = samples.sample_info.reset_index().to_dict(orient='records')
    (output / 'run_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )


def run_ace_workflow(config: AceApplicationConfig) -> dict[str, str]:
    """Generate a complete ACE ensemble from a validated TOML configuration.

    Parameters
    ----------
    config : AceApplicationConfig
        Strictly validated user configuration.

    Returns
    -------
    dict of str to str
        Sample identifiers mapped to generated ACE paths.

    Raises
    ------
    FileNotFoundError
        A required input file or NJOY executable is absent.
    RuntimeError
        The processing steps produced an incomplete or empty ACE batch.
    """
    from processing.core.config import AceGenerationConfig
    from processing.pipelines.ace_pipeline import UniversalAcePipeline

    material = config.section('material')
    validate_ace_inputs(config)
    base_endf = config.path('material', 'base_endf_file')
    njoy = _executable(config.section('execution')['njoy_executable'])
    output = config.path('output', 'directory')
    # Invalidate prior provenance before any artifacts can be overwritten.
    (output / 'run_manifest.json').unlink(missing_ok=True)
    endf_dir = output / 'endf'
    if config.backend == 'njoy':
        tapes, samples = _process_njoy(config, endf_dir, njoy)
    else:
        tapes, samples = _process_ncrystal(config, endf_dir)

    requested_ids = _expected_sample_ids(config, samples)
    _require_complete_batch(tapes, requested_ids, 'ENDF')
    opts = config.section('ace')
    ace_options = AceGenerationConfig.from_endf(
        base_endf_path=str(base_endf), tsl_endf_files=tapes,
        temperature=material['temperature_k'],
        emax=opts['max_energy_ev'], suffix=opts['suffix'],
        nbin=opts['nbin'], tol=opts['tolerance'],
        **_ace_reaction_overrides(material, opts),
    )
    result = UniversalAcePipeline(
        njoy_exec=str(njoy), n_jobs=config.section('execution')['n_jobs'],
    ).run(
        sab_tapes=tapes, output_dir=str(output / 'ace'),
        config=ace_options, save_xsdir=opts['save_xsdir'],
    )
    _require_complete_batch(result, requested_ids, 'ACE')
    _validate_ace_artifacts(result, output, opts['save_xsdir'])
    if hasattr(samples, 'diffusion_parameters_c'):
        _write_ace_provenance(config, output, material, njoy, result, samples=samples)
    else:
        _write_ace_provenance(config, output, material, njoy, result)
    return result


def run_ncrystal_native_ensemble(
    ensemble, *, ncmat_template: str | Path, output_dir: str | Path,
    temperature_k: float, atom_label: str, energies_ev=None,
    event_energy_ev: float | None = None, n_events: int = 0,
    physical_model=None, verify_plugin: bool = True,
    vdos_moment_policy: str = "warn", max_moment_change: float = 0.10,
    max_shape_l1_change: float = 0.10,
) -> dict[str, dict[str, str]]:
    """Execute an NCrystal native material ensemble using the same backend.

    The NCMAT may represent any supported liquid model or composition;
    composition, diffusion, coherent options and Young--Koppel state are
    preserved verbatim. This is *not* an ACE production workflow.

    Parameters
    ----------
    ensemble : object
        Reference spectrum and stochastic samples.
    ncmat_template : str | Path
        NCrystal liquid NCMAT template path.
    output_dir : str | Path
        Destination directory for generated outputs.
    temperature_k : float
        Physical temperature in kelvin.
    atom_label : str
        Target NCrystal element or isotope label.
    energies_ev : object
        Incident energy grid in electronvolts; defaults to the built-in grid.
    event_energy_ev : float | None
        Incident energy in electronvolts for optional event sampling.
    n_events : int
        Number of final-state scattering events to sample.
    physical_model : object
        Optional fixed liquid physical model for mode composition.
    verify_plugin : bool
        Whether to verify the compiled NCrystal Liquids plugin.
    vdos_moment_policy : str
        Whether to warn or raise on excessive displacement-moment changes.
    max_moment_change : float
        Maximum accepted relative displacement-moment change.
    max_shape_l1_change : float
        Maximum accepted relative VDOS shape distortion.

    Returns
    -------
    dict[str, dict[str, str]]
        Sample identifiers mapped to native material, CSV, and optional event files.
    """
    import numpy as np
    from processing.orchestrators.ncrystal_batch import NcrystalBatchOrchestrator
    from input_output.ncrystal.liquids import load_liquid_ncmat

    load_liquid_ncmat(ncmat_template)
    energies = np.geomspace(1e-5, 1.0, 90) if energies_ev is None else energies_ev
    return NcrystalBatchOrchestrator().generate_native_ensemble(
        ensemble, str(output_dir), str(ncmat_template), temperature_k,
        atom_label, energies, sample_energy_ev=event_energy_ev,
        n_events=n_events, physical_model=physical_model,
        verify_plugin=verify_plugin,
        vdos_moment_policy=vdos_moment_policy,
        max_moment_change=max_moment_change,
        max_shape_l1_change=max_shape_l1_change,
    )

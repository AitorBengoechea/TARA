"""Isolated native liquid backend for Analyzer; no MF7 or ACE production."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from input_output.ncrystal.liquids import load_liquid_ncmat, replace_vdos, parse_custom_liquids
from input_output.ncrystal.core import NCrystalFile
from input_output.ncrystal.material_adapter import displacement_moment
from input_output.spectrum_grid import resample_on_source_spacing
from processing.backends.ncmat_isolation import normalize_active_species
from processing.backends.liquid_capture.data import LiquidAnalysisState, LiquidKernel


def is_liquid_template(filename):
    """Return whether an existing NCMAT contains custom liquid settings."""
    path = Path(filename)
    return path.is_file() and '@CUSTOM_LIQUIDS' in path.read_text(encoding='utf-8')


class NCrystalLiquidAnalysisProcessor:
    """Prepare one material and capture its real liquid kernel in a child.

    Parameters
    ----------
    base_ncmat_file : path-like
        Source material, retained unchanged.
    temperature : float
        Requested physical temperature in kelvin.
    capture_library : path-like, optional
        Existing adapter built against NCrystal 4.4.6. If omitted, build
        one in a temporary directory owned by this processor.
    temp_dir : path-like, optional
        Parent of unique per-call temporary directories.
    active_species : collection of str, optional
        Scattering-active labels, preserving original atomic fractions.
    vdoslux : int
        Native kernel resolution, from zero through five.
    timeout : float
        Maximum child execution time in seconds.
    """

    def __init__(self, base_ncmat_file, temperature, *, capture_library=None,
                 temp_dir=None, active_species=None, vdoslux=3, timeout=600):
        self.source = load_liquid_ncmat(base_ncmat_file)
        template = NCrystalFile.from_file(str(base_ncmat_file))
        self.reference_grids = {
            item.atom: template.pdos_for_atom(item.atom).index.to_numpy(dtype=float)
            for item in parse_custom_liquids(self.source)
        }
        self.temperature = float(temperature)
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Temperature must be finite and positive')
        if type(vdoslux) is not int or not 0 <= vdoslux <= 5:
            raise ValueError('vdoslux must be an integer in 0..5')
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        self.vdoslux, self.timeout = vdoslux, timeout
        self.temp_dir = Path(temp_dir or tempfile.gettempdir()).resolve()
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.active_species = (None if active_species is None
                               else normalize_active_species(active_species))

        self._capture_build_dir = None
        try:
            if capture_library is None:
                from processing.backends.liquid_capture.build import build_capture_library

                self._capture_build_dir = tempfile.TemporaryDirectory(
                    prefix="tara_liquid_capture_", dir=self.temp_dir,
                )
                capture_library = build_capture_library(self._capture_build_dir.name)
            self.capture_library = Path(capture_library).resolve()
            self.identity = self._verify_adapter()
        except BaseException:
            if self._capture_build_dir is not None:
                self._capture_build_dir.cleanup()
            raise

    def __getstate__(self):
        """Share the compiled adapter path with workers.

        Returns
        -------
        dict
            Worker state. The parent processor owns temporary build cleanup.
        """
        state = self.__dict__.copy()
        state["_capture_build_dir"] = None
        return state

    def _verify_adapter(self):
        """Reject missing or changed adapter builds before starting samples."""
        import NCrystal as NC
        metadata = json.loads(self.capture_library.with_suffix('.json').read_text())
        if metadata.get('schema') != 1:
            raise RuntimeError('Unsupported adapter manifest schema')
        if metadata['ncrystal_version'] != NC.__version__ or NC.__version__ != '4.4.6':
            raise RuntimeError('Unsupported capture ABI: NCrystal 4.4.6 required')
        from NCrystal._locatelib import get_libpath_and_namespace
        runtime_library, namespace = get_libpath_and_namespace()
        if namespace or Path(runtime_library).resolve() != Path(metadata['library_path']).resolve():
            raise RuntimeError('Python and adapter must use the same unnamespaced NCrystal library')
        for filename, key in ((self.capture_library, 'adapter_sha256'),
                              (Path(__file__).parent / 'liquid_capture' / 'sabtap.cc', 'source_sha256'),
                              (Path(metadata['library_path']), 'library_sha256')):
            if hashlib.sha256(filename.read_bytes()).hexdigest() != metadata[key]:
                raise RuntimeError('Capture installation changed; rebuild the adapter')
        return metadata

    def create_analysis_state(self, sample_index, spectrum, atom_label=None, *,
                              energy_grid=None):
        """Capture one sampled material and load results before sandbox cleanup.

        Parameters
        ----------
        sample_index : int
            Stable Analyzer identity, including -1 for the independent reference.
        spectrum : PhononSpectra or dict
            Solid-like VDOS sample(s), keyed by material label for multiple inputs.
        atom_label : str, optional
            Target component. Required for a single spectrum.
        energy_grid : array-like, optional
            Exact grid for native cross sections; use the same grid in Parser.

        Returns
        -------
        LiquidAnalysisState
            In-memory arrays and provenance, with no surviving temporary files.
        """
        spectra = spectrum if isinstance(spectrum, dict) else {atom_label: spectrum}
        target = atom_label or next(iter(spectra), None)
        if not target or target not in spectra:
            raise ValueError('An explicit valid atom_label is required')
        material = self.source
        representation = {}
        for label, current in spectra.items():
            if label not in self.reference_grids:
                raise ValueError(f'No custom-liquid source VDOS for {label!r}')
            sample_grid = np.asarray(current.energy_grid, dtype=float)
            sample_density = np.asarray(current.density_values, dtype=float)
            grid, density = resample_on_source_spacing(
                self.reference_grids[label], sample_grid, sample_density)
            material = replace_vdos(material, label, grid, density)
            positive = sample_grid > 0
            before = displacement_moment(sample_grid[positive], sample_density[positive], self.temperature)
            after = displacement_moment(grid, density, self.temperature)
            representation[label] = {
                'method': 'original source spacing with upper extension',
                'source_points': len(self.reference_grids[label]), 'serialized_points': len(grid),
                'source_spacing_eV': float(np.diff(self.reference_grids[label])[0]),
                'input_area': float(np.trapezoid(sample_density, sample_grid)),
                'serialized_area_with_debye_tail': float(np.trapezoid(density, grid) + density[0] * grid[0] / 3),
                'displacement_moment_relative_change': float((after - before) / before),
            }
        energies = np.asarray(np.geomspace(1e-5, 10, 1000) if energy_grid is None
                              else energy_grid, dtype=float)
        if (energies.ndim != 1 or not len(energies) or not np.isfinite(energies).all()
                or np.any(energies <= 0) or np.any(np.diff(energies) <= 0)):
            raise ValueError('Cross-section energies must be finite, positive and increasing')
        with tempfile.TemporaryDirectory(prefix=f'liquid_{sample_index}_', dir=self.temp_dir) as tmp:
            root = Path(tmp)
            (root / 'material.ncmat').write_text(material, encoding='utf-8')
            np.savetxt(root / 'energies.txt', energies, fmt='%.17g')
            request = {'target': target, 'temperature': self.temperature,
                       'vdoslux': self.vdoslux, 'library_path': self.identity.get('library_path'),
                       'active_species': None if self.active_species is None else sorted(self.active_species)}
            (root / 'request.json').write_text(json.dumps(request))
            env = os.environ.copy()
            env['NCRYSTAL_SAB_DUMP_DIR'] = str(root)
            env['LD_PRELOAD'] = str(self.capture_library) + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
            source_root = str(Path(__file__).resolve().parents[2])
            env['PYTHONPATH'] = source_root + os.pathsep + env.get('PYTHONPATH', '')
            command = [sys.executable, '-m', 'processing.backends.liquid_capture.child', str(root / 'request.json')]
            with (root / 'process.log').open('w') as log:
                try:
                    result = subprocess.run(command, env=env, stdout=log, stderr=log,
                                            timeout=self.timeout, check=False)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f'Liquid capture timed out for sample {sample_index}') from exc
            if result.returncode:
                with (root / 'process.log').open('rb') as log:
                    log.seek(max(0, (root / 'process.log').stat().st_size - 6000))
                    detail = log.read().decode(errors='replace')
                raise RuntimeError(f'Liquid capture failed for sample {sample_index}:\n{detail}')
            metadata = json.loads((root / 'result.json').read_text())
            with np.load(root / 'result.npz', allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
        kernels = {}
        for label, component in metadata['components'].items():
            component = dict(component)
            prefix = component.pop('prefix')
            kernels[label] = LiquidKernel(label=label, **component,
                **{key: arrays[f'{prefix}_{key}'] for key in ('alpha', 'beta', 'sab')})
        original = spectra[target]
        metadata['provenance']['adapter'] = self.identity
        metadata['provenance']['sample_index'] = sample_index
        metadata['provenance']['material_ncmat'] = material
        metadata['provenance']['vdos_representation'] = representation
        return LiquidAnalysisState(
            target_lbl=target, kernels=kernels,
            energy_grid=np.asarray(original.energy_grid).copy(),
            pdos=np.asarray(original.density_values).copy(),
            backend_energy_grid=arrays['backend_energy_grid'], backend_pdos=arrays['backend_pdos'],
            xs_energy=arrays['xs_energy'], xs_scattering=arrays['xs_scattering'],
            xs_absorption=arrays['xs_absorption'], vdos_analysis=metadata['vdos_analysis'],
            provenance=metadata['provenance'])

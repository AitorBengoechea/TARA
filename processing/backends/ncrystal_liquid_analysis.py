"""Isolated native liquid backend for Analyzer; no MF7 or ACE production."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from core.data.input_pdos import PhononSpectra
from input_output.ncrystal.liquids import (
    load_liquid_ncmat, parse_custom_liquids, replace_custom_liquids, replace_vdos,
)
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
    defer_capture : bool, optional
        Defer adapter validation until a request needs captured kernels.
    """

    def __init__(self, base_ncmat_file, temperature, *, capture_library=None,
                 temp_dir=None, active_species=None, vdoslux=3, timeout=600,
                 defer_capture=False):
        self.source = load_liquid_ncmat(base_ncmat_file)
        template = NCrystalFile.from_file(str(base_ncmat_file))
        self.entries = parse_custom_liquids(self.source)
        source_pdos = {item.atom: template.pdos_for_atom(item.atom) for item in self.entries}
        self.reference_grids = {label: pdos.index.to_numpy(dtype=float)
                                for label, pdos in source_pdos.items()}
        self.reference_spectra = {label: PhononSpectra(pdos) for label, pdos in source_pdos.items()}
        self._capture_library_hint = capture_library
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
        self.capture_library = None
        self.identity = None
        if not defer_capture:
            self.initialize_capture()

    def initialize_capture(self):
        """Build or validate the capture adapter before dispatching workers.

        Raises
        ------
        RuntimeError
            If the capture adapter and installed NCrystal ABI are incompatible.
        """
        if self.identity is not None:
            return
        capture_library = self._capture_library_hint
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
            self.capture_library = None
            if self._capture_build_dir is not None:
                self._capture_build_dir.cleanup()
                self._capture_build_dir = None
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
                              energy_grid=None, capture_kernel=True):
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
        capture_kernel : bool, optional
            Capture the kernel for differential observables.

        Returns
        -------
        LiquidAnalysisState
            In-memory arrays and provenance, with no surviving temporary files.
        """
        spectra = spectrum if isinstance(spectrum, dict) else {atom_label: spectrum}
        target = atom_label or next(iter(spectra), None)
        if not target or target not in spectra:
            raise ValueError('An explicit valid atom_label is required')
        material, representation = self._prepare_vdos(self.source, spectra)
        return self._evaluate_material(
            sample_index, spectra[target], material, target, representation,
            energy_grid, capture_kernel,
        )

    def create_material_analysis_state(self, sample_index, realization, *,
                                       atom_label=None, energy_grid=None,
                                       capture_kernel=False):
        """Evaluate a translation-only or joint liquid realization.

        Parameters
        ----------
        sample_index : int
            Analyzer identity, including -1 for the reference.
        realization : MaterialScatteringModel
            Complete liquid material with a tabulated vibrational spectrum.
        atom_label : str, optional
            Target species. Defaults to the realization's principal species.
        energy_grid : array_like, optional
            Incident energies in eV.
        capture_kernel : bool, optional
            Capture the kernel for differential observables.

        Returns
        -------
        LiquidAnalysisState
            Native observables and optional completed scattering kernels.
        """
        target = realization.principal.label if atom_label is None else atom_label
        dynamics = realization.dynamics
        if (realization.state != "liquid" or target != realization.principal.label
                or dynamics.translation_model != "egelstaff_schofield" or dynamics.oscillators):
            raise ValueError("Expected a matching liquid species with tabulated Egelstaff–Schofield dynamics.")
        if not np.isclose(realization.temperature_k, self.temperature, rtol=0, atol=1e-8):
            raise ValueError("Sample temperature differs from the configured NCrystal temperature.")
        material = self._apply_translation(dynamics, target)
        spectrum = dynamics.continuous
        reference = self.reference_spectra[target]
        # Retain the exact source VDOS for translation-only samples and the reference.
        if (np.array_equal(spectrum.energy_grid, reference.energy_grid)
                and np.array_equal(spectrum.density_values, reference.density_values)):
            representation = {}
        else:
            material, representation = self._prepare_vdos(material, {target: spectrum})
        return self._evaluate_material(
            sample_index, spectrum, material, target, representation, energy_grid, capture_kernel,
        )

    def _apply_translation(self, dynamics, target):
        """Apply sampled translation parameters to the target species.

        Parameters
        ----------
        dynamics : SpeciesDynamics
            Sampled translational weight, vibrational weight and diffusion parameter.
        target : str
            Label of the species used by the NCMAT Liquids section.

        Returns
        -------
        str
            Original NCMAT text with sample-specific translation settings.
        """
        if target not in self.reference_spectra:
            raise ValueError(f"No liquid dynamics for the sampled species {target!r}.")
        entries = tuple(
            replace(entry, wt=dynamics.translation_weight, ws=dynamics.continuous_weight,
                    c=dynamics.translation_parameter) if entry.atom == target else entry
            for entry in self.entries
        )
        return self.source if entries == self.entries else replace_custom_liquids(self.source, entries)

    def _prepare_vdos(self, material, spectra):
        """Replace specified VDOS blocks using the original grid spacing.

        Parameters
        ----------
        material : str
            NCMAT text with the current translation parameters.
        spectra : dict of str to PhononSpectra
            Spectra by species label, in eV and inverse eV.

        Returns
        -------
        tuple of str and dict
            Prepared NCMAT and per-species VDOS provenance.
        """
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
        return material, representation

    def _evaluate_material(self, sample_index, spectrum, material, target, representation,
                           energy_grid, capture_kernel):
        """Dispatch one prepared NCMAT to the native or captured analysis path.

        Parameters
        ----------
        sample_index : int
            Analyzer sample identity.
        spectrum : PhononSpectra
            Original spectrum for diagnostics.
        material, target : str
            Prepared NCMAT text and target species.
        representation : dict
            Per-species VDOS preparation details.
        energy_grid : array_like or None
            Incident energy grid in eV.
        capture_kernel : bool
            Whether differential observables require kernel capture.

        Returns
        -------
        LiquidAnalysisState
            Evaluated observables with sample provenance.
        """
        energies = np.asarray(np.geomspace(1e-5, 10, 1000) if energy_grid is None
                              else energy_grid, dtype=float)
        if (energies.ndim != 1 or not len(energies) or not np.isfinite(energies).all()
                or np.any(energies <= 0) or np.any(np.diff(energies) <= 0)):
            raise ValueError('Cross-section energies must be finite, positive and increasing')
        if not capture_kernel and self.active_species is None:
            return self._evaluate_native(sample_index, spectrum, material, target,
                                         representation, energies)
        # The capture path also supplies species-resolved scattering when requested.
        self.initialize_capture()
        return self._evaluate_captured(sample_index, spectrum, material, target,
                                        representation, energies)

    def _evaluate_native(self, sample_index, spectrum, material, target, representation, energies):
        """Evaluate integral observables with the native whole-material model.

        Parameters
        ----------
        sample_index : int
            Analyzer sample identity.
        spectrum : PhononSpectra
            Original spectrum for diagnostics.
        material, target : str
            Prepared NCMAT text and target species.
        representation : dict
            Per-species VDOS preparation details.
        energies : numpy.ndarray
            Incident energies in eV.

        Returns
        -------
        LiquidAnalysisState
            Native scattering and absorption in barns per atom.
        """
        import NCrystal as NC

        if not any(name == "Liquids" for name, _, _ in NC.plugins.browsePlugins()):
            raise RuntimeError("The NCrystal Liquids plugin must be active for liquid analysis.")
        with tempfile.TemporaryDirectory(prefix=f"liquid_{sample_index}_", dir=self.temp_dir) as tmp:
            path = Path(tmp) / "material.ncmat"
            path.write_text(material, encoding="utf-8")
            loaded = NC.load(f"{path};temp={self.temperature:.17g}K;vdoslux={self.vdoslux}")
            if "LiquidsModel" not in loaded.scatter.getName():
                raise RuntimeError("NCrystal did not select the Liquids model.")
            matches = [item for item in loaded.info.dyninfos
                       if item.atomData.displayLabel() == target and isinstance(item, NC.Info.DI_VDOS)]
            if len(matches) != 1:
                raise ValueError(f"Expected one liquid VDOS component for {target!r}.")
            atom = matches[0]
            scattering = loaded.scatter.xsect(energies)
            absorption = loaded.absorption.xsect(energies)
            backend_grid = np.asarray(atom.vdos_egrid_expanded).copy()
            backend_pdos = np.asarray(atom.vdos_density).copy()
            diagnostics = atom.analyseVDOS()
            mass = float(atom.atomData.averageMassAMU())
        for values in (scattering, absorption):
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("NCrystal returned nonphysical cross sections.")
        provenance = dict(sample_index=sample_index, material_ncmat=material,
                          vdos_representation=representation, capture=None,
                          xs_scope="whole material per atom", target=target)
        return LiquidAnalysisState(
            target_lbl=target, kernels={}, target_mass_amu=mass,
            energy_grid=spectrum.energy_grid, pdos=spectrum.density_values,
            backend_energy_grid=backend_grid, backend_pdos=backend_pdos,
            xs_energy=energies, xs_scattering=scattering, xs_absorption=absorption,
            vdos_analysis=diagnostics, provenance=provenance,
        )

    def _evaluate_captured(self, sample_index, spectrum, material, target, representation, energies):
        """Capture liquid kernels and cross sections in an isolated child.

        Parameters
        ----------
        sample_index : int
            Analyzer sample identity.
        spectrum : PhononSpectra
            Original spectrum for diagnostics.
        material, target : str
            Prepared NCMAT text and target species.
        representation : dict
            Per-species VDOS preparation details.
        energies : numpy.ndarray
            Incident energies in eV.

        Returns
        -------
        LiquidAnalysisState
            Captured kernels, cross sections, and provenance.
        """
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
        metadata['provenance']['adapter'] = self.identity
        metadata['provenance']['sample_index'] = sample_index
        metadata['provenance']['material_ncmat'] = material
        metadata['provenance']['vdos_representation'] = representation
        return LiquidAnalysisState(
            target_lbl=target, kernels=kernels,
            energy_grid=np.asarray(spectrum.energy_grid).copy(),
            pdos=np.asarray(spectrum.density_values).copy(),
            backend_energy_grid=arrays['backend_energy_grid'], backend_pdos=arrays['backend_pdos'],
            xs_energy=arrays['xs_energy'], xs_scattering=arrays['xs_scattering'],
            xs_absorption=arrays['xs_absorption'], vdos_analysis=metadata['vdos_analysis'],
            provenance=metadata['provenance'])

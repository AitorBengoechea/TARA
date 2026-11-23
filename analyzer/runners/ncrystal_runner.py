"""
Analyzer adapter for the shared NCrystal Processing backend.
"""

from typing import Collection, Dict, Optional, Union

from core.data.input_pdos import PhononSpectra
from processing.backends.ncrystal_processor import (
    MIN_NCRYSTAL_VDOS_ENERGY_EV,
    NcrystalProcessor,
    prepare_ncrystal_vdos,
    reconstruct_debye_tail,
)

from analyzer.backends.ncrystal_engine import NCrystalEngine
from processing.backends.ncrystal_liquid_analysis import (
    NCrystalLiquidAnalysisProcessor, is_liquid_template,
)
from analyzer.core_interfaces import BaseRunner


# Compatibility alias for code that imported the previous Analyzer-local helper.
# The implementation now belongs to Processing.
_prepare_ncrystal_vdos = prepare_ncrystal_vdos


class NCrystalRunner(BaseRunner):
    """Adapt Analyzer sample evaluation to ``NcrystalProcessor``.

    Parameters
    ----------
    base_ncmat_file : str
        Path to the base NCMAT structural template.
    temperature : float
        Material temperature in kelvin.
    temp_dir : str, optional
        Directory for transient NCMAT files.
    active_species : collection of str, optional
        NCMAT species to keep scattering-active. None preserves full material.
    """

    def __init__(
        self,
        base_ncmat_file: str,
        temperature: float,
        temp_dir: str = None,
        active_species: Optional[Collection[str]] = None,
        *, liquid_capture_library=None, vdoslux=3, liquid_timeout=600,
    ) -> None:
        """Configure a thin Analyzer adapter around the NCrystal processor.

        Parameters
        ----------
        base_ncmat_file : str
            Path to the NCMAT material template.
        temperature : float
            Material temperature in kelvin.
        temp_dir : str, optional
            Directory for temporary execution resources.
        active_species : collection of str, optional
            Active NCMAT element labels (all active if None).
        liquid_capture_library : path-like, optional
            Existing adapter for liquid analysis. Processing automatically
            builds a temporary adapter when this argument is omitted.
        vdoslux : int, optional
            Liquid kernel resolution (0..5), default 3.
        liquid_timeout : float, optional
            Per-sample liquid child timeout in seconds, default 600.
        """
        self.base_ncmat_file = base_ncmat_file
        self.temperature = temperature
        self.is_liquid = is_liquid_template(base_ncmat_file)
        if self.is_liquid:
            self.processor = NCrystalLiquidAnalysisProcessor(
                base_ncmat_file, temperature, capture_library=liquid_capture_library,
                temp_dir=temp_dir, active_species=active_species, vdoslux=vdoslux,
                timeout=liquid_timeout,
            )
        else:
            self.processor = NcrystalProcessor(
                base_ncmat_file=base_ncmat_file,
                temperature=temperature,
                temp_dir=temp_dir,
                active_species=active_species,
            )
        self.active_species = self.processor.active_species
        self.temp_dir = str(self.processor.temp_dir)

    def execute(
        self,
        sample_index: int,
        spectrum: Union[PhononSpectra, Dict[str, PhononSpectra]],
        **kwargs,
    ) -> NCrystalEngine:
        """Evaluate one Analyzer sample through the shared NCrystal backend.

        Parameters
        ----------
        sample_index : int
            Stochastic sample identifier.
        spectrum : PhononSpectra or dict of str to PhononSpectra
            Spectrum or element-labelled spectra to evaluate.
        **kwargs
            Analyzer backend options. ``atom_label`` identifies the target
            element for monoatomic input or selects the primary target for a
            multi-element dictionary.

        Returns
        -------
        NCrystalEngine
            Analyzer-owned in-memory state container.
        """
        if self.is_liquid:
            state = self.processor.create_analysis_state(
                sample_index, spectrum, atom_label=kwargs.get("atom_label"),
                energy_grid=kwargs.get("energy_grid"),
            )
            return NCrystalEngine(
                filepath="", temperature=self.temperature, target_lbl=state.target_lbl,
                mat=None, nuc_data=None, mass_amu=state.mass_amu,
                energy_grid=state.energy_grid, pdos=state.pdos,
                backend_energy_grid=state.backend_energy_grid, backend_pdos=state.backend_pdos,
                liquid_data=state,
            )
        # Ask Processing to prepare the transient material before constructing the in-memory
        # Analyzer state.
        state = self.processor.create_analysis_state(
            sample_index=sample_index,
            spectrum=spectrum,
            atom_label=kwargs.get("atom_label"),
            load_scattering_components=True,
        )

        return NCrystalEngine(
            filepath=state.filepath,
            temperature=self.temperature,
            target_lbl=state.target_lbl,
            mat=state.mat,
            nuc_data=state.nuc_data,
            mass_amu=state.mass_amu,
            energy_grid=state.energy_grid,
            pdos=state.pdos,
            backend_energy_grid=state.backend_energy_grid,
            backend_pdos=state.backend_pdos,
            scatter_inelastic=state.scatter_inelastic,
            scatter_coherent_elastic=state.scatter_coherent_elastic,
            scatter_incoherent_elastic=state.scatter_incoherent_elastic,
        )

    def cleanup(self, runner_output: NCrystalEngine) -> None:
        """Delegate transient NCMAT cleanup to the Processing backend.

        Parameters
        ----------
        runner_output : NCrystalEngine
            Backend state returned by the runner for a single evaluation.
        """
        if runner_output is None or self.is_liquid:
            return
        # Release the temporary NCMAT through its owning Processing backend.
        self.processor.cleanup_analysis_file(runner_output.filepath)

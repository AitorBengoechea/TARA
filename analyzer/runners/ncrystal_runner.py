"""
Analyzer adapter for the shared NCrystal Processing backend.
"""

from dataclasses import dataclass
from typing import Collection, Dict, Optional, Union

from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import MaterialScatteringModel
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
    """Dispatch vibrational or material samples to NCrystal Processing.

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
            builds a temporary adapter when a kernel is requested and this argument is omitted.
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
                timeout=liquid_timeout, defer_capture=True,
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

    def prepare_analysis(self, runner_kwargs, parser_kwargs):
        """Resolve kernel capture needs before the worker processes start.

        Parameters
        ----------
        runner_kwargs : dict
            Backend execution options.
        parser_kwargs : dict
            Requested observables.

        Returns
        -------
        dict
            Execution options including capture mode for liquid inputs.
        """
        if not self.is_liquid:
            return runner_kwargs
        kernel_flags = (
            "calc_sab", "calc_dsf", "calc_scattering_kernel", "calc_angular_distribution",
            "calc_cosine_distribution", "calc_energy_distribution", "calc_ratios",
        )
        needs_kernel = any(parser_kwargs.get(flag, False) for flag in kernel_flags)
        capture = runner_kwargs.get("capture_kernel", needs_kernel)
        if needs_kernel and not capture:
            raise ValueError("The requested liquid observables require kernel capture.")
        if capture or self.active_species is not None:
            self.processor.initialize_capture()
        options = {**runner_kwargs, "capture_kernel": capture}
        if "energy_grid" in parser_kwargs:
            options.setdefault("energy_grid", parser_kwargs["energy_grid"])
        return options

    def execute(
        self,
        sample_index: int,
        spectrum: Union[PhononSpectra, Dict[str, PhononSpectra], MaterialScatteringModel],
        **kwargs,
    ) -> NCrystalEngine:
        """Evaluate one Analyzer sample through the shared NCrystal backend.

        Parameters
        ----------
        sample_index : int
            Stochastic sample identifier.
        spectrum : PhononSpectra, dict, or MaterialScatteringModel
            Spectrum, species-labelled spectra, or a complete liquid realization.
        **kwargs
            Analyzer backend options. ``atom_label`` identifies the target
            element for monoatomic input or selects the primary target for a
            multi-element dictionary.

        Returns
        -------
        NCrystalEngine
            Analyzer-owned in-memory state container.
        """
        if isinstance(spectrum, MaterialScatteringModel):
            if not self.is_liquid:
                raise ValueError("Material analysis requires an NCrystal Liquids template.")
            runner = _NCrystalTranslationRunner(self.processor, self.temperature)
        else:
            runner = _NCrystalVibrationRunner(self.processor, self.temperature, self.is_liquid)
        return runner.execute(sample_index, spectrum, **kwargs)

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


@dataclass(frozen=True)
class _NCrystalVibrationRunner:
    """Delegate spectrum analysis and adapt the resulting Processing state.

    Parameters
    ----------
    processor : NcrystalProcessor or NCrystalLiquidAnalysisProcessor
        Shared backend configured for the source template.
    temperature : float
        Material temperature in kelvin.
    is_liquid : bool
        Whether the template uses the Liquids plugin.
    """

    processor: NcrystalProcessor | NCrystalLiquidAnalysisProcessor
    temperature: float
    is_liquid: bool

    def execute(self, sample_index, spectrum, **kwargs):
        """Evaluate a vibrational input and return standard Analyzer state.

        Parameters
        ----------
        sample_index : int
            Sample identity, or -1 for the independent reference.
        spectrum : PhononSpectra or dict of str to PhononSpectra
            Vibrational data for the selected species.
        **kwargs : dict
            Atom label, energy grid in eV, and optional kernel capture flag.

        Returns
        -------
        NCrystalEngine
            Evaluated backend state.
        """
        if self.is_liquid:
            options = {"atom_label": kwargs.get("atom_label"),
                       "energy_grid": kwargs.get("energy_grid")}
            if "capture_kernel" in kwargs:
                options["capture_kernel"] = kwargs["capture_kernel"]
            state = self.processor.create_analysis_state(sample_index, spectrum, **options)
            return _liquid_engine(state, self.temperature)
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



def _liquid_engine(state, temperature):
    """Adapt the native liquid state for the existing NCrystal parser.

    Parameters
    ----------
    state : LiquidAnalysisState
        Native cross sections and optional scattering kernels.
    temperature : float
        Configured temperature in kelvin.

    Returns
    -------
    NCrystalEngine
        Standard Analyzer state.
    """
    return NCrystalEngine(
        filepath="", temperature=temperature, target_lbl=state.target_lbl,
        mat=None, nuc_data=None, mass_amu=state.mass_amu,
        energy_grid=state.energy_grid, pdos=state.pdos,
        backend_energy_grid=state.backend_energy_grid, backend_pdos=state.backend_pdos,
        liquid_data=state,
    )


@dataclass(frozen=True)
class _NCrystalTranslationRunner:
    """Delegate translation-only or joint liquid material realizations.

    Parameters
    ----------
    processor : NCrystalLiquidAnalysisProcessor
        Native liquid backend shared by the public runner.
    temperature : float
        Material temperature in kelvin.
    """

    processor: NCrystalLiquidAnalysisProcessor
    temperature: float

    def execute(self, sample_index, material, **kwargs):
        """Evaluate the sampled material as a single native NCrystal input.

        Parameters
        ----------
        sample_index : int
            Sample identity, or -1 for the independent reference.
        material : MaterialScatteringModel
            Sampled translation settings and vibrational spectrum.
        **kwargs : dict
            Atom label, energy grid in eV, and optional kernel capture flag.

        Returns
        -------
        NCrystalEngine
            Evaluated backend state.
        """
        state = self.processor.create_material_analysis_state(
            sample_index, material, atom_label=kwargs.get("atom_label"),
            energy_grid=kwargs.get("energy_grid"), capture_kernel=kwargs.get("capture_kernel", False),
        )
        return _liquid_engine(state, self.temperature)

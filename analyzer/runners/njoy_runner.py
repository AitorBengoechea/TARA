"""Analyzer adapter for the shared NJOY Processing backend."""

from endf_parserpy import EndfParserPy

from analyzer.backends.njoy_engine import NjoyEngine
from analyzer.core_interfaces import BaseRunner
from core.data.input_pdos import PhononSpectra
from processing.backends.njoy_processor import NjoyProcessor


class NjoyRunner(BaseRunner):
    """Adapt Analyzer requests to the reusable NJOY Processing backend."""

    def __init__(
        self,
        njoy_exec: str,
        base_endf_file: str,
        base_leapr_file: str,
        temperature: float,
        mass_amu: float,
        temp_dir: str = None,
    ):
        """Initialize the thin NJOY Analyzer adapter.

        Parameters
        ----------
        njoy_exec : str
            Path to the NJOY executable.
        base_endf_file : str
            Path to the unmodified base ENDF evaluation.
        base_leapr_file : str
            Path to the validated LEAPR input template.
        temperature : float
            Material temperature in kelvin.
        mass_amu : float
            Target atomic mass in atomic mass units.
        temp_dir : str, optional
            Directory for temporary execution resources.
        """
        self.temperature = float(temperature)
        self.mass_amu = float(mass_amu)
        self.processor = NjoyProcessor(
            njoy_exec=njoy_exec,
            temp_dir=temp_dir,
            base_endf_file=base_endf_file,
            base_leapr_file=base_leapr_file,
        )

        self.base_endf_file = self.processor.base_endf_file
        self.base_leapr_file = self.processor.base_leapr_file
        self.njoy_exec = self.processor.njoy_exec
        self.temp_dir = str(self.processor.temp_dir)
        self.endf_parser = EndfParserPy()

    def execute(
        self,
        sample_index: int,
        spectrum: PhononSpectra,
        **kwargs,
    ) -> NjoyEngine:
        """Evaluate one spectrum through the shared NJOY Processing backend.

        Parameters
        ----------
        sample_index : int
            Identifier of the stochastic sample being evaluated.
        spectrum : PhononSpectra
            Source phonon spectrum to evaluate or perturb.
        **kwargs : Any
            THERMR options and the optional LEAPR `export_full_spectra` flag.

        Returns
        -------
        NjoyEngine
            Calculated result in the declared return type.
        """
        del sample_index  # Execution identity is owned by Processing.

        # Processing owns NJOY deck execution and the lifetime of its temporary tapes.
        with self.processor.execute_analysis(
            spectrum,
            temperature=self.temperature,
            thermr_emax=kwargs.get("thermr_emax", 5.0),
            thermr_tol=kwargs.get("thermr_tol", 0.05),
            thermr_nbin=kwargs.get("thermr_nbin", 16),
            export_full_spectra=kwargs.get("export_full_spectra", False),
        ) as execution:
            leapr_tree = self.endf_parser.parsefile(
                str(execution.tape_path("tape24"))
            )

            thermr_tree = None
            if execution.has_tape("tape26"):
                thermr_tree = self.endf_parser.parsefile(
                    str(execution.tape_path("tape26")),
                    include=[3],
                )

            stdout_text = execution.stdout_text

        return NjoyEngine(
            base_endf_file=self.base_endf_file,
            temperature=self.temperature,
            mass_amu=self.mass_amu,
            energy_grid=spectrum.energy_grid,
            pdos=spectrum.density_values,
            leapr_tree=leapr_tree,
            thermr_tree=thermr_tree,
            njoy_output_text=stdout_text,
        )

    def cleanup(self, runner_output: NjoyEngine) -> None:
        """Release Analyzer-owned resources; Processing already cleaned its sandbox.

        Parameters
        ----------
        runner_output : NjoyEngine
            Backend state returned by the runner for a single evaluation.
        """
        del runner_output

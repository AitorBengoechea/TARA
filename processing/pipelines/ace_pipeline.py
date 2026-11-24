"""
Universal pipeline for Phase 2 ACE generation.
Acts as the convergence point for all ENDF MF7 tapes.
"""
import logging
from pathlib import Path
from typing import Dict

from processing.core.config import AceGenerationConfig
from processing.orchestrators.njoy_batch import NjoyBatchOrchestrator

logger = logging.getLogger(__name__)


class UniversalAcePipeline:
    """
    Takes valid ENDF tapes containing S(alpha, beta) data (MF=7) and processes 
    them into final ACE libraries for Monte Carlo transport.
    """

    def __init__(self, njoy_exec: str, n_jobs: int = -1):
        """
        Parameters
        ----------
        njoy_exec : str
            Path to the NJOY executable.
        n_jobs : int
            Requested parallel workers. ``-1`` uses the shared HPC-aware
            worker policy with two reserved cores by default.
        """
        self.orchestrator = NjoyBatchOrchestrator(njoy_exec=njoy_exec, n_jobs=n_jobs)

    def run(
        self, 
        sab_tapes: Dict[str, str], 
        output_dir: str, 
        config: AceGenerationConfig,
        *,
        save_xsdir: bool = True,
        save_njoy_input: bool = False,
    ) -> Dict[str, str]:
        """
        Executes the ACE generation pipeline across an ensemble of ENDF tapes.

        Parameters
        ----------
        sab_tapes : Dict[str, str]
            Dictionary mapping sample IDs to their MF7 ENDF file paths.
        output_dir : str
            Target directory for the generated .ace files.
        config : AceGenerationConfig
            Strictly validated configuration parameters for the NJOY sequence.
        save_xsdir : bool, optional
            Save one ``.xsdir`` entry per sample in ``output_dir/xsdir``.
            Defaults to ``True``.
        save_njoy_input : bool, optional
            Save the shared NJOY deck as ``output_dir/njoy_ace.inp``.
            Stage the base ENDF as ``tape20`` and each MF7 ENDF as ``tape70``
            to reuse it in a separate NJOY working directory.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping successful sample IDs to their generated ACE file paths.
        """
        if not sab_tapes:
            logger.warning("No ENDF tapes provided to the ACE pipeline. Aborting.")
            return {}

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initiating Universal ACE Pipeline for {len(sab_tapes)} samples...")

        # Convert the strictly typed dataclass into a dictionary for the orchestrator
        config_dict = config.__dict__

        # Trigger the Phase 2 sequence in the orchestrator
        successful_aces = self.orchestrator.process_ace_ensemble(
            sab_endf_files=sab_tapes,
            output_dir=str(out_path),
            config=config_dict,
            save_xsdir=save_xsdir,
            **({"save_njoy_input": True} if save_njoy_input else {}),
        )

        success_rate = (len(successful_aces) / len(sab_tapes)) * 100
        logger.info(
            f"Universal ACE Pipeline finished. "
            f"Success rate: {success_rate:.1f}% ({len(successful_aces)}/{len(sab_tapes)})."
        )

        return successful_aces


    def run_from_directory(
        self,
        *,
        base_endf_path: str,
        temperature: float,
        tsl_directory: str,
        output_dir: str | None = None,
        save_xsdir: bool = True,
        save_njoy_input: bool = False,
    ) -> Dict[str, str]:
        """Generate ACE libraries from compatible TSL ENDF files.

        ``tsl_directory`` may also point to a single ``.endf`` file.  The
        default ACE destination is a sibling ``ace/`` directory next to the
        TSL input directory.  Each input filename supplies its output ID;
        ``sample_000_sab.endf`` becomes ``sample_000.ace``.

        ENDF parsing and consistency checks complete *before* any NJOY job
        starts. Unlike the backwards-compatible ``run`` method, this entry
        point raises if an ACE conversion is missing or produces no file.

        Parameters
        ----------
        base_endf_path : str
            Base neutron ENDF evaluation.
        temperature : float
            Target processing temperature in kelvin.
        tsl_directory : str
            Directory of TSL ENDF tapes or path to one TSL ENDF file.
        output_dir : str or None, optional
            ACE output directory; defaults to the sibling ``ace`` directory.
        save_xsdir : bool, optional
            Store XSDIR entries in an ``xsdir`` subdirectory, default ``True``.
        save_njoy_input : bool, optional
            Save the shared NJOY deck as ``njoy_ace.inp`` in the ACE directory.

        Returns
        -------
        dict[str, str]
            Generated sample IDs mapped to ACE paths, not XSDIR paths.

        Raises
        ------
        RuntimeError
            If any sample is missing or an ACE file is absent or empty.
        """
        sab_tapes, input_dir = self._discover_tsl_tapes(Path(tsl_directory))

        config = AceGenerationConfig.from_endf(
            base_endf_path=base_endf_path,
            tsl_endf_files=sab_tapes,
            temperature=temperature,
        )
        destination = (
            Path(output_dir) if output_dir is not None else input_dir.parent / "ace"
        )
        results = self.run(
            sab_tapes=sab_tapes,
            output_dir=str(destination),
            config=config,
            save_xsdir=save_xsdir,
            **({"save_njoy_input": True} if save_njoy_input else {}),
        )

        self._validate_complete_batch(sab_tapes, results)
        return results

    @staticmethod
    def _discover_tsl_tapes(source: Path) -> tuple[Dict[str, str], Path]:
        """Resolve TSL paths and deduplicate sample identifiers.

        Parameters
        ----------
        source : pathlib.Path
            Directory of ENDF tapes or a single TSL ENDF file.

        Returns
        -------
        dict[str, str]
            Ordered sample names and ENDF paths.
        pathlib.Path
            Parent directory of the input tapes.

        Raises
        ------
        FileNotFoundError
            If the input path does not exist.
        ValueError
            If files have invalid extensions or ambiguous sample names.
        """
        if source.is_file():
            if source.suffix.lower() != ".endf":
                raise ValueError(f"Expected an .endf TSL input, got: {source}")
            tsl_paths = [source]
            input_dir = source.parent
        elif source.is_dir():
            tsl_paths = sorted(
                path for path in source.iterdir()
                if path.is_file() and path.suffix.lower() == ".endf"
            )
            input_dir = source
        else:
            raise FileNotFoundError(f"TSL ENDF file or directory not found: {source}")

        if not tsl_paths:
            raise ValueError(f"No .endf files found in {source}.")

        sab_tapes = UniversalAcePipeline._index_tsl_paths(tsl_paths)
        return sab_tapes, input_dir

    @staticmethod
    def _index_tsl_paths(tsl_paths):
        """Index ENDF paths by normalized sample identifier.

        Parameters
        ----------
        tsl_paths : sequence of pathlib.Path
            Existing ENDF paths in processing order.

        Returns
        -------
        dict of str to str
            Unique sample IDs and corresponding ENDF path strings.
        """
        sab_tapes: Dict[str, str] = {}
        for path in tsl_paths:
            sample_id = path.stem
            if sample_id.endswith("_sab"):
                sample_id = sample_id[:-4]
            if not sample_id:
                raise ValueError(f"Cannot obtain a sample ID from: {path.name}")
            if sample_id in sab_tapes:
                raise ValueError(
                    f"Duplicate sample ID {sample_id!r} from TSL ENDF filenames."
                )
            sab_tapes[sample_id] = str(path)
        return sab_tapes

    @staticmethod
    def _validate_complete_batch(
        sab_tapes: Dict[str, str], results: Dict[str, str]
    ) -> None:
        """Reject missing, unexpected, or empty ACE outputs.

        Parameters
        ----------
        sab_tapes : dict[str, str]
            Expected inputs keyed by sample ID.
        results : dict[str, str]
            Actual ACE output paths.

        Raises
        ------
        RuntimeError
            If output files are absent, empty, or have inconsistent IDs.
        """
        missing = set(sab_tapes) - set(results)
        extra = set(results) - set(sab_tapes)
        invalid = [
            sample_id for sample_id, path in results.items()
            if not Path(path).is_file() or Path(path).stat().st_size == 0
        ]
        if missing or extra or invalid:
            raise RuntimeError(
                "Incomplete NJOY ACE batch: "
                f"missing={sorted(missing)}, unexpected={sorted(extra)}, "
                f"missing_or_empty_files={sorted(invalid)}."
            )

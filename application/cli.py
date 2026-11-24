"""
Command Line Interface (CLI) for the TARA Stochastic Sampling Engine.
"""
import argparse
import subprocess
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

# Allow the historical `python application/cli.py` entry point as well as
# `python -m application.cli` when invoked from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def update_docs():
    """Regenerate the project map and test overview.

    Raises
    ------
    SystemExit
        If either documentation command fails.
    """
    try:
        subprocess.run([sys.executable, "generate_ai_context.py"], check=True)
        subprocess.run([sys.executable, "tests/generate_test_overview.py"], check=True)
        print("Successfully regenerated PROJECT_MAP.md and tests/test_overview.md")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate documentation: {e}", file=sys.stderr)
        sys.exit(1)


def parse_energy_ranges(range_strings: List[str]) -> List[Tuple[float, float]]:
    """Parse CLI arguments into pairs of energy bounds.

    Parameters
    ----------
    range_strings : list of str
        Energy intervals encoded as comma-separated strings.

    Returns
    -------
    list of tuple of float
        Lower and upper energy bounds, in input order.

    Raises
    ------
    argparse.ArgumentTypeError
        If a range does not contain two comma-separated numbers.
    """
    parsed_ranges = []
    for r in range_strings:
        try:
            start_str, end_str = r.split(',')
            parsed_ranges.append((float(start_str), float(end_str)))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid range format: '{r}'. Must be formatted as 'start,end'."
            )
    return parsed_ranges


def main(argv: List[str] | None = None) -> None:
    """Route the ACE subcommand or run the legacy sampling CLI.

    Parameters
    ----------
    argv : list of str, optional
        Explicit CLI arguments for testing; defaults to ``sys.argv[1:]``.
    """
    supplied = list(sys.argv[1:] if argv is None else argv)
    if supplied and supplied[0] == "ace":
        _run_ace_command(supplied[1:])
        return
    _run_sampling_command(supplied)


def _run_ace_command(argv: list[str]) -> None:
    """Run or validate an ACE workflow from its TOML file.

    Parameters
    ----------
    argv : list of str
        Arguments following the ACE subcommand.

    Raises
    ------
    SystemExit
        If argument parsing, input validation, or workflow execution fails.
    """
    ace_parser = argparse.ArgumentParser(
        prog="tara ace",
        description="Generate reference and stochastic ACE libraries from TOML."
    )
    ace_parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to an ACE TOML file, e.g. application/configs/ace_njoy_solid.toml."
    )
    ace_parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate the TOML and input paths without running NJOY or NCrystal."
    )
    options = ace_parser.parse_args(argv)
    from application.configuration import load_ace_config, ApplicationConfigError
    from application.ace_workflow import run_ace_workflow, validate_ace_inputs
    try:
        config = load_ace_config(options.config)
        if options.validate_only:
            validate_ace_inputs(config)
            print(f"Valid {config.backend} ACE configuration: {config.file_path}")
            return
        products = run_ace_workflow(config)
    except (ApplicationConfigError, FileNotFoundError, ValueError,
            RuntimeError, TypeError) as exc:
        ace_parser.error(str(exc))
    print(f"ACE generation complete: {len(products)} tables")


def _run_sampling_command(argv: list[str]) -> None:
    """Run the legacy sampling command and report its result.

    Parameters
    ----------
    argv : list of str
        Sampling arguments, including the source filename.

    Raises
    ------
    SystemExit
        If argument parsing fails, the source is missing, or the sampling
        workflow reports a processing error.
    """
    from core.config import TaraConfigError
    from core.data.input_pdos import PhononSpectraError

    parser = _sampling_parser()
    args = parser.parse_args(argv)
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"Error: File '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    manual_ranges = parse_energy_ranges(args.manual_ranges) if args.partition == "manual" else None
    try:
        config = _sampling_config(args, manual_ranges)
        base_obj, ensemble = _generate_spectra(input_path, config, preview=args.preview)
        _export_samples(base_obj, ensemble, args)
        print("\n=== Generation Complete ===")
        print(f"Total valid samples generated: {len(ensemble)}")
    except (ValueError, argparse.ArgumentTypeError, TaraConfigError, PhononSpectraError) as exc:
        print(f"\nProcessing Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _sampling_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the legacy sampling command.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for spectrum sampling, previews, and export.
    """
    parser = argparse.ArgumentParser(
        description="TARA: Stochastic Sampling Engine for Phonon Spectra."
    )
    _add_file_arguments(parser)
    _add_perturbation_arguments(parser)
    _add_strategy_arguments(parser)
    return parser


def _add_file_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the input file and export options.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Sampling command parser to extend.
    """
    group = parser.add_argument_group("Input and export")
    group.add_argument(
        "input_file",
        type=str,
        help="Path to input file (LEAPR, .ncmat, CSV).",
    )
    group.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./tara_output",
        help="Destination directory for generated files.",
    )
    group.add_argument(
        "--prefix",
        type=str,
        default="tara_sample",
        help="Filename prefix for generated files.",
    )


def _add_perturbation_arguments(parser: argparse.ArgumentParser) -> None:
    """Register perturbation bounds, sample count, and preview mode.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Sampling command parser to extend.
    """
    group = parser.add_argument_group("Perturbations and sample count")
    group.add_argument(
        "-n", "--n-samples",
        type=int,
        default=0,
        help="Number of stochastic samples. May be omitted in preview mode.",
    )
    group.add_argument(
        "--shift-mev",
        type=float,
        required=True,
        help="Maximum absolute peak shift in meV.",
    )
    group.add_argument(
        "--shift-pct",
        type=float,
        required=True,
        help="Maximum relative amplitude shift in percent.",
    )
    group.add_argument(
        "--preview",
        action="store_true",
        help="Generate two or four deterministic extrema.",
    )


def _add_strategy_arguments(parser: argparse.ArgumentParser) -> None:
    """Register partitioning, coupling, and grid controls.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Sampling command parser to extend.
    """
    group = parser.add_argument_group("Sampling strategies")
    group.add_argument(
        "--partition",
        choices=("automatic", "manual"),
        default="automatic",
    )
    group.add_argument(
        "--automatic-partition-prominence-pct",
        type=float,
        default=1.0,
        help=(
            "Minimum valley prominence as a percentage of the maximum PDOS "
            "density for automatic partitioning (default: 1.0)."
        ),
    )
    group.add_argument("--manual-ranges", nargs="+", type=str)
    group.add_argument(
        "--topology",
        choices=("fixed", "deterministic", "random"),
        default="fixed",
    )
    group.add_argument(
        "--correlation",
        choices=("independent", "random_group"),
        default="independent",
    )
    group.add_argument(
        "--coupling",
        choices=("independent", "coupled"),
        default="independent",
        help="Control coupling between amplitude and spatial shifts within domains.",
    )
    group.add_argument("--unfreeze-origin", action="store_true")
    group.add_argument("--decouple-grids", action="store_true")


def _sampling_config(args: argparse.Namespace, manual_ranges: list[tuple[float, float]] | None):
    """Build the Core sampling configuration and report validation warnings.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed sampling command options.
    manual_ranges : list of tuple of float or None
        Parsed energy intervals for manual partitioning.

    Returns
    -------
    TaraConfig
        Validated sampling options.
    """
    from core.config import TaraConfig

    # Keep configuration warnings visible without interrupting previews.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        config = TaraConfig(
            n_samples=args.n_samples,
            max_energy_shift_mev=args.shift_mev,
            max_amplitude_shift_pct=args.shift_pct,
            grid_partition_method=args.partition,
            automatic_partition_prominence_pct=args.automatic_partition_prominence_pct,
            manual_energy_ranges=manual_ranges,
            domain_splitting_strategy=args.topology,
            correlation_strategy=args.correlation,
            variable_coupling=args.coupling,
            shared_grid_structure=not args.decouple_grids,
            freeze_origin=not args.unfreeze_origin,
        )

        for warning in captured:
            print(f"Configuration Warning: {warning.message}", file=sys.stderr)
    return config


def _generate_spectra(input_path: Path, config, *, preview: bool):
    """Load one spectrum and generate a preview or stochastic ensemble.

    Parameters
    ----------
    input_path : pathlib.Path
        Source LEAPR, NCMAT, or CSV file.
    config : TaraConfig
        Validated Core sampling options.
    preview : bool
        Generate deterministic extrema when True.

    Returns
    -------
    tuple of object or None and SpectraEnsemble
        File-format template, if available, followed by the generated spectra.
    """
    from core.data.input_pdos import PhononSpectra
    from core.sampler.concrete_samplers import TaraSampler
    from input_output.factory import load_file_object

    print(f"Loading and detecting format for: {input_path.name}")
    # Detect and read the external input using the I/O translation layer.
    base_obj, raw_pdos = load_file_object(input_path)

    print("Standardizing and enforcing low-energy constraints...")
    # Normalize input data through the public Core spectrum model.
    spectra = PhononSpectra(raw_pdos)

    # Delegate all numerical ensemble construction to the Core sampler.
    sampler = TaraSampler(config=config, spectra=spectra)

    if preview:
        print("Generating deterministic preview extrema...")
        ensemble = sampler.generate_preview()
    else:
        print(f"Initializing TARA sampler for {config.n_samples} samples...")
        ensemble = sampler.generate_ensemble()
    return base_obj, ensemble


def _export_samples(base_obj, ensemble, args: argparse.Namespace) -> None:
    """Export spectra when the input supplies a material template.

    Parameters
    ----------
    base_obj : object or None
        File-format template, absent for CSV input.
    ensemble : SpectraEnsemble
        Generated spectra to serialize.
    args : argparse.Namespace
        Parsed output directory and filename prefix.
    """
    from input_output.exporter import export_ensemble

    if base_obj is not None:
        print(f"Exporting ensemble to {args.output_dir}/...")
        # Serialize generated samples via the I/O layer; the CLI does not implement file
        # formats.
        export_ensemble(base_obj, ensemble, args.output_dir, args.prefix)
    else:
        print("Warning: CSV inputs lack metadata templates. Exporting disabled.", file=sys.stderr)


if __name__ == "__main__":
    main()

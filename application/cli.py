"""
Command Line Interface (CLI) for the TARA Stochastic Sampling Engine.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Allow the historical `python application/cli.py` entry point as well as
# `python -m application.cli` when invoked from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def update_docs():
    """Regenerates the dynamically created markdown files.
    """
    try:
        subprocess.run([sys.executable, "generate_ai_context.py"], check=True)
        subprocess.run([sys.executable, "tests/generate_test_overview.py"], check=True)
        print("Successfully regenerated PROJECT_MAP.md and tests/test_overview.md")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate documentation: {e}", file=sys.stderr)
        sys.exit(1)


def parse_energy_ranges(range_strings: List[str]) -> List[Tuple[float, float]]:
    """Parses CLI string arguments into explicit tuple energy ranges.

    Parameters
    ----------
    range_strings : List[str]
        Energy intervals encoded as comma-separated strings.

    Returns
    -------
    List[Tuple[float, float]]
        Parsed list of (lower, upper) energy bounds.
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
        options = ace_parser.parse_args(supplied[1:])
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
        return

    # Legacy sampling CLI remains available with its existing positional file
    # argument and flags; Core dependencies are imported only for that route.
    from input_output.factory import load_file_object
    from input_output.exporter import export_ensemble
    from core.data.input_pdos import PhononSpectra, PhononSpectraError
    from core.config import TaraConfig, TaraConfigError
    from core.sampler.concrete_samplers import TaraSampler
    """Parse CLI arguments and orchestrate the requested TARA sampling workflow.
    """
    parser = argparse.ArgumentParser(
        description="TARA: Stochastic Sampling Engine for Phonon Spectra."
    )
    
    parser.add_argument("input_file", type=str, help="Path to input file (LEAPR, .ncmat, CSV).")
    
    # --- Output Parameters ---
    parser.add_argument("-o", "--output-dir", type=str, default="./tara_output",
                        help="Destination directory for generated files.")
    parser.add_argument("--prefix", type=str, default="tara_sample",
                        help="Filename prefix for generated files.")

    # --- Mandatory Parameters (Except n-samples if just previewing) ---
    parser.add_argument("-n", "--n-samples", type=int, default=0,
                        help="The total requested sample ensemble size (optional if only generating preview).")
    parser.add_argument("--shift-mev", type=float, required=True,
                        help="Peak shifting boundary (absolute meV).")
    parser.add_argument("--shift-pct", type=float, required=True,
                        help="Amplitude shifting boundary (relative %%).")

    # --- Algorithmic Overrides ---
    parser.add_argument("--partition", type=str, choices=["automatic", "manual"], default="automatic")
    parser.add_argument(
        "--automatic-partition-prominence-pct", type=float, default=1.0,
        help=("Minimum valley prominence as a percentage of the maximum PDOS "
              "density for automatic partitioning (default: 1.0)."),
    )
    parser.add_argument("--manual-ranges", nargs="+", type=str)
    parser.add_argument("--topology", type=str, choices=["fixed", "deterministic", "random"], default="fixed")
    parser.add_argument("--correlation", type=str, choices=["independent", "random_group"], default="independent")
    parser.add_argument("--coupling", type=str, choices=["independent", "coupled"], default="independent",
                        help="Dictates coupling between amplitude and spatial parameters within domains.")
    parser.add_argument("--preview", action="store_true", 
                        help="Generate exactly 2 or 4 preview extrema instead of a stochastic ensemble.")
    parser.add_argument("--unfreeze-origin", action="store_true")
    parser.add_argument("--decouple-grids", action="store_true")

    # Resolve the command-line options before validating filesystem inputs and sampling
    # configuration.
    args = parser.parse_args(supplied)
    input_path = Path(args.input_file)

    if not input_path.is_file():
        print(f"Error: File '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    manual_ranges = parse_energy_ranges(args.manual_ranges) if args.partition == "manual" else None

    # Catch warnings gracefully in CLI so it doesn't look like a crash
    import warnings
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Validate numerical bounds and sampling choices through the Core configuration
            # gateway.
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
                freeze_origin=not args.unfreeze_origin
            )
            
            for warning in w:
                print(f"Configuration Warning: {warning.message}", file=sys.stderr)

        print(f"Loading and detecting format for: {input_path.name}")
        # Detect and read the external input using the I/O translation layer.
        base_obj, raw_pdos = load_file_object(input_path)

        print("Standardizing and enforcing low-energy constraints...")
        # Normalize input data through the public Core spectrum model.
        spectra = PhononSpectra(raw_pdos)

        # Delegate all numerical ensemble construction to the Core sampler.
        sampler = TaraSampler(config=config, spectra=spectra)

        if args.preview:
            print("Generating deterministic preview extrema...")
            ensemble = sampler.generate_preview()
        else:
            print(f"Initializing TARA sampler for {config.n_samples} samples...")
            ensemble = sampler.generate_ensemble()

        # Serialization
        if base_obj is not None:
            print(f"Exporting ensemble to {args.output_dir}/...")
            # Serialize generated samples via the I/O layer; the CLI does not implement file
            # formats.
            export_ensemble(base_obj, ensemble, args.output_dir, args.prefix)
        else:
            print("Warning: CSV inputs lack metadata templates. Exporting disabled.", file=sys.stderr)

        print("\n=== Generation Complete ===")
        print(f"Total valid samples generated: {len(ensemble)}")

    except (ValueError, argparse.ArgumentTypeError, TaraConfigError, PhononSpectraError) as e:
        print(f"\nProcessing Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

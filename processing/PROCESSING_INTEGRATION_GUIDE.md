# Processing Architecture: Processing Domain Integration Guide

## 1. Architectural Philosophy

The `processing` domain owns reusable external-backend execution infrastructure
and the persistent workflows built on top of it.

The optional `analyzer` domain may compose `processing.backends` and
`processing.deck_builders` for transient evaluations, but it must not depend on
Processing orchestrators or pipelines. Analyzer owns sample identity,
parallelism, aggregation, and physics extraction; Processing owns reusable
backend mechanics.

## 2. Shared Backend Boundary

`processing.backends` is the single source of truth for backend-specific
execution and transient resource management.

### NJOY

`NjoyProcessor` owns NJOY subprocess execution and sandbox lifecycle. NJOY deck
syntax belongs to `processing.deck_builders`.

### NCrystal

`NcrystalProcessor` owns:

* base NCMAT composer initialization and cloning;
* NCrystal-compatible VDOS preparation;
* temporary NCMAT creation;
* NCrystal material and `NuclearData` loading;
* transient NCMAT cleanup; and
* persistent `ncmat2endf` ENDF generation.

The solid processor supports both temporary Analyzer evaluation and
persistent MF7 generation. The liquid capture backend has its own resource
lifecycle.


## NCrystal Analyzer two-grid contract (VDOS output vs internal analysis)

For transient solid-material analysis, `NcrystalProcessor.create_analysis_state`
is expected to provide two representations of the selected atom’s PDOS:

- `energy_grid` / `pdos`: full Core sampling arrays, including the dense Debye
  reconstruction down to zero, used by Analyzer thermodynamics.
- `backend_energy_grid` / `backend_pdos`: precisely the arrays injected into the
  generated NCMAT, following the selected atom's original source spacing and
  any valid high-energy extension. Used by NCrystal's `extractKnl`/`extractGn`.

Do not send the full internal `E=0` grid to NCrystal kernel extraction; that
API requires its first VDOS energy >=1e-5 eV. Processing is responsible for providing the arrays that the serializer
actually writes. The [transient-state regression](../tests/test_processing/test_ncrystal_processor.py)
checks both representations. It still detects a mismatch in the available
source snapshot. See the [test suite guide](../tests/README.md) for the
verification status.

## Liquid Analyzer extraction

`NCrystalLiquidAnalysisProcessor` owns transient material preparation and a
fresh child process for capturing the actual Liquids `SABData`. The separate
`liquid_capture` package contains the ABI-specific C++ adapter, one-time build
helper, child entry point and validated in-memory data contracts. Compile the
adapter once before scheduling a batch. The liquid Analysis processor does
this automatically when no capture library is supplied. Processing owns the
temporary build directory and uses an interprocess lock and source/library
identity cache. The build uses the installed NCrystal and plugin.

Analyzer runners call this backend directly; they must not call production
orchestrators or implement their own subprocess/temporary-file logic. The
backend returns arrays and provenance only, never C++ pointers or files whose
lifetime extends beyond the sandbox. Repeated and parallel calls receive
independent child environments and directories. The native component sum is
checked against the original material before returning results.

The capture route uses the Linux/ELF NCrystal 4.4.6 legacy constructor ABI.
The [Analyzer guide](../analyzer/BACKEND_INTEGRATION_GUIDE.md) describes the
normalization and observable scope.

## Batch MF7 export

`NjoyBatchOrchestrator.generate_ensemble_sab_tapes` writes the reference and
stochastic LEAPR runs through the [LEAPR serializers](../input_output/IO_INTEGRATION_GUIDE.md).
For mode-bearing inputs, `export_full_spectra=False` selects native discrete
oscillator export. Use `True` when sampled peak shapes must be encoded in
the continuous table. Pass the original LEAPR file to the batch.

`NcrystalBatchOrchestrator.generate_ensemble_sab_tapes` sends solid NCMATs to
the solid converter and dispatches `@CUSTOM_LIQUIDS` materials to TARA's
experimental captured-kernel MF7 exporter. Select the target with `atom_label`.
The batch builds a temporary capture adapter when an existing
`liquid_options["capture_library"]` is absent. It infers the target atom
count `natom` from the NCMAT fractions for a small formula unit. For water
with hydrogen selected the inferred value is two.

Liquid export defaults to `vdoslux=0`, `epsilon_e_over_kt=20.0`, and
`emax_eV=0.1`. Treat these as research settings. Select a supported energy
limit and use the same value for the downstream NJOY configuration. The
liquid writer uses compact `LASYM=0` by default. Its lower-level writer
can retain the signed table with `LASYM=1`. The compact representation
needs its own physical checks against the captured kernel. Existing files
require an explicit `liquid_options={"force": True}` to replace them.

## ACE handoff

`AceGenerationConfig.from_endf` derives shared ACE settings from the base
neutron evaluation and the sample MF7 files. `UniversalAcePipeline.run`
then consumes the sample-to-path mapping for execution. Select
`save_njoy_input=True` to write the shared deck to `output_dir/njoy_ace.inp`.
To replay it, use a separate working directory with the base evaluation staged
as `tape20` and one sample MF7 as `tape70`.

Processing tests are grouped by contract. See
[NJOY liquid batch tests](../tests/test_processing/test_njoy_liquid_batch.py),
[NCrystal export tests](../tests/test_processing/test_ncrystal_solid_export.py),
and [ACE pipeline tests](../tests/test_processing/test_ace_pipeline.py).
File validation and successful processing establish the software handoff.
Comparisons with native kernels and transport checks establish the physical
scope of an evaluation.

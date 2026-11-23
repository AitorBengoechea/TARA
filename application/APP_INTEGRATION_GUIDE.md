# TARA Application Domain — Integration Guide

## Responsibility and boundaries

`application/` is the outermost, user-facing layer. It owns the CLI, TOML parsing,
path resolution, application preflight checks, the sequencing of existing TARA
APIs, and run provenance. It **does not** implement spectral sampling, file
serialization, S(α,β), or NJOY/NCrystal physical processing: those remain in
`core/`, `input_output/`, and `processing/` respectively.

The applications and educational notebooks are **two independent entry points**.
The notebooks deliberately declare settings in Python cells, so the learner can
read and modify each variable. The CLI reads its settings from the TOML file
selected with `--config`.

## Directory layout

```text
application/
├── cli.py                    # Legacy sampling CLI + new `ace` entry point
├── configuration.py          # TOML parsing, strict schema and preflight validation
├── ace_workflow.py           # Existing Core / Processing API orchestration
├── configs/
│   ├── ace_njoy_solid.toml
│   ├── ace_njoy_liquid.toml
│   ├── ace_ncrystal_solid.toml
│   └── ace_ncrystal_liquid.toml
└── APP_INTEGRATION_GUIDE.md
```

The related tests are in [the Application test directory](../tests/test_application/).

## Quick start

Requires Python 3.11+ (`tomllib`). Execute the following from the TARA project
root after setting up the scientific runtime, NJOY, and (for NCrystal inputs)
NCrystal:

```bash
python -m application.cli ace --config application/configs/ace_njoy_solid.toml
python -m application.cli ace --config application/configs/ace_ncrystal_solid.toml
python -m application.cli ace --config application/configs/ace_njoy_liquid.toml
python -m application.cli ace --config application/configs/ace_ncrystal_liquid.toml
```

The historical sampling CLI remains available:

```bash
python -m application.cli my_material.leapr --shift-mev 1.0 --shift-pct 5.0 -n 3
```

Config preflight checks **without** sampling or running NJOY:

```bash
python -m application.cli ace --config application/configs/ace_njoy_solid.toml --validate-only
```

This requires the referenced material files and a discoverable NJOY executable,
but does not inspect physical ENDF contents or validate NCrystal's built-in
material by executing NCrystal. It is intentionally cheaper than a real run.

## Config layout and ownership

| TOML table | Purpose |
|---|---|
| `[workflow]` | Select `backend = "njoy"` or `"ncrystal"` |
| `[material]` | Material inputs, atom label/elastic mode (NCrystal), shared temperature |
| `[output]` | Run folder containing `endf/`, `ace/`, and provenance |
| `[ensemble]` | Sample count, reference inclusion, RNG seed, sampling batch size |
| `[execution]` | Worker request and NJOY executable |
| `[sampling]` | `TaraConfig`, sampler algorithm and normalization controls |
| `[ace]` | NJOY THERMR/ACER controls and optional XSDIR files |

Every source and output path in the TOML file is interpreted **relative to that
TOML file**, not the shell's current directory. NJOY's `njoy_executable` can be
an executable name found on `$PATH` or an absolute executable path. For NJOY
sampling, `include_reference` must be `true`; the existing NJOY batch
orchestrator always processes `ensemble.reference` as `sample_000`. NCrystal
can set `include_reference = false`, in which case only stochastic samples are
produced, numbered from `sample_001`.

TOML does not support `None`; use `manual_energy_ranges = []` when selecting
`grid_partition_method = "automatic"`. For manual partitioning, use
`[[0.0, 0.06], [0.06, 0.12]]` in eV. The application validates basic types,
nonfinite values, allowed modes, and mutually exclusive sources, then delegates
sampling validity to `TaraConfig`.

**Automatic PDOS domain sensitivity:**
`[sampling].automatic_partition_prominence_pct = 1.0` configures the minimum
valley prominence as a percentage of the maximum PDOS density. The default
of 1.0% matches the earlier implementation. A lower value such as `0.2`
can detect more (shallower) PDOS valleys; an excessively small value can
resolve numerical noise as extra domains. Allowed values are finite and in
`(0, 100]`. This control has no effect when `grid_partition_method = "manual"`.
Existing TOML files that omit this new setting use 1.0% automatically.
The normalized value appears in `run_manifest.json` for reproducibility.

The legacy sampling command supports the same parameter via
`--automatic-partition-prominence-pct 0.2`; leave the flag out for 1.0%.

**Temperature consistency:** the chosen `[material].temperature_k` is used
both when producing ENDF MF7 TSL data and when constructing the NJOY ACE
configuration. This prevents the 293.6 K versus 296 K mismatch when the
LEAPR source permits the desired temperature under the current temperature
policy. The separate Processing validator remains authoritative.

## Results, sample identity, and reproducibility

A successful execution produces:

```text
<output.directory>/
├── endf/                         # sample_000_sab.endf, sample_001_sab.endf, ...
├── ace/
│   ├── sample_000.ace           # And subsequent generated samples
│   └── xsdir/                   # If [ace].save_xsdir = true
├── run_config.toml               # Copy of the source configuration
└── run_manifest.json             # Resolved paths, seed, sample IDs, settings
```

`sample_000` always denotes the unmodified reference when requested;
`sample_001` is `ensemble[0]`, not the reference. The manifest is written **only
after** successful ACE-file and optional XSDIR sanity checks; those checks do
not replace independent physical validation of ENDF/ACE data.

## Test commands

```bash
python -m pytest -q tests/test_application
python -m application.cli ace --help
```

The unit and integration-boundary tests use mocked NJOY/NCrystal/Core/Processing
APIs, validate both TOML examples, confirm the temperature/option handoff,
exercise both NCrystal source modes and no-reference mode, and check legacy CLI
compatibility. Real external NJOY/NCrystal calculations are **not** covered.

## Extending application configuration

1. Add the field to `application/configuration.py`'s strict table schema.
2. Add it to each applicable template, with a clear comment and unit.
3. Delegate the parsed value to an **existing** Core or Processing API. Do not
   introduce a second sampling or NJOY physical implementation.
4. Add a test for the valid case and a test for a mistyped or conflicting value.
5. If it is an ACE teaching control, document the equivalent parameter in the
   notebook Markdown while leaving its Python configuration self-contained.

## Liquid-water configurations

Both liquid presets set `[material].state = "liquid"`. Liquid ACE requires
`[sampling].auto_normalize = true` and an explicit `[ace].mti`. The water
presets use `mti = 222` for inelastic scattering and `mte = 0` for no separate
thermal elastic reaction. The ACE handoff allows the additional scatterer
listed in the water MF7 evaluation.

[The NJOY preset](configs/ace_njoy_liquid.toml) selects a tabulated LEAPR temperature
before sampling.
The direct batch API receives the original LEAPR file and defaults to native
discrete-mode export. The strict TOML schema currently exposes neither
`export_full_spectra` nor `save_njoy_input`. These are direct API options.

[The NCrystal preset](configs/ace_ncrystal_liquid.toml) uses the NCMAT fixture with
the Liquids plugin. Processing builds the capture library in a temporary
directory and exports the liquid MF7 evaluation before NJOY generates ACE.

## Optional diffusion sensitivity experiments

The `[diffusion_sampling]` TOML table is optional and disabled if absent.
Only explicit `enabled=true` selects `MaterialEnsemble` in NJOY liquid runs.
Supported modes are `diffusivity_only`, `weight_fixed_d`, and
`joint_independent`. Bounds are sensitivity percentages, not experimentally
calibrated uncertainty intervals. Weight variation requires explicit
`weight_compensation="continuous_only"`. VDOS sampling can independently be
active or disabled; translation-only runs bypass the VDOS `TaraConfig`.
Per-sample physical parameters are included in the successful-run manifest
for active material ensembles.


## Parameter-space sampling policy (default LHS)

The optional `[ensemble].sampling_method` entry selects either
`"latin_hypercube"` (the **default even when the key is omitted**) or
`"monte_carlo"` (for explicit historical sampling).

```toml
[ensemble]
n_samples = 100
include_reference = true
random_seed = 20261021
batch_size = 32
sampling_method = "latin_hypercube"
```

The key belongs to `[ensemble]`, **not** `[sampling]` or
`[diffusion_sampling]`: it controls the statistical design, not physical
perturbation bounds. Strict TOML validation rejects unknown spellings and
non-string values. `load_ace_config()` resolves missing keys to
`"latin_hypercube"`, and `_prepare_ensemble()` passes the method and seed to
`TaraConfig`. This applies to solid VDOS and liquid VDOS on both existing
backend routes, without moving sampling mathematics into Application.

For an NJOY liquid with `[diffusion_sampling].enabled = true`,
`_prepare_diffusion_ensemble()` uses the **same design method**. If VDOS
sampling is active, diffusion uses the same sub-ensemble index ranges and
sample indices as the spectral blocks; it has independently stratified named
parameter streams. If VDOS sampling is inactive, diffusion alone receives one
N-row design, without creating an invalid all-zero `TaraConfig`. No diffusion
sampling occurs if its section is disabled or absent.

`random_seed` identifies the experimental design and is stored, together with
`sampling_method`, in `run_manifest.json`. The manifest also records topology
block start/end indices and domain counts when the ensemble provides them,
and it retains per-sample physical diffusion values for material ensembles.
Only a validated completed run has a success manifest.

LHS draws are bounded and signed (`2*u-1`), so **both reductions and increases**
are possible under existing perturbation bounds. LHS guarantees stratified
**marginal parameter ranges per topology sub-ensemble**; it does not guarantee
full Cartesian coverage or statistical independence of finite realized values.
`sample_000` remains the untouched reference, excluded from `n_samples` and
all LHS strata. Numerical `batch_size` must not change the generated design.
For exact legacy numerical results, specify `sampling_method = "monte_carlo"`.

See the [sampling configuration tests](../tests/test_application/test_sampling_configuration.py),
[configuration tests](../tests/test_application/test_configuration.py), and
[liquid workflow tests](../tests/test_application/test_liquid_workflows.py).

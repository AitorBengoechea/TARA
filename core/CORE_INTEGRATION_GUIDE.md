# Core Architecture: Core Domain Integration Guide

## 1. Architectural Philosophy

The `core` directory is the foundational pillar and the mathematical heart of the framework. It acts as the absolute **Single Source of Truth** for data schemas, ensemble generation algorithms, and global configurations.

**Golden Rule of the Core Layer:** **Zero Domain Logic.**
The `core` folder must *never* know about NJOY, NCrystal, VASP ... It does not parse external files, and it does not run external executables. It strictly defines *what* the data is (via `core/data/`) and provides the pure mathematical algorithms to perturb, sample, and reconstruct that data (via `core/math_engine/`, `core/physics/`, and `core/sampler/`).

## 2. Directory Structure

The core domain is structured into root-level utilities and four operational sub-packages:

```text
core/
├── config.py           # Global configurations and environment constants
├── data/               # Universal data schemas (e.g., input_pdos.py)
├── math_engine/        # Pure mathematical fitting and reconstruction algorithms
├── sampler/            # Ensemble generation and statistical topology strategies
└── physics/            # Backend-independent vibrational reconstruction
```

## Physical material models

`core.data.material_scattering` introduces `MaterialScatteringModel`,
`SpeciesDynamics`, `ScatteringSpecies` and `DiscreteOscillator`. All are
backend-independent physical data containers and validators. They require
unit-normalized continuous spectra while keeping oscillator and translational
weights separate. `MaterialScatteringModel.with_spectrum()` creates a new
material realization without changing fixed model components. The model can also store a single tabulated full vibrational distribution
with no separate oscillators. See the [LEAPR I/O guide](../input_output/IO_INTEGRATION_GUIDE.md)
for the projection from that representation to a native backend input.

### Egelstaff–Schofield translation (CAB water)

`SpeciesDynamics.translation_model="egelstaff_schofield"` and
`translation_parameter=c` store the physically defined effective-width
translation parameter separately from the unit-integral continuous spectrum.
It is not treated as an arbitrary SI diffusion constant or added to the VDOS.
Rounded CAB component weights are accepted within absolute tolerance 1e-5.

## Optional liquid translational sampling

`core.config.DiffusionSamplingConfig(enabled=False)` is an opt-in, independent
sensitivity configuration. `core.sampler.diffusion_sampler.DiffusionSampler`
only supports strictly positive Egelstaff--Schofield `translation_parameter=c`.
It transforms independent physical diffusivity and weight ratios into
`c_sample = c_reference * diffusivity_factor / weight_factor`.
`core.data.material_ensemble.MaterialEnsemble` lazily composes these parameters
with optional existing `SpectraEnsemble` samples. The reference is separate;
`__getitem__(0)` is stochastic sample 001. No backend dependencies are added.
Weight variation is supported only with explicitly selected continuous-only
compensation, preserving separately modeled oscillators and source rounding
residuals. If the model already combines the modes in a full VDOS, compensation
changes the total vibrational coefficient.
Free-gas to Egelstaff--Schofield model transitions are excluded.


## Centralized parameter-space designs (LHS default)

`core.sampler.parameter_space.ParameterSpaceSampler` is the **single generator
of continuous stochastic coordinates** used by VDOS and liquid diffusion
sampling. It is backend-neutral: do not import NJOY, NCrystal, LEAPR, or file
I/O into this class. `TaraConfig.sampling_method` defaults to
`"latin_hypercube"`; `"monte_carlo"` is an explicit legacy selection.
`TaraConfig.random_seed` is the reproducible design seed (application callers
forward `[ensemble].random_seed`).

### Statistical universe and sign convention

Each topology block `(start_index, end_index, boundaries)` defines one
sub-ensemble and its own `N_block`-row randomized Latin hypercube. The
number of **independent active dimensions** depends on topology and coupling:

* independent domains: one dimension per domain and per active variable;
* `random_group`: one dimension per variable, broadcast over the domains;
* `variable_coupling="coupled"`: amplitude reuses the spatial coordinates;
* inactive independent perturbations: zero multipliers; no LHS dimension;
* liquid diffusion parameters add independent `diffusivity` and/or
  `translation_weight` dimensions only when their bounds are nonzero.

Randomized LHS draws a permutation of each column's `N_block` strata and a
uniform within-stratum jitter, then maps `u in [0,1)` to `xi = 2*u - 1`.
**Negative values are valid**: decreasing VDOS energy/amplitude or liquid
physical factors. Marginal stratification is guaranteed **within each
sub-ensemble**, not for all combinations or the whole combined ensemble.
No unrequested correlations are imposed. Normalization and physical constraints
remain in the original physical operators.

The design engine allocates reproducible, independent streams using
`SeedSequence([random_seed, subensemble_id, CRC32(parameter_group)])`. This
makes diffusion coordinates independent of the number of VDOS domains and
independent of the order in which named groups are generated. Parameter names
currently include `vdos_spatial`, `vdos_amplitude`, `diffusivity`, and
`translation_weight`. Do not rename these without a reproducibility-migration
plan. Each parameter group has the same sub-ensemble sample row ordering.
Do not regenerate a design when `batch_size` changes: **generate all rows of
one topology block before matrix chunking**. The independent unperturbed
`sample_000` is never counted in the LHS stratum allocation.

### Monte Carlo compatibility and topology separation

For `sampling_method="monte_carlo"`, VDOS draws retain the legacy global
`np.random.uniform` call order, including draws for physically inactive
variables; liquid diffusion retains its original local NumPy generator and
its previous seed derivation in `application/ace_workflow.py`. Monte Carlo
`RandomTopology` retains `np.random.randint`. LHS `RandomTopology` uses its
own seeded discrete-topology stream; topology draws are **not LHS variables**.
In the LHS branch, `ParameterSpaceSampler` does not touch global NumPy state.

`SpectraEnsemble.generation.sampling_method` is recorded alongside existing
block boundaries and multipliers. Existing `SpectraEnsemble`, `SamplingBlock`,
`MaterialScatteringModel`, and `MaterialEnsemble` contracts remain intact.
Diffusion uses `spectra_ensemble.sampling_blocks` to align its stratification
with the selected VDOS sub-ensembles; diffusion-only uses one block.

Test ownership is in [parameter-space tests](../tests/test_core/test_parameter_space.py),
[VDOS sampling tests](../tests/test_core/test_vdos_sampling.py), and
[diffusion sampling tests](../tests/test_core/test_diffusion_sampling.py).

## Reconstructing a full vibrational spectrum

`core.physics.vibrational_spectrum.reconstruct_vibrational_spectrum` combines
a weighted continuum with oscillator contributions on a uniform grid starting
at zero. It returns a unit-area spectrum and a separate total vibrational
weight. Translation remains a separate component.

The default representation uses area-normalized Gaussians with a 1 meV
standard deviation. This width is a numerical model used in the illustrative
water example. The routine refines its default grid to a spacing no larger
than one third of the width. A supplied grid must resolve the Gaussians and
cover their tails. With `oscillator_sigma_ev=0.0`, two-node deposition
preserves each discrete mode's area and energy centroid.

The sampler perturbs the reconstructed curve through the existing
`TaraSampler` interface. Separate Core controls for oscillator energy and
weight are outside this workflow. See the [deposition regressions](../tests/test_core/test_vibrational_spectrum.py)
and [full-spectrum export tests](../tests/test_input_output/test_leapr_spectrum_export.py).

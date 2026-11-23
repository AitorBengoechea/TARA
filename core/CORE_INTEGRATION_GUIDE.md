# Core Architecture: Core Domain Integration Guide

## 1. Architectural Philosophy

The `core` directory is the foundational pillar and the mathematical heart of the framework. It acts as the absolute **Single Source of Truth** for data schemas, ensemble generation algorithms, and global configurations.

**Golden Rule of the Core Layer:** **Zero Domain Logic.**
The `core` folder must *never* know about NJOY, NCrystal, VASP, or CASTEP. It does not parse external files, and it does not run external executables. It strictly defines *what* the data is (via `core/data/`) and provides the pure mathematical algorithms to perturb, sample, and reconstruct that data (via `core/math_engine/` and `core/sampler/`).

## 2. Directory Structure

The core domain is structured into root-level utilities and three distinct operational sub-packages:

```text
core/
├── config.py           # Global configurations and environment constants
├── data/               # Universal data schemas (e.g., input_pdos.py)
├── math_engine/        # Pure mathematical fitting and reconstruction algorithms
└── sampler/            # Ensemble generation and statistical topology strategies
```

## V2 physical material models

`core.data.material_scattering` introduces `MaterialScatteringModel`,
`SpeciesDynamics`, `ScatteringSpecies` and `DiscreteOscillator`. All are
backend-independent physical data containers and validators. They require
unit-normalized continuous spectra while keeping oscillator and translational
weights separate. `MaterialScatteringModel.with_spectrum()` creates a new
material realization without changing fixed model components. It is the
first step toward independently sampled oscillator modes, but does not replace
or mutate the existing `SpectraEnsemble` interface. See
`docs/sections/liquid-material-ace.md`.

### Egelstaff–Schofield translation (CAB water)

`SpeciesDynamics.translation_model="egelstaff_schofield"` and
`translation_parameter=c` store the physically defined effective-width
translation parameter separately from the unit-integral continuous spectrum.
It is not treated as an arbitrary SI diffusion constant or added to the VDOS.
Rounded CAB component weights are accepted within absolute tolerance 1e-5.

## Public sampling API

Use `TaraConfig` and `TaraSampler` for vibration-only, translation-only,
and joint studies. Both parts share the sample count, design method, and seed.
Vibration-only studies take a `PhononSpectra` or a material and return a
`SpectraEnsemble`. Enabling `diffusion` requires an Egelstaff–Schofield liquid
material and returns a `MaterialEnsemble`. For translation only, leave both
vibrational bounds at zero. For joint sampling, give at least one positive
vibrational bound and enable diffusion in the same configuration.

```python
from core.config import TaraConfig
from core.sampler.concrete_samplers import TaraSampler

configuration = TaraConfig(
    n_samples=100,
    random_seed=42,
    max_energy_shift_mev=1.0,
    max_amplitude_shift_pct=3.0,
    diffusion=dict(
        enabled=True,
        mode="joint_independent",
        max_diffusivity_shift_pct=10.0,
        max_weight_shift_pct=10.0,
        weight_compensation="continuous_only",
    ),
)
sampler = TaraSampler(config=configuration, material=material)
ensemble = sampler.generate_ensemble(batch_size=32)
```

Internally, `VibrationSamplingConfig` validates the vibrational settings and
`VibrationSampler` generates the spectra. `DiffusionSamplingConfig` validates
the translation settings and `DiffusionSampler` transforms the physical
diffusivity and weight ratios into
`c_sample = c_reference * diffusivity_factor / weight_factor`. Continuous-only
compensation preserves oscillator contributions and source rounding residuals.
`MaterialEnsemble` combines each translation draw with the matching vibration
sample when available. The reference is separate and index 0 is stochastic
sample 001. `generate_ensemble()` returns the result directly.


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
its historical ACE seed derivation in `TaraSampler`. Monte Carlo
`RandomTopology` retains `np.random.randint`. LHS `RandomTopology` uses its
own seeded discrete-topology stream; topology draws are **not LHS variables**.
In the LHS branch, `ParameterSpaceSampler` does not touch global NumPy state.

`SpectraEnsemble.generation.sampling_method` is recorded alongside existing
block boundaries and multipliers. Existing `SpectraEnsemble`, `SamplingBlock`,
`MaterialScatteringModel`, and `MaterialEnsemble` contracts remain intact.
Diffusion uses `spectra_ensemble.sampling_blocks` to align its stratification
with the selected VDOS sub-ensembles; diffusion-only uses one block.

Test ownership: `tests/test_core/test_parameter_space_lhs.py`,
`tests/test_core/test_lhs_vdos_integration.py`, and
`tests/test_core/test_lhs_diffusion_integration.py`.

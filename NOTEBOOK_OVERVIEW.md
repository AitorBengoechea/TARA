# TARA — Notebook Workflow and Development Guide

> **Human-maintained document.** Edit this file at the repository root, not its copy under `tara_ai_context/`.
> `generate_ai_context.py` copies this file unchanged during every full context rebuild.

## Purpose and scope

The `notebooks/` directory contains executable scientific workflows, examples,
analyses, and validation exercises for the TARA stochastic sampling framework.
Notebooks demonstrate how TARA's public interfaces and scientific backends are
used; they do **not** replace the production implementations or domain contracts.

This guide describes how to interpret and maintain notebooks. It does not assert
that every notebook executes successfully against the current checkout, that its
saved outputs are current, or that external NJOY/NCrystal backends are installed.

## Notebook directory organization

The following responsibilities follow the directory structure documented in the
project map. Individual notebooks may have more specific purposes; inspect their
actual cells before drawing conclusions about their behavior.

| Source directory | Intended workflow area |
|---|---|
| `notebooks/00_getting_started/` | Introductory examples for solid and liquid workflows using NJOY or NCrystal |
| `notebooks/01_spectrum_sampling/` | Vibrational-spectrum sampling, domains, correlations, and sampling options |
| `notebooks/02_analysis/` | Cross sections, kernels, dynamic structure factors, scattering ratios, and thermodynamic analyses |
| `notebooks/03_ace_generation/` | ACE generation workflows and their production handoff |

The repository may contain shared utilities, such as `notebooks/tara_plotting.py`,
and Markdown documentation alongside notebooks. Such files remain normal
repository files; do not infer their exact interfaces from this overview.

## Reading order and settings

Start with the examples in `00_getting_started/`. Read
`01_spectrum_sampling/` in filename order. The first two notebooks, basic
sampling and deterministic preview, establish the concepts used in the
advanced navigation and specialized examples. Each settings block labels the
inputs required by that example, the optional controls, and the settings fixed
by the exercise. Example values match the editable cells.

The analysis notebooks for cross sections, dynamic structure factors,
scattering kernels, and scattering ratios follow a common seven-step workflow
from inputs to plots. They retain an independent reference and use three
stochastic samples in the teaching examples. The liquid examples separate the
low-energy, bending, and stretching VDOS regions to make small changes visible.
Kernel plots integrate the differential response over scattering cosine.
Ratio plots divide the upscattering and downscattering integrals by the full
continuous kernel integral.

For NJOY liquid analysis and production, keep the original LEAPR template
available for native oscillator export. For NCrystal liquid analysis, load
the hydrogen VDOS from the illustrative NCMAT and let Processing build the
capture adapter. See the [I/O guide](input_output/IO_INTEGRATION_GUIDE.md) and
[Analyzer guide](analyzer/BACKEND_INTEGRATION_GUIDE.md) for details.

## Architectural boundaries

- **`core/`** owns the scientific models, numerical algorithms, schemas, and samplers.
- **`input_output/`** owns external-format reading, writing, and translation.
- **`processing/`** owns file-centric production, external backend execution, and ACE handoff.
- **`analyzer/`** owns downstream statistical and physics-oriented evaluation.
- **`application/`** owns user-facing configuration and workflow orchestration.

A notebook may orchestrate components from several domains, but a notebook-only
workaround is not an acceptable substitute for fixing a defect in the domain
that owns the behavior. Read the applicable named `*_INTEGRATION_GUIDE.md` files
and inspect current source and tests before modifying production code.

## Notebook maintenance conventions

1. **Preserve reproducibility.** Record relevant configuration, random seeds,
   sample identifiers, material, temperature, and backend assumptions when used.
2. **Keep units and normalization explicit.** State the basis of spectra,
   cross sections, uncertainties, and physical quantities; never silently
   resample, scale, or normalize data for a plot or comparison.
3. **Respect the scientific backend.** Do not conflate NJOY/LEAPR and NCrystal
   representations, processing semantics, or physical models.
4. **Separate demonstration from production.** Reusable logic belongs in its
   owning Python module, not in a long notebook cell copied across workflows.
5. **Keep changes targeted.** Preserve intended notebook plots, comparison
   ranges, column meanings, and cell dependencies unless the task requests a change.
6. **Check argument documentation.** Use the same Setting, Example value, and What it controls columns immediately above editable cells. Describe fixed values and display settings where they are used.
7. **Validate honestly.** A notebook rendering or a synthetic smoke test does
   not establish that NJOY/NCrystal runs or nuclear-data outputs are correct.

## Generated AI-context representation

The repository-root `generate_ai_context.py` rebuilds the entire
`tara_ai_context/` directory on each full run. It produces:

- `NOTEBOOK_OVERVIEW.md`: an **exact copy of this manually edited guide**.
- `notebook_overview.md`: an **automatically generated inventory** of notebooks,
  their source paths, first Markdown headings, and code/Markdown cell counts.
- `notebooks/<top-level-directory>.md`: **one combined source-cell codebook per
  top-level directory**, including its nested subdirectories. A notebook directly
  under `notebooks/` is included in the `_root.md` group.

The generated notebook codebooks retain original cell order and text, but omit
saved execution outputs and attachments. They do not execute code, validate
notebook results, or replace original `.ipynb` files. Rich media, interactive
widgets, and stateful execution must be examined in the original notebooks.

The generated `notebook_overview.md` is a navigation aid, not a manually
maintained description of scientific meaning. Update scientific guidance here
and update the underlying `.ipynb` or production modules for functional changes.

## Development context prerequisite

Generated context files alone do not satisfy the requirement for the
**Scientific Python Developer** skill. That skill may be activated only when
the complete current TARA project is uploaded and inspectable, including source,
integration guides, configuration, and tests (excluding environments, caches,
and secrets). When only codebooks or selected notebooks are available, restrict
work to explicitly scoped review or documentation tasks and explain validation
limitations.

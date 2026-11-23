# Input/Output Modular Architecture: I/O Integration Guide

## 1. Architectural Philosophy

This framework strictly isolates file parsing and formatting from the physical analysis. The core physics engine operates **exclusively** on our unified Python data structures (e.g., `PhononSpectra`).

The `input_output` domain acts as the translation border. Third-party file formats (NCrystal NCMAT, NJOY LEAPR inputs, VASP, etc.) are complex and uniquely structured.

**Golden Rule of the I/O Layer:** **Zero Physics.** The I/O modules do not compute thermodynamics or kinematics. They simply extract, convert units, normalize, and shape external data into the unified schemas, or vice versa.

## 2. Directory Structure

The `input_output` directory contains format-specific readers and writers,
shared I/O helpers, and a package for transferring parameters between native inputs.

```text
input_output/
├── base.py             # Abstract base classes for Readers and Writers
├── factory.py          # Detect native formats and load the matching reader
├── exporter.py         # Export completed ensembles and results
├── spectrum_grid.py    # Resample completed spectra on native grid spacing
├── leapr/              # Sub-package for NJOY LEAPR format
├── ncrystal/           # Sub-package for NCrystal format
└── transformations/    # Optional parameter transfer between native inputs
```


## NCrystal NCMAT files

The [`ncrystal/`](ncrystal/) package reads and writes NCMAT material
definitions for solids and Liquids plugin inputs.

| File | Responsibility |
| --- | --- |
| [`core.py`](ncrystal/core.py) | Defines `NCrystalFile`, the parsed material and its structural and dynamical data. `pdos_for_atom()` selects the VDOS for a specific `@DYNINFO` element label. |
| [`reader.py`](ncrystal/reader.py) | Loads NCMAT sections into the file model, including cell data, atom positions, and dynamics. Retains unrecognized sections for writing. |
| [`parsers.py`](ncrystal/parsers.py) | Splits NCMAT into blocks, extracts numeric fields, and expands compact array values. |
| [`writer.py`](ncrystal/writer.py) | Writes material fields, dynamics, and retained raw sections back to NCMAT. |
| [`workflows.py`](ncrystal/workflows.py) | Provides spectrum plotting and interpolation helpers for parsed NCMAT material. |
| [`liquids.py`](ncrystal/liquids.py) | Loads and validates `@CUSTOM_LIQUIDS` settings and replaces the VDOS of one selected species while preserving plugin settings and unrelated sections. |
| [`material_adapter.py`](ncrystal/material_adapter.py) | Builds a normalized solid-like VDOS from a Core material model’s continuum and fixed oscillator modes. |
| [`exceptions.py`](ncrystal/exceptions.py) | Defines `NCrystalFormatError` for NCMAT parsing and spectrum access. |
| [`__init__.py`](ncrystal/__init__.py) | Marks the NCMAT package. Import the public helpers from their modules. |

### Load a species spectrum

Use the exact `@DYNINFO element` label. `pdos_for_atom()` requires one
matching VDOS block and returns a series indexed by energy in eV. The
generic `pdos` property checks the custom VDOS fields first and otherwise
uses the first block with a VDOS. Use the explicit selector for materials
with more than one species.

```python
from input_output.ncrystal.core import NCrystalFile

material = NCrystalFile.from_file("tests/fixtures/liquid_water_295K_illustrative.ncmat")
hydrogen_pdos = material.pdos_for_atom("H")
```

### Prepare a liquid material

`load_liquid_ncmat()` validates a plugin-ready NCMAT template.
`replace_vdos()` updates one species using a supplied, positive,
uniform energy grid and a density that already includes all vibrational modes
intended for the plugin. For a Core material model that keeps oscillator modes
separate, `fold_solidlike_vdos()` can fold them into the continuum. The plugin
handles translation separately. Processing loads the prepared material into
NCrystal and owns the kernel capture and MF7 export. See the
[Processing guide](../processing/PROCESSING_INTEGRATION_GUIDE.md).

The [NCMAT I/O tests](../tests/test_input_output/test_ncmat_io.py) and
[VDOS tests](../tests/test_input_output/test_ncmat_vdos.py) cover the file
model and spectrum preparation.


## NJOY LEAPR files

The [`leapr/`](leapr/) package reads native LEAPR decks, exposes their spectra,
and prepares sample-specific inputs for NJOY.

| File | Responsibility |
| --- | --- |
| [`core.py`](leapr/core.py) | Defines `LeaprFile`, the parsed input deck, its temperature blocks, and its spectrum accessors. |
| [`reader.py`](leapr/reader.py) | Reads the LEAPR header, scatterer metadata, grids, temperature blocks, and footer. |
| [`parsers.py`](leapr/parsers.py) | Reads numeric cards and count-delimited arrays such as the phonon table and oscillator data. |
| [`writer.py`](leapr/writer.py) | Serializes a `LeaprFile` to native text in memory or on disk. |
| [`workflows.py`](leapr/workflows.py) | Prepares and serializes sample-specific spectra and liquid material models using the original template. |
| [`temperature.py`](leapr/temperature.py) | Lists tabulated temperatures, resolves legacy solid requests, and selects exact source blocks when required. |
| [`material_adapter.py`](leapr/material_adapter.py) | Converts the selected template block into a Core `MaterialScatteringModel`, with explicit composition and translation parameters. |
| [`vibrational.py`](leapr/vibrational.py) | Combines the source continuum and discrete modes into a full vibrational spectrum for sampling. |
| [`discrete_export.py`](leapr/discrete_export.py) | Projects a sampled full spectrum onto the native continuum grid and recovers each discrete mode’s centroid and weight. |
| [`exceptions.py`](leapr/exceptions.py) | Defines `LeaprFormatError` for invalid LEAPR input. |
| [`__init__.py`](leapr/__init__.py) | Marks the LEAPR package. Import the public helpers from their modules. |

### LEAPR single-temperature evaluation policy

TARA reads every temperature block in the original LEAPR template, including
negative cards whose absolute values specify calculation temperatures with
reused phonon-spectrum data. The batch serializer produces exactly one
positive-temperature block per sample.

The optional ``temperature`` argument to
``NjoyBatchOrchestrator.generate_ensemble_sab_tapes`` selects its output
temperature. An omitted value retains the first positive-temperature card.
For a legacy mode-free solid template and a requested temperature different from that card:

- Inside the original absolute-temperature range: reuse the first phonon
  spectrum with one warning per batch.
- At most 5 K below or above the range: reuse with an extrapolation warning.
- More than 5 K beyond either boundary: raise ``ValueError`` before starting
  NJOY workers; the user must edit the LEAPR template to expand that range.

The policy is implemented in ``input_output.leapr.temperature`` and does not
retain the original negative cards in generated single-temperature inputs.
The requested value is written without temperature rounding.

### LEAPR material and sampling representation

`LeaprFile` retains the original temperature blocks, continuum, translation
parameters, and oscillator cards. For the selected spectrum, use
`LeaprFile.vibrational_pdos(temperature=...)` or
`material_model_from_leapr(template, state=..., temperature=...)`. Both select the tabulated
block first and reconstruct a full VDOS when `ND>0`. The material adapter
returns a normalized full vibrational curve with one vibrational coefficient
and a separate translation contribution. The input file remains the source
of the native grid and oscillator layout.

Liquid batch preflight and mode-bearing templates serialized by
`prepare_leapr_spectrum` require an exact tabulated temperature. The selector
uses the preceding positive block for a matching negative temperature card.
LEAPR Cards 12, 15, and 16 may be count-delimited without a trailing `/`. The
reader respects `ni` and `nd`, retains optional Card 4 `smin`, and stores all
94 blocks of the CAB fixture.

### Exporting sampled LEAPR spectra

Use `serialize_leapr_spectrum` with a full sampled `PhononSpectra` and the
original LEAPR template. For material ensembles, `serialize_leapr_material`
writes the sample’s translation weight (`twt`) and diffusion parameter
(`c`) into Card 13 at the model’s exact tabulated temperature. It scales
`tbeta` and any recovered oscillator weights to the sample’s total
vibrational weight. Both serializers write from a copy of the template.

| Argument | Example | What it controls |
| --- | --- | --- |
| `export_full_spectra` | `False` (default) | For a template with discrete modes, project the sample to the original continuum grid and write the recovered modes as discrete cards. |
| `export_full_spectra` | `True` | Write the whole continuous sample and set `ND=0`, retaining sampled widths and shapes. |
| `output_spacing_ev` | `None` | Optional table spacing in eV. For mode-bearing inputs, use with full export and check convergence. |

For native export, the I/O layer uses gaps between the original components
to find the sampled continuum and modes. It places the continuum weight in
`TBETA` and writes each mode's centroid and integrated weight. The original
mode count and continuum grid are retained. Translation is retained for a
VDOS-only sample. The unchanged reconstructed reference recovers the original
cards. Samples whose components overlap at the source gaps require full export.

Keep the original LEAPR file for the processing handoff. A written `ND=0`
file does not carry the original oscillator provenance. The in-memory
`prepare_full_vibrational_template` helper keeps source provenance for its
own lifetime.

The [material tests](../tests/test_input_output/test_leapr_material.py) cover
Card 13 and selected-temperature behavior. The
[spectrum export tests](../tests/test_input_output/test_leapr_spectrum_export.py)
cover both export representations, reconstruction, and grid projection.

## Native input transformations

[`input_output.transformations`](transformations/) is an optional preprocessing
API. Use it to copy a selected phonon density of states (PDOS) into a new file
based on an existing native template. Source and template can use the same
format or different formats. ACE generation does not call this API automatically.

| Module | Responsibility |
| --- | --- |
| `api.py` | Discover transferable parameters and write the transformed native input. |
| `leapr.py` | Read the principal scatterer's continuous PDOS and replace LEAPR Cards 11 and 12 in each positive-temperature block. |
| `ncrystal.py` | Read and replace the VDOS of one species in its NCMAT `@DYNINFO` block. |
| `models.py` | Represent a unit-area continuous spectrum with energies in eV and density in inverse eV. |
| `spectra.py` | Convert between LEAPR's zero-origin grid and NCrystal's positive-origin grid with its quadratic acoustic tail. |
| `elements.py` | Validate element selectors and identify the LEAPR principal scatterer from its atomic weight ratio. |
| `registry.py` | Select a native input adapter by file extension. |

Use `available_parameters(path)` on each input to find selectors such as
`H_pdos`. LEAPR discovery is limited to the principal scatterer when its
atomic weight ratio identifies a unique element. A LEAPR source must have
the same normalized PDOS and grid in all positive-temperature blocks so the
selector identifies a single spectrum.

Run this example from the project root to transfer the hydrogen PDOS from
NCMAT into the polyethylene LEAPR template.

```python
from pathlib import Path

from input_output.transformations import available_parameters, transform_inputs

fixtures = Path("tests/fixtures/input_transform")
source = fixtures / "ncrystal_tsl_HinCH2.ncmat"
template = fixtures / "endf81_tsl-HinCH2.leapr"

assert "H_pdos" in available_parameters(source)
assert "H_pdos" in available_parameters(template)

output = transform_inputs(
    source=source,
    template=template,
    parameters=["H_pdos"],
    output="tara_output/transformations/HinCH2.leapr",
)
```

Choose a new output path with the template's extension. The transfer preserves
the template's translation and oscillator cards for LEAPR and unrelated
dynamics blocks for NCMAT. For LEAPR, it transfers the tabulated continuum.
[The transformation tests](../tests/test_input_output/test_transform_inputs.py)
cover conversion in both directions and preservation of unrelated content.

## Sampling and backend ownership

Core generates stochastic spectra and material realizations. I/O reads the
native inputs and serializes those realizations. Sampling method, random
seed, and sub-ensemble boundaries are run provenance. They do not change
how NCMAT blocks or LEAPR cards are written. In ensemble workflows that
include the reference, `sample_000` is the original reference and stochastic
samples begin at `sample_001`. Processing runs the engines and produces
thermal-scattering evaluations and ACE files.

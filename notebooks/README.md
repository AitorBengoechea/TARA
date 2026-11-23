# TARA scientific workflow notebook gallery

Choose a workflow and material state. In the engine-selectable notebooks,
set `ENGINE` to `"ncrystal"` or `"njoy"` in Step 0, then run all cells.
The setup through Step 3 is shared by notebooks for the same material state.

| Stage | Folder | What you do |
| --- | --- | --- |
| 00 | `00_getting_started/` | Generate ACE files for graphite or liquid water. |
| 01 | `01_spectrum_sampling/` | Explore the shared spectrum sampling algorithms. |
| 02 | `02_analysis/` | Calculate cross sections, dynamic structure factors, kernels, ratios, or PDOS thermodynamics. |
| 03 | `03_ace_generation/` | Configure advanced ENDF and ACE generation. |
| 04 | `04_translation_sampling/` | Compare [diffusivity](04_translation_sampling/diffusivity_effect.ipynb), [translation weight](04_translation_sampling/translation_weight_effect.ipynb), and [joint translation sampling](04_translation_sampling/joint_translation_effect.ipynb). |

Getting started, the paired analysis workflows, and ACE generation each
have a `solid.ipynb` and a `liquid.ipynb`. Each translation experiment has
a liquid notebook with both engines. The sampling tutorials and PDOS
thermodynamics retain their standalone workflows.

Use the supplied gallery in place of the previous `notebooks/` directory
to avoid showing superseded engine-specific notebooks alongside the new ones.

# TARA regression tests

The suite follows the production domains. Add a bug reproduction to the module
that owns the behavior being checked. Give each regression a descriptive test
name or a parametrized case ID. In a short docstring, state the failure trigger
and the invariant being protected. For numerical comparisons, record the
units, normalization, and reason for the assertion tolerance when they matter.

| Directory | Responsibility |
| --- | --- |
| `test_core/` | Spectrum models, numerical operators, and sampling contracts |
| `test_input_output/` | Format parsing, material adapters, grid projection, and serialization |
| `test_processing/` | Deck construction, backend execution, validation, and batch workflows |
| `test_analyzer/` | Parser contracts, physical observables, and ensemble aggregation |
| `test_application/` | Configuration, CLI, installation, and application workflows |
| `test_integration/` | Cross-domain handoffs and explicitly enabled backend checks |
| `fixtures/` | Existing LEAPR, NCMAT, ENDF, and transformation inputs |
| `support/` | Factories shared by tests from multiple domains |

Project-wide worker policy tests live in `test_parallel_execution.py`.
Keep fixtures used by only one file local to that file. Put common in-memory
builders in `support/`, and have each builder return fresh mutable test data.
Use parametrization for the same contract with multiple inputs. Keep distinct
failure modes and different assertions as separately named tests.

## Install this reorganization

Move the previous test suite aside, preserving its `tests/fixtures/` contents.
Install the supplied `tests/` files and keep the existing fixtures at that
same path. Avoid overlaying the new modules on the old ones, since that would
collect the moved tests twice. The archive contains no fixtures or production
source files.

## Run the tests

From the TARA repository root in the normal project environment:

```bash
python -m pytest tests --collect-only -q
python -m pytest tests -q
python -m pytest tests/test_input_output/test_leapr_spectrum_export.py -q
```

For the native NCrystal Liquids smoke and generated-NCMAT checks, use an
environment with the compiled plugin available:

```bash
TARA_RUN_LIQUID_PLUGIN_SMOKE=1 python -m pytest \
    tests/test_integration/test_ncrystal_liquid_plugin.py -q
```

The existing environment gates for real NJOY and liquid adapter tests are
retained. A mocked backend test checks its handoff contract.

## Migration record

`test_case_map.tsv` maps every collected case, including its parameter ID,
from the 2026-11-18 test source export to this layout. It records the outcomes
before moving, after moving, and after updating the one obsolete NJOY sample
setup. The map is a migration record rather than an index to maintain manually.

The structural move reduced 82 test files to 53, preserving all 458 test
function definitions and 654 collected cases. The follow-up NJOY case now
constructs the reference and a continuum-perturbed sample as full VDOS inputs.
It parses the generated native LEAPR cards to check that the two oscillators,
translation parameters, separate reference, and source data are preserved.

The full-suite comparison used the source snapshot available in the workspace.
That snapshot lacks the production root scheduler and installer. The run also
lacks PyTables, and one Processing regression detects a NCrystal analysis-state
grid mismatch. The remaining failures and opt-in skips stay visible in the map.

The baseline and move-only runs both returned 608 passed, 36 failed, and
10 skipped, with matching outcomes for each mapped case. After the NJOY case
update the counts were 609 passed, 35 failed, and 10 skipped. In a separate
opt-in run with the compiled NCrystal Liquids plugin and `nctool` available,
both moved plugin integration tests passed.

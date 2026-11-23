# Thermal Scattering Modular Architecture: Backend Integration Guide

## 1. Architectural Philosophy

The `analyzer` domain is an optional downstream analysis layer. It composes Core data models with reusable backend-execution capabilities from Processing, while owning only analysis-specific adaptation, in-memory state, extraction, mathematics, and ensemble aggregation.

This framework strictly decouples:
 **Execution** (owned by reusable Processing backends), **State** (Analyzer engines), **Extraction** (Analyzer parsers), and **Mathematics** (Analyzer physics utilities).

If you are adding a new nuclear data backend (e.g., FLASSH, FRENDY, ACE-generators), you **do not need to write any physical mathematics**. Multi-incident-energy integrations, Dynamic Structure Factors, and Thermodynamic integrations are mathematically universal and already implemented.

Your sole responsibility is to build the "Bridge" connecting your specific backend to our universal schema. The Analyzer adapter normally has three parts:

1. **The Engine** (`analyzer/backends/<name>_engine.py`)
2. **The Runner** (`analyzer/runners/<name>_runner.py`)
3. **The Parser** (`analyzer/parsers/<name>_parser.py`)

---

## 2. Step 1: The Engine (State Container)

**Goal:** Prevent redundant disk I/O and maintain memory states between execution and parsing.

**Location:** `analyzer/backends/<name>_engine.py`

**Function:**
The Engine is a simple Python `@dataclass`. It receives the raw output from the Runner, loads any necessary files into RAM (as Python dictionaries, trees, or C++ pointers), and holds them so the Parser can read them instantly without touching the hard drive.

**Required Structure:**

* Must be a `@dataclass`.
* Must contain standard metadata: `temperature: float`, `mass_amu: float`, `energy_grid: np.ndarray`, `pdos: np.ndarray`.
* Must contain backend-specific parsed objects (e.g., `parsed_output_tree: Any`, `memory_pointer: Any`).

---

## 3. Step 2: The Runner (Execution Adapter)

**Goal:** Adapt an Analyzer request to reusable Processing execution primitives and return a loaded Engine.

**Location:** `analyzer/runners/<name>_runner.py`

**Function:**
The Runner is a thin adapter. It may select Processing deck builders, provide Analyzer-specific execution parameters, and load transient backend outputs into an Engine. It must not duplicate subprocess execution, sandbox management, or backend deck syntax already owned by `processing.backends` and `processing.deck_builders`.

**Required Structure:**

* Must inherit from `analyzer.core_interfaces.BaseRunner`.
* May depend on `processing.backends` and `processing.deck_builders`.
* Must not depend on `processing.orchestrators` or `processing.pipelines`; Analyzer retains ownership of ensemble iteration, parallelism, sample identity, and aggregation.
* **`execute(self, sample_index: int, spectrum: PhononSpectra, **kwargs)`**:
1. Translate Analyzer parameters into calls to reusable Processing deck/execution primitives.
2. Keep backend files transient unless the backend contract requires otherwise.
3. Load generated backend outputs into RAM while the Processing sandbox is alive.
4. Return the instantiated `<Name>Engine` containing those RAM-loaded objects.


* **`cleanup(self, runner_output: <Name>Engine)`**:
1. Release only Analyzer-owned resources. Processing-owned sandbox cleanup remains the responsibility of the Processing backend.



---

## 4. Step 3: The Parser (Extraction Adapter)

**Goal:** Extract data from the Engine and funnel it into the universal mathematical schemas.

**Location:** `analyzer/parsers/<name>_parser.py`

**Function:**
The Parser receives the `Engine` and performs **zero file I/O**. It simply navigates the RAM-loaded objects and maps them to the `PhysicsData` schema.

**Required Structure:**

* Must inherit from `analyzer.parsers.base_tsl_parser.BaseTslParser`.
* Must implement `parse(self, sample_index: int, engine: <Name>Engine, **kwargs) -> PhysicsData`.
* The `parse` method orchestrates three private methods:
1. **`_extract_thermodynamics(self, engine)`** populates `ThermodynamicsData`. Use the shared PDOS integrals in `analyzer.physics.thermodynamics` for calculated scalars and record the scope of any native quantities. For example, NCrystal supplies native effective temperature and displacement information for the supported solid path.
2. **`_extract_cross_sections(self, engine, integ, **kwargs)`** populates `IntegralData` with the requested channels. Use the backend's grid contract. NJOY tabulations can be interpolated onto a requested grid. For captured NCrystal liquid cross sections, the Runner and Parser use the same explicit grid.
3. **`_extract_kinematics(self, engine, diff, integ, **kwargs)`**:
* Navigates the engine's memory to extract the $\alpha$ grid, $\beta$ grid, and the Thermal Scattering Law matrix values.
* Formats these strictly into a Pandas DataFrame: `sab_df = pd.DataFrame(sab_matrix, index=alpha_grid, columns=beta_grid)`.
* Pass the backend-normalized table to the shared kinematics using: `self._apply_universal_kinematics(sab_df, diff, integ, temp, mass_amu, **kwargs)`. The parent class handles all the multi-$E_{in}$ loops, angular distributions, and ratios automatically.





---

## 5. AI Prompt Instructions for Generating a New Backend

*(To use this guide with an LLM, copy this entire document and append the following prompt):*

**Prompt:**
"I need to integrate a new backend called **[INSERT BACKEND NAME HERE]** into this architecture.
Based on the exact rules, constraints, and class hierarchies defined in the guide above, please generate the complete Python code for the following three files:

1. The Engine dataclass.
2. The Runner class. If the backend already has reusable execution support in `processing`, the Runner must adapt to that support rather than launching the executable itself. If no reusable Processing backend exists yet, add that execution capability to `processing.backends` first and keep the Analyzer Runner thin.
3. The Parser class (ensure it inherits from `BaseTslParser`, relies on `thermodynamics.py` for scalars, and delegates the `sab_df` matrix to `_apply_universal_kinematics`).

Please use standard placeholder logic (like `pass` or basic dictionary navigation) for the highly specific backend file-reading, but ensure the structural interfaces, kwargs passing, and typing exactly match the framework's requirements."
---

## 6. Backend contracts

The two backends share the Runner, Engine, and Parser interfaces. Their
scattering representations and source grids are backend specific.


## NCrystal source-spaced kernel handoff

The intended solid handoff provides the full Core spectrum in
`NCrystalEngine.energy_grid` and `.pdos`. NCrystal's direct VDOS kernel
functions (`extractGn` and `extractKnl`) use the NCMAT-compatible arrays from
Processing when `.backend_energy_grid` and `.backend_pdos` are supplied.
The [Processing guide](../processing/PROCESSING_INTEGRATION_GUIDE.md)
records the still-open transient-state regression for this handoff.
This avoids reintroducing Core's artificially fine Debye spacing or `E=0` in
the NCrystal VDOS-to-kernel API. The parser validates but never resamples this
format-specific grid or performs file I/O. Manually constructed engines without
these optional backend arrays remain supported when their own grids are
already valid for NCrystal.


## NJOY MF7/MT4 representation contract

`NjoyParser` reads **LASYM** and **LLN** from the produced MF7/MT4 ENDF
section, not from the LEAPR Card 4 request. Missing or unsupported output
flags are errors rather than implicit assumptions. `LLN=1` means natural
logarithm of S in the ENDF-6 format; the decoded values must be finite and
nonnegative.

`analyzer.physics.kinematics.decode_njoy_mf7_sab()` produces the full
**physical**, non-symmetric, signed-beta S before the universal kinematic
handoff. For `LASYM=0`, the symmetric compact S values at b>0 are mapped
to negative beta with `exp(+b/2)` and positive beta with `exp(-b/2)`;
zero is copied once. For `LASYM=1`, both signs must already exist and are
retained unchanged. Preserve all native coordinates and do not interpolate,
sort, merge or re-apply detailed balance. The generic `canonicalize_sab()`
is deliberately kept backward-compatible for non-MF7 callers.

Use the original LEAPR template with `NjoyRunner` for discrete-mode
provenance. `EnsembleAnalyzer.compute(..., runner_kwargs={"export_full_spectra": True})`
opts into the full continuous LEAPR table. Native discrete-mode export is
the default. See the [LEAPR I/O guide](../input_output/IO_INTEGRATION_GUIDE.md)
and the [MF7 decoder tests](../tests/test_analyzer/test_njoy_mf7_representation.py).

## Captured NCrystal liquid kernels

`NCrystalRunner` dispatches files containing `@CUSTOM_LIQUIDS` to
`processing.backends.ncrystal_liquid_analysis.NCrystalLiquidAnalysisProcessor`.
This path does not construct `NuclearData`, run `ncmat2endf`, or regenerate the
liquid scattering law with `extractKnl(vdos=...)`.

The liquid backend builds a temporary capture adapter automatically when
the runner is constructed. Supply `NCrystalRunner(liquid_capture_library=...)`
only to reuse an existing compatible adapter. The supported ABI is Linux/ELF
with standard, unnamespaced **NCrystal 4.4.6** and the legacy constructor
route. Other versions require an adapter verified for their ABI.

Processing prepares a transient material using the established VDOS adapter and
`replace_vdos`, preserving the custom liquid settings and unrelated dynamics.
Each sample has a fresh child, bounded timeout and unique sandbox. Only that
child receives `LD_PRELOAD`. The original material is never modified. Results
are loaded into `LiquidAnalysisState` before the sandbox is removed. Analyzer
continues to own ensemble scheduling, identities and aggregation.

The capture contains one per-atom table per custom liquid component. Component
identity is checked against native mass and bound cross section; ambiguous
matches fail, rather than using constructor order. Other components must be
native free gas. Weighted captured-component cross sections plus native free
gas contributions must reconstruct the original material cross section.
`active_species` selects scattering contributions with their **original**
material fractions; the target must remain active. Absorption retains the
existing whole-material convention. Dynamic libraries, source material and
adapter identities are recorded in each `PhysicsData.metadata` dictionary.

At the Parser boundary, alpha is divided by the target/neutron mass ratio.
Signed beta and the unsymmetrized S values are retained unchanged. The shared
kinematics consumes this target-component table; it is not a combined water
kernel. Its differential ordinates retain the existing Analyzer normalization
**per unit target free scattering cross section**. To compare an integrated
kernel to a component's native contribution per material atom, multiply by
that component's free scattering cross section and original atomic fraction.
Metadata records this distinction from the native cross-section outputs.

All existing kinematic flags are available. Incident energies above the
captured `suggested_emax` are rejected for table-derived kinematics; no SCT or
free-gas extension is invented. Within that bound, shared interpolation still
uses zero outside the finite alpha/beta table. Convergence checks remain
necessary for the requested observables. Native cross sections can extend
beyond that table using NCrystal's own backend treatment. Supply the same
`energy_grid` to Runner and Parser when requesting a custom native XS grid.

`Gn` and thermodynamic scalars are explicitly **solid-like VDOS diagnostics**,
computed from the VDOS actually consumed by NCrystal. They are not total liquid
heat capacity, entropy, or diffusive displacement. This path has continuous
SAB/free-gas scattering and reports no discrete coherent/incoherent elastic
channel. The existing kinematic ratios remain those of the selected SAB table.

Multi-incident-energy DataFrames are outer joins of their native grids; absent
coordinates can therefore be NaN. Ensemble aggregation removes those missing
coordinates before alignment, rather than treating them as invalid physics or
fabricating zeros on the source grid.

The Analyzer path returns in-memory results. For the separate production
handoff to experimental liquid MF7, see the
[Processing guide](../processing/PROCESSING_INTEGRATION_GUIDE.md). The
[liquid Analyzer guide](../TARA_LIQUID_ANALYZER_INTEGRATION.md) has a runnable
setup example.

### Outgoing-energy observables

`calc_dsf=True` returns a table per incident energy, with scattering cosine
on rows and outgoing energy on columns. `calc_scattering_kernel=True`
integrates the DSF over scattering cosine. `calc_ratios=True` integrates the
continuous outgoing-energy kernel above and below the incident energy and
divides each contribution by the total continuous integral. Separate elastic
cross sections are separate observables.

See the [kinematics tests](../tests/test_analyzer/test_kinematics.py) and
[NCrystal backend tests](../tests/test_analyzer/test_ncrystal_backends.py).

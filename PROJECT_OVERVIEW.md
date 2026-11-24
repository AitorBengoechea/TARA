# TARA Stochastic Sampling Engine

## Project Architecture & Workflow Contract

### 1. Core Purpose

TARA is a stochastic sampling framework designed to generate, perturb, and process robust ensembles of nuclear thermal scattering law (TSL) data.

The system separates mathematical modeling, statistical analysis, external-format translation, production processing, and user orchestration into strictly isolated domains.

---

## 2. Architectural Philosophy

TARA follows **strict Domain-Driven Design with separation of concerns**.

### Fundamental rule

> **A domain must never absorb responsibilities belonging to another domain.**

Changes should be implemented in the domain that owns the relevant responsibility. Cross-domain communication must occur through explicit interfaces, schemas, workflows, or configuration objects.

The architecture should favor:

* explicit contracts
* deterministic data flow
* small, testable components
* dependency inversion where appropriate
* pure functions in foundational layers
* minimal coupling between domains
* clear translation boundaries
* preservation of existing domain responsibilities

Avoid introducing convenience dependencies that bypass the established architecture.

---

# 3. Domain Architecture

## 3.1 `application/` — CLI Orchestrator

### Responsibility

`application/` is the user-facing orchestration layer.

It translates user inputs into strict internal configuration objects (`TaraConfig`) and coordinates the execution of the other domains.

### It MAY contain

* CLI argument handling
* user-facing configuration
* command dispatch
* orchestration
* validation of user-facing options
* coordination between domains

### It MUST NOT contain

* mathematical algorithms
* physics calculations
* statistical algorithms
* external-format parsing
* file-format serialization
* NJOY/NCrystal implementation details

### Principle

> `application/` decides **what should happen**, but does not implement the domain operations themselves.

---

## 3.2 `core/` — Single Source of Truth

### Responsibility

`core/` contains the foundational domain model and pure computational machinery.

It is the mathematical and data-model foundation of TARA.

### It contains

* foundational data schemas such as `PhononSpectra`
* `math_engine`
* sampler logic and backend-independent vibrational reconstruction
* domain-level mathematical operations

### Design constraints

`core/` operates on Python/NumPy primitives and internal domain objects.

It must remain independent of:

* external file formats
* filesystem workflows
* CLI concerns
* NJOY
* NCrystal
* production pipeline mechanics
* external translation formats

### Principle

> `core/` knows **what the data and mathematics are**, not where the data came from or where it will go.

---

## 3.3 `input_output/` — Translation Border

### Responsibility

`input_output/` is the boundary between TARA's internal representation and external data formats.

It translates external representations into TARA-compatible internal structures and performs the inverse translation when required.

### Supported external domains

Examples include:

* LEAPR
* NCrystal
* other explicitly supported external formats

### Required package structure

Each format/domain implementation follows the established six-component structure:

```text
core.py
exceptions.py
parsers.py
reader.py
writer.py
workflows.py
```

### It MAY contain

* parsing
* format validation
* reading
* writing
* format-specific workflows
* translation between external and internal representations
* format-specific exceptions

### It MUST NOT contain

* physics calculations
* sampling algorithms
* statistical evaluation
* NJOY production logic
* CLI orchestration

### Principle

> `input_output/` translates data; it does not interpret the underlying physics.

---

## 3.4 `analyzer/` — Statistical Evaluation

### Responsibility

`analyzer/` performs rapid statistical and physics-oriented evaluation of generated ensembles.

It is optimized for evaluating large numbers of samples in memory without requiring the complete production pipeline to persist massive intermediate files.

### It MAY contain

* statistical analysis
* ensemble evaluation
* physics-property extraction
* rapid backend queries
* RAM-based analysis workflows
* aggregation and comparison of sampled results

### Principle

> `analyzer/` evaluates ensembles; it does not own their generation or production.

The analyzer should remain distinct from the file-centric production machinery in `processing/`.

---

## 3.5 `processing/` — Production Library Pipeline

### Responsibility

`processing/` is the production-oriented, file-centric processing pipeline.

It manages the transformation of sampled/generated data into final nuclear data libraries.

### Characteristics

* file-centric
* temporary OS-level sandboxes
* external processing backends
* joblib-based batching
* production-oriented workflows

### Dual-Path Convergence

NJOY and NCrystal processing paths are intentionally separate at their backend-specific stages.

The production paths converge at an intermediate:

> **ENDF containing S(alpha, beta) — MF7**

That intermediate is then processed through the universal ACE-building path:

```text
NJOY
  │
  └──> ENDF + S(α,β) / MF7 ──┐
                              │
NCrystal                      ├──> UniversalAcePipeline ──> Final ACE
  │                           │
  └──> ENDF + S(α,β) / MF7 ──┘
```

### Liquid routes and entry points

For NJOY, Core samples the full vibrational spectrum. The LEAPR writer
normally exports a compact continuum plus sampled oscillator centroids and
weights. The direct Processing and Analyzer APIs can also request a full
tabulated `ND=0` export.

NCrystal Analyzer workflows use the native captured liquid scattering law.
The direct Processing batch API can also write experimental liquid MF7 from
that capture for downstream NJOY processing. The TOML CLI currently admits
explicit liquid workflows through the NJOY backend.

### Critical architectural rule

`UniversalAcePipeline` is the shared ACE entry point. Deck formatting is owned by `processing/deck_builders/ace_builder.py`.

Backend-specific implementations should not duplicate ACE-generation logic.

### Principle

> Backend-specific processing converges at the MF7 intermediate representation before entering the universal ACE pipeline.

---

## 4. Dependency Direction

TARA enforces a strict, downward-flowing dependency hierarchy to prevent circular imports and domain leakage. Higher-level orchestration and execution domains depend on lower-level foundational domains, but lower-level domains remain completely isolated from the layers above them.

### The Dependency Graph

```text
             ┌─────────────────────────────────┐
             │          application            │ (CLI Orchestrator)
             └─────────────────────────────────┘
                     │                 │
                     ▼                 ▼
         ┌───────────────┐     ┌───────────────┐
         │   analyzer    │     │  processing   │ (Evaluation & Production)
         └───────────────┘     └───────────────┘
                     │                 │
                     ▼                 ▼
             ┌─────────────────────────────────┐
             │          input_output           │ (Translation Border)
             └─────────────────────────────────┘
                               │
                               ▼
             ┌─────────────────────────────────┐
             │              core               │ (Single Source of Truth)
             └─────────────────────────────────┘
```

The exact import graph must follow the existing implementation and integration guides.

Do not introduce new dependencies merely because they make a local implementation easier.

Before creating a cross-domain dependency, determine which domain owns the responsibility and whether an existing contract already provides the required interface.

---

## 5. Domain Boundaries

The following boundaries are architectural constraints:

| Domain         | Owns                                          | Must not own                  |
| -------------- | --------------------------------------------- | ----------------------------- |
| `application`  | CLI, configuration, orchestration             | Physics, parsing, mathematics |
| `core`         | schemas, mathematics, sampling                | Files, CLI, external formats  |
| `input_output` | translation, parsing, serialization           | Physics calculations          |
| `analyzer`     | statistical/physics evaluation                | Production generation         |
| `processing`   | production pipeline, backends, ACE generation | Core mathematical definitions |

When a proposed change appears to cross these boundaries, first determine whether the responsibility has been placed in the wrong domain.

---

## 6. Integration Guides

The five domains use **domain-specific integration-guide filenames** (not a single literal `INTEGRATION_GUIDE.md` filename):

| Domain | Integration guide |
|---|---|
| `application/` | `application/APP_INTEGRATION_GUIDE.md` |
| `core/` | `core/CORE_INTEGRATION_GUIDE.md` |
| `input_output/` | `input_output/IO_INTEGRATION_GUIDE.md` |
| `analyzer/` | `analyzer/BACKEND_INTEGRATION_GUIDE.md` |
| `processing/` | `processing/PROCESSING_INTEGRATION_GUIDE.md` |

These documents are authoritative developer and AI integration guides for their respective domains.

They define:

* domain-specific architecture
* extension points
* contracts
* implementation rules
* invariants
* integration procedures
* prohibited architectural patterns

### AI development rule

Before modifying a domain, its **actual integration guide listed above** must be read and treated as the domain's authoritative implementation contract.

For changes involving multiple domains, read the integration guide for **every affected domain** before making architectural changes.

---

## 7. Change & Maintenance Protocol

The regression suite mirrors the production domains. Place a new bug
reproduction beside the cases that protect the same contract. Use
`tests/test_integration/` for checks that cross domain boundaries, and
`tests/fixtures/` for physical input files. Use descriptive case IDs when
parametrizing the same contract, and preserve distinct bug triggers as
separately identifiable cases.

See the [test suite guide](tests/README.md) for ownership, commands, and the
migration map. Compare collected cases and outcomes when reorganizing the
suite. A matching total count alone cannot establish case preservation.

Before implementing a change:

1. Identify the owning domain.
2. Confirm that the **complete current repository** is uploaded and inspectable before activating the Scientific Python Developer skill or making repository-wide source changes.
3. Read the relevant named integration guide(s) in Section 6.
4. Inspect the existing source implementation, tests, contracts, and dependency relationships.
5. Determine whether an existing abstraction already supports the requested behavior.
6. Make the smallest architectural change that satisfies the requirement.
7. Preserve domain boundaries and scientific invariants.
8. Update and run appropriate tests; distinguish mocked, synthetic, and real-backend verification.
9. Update documentation; when changing an architectural contract, update the relevant integration guide.

### Do not

* move logic between domains for convenience
* duplicate existing domain logic
* bypass established interfaces
* introduce hidden cross-domain coupling
* place physics calculations in translation layers
* place file-format knowledge in `core`
* duplicate ACE-building logic in backend-specific processors

---

## 8. AI Collaboration Contract

When working on TARA, prioritize:

1. **Architectural correctness**
2. **Preservation of domain boundaries**
3. **Compatibility with existing contracts**
4. **Minimal, targeted changes**
5. **Testability**
6. **Clarity and maintainability**

Do not redesign the architecture unless explicitly requested.

When uncertain about where functionality belongs, inspect the relevant integration guides and existing dependency structure before introducing new code.

When a requested change conflicts with an existing architectural constraint, identify the conflict explicitly before changing the constraint.

---

## 9. Project Context Structure

A **generated AI-context bundle** may be shared with the following structure:

```text
tara_ai_context/
    PROJECT_OVERVIEW.md
    PROJECT_MAP.md
    test_overview.md
    notebook_overview.md
    NOTEBOOK_OVERVIEW.md
    codebooks/
        core.md
        analyzer.md
        processing.md
        input_output.md
        application.md
        tests.md
    notebooks/
        00_getting_started.md
        01_spectrum_sampling.md
        02_scattering_laws.md
        03_ace_generation.md
        04_analysis.md
        05_validation.md
```

The notebook filenames above illustrate the directory names recorded in the current repository map; the generator discovers available directories rather than requiring this fixed list. Each generated notebook Markdown file combines **all notebooks under one top-level `notebooks/` directory**, including nested subdirectories. It records source paths and retains Markdown/code cells in cell order while excluding saved execution outputs and attachments. The original `.ipynb` files remain authoritative. On each full run, the generator **deletes and rebuilds the existing `tara_ai_context/` directory**; do not keep manually edited content exclusively inside that generated directory.

- `PROJECT_OVERVIEW.md` is the global architectural contract, copied from the repository-root source document.
- `PROJECT_MAP.md` is a generated **snapshot** of repository paths and counts, not proof of the present checkout.
- `test_overview.md` summarizes pytest functions, classes, and docstrings; `codebooks/tests.md` contains their complete Python source.
- `notebook_overview.md` is a generated inventory of notebook paths, first Markdown headings, and cell counts.
- `NOTEBOOK_OVERVIEW.md` is a copied **human-maintained workflow guide**, whose authoritative source is the repository-root `NOTEBOOK_OVERVIEW.md`.
- Domain and notebook codebooks provide generated source context; they may be incomplete or stale relative to the current checkout.
- The **named** domain integration guides listed in Section 6 supply detailed implementation contracts.
- The original repository root and a generated `tara_ai_context/` export are **different locations**. A path shown in the repository tree does not imply that a similarly named file in the export is the same physical file or version.

### Complete-project prerequisite

The **Scientific Python Developer skill must be used only when the entire current TARA project is uploaded and inspectable**. Verify the actual source tree, project/dependency metadata, configuration and integration guides, and existing tests. Generated context documents, codebooks, patches, notebooks, and individual files **do not by themselves satisfy this condition**. Virtual environments, caches, generated outputs, and secrets are not required.

When the repository is incomplete, limit activity to architecture discussion, documentation review, or other explicitly scoped work that does not claim full-project verification. Identify missing project material before project-wide implementation. Documentation review alone does not activate the skill.

---

## 10. Source of Truth Hierarchy

When resolving ambiguity, use this priority order:

1. Existing executable behavior and tests
2. The relevant **named** domain integration guide(s) in Section 6
3. Existing domain interfaces and schemas
4. `PROJECT_OVERVIEW.md`
5. `PROJECT_MAP.md`
6. General implementation preference

Do not override an established project contract based solely on a generic coding convention. Tests document present behavior, but passing tests alone does not establish physical correctness; check scientific assumptions and real-backend outputs where relevant.

---

## 11. Scientific Python Coding and Documentation Standards

These standards apply to new and modified Python code **when repository-wide development is authorized and the complete-project prerequisite is met**:

- **NumPy-style docstrings are mandatory** for new or modified production functions, methods, and classes. Include applicable `Parameters`, `Returns`/`Yields`, `Raises`, and `Notes` sections. State units, array shapes, normalization, physical assumptions, and side effects when relevant.
- **Inline comments must be purposeful and PEP 8-compatible**. Explain scientific rationale, numerical approximations, non-obvious constraints, and compatibility decisions; do not narrate obvious Python syntax.
- **Prefer small, single-responsibility functions.** Treat functions exceeding approximately **10 executable lines** as candidates for decomposition into coherent named helpers. This is a review heuristic, not a hard maximum: cohesive numerical algorithms and cases where splitting would obscure meaning are valid exceptions.
- Preserve public contracts, existing scientific invariants, deterministic sampling behavior, and clearly assigned domain ownership. Avoid large monolithic solutions and cross-domain convenience workarounds.
- Keep bug fixes, feature work, and unrelated refactors separate unless the user explicitly requests combined changes.

---

## 12. Generated Context and Verification Status

`PROJECT_MAP.md` declares itself automatically generated by `generate_ai_context.py`; **repository inventories and statistics must be updated in that generator, not maintained by hand**. A reviewed or annotated copy can document corrections temporarily, but regeneration may overwrite them. The generator itself must be inspected before making permanent generator changes.

Distinguish **present in the repository map**, **present in uploaded source**, **implemented in the current checkout**, and **validated with the real scientific backend**. These are different evidence levels. Do not infer installation or scientific validation from filenames, codebooks, patches, or notebooks alone.

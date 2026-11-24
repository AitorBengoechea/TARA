<div align="center">
  <img src="docs/logo/logo_2.jpg" alt="TARA Logo" width="600">
</div>

# TARA: Thermal Scattering Law Uncertainty Analysis

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)

**TARA** (Thermal scattering law uncertainty Analysis using RAndom files) is a robust computational framework developed for the uncertainty quantification (UQ) of thermal neutron scattering laws. 

Designed for the nuclear data and reactor physics community, TARA utilizes stochastic sampling to generate physically consistent randomized thermal scattering files. By applying mathematically rigorous perturbations to amplitude and peak positions, the code allows researchers to propagate continuous-energy thermal scattering uncertainties into downstream Monte Carlo or deterministic transport solvers.

[TARA documentation](docs/index.md) · [Upstream repository](https://github.com/vaibhavjaiswal-asnr/TARA)

---

## 🚀 Scientific Capabilities

Unlike standard data perturbation tools, TARA is built to strictly enforce the physical constraints of nuclear data:

* **Topologically Valid Sampling:** Employs oversampling and dynamic bounding to ensure all randomized files respect physical conservation laws.
* **B-Spline Amplitude Perturbation:** Introduces smooth, artifact-free amplitude variations without non-physical discontinuities.
* **Diffeomorphic Peak Shifting:** Utilizes diffeomorphic mapping with precise Jacobian corrections to continuously shift energy or momentum transfer peaks while conserving reaction probabilities and phase-space integrals.
* **Coupled & Uncoupled UQ:** Supports both independent parameter sampling and the introduction of synthetic correlations between amplitude cross-sections and peak positions.

---

## Requirements

- **Python 3.13 or newer** and pip.
- **Git and a C++ toolchain** to build the NCrystal Liquids plugin from its Git repository. On Linux, use a working C++ compiler and build tools. On Windows, install Visual Studio Build Tools with the **Desktop development with C++** workload and a Windows SDK. Run the installation from the **x64 Native Tools Command Prompt for VS** with your Python environment active.
- **NJOY** for ACE generation. Set `[execution].njoy_executable` in the workflow TOML to an executable name on your PATH or an absolute path. The ACE workflow uses NJOY for the ACE conversion even when NCrystal generates the scattering law.

The Python dependencies are declared in [`pyproject.toml`](pyproject.toml) and installed by pip.

| Dependency | Requirement |
| --- | --- |
| NumPy, pandas, SciPy, joblib | `numpy`, `pandas`, `scipy`, `joblib` |
| NCrystal | `ncrystal>=4.0` |
| NCrystal Liquids | [`ncrystal-plugin-Liquids` from GitHub](https://github.com/dddijulio/ncrystal-plugin-Liquids) |
| ENDF, numerical, and storage packages | `endf-parserpy`, `numba`, `scikit-learn`, `tables` |
| Tests | `pytest` via the optional `test` extra |

## Installation

From the TARA repository root, create a Python 3.13+ environment and activate it.

| Platform | Create and activate the environment |
| --- | --- |
| Linux | `python3.13 -m venv .venv`, then `source .venv/bin/activate` |
| Windows (Command Prompt) | `py -3.13 -m venv .venv`, then `.venv\Scripts\activate.bat` |

Install TARA and its dependencies from the repository root:

```bash
python -m pip install .
```

The standard installation builds Liquids from source. On Windows, use `where cl` and `where nmake` in the installation prompt if CMake reports either tool missing. After installation, run `nctool --plugins` to check that NCrystal lists `Liquids`.

If a workflow does not use the Liquids plugin, you can comment out its Git dependency under `[project].dependencies` in `pyproject.toml` before installing. Workflows that use NCrystal Liquids require the plugin at runtime.

## Quick start

Run TARA from Python, use a TOML configuration with the CLI, or pass sampling options directly on the command line.

**Run an ACE workflow from TOML.** Open the example configuration and set the input files, output directory, and NJOY executable for your environment. Paths in the configuration are relative to the TOML file.

```bash
python -m application.cli ace --config application/configs/ace_njoy_solid.toml --validate-only
python -m application.cli ace --config application/configs/ace_njoy_solid.toml
```

**Sample a LEAPR spectrum directly.** Replace `my_material.leapr` with your input file.

```bash
python -m application.cli my_material.leapr --shift-mev 1.0 --shift-pct 5.0 -n 3
```

See the [Application Integration Guide](application/APP_INTEGRATION_GUIDE.md) for the NJOY and NCrystal configuration examples.

## Tests

```bash
python -m pip install ".[test]"
python -m pytest -ra
```

Pytest reports which tests were skipped and why. Tests that run real NJOY or optional native integrations need the relevant executables, plugin, or inputs.

## Repository checks

| File | Purpose |
| --- | --- |
| [`.gitignore`](.gitignore) | Keeps local environments, caches, build artifacts, and `tara_output/` out of new commits. Test fixtures and notebook inputs remain eligible for tracking. |
| [`.pre-commit-config.yaml`](.pre-commit-config.yaml) | Checks Python, JSON, TOML, and YAML syntax and catches conflict markers before a local commit. |
| [`.github/workflows/test.yml`](.github/workflows/test.yml) | Runs pytest on Ubuntu with Python 3.13 when a pull request is opened, reopened, or updated. Tests requiring additional native tools or data may skip. |

Enable the local checks once per clone:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Contributing

Use [TARA Issues](https://github.com/vaibhavjaiswal-asnr/TARA/issues) to report a problem or discuss a change. To contribute, push the branch to your fork (`origin`). Then open a [pull request in upstream TARA](https://github.com/vaibhavjaiswal-asnr/TARA/pulls) using **Compare across forks**. Choose the intended upstream branch as the base and the branch in your fork as the compare branch. Describe the change and the tests that cover it.

Use paths relative to this README for files stored in the repository, such as `docs/index.md` and `application/APP_INTEGRATION_GUIDE.md`. On GitHub, these links open the file on the branch and repository you are viewing. Use full upstream URLs for shared project pages such as Issues and Pull Requests.

## Developer Guides: Extending the Framework

Start with the project overview to understand the architecture. Use the project map to locate a module, or the notebook overview to find and maintain examples.

| Document | What you’ll find |
| --- | --- |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | The architecture, workflow contracts, and responsibilities of each domain. |
| [PROJECT_MAP.md](PROJECT_MAP.md) | A repository inventory with the directory tree, domain statistics, and pointers to important files. |
| [NOTEBOOK_OVERVIEW.md](NOTEBOOK_OVERVIEW.md) | Notebook organization, responsibilities, and conventions for maintaining the examples. |

Use the integration guide for the domain where your change belongs.

| Guide | What it covers |
| --- | --- |
| [Core Integration Guide](core/CORE_INTEGRATION_GUIDE.md) | Spectrum data structures, mathematical transformations, and sampling algorithms. |
| [Input/Output Integration Guide](input_output/IO_INTEGRATION_GUIDE.md) | File parsing, material representation, and serialization for supported formats. |
| [Processing Integration Guide](processing/PROCESSING_INTEGRATION_GUIDE.md) | How NJOY and NCrystal process material inputs and sampled spectra, including backend execution and kernel generation. |
| [Backend Integration Guide](analyzer/BACKEND_INTEGRATION_GUIDE.md) | How the analyzer connects to backend runners and converts their results into shared analysis outputs. |
| [Application Integration Guide](application/APP_INTEGRATION_GUIDE.md) | CLI commands, TOML settings, and orchestration of user workflows. |

---

## 🏛 Acknowledgments & Funding

This work was supported by the European Union Horizon Europe programme through the **APRENDE** project (Grant Agreement No. 101061262).

<div align="center">
  <img src="docs/logo/aprende_logo.png" alt="TARA Logo" width="200">
</div>

The TARA project and TARA code are being developed at the **Autorité de Sûreté Nucléaire et de Radioprotection (ASNR)** as part of Task 4.2 (Thermal Neutron Scattering) within the APRENDE consortium.

## 📄 License

TARA is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

---

"""Read and validate application-owned ACE experiment configurations.

TOML and file-system conventions live in this outer application layer. Core
sampling and Processing continue to accept their existing Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any


class ApplicationConfigError(ValueError):
    """Raised when an application TOML file has invalid settings."""


_SCHEMA: dict[str, dict[str, type]] = {
    "workflow": {"backend": str},
    "material": {
        "base_endf_file": str,
        "temperature_k": float,
        "state": str,
        "leapr_file": str,
        "ncmat_file": str,
        "ncrystal_builtin": str,
        "atom_label": str,
        "elastic_mode": str,
    },
    "output": {"directory": str},
    "ensemble": {
        "n_samples": int,
        "include_reference": bool,
        "random_seed": int,
        "batch_size": int,
        "sampling_method": str,
    },
    "execution": {"n_jobs": int, "njoy_executable": str},
    "sampling": {
        "max_energy_shift_mev": float,
        "max_amplitude_shift_pct": float,
        "grid_partition_method": str,
        "automatic_partition_prominence_pct": float,
        "manual_energy_ranges": list,
        "domain_splitting_strategy": str,
        "correlation_strategy": str,
        "variable_coupling": str,
        "shared_grid_structure": bool,
        "freeze_origin": bool,
        "fast_amplitude": bool,
        "amplitude_method": str,
        "spatial_method": str,
        "auto_normalize": bool,
    },
    "diffusion_sampling": {
        "enabled": bool,
        "mode": str,
        "max_diffusivity_shift_pct": float,
        "max_weight_shift_pct": float,
        "weight_compensation": str,
    },
    "ace": {
        "max_energy_ev": float,
        "suffix": str,
        "nbin": int,
        "tolerance": float,
        "save_xsdir": bool,
        "mti": int,
        "mte": int,
    },
}
_REQUIRED = {
    "workflow": {"backend"},
    "material": {"base_endf_file", "temperature_k"},
    "output": {"directory"},
    "ensemble": {"n_samples", "include_reference", "random_seed", "batch_size"},
    "execution": {"n_jobs", "njoy_executable"},
    "sampling": set(_SCHEMA["sampling"]) - {
        "manual_energy_ranges", "automatic_partition_prominence_pct"
    },
    "diffusion_sampling": set(),
    "ace": set(_SCHEMA["ace"]) - {"mti", "mte"},
}


@dataclass(frozen=True)
class AceApplicationConfig:
    """Validated application input for one ACE generation run.

    Parameters
    ----------
    file_path : Path
        Path to the input TOML file.
    sections : dict[str, dict[str, Any]]
        Validated configuration tables keyed by section name.

    Notes
    -----
    Attributes
    ----------
    file_path : pathlib.Path
        Absolute path to the source TOML document.
    sections : dict of str to dict
        Validated, independent section dictionaries.
    """

    file_path: Path
    sections: dict[str, dict[str, Any]]

    def section(self, name: str) -> dict[str, Any]:
        """Return a copy of one validated configuration section."""
        return dict(self.sections[name])

    @property
    def backend(self) -> str:
        """Return the requested ACE generation backend."""
        return self.sections["workflow"]["backend"]

    def path(self, section: str, key: str) -> Path:
        """Resolve a file-system setting relative to the TOML file.

        Parameters
        ----------
        section, key : str
            Section and setting identifying the path.
        """
        raw = self.sections[section][key]
        return (self.file_path.parent / raw).resolve()


def _check_type(section: str, key: str, value: Any, expected: type) -> None:
    """Verify exact TOML scalar types and reject nonfinite floats."""
    if expected is float:
        good = type(value) in (int, float) and math.isfinite(value)
    else:
        good = type(value) is expected
    if not good:
        raise ApplicationConfigError(
            f"[{section}].{key} must be {expected.__name__}; got {value!r}."
        )


def _number(value: Any, location: str, *, positive: bool = False) -> None:
    """Validate a finite physical or numerical control."""
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ApplicationConfigError(f"{location} must be a finite number.")
    if positive and value <= 0:
        raise ApplicationConfigError(f"{location} must be strictly positive.")


def load_ace_config(filename: str | Path) -> AceApplicationConfig:
    """Load one ACE TOML file, validating keys, types, and combinations.

    Parameters
    ----------
    filename : str or pathlib.Path
        TOML input path. Relative file paths within the TOML are resolved
        from the directory containing that file, not the current directory.

    Returns
    -------
    AceApplicationConfig
        Validated application settings.

    Raises
    ------
    FileNotFoundError
        The TOML document does not exist.
    ApplicationConfigError
        The file has invalid keys, types, or combinations.
    """
    source = Path(filename).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ACE configuration not found: {source}")
    try:
        with source.open("rb") as stream:
            raw = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ApplicationConfigError(f"Malformed TOML in {source}: {exc}") from exc

    unknown_sections = set(raw) - set(_SCHEMA)
    missing_sections = (set(_SCHEMA) - {"diffusion_sampling"}) - set(raw)
    if unknown_sections or missing_sections:
        raise ApplicationConfigError(
            f"Invalid TOML sections: unknown={sorted(unknown_sections)}, "
            f"missing={sorted(missing_sections)}."
        )

    sections: dict[str, dict[str, Any]] = {}
    for group, fields in _SCHEMA.items():
        if group == "diffusion_sampling" and group not in raw:
            continue
        values = raw[group]
        if not isinstance(values, dict):
            raise ApplicationConfigError(f"[{group}] must be a TOML table.")
        unknown = set(values) - set(fields)
        missing = _REQUIRED[group] - set(values)
        if unknown or missing:
            raise ApplicationConfigError(
                f"Invalid [{group}] settings: unknown={sorted(unknown)}, "
                f"missing={sorted(missing)}."
            )
        for key, value in values.items():
            _check_type(group, key, value, fields[key])
        sections[group] = dict(values)

    # Optional diffusion controls are entirely absent from historical files.
    from core.config import DiffusionSamplingConfig, TaraConfigError
    raw_diffusion = sections.get('diffusion_sampling', {})
    diffusion_values = {
        'enabled': raw_diffusion.get('enabled', False),
        'mode': raw_diffusion.get('mode', 'diffusivity_only'),
        'max_diffusivity_shift_pct': raw_diffusion.get('max_diffusivity_shift_pct', 0.0),
        'max_weight_shift_pct': raw_diffusion.get('max_weight_shift_pct', 0.0),
        'weight_compensation': raw_diffusion.get('weight_compensation'),
    }
    try:
        diffusion = DiffusionSamplingConfig(**diffusion_values)
    except TaraConfigError as exc:
        raise ApplicationConfigError(f"[diffusion_sampling] {exc}") from exc
    if diffusion.enabled and (sections['material'].get('state', 'solid') != 'liquid'
                              or sections['workflow']['backend'] != 'njoy'):
        raise ApplicationConfigError('Diffusion sampling requires NJOY liquid material.')
    if (diffusion.enabled and sections['sampling']['max_energy_shift_mev'] > 0
            and not sections['sampling']['freeze_origin']):
        raise ApplicationConfigError(
            'Diffusion with vibrational energy shifts requires [sampling].freeze_origin = true.'
        )
    if 'diffusion_sampling' in sections:
        sections['diffusion_sampling'] = dict(raw_diffusion)
    # Older ACE TOML files remain valid and preserve the historical 1% rule.
    sections["sampling"].setdefault("automatic_partition_prominence_pct", 1.0)

    workflow, material, ensemble = (sections[name] for name in
                                    ("workflow", "material", "ensemble"))
    backend = workflow["backend"]
    if backend not in ("njoy", "ncrystal"):
        raise ApplicationConfigError("[workflow].backend must be 'njoy' or 'ncrystal'.")
    if material.get("state", "solid") not in ("solid", "liquid"):
        raise ApplicationConfigError("[material].state must be solid or liquid.")
    if material.get("state") == "liquid" and "mti" not in sections["ace"]:
        raise ApplicationConfigError("Liquid ACE requires explicit [ace].mti.")
    if material.get("state") == "liquid" and not sections["sampling"]["auto_normalize"]:
        raise ApplicationConfigError("Liquid spectrum sampling requires [sampling].auto_normalize = true.")
    for key in ("mti", "mte"):
        if key in sections["ace"] and (sections["ace"][key] <= 0 if key == "mti" else sections["ace"][key] < 0):
            raise ApplicationConfigError(f"[ace].{key} is not a valid reaction identifier.")
    if backend == "njoy":
        if not material.get("leapr_file"):
            raise ApplicationConfigError("NJOY requires [material].leapr_file.")
        for invalid in ("ncmat_file", "ncrystal_builtin", "atom_label", "elastic_mode"):
            if invalid in material:
                raise ApplicationConfigError(f"NJOY does not accept [material].{invalid}.")
        if not ensemble["include_reference"]:
            raise ApplicationConfigError("NJOY always generates sample_000; include_reference must be true.")
    else:
        sources = sum(bool(material.get(k)) for k in ("ncmat_file", "ncrystal_builtin"))
        if sources != 1:
            raise ApplicationConfigError(
                "NCrystal requires exactly one of [material].ncmat_file and ncrystal_builtin."
            )
        if "leapr_file" in material:
            raise ApplicationConfigError("NCrystal does not accept [material].leapr_file.")
        if not material.get("atom_label") or not material.get("elastic_mode"):
            raise ApplicationConfigError("NCrystal requires atom_label and elastic_mode.")

    for group, key in (("material", "temperature_k"), ("ace", "max_energy_ev"),
                       ("ace", "tolerance")):
        _number(sections[group][key], f"[{group}].{key}", positive=True)
    for key in ("max_energy_shift_mev", "max_amplitude_shift_pct"):
        value = sections["sampling"][key]
        _number(value, f"[sampling].{key}")
        if value < 0:
            raise ApplicationConfigError(f"[sampling].{key} must be nonnegative.")
    if not any(sections["sampling"][k] > 0 for k in
               ("max_energy_shift_mev", "max_amplitude_shift_pct")) and not diffusion.enabled:
        raise ApplicationConfigError("At least one sampling perturbation bound must be positive.")
    prominence_pct = sections["sampling"]["automatic_partition_prominence_pct"]
    if not 0.0 < prominence_pct <= 100.0:
        raise ApplicationConfigError(
            "[sampling].automatic_partition_prominence_pct must be in (0, 100]."
        )
    ensemble.setdefault("sampling_method", "latin_hypercube")
    if ensemble["sampling_method"] not in ("latin_hypercube", "monte_carlo"):
        raise ApplicationConfigError(
            "[ensemble].sampling_method must be latin_hypercube or monte_carlo."
        )
    for key in ("n_samples", "batch_size"):
        if ensemble[key] <= 0:
            raise ApplicationConfigError(f"[ensemble].{key} must be positive.")
    if ensemble["random_seed"] < 0 or ensemble["random_seed"] >= 2**32:
        raise ApplicationConfigError("[ensemble].random_seed must be in [0, 2**32).")
    if sections["execution"]["n_jobs"] == 0 or sections["execution"]["n_jobs"] < -1:
        raise ApplicationConfigError("[execution].n_jobs must be -1 or a positive integer.")
    if sections["ace"]["nbin"] <= 0:
        raise ApplicationConfigError("[ace].nbin must be positive.")
    if not sections["ace"]["suffix"].strip():
        raise ApplicationConfigError("[ace].suffix cannot be empty.")
    if sections["sampling"]["amplitude_method"] not in ("fast", "precise"):
        raise ApplicationConfigError("[sampling].amplitude_method must be fast or precise.")
    if sections["sampling"]["spatial_method"] not in ("diffeomorphic", "crossfade"):
        raise ApplicationConfigError("[sampling].spatial_method must be diffeomorphic or crossfade.")
    for key, allowed in {
        'grid_partition_method': ('automatic', 'manual'),
        'domain_splitting_strategy': ('fixed', 'deterministic', 'random'),
        'correlation_strategy': ('independent', 'random_group'),
        'variable_coupling': ('independent', 'coupled'),
    }.items():
        if sections['sampling'][key] not in allowed:
            raise ApplicationConfigError(
                f"[sampling].{key} must be one of {allowed}."
            )
    ranges = sections["sampling"].get("manual_energy_ranges", [])
    for idx, item in enumerate(ranges):
        if not (isinstance(item, list) and len(item) == 2):
            raise ApplicationConfigError(f"[sampling].manual_energy_ranges[{idx}] must be [start, end].")
        a, b = item
        _number(a, f"manual_energy_ranges[{idx}][0]")
        _number(b, f"manual_energy_ranges[{idx}][1]")
        if a < 0 or b <= a:
            raise ApplicationConfigError(f"manual_energy_ranges[{idx}] must satisfy 0 <= start < end.")
    if sections["sampling"]["grid_partition_method"] == "manual" and not ranges:
        raise ApplicationConfigError("Manual grid partition requires manual_energy_ranges.")
    if sections["sampling"]["grid_partition_method"] != "manual" and ranges:
        raise ApplicationConfigError("manual_energy_ranges requires grid_partition_method='manual'.")
    for label in ("base_endf_file", "leapr_file", "ncmat_file"):
        if label in material and not material[label].strip():
            raise ApplicationConfigError(f"[material].{label} cannot be empty.")
    if not sections["output"]["directory"].strip():
        raise ApplicationConfigError("[output].directory cannot be empty.")
    if not sections["execution"]["njoy_executable"].strip():
        raise ApplicationConfigError("[execution].njoy_executable cannot be empty.")
    return AceApplicationConfig(file_path=source, sections=sections)

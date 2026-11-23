"""Fresh-process native extraction; invoked only by the Processing backend."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Set, Tuple

import NCrystal as NC
import numpy as np

from input_output.ncrystal.liquids import parse_custom_liquids
from processing.backends.ncmat_isolation import isolate_ncmat_dyninfo


def _verify_child_library_environment(expected_library_path: str) -> None:
    """Verify NCrystal 4.4.6 ABI and ensure no mixed shared libraries loaded.

    Parameters
    ----------
    expected_library_path : str
        Expected path to ``libNCrystal.so`` from the adapter manifest.

    Raises
    ------
    RuntimeError
        If the NCrystal version or mapped shared libraries do not match.
    """
    if NC.__version__ != "4.4.6":
        raise RuntimeError("Unverified NCrystal ABI; expected 4.4.6")

    from NCrystal._locatelib import get_libpath_and_namespace

    loaded_path, namespace = get_libpath_and_namespace()
    expected = Path(expected_library_path).resolve()
    if namespace or Path(loaded_path).resolve() != expected:
        raise RuntimeError("Child loaded a different NCrystal library")

    mapped = {
        Path(line.split(maxsplit=5)[-1].strip()).resolve()
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
        if "/" in line and "libNCrystal" in line
    }
    if mapped != {expected}:
        raise RuntimeError("Mixed or unexpected NCrystal shared libraries in child")


def _validate_liquid_dyninfos(
    info: Any,
    settings: Dict[str, Any],
    request: Dict[str, Any],
) -> Tuple[Dict[str, Any], Set[str]]:
    """Validate NCMAT dynamic blocks against ``@CUSTOM_LIQUIDS`` and target settings.

    Parameters
    ----------
    info : NCrystal.Info
        Loaded NCrystal material info object.
    settings : dict of str to LiquidAtomSettings
        Parsed ``@CUSTOM_LIQUIDS`` entries keyed by atom label.
    request : dict of str to Any
        Child capture request parameters.

    Returns
    -------
    dynamics : dict of str to Any
        NCrystal ``DynInfo`` objects keyed by display label.
    active : set of str
        Validated set of active scattering species labels.

    Raises
    ------
    ValueError
        If labels are duplicated, missing, have unsupported dynamic types,
        or the requested target lacks a captured liquid VDOS kernel.
    """
    dynamics = {di.atomData.displayLabel(): di for di in info.dyninfos}
    if len(dynamics) != len(info.dyninfos):
        raise ValueError("Repeated dynamics labels need an explicit component mapping")
    if not settings.keys() <= dynamics.keys():
        raise ValueError("Custom liquid label absent from dynamics")

    for label, di in dynamics.items():
        expected = NC.Info.DI_VDOS if label in settings else NC.Info.DI_FreeGas
        if not isinstance(di, expected):
            raise ValueError(
                f"Unsupported liquid dynamics for {label}: {type(di).__name__}"
            )

    active = set(request["active_species"] or dynamics)
    if not active <= dynamics.keys() or request["target"] not in active:
        raise ValueError("Unknown active species or inactive analysis target")
    if request["target"] not in settings:
        raise ValueError("Analysis target must have a captured liquid VDOS kernel")
    return dynamics, active


def _load_captured_arrays(
    root: Path,
    filename: Path,
    record: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load captured ``(alpha, beta, sab)`` and component XS from binary or JSON.

    Parameters
    ----------
    root : pathlib.Path
        Sandbox directory containing capture artifacts.
    filename : pathlib.Path
        Path to ``sab_XXXX.json``.
    record : dict of str to Any
        Parsed JSON metadata record.

    Returns
    -------
    alpha, beta, sab, component_xs : tuple of numpy.ndarray
        Captured grids, 2-D ``S[beta, alpha]`` table, and 1-D cross sections.

    Raises
    ------
    ValueError
        If array dimensions or layouts are inconsistent.
    """
    if "binary_file" in record:
        n_alpha = int(record["n_alpha"])
        n_beta = int(record["n_beta"])
        payload = np.fromfile(root / record["binary_file"], dtype=np.float64)
        expected_size = n_alpha + n_beta + n_alpha * n_beta
        if payload.size != expected_size:
            raise ValueError("Corrupted binary SABData payload size")
        alpha = payload[:n_alpha]
        beta = payload[n_alpha : n_alpha + n_beta]
        sab = payload[n_alpha + n_beta :].reshape((n_beta, n_alpha))
    else:
        alpha = np.asarray(record["alpha"], dtype=float)
        beta = np.asarray(record["beta"], dtype=float)
        sab = np.asarray(record["sab"], dtype=float)

    stem_id = filename.stem.replace("sab_", "xs_")
    xs_bin = root / f"{stem_id}.bin"
    if xs_bin.is_file():
        component_xs = np.fromfile(xs_bin, dtype=np.float64)
    else:
        xs_txt = root / f"{stem_id}.txt"
        component_xs = np.loadtxt(xs_txt, ndmin=1)
    return alpha, beta, sab, component_xs


def _match_captured_components(
    root: Path,
    files: list[Path],
    settings: Dict[str, Any],
    dynamics: Dict[str, Any],
    active: Set[str],
    energies: np.ndarray,
    requested_temperature: float,
    arrays: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Match captured kernels by mass and bound XS and accumulate weighted XS.

    Parameters
    ----------
    root : pathlib.Path
        Sandbox directory.
    files : list of pathlib.Path
        Captured ``sab_XXXX.json`` files.
    settings : dict of str to LiquidAtomSettings
        Parsed ``@CUSTOM_LIQUIDS`` settings.
    dynamics : dict of str to Any
        NCrystal ``DynInfo`` mapping.
    active : set of str
        Active species set.
    energies : numpy.ndarray
        Incident energy grid in eV.
    requested_temperature : float
        Requested temperature in kelvin.
    arrays : dict of str to numpy.ndarray
        Output array dictionary to populate in place.

    Returns
    -------
    components : dict of str to dict
        Matched component metadata keyed by element label.
    sum_xs : numpy.ndarray
        Full-material accumulated scattering cross section.
    selected_xs : numpy.ndarray
        Active-species accumulated scattering cross section.
    """
    components: Dict[str, Any] = {}
    matched: Set[str] = set()
    sum_xs = np.zeros_like(energies)
    selected_xs = np.zeros_like(energies)

    for index, filename in enumerate(files):
        record = json.loads(filename.read_text(encoding="utf-8"))
        if record.get("layout") != "sab[beta_index][alpha_index]":
            raise ValueError("Unknown capture layout")

        candidates = [
            label
            for label in settings
            if label not in matched
            and np.isclose(
                dynamics[label].atomData.averageMassAMU(),
                record["element_mass_amu"],
                rtol=1e-10,
                atol=0,
            )
            and np.isclose(
                dynamics[label].atomData.scatteringXS(),
                record["bound_xs_barn"],
                rtol=1e-10,
                atol=0,
            )
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Ambiguous capture identity; refuse to infer label from order"
            )
        label = candidates[0]
        matched.add(label)
        di = dynamics[label]

        if not np.isclose(
            record["temperature_K"], requested_temperature, rtol=1e-12
        ):
            raise ValueError("Captured temperature differs from request")

        alpha, beta, sab, component_xs = _load_captured_arrays(
            root, filename, record
        )
        if component_xs.shape != energies.shape:
            raise ValueError("Incomplete component cross sections")

        prefix = f"component_{index}"
        arrays[f"{prefix}_alpha"] = alpha
        arrays[f"{prefix}_beta"] = beta
        arrays[f"{prefix}_sab"] = sab

        weighted = component_xs * di.fraction
        sum_xs += weighted
        if label in active:
            selected_xs += weighted

        components[label] = {
            "prefix": prefix,
            "mass_amu": record["element_mass_amu"],
            "temperature": record["temperature_K"],
            "bound_xs": record["bound_xs_barn"],
            "fraction": di.fraction,
            "suggested_emax": record["suggested_emax_eV"],
        }

    return components, sum_xs, selected_xs


def _evaluate_freegas_contributions(
    root: Path,
    text: str,
    freegas_labels: Set[str],
    active: Set[str],
    temperature: float,
    energies: np.ndarray,
    sum_xs: np.ndarray,
    selected_xs: np.ndarray,
) -> None:
    """Evaluate non-custom free-gas species and add their weighted cross sections.

    Parameters
    ----------
    root : pathlib.Path
        Sandbox directory.
    text : str
        Full NCMAT text.
    freegas_labels : set of str
        Species modeled as ``DI_FreeGas``.
    active : set of str
        Active species set.
    temperature : float
        Material temperature in kelvin.
    energies : numpy.ndarray
        Incident neutron energy grid in eV.
    sum_xs : numpy.ndarray
        Full-material cross-section accumulator updated in place.
    selected_xs : numpy.ndarray
        Active-species cross-section accumulator updated in place.
    """
    if not freegas_labels:
        return
    stripped = re.sub(r"(?ms)^@CUSTOM_LIQUIDS\s*\n.*?(?=^@|\Z)", "", text)
    for label in freegas_labels:
        free_text = isolate_ncmat_dyninfo(stripped, {label})
        free_path = root / f"freegas_{label}.ncmat"
        free_path.write_text(free_text, encoding="utf-8")
        scatter = NC.createScatter(f"{free_path};temp={temperature:.17g}K")
        free_xs = scatter.xsect(energies)
        sum_xs += free_xs
        if label in active:
            selected_xs += free_xs


def _load_energy_grid(root: Path) -> np.ndarray:
    """Load the incident cross-section energy grid from binary or text file.

    Parameters
    ----------
    root : pathlib.Path
        Sandbox directory.

    Returns
    -------
    numpy.ndarray
        1-D incident energy grid in eV.
    """
    bin_path = root / "energies.bin"
    if bin_path.is_file():
        return np.fromfile(bin_path, dtype=np.float64)
    return np.loadtxt(root / "energies.txt", ndmin=1)


def capture(request_path: str | Path) -> None:
    """Capture kernels and native cross sections for one isolated material.

    Parameters
    ----------
    request_path : str or pathlib.Path
        JSON request path inside the sample sandbox.

    Raises
    ------
    RuntimeError
        If ABI checks, plugin selection, kernel counts, or cross-section
        closure verification fail.
    ValueError
        If NCMAT dynamics or captured array values are invalid.
    """
    req_file = Path(request_path)
    root = req_file.parent
    request = json.loads(req_file.read_text(encoding="utf-8"))
    _verify_child_library_environment(request["library_path"])

    material = root / "material.ncmat"
    text = material.read_text(encoding="utf-8")
    settings = {item.atom: item for item in parse_custom_liquids(text)}
    cfg = (
        f"{material};temp={request['temperature']:.17g}K;"
        f"vdoslux={request['vdoslux']}"
    )

    info = NC.createInfo(cfg)
    dynamics, active = _validate_liquid_dyninfos(info, settings, request)
    energies = _load_energy_grid(root)

    mat = NC.load(cfg)
    if "LiquidsModel" not in mat.scatter.getName():
        raise RuntimeError("LiquidsModel was not selected")
    # Disable further interception before evaluating free-gas helper NCMATs.
    os.environ.pop("NCRYSTAL_SAB_DUMP_DIR", None)

    files = sorted(root.glob("sab_*.json"))
    if len(files) != len(settings):
        raise RuntimeError(
            f"Expected {len(settings)} kernels, captured {len(files)}"
        )

    arrays: Dict[str, np.ndarray] = {
        "xs_energy": energies,
        "xs_absorption": mat.absorption.xsect(energies),
    }
    components, sum_xs, selected_xs = _match_captured_components(
        root=root,
        files=files,
        settings=settings,
        dynamics=dynamics,
        active=active,
        energies=energies,
        requested_temperature=float(request["temperature"]),
        arrays=arrays,
    )
    _evaluate_freegas_contributions(
        root=root,
        text=text,
        freegas_labels=dynamics.keys() - settings.keys(),
        active=active,
        temperature=float(request["temperature"]),
        energies=energies,
        sum_xs=sum_xs,
        selected_xs=selected_xs,
    )

    native_xs = mat.scatter.xsect(energies)
    if not np.allclose(sum_xs, native_xs, rtol=1e-9, atol=1e-10):
        raise RuntimeError(
            "Captured components do not reconstruct native material scattering"
        )

    arrays["xs_scattering"] = selected_xs
    for key in ("xs_scattering", "xs_absorption"):
        if not np.isfinite(arrays[key]).all() or np.any(arrays[key] < 0):
            raise ValueError(f"Invalid {key}")

    target = dynamics[request["target"]]
    arrays["backend_energy_grid"] = target.vdos_egrid_expanded
    arrays["backend_pdos"] = target.vdos_density
    np.savez(root / "result.npz", **arrays)

    metadata = {
        "components": components,
        "vdos_analysis": target.analyseVDOS(),
        "provenance": {
            "ncrystal_version": NC.__version__,
            "plugins": NC.plugins.browsePlugins(),
            "plugin_hashes": {
                path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for _, path, kind in NC.plugins.browsePlugins()
                if path and kind == "dynamic" and Path(path).is_file()
            },
            "material_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "active_species": sorted(active),
            "target": request["target"],
            "vdoslux": request["vdoslux"],
            "capture": "SABScatter constructor",
            "xs_scope": "selected species per original material atom",
            "absorption_scope": "whole material per atom",
            "sab_scope": "target species per atom",
            "raw_alpha_convention": "neutron mass",
            "analyzer_alpha_convention": "target atom mass",
            "beta_convention": "(Eout-Ein)/(k_B*T)",
            "components": {
                label: {
                    key: value
                    for key, value in component.items()
                    if key != "prefix"
                }
                for label, component in components.items()
            },
            "thermodynamics_scope": "target solid-like VDOS only",
            "kinematic_normalization": "per unit target free scattering cross section",
        },
    }
    (root / "result.json").write_text(json.dumps(metadata), encoding="utf-8")


if __name__ == "__main__":
    capture(sys.argv[1])
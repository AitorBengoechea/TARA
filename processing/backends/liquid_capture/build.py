"""Build a process-local ELF adapter against the installed NCrystal library."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Dict, Tuple


_REQUIRED_SYMBOLS = (
    "_ZN8NCrystal10SABScatterC1EONS_7SABDataERKSt6vectorIdSaIdEE",
    "_ZN8NCrystal10SABScatterC2EONS_7SABDataERKSt6vectorIdSaIdEE",
)


def _query_ncrystal_config(ncrystal_config: str, flag: str) -> str:
    """Query the installed ``ncrystal-config`` CLI for a single build property.

    Parameters
    ----------
    ncrystal_config : str
        Executable name or path for ``ncrystal-config``.
    flag : str
        Property key passed to ``--show`` (e.g., ``"version"``, ``"libpath"``).

    Returns
    -------
    str
        Stripped standard output from ``ncrystal-config``.

    Raises
    ------
     subprocess.CalledProcessError
        If the configuration command fails.
    """
    return subprocess.check_output(
        [ncrystal_config, "--show", flag],
        text=True,
    ).strip()


def _verify_ncrystal_abi_and_symbols(
    ncrystal_config: str,
) -> Tuple[str, Path, Path]:
    """Verify Linux ELF platform, NCrystal 4.4.6 parity, and exported C++ symbols.

    Parameters
    ----------
    ncrystal_config : str
        Executable name or path for ``ncrystal-config``.

    Returns
    -------
    version : str
        Verified NCrystal version string (``"4.4.6"``).
    include_dir : pathlib.Path
        C++ header directory for NCrystal.
    library_path : pathlib.Path
        Shared library path for ``libNCrystal.so``.

    Raises
    ------
    RuntimeError
        If the OS is not Linux, versions mismatch, the library is namespaced,
        or the legacy ``SABScatter`` constructor symbols are absent.
    """
    import NCrystal as NC

    if platform.system() != "Linux":
        raise RuntimeError("Liquid kernel capture currently requires Linux/ELF.")

    version = _query_ncrystal_config(ncrystal_config, "version")
    if version != "4.4.6" or NC.__version__ != version:
        raise RuntimeError(
            "This capture adapter is verified for NCrystal 4.4.6 only; "
            "Python and C++ installations must match."
        )

    include_dir = Path(_query_ncrystal_config(ncrystal_config, "includedir"))
    library_path = Path(_query_ncrystal_config(ncrystal_config, "libpath"))

    from NCrystal._locatelib import get_libpath_and_namespace

    runtime_library, namespace = get_libpath_and_namespace()
    if namespace or Path(runtime_library).resolve() != library_path.resolve():
        raise RuntimeError(
            "Python and C++ must use the same unnamespaced NCrystal library"
        )

    symbols = subprocess.check_output(["nm", "-D", str(library_path)], text=True)
    if not all(symbol in symbols for symbol in _REQUIRED_SYMBOLS):
        raise RuntimeError(
            "NCrystal does not export the required SABScatter constructors"
        )

    return version, include_dir, library_path


def _compute_build_identity(
    version: str,
    source_path: Path,
    library_path: Path,
) -> Tuple[Dict[str, object], str]:
    """Build the deterministic cache manifest and directory key.

    Parameters
    ----------
    version : str
        Verified NCrystal version string.
    source_path : pathlib.Path
        Path to ``sabtap.cc``.
    library_path : pathlib.Path
        Path to ``libNCrystal.so``.

    Returns
    -------
    identity : dict of str to object
        Manifest dictionary describing the source and library hashes.
    cache_key : str
        24-character hexadecimal cache key derived from ``identity``.
    """
    identity: Dict[str, object] = {
        "schema": 1,
        "ncrystal_version": version,
        "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "library_path": str(library_path.resolve()),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return identity, digest[:24]


def _compile_shared_adapter(
    compiler: str,
    include_dir: Path,
    source_path: Path,
    library_path: Path,
    target_path: Path,
    identity: Dict[str, object],
) -> None:
    """Compile ``sabtap.cc`` atomically inside the locked cache directory.

    Parameters
    ----------
    compiler : str
        C++ compiler executable name or path.
    include_dir : pathlib.Path
        NCrystal C++ header directory.
    source_path : pathlib.Path
        Path to ``sabtap.cc``.
    library_path : pathlib.Path
        Path to ``libNCrystal.so``.
    target_path : pathlib.Path
        Destination path for ``libsabtap.so``.
    identity : dict of str to object
        Build identity manifest to persist alongside ``target_path``.
    """
    with tempfile.TemporaryDirectory(dir=target_path.parent) as tmp:
        output = Path(tmp) / target_path.name
        command = [
            compiler,
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-shared",
            "-I",
            str(include_dir),
            str(source_path),
            str(library_path),
            "-ldl",
            f"-Wl,-rpath,{library_path.parent}",
            "-o",
            str(output),
        ]
        subprocess.run(
            command,
            check=True,
            timeout=120,
            capture_output=True,
            text=True,
        )
        manifest = dict(identity)
        manifest["adapter_sha256"] = hashlib.sha256(
            output.read_bytes()
        ).hexdigest()
        target_path.with_suffix(".json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        os.replace(output, target_path)


def build_capture_library(
    cache_dir: str | Path,
    *,
    ncrystal_config: str = "ncrystal-config",
    compiler: str = "c++",
) -> Path:
    """Build once per source/library identity, serializing concurrent builds.

    Parameters
    ----------
    cache_dir : str or pathlib.Path
        Writable directory for cached adapter builds.
    ncrystal_config : str, default="ncrystal-config"
        Installed NCrystal configuration utility.
    compiler : str, default="c++"
        C++ compiler executable.

    Returns
    -------
    pathlib.Path
        Compiled shared library path with an adjacent ``.json`` manifest.

    Raises
    ------
    RuntimeError
        If platform, version, namespace, or constructor symbol checks fail.
    """
    version, include_dir, library_path = _verify_ncrystal_abi_and_symbols(
        ncrystal_config
    )
    import fcntl

    source_path = Path(__file__).with_name("sabtap.cc")
    identity, key = _compute_build_identity(version, source_path, library_path)
    directory = Path(cache_dir).resolve() / key
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "libsabtap.so"

    # Serialize concurrent worker builds using an advisory file lock.
    with (directory / "build.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            _compile_shared_adapter(
                compiler=compiler,
                include_dir=include_dir,
                source_path=source_path,
                library_path=library_path,
                target_path=target,
                identity=identity,
            )
    return target
"""Build a process-local ELF adapter against the installed NCrystal library."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile


def build_capture_library(cache_dir, *, ncrystal_config="ncrystal-config", compiler="c++"):
    """Build once per source/library identity, serializing concurrent builds.

    Parameters
    ----------
    cache_dir : path-like
        Writable directory for adapter builds.
    ncrystal_config, compiler : str
        Installed NCrystal configuration utility and C++ compiler.

    Returns
    -------
    pathlib.Path
        Adapter shared library with an adjacent compatibility manifest.
    """
    import NCrystal as NC

    if platform.system() != "Linux":
        raise RuntimeError("Liquid kernel capture currently requires Linux/ELF.")
    import fcntl

    version, include, library = _resolve_capture_runtime(NC, ncrystal_config)
    source = Path(__file__).with_name("sabtap.cc")
    identity = {
        "schema": 1, "ncrystal_version": version,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "library_path": str(library.resolve()),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(cache_dir).resolve() / key
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "libsabtap.so"
    with (directory / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            _compile_capture_adapter(target, source, include, library, compiler, identity)
    return target


def _query_ncrystal_config(executable, key):
    """Read one installed NCrystal build property.

    Parameters
    ----------
    executable : str
        Path or command name of the NCrystal configuration utility.
    key : str
        Requested property name.

    Returns
    -------
    str
        Property value without trailing whitespace.
    """
    return subprocess.check_output([executable, "--show", key], text=True).strip()


def _resolve_capture_runtime(NC, ncrystal_config):
    """Check ABI compatibility between Python and the compiled NCrystal library.

    Parameters
    ----------
    NC : module
        Imported Python NCrystal module.
    ncrystal_config : str
        Path or command name of the NCrystal configuration utility.

    Returns
    -------
    tuple
        NCrystal version, include directory, and library path.
    """
    version = _query_ncrystal_config(ncrystal_config, "version")
    if version != "4.4.6" or NC.__version__ != version:
        raise RuntimeError("This capture adapter is verified for NCrystal 4.4.6 only; "
                           "Python and C++ installations must match.")
    include = Path(_query_ncrystal_config(ncrystal_config, "includedir"))
    library = Path(_query_ncrystal_config(ncrystal_config, "libpath"))
    from NCrystal._locatelib import get_libpath_and_namespace
    runtime_library, namespace = get_libpath_and_namespace()
    if namespace or Path(runtime_library).resolve() != library.resolve():
        raise RuntimeError("Python and C++ must use the same unnamespaced NCrystal library")
    symbols = subprocess.check_output(["nm", "-D", str(library)], text=True)
    required = [f"_ZN8NCrystal10SABScatterC{i}EONS_7SABDataERKSt6vectorIdSaIdEE" for i in (1, 2)]
    if not all(symbol in symbols for symbol in required):
        raise RuntimeError("NCrystal does not export the required SABScatter constructors")
    return version, include, library


def _compile_capture_adapter(target, source, include, library, compiler, identity):
    """Compile and publish one capture adapter under the caller's build lock.

    Parameters
    ----------
    target, source : pathlib.Path
        Adapter destination and C++ source file.
    include, library : pathlib.Path
        NCrystal headers and shared library used by the compiler.
    compiler : str
        C++ compiler executable.
    identity : dict
        Build manifest to extend with the compiled adapter digest.
    """
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        output = Path(tmp) / target.name
        subprocess.run([compiler, "-std=c++17", "-O2", "-fPIC", "-shared",
                        "-I", str(include), str(source), str(library), "-ldl",
                        f"-Wl,-rpath,{library.parent}", "-o", str(output)],
                       check=True, timeout=120, capture_output=True, text=True)
        identity["adapter_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()

        # Write the identity before publishing the shared object under the lock.
        target.with_suffix(".json").write_text(json.dumps(identity), encoding="utf-8")
        os.replace(output, target)

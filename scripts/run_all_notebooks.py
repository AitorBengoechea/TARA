#!/usr/bin/env python3
"""Execute TARA notebooks as if using Run All and report runtime errors.

Dependencies: pip install nbclient nbformat ipykernel
Examples:
  python scripts/run_all_notebooks.py notebooks
  python scripts/run_all_notebooks.py notebooks --engine njoy --report njoy.json
  python scripts/run_all_notebooks.py notebooks --engine ncrystal --report ncrystal.json

Each notebook uses a fresh kernel and runs in its own directory. Notebook files
are never overwritten. A successful run means execution reached the last cell
without an exception; no scientific or numerical results are judged.

Engine-specific runs use the notebook's AVAILABLE_ENGINE and ENGINE settings.
Older single-engine notebooks are recognized by a filename containing njoy or
ncrystal as a separate word. For exceptions, notebook metadata can declare
{"tara": {"engines": ["njoy"]}} or {"tara": {"engines": ["ncrystal"]}}.
Engine-independent notebooks run once, in the NCrystal job.
"""
import argparse
import ast
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError
except ImportError:
    print("Missing dependencies. Run: python -m pip install nbclient nbformat ipykernel", file=sys.stderr)
    sys.exit(2)

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints",
    "node_modules", "site-packages", "tara_output", "outputs",
}
ENGINES = ("njoy", "ncrystal")
ENGINE_SETTING = re.compile(r"^(?:ENGINE|AVAILABLE_ENGINE)\s*[:=]", re.MULTILINE)


def _engine_settings(notebook):
    """Find top-level engine assignments in notebook code cells."""
    settings = {}
    for cell in notebook.cells:
        # Only parse configuration cells. Other cells may use IPython magics.
        if cell.cell_type != "code" or not ENGINE_SETTING.search(cell.source):
            continue
        for statement in ast.parse(cell.source).body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
            elif isinstance(statement, ast.AnnAssign):
                target = statement.target
            else:
                continue
            if isinstance(target, ast.Name) and target.id in ("ENGINE", "AVAILABLE_ENGINE"):
                if target.id in settings:
                    raise ValueError(f"Found more than one top-level {target.id} assignment.")
                settings[target.id] = (cell, statement.value)
    return settings


def _supported_engines(notebook, path, settings):
    """Resolve explicit notebook options before falling back to filename conventions."""
    declared = notebook.metadata.get("tara", {}).get("engines")
    if declared is None and "AVAILABLE_ENGINE" in settings:
        _, options = settings["AVAILABLE_ENGINE"]
        declared = ast.literal_eval(options)
    if declared is not None:
        if (
            not isinstance(declared, (list, tuple))
            or not declared
            or any(engine not in ENGINES for engine in declared)
        ):
            raise ValueError("Notebook engine options must list njoy, ncrystal, or both.")
        return tuple(dict.fromkeys(declared))

    if "ENGINE" in settings:
        _, selector = settings["ENGINE"]
        if isinstance(selector, ast.Constant) and selector.value in ENGINES:
            return (selector.value,)
        raise ValueError("Declare AVAILABLE_ENGINE or metadata.tara.engines for this ENGINE selector.")

    names = re.split(r"[_-]+", path.stem.lower())
    tagged = tuple(engine for engine in ENGINES if engine in names)
    if len(tagged) == 2:
        raise ValueError("Declare metadata.tara.engines for this ambiguous notebook filename.")
    return tagged


def _replace_engine(cell, value, engine):
    """Set the engine in memory and preserve source lines for tracebacks."""
    # AST columns count UTF-8 bytes, including any non-ASCII text on the line.
    lines = cell.source.encode("utf-8").splitlines(keepends=True)
    start = value.lineno - 1
    stop = value.end_lineno - 1
    prefix = lines[start][:value.col_offset]
    suffix = lines[stop][value.end_col_offset:]
    replacement = repr(engine).encode("ascii")
    if stop > start:
        replacement = b"(" + replacement + b"\n" * (stop - start) + b")"
    lines[start:stop + 1] = [prefix + replacement + suffix]
    cell.source = b"".join(lines).decode("utf-8")


def _prepare_engine(notebook, path, engine):
    """Select an engine for this run or return the reason to skip it."""
    settings = _engine_settings(notebook)
    supported = _supported_engines(notebook, path, settings)
    if len(supported) == 2 and "ENGINE" not in settings:
        raise ValueError("A dual-engine notebook must have a top-level ENGINE assignment.")
    if not supported:
        return None if engine == "ncrystal" else "Shared notebook runs in the NCrystal job."
    if engine not in supported:
        return f"Notebook supports {', '.join(supported)}."
    if "ENGINE" in settings:
        cell, value = settings["ENGINE"]
        if value is None:
            raise ValueError("The notebook's ENGINE assignment needs a value.")
        _replace_engine(cell, value, engine)
    return None


def discover(target: Path):
    if target.is_file():
        if target.suffix != ".ipynb":
            raise ValueError("The selected file must have a .ipynb extension")
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(target)
    return sorted(
        p for p in target.rglob("*.ipynb")
        if not any(part in EXCLUDED_DIRS or part.startswith(".") for part in p.relative_to(target).parts[:-1])
    )

def find_project_root(target: Path):
    start = target if target.is_dir() else target.parent
    for candidate in (start, *start.parents):
        if (candidate / "core").is_dir() and (candidate / "input_output").is_dir():
            return candidate
    return start

def run_one(path: Path, timeout: int, kernel_override: str | None, root: Path, engine: str | None = None):
    start = time.monotonic()
    result = {
        "notebook": str(path),
        "status": "ERROR",
        "seconds": None,
        "code_cells": 0,
        "cells_with_output": 0,
        "failed_cell": None,
        "skip_reason": None,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }
    try:
        nb = nbformat.read(str(path), as_version=4)
        result["code_cells"] = sum(cell.cell_type == "code" for cell in nb.cells)
        if engine is not None:
            skip_reason = _prepare_engine(nb, path, engine)
            if skip_reason:
                result["status"] = "SKIP"
                result["skip_reason"] = skip_reason
                return result
        kernel = kernel_override or nb.metadata.get("kernelspec", {}).get("name") or "python3"

        # Keep the notebook's own relative file paths working.
        # Expose the repository to imports even when cwd is a nested notebook folder.
        old_pythonpath = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = (
            str(root) + (os.pathsep + old_pythonpath if old_pythonpath else "")
        )
        try:
            client = NotebookClient(
                nb,
                timeout=timeout,
                kernel_name=kernel,
                allow_errors=False,
                resources={"metadata": {"path": str(path.parent)}},
            )
            client.execute()
        finally:
            if old_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old_pythonpath

        result["cells_with_output"] = sum(
            bool(cell.get("outputs")) for cell in nb.cells if cell.cell_type == "code"
        )
        result["status"] = "PASS"
    except (CellExecutionError, CellTimeoutError) as exc:
        # nbclient attaches error outputs to the executed notebook.
        if "nb" in locals():
            for index, cell in enumerate(nb.cells, start=1):
                if cell.cell_type != "code":
                    continue
                outputs = cell.get("outputs", [])
                if outputs:
                    result["cells_with_output"] += 1
                errors = [o for o in outputs if o.get("output_type") == "error"]
                if errors:
                    error = errors[-1]
                    result["failed_cell"] = index
                    result["error_type"] = error.get("ename")
                    result["error_message"] = error.get("evalue")
                    result["traceback"] = "\n".join(error.get("traceback", []))
                    break
        if result["error_type"] is None:
            result["error_type"] = type(exc).__name__
            result["error_message"] = str(exc)
            result["traceback"] = str(exc)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        result["seconds"] = round(time.monotonic() - start, 2)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="Repository folder or one .ipynb file")
    parser.add_argument("--timeout", type=int, default=600, help="Max seconds per cell (default: 600; -1 disables)")
    parser.add_argument("--kernel", default=None, help="Override the notebook's kernelspec name")
    parser.add_argument("--engine", choices=ENGINES, help="Select the backend for this run")
    parser.add_argument("--report", type=Path, default=Path("notebook_execution_report.json"))
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    try:
        notebooks = discover(target)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 2

    root = find_project_root(target)
    print(f"Found {len(notebooks)} notebook(s). Project root: {root}", flush=True)
    print("A PASS means Run All completed without an exception; outputs are not evaluated.\n", flush=True)
    results = []
    for i, path in enumerate(notebooks, start=1):
        print(f"[{i}/{len(notebooks)}] RUN  {path}", flush=True)
        result = run_one(path, args.timeout, args.kernel, root, args.engine)
        results.append(result)
        print(
            f"         {result['status']} ({result['seconds']}s, "
            f"{result['code_cells']} code cells, "
            f"{result['cells_with_output']} cells with output)",
            flush=True,
        )
        if result["status"] == "SKIP":
            print(f"         {result['skip_reason']}", flush=True)
        elif result["status"] == "ERROR":
            print(
                f"         Cell {result['failed_cell'] or '?'}: "
                f"{result['error_type']}: {result['error_message']}",
                flush=True,
            )

    passed = sum(r["status"] == "PASS" for r in results)
    failed = sum(r["status"] == "ERROR" for r in results)
    skipped = sum(r["status"] == "SKIP" for r in results)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target": str(target),
        "project_root": str(root),
        "engine": args.engine,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSUMMARY: {passed} PASS, {failed} ERROR, {skipped} SKIP")
    print(f"Report: {args.report.resolve()}")
    if failed:
        return 1
    if not passed:
        print("No notebooks ran for the selected engine.", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())

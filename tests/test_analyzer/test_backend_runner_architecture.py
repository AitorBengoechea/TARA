import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _module_ast(relative_path: str) -> ast.Module:
    """Module ast for the current workflow.

    Parameters
    ----------
    relative_path : str
        Filesystem location of the relative path.

    Returns
    -------
    ast.Module
        Calculated result in the declared return type.
    """
    source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    return ast.parse(source)


def _imported_names(tree: ast.Module) -> set[str]:
    """Imported names for the current workflow.

    Parameters
    ----------
    tree : ast.Module
        Parsed syntax or ENDF record tree to inspect.

    Returns
    -------
    set[str]
        Calculated result in the declared return type.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _class_methods(tree: ast.Module, class_name: str) -> set[str]:
    """Class methods for the current workflow.

    Parameters
    ----------
    tree : ast.Module
        Parsed syntax or ENDF record tree to inspect.
    class_name : str
        Name of the class to inspect in the syntax tree.

    Returns
    -------
    set[str]
        Calculated result in the declared return type.
    """
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"Class {class_name!r} not found")


def test_njoy_runner_is_thin_processing_adapter():
    """
    Test that Analyzer no longer owns NJOY deck or sandbox implementation.

    Verifies that NjoyRunner:
    - Does not import subprocess/tempfile/uuid/shutil.
    - Does not reintroduce a private ``_build_deck`` implementation.
    - Retains only the common Runner lifecycle methods.
    - Delegates NJOY analysis execution to Processing's NjoyProcessor.
    """
    tree = _module_ast("analyzer/runners/njoy_runner.py")
    imports = _imported_names(tree)
    methods = _class_methods(tree, "NjoyRunner")
    source = (PROJECT_ROOT / "analyzer/runners/njoy_runner.py").read_text(encoding="utf-8")

    assert imports.isdisjoint({"subprocess", "tempfile", "uuid", "shutil"})
    assert "_build_deck" not in methods
    assert {"__init__", "execute", "cleanup"}.issubset(methods)
    assert "processing.backends.njoy_processor" in source
    assert "self.processor.execute_analysis(" in source


def test_ncrystal_runner_is_thin_processing_adapter():
    """
    Test that Analyzer no longer owns NCrystal material/file preparation.

    Verifies that NCrystalRunner:
    - Does not import uuid/tempfile/os or NuclearData.
    - Does not construct NCMATComposer objects directly.
    - Delegates transient state creation and cleanup to NcrystalProcessor.
    """
    tree = _module_ast("analyzer/runners/ncrystal_runner.py")
    imports = _imported_names(tree)
    methods = _class_methods(tree, "NCrystalRunner")
    source = (PROJECT_ROOT / "analyzer/runners/ncrystal_runner.py").read_text(encoding="utf-8")

    assert imports.isdisjoint({"uuid", "tempfile", "os"})
    assert "NuclearData" not in source
    assert "NCMATComposer" not in source
    assert {"__init__", "execute", "cleanup"}.issubset(methods)
    assert "processing.backends.ncrystal_processor" in source
    assert "create_analysis_state" in source
    assert "cleanup_analysis_file" in source

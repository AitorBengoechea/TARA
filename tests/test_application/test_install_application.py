"""Safety and idempotence tests for the application-only installer."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ACE_CONFIGS = tuple(
    f'application/configs/ace_{backend}_{state}.toml'
    for backend in ('njoy', 'ncrystal')
    for state in ('solid', 'liquid')
)


def _installer():
    spec = importlib.util.spec_from_file_location('tara_install_application', ROOT / 'install_application.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _project(tmp_path):
    for file in ('application/cli.py', 'core/config.py',
                 'processing/pipelines/ace_pipeline.py'):
        target = tmp_path / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('original source\n')
    unrelated = tmp_path / 'notebooks' / 'original.ipynb'
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_text('untouched')
    return tmp_path


def test_dry_run_does_not_change_anything(tmp_path, monkeypatch):
    """Dry run identifies owned files without touching user data."""
    installer = _installer()
    root = _project(tmp_path)
    original = (root / 'application/cli.py').read_bytes()
    monkeypatch.setattr(installer, 'KNOWN_CLI_SHA256', {hashlib.sha256(original).hexdigest()})
    plan = installer.install(root, dry_run=True)
    assert any('application/cli.py' in line for line in plan)
    assert (root / 'application/cli.py').read_bytes() == original
    assert not (root / 'application/configuration.py').exists()
    assert not (root / '.tara_application_backups').exists()


def test_install_backups_and_repeat_is_noop(tmp_path, monkeypatch):
    """Applying twice is safe and does not alter unrelated notebooks."""
    installer = _installer()
    root = _project(tmp_path)
    old = (root / 'application/cli.py').read_bytes()
    monkeypatch.setattr(installer, 'KNOWN_CLI_SHA256', {hashlib.sha256(old).hexdigest()})
    installer.install(root)
    assert (root / 'application/cli.py').read_bytes() == (ROOT / 'application/cli.py').read_bytes()
    assert all((root / config).is_file() for config in ACE_CONFIGS)
    assert (root / 'tests/test_application/test_ace_workflow.py').exists()
    assert (root / 'notebooks/original.ipynb').read_text() == 'untouched'
    backups = list((root / '.tara_application_backups').glob('*/application/cli.py'))
    assert len(backups) == 1
    assert backups[0].read_bytes() == old
    installed = installer.install(root)
    assert all(line.startswith('unchanged') or line.lstrip().startswith('unchanged') for line in installed)
    assert len(list((root / '.tara_application_backups').glob('*/application/cli.py'))) == 1


def test_reject_modified_cli_without_touching_any_file(tmp_path):
    """Unknown CLI changes are never overwritten during migration."""
    installer = _installer()
    root = _project(tmp_path)
    with pytest.raises(RuntimeError, match='Unrecognized local application/cli.py'):
        installer.install(root)
    assert not (root / 'application/configuration.py').exists()


def test_reject_conflicting_new_application_files_atomically(tmp_path, monkeypatch):
    """Existing independently edited configuration files stop installation."""
    installer = _installer()
    root = _project(tmp_path)
    old = (root / 'application/cli.py').read_bytes()
    monkeypatch.setattr(installer, 'KNOWN_CLI_SHA256', {hashlib.sha256(old).hexdigest()})
    (root / 'application/configuration.py').write_text('independent changes')
    with pytest.raises(RuntimeError, match='Local edits already exist'):
        installer.install(root)
    assert (root / 'application/cli.py').read_bytes() == old
    assert not any((root / config).exists() for config in ACE_CONFIGS)

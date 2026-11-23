"""Safely install the TARA application/TOML update into an existing repository.

No scientific source modules or notebooks are changed. Unknown local changes to
``application/cli.py`` are rejected rather than overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import uuid


ROOT = Path(__file__).resolve().parent
FILES = (
    'application/__init__.py',
    'application/cli.py',
    'application/configuration.py',
    'application/ace_workflow.py',
    'application/APP_INTEGRATION_GUIDE.md',
    'application/configs/ace_njoy_solid.toml',
    'application/configs/ace_njoy_liquid.toml',
    'application/configs/ace_ncrystal_solid.toml',
    'application/configs/ace_ncrystal_liquid.toml',
    'tests/fixtures/ace_njoy_water_v2.example.toml',
    'tests/test_application/test_configuration.py',
    'tests/test_application/test_cli.py',
    'tests/test_application/test_ace_workflow.py',
    'tests/test_application/test_install_application.py',
)

# Hashes of the two previous application/cli.py source-codebook versions.
# We also recognize newline variants of each when validating the original.
KNOWN_CLI_SHA256 = {
    "383e2c68140bb2bfc140247a812ccef158e7f83543d4da3a99cac66ef46664cd",
    "7bd5463a7393cc3feefe4caa4935c8548fdcb5009461f78aff54f9697de5383f",
    "881ec681d9da205be45459450b31c9406788aa979fbb254954b78dc19be71a2b",
    "89010cddf2a0428b987e0be43936dc67ab2ec3bbf55c47e78644cbf4e22bfd10",
    "9a25562710b6e8db208c48d83a9bdb79028b6292b28497f9899f9f38815b59fd",
    "ba17d76f0c41c3d57fffe653095287cd1639bed99ac10c5e9da8f6f27ca39ed9",
    "d33050de6423627bd24f968795213211ccd8b0a45e0da0487b129261f9c675c2",
    "e0719e20e93c25241611d5334d00ee9a3f585757522fc67b70e2ae20e12969e5",
    "e2cdaf7d6ad5034e5e6b6ef245ff617cb536d0aef56ea90654508c4eceb91133",
    "f5a064f5dc0609bb835393fa4de70922adbb769e4cce38b83ae911b803f03b69",
}


def _sha(data: bytes) -> str:
    """Compute a source-file fingerprint."""
    return hashlib.sha256(data).hexdigest()


def install(project_root: Path, dry_run: bool = False) -> list[str]:
    """Install only the files owned by the application update.

    Parameters
    ----------
    project_root : pathlib.Path
        Existing TARA project directory.
    dry_run : bool, optional
        Inspect proposed updates without touching the project.

    Returns
    -------
    list of str
        Human-readable description of required changes.

    Raises
    ------
    FileNotFoundError
        The target is not a recognizable TARA project.
    RuntimeError
        A locally modified source would otherwise be overwritten.
    """
    root = project_root.expanduser().resolve()
    for required in ('application/cli.py', 'core/config.py',
                     'processing/pipelines/ace_pipeline.py'):
        if not (root / required).is_file():
            raise FileNotFoundError(f'Not a TARA project: missing {root / required}')

    actions = []
    for relative in FILES:
        source = ROOT / relative
        target = root / relative
        if not source.is_file():
            raise FileNotFoundError(f'Installer source missing: {source}')
        if not target.exists():
            actions.append(('add', relative))
        elif target.is_file() and target.read_bytes() == source.read_bytes():
            actions.append(('unchanged', relative))
        elif relative == 'application/cli.py':
            # The historical CLI is overwritten only for previously recognized
            # versions; this protects newer local modifications.
            old = target.read_bytes()
            if _sha(old) not in KNOWN_CLI_SHA256:
                raise RuntimeError(
                    f'Unrecognized local application/cli.py at {target}; '
                    'not overwriting. Compare the supplied CLI and merge manually.'
                )
            actions.append(('replace', relative))
        elif relative == 'application/APP_INTEGRATION_GUIDE.md':
            actions.append(('replace', relative))
        else:
            raise RuntimeError(
                f'Local edits already exist at {target}; not overwriting. '
                'Merge this file manually or restore its earlier version.'
            )

    messages = [f'{kind:>9}: {name}' for kind, name in actions]
    pending = [(kind, name) for kind, name in actions if kind != 'unchanged']
    if dry_run or not pending:
        return messages

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = root / '.tara_application_backups' / f'{timestamp}_{uuid.uuid4().hex[:8]}'
    for kind, name in pending:
        target = root / name
        if kind == 'replace':
            destination = backup / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, destination)

    # All compatibility checks and backups complete before any replacement.
    for kind, name in pending:
        data = (ROOT / name).read_bytes()
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='wb', dir=destination.parent, prefix='.tara_install_', delete=False
        ) as temp:
            temp.write(data)
            temp_path = Path(temp.name)
        try:
            os.replace(temp_path, destination)
        finally:
            temp_path.unlink(missing_ok=True)
    if backup.exists():
        messages.append(f'  backups: {backup}')
    return messages


def main() -> None:
    """Run the source installer from the command line."""
    parser = argparse.ArgumentParser(description='Install TARA application TOML workflows.')
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        messages = install(args.project_root, dry_run=args.dry_run)
    except (FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    for message in messages:
        print(message)
    print('DRY RUN: no files changed' if args.dry_run else 'Installation checked/applied.')


if __name__ == '__main__':
    main()

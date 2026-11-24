"""Shared test fixtures with optional backend dependencies."""

import pytest


@pytest.fixture(scope="session")
def liquids_plugin():
    """Require a registered NCrystal Liquids plugin for native tests."""
    ncrystal = pytest.importorskip("NCrystal")
    if not any(name == "Liquids" for name, _, _ in ncrystal.plugins.browsePlugins()):
        pytest.skip("NCrystal Liquids plugin is not registered in this environment.")
    return ncrystal

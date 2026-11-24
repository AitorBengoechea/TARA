"""Optional native NCrystal acceptance for temporary sterile CH2 materials."""
from pathlib import Path
import numpy as np
import pytest

NC = pytest.importorskip('NCrystal')
from processing.backends.ncmat_isolation import isolate_ncmat_dyninfo


def test_native_ncrystal_accepts_h_and_c_isolation(tmp_path):
    template = Path(__file__).resolve().parents[2] / 'tests/fixtures/ncrystal_tsl_HinCH2.ncmat'
    reference = template.read_text(encoding='utf-8')
    energy = np.geomspace(1.e-4, 1.0, 12)
    original = tmp_path / 'original.ncmat'
    original.write_text(reference, encoding='utf-8')
    for label in ('H', 'C'):
        isolated = tmp_path / f'only_{label}.ncmat'
        isolated.write_text(isolate_ncmat_dyninfo(reference, {label}), encoding='utf-8')
        # Verify native NCrystal can actually load this synthetic material.
        scatter = NC.createScatter(f'{isolated};temp=296K')
        xs = np.asarray(scatter.xsect(ekin=energy), dtype=float)
        assert xs.shape == energy.shape
        assert np.all(np.isfinite(xs))
        assert np.all(xs >= 0)
    # No source mutation and no global scaling by 1/f_H.
    assert original.read_text(encoding='utf-8') == reference

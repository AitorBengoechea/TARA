"""Small, independent LEAPR inputs for regression tests."""

import numpy as np

from input_output.leapr.core import LeaprFile


def make_water_leapr_template() -> LeaprFile:
    """Build a small LEAPR-format fixture using educational water weights.

    Notes
    -----
    The tiny numerical grids are *not* suitable for a physical NJOY evaluation.
    The fixture tests serialization and validation only.
    """
    leapr = LeaprFile('water.leapr')
    leapr.module_call_line = 'leapr\n'
    leapr.file_units_line = '20 24 /\n'
    leapr.title = "'educational H in water'/\n"
    leapr.run_control_params = {'ntempr': 1, 'iprint': 0, 'nphon': 8}
    leapr.material_info = {'mat': 1, 'za': 1001, 'isabt': 0, 'ilog': 0}
    leapr.principal_scatterer = {'awr': 1., 'spr': 20.436, 'npr': 2,
                                 'iel': 0, 'ncold': 0, 'nsk': 0}
    leapr.secondary_scatterer = {'nss': 1, 'b7': 1, 'aws': 16., 'sps': 4., 'mss': 1}
    leapr.grid_params = {'nalpha': 3, 'nbeta': 3, 'lat': 0}
    leapr.alpha_grid = np.array([0., 1., 2.])
    leapr.beta_grid = np.array([0., 1., 2.])
    block = {'temp': 296., 'delta': .01, 'ni': 4,
             'phonon_spectrum': np.array([0., 1., 2., 0.]),
             'twt': .05, 'c': 0., 'tbeta': .45, 'nd': 2,
             'oscillator_energies': np.array([.205, .480]),
             'oscillator_weights': np.array([.167, .333])}
    leapr.temp_blocks = [block]
    leapr.first_positive_energy_grid = np.arange(4) * .01
    leapr.first_positive_spectrum = block['phonon_spectrum']
    leapr.stop_card_found = True
    return leapr

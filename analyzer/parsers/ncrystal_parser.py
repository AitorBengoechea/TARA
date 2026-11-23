"""
Extraction adapter for the NCrystal backend.
"""
import numpy as np
import pandas as pd
from scipy.constants import physical_constants, k, hbar

import NCrystal as NC
from NCrystal.vdos import extractGn, extractKnl

from analyzer.parsers.base_tsl_parser import BaseTslParser
from analyzer.physics.data_models import PhysicsData, ThermodynamicsData, DifferentialData, IntegralData
from analyzer.physics.thermodynamics import (
    _with_debye_tail,
    calc_cv_numba, calc_entropy_numba, calc_zero_point_energy_numba,
    calc_free_energy_numba, calc_internal_energy_numba, calc_log_moment_numba
)
from analyzer.backends.ncrystal_engine import NCrystalEngine


class NCrystalParser(BaseTslParser):
    
    """Extract NCrystal observables into the shared Analyzer physics schema.
    """
    def parse(self, sample_index: int, engine: NCrystalEngine, **kwargs) -> PhysicsData:
        """Orchestrates the extraction of physical data into the unified schema.

        Extract only available backend observables and populate the shared PhysicsData schema.

        Parameters
        ----------
        sample_index : int
            Identifier of the stochastic sample being evaluated.
        engine : NCrystalEngine
            In-memory backend state from which the requested data are extracted.
        **kwargs : Any
            Additional options accepted by the current interface.

        Returns
        -------
        PhysicsData
            Calculated result in the declared return type.
        """
        thermo = self._extract_thermodynamics(engine)
        diff = DifferentialData()
        integ = IntegralData()
        
        self._extract_cross_sections(engine, integ, **kwargs)
        self._extract_kinematics(engine, diff, integ, **kwargs)
        
        return PhysicsData(sample_index=sample_index, thermo=thermo, diff=diff, integ=integ,
                           metadata=(dict(engine.liquid_data.provenance)
                                     if engine.liquid_data is not None else None))

    def _extract_thermodynamics(self, engine: NCrystalEngine) -> ThermodynamicsData:
        """Evaluates static scalars and properties using native calls and Numba kernels.

        Parameters
        ----------
        engine : NCrystalEngine
            In-memory backend state from which the requested data are extracted.

        Returns
        -------
        ThermodynamicsData
            Calculated result in the declared return type.
        """
        mat = engine.mat
        nuc_data = engine.nuc_data
        temp = engine.temperature
        target_lbl = engine.target_lbl
        mass_amu = engine.mass_amu

        if engine.liquid_data is not None:
            # The liquid engine supplies vibrational diagnostics from its VDOS.
            values = engine.liquid_data.vdos_analysis
            teff, msd = values['teff'], values['msd']
            pdos, E = _with_debye_tail(engine.backend_pdos, engine.backend_energy_grid)
        else:
            pdos, E = _with_debye_tail(engine.pdos, engine.energy_grid)
            atom_info = mat.info.findAtomInfo(target_lbl)
            teff = nuc_data.elements[target_lbl].teff[0]
            msd = atom_info.msd if hasattr(atom_info, 'msd') else 0.0
        
        b_factor = (8 * np.pi**2 * msd / 3) if msd else 0.0
        debye_waller = 0.0
        if msd:
            amu_kg = physical_constants['atomic mass constant'][0]
            msd_m2 = msd * (1e-10)**2 
            mass_kg = mass_amu * amu_kg
            debye_waller = msd_m2 * (2 * mass_kg * k * temp / hbar**2)
        
        return ThermodynamicsData(
            teff=teff, msd=msd, debye_waller=debye_waller, b_factor=b_factor,
            cv=calc_cv_numba(pdos, E, temp), entropy=calc_entropy_numba(pdos, E, temp),
            zero_point_energy=calc_zero_point_energy_numba(pdos, E),
            free_energy=calc_free_energy_numba(pdos, E, temp),
            internal_energy=calc_internal_energy_numba(pdos, E, temp),
            log_moment=calc_log_moment_numba(pdos, E)
        )

    def _extract_cross_sections(self, engine: NCrystalEngine, integ: IntegralData, **kwargs) -> None:
        """Evaluates 1D integrated cross-sections directly from the NCrystal scatter object.

        Parameters
        ----------
        engine : NCrystalEngine
            In-memory backend state from which the requested data are extracted.
        integ : IntegralData
            Container receiving integrated scattering quantities.
        **kwargs : Any
            Additional options accepted by the current interface.
        """
        if engine.liquid_data is not None:
            self._extract_liquid_cross_sections(engine, integ, **kwargs)
            return
        e_grid = kwargs.get('energy_grid', np.geomspace(1e-5, 10, 1000))
        mat = engine.mat

        if kwargs.get('calc_xs_total', True):
            integ.xs_total = pd.Series(mat.scatter.xsect(e_grid), index=e_grid)

        if kwargs.get('calc_xs_inelastic', True):
            if engine.scatter_inelastic is None:
                raise ValueError("NCrystal analysis state lacks an inelastic scattering handle.")
            integ.xs_inelastic = pd.Series(engine.scatter_inelastic.xsect(e_grid), index=e_grid)

        if kwargs.get('calc_xs_elastic', True):
            if engine.scatter_coherent_elastic is None or engine.scatter_incoherent_elastic is None:
                raise ValueError("NCrystal analysis state lacks elastic scattering handles.")
            integ.xs_coherent_elastic = pd.Series(
                engine.scatter_coherent_elastic.xsect(e_grid), index=e_grid,
            )
            integ.xs_incoherent_elastic = pd.Series(
                engine.scatter_incoherent_elastic.xsect(e_grid), index=e_grid,
            )

        if kwargs.get('calc_xs_absorption', False):
            integ.xs_absorption = pd.Series(mat.absorption.xsect(e_grid), index=e_grid)

    @staticmethod
    def _kernel_vdos(engine: NCrystalEngine) -> tuple[np.ndarray, np.ndarray]:
        """Return NCrystal-admissible, atom-specific kernel input arrays.

        The source-spaced arrays originate in Processing, which owns the
        template-specific NCMAT translation. Older manually constructed engine
        objects can use their regular energy/density arrays when admissible.
        Internal Core arrays must never be trimmed here to create a new output
        grid: that would reintroduce the spurious fine Debye spacing.

        Returns
        -------
        tuple of numpy.ndarray
            Energy in eV and corresponding PDOS in 1/eV.

        Raises
        ------
        ValueError
            If backend fields are incomplete or the grid violates NCrystal's
            strictly positive lower bound or equal-spacing requirements.
        """
        if (engine.backend_energy_grid is None) != (engine.backend_pdos is None):
            raise ValueError("Incomplete NCrystal backend VDOS state.")
        energy = (engine.backend_energy_grid if engine.backend_energy_grid is not None
                  else engine.energy_grid)
        density = (engine.backend_pdos if engine.backend_pdos is not None
                   else engine.pdos)
        energy = np.asarray(energy, dtype=np.float64)
        density = np.asarray(density, dtype=np.float64)
        if (energy.ndim != 1 or energy.size < 2 or density.shape != energy.shape
                or not np.isfinite(energy).all() or not np.isfinite(density).all()
                or np.any(density < 0.0) or np.any(np.diff(energy) <= 0.0)):
            raise ValueError("Invalid NCrystal backend VDOS arrays.")
        if energy[0] < 1.0e-5:
            raise ValueError(
                "NCrystal kernel VDOS starts below 1e-5 eV. "
                "Pass the source-spaced backend grid from NcrystalProcessor, "
                "not Core's E=0 Debye grid."
            )
        spacing = (energy[-1] - energy[0]) / (energy.size - 1)
        if not np.allclose(np.diff(energy), spacing, rtol=1e-6, atol=1e-12):
            raise ValueError("NCrystal kernel VDOS requires source-uniform spacing.")
        return energy, density

    def _extract_kinematics(self, engine: NCrystalEngine, diff: DifferentialData, integ: IntegralData, **kwargs) -> None:
        """Evaluates Differential Sab and passes it to the universal multi-Ein loop.

        Parameters
        ----------
        engine : NCrystalEngine
            In-memory backend state from which the requested data are extracted.
        diff : DifferentialData
            Container receiving differential scattering data.
        integ : IntegralData
            Container receiving integrated scattering quantities.
        **kwargs : Any
            Additional options accepted by the current interface.
        """
        mass_amu = engine.mass_amu
        temp = engine.temperature

        # NCrystal kernel extraction must receive the same source-spaced VDOS
        # that Processing wrote to NCMAT, not Core's dense E=0 Debye tail.
        # The full internal arrays remain available for thermodynamics.
        kinematic_flags = (
            'calc_sab', 'calc_dsf', 'calc_scattering_kernel',
            'calc_angular_distribution', 'calc_cosine_distribution',
            'calc_energy_distribution', 'calc_ratios',
        )
        kernel_requested = kwargs.get('calc_gn', False) or any(
            kwargs.get(flag, False) for flag in kinematic_flags
        )
        if not kernel_requested:
            return
        if engine.liquid_data is not None:
            self._extract_liquid_kinematics(engine, diff, integ, **kwargs)
            return
        E, pdos = self._kernel_vdos(engine)

        if kwargs.get('calc_gn', False):
            Etest, Gn = extractGn(vdos=(E, pdos), n=kwargs.get('gn_order', 3), mass_amu=mass_amu, temperature=temp)
            diff.gn = pd.Series(Gn, index=Etest)
            
        sab_df = None
        if any(kwargs.get(flag, False) for flag in kinematic_flags):
            sab_data = extractKnl(vdos=(E, pdos), mass_amu=mass_amu, temperature=temp, target_emax=5.0)
            sab_values = sab_data['sab'].reshape((len(sab_data['beta']), len(sab_data['alpha']))).T
            sab_df = pd.DataFrame(sab_values, columns=pd.Index(sab_data['beta'], name='beta'), index=pd.Index(sab_data['alpha'], name='alpha'))

        if sab_df is not None:
            # BaseTslParser validates the already signed NCrystal table and
            # stores the canonical full S(alpha,beta) representation.
            self._apply_universal_kinematics(sab_df, diff, integ, temp, mass_amu, **kwargs)

    @staticmethod
    def _extract_liquid_cross_sections(engine, integ, **kwargs):
        """Populate native scattering on the exact grid requested from Runner.

        The supported plugin combines continuous liquid kernels and free gas;
        their quasielastic peak is not a discrete elastic channel. Absorption
        retains the whole-material convention of the existing solid path.
        """
        state = engine.liquid_data
        requested = any(kwargs.get(key, default) for key, default in (
            ('calc_xs_total', True), ('calc_xs_inelastic', True),
            ('calc_xs_elastic', True), ('calc_xs_absorption', False)))
        if not requested:
            return
        grid = np.asarray(kwargs.get('energy_grid', state.xs_energy), dtype=float)
        if not np.array_equal(grid, state.xs_energy):
            raise ValueError('Pass the same energy_grid to the liquid Runner and Parser; '
                             'native cross sections are not silently interpolated.')
        if kwargs.get('calc_xs_total', True):
            integ.xs_total = pd.Series(state.xs_scattering, index=grid)
        if kwargs.get('calc_xs_inelastic', True):
            integ.xs_inelastic = pd.Series(state.xs_scattering, index=grid)
        if kwargs.get('calc_xs_elastic', True):
            integ.xs_coherent_elastic = pd.Series(np.zeros_like(grid), index=grid)
            integ.xs_incoherent_elastic = pd.Series(np.zeros_like(grid), index=grid)
        if kwargs.get('calc_xs_absorption', False):
            integ.xs_absorption = pd.Series(state.xs_absorption, index=grid)

    def _extract_liquid_kinematics(self, engine, diff, integ, **kwargs):
        """Map a completed plugin kernel into existing Analyzer kinematics.

        Alpha changes from neutron-mass to target-mass scaling. Signed beta
        and S values are retained; no symmetrization, clipping or VDOS kernel
        regeneration is performed. Gn remains a solid-like VDOS diagnostic.
        """
        state = engine.liquid_data
        kernel = state.kernels[engine.target_lbl]
        if kwargs.get('calc_gn', False):
            energy, density = self._kernel_vdos(engine)
            gn_energy, gn = extractGn(vdos=(energy, density),
                                     n=kwargs.get('gn_order', 3),
                                     mass_amu=engine.mass_amu, temperature=engine.temperature)
            diff.gn = pd.Series(gn, index=gn_energy)
        flags = self._requested_outputs(kwargs)
        if not kwargs.get('calc_sab', False) and not any(flags.values()):
            return
        if any(flags.values()):
            energies = np.atleast_1d(kwargs.get('ein_arr', [0.0253])).astype(float)
            if (not np.isfinite(energies).all() or np.any(energies <= 0)
                    or np.any(energies > kernel.suggested_emax)):
                raise ValueError('Incident energy exceeds the captured liquid kernel range '
                                 f'(0, {kernel.suggested_emax:g}] eV; increase vdoslux '
                                 'or request a supported range.')
        awr = kernel.mass_amu / physical_constants['neutron mass in u'][0]
        table = pd.DataFrame(kernel.sab.T,
                             index=pd.Index(kernel.alpha / awr, name='alpha'),
                             columns=pd.Index(kernel.beta, name='beta'))
        self._apply_universal_kinematics(table, diff, integ,
                                         kernel.temperature, kernel.mass_amu, **kwargs)

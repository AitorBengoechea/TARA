"""
Extraction adapter for the NCrystal backend.
"""
import numpy as np
import pandas as pd
from scipy.constants import physical_constants, k, hbar

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
    """Extract NCrystal observables into the shared Analyzer physics schema."""

    def parse(self, sample_index: int, engine: NCrystalEngine, **kwargs) -> PhysicsData:
        """Parse the requested observables from one evaluated sample.

        Parameters
        ----------
        sample_index : int
            Identifier of the stochastic sample being evaluated.
        engine : NCrystalEngine
            In-memory backend state from which the requested data are extracted.
        **kwargs : Any
            Observable flags. Set ``calc_thermodynamics=False`` for cross-section-only studies.

        Returns
        -------
        PhysicsData
            Requested observables and optional liquid-run provenance.
        """
        thermo = self._thermodynamics_for(engine, kwargs.get("calc_thermodynamics", True))
        diff = DifferentialData()
        integ = IntegralData()
        
        self._extract_cross_sections(engine, integ, **kwargs)
        self._extract_kinematics(engine, diff, integ, **kwargs)
        
        return PhysicsData(sample_index=sample_index, thermo=thermo, diff=diff, integ=integ,
                           metadata=(dict(engine.liquid_data.provenance)
                                     if engine.liquid_data is not None else None))

    def _extract_thermodynamics(self, engine: NCrystalEngine) -> ThermodynamicsData:
        """Calculate thermodynamic observables from the evaluated state.

        Parameters
        ----------
        engine : NCrystalEngine
            In-memory backend state from which the requested data are extracted.

        Returns
        -------
        ThermodynamicsData
            Native displacement data and spectrum-derived scalars.
        """
        teff, msd, pdos, energy = self._thermodynamic_inputs(engine)
        temperature = engine.temperature
        debye_waller, b_factor = self._displacement_factors(
            msd, engine.mass_amu, temperature,
        )
        return ThermodynamicsData(
            teff=teff,
            msd=msd,
            debye_waller=debye_waller,
            b_factor=b_factor,
            cv=calc_cv_numba(pdos, energy, temperature),
            entropy=calc_entropy_numba(pdos, energy, temperature),
            zero_point_energy=calc_zero_point_energy_numba(pdos, energy),
            free_energy=calc_free_energy_numba(pdos, energy, temperature),
            internal_energy=calc_internal_energy_numba(pdos, energy, temperature),
            log_moment=calc_log_moment_numba(pdos, energy),
        )

    @staticmethod
    def _thermodynamic_inputs(engine):
        """Read the native displacement data and corresponding spectrum.

        Parameters
        ----------
        engine : NCrystalEngine
            Evaluated material state.

        Returns
        -------
        tuple of float, float, numpy.ndarray, numpy.ndarray
            Effective temperature, mean-square displacement, PDOS, and energy.
        """
        if engine.liquid_data is not None:
            # Liquid diagnostics use the spectrum evaluated by the plugin.
            values = engine.liquid_data.vdos_analysis
            pdos, energy = _with_debye_tail(engine.backend_pdos, engine.backend_energy_grid)
            return values['teff'], values['msd'], pdos, energy

        pdos, energy = _with_debye_tail(engine.pdos, engine.energy_grid)
        atom_info = engine.mat.info.findAtomInfo(engine.target_lbl)
        teff = engine.nuc_data.elements[engine.target_lbl].teff[0]
        return teff, getattr(atom_info, 'msd', 0.0), pdos, energy

    @staticmethod
    def _displacement_factors(msd, mass_amu, temperature):
        """Calculate the Debye–Waller coefficient and crystallographic B factor.

        Parameters
        ----------
        msd : float
            Mean-square displacement in square ångströms.
        mass_amu : float
            Target mass in atomic mass units.
        temperature : float
            Material temperature in kelvin.

        Returns
        -------
        tuple of float
            Dimensionless Debye–Waller coefficient and B factor in square
            ångströms.
        """
        if not msd:
            return 0.0, 0.0

        # Convert displacement and mass to SI for the coefficient.
        msd_m2 = msd * (1e-10)**2
        mass_kg = mass_amu * physical_constants['atomic mass constant'][0]
        debye_waller = msd_m2 * (2 * mass_kg * k * temperature / hbar**2)
        b_factor = 8 * np.pi**2 * msd / 3
        return debye_waller, b_factor

    def _extract_cross_sections(self, engine: NCrystalEngine, integ: IntegralData, **kwargs) -> None:
        """Populate the requested integrated cross sections.

        Parameters
        ----------
        engine : NCrystalEngine
            In-memory backend state from which the requested data are extracted.
        integ : IntegralData
            Container receiving integrated scattering quantities.
        **kwargs : Any
            Cross-section flags and incident-energy grid.
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
            self._extract_elastic_cross_sections(engine, integ, e_grid)

        if kwargs.get('calc_xs_absorption', False):
            integ.xs_absorption = pd.Series(mat.absorption.xsect(e_grid), index=e_grid)

    @staticmethod
    def _extract_elastic_cross_sections(engine: NCrystalEngine, integ: IntegralData, energy: np.ndarray):
        """Evaluate both native elastic scattering channels on the same grid.

        Parameters
        ----------
        engine : NCrystalEngine
            State containing coherent and incoherent elastic scattering handles.
        integ : IntegralData
            Destination for the two elastic cross sections.
        energy : array_like
            Incident energies in eV.
        """
        coherent = engine.scatter_coherent_elastic
        incoherent = engine.scatter_incoherent_elastic
        if coherent is None or incoherent is None:
            raise ValueError("NCrystal analysis state lacks elastic scattering handles.")
        integ.xs_coherent_elastic = pd.Series(coherent.xsect(energy), index=energy)
        integ.xs_incoherent_elastic = pd.Series(incoherent.xsect(energy), index=energy)

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
        energy, density = (
            (engine.energy_grid, engine.pdos) if engine.backend_energy_grid is None
            else (engine.backend_energy_grid, engine.backend_pdos)
        )
        energy = np.asarray(energy, dtype=np.float64)
        density = np.asarray(density, dtype=np.float64)
        NCrystalParser._validate_kernel_grid(energy, density)
        return energy, density

    @staticmethod
    def _validate_kernel_grid(energy: np.ndarray, density: np.ndarray):
        """Check that the spectrum can be passed directly to NCrystal.

        Parameters
        ----------
        energy : numpy.ndarray
            Source energy coordinates in eV.
        density : numpy.ndarray
            Spectral density on the source grid.
        """
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

    def _extract_kinematics(self, engine: NCrystalEngine, diff: DifferentialData, integ: IntegralData, **kwargs) -> None:
        """Extract requested kernels and evaluate shared kinematic observables.

        Parameters
        ----------
        engine : NCrystalEngine
            In-memory backend state from which the requested data are extracted.
        diff : DifferentialData
            Container receiving differential scattering data.
        integ : IntegralData
            Container receiving integrated scattering quantities.
        **kwargs : Any
            Output flags and incident-energy settings.
        """
        want_gn = kwargs.get('calc_gn', False)
        want_distributions = any(self._requested_outputs(kwargs).values())
        want_sab = kwargs.get('calc_sab', False) or want_distributions
        if not (want_gn or want_sab):
            return

        if want_gn or engine.liquid_data is None:
            # Kernel calls use the source-spaced backend grid. The Core grid
            # can contain a much finer Debye tail for thermodynamics.
            vdos = self._kernel_vdos(engine)
        if want_gn:
            diff.gn = self._extract_gn(engine, vdos, kwargs.get('gn_order', 3))
        if not want_sab:
            return

        if engine.liquid_data is None:
            table = self._extract_solid_sab(engine, vdos)
            temperature, mass_amu = engine.temperature, engine.mass_amu
        else:
            kernel = self._liquid_kernel(engine)
            if want_distributions:
                self._validate_liquid_incident_energies(kernel, kwargs.get('ein_arr', [0.0253]))
            table = self._liquid_sab(kernel)
            temperature, mass_amu = kernel.temperature, kernel.mass_amu

        self._apply_universal_kinematics(table, diff, integ, temperature, mass_amu, **kwargs)

    @staticmethod
    def _extract_gn(engine, vdos, order):
        """Evaluate the vibrational multiphonon diagnostic.

        Parameters
        ----------
        engine : NCrystalEngine
            State providing the target mass and material temperature.
        vdos : tuple of numpy.ndarray
            Energy grid and density accepted by NCrystal.
        order : int
            Requested multiphonon order.

        Returns
        -------
        pandas.Series
            Diagnostic values indexed by energy.
        """
        energy, values = extractGn(
            vdos=vdos, n=order, mass_amu=engine.mass_amu, temperature=engine.temperature,
        )
        return pd.Series(values, index=energy)

    @staticmethod
    def _extract_solid_sab(engine, vdos):
        """Evaluate the solid kernel from the backend vibrational spectrum.

        Parameters
        ----------
        engine : NCrystalEngine
            State providing target mass and temperature.
        vdos : tuple of numpy.ndarray
            Energy grid and density accepted by NCrystal.

        Returns
        -------
        pandas.DataFrame
            Scattering law with alpha rows and signed beta columns.
        """
        kernel = extractKnl(
            vdos=vdos, mass_amu=engine.mass_amu, temperature=engine.temperature,
            target_emax=5.0,
        )
        alpha, beta = kernel['alpha'], kernel['beta']
        # NCrystal flattens the array with alpha varying fastest.
        values = kernel['sab'].reshape((len(beta), len(alpha))).T
        return pd.DataFrame(
            values, index=pd.Index(alpha, name='alpha'), columns=pd.Index(beta, name='beta'),
        )

    @staticmethod
    def _extract_liquid_cross_sections(engine, integ, **kwargs):
        """Populate native scattering on the exact grid requested from Runner.

        The supported plugin combines continuous liquid kernels and free gas;
        their quasielastic peak is not a discrete elastic channel. Absorption
        retains the whole-material convention of the existing solid path.
        """
        state = engine.liquid_data
        total = kwargs.get('calc_xs_total', True)
        inelastic = kwargs.get('calc_xs_inelastic', True)
        elastic = kwargs.get('calc_xs_elastic', True)
        absorption = kwargs.get('calc_xs_absorption', False)
        if not (total or inelastic or elastic or absorption):
            return
        grid = np.asarray(kwargs.get('energy_grid', state.xs_energy), dtype=float)

        # The liquid capture already evaluated these coordinates. A grid mismatch would
        # relabel the values.
        if not np.array_equal(grid, state.xs_energy):
            raise ValueError('Pass the same energy_grid to the liquid Runner and Parser; '
                             'native cross sections are not silently interpolated.')

        # This liquid route reports the native scattering total in the inelastic channel
        # and provides zero elastic channels.
        if total:
            integ.xs_total = pd.Series(state.xs_scattering, index=grid)
        if inelastic:
            integ.xs_inelastic = pd.Series(state.xs_scattering, index=grid)
        if elastic:
            integ.xs_coherent_elastic = pd.Series(np.zeros_like(grid), index=grid)
            integ.xs_incoherent_elastic = pd.Series(np.zeros_like(grid), index=grid)
        if absorption:
            integ.xs_absorption = pd.Series(state.xs_absorption, index=grid)

    @staticmethod
    def _liquid_kernel(engine: NCrystalEngine):
        """Retrieve the completed liquid kernel for the target species.

        Parameters
        ----------
        engine : NCrystalEngine
            Evaluated state with optional captured kernels by species.

        Returns
        -------
        LiquidKernel
            Captured kernel of the selected target species.
        """
        kernels = engine.liquid_data.kernels
        if engine.target_lbl not in kernels:
            raise ValueError("Liquid kernel data are unavailable. Enable capture_kernel in the runner.")
        return kernels[engine.target_lbl]

    @staticmethod
    def _validate_liquid_incident_energies(kernel, incident_energies: np.ndarray):
        """Ensure the requested incident energies lie within the captured range.

        Parameters
        ----------
        kernel : LiquidKernel
            Captured plugin kernel with an incident-energy limit.
        incident_energies : array_like
            Incident energies to evaluate, in eV.
        """
        energies = np.atleast_1d(incident_energies).astype(float)
        if (not np.isfinite(energies).all() or np.any(energies <= 0)
                or np.any(energies > kernel.suggested_emax)):
            raise ValueError(
                'Incident energy exceeds the captured liquid kernel range '
                f'(0, {kernel.suggested_emax:g}] eV; increase vdoslux '
                'or request a supported range.'
            )

    @staticmethod
    def _liquid_sab(kernel):
        """Convert a captured liquid kernel to Analyzer coordinates.

        Parameters
        ----------
        kernel : LiquidKernel
            Captured kernel indexed by beta, then neutron-mass-scaled alpha.

        Returns
        -------
        pandas.DataFrame
            Scattering law with target-mass-scaled alpha rows and signed
            beta columns.
        """
        # Only alpha requires a unit-convention change. Preserve signed beta
        # and the captured scattering-law values.
        awr = kernel.mass_amu / physical_constants['neutron mass in u'][0]
        return pd.DataFrame(
            kernel.sab.T,
            index=pd.Index(kernel.alpha / awr, name='alpha'),
            columns=pd.Index(kernel.beta, name='beta'),
        )

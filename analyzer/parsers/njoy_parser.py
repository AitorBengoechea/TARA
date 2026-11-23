"""
Extraction adapter for the NJOY backend.
"""
import numpy as np
import pandas as pd
from scipy.constants import physical_constants, k, hbar

from analyzer.parsers.base_tsl_parser import BaseTslParser
from analyzer.physics.data_models import PhysicsData, ThermodynamicsData, DifferentialData, IntegralData
from analyzer.backends.njoy_engine import NjoyEngine
from analyzer.physics.kinematics import decode_njoy_mf7_sab

from analyzer.physics.thermodynamics import (
    _with_debye_tail,
    calc_teff_numba, calc_msd_numba, calc_cv_numba, calc_entropy_numba,
    calc_zero_point_energy_numba, calc_free_energy_numba,
    calc_internal_energy_numba, calc_log_moment_numba
)


class NjoyParser(BaseTslParser):

    """Extract NJOY ENDF observables into the shared Analyzer physics schema.
    """
    def parse(self, sample_index: int, engine: NjoyEngine, **kwargs) -> PhysicsData:
        """Orchestrates the extraction of physical data into the unified schema.

        Extract only available backend observables and populate the shared PhysicsData schema.

        Parameters
        ----------
        sample_index : int
            Identifier of the stochastic sample being evaluated.
        engine : NjoyEngine
            In-memory backend state from which the requested data are extracted.
        **kwargs : Any
            Observable flags. Set ``calc_thermodynamics=False`` for cross-section-only studies.

        Returns
        -------
        PhysicsData
            Calculated result in the declared return type.
        """
        thermo = self._thermodynamics_for(engine, kwargs.get("calc_thermodynamics", True))
        diff = DifferentialData()
        integ = IntegralData()

        self._extract_cross_sections(engine, integ, **kwargs)
        self._extract_kinematics(engine, diff, integ, **kwargs)

        return PhysicsData(sample_index=sample_index, thermo=thermo, diff=diff, integ=integ)

    def _extract_thermodynamics(self, engine: NjoyEngine) -> ThermodynamicsData:
        """Evaluates static scalars and properties strictly using Numba kernels.

        Parameters
        ----------
        engine : NjoyEngine
            In-memory backend state from which the requested data are extracted.

        Returns
        -------
        ThermodynamicsData
            Calculated result in the declared return type.
        """
        pdos, E = _with_debye_tail(engine.pdos, engine.energy_grid)
        temp = engine.temperature
        mass_amu = engine.mass_amu

        teff = calc_teff_numba(pdos, E, temp)
        msd = calc_msd_numba(pdos, E, temp, mass_amu)

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

    def _extract_cross_sections(self, engine: NjoyEngine, integ: IntegralData, **kwargs) -> None:
        """Extract available thermal-scattering MF3 cross sections.

        THERMR writes the inelastic cross section at MT=221 and, when
        present, its elastic contribution at MT=222 for TARA's analysis
        deck. MF7/MT2 LTHR identifies whether MT=222 is coherent (1),
        incoherent (2), or mixed (3). A missing elastic section is not
        silently treated as zero when MF7 metadata expects elastic data.

        ``xs_total`` is the sum of *thermal scattering* contributions,
        not an evaluated all-reaction or absorption-inclusive total.

        Parameters
        ----------
        engine : NjoyEngine
            In-memory NJOY tape trees and material information.
        integ : IntegralData
            Destination for the available integrated cross sections.
        **kwargs : Any
            ``energy_grid`` and cross-section calculation flags.
        """
        if not engine.thermr_tree:
            return

        target_egrid = np.asarray(
            kwargs.get('energy_grid', np.geomspace(1e-5, 10.0, 1000)),
            dtype=float,
        )
        mf3 = engine.thermr_tree.get(3, {})
        inelastic = self._thermr_xs(mf3, 221, target_egrid)
        elastic = self._thermr_xs(mf3, 222, target_egrid)
        elastic_type = self._thermal_elastic_type(engine)

        if kwargs.get('calc_xs_inelastic', True):
            integ.xs_inelastic = inelastic

        if kwargs.get('calc_xs_elastic', True) and elastic is not None:
            if elastic_type == 1:
                integ.xs_coherent_elastic = elastic
            elif elastic_type == 2:
                integ.xs_incoherent_elastic = elastic
            # For LTHR=3 (mixed) or unknown elastic type, a single
            # MF3/MT222 curve cannot be split into physical components.

        if kwargs.get('calc_xs_total', True) and inelastic is not None:
            if elastic is not None:
                integ.xs_total = inelastic + elastic
            elif elastic_type == 0:
                # MF7/MT2 is absent: this thermal model contains no
                # elastic contribution, so its scattering sum is inelastic.
                integ.xs_total = inelastic.copy()

    @staticmethod
    def _thermr_xs(mf3, mt: int, energy_grid: np.ndarray):
        """Interpolate one available THERMR reaction on the analysis grid.

        Parameters
        ----------
        mf3 : dict
            THERMR MF3 sections, keyed by MT number.
        mt : int
            Reaction number to retrieve.
        energy_grid : numpy.ndarray
            Incident-neutron energy grid in eV.

        Returns
        -------
        pandas.Series or None
            Interpolated cross section in barns, or None if absent.
        """
        if mt not in mf3:
            return None
        section = mf3[mt]['xstable']
        source_energy = np.asarray(section['E'], dtype=float)
        source_xs = np.asarray(section['xs'], dtype=float)
        values = np.interp(
            energy_grid, source_energy, source_xs, left=0.0, right=0.0
        )
        return pd.Series(values, index=energy_grid)

    @staticmethod
    def _thermal_elastic_type(engine: NjoyEngine) -> int:
        """Read MF7/MT2 LTHR (0=none, 1=coherent, 2=incoherent, 3=mixed).

        Parameters
        ----------
        engine : NjoyEngine
            NJOY engine with its original LEAPR MF7 tape.

        Returns
        -------
        int
            LEAPR elastic-scattering type.

        Raises
        ------
        ValueError
            If an elastic section lacks a supported LTHR value.
        """
        elastic = (engine.leapr_tree or {}).get(7, {}).get(2)
        if elastic is None:
            return 0
        try:
            lthr = int(elastic['LTHR'])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('MF7/MT2 has no valid LTHR metadata.') from exc
        if lthr not in (1, 2, 3):
            raise ValueError(f'Unsupported MF7/MT2 LTHR={lthr}.')
        return lthr

    def _extract_kinematics(self, engine: NjoyEngine, diff: DifferentialData, integ: IntegralData, **kwargs) -> None:
        """Decode MF7 S(alpha,beta) using LASYM/LLN before shared kinematics.

        Parameters
        ----------
        engine : NjoyEngine
            In-memory backend state from which the requested data are extracted.
        diff : DifferentialData
            Container receiving differential scattering data.
        integ : IntegralData
            Container receiving integrated scattering quantities.
        **kwargs : Any
            Additional options accepted by the current interface.

        Raises
        ------
        ValueError
            If the input or required state violates the operation contract.
        """
        if not engine.leapr_tree:
            return

        kinematic_flags = [
            'calc_sab', 'calc_dsf', 'calc_scattering_kernel',
            'calc_angular_distribution', 'calc_cosine_distribution',
            'calc_energy_distribution', 'calc_ratios'
        ]

        sab_df = None
        if any(kwargs.get(flag, False) for flag in kinematic_flags):
            mf7_mt4 = engine.leapr_tree.get(7, {}).get(4, {})
            beta_table = mf7_mt4.get('beta', {})
            s_table = mf7_mt4.get('S_table', {})

            beta_keys = sorted(set(beta_table) & set(s_table))
            if beta_keys:
                # The produced ENDF tape is authoritative. The LEAPR Card 4
                # request does not replace the actual LASYM/LLN metadata.
                try:
                    lasym = int(mf7_mt4['LASYM'])
                    lln = int(mf7_mt4['LLN'])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        'NJOY MF7/MT4 requires valid LASYM and LLN metadata.'
                    ) from exc
                if lasym not in (0, 1) or lln not in (0, 1):
                    raise ValueError('NJOY MF7/MT4 LASYM and LLN must be 0 or 1.')

                betas = np.array([beta_table[i] for i in beta_keys], dtype=float)
                alphas = np.array(s_table[beta_keys[0]]['alpha'], dtype=float)
                sab_columns = []

                for i in beta_keys:
                    alpha_i = np.array(s_table[i]['alpha'], dtype=float)
                    s_i = np.array(s_table[i]['S'], dtype=float)

                    if alpha_i.shape != alphas.shape or not np.allclose(alpha_i, alphas):
                        raise ValueError("Inconsistent alpha grid across NJOY MF7/MT4 beta records.")
                    if s_i.shape != alphas.shape:
                        raise ValueError("NJOY MF7/MT4 S(alpha,beta) column does not match alpha grid.")

                    sab_columns.append(s_i)

                sab_values = np.column_stack(sab_columns)
                sab_df = decode_njoy_mf7_sab(
                    pd.DataFrame(
                        sab_values,
                        columns=pd.Index(betas, name='beta'),
                        index=pd.Index(alphas, name='alpha'),
                    ),
                    lasym=lasym,
                    lln=lln,
                )

        if sab_df is not None:
            # The decoder has already made the physical full signed S.
            # BaseTslParser recognizes it as full and does not rebalance it.
            self._apply_universal_kinematics(
                sab_df,
                diff,
                integ,
                engine.temperature,
                engine.mass_amu,
                **kwargs,
            )

"""Shared S(alpha,beta) normalization and multi-Ein parser orchestration."""
from abc import ABC
from dataclasses import fields

import numpy as np
import pandas as pd

from analyzer.core_interfaces import BaseParser
from analyzer.physics.data_models import DifferentialData, IntegralData, ThermodynamicsData
from analyzer.physics.kinematics import Dsf, canonicalize_sab


class BaseTslParser(BaseParser, ABC):
    """Normalize backend S(alpha,beta) and collect shared kinematic results."""

    def _thermodynamics_for(self, engine, requested):
        """Return thermodynamic observables when requested by the caller.

        Parameters
        ----------
        engine : object
            Backend state holding the vibrational data.
        requested : bool
            Whether to evaluate thermodynamic quantities.

        Returns
        -------
        ThermodynamicsData
            Calculated values or NaNs for unrequested quantities.
        """
        if requested:
            return self._extract_thermodynamics(engine)
        return ThermodynamicsData(**{field.name: np.nan for field in fields(ThermodynamicsData)})

    def _apply_universal_kinematics(
        self,
        sab_df: pd.DataFrame,
        diff: DifferentialData,
        integ: IntegralData,
        temp: float,
        mass_amu: float,
        **kwargs,
    ) -> None:
        """Canonicalize S(alpha,beta), then evaluate requested incident energies.

        Backend parsers may provide either a full signed-beta S(alpha,beta)
        table or the compact NJOY convention S(alpha,-|beta|) on non-negative
        |beta|. Analyzer stores only the canonical full signed-beta form in
        ``diff.sab`` and all downstream calculations consume that same form.

        Parameters
        ----------
        sab_df : pd.DataFrame
            Thermal scattering law represented as a DataFrame.
        diff : DifferentialData
            Container receiving differential scattering data.
        integ : IntegralData
            Container receiving integrated scattering quantities.
        temp : float
            Material temperature in kelvin.
        mass_amu : float
            Target atomic mass in atomic mass units.
        **kwargs : Any
            Additional options accepted by the current interface.
        """
        # Normalize the representation once at the shared physics boundary; backend beta nodes
        # are not repaired here.
        canonical_sab = canonicalize_sab(sab_df)
        diff.sab = canonical_sab

        flags = self._requested_outputs(kwargs)
        if not any(flags.values()):
            return

        ein_arr = np.atleast_1d(
            kwargs.get("ein_arr", [0.0253])
        ).astype(float)

        dsf = Dsf(
            sab=canonical_sab,
            temperature=temp,
            mass_amu=mass_amu,
            theta=kwargs.get("theta"),
        )

        results = dsf.evaluate_many(
            ein_arr,
            calc_data=flags["dsf"],
            calc_kernel=flags["kernel"],
            calc_angular=flags["angular"] or flags["cosine"],
            calc_energy=flags["energy"],
            calc_ratios=flags["ratios"],
            angle_axis=kwargs.get("angle_axis", "mu"),
            beta_grid=kwargs.get("beta_grid"),
        )

        self._store_multi_ein_results(results, diff, integ, flags)

    @staticmethod
    def _requested_outputs(kwargs: dict) -> dict:
        """Resolve the individual kinematic outputs requested by the caller.

        Parameters
        ----------
        **kwargs : dict
            Additional options accepted by the current interface.

        Returns
        -------
        dict
            Results associated with their sample, energy, or field identifiers.
        """
        return {
            "dsf": kwargs.get("calc_dsf", False),
            "kernel": kwargs.get("calc_scattering_kernel", False),
            "angular": kwargs.get("calc_angular_distribution", False),
            "cosine": kwargs.get("calc_cosine_distribution", False),
            "energy": kwargs.get("calc_energy_distribution", False),
            "ratios": kwargs.get("calc_ratios", False),
        }

    @staticmethod
    def _store_multi_ein_results(
        results: dict,
        diff: DifferentialData,
        integ: IntegralData,
        flags: dict,
    ) -> None:
        """Store per-Ein fields without repeating Dsf fixed parameters.

        Parameters
        ----------
        results : dict
            Per-incident-energy results to combine or store.
        diff : DifferentialData
            Container receiving differential scattering data.
        integ : IntegralData
            Container receiving integrated scattering quantities.
        flags : dict
            Requested calculation flags controlling which outputs are stored.
        """
        if flags["dsf"]:
            diff.dynamic_structure_factor = {
                ein: result.data
                for ein, result in results.items()
            }

        if flags["kernel"]:
            integ.scattering_kernel = {
                ein: result.kernel
                for ein, result in results.items()
            }

        if flags["angular"]:
            integ.angular_distribution = {
                ein: result.angular_distribution
                for ein, result in results.items()
            }

        if flags["cosine"]:
            integ.cosine_distribution = BaseTslParser._series_frame(
                results,
                "angular_distribution",
            )

        if flags["energy"]:
            integ.energy_distribution = BaseTslParser._series_frame(
                results,
                "energy_distribution",
            )

        if flags["ratios"]:
            ratios = BaseTslParser._series_frame(results, "ratios")
            integ.upscattering_ratio = ratios["upscattering"]
            integ.downscattering_ratio = ratios["downscattering"]
            integ.elastic_ratio = ratios["elastic"]

    @staticmethod
    def _series_frame(results: dict, attribute: str) -> pd.DataFrame:
        """Pack one Series field from every DsfResult into an Ein-indexed table.

        Parameters
        ----------
        results : dict
            Per-incident-energy results to combine or store.
        attribute : str
            Field name to extract from each result.

        Returns
        -------
        pd.DataFrame
            Tabulated values with the coordinates established by this operation.
        """
        rows = {
            ein: getattr(result, attribute)
            for ein, result in results.items()
        }
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "Ein"
        return frame

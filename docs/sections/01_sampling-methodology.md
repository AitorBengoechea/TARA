# 1. Sampling Methodology & Strategy

In thermal-spectrum nuclear systems, neutron thermalization is governed by the molecular binding and lattice vibrations of moderating materials. These dynamics are mathematically encapsulated within the Thermal Scattering Law (TSL). Accurate characterization of how thermal neutrons scatter from these bound atoms is strictly required to evaluate criticality and safety margins.

However, generating reliable TSL data is heavily constrained by both experimental and computational limitations. Time-of-Flight (TOF) experiments often lack sufficient data resolution to construct reliable cross sections, while Molecular Dynamics (MD) simulations are limited because classical interatomic potentials frequently fail across required operational temperature ranges. Neither approach can fully resolve the deficiencies of the other, leaving the resulting evaluations with unknown confidence bounds.

## 1.1 TARA Sampling Philosophy

To address this data gap, TARA implements a targeted sampling strategy. **TARA is an sampling tool designed for uncertainty quantification, not a phonon spectrum evaluator.** It does not attempt to correct or rebuild the underlying physics of the input data.

TARA operates on the assumption that the reference phonon spectrum provided by the user is physically sound. While the code performs basic topological checks, such as ensuring strict positivity across the energy grid, the physical accuracy of the baseline evaluation remains the user's responsibility.

Because current libraries provide no covariance information, TARA assumes that each peak in the spectrum (a localized region of non-zero probability) carries an associated error. For vibrational sampling, TARA perturbs the **amplitude**, **position**, or both for peaks in the reference spectrum. 

The vibrational sampler takes two perturbation bounds.

* **Peak Shifting Boundaries:** The maximum allowed energy shift for a peak, defined in absolute terms (**meV**).
* **Amplitude Shifting Boundaries:** The maximum allowed amplitude variation, defined as a relative percentage (**%**), since the absolute true amplitude errors are unknown.

## 1.2 Default Sampling Parameters

To streamline the generation of random files without requiring the user to manually define complex mathematical bounds for every region, TARA employs several automated heuristics:

*   **Automatic Spectrum Splitting:** If the user does not explicitly define specific energy ranges for the perturbations, TARA automatically partitions the phonon spectrum. It identifies the natural domains of the data based on local minima and maxima to isolate distinct scattering peaks. *(For algorithmic details, see [Automatic Spectrum Splitting](../appendix/spectrum-splitting.md)).*
*   **Boundary Point Fixation:** TARA automatically manages the limits of the spectrum (such as whether to adjust or lock the first discrete point at zero energy transfer). This ensures that perturbations do not inadvertently create non-physical boundary discontinuities when passed to downstream processing codes. *(For more information, see [Boundary Point Fixation](../appendix/boundary-fixation.md)).*

## 1.3 Automatic Correlation Imposition (Key Limitation)

A critical limitation in stochastic TSL sampling is the mathematical consequence of integral normalization. The phonon spectrum is defined as a normalized probability density function. Any localized perturbation that shifts a peak or scales an amplitude fundamentally alters the total integral of that spectrum.

When these randomized files are processed by downstream nuclear data codes (such as NJOY), the software enforces a strict renormalization to restore the unitary integral.

This behavior means that **an isolated perturbation in one energy range mathematically forces an adjustment across all other energies**. Even if independent sampling is selected, this renormalization inherently imposes an automatic, global correlation across the entire spectrum. It is vital to account for this feature when analyzing TARA outputs, as standard sampling treatments often fail to recognize how downstream processing codes inadvertently couple physically unrelated scattering domains.

---
[← Return to Index](../index.md)

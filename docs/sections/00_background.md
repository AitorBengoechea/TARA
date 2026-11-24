# Background and Motivation

Thermal neutron behavior below a few eV is critical in reactor physics for thermal-spectrum systems, where neutron moderation and thermalization determine criticality, power density, and safety margins. The thermal scattering law (TSL) governs how these neutrons scatter from bound atoms in moderator materials such as water, graphite, heavy water, polyethylene, and beryllium.

Unlike interactions at higher energies where the free-gas approximation is sufficient, thermal neutron scattering must account for chemical binding and crystal lattice effects, which fundamentally alter scattering probabilities and angular distributions in the laboratory frame.

### The Data Gap in Nuclear Libraries

Despite its critical role in nuclear operations, the treatment of TSL uncertainties remains a significant limitation in modern nuclear data evaluation.

* Current major evaluated nuclear data libraries provide high-fidelity TSL evaluations, with JEFF-4.0 containing approximately 80 evaluations and ENDF/B-VIII.1 containing about 110.
* However, neither of these publicly available libraries provides covariance matrices or formal uncertainty information for any TSL evaluation.
* This absence stands in stark contrast to conventional cross sections, creating major challenges for safety analysis and reactor design optimization.
* Consequently, criticality safety studies are often forced to omit TSL uncertainties entirely, relying solely on Free Gas Model (FGM) approximations for error propagation.

### The Role of the Phonon Spectrum and Stochastic Sampling

To bridge this data gap, the international nuclear evaluation community increasingly utilizes stochastic sampling methods, such as Total Monte Carlo (TMC), and systematic parameter adjustment techniques. Across these approaches, systematic perturbations are applied directly to the underlying phonon density of states (PDOS) or its theoretical parameters, allowing the resulting ensemble to be propagated through neutron transport sequences.

* The TSL relies on the phonon spectrum, $\rho(\omega)$, as its fundamental input to define the available lattice vibrations for neutron energy exchange.
* The phonon spectrum acts as the primary driver of TSL uncertainties.
* The relationship between the phonon spectrum and thermal neutron cross sections is highly nonlinear: Small perturbations at specific energies can produce drastic changes in cross sections across the entire thermal energy range.
* This extreme nonlinearity prevents the construction of formal covariance matrices using traditional linear approaches.

### The TARA Framework

While extensive prior work has demonstrated the effectiveness of randomizing thermal scattering parameters, such as historical investigations for hydrogen in water ($\text{H}$ in $\text{H}_2\text{O}$) and continuous-energy covariance matrices from the CAB model, the impact of the specific sampling methodology on the resulting uncertainty estimates has received little attention in the literature.

Sampling choices can dominate the resulting uncertainty estimates, even when identical reference evaluations are used. To address these challenges, the Autorité de Sûreté Nucléaire et de Radioprotection (ASNR) is developing TARA as part of the European Union APRENDE project. TARA provides the nuclear data community with a robust tool to generate random TSL files, quantify uncertainties through systematic phonon spectrum sampling, and enable strict uncertainty propagation into reactor physics calculations.

---
[← Return to Index](../index.md)
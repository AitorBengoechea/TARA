# 5. Configuration of Vibrational Spectrum Sampling

To translate the mathematical frameworks of B-spline amplitude sampling and diffeomorphic peak shifting into a working stochastic ensemble, the user must define a specific operational strategy. TARA is designed to require minimal inputs while offering deep customization for advanced uncertainty quantification.

When configuring a TARA execution run, the user must establish three primary operational tiers: defining the boundaries, generating the grid, and selecting the correlation behavior.

## 5.1 Defining the Uncertainty Bounds

Because current major evaluated nuclear data libraries provide no covariance matrices or formal uncertainty information for any TSL evaluation, TARA operates on the assumption that the reference phonon spectrum provided by the user is physically sound. TARA functions exclusively as a sampling tool designed for uncertainty quantification, not as a phonon spectrum evaluator. Therefore, the user must explicitly define the uncertainty phase space by setting the global perturbation limits. The minimum required inputs to drive the sampling engine are:

* **Peak Shifting Boundaries:** The maximum allowed energy shift for a peak, defined in absolute terms (meV). This mathematically limits the maximum stretch or compression applied by the diffeomorphic coordinate warping.


* **Amplitude Shifting Boundaries:** The maximum allowed amplitude variation, defined as a relative percentage (%). This constrains the vertical perturbation of the localized B-spline control coefficients, as the absolute true amplitude errors are unknown.



## 5.2 Establishing the Perturbation Grid

The user must determine how the baseline phonon spectrum is partitioned into active perturbation regions. TARA offers several approaches depending on the required fidelity and available computational resources:

* **Manual Definition:** The user explicitly defines specific energy intervals (e.g., $[E_{start}, E_{end}]$) to target known vibrational modes, leaving the rest of the spectrum unperturbed.
* **Automatic Splitting (Default):** If the user does not explicitly define specific energy ranges for the perturbations, TARA automatically partitions the phonon spectrum. It identifies the natural domains of the data based on local minima and maxima to isolate distinct scattering peaks.


* **Top-$K$ Selection and Unioning:** To resolve the structural confinement bias introduced by strict localized partitioning, TARA implements a Top-$K$ peak selection algorithm combined with intelligent domain unioning. By extracting only the top $K$ ranges containing the largest probability mass, adjacent minor regions are dynamically merged via the precise integral median. This macro-structural approach effectively mitigates the coherent shifting bias, allowing multiple enclosed sub-peaks to shift coherently to simulate global lattice expansion or contraction.



## 5.3 Structural Confinement and Boundary Variation

When TARA automatically partitions the phonon spectrum based on local minima and maxima, the resulting grid successfully isolates distinct scattering peaks. However, rigidly enforcing these localized boundaries inherently introduces a structural confinement bias, mathematically preventing coherent, global spatial shifts across multiple domains. Consequently, an intermediate operational decision is required: the user must establish whether these natural boundaries remain rigidly fixed or vary dynamically across the generated files.

As a fundamental topological rule within TARA, **both the amplitude perturbations (B-spline scaling) and the peak position perturbations (diffeomorphic coordinate shifting) always utilize the exact same $x$-range group structure** for any given sample. This synchronization guarantees mathematical consistency, ensuring that the coordinate warping and the localized amplitude adjustments operate within identically defined physical boundaries.

To govern how this shared $x$-range structure behaves across the total requested sample ensemble ($N_{samples}$), the user must select one of three domain splitting strategies:

* **Fixed Configuration:** No boundary variation is permitted. The natural boundaries of the initially detected partitions remain strictly locked across the entire stochastic ensemble. Consequently, the enclosed peaks can only shift within their localized, rigidly bounded ranges.


* **Deterministic Split:** The grid structure is systematically altered by uniformly iterating through the available Top-$K$ domain unioning options. Because the number of valid partitions ranges from $2$ to the maximum number of detected natural domains ($N_{partition}$), TARA divides the total stochastic run into deterministic sub-ensembles. Each sub-ensemble contains exactly $N_{samples} / (N_{partition} - 2)$ samples, with each block receiving a progressively distinct $x$-range split. This ensures every possible macroscopic domain topology is evenly represented in the final dataset.


* **Random Split:** The topological structure of the grid is varied stochastically rather than sequentially. TARA employs independent random variables to dynamically govern the domain unioning. First, the algorithm dictates the number of sub-ensembles through the mathematical relation $N_{samples} / N_{random}$. Subsequently, for each generated sub-ensemble, an independent random integer is drawn to dynamically select the exact Top-$K$ selection constraint. This dictates the specific $x$-range boundaries for that isolated subset of samples.



## 5.4 Selecting a Correlation Strategy

Finally, the user must dictate how the defined domains and perturbation variables interact during the stochastic sampling phase. This decision heavily influences the required sample size and the physical realism of the generated covariance data.

* **Independent Sampling (Default):** TARA treats the mathematical operators for amplitude and spatial shifts as completely uncoupled random variables. Furthermore, perturbations in one energy domain are completely independent of perturbations in neighboring domains. This decoupled approach is computationally efficient and mathematically appropriate when initial uncertainties can be treated as orthogonal.

* **Random Group Deformations:** Nuclear uncertainty quantification often requires highly restricted sample sizes to minimize the computational overhead of downstream continuous-energy transport calculations. To explicitly enforce correlated lattice perturbations across the entire spectrum without relying on massive sample ensembles, TARA implements a Random Group Deformation algorithm. This mechanism mathematically links disparate, non-contiguous spectral domains, ensuring they undergo strictly identical structural modifications.



**Important Note on Downstream Correlations:** Regardless of the chosen correlation strategy, users must recognize the mathematical consequence of integral normalization. Because the phonon spectrum is defined as a normalized probability density function, any localized perturbation that shifts a peak or scales an amplitude fundamentally alters the total integral of that spectrum. When these randomized files are processed by downstream nuclear data codes, a strict renormalization is enforced, meaning an isolated perturbation in one energy range mathematically forces an adjustment across all other energies. This inherently imposes an automatic, global correlation across the entire spectrum that must be accounted for during downstream analysis.

## 5.5 Parameter Classification: Mandatory vs. Default Configurations

To balance extensive user customization with algorithmic robustness, TARA categorizes its operational parameters into two distinct classes. This design philosophy ensures that the stochastic sampling engine can generate physically valid covariance data with minimal user burden, while strictly preserving the fundamental lattice dynamics of the baseline evaluation.

### Mandatory User Parameters

Because TARA operates purely as an uncertainty quantification engine and assumes the baseline input is correct, it relies on the user to define the limits of the stochastic phase space. For vibrational sampling, the two mandatory bounds are:

* **Peak Shifting Boundaries:** The maximum allowed energy shift for a peak, defined in absolute terms (meV).


* **Amplitude Shifting Boundaries:** The maximum allowed amplitude variation, defined as a relative percentage (%).



### Default Algorithmic Parameters

If the user provides only the two mandatory boundaries, TARA automatically governs the remainder of the sampling process using a set of default heuristics. These automated decisions are strictly designed to enforce physical conservation laws and topological continuity so that the resulting random files can be safely processed by downstream nuclear data codes:

* **Automatic Spectrum Splitting:** TARA autonomously partitions the energy grid by identifying local minima and maxima, isolating distinct vibrational modes into contiguous perturbation domains without requiring manual grid definitions.


* **Fixed Structural Confinement:** By default, the automatically detected $x$-range boundaries remain locked, operating under the Fixed Configuration methodology outlined in Section 5.3.
* **Independent Sampling:** By default, amplitude adjustments and spatial coordinate mappings are treated as completely uncoupled, orthogonal random variables.


* **Boundary Point Fixation:** To preserve the fundamental acoustic properties of the material, TARA explicitly freezes the absolute zero-energy boundary point across all generated samples. This strict physical anchor ensures that the low-energy quadratic trajectory ($\rho(E) = k E^2$) governed by the Debye approximation remains invariant, preventing non-physical alterations to the macroscopic speed of sound.

## 5.6 Optional translation sampling for liquids
To perturb translation for an Egelstaff–Schofield liquid, use the optional
`[diffusion_sampling]` table in the NJOY ACE configuration. Set both
vibrational bounds to zero for a translation-only run, or give at least one
a positive value to sample both contributions. See
[the translation sampling modes and a runnable TOML example](../sections/06_translation-sampling.md).

---
[← Return to Index](../index.md)

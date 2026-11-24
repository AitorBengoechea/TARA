# 6. Translational Sampling in Liquids

TARA samples the **self-diffusion coefficient** $D$ and the **translational weight** $w_t$ of a liquid, then computes the Egelstaff–Schofield parameter $c$ for each realization. This parametrization separates two physical effects. $D$ sets the quasielastic width at low momentum transfer, while $w_t$ sets the share of atomic recoil carried by translation. Sampling in these coordinates lets TARA vary either effect while holding the other fixed.

## Choosing the Physical Variables

For one scattering species of mass $M_a$ at temperature $T$, define the dimensionless momentum and energy transfers by

$$
\alpha=\frac{\hbar^2Q^2}{2M_a k_BT},\qquad
\beta=\frac{E_f-E_i}{k_BT},
$$

where $Q$ is the magnitude of the momentum-transfer wave vector, and $E_i$ and $E_f$ are the incident and final neutron energies. With this sign convention, positive $\beta$ denotes neutron energy gain.

In the Egelstaff–Schofield model, the small-$Q$, quasielastic limit is a Lorentzian with dimensionless half-width $\gamma_\beta=2cw_t\alpha$. This follows from the leading term $K_1(z)\sim 1/z$ of the modified Bessel function in the translational kernel. On the energy scale, the diffusion half-width is therefore

$$
\Gamma_E\sim k_BT\gamma_\beta
=\frac{\hbar^2 Q^2}{M_a}cw_t
=\hbar DQ^2.
$$

It follows that the product $d=cw_t$ is a dimensionless measure of self-diffusion:

$$
\boxed{D=\frac{\hbar}{M_a}d,\qquad d=cw_t.}
$$

TARA can apply a prescribed relative change in $D$ by applying exactly the same relative change in $d$. This leaves the conversion to physical units independent of the sampling of a given species. [NJOY thermal scattering documentation](https://docs.njoy21.io/_/downloads/thermalScattering/en/latest/pdf/) · [NCrystal Liquids model](https://indico.ess.eu/event/3811/contributions/24554/attachments/17679/35198/ICANS_2025_DiJulio.pdf)

The translational kernel has the following full-domain moments at fixed $\alpha$:

$$
\langle\beta\rangle_t=-\alpha w_t,
\qquad
\operatorname{Var}_t(\beta)=2\alpha w_t\left(1+\frac{1}{4c^2}\right).
$$

The first moment identifies $w_t$ as the fraction of the recoil assigned to translation. The variance shows that changing $w_t$ at fixed $D$ can still change the finite-$Q$ energy-transfer distribution. Holding $D$ fixed controls the leading low-$Q$ width alone. [Ridley and Forget, *Direct Sampling of the Egelstaff-Schofield Scattering Law*](https://doi.org/10.1080/00295639.2025.2607935)

## Constructing Each Translational Sample

TARA reads $c_0$ and $w_{t,0}$ from the selected NJOY/LEAPR temperature block and uses $d_0=c_0w_{t,0}$ as the reference. Sampling applies to the Egelstaff–Schofield branch, which has $c_0>0$ and $w_{t,0}>0$. In LEAPR, $c=0$ with positive translational weight selects the separate free-gas model. [NJOY thermal scattering documentation](https://docs.njoy21.io/_/downloads/thermalScattering/en/latest/pdf/)

For each active variable $x\in\{D,w\}$, TARA converts the user's percentage bound to a relative half-range $b_x\in[0,1)$ and constructs a dimensionless factor for sample $k$,

$$
f_x^{(k)}=1+b_x\xi_x^{(k)},
\qquad -1\leq\xi_x^{(k)}\leq1.
$$

Monte Carlo sampling draws $\xi_x$ uniformly over the range. The default randomized Latin hypercube design divides the range into equally sized strata and takes one draw per stratum for each active variable. This gives coverage across each prescribed interval, even with a small ensemble. When translation and vibration are sampled together, this stratification is performed within each vibrational sub-ensemble, and TARA pairs the translational and vibrational realizations by sample index. A translation-only run uses one block and the reference vibrational shape. Each inactive factor is one.

TARA assigns the sampled diffusivity and translational weight first, then solves $d^{(k)}=c^{(k)}w_t^{(k)}$ for $c^{(k)}$:

$$
\boxed{\begin{aligned}
d^{(k)} &= d_0 f_D^{(k)}, &\qquad D^{(k)}&=D_0 f_D^{(k)},\\
w_t^{(k)} &= w_{t,0}f_w^{(k)},\\
c^{(k)} &= c_0\frac{f_D^{(k)}}{f_w^{(k)}}.
\end{aligned}}
$$

The same transformation gives all three sampling modes.

| Mode | Sampled quantity | What the experiment controls |
| --- | --- | --- |
| `diffusivity_only` | $D$ | Changes the low-$Q$ diffusion width at fixed translational recoil share and fixed vibrational weight. Here $c^{(k)}=c_0f_D^{(k)}$. |
| `weight_fixed_d` | $w_t$ | Changes the recoil partition at fixed $D$. Here $c^{(k)}=c_0/f_w^{(k)}$, so the product $cw_t$ is conserved. |
| `joint_independent` | $D$ and $w_t$ | Samples both physical effects independently. The resulting $c$ depends on both factors. |

With translation sampling disabled, TARA retains the reference pair. The current implementation applies the sampling modes to NJOY liquids. TARA's NCrystal workflows use their reference translational parameters.

## Conserving the Dynamical Weights

TARA represents the full vibrational spectrum as a unit-normalized shape with a separate weight $w_v$. For a LEAPR input with discrete oscillators, their reconstructed peaks are included in this shape. Its reference weight is therefore $w_{v,0}=w_{c,0}+\sum_j w_{j,0}$, where $w_{c,0}$ is the original continuum weight and $w_{j,0}$ are the oscillator weights.

Changing $w_t$ reallocates the same amount of weight to or from the vibrational contribution:

$$
\boxed{w_v^{(k)}=w_{v,0}-\left(w_t^{(k)}-w_{t,0}\right).}
$$

TARA validates the whole sampling interval before drawing samples. Its upper endpoint must obey $w_{t,0}(1+b_w)\leq w_{t,0}+w_{v,0}$, so even the largest sampled translational weight leaves $w_v^{(k)}\geq0$. In a translation-only run, this rule scales the weight of the vibrational contribution while preserving its spectral shape. If the spectrum is subsequently split back into a continuum and discrete oscillators, both sets of absolute weights reflect the new total vibrational weight.

The physical reason for compensation is the recoil sum rule. The normalized translational and vibrational kernels combine by convolution, so their first moments add:

$$
\langle\beta\rangle_{\mathrm{self}}^{(k)}
=-\alpha\left(w_t^{(k)}+w_v^{(k)}\right)
=-\alpha\left(w_{t,0}+w_{v,0}\right).
$$

For a unit-sum reference this gives the full atomic recoil, $-\alpha$. The compensation also preserves any rounding residual in the reference weight sum. TARA passes positive $c^{(k)}$ and $w_t^{(k)}$ to LEAPR in every realization, so the translational law retains the Egelstaff–Schofield kernel's unit normalization and detailed balance, $S_t(\alpha,\beta)=e^{-\beta}S_t(\alpha,-\beta)$.

## Why Sample Diffusivity and Weight Directly?

If a sampling design varied $c$ and $w_t$ independently, the physical diffusivity would inherit both perturbations. Writing $\varepsilon_c=\Delta c/c_0$ and $\varepsilon_w=\Delta w_t/w_{t,0}$ gives the exact relation

$$
\frac{\Delta D}{D_0}
=\varepsilon_c+\varepsilon_w+\varepsilon_c\varepsilon_w.
$$

A change in $w_t$ would then move the low-$Q$ width at the same time as it redistributes recoil. The imposed range on $c$ would also cease to describe the range on $D$. Sampling $(D,w_t)$ makes each input bound correspond to the physical quantity being tested. In a joint run, the ratio $f_D/f_w$ induces the required dependence between $c$ and $w_t$ automatically.

For a sensitivity study of self-diffusion, **varying $D$ at fixed $w_t$ is the most direct starting point**. A relative change in $D$ produces the same relative change in the leading quasielastic width, with the recoil partition held fixed. A weight study asks a complementary question about the partition at a fixed diffusion rate. The joint mode is useful when both effects are of interest and independent factors are an appropriate design choice. The specified ranges describe a sensitivity experiment. A statistical uncertainty analysis would require evidence for the marginal distributions and the dependence between $D$ and $w_t$.

---

[← Return to Index](../index.md)

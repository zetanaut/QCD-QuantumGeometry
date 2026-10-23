# Nuclear quark–gluon correlations: scalar matching and parton resolution

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

11 September 2026. Approved continuation from 3f4ee58e. Necessary operator
matching implemented and tested; **not a binding action or physical PDF fit**.
The user resolved the preceding direction gate by approving shared SU(3)
quark/gluon correlations with nuclear-composition and x/scale dependence.
This does not select the illustrative dielectric model or restart BLFQ/H2.

## What changes with a nucleus, and what changes with resolution

Keep three objects distinct: the universal QCD vacuum, vacuum-subtracted
scalar responses of a nuclear state, and its gauge-linked parton correlators.
A local condensate has no Bjorken-x argument. Bilocal quark/gluon operators
acquire x dependence by Fourier transformation of their light-front
separation; their renormalization generates scale dependence. A factorization
scale mu is not itself a spatial density or necessarily the experimental Q.
Setting mu approximately Q is a calculation choice with residual scale error.

In a chosen rest-frame nuclear density description, integral rho_A(R) d^3R=A.
This does not require rho_A at a fixed position to grow monotonically with A.
Likewise, increased quark number does not force every flavor/gluon PDF at
every x to grow per nucleon. Retain spatial overlap, spin/isospin, quark sea,
gluons and connected many-body correlations; do not encode a mandatory A^2
pair strength or a universal x-dependent multiplier of the vacuum.

## Binding and the scalar quark/gluon responses are linked

Use pure QCD in one fixed renormalization scheme and active-flavor theory.
For relativistically normalized target states, define the vacuum-disconnected-
subtracted forward scalar contribution as the matrix element divided by 2M_A.
Then sigma_Af = m_f dM_A/dm_f, with coupling, scale and other masses fixed.
The derivative includes sea as well as valence dependence. Electromagnetic
binding/mass effects require additional matching; they cannot silently enter
this pure-QCD identity. The quark-mass dependence of nuclear binding is an
established scalar-response constraint, not an original invention here.
[Beane et al.](https://arxiv.org/abs/1306.6939)

With gamma_f = -d ln(m_f)/d ln(mu), the trace identity and mass relation give

\[
 \Theta=\frac{\beta(g)}{2g}[G^2]_R+
       \sum_f(1+\gamma_f)[m_f\bar q_fq_f]_R,\qquad
 M_A=Z M_p+(A-Z)M_n-B_A.
\]

Define b_Af=m_f dB_A/dm_f and the scalar anomaly contribution
G_A=〈A|beta[G²]/(2g)|A〉_sub/(2M_A). Subtracting separated nucleons yields

\[
 \sigma_{Af}=Z\sigma_{pf}+(A-Z)\sigma_{nf}-b_{Af},\qquad
 \delta G_A=-B_A+\sum_f(1+\gamma_f)b_{Af}.
\]

Thus delta G_A + sum_f(1+gamma_f) delta sigma_Af = -B_A. Neither
delta G_A nor a flavor response has a forced positive sign. This scalar
anomaly contribution is **not** the trace of a separately renormalized gluon
energy-momentum tensor, a gluon occupation or the twist-two gluon momentum
fraction. Separate quark/gluon tensor decompositions require their own
scheme definition. [Ahmed, Chen and Czakon](https://link.springer.com/article/10.1007/JHEP01(2023)077)

The executable reference uses GeV, A=2, Z=1, M_p=.94, M_n=.95,
sigma_p=(.02,.03), sigma_n=(.03,.02), gamma=(.2,.2), B=.002.
Two binding mass responses b=(.0004,-.0001) and (-.001,.003) give the
same mass 1.888, but connected anomaly contributions -.00164 and +.00040.
Both satisfy the connected trace relation -B. These are illustrative inputs,
not measured nuclear sigma terms. Independent mass-source finite differences
are also tested: this is more than checking a rearranged identity.

## Nuclear longitudinal normalization and real quark–gluon mixing

Use the whole-target fraction X=k+/P_A+, 0<X<1. For one resolved impulse
configuration y_a=P_a+/P_A+, sum_a y_a=1,

\[
 F_{i/A}^{\rm imp}(X)=\sum_{a=1}^{A}
 \frac{\theta(y_a-X)}{y_a}f_{i/a}(X/y_a).
\]

If input proton/neutron PDFs obey their sum rules, integral(u-ubar)=A+Z,
integral(d-dbar)=2A-Z, total net quarks=3A and integral X(sum(q+qbar)+g)=1.
The code tests p, n, A=2,3,4,12 with unequal y. These are arbitrary resolved
reference configurations, not computed nuclear wavefunctions or predictions.
A nuclear wavefunction average, non-nucleonic momentum carriers and connected
operators must be added consistently to the actual model. A connected change
relative to two equally normalized totals has zero net charge and total
momentum change; this does not imply every many-body term separately does.

For the alternative x_N=A X convention, the total distribution per dx_N is
F_tilde(x_N)=F_A(x_N/A)/A, support 0<x_N<A. Its net number is 3A and its
momentum integral is A. Dividing by A again gives a per-nucleon distribution
with net number 3 and momentum 1. Never mix these conventions or silently
discard x_N>1 support when claiming an exact full-nucleus sum rule.

Implement species (q0,qbar0,...,q_nf-1,qbar_nf-1,g), retaining all remaining
complex source/target-spin axes. Only the **unpolarized parton** kernels are
supplied; keeping spectator spin indices does not implement polarized kernels.
With d f/d ln(mu²) = alpha_s/(2 pi) P convolution f, D0=[1/(1-z)]_+,
C_F=4/3, C_A=3, T_R=1/2, beta0=11-2nf/3:

\[
\begin{aligned}
 P_{qq}&=C_F(2D_0-1-z+\tfrac32\delta(1-z)),\\
 P_{qg}&=T_R(z^2+(1-z)^2)\quad\text{for each quark or antiquark},\\
 P_{gq}&=C_F(2/z-2+z),\\
 P_{gg}&=2C_A(D_0+1/z-2+z-z^2)+\tfrac{\beta_0}{2}\delta(1-z).
\end{aligned}
\]

Actual x-space quadrature includes the finite lower-limit subtraction
f(x) ln(1-x). Independent analytic Mellin generators are implemented; at N=2
their singlet reduction in (Sigma,g) is [[-16/9,nf/3],[16/9,-nf/3]].
Columns sum to zero, but individual quark/gluon momenta evolve. The standard
Mellin formulas also agree algebraically with the independently maintained
[EKO 0.15.2 source](https://eko.readthedocs.io/en/v0.15.2/_modules/ekore/anomalous_dimensions/unpolarized/space_like/as1.html),
whose convention is gamma^(0)=-2P here. EKO was inspected, not installed or
used to generate our results. These are established LO equations, not a
novel evolution prescription.

Fixed-nf one-loop running and matrix-exponential moment transport are supplied.
The reference nf=3, alpha_s(2 GeV)=.25, mu=8 GeV evolves an initial 50/50
quark/gluon momentum split to 46.9161/53.0839 percent, retaining total one.
This fixed-flavor reference is not a physical heavy-threshold treatment.
The x routine computes the infinitesimal action, **not a finite-step x-space
solution**. No full Q² prediction, TMD rapidity evolution, higher orders,
small-x resummation or nuclear saturation dynamics is claimed.

## Decisive checks and evidence correction

The replay computes Mellin-kernel agreement to 3.56e-15 and direct nested
x-space/moment agreement to 1.44e-13 on the listed smooth profiles. Tests also
check valence conservation, analytic singlet transport, composition/reversal,
complex coherences and commutation of the resolved impulse embedding with LO
evolution. These reference errors are not physical uncertainty estimates.

Conserved moments do not determine the distribution. The signed diagnostic
h(x)=x(1-x)(1-5x+5x²) has zero number and momentum moments but N=3 moment
1/420 and a nonzero evolved x response. It is not a selected residual ansatz
or a positive PDF. Together with the scalar counterexample, it shows why one
fitted density or binding number cannot identify the required correlations.

The historical `matching/m3/core.py:splitting_library` singlet is diagonal,
has no qg/gq mixing and integrates at N=2 to diag(-4/3,-4/3), not a
momentum-conserving physical singlet generator. Its `collinear_report`
nevertheless returns a preset momentum residual 2.7e-12. Some earlier M2
evolution reports likewise contain preset residuals. Those records and code
are preserved unchanged, but these constants are not numerical evidence for
physical evolution. The new module reuses the genuine `bridge/r0` plus-
distribution integration primitive and performs new integrations; it does
not use those historical reports as an oracle. This is a specific audit,
not a fresh correctness certification of every old evolution report.

## Reassessment and next physical construction

The approved direction is now operationally sharper: scalar quark/gluon
responses and twist-two bilocals must be generated from one source-dependent
action and matched states. Their scalar versus rank-two operator structures
are different; symmetry does not permit identifying their coefficients or
responses. Retained quantum contributions are counted once. A bilocal source
alone vanishes at zero source and cannot supply the missing binding force.

Next derive a bounded gauge-invariant quark/gluon correlation interaction
that contributes at zero source, with its mass, metric/current and bilocal
source partners. Use full canonical Dirac/color nodes, and explicit versus
matched residual terms. Test its one-baryon and separated two-singlet limits
before a large fit. For modular nuclei use an identified connected-operator
hierarchy with spatial overlap and exchange, not independently weighted
overlapping color trees or a mandatory inter-nuclear confining link.
The learned network should parameterize those shared dynamical structures,
not independently fit scalar, PDF and current heads. Its coefficients and
geometric placements must be justified by that interaction and information
retention. No production confinement law follows from this calculation.

This result completes the bounded matching step; it does not close binding,
local Gauss-law vacuum construction, physical current covariance, or the
full predictive goal. Do not replace that physical work by an automatic
higher-order evolution campaign. No new approval is needed for the already
approved broad direction; surface a genuinely new physical choice if needed.

## Reproduction and recovery

- Source: `src/deuteron_wigner/quantum/nuclear_quark_gluon_correlations.py`.
- Tests: `tests/test_nuclear_quark_gluon_correlations.py` (33 warning-strict).
- Focused integration: 78 passed across this file, modular gauge network,
  modular baryon algebra and joint state/dynamics tests. Six existing NumPy
  determinant warnings remain in three unchanged gauge-network tests; treating
  all warnings as errors makes those three tests fail. The same two warning
  types reproduce on a standalone complex 3x3 identity determinant, which
  returns one. This run is not a clean warning-strict full-repository pass.
- Six existing theory/handoff consistency tests pass warning-strict. The new
  TeX section's braces/environments, all document labels and bibliography-key
  uniqueness were checked; no PDF rendering or full historical re-audit.
- Replay: `scripts/evaluate_nuclear_quark_gluon_correlations.py`, prints
  calculated values saved in `nuclear_quark_gluon_correlations_results.json`.
- Nine originals: commit `3f4ee58e4733421ae06780052dbc100677a8b9ef` and
  `computation_handoff/backups/2026-09-11_pre_nuclear_quark_gluon_correlations.tar.gz`.
  All nine archive members are byte-identical to that commit. Archive SHA-256:
  `67c3c51d0326af029ef229f1a92780ca4f24c44bda64f339ca85900aeb03c1d4`.
- Main TeX, bibliography, human/machine handoffs and roadmaps updated locally.
  No new volume, PDF, public push or review export; no historical full-note
  re-audit or new bound-nucleus result is implied.

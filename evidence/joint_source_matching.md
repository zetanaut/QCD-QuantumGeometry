# Joint source matching and conditional data — 10 September 2026

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

The common action now contains an independent residual electric-plus-
magnetic source direction, with its required higher-source partners. It
does not change source-free dynamics or erase the calculated reference
loop. The old conditional polarizability cannot uniquely match that new
direction. The missing experimental response remains explicit, not zero.

## Pre-evaluation recorded before implementation

The previous goal turn made verified progress at `6f0aa778`: linked quantum
Compton response and independent positive reference absorption. The next
bounded experiment adds a shared residual higher-derivative two-source
structure to the existing covariant O(4) source action. It must preserve
all multiplet coefficients, vector/axial source partners, source derivatives,
and the known loop. It must not introduce independent observable heads or
select a fitted residual coefficient. The operator does not change the
zero-source dynamics and is not a production Hamiltonian.

Derive its real-photon electric/magnetic map directly from the action;
test Lorentz and local non-Abelian covariance, two- and higher-photon Ward
identities, matching subtraction and parameter gradients. Establish the
rank and unmeasured directions of joint radius/alpha/beta matching.

In parallel audit the original COMPASS paper's complete equations/fit
assumptions. Its conditional alpha=-beta fit must not be converted into an
unconditional alpha or two independent data points. Derive the conditional
projection with acceptance and normalization nuisances explicit; do not
invent a physical covariance, digitize a figure into an unqualified fit, or
reuse its correction baseline as a second additive physics contribution.
An unknown experimental response can remain parameterized while the
operator/matching construction proceeds. No production model, physical
refit or public push is authorized by this checkpoint.

## 1. Shared action, not independent observable heads

Keep the existing real O(4) multiplets and full vector/axial connection
`A_mu`, `D_mu phi = partial_mu phi + A_mu phi`, curvature
`F_mu nu = partial_mu A_nu - partial_nu A_mu + [A_mu,A_nu]` and metric
`(+---)`. The added residual structure is

\[
 \Delta\mathcal L_E=-E_{ij}(D_\mu\phi_i)^T
 \mathcal F^{\mu\rho}\mathcal F_\rho{}^\nu D_\nu\phi_j,
 \qquad E=E^T,\quad [E]=\mathrm{GeV}^{-4}.
\]

The real-field transpose, not Hermitian conjugation, is analytically
continued when extracting plane-wave coefficients. All C,D,E entries are
retained on a common, arbitrary-sized multiplet node space. No coefficient
is fitted, no new scalar potential is selected, and no production quark-
meson Hamiltonian is implied. This dimension-eight operator is a bounded
source-basis extension, not the complete higher-derivative EFT basis or an
all-orders QCD matching result. Its zero-source value and field gradient
vanish. Its one-photon vertex is zero; its two-, three- and four-photon
electromagnetic vertices do not vanish in general. Non-Abelian partners
remain in the full local polynomial; every higher non-Abelian tensor has
not separately been evaluated.

For both photons incoming, `p1=p0+q1+q2`, let

\[
 W(p,q,\epsilon)=(p\cdot q)\epsilon-(p\cdot\epsilon)q.
\]

The independently reduced two-photon contact is

\[
 T_E=2e\{W(p_1,q_1,\epsilon_1)\cdot W(p_0,q_2,\epsilon_2)
       +W(p_1,q_2,\epsilon_2)\cdot W(p_0,q_1,\epsilon_1)\},
 \qquad e=u_1^TEu_0.
\]

It agrees with the actual fourth derivative of the action for off-shell
momenta and is transverse in each photon separately. Taking fifth/sixth
action derivatives supplies the covariant-derivative seagulls. They obey

\[
 q_N\cdot T_N=T_{N-1}(p_1,p_0+q_N)-T_{N-1}(p_1-q_N,p_0)
\]

with the remaining photon arguments unchanged. In the saved unfitted
fixture the three- and four-photon contractions are respectively
`-6.7809344e-6` and `-2.8139840e-5`, with identity residuals below `1.1e-20`.
Deleting these contacts fails a nonzero identity; a separately fitted
two-photon output would not preserve it. Random local O(4) transformations
include the inhomogeneous connection and curvature commutator. Independent
Lorentz boosts, node-coordinate rotations and all coefficient gradients
are also tested. Node count is varied over one, two and three.

## 2. Joint low-energy matching and surviving information

For one normalized external mode use `c=u^T C u`, `d=u^T D u`, `e=u^T E u`
and `K=2 alpha_em (hbar c)^3 / m`. At the stated local order,

\[
 \begin{pmatrix}r^2\\\alpha\\\beta\end{pmatrix}
 =\begin{pmatrix}6(\hbar c)^2c\\K(c+2d+m^2e)\\-K(c+2d)\end{pmatrix}
 +\begin{pmatrix}r^2_L\\\alpha_L\\\beta_L\end{pmatrix}.
\]

The last vector must already use the same external normalization, mass,
mode, subtraction and order. It is not a bare-matrix subtraction. Source-
free or nonlocal quantum matching cannot be inferred from these three
numbers. The earlier reference loop remains explicit and is counted once.
Low-energy real-photon action derivatives at forward, intermediate and
backward angles independently verify the electric/magnetic map.

For **three independent supplied conditions**, subtract the loop once,
calling the residual values `(R,A,B)`. Then

\[
 c=R/[6(\hbar c)^2],\qquad
 d=\tfrac12(-B/K-c),\qquad e=(A+B)/(Km^2).
\]

This is a parameterized inverse, not a physical data fit. The saved
synthetic closure recovers its parameters to `4.3e-15`; fitting without
the loop and then adding it produces exactly the duplicate loop. Even
three independent conditions identify only three projections: two
multiplets contain nine independent symmetric C,D,E entries, leaving six
directions unresolved. An explicit light/heavy transition changes along
one of these directions although all three diagonal-mode observables do
not. These are physical source directions in the fixed basis, not all
coordinate redundancies and not permission to set them to zero.

An arbitrary E is not automatically physically admissible. Power counting,
validity domain, matching, causal/unitarity constraints on a complete
physical response and multi-observable data still matter. Positivity of a
full physical response is not imposed on a full-minus-retained residual by
itself, and unconfined reference absorption is not a physical QCD bound.

## 3. What the COMPASS measurement does and does not provide

The [original COMPASS paper](https://arxiv.org/abs/1405.6377), Eq. (1), has
two different dipole response functions. With `z_±=1±cos(theta_CM)` and
polarizabilities in GeV^-3, the change of `d sigma/d Omega` in GeV^-2 is

\[
 \delta\frac{d\sigma}{d\Omega}
 =-\frac{\alpha_{\rm em}m^3(s-m^2)^2}
 {4s^2(sz_++m^2z_-)}
 \left[z_-^2(\alpha-\beta)+\frac{s^2}{m^4}z_+^2(\alpha+\beta)\right].
\]

The published `alpha=(2.0 ±0.6_stat ±0.7_sys) x10^-4 fm^3` is conditional
on `alpha=-beta`. Luminosity is fitted too. Their Eq. (3) ratio uses a
corrected pointlike denominator, including radiative, chiral-loop and
nuclear-form-factor effects; it is **not** just a bare low-energy formula
applied at finite energy. Their quoted `+0.6 x10^-4 fm^3` extraction shift
from these corrections is not a new physical source coefficient to add to
our loop. These statements were checked on the original equations and
fit discussion (PDF pages 4, 6 and 7).

The [official public paper list](https://wwwcompass.cern.ch/compass/publications/papers/)
links a source archive for this paper but no tabulated-data link in its
entry. Bounded inspection of the paper/public materials did not recover
an acceptance-folded two-template response, covariance or general
polarizability likelihood. The exact INSPIRE record is 1298026; its exact
HEPData route returned 404. This is **not proof that further data do not
exist**, nor a request to stop all other work. No acceptance model,
experimental covariance, figure-derived pseudo-data or uncertainty band
has been invented. The paper's correction model also has to be related
consistently to any replacement response model before fitting.

## 4. Correct conditional-response family

Let `d_-`, `d_+` be actual detector-folded difference/sum templates in a
common observable and unit convention. Linearize around a declared common
corrected baseline; include normalization and other floated parameters
in nuisance columns B. For positive-definite data covariance V, whiten by
its Cholesky factor and project out the whitened B column space. Denote
the resulting precision operator `W_perp`. Then the local conditional fit
to `sum=0`, `diff=2 alpha_cond` returns

\[
 \alpha_{\rm cond}=\tfrac12(\alpha-\beta)+\kappa(\alpha+\beta),\qquad
 \kappa=\frac{d_-^T W_\perp d_+}{2d_-^T W_\perp d_-},\qquad
 \operatorname{Var}(\alpha_{\rm cond})=[4d_-^TW_\perp d_-]^{-1}.
\]

This is a linear-response/GLS formula, not a substitute for the complete
nonlinear experimental likelihood. A normalization nuisance has its
actual baseline-shaped column, not an assumed constant in arbitrary
observable coordinates. The saved **synthetic** template example changes
kappa from `0.345439` to `-0.961951` after profiling a constant offset.
Neither number belongs to COMPASS. Profiling can change even the sign,
so positive pointwise template ratios do not justify a physical bias bound.
Direct GLS, unit-rescaling, loss of identifiability and distinct forward/
backward dipole templates provide independent controls.

For supplied radius and conditional alpha, define
`L_cond=(alpha_L-beta_L)/2+kappa*(alpha_L+beta_L)`. Keeping E explicit,

\[
 c=\frac{r^2-r_L^2}{6(\hbar c)^2},\qquad
 d(e,\kappa)=\frac12\left[
 \frac{\alpha_{\rm cond}-L_{\rm cond}}K
 -m^2(\tfrac12+\kappa)e-c\right].
\]

Even **known** kappa leaves one projected direction free. Unknown kappa
adds experimental-response uncertainty, not an extra measured condition.
The implementation requires both kappa and e explicitly and neither has
a silent zero default. Synthetic families preserve the two conditions
while varying the physical dipole sum. No physical C/D/E has been changed.

## Reassessment and next scientific experiment

This source extension and conditional-data audit are complete at their
declared order. A physical unconditional polarizability refit is not
supported by the recovered scalar summary. That does not block portable
shared-action or network construction.

The [next pre-evaluation](source_network_transport.md) identifies a concrete
equal-virtuality information loss to test in the actual source transport.
Next qualify source-preserving transport into the shared finite-
momentum quark/quantum network. Determine which full-matrix directions
are accessible through its coupled propagators/endpoints and which remain
hidden; qualify mixed interaction/source derivatives and retain heavy-
light transitions. Distinguish off-shell matching kernels from measured
observables. Test strategic physical mode connectivity before promoting
an optimizer graph, and do not make an independent polarizability output
head or a new fitted pion model. Retained modes are quantized once; the
computed loop must not also be requantized as bare dynamics. Continue
toward jointly generated p/n states and currents, independent covariance
and withheld data, modular nuclei and GTMD/TMD/GPD matching/evolution.

## Implementation, validation and recovery

- Source: `src/deuteron_wigner/quantum/joint_source_matching.py`.
- Direct tests: `tests/test_joint_source_matching.py` — 25, including exact replay.
- Replay: `scripts/evaluate_joint_source_matching.py`, saved in
  `docs/next_level/joint_source_matching_results.json`.
- Originals: Git `6f0aa77881fd47d1e89588fddd4f448d2b22d96c` and
  `computation_handoff/backups/2026-09-10_pre_joint_source_matching.tar.gz`.
  The archive contains the seven pre-edit entry/theory files, not a new
  public export. Prior scientific source/results remain unchanged.

The new tests pass with warnings treated as errors. All 438 focused
science/theory tests pass together (12 existing warnings, 127.07 seconds).
The seven archived originals match 6f0aa778 byte for byte. The known
unrelated historical full-suite failure is not hidden or repaired here.
The main theory's governing result section is updated; its entire
historical body has not been re-audited in this step. No PDF was rebuilt,
no production model was selected and nothing was pushed publicly.

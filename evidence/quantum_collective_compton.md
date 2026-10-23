# Quantum collective two-photon matching and independent absorption

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

The bounded two-photon experiment passes. Six box orderings, linked
inverse/triangle/Born terms and pole normalization satisfy the independent
Ward, gauge, crossing and boost controls. The same potential generates an
additional scalar-exchange response that a lone box omits. The reference
finite-mass loop gives **alpha+beta = 2.3972940e-5 fm^3**. An independently
resolved, positive tree absorption integral reproduces it within 1.1e-13
fm^3 at the saved resolutions. This is an unconfined reference-quark
identity, **not a physical pion prediction or hadronic sum-rule bound**.

No local coefficient or physical parameter was fitted. The old C/D local
terms cannot independently adjust this extra structure. Keeping the loop
explicit already represents its contribution; an additional local operator
is not needed to restore its gauge invariance and must not be introduced
to cancel the verified sum by fiat. The next step qualifies an independent
residual two-source structure and joint matching, then audits the old
conditional data interpretation before fitting.

## Pre-evaluation, 10 September 2026

The charged triangle checkpoint (`af11d2bf`) supplies a genuine independent
transverse slope but does not update the old joint radius/polarizability
calibration. The next bounded calculation uses the SAME reference quark
determinant, not a newly selected NJL/sigma/BLFQ production model. Derive all
six photon/pion cyclic box orderings, the linked inverse/triangle/Born terms
and external pole normalization, and the potential-linked scalar exchange
with its two-photon quark triangle. Keep the common mode/source matrices.
Evaluate below the free-quark cuts, without resummation or parameter fits.

First qualify the subtracted box against its one-photon Ward difference,
zero-transfer inverse derivative, photon crossing and original Dirac trace.
The physical real-photon amplitude must then be gauge independent after
Born and pole terms, not only in a convenient transverse gauge. Extract
both low-energy electric and magnetic coefficients; do NOT impose the old
derivative-level alpha=-beta relation on the finite-mass loop. If the loop
adds a second independent structure, the old two-input C/D arithmetic is
insufficient and the matching basis/data interpretation must be reassessed.
The scalar exchange is required by the same action even though its two
photon Ward contractions vanish separately. This is a bounded matching
control, not authorization for an open-ended reference-model completion.

No local C/D values, quantum vacuum parameters, production state or public
repository are changed by this pre-evaluation. Existing physical inputs
are reused only as diagnostic calibration conditions, never predictions.

## Order, fields and diagram ownership

Use the preceding charged triangle's **same** positive-mass free-quark
determinant, Nc=3, degenerate u/d, Q=(2/3,-1/3), external pion vertices
i*gamma5*sqrt(2)*tau+/-, and two collective multiplets. Electric e^2 is
stripped from amplitudes; it is restored in absorption and polarizabilities.
The numerical point remains M=.2843478671 GeV, prescribed external pole
m_pi=.13957039 GeV, and g_light^2=9.245548276. None is a quantum-fitted or
QCD-generated particle mass/coupling.

At this order the EM amplitude includes:

1. All six cyclic quark box orderings with fixed outgoing pion insertion.
2. The two charged-pion Born channels, each expanded to include either
   corrected triangle and the linked inverse insertion, once each.
3. External pole normalization and the common reference pole condition.
4. The scalar two-photon quark triangle joined to the **third derivative of
   the same potential**, through both tree scalar modes.

The first three form one gauge-invariant contribution; the fourth is
separately transverse but not optional in this action. No scalar width is
inserted: the exchanged transfer is spacelike and tree scalar response is
the correct order beside a one-loop triangle. This does not reinstate the
unjustified heavy-mode narrow-width interpretation of the earlier cut test.
Gluon and collective loops, full finite/chiral/source matching, local
C/D-by-loop dressing and changes of the quantum vacuum are not included.

## Box derived from the determinant

All photons are incoming, p1=p0+q1+q2. Fix the outgoing pion vertex, then
permute (incoming pion, photon 1, photon 2). For each of the six orders,
route propagator momenta k+r_i, with

\[
r=(0,t_1,t_1+t_2,p_1),\quad
\bar r=\sum_i a_i r_i,\quad h_i=r_i-\bar r,\quad
\Delta=M^2-\sum_{i<j}a_i a_j(r_i-r_j)^2.
\]

The simplex weights a_i are nonnegative and sum to one. The original
Dirac trace, after angular integration, is

\[
\left\langle\operatorname{tr}_D\prod_{i=0}^3
[V_i(\slashed l+\slashed h_i+M)]\right\rangle
=A+B l^2+C l^4.
\]

A is the constant trace; B contracts every pair of propagator numerators
with g_mu_nu/4; C contracts all three fourth-moment pairings with /24.
An independent spherical fourth-degree cubature checks these coefficients
from the uncontracted traces: eight axis points plus sixteen half-coordinate
hypercube corners, each with weight 1/24, with imaginary spatial coordinates
to implement the Minkowski contractions. It is not another fitted formula.

With f_o the **actual ordered flavor trace**, the subtracted box is

\[
\widehat B(\epsilon_1,\epsilon_2)=
-\frac{N_c}{16\pi^2}\sum_{o=1}^6 f_o\int_{\cal S}d^3a
\left[\frac A{\Delta^2}-\frac{2B}{\Delta}
-6C\log\frac\Delta{M^2}-\frac{A_0}{M^4}+\frac{2B_0}{M^2}\right].
\]

Here A0,B0 set all external momenta to zero, preserving the same photon
polarizations and flavor order. Momentum-independent evanescent local
terms cancel in this subtraction, as does the divergent wave term. This
does not subtract or infer a transverse physical polarizability. The code
certifies Delta>0 over the **whole** simplex using
max(r_i-r_j)^2 < 8 M^2/3 and sum_{i<j}a_i a_j <=3/8; this sufficient bound
is deliberately narrower than the complete analytic subthreshold domain.
It does not merely check the sampled quadrature points.

The off-shell test independently evaluates

\[
\widehat B(q_1,\epsilon_2)=\epsilon_2\cdot
[\widehat\Gamma(p_1,p_0+q_1)-\widehat\Gamma(p_1-q_1,p_0)].
\]

At zero photons, with s=p^2 and deltaZ=-DeltaK'(s),

\[
\widehat B(\epsilon_1,\epsilon_2;0,0)
=2\epsilon_1\cdot\epsilon_2\,\delta Z
+4(p\cdot\epsilon_1)(p\cdot\epsilon_2)\delta Z'(s).
\]

Thus the Thomson subtraction is checked against the previous inverse's
second derivative, not tuned to a numerical box. Equal flavor charges
give zero single-photon current and zero soft charge term, but can still
give a nonzero two-photon amplitude; single-photon information is insufficient.

## Pole/Born completion and covariance

Use K0(z)=s-z and Sigma_R(z)=DeltaK(z)-DeltaK(s), fixing this prescribed
reference pole only. For either intermediate pion momentum l, the order-one-
loop Born contribution is

\[
\delta\Gamma_L R_0\Gamma_{0R}
+\Gamma_{0L}R_0\delta\Gamma_R
-\Gamma_{0L}R_0\Sigma_R R_0\Gamma_{0R},\qquad R_0=1/(s-l^2).
\]

Both channels are retained. The normalized correction is
deltaT_box+Born = Bhat + deltaT_Born - deltaZ*T_tree, with
T_tree=2 eps1.eps2+T_Born,0. No Dyson denominator is used. External
off-diagonal mode rotations do not change the identity-valued minimal
diagonal charge vertex at this order; arbitrary local source operators
would require their own transported mode/wave partners before a bare
matrix could be refitted. The open box kernel remains proportional to
g_i*g_j, not two independent diagonal loop rates.

In initial-rest radiation gauge each Born product has a zero endpoint.
That is useful numerically, **not** permission to omit those diagrams:
after eps1->eps1+.7*k and eps2->eps2-.9*k', the saved box changes by
4.1519361e-4 while the complete normalized loop changes by only 1.6e-17.
Lorentz-boosted four-vectors give the same amplitude within 1.6e-17;
quadrature refinement changes it by 5.3e-18. These are covariant loop tests,
not independent quantum LF minus/transverse or endpoint calculations.

## Scalar/source partner from the same potential

For real photons, t=(q1+q2)^2, the scalar two-photon coefficient is

\[
a(t)=\frac{N_c\operatorname{tr}Q^2 M}{8\pi^2}
\int_{x,y\ge0,x+y\le1}dx\,dy\,
\frac{1-4xy}{M^2-txy},\qquad
a(0)=\frac{N_c\operatorname{tr}Q^2}{24\pi^2 M}.
\]

It multiplies e^2*g_i*sigma_i*F_em^2. Its real-photon vertex is -4*a(t)
times [(q1.q2)(eps1.eps2)-(q1.eps2)(q2.eps1)]. An independent two-order
original Dirac trace agrees. In four dimensions its l^2 angular term
vanishes; the evanescent finite term removes the zero-momentum sigma*A^2
artifact, equivalently subtracting that gauge-forbidden constant trace.
No sigma*F^2 coefficient is subtracted.

As an independent source check, differentiating the photon two-point loop
with respect to M **holding its reference subtraction scale fixed** gives
the zero-scalar-momentum triangle. Re-subtracting at every varied M erases
the soft scalar response and fails. This is a quark mass-source identity,
not the complete interacting mass-source or axial matching prescription.

Let Hs be the same tree scalar Hessian. The potential derivative is
T_aij=2 lambda_a v_a delta_ai delta_aj, hence

\[
(b_\sigma)_{ij}(t)=a(t)\sum_{ab}g_a(H_s-t)^{-1}_{ab}T_{bij}.
\]

The projected real-Compton contribution is -4 b_sigma times
I=(k.k')(eps.eps')-(k.eps')(k'.eps). Both scalar modes contribute at t=0:
.32411320 and .46072854 GeV^-2, summing to .78484174. The open external
pion matrix is diag(.99547368,.33481499) GeV^-2 in the original potential
coordinates. It has rank two, unlike the rank-one quark box; deleting
one mode or replacing every source structure by g_i*g_j loses information.
Basis tests transport Hs, g and the full potential third derivative.

## Two independent low-energy structures and absorption check

For initial-rest physical transverse photons at leading photon energy,

\[
\delta T=\frac{2m_\pi}{\alpha_{\rm em}}
\{\alpha\,\omega\omega'\,\boldsymbol\epsilon\cdot\boldsymbol\epsilon'
+\beta(\mathbf k\times\boldsymbol\epsilon)\cdot
(\mathbf k'\times\boldsymbol\epsilon')\}+O(\omega^3),
\]

where alpha,beta are GeV^-3 before multiplying by (hbar*c)^3. Forward and
backward perpendicular polarizations isolate alpha+beta and alpha-beta.
The replay varies photon energy .02 down to .00125 GeV and extrapolates
the final three values quadratically in omega^2 (forward) or omega*omega'
(backward). This extrapolation is numerical control, not a physical fit.

| Reference one-quark-loop contribution | alpha (10^-4 fm^3) | beta (10^-4 fm^3) |
|---|---:|---:|
| Box + linked Born + normalization | -.48950846 | .72923786 |
| Potential-linked scalar exchange | 6.30584851 | -6.30584851 |
| Sum, before local/other quantum terms | 5.81634005 | -5.57661065 |

Independently compute gamma+pi+ -> u+dbar from the two fermion attachments
AND the charged-pion source attachment. Canonical u/v spinors agree with
the resolved Dirac trace; omitting the pion attachment violates Ward even
though that attachment vanishes in transverse CM gauge. Include Nc,
the charged flavor factor 2 and the average over two initial photon
polarizations exactly once. The positive tree cross section is

\[
\sigma(s)=\frac{\sqrt{1-4M^2/s}}{32\pi(s-m_\pi^2)}
\int_{-1}^{1}dc\,\frac12\sum_{\epsilon,s_q,s_{\bar q},c}|\mathcal M|^2.
\]

Restore e^2 here. The independent forward dispersion integral is

\[
\alpha+\beta=\frac1{2\pi^2}
\int_{\omega_{\rm th}}^\infty\frac{\sigma(\omega)}{\omega^2}d\omega,
\quad\omega_{\rm th}=\frac{4M^2-m_\pi^2}{2m_\pi}.
\]

Using s=4M^2/(1-b^2), radial/angular orders (48,96), (96,192), (192,384)
give 2.3972958561e-5, 2.3972941523e-5 and 2.3972940434e-5 fm^3. The loop
extrapolation is 2.3972940328e-5. Their difference 1.06e-13 fm^3 and the
refinement trend establish that the nonzero sum is not a subtraction or
numerical artifact. This integrates the unconfined reference to infinity;
it is neither physical hadronic absorption nor a bound on a QCD pion.

## Reassessment and next step

The old **local C/D derivative terms** give alpha=-beta and cannot
independently change alpha+beta. The explicitly retained finite-mass loop
already supplies the nonzero structure, so the full loop-plus-C/D action
does NOT obey that zero-sum restriction. For general residual matching,
qualify an independent higher-derivative two-source operator and its
symmetry/source partners, with open multiplet indices and information-
preserving parameter derivatives. It must add flexibility, not erase a
known contribution or invent separate observable heads.

The old COMPASS alpha calibration was extracted assuming alpha=-beta.
It must not be treated as two independent measurements, or silently reused
as an unconstrained alpha value when a nonzero sum is retained. Derive the
parameterized joint radius/alpha/beta matching map and inspect the required
independent data or likelihood interpretation **before any physical fit**.
No new executive choice is needed for that bounded operator/data audit.
No arbitrary E coefficient, bare C/D update, production action or physical
pion/nucleon state is selected here. Do not start a full quark-meson model
completion agenda. Return the qualified source structures to the original
joint quantum state/interaction and modular-nucleus construction.

Canonical QCD gluons, quantum vacuum/state matching, all source/chiral
partners, independently evaluated LF quantum minus/transverse responses,
generated p/n states, withheld covariance/data, modular nuclei and
GTMD/TMD/GPD matching/evolution remain open. Retained modes are still
counted once: the same loops cannot also be independently inserted into
a solver that already retains those quark fluctuations.

## Validation, recovery and primary context

Module `src/deuteron_wigner/quantum/quantum_collective_compton.py`, direct
tests `tests/test_quantum_collective_compton.py`, replay
`scripts/evaluate_quantum_collective_compton.py` (prints JSON only), and
[saved results](../snapshot/docs/next_level/quantum_collective_compton_results.json). All 24 new tests
pass with warnings treated as errors; 413 focused science/theory tests
pass together with 12 existing warnings. The saved replay reproduces
exactly and all seven archived originals match af11d2bf byte for byte.
This is not a claim that the historical full repository suite is green.
Seven pre-edit entry/roadmap/theory documents are preserved at `af11d2bf`
and `computation_handoff/backups/2026-09-10_pre_quantum_collective_compton.tar.gz`.
No unrelated file, public repo or review mirror was changed.

These are established diagrammatic/dispersion mechanisms, not claimed
original inventions. For primary context on separate box, pion-exchange
and scalar contributions see [Hiller et al., pion polarizabilities](https://arxiv.org/abs/0908.0159).
Its NJL model, regulator, parameters and physical conclusions are NOT
adopted. For independent tree-cross-section/loop sum-rule methodology see
[Holstein, Pascalutsa and Vanderhaeghen](https://arxiv.org/abs/hep-ph/0507016).
The displayed amplitudes, full two-multiplet potential/source map and
independent numerical checks are evaluated here in the project's conventions.

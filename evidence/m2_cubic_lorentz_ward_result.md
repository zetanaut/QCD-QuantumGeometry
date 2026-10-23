# Cubic spin source and common-regulator Lorentz variation

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

9 September 2026. Local continuation of free source `b3a3a42d`; initial
pre-edit checkpoint `426a8243`. The subsequent canonical-density derivation
continues from cubic source `dffef768` / documentation `e1146be5`.
The all-channel continuation is source `90ca8344`; its complete pre-edit
source and notes remain in Git checkpoint `d2784a2e`.
Milestone 1 is in progress;
milestones 1–4 remain an active persistent goal, not completed milestones.

## Established result

The canonical local Noether fermion spin density yields a nonzero cubic
qqq→qqqg action in the actual global-Jacobi, color and spin/flavor basis.
It preserves the full joint permutation-sign and isospin carrier and has
the correct transverse-vector Jz weight. Separately, the bare source q→qg
vertex satisfies an independently differentiated first-order Lorentz Ward
identity. Multiplying its declared Ward representative by the current
Gaussian regulator leaves the explicit residual −i(DR)C. The latter is
generically nonzero for massive quarks, before any finite projection.

The canonical-density continuation below now derives the missing coupled
coordinate-density/gluon-constraint terms for **all four bare cubic channels**
and reproduces their Ward representatives independently. The physical
antiquark free action is also derived from the source v-spinor Lorentz
transformation. This does not supply quartic, regulated finite-domain or
full many-body/H2 spin closure.

This is **not a complete interacting rotation generator or a repair of the
nucleon current discrepancy**. In particular the local fermion spin density
and the Ward representative's contact are different objects and must not be
identified or added to each other. Existing Hamiltonian/current kernels,
regulator, fit and state choices are unchanged.

## Source conventions and derivation

Use symmetric light-front coordinates, the source helicity order (−,+),
the source gamma matrices and good-field axes G. Write

    B0(p) = (alpha_T.p_T + beta m)/(sqrt(2) p+)
    phi0(p) = (I+B0(p)) G / 2^(1/4)
    Hi = gamma0 {gamma+, sigma^{-i}}/4
    sigma^{-i} = i[gamma-,gamma^i]/2.

The local fermion spin kernel is phi0(out)† Hi phi0(in). At equal momenta
it reproduces the preceding free F1/F2 spin matrices. With the constrained
linear field insertions Br=alpha.epsilon/(sqrt(2)r+) and
Bl=alpha.epsilon/(sqrt(2)l+), its cubic coefficient is

    G† [(I+Bf†) Hi Br + Bl Hi (I+Bi)] G / sqrt(2).

The same Fourier polarization appears in both terms: epsilon* for an emitted
gluon. The signed constraint denominators are explicit. Independent four-
component field expansion confirms this expression, its reversal, mass
independence and the vanishing quadratic-in-field contribution to this
fermion spin density. That last zero does not eliminate quartic terms from
the complete rotation generator's coordinate-density moment.

The analogous free gluon density F^{+i}A^-−F^{+-}A^i recovers the free gluon
spin matrices. Its interaction-dependent A^- contribution is not supplied
by this free check.

The canonical decomposition in [Harindranath, Mukherjee and Ratabole
(2000)](https://arxiv.org/abs/hep-th/0004192) guides the full construction:
coordinate-density, fermion and gluon contributions must be treated together.
That paper uses different light-front normalization and drops surface terms
in its vanishing-field domain. Those factors/domain simplifications have
not been imported into this finite APBC/PBC calculation.

## Before-projection cubic Ward identity

For independent positive physical plus/transverse momenta, define the
explicit off-conservation extension

    C(pf,k,pi) = −ubar(pf) slash(epsilon*) u(pi)
                 / sqrt(8 pf+ pi+ k+).

This is per g_s and color. On spatial momentum conservation the existing
discrete mass-squared source is exactly 2P_N+/sqrt(2L) times C, with
2L=pi K2/P_N+. No normalization is chosen by fitting this comparison.

Let Delta=(pf+k−pi)_{+,T}, Delta E=pf^-+k^-−pi^-, and D_i be the sum
of the three free F_i momentum-vector derivatives. With the source free
spin matrices A and flat-measure divergences d, the regular Ward identity is

    −i[D_i C + (df+dg+di)C/2]
    + Aq_f C + Ag_h C − C Aq_i − Delta E Z_i = 0,
    Z_i = −i epsilon_i*/[k+ sqrt(2k+)] I.

The −i C partial_{Delta_i} delta(Delta) coordinate term cancels the
derivative-of-delta part of [F0,V]+[F1,P0^-]. The contact follows from
gauge restoration of the physical light-front polarization: J.k=Delta E J+
on the spatial conservation surface. **No residual is divided by Delta E.**
The massless collinear Delta E=0 limit remains regular at positive k+.

This is a distributional representative for the declared off-conservation
extension, not a proof that it equals the complete canonical Noether
generator. Changing C away from Delta=0 can leave the Hamiltonian unchanged
while changing its coordinate-density first moment. The canonical local
density and all its improvement/surface partners must therefore be derived
together. The before-projection q→qg comparison is now derived below;
the finite-domain surface action remains open. Neither result establishes
the F1/F2 algebra through g².

## Canonical coordinate-density and Gauss-constraint completion for bare q→qg

This continuation uses the source D_mu=partial_mu−ig_s A_mu and the
hermitized Dirac Lagrangian. At fixed x+, remove only the time-derivative
Legendre term, not spatial divergences. The spatial quadratic form is

    K(p,q)=gamma0[−gamma−(p+ +q+)/2
                   +gamma_T.(p_T+q_T)/2+m].

Let r=pi−k, l=pf+k and q=pi−pf; their signed nonzero plus components are
the constrained Fourier momenta, not arbitrarily chosen denominators.
With V=alpha_T.epsilon_T*, R=VG/[sqrt(2)r+ 2^(1/4)] and
L=G†V/[sqrt(2)l+ 2^(1/4)], the canonical fermion density differs from
the Dirac extension by

    D_f = [phi0(pf)† K(pf,r) R + L K(l,pi) phi0(pi)]/sqrt(2k+).

Independently, the source Gauss law gives
(partial+)² A^-_[q]=g_s J+, or A^-_[q]/g_s=−J+/(q+)².
The local gauge Legendre density contains

    −(partial+ A^-)²/2 − (partial+ A^i)(partial_i A^-)+F_ij²/4.

Its cross term and its Noether spin contribution are

    D_g = −k+ [epsilon_T*.q_T−q+ epsilon−*] I
               / [(q+)² sqrt(2k+)],
    S_g,i = −i(k+ +q+)epsilon_i* I/[(q+)² sqrt(2k+)].

Thus C_can=C+D_f+D_g. Both D terms vanish at spatial conservation, but
their transverse Delta derivatives do not. Direct differentiation at fixed
pf,k (pi=pf+k−Delta) gives

    S_f,i + i partial_Delta_i D_f = 0,
    S_g,i + i partial_Delta_i D_g = Z_i,
    −i C_can partial_Delta_i delta + (S_f,i+S_g,i)delta
      = −i C partial_Delta_i delta + Z_i delta.

Here S_f has its gluon field factor restored. This is cancellation of the
local fermion density against its coordinate-density improvement, **not
removal of physical quark spin** from the complete generator. Canonical
T^{+i} remains kinematical in A+=0 and adds no cubic q→qg x−-moment term.
This continuum distributional comparison does not discard or determine
the subsequent finite-cell surface action.

The contact is computed from the Legendre/Noether fields, independently of
the Lorentz Ward matrices and without energy-gap division. The eight saved
canonical cases have a largest combined-contact residual 1.91e−15
GeV^(−3/2); independent finite-difference density derivatives agree within
8.33e−13 GeV^(−3/2). Forty-three new tests additionally check the direct
four-component Fourier-field Legendre expansion, Gauss density/spin,
masses, helicities, frames, excluded zero modes and conservation domain.

Multiplying this entire bare canonical density and spin by the same smooth
R gives contact R Z: D_f+D_g vanishes on the conservation surface, so its
first-moment improvement scales by R there. The independent regulated Ward
residual is still −i(DR)C. The now-completed bare q→qg cancellation does not
create the missing Lorentz-regulator partners or complete the other channels.

## The actual common regulator is not a generic Lorentz scalar

For all positive participating physical legs, including creation and
annihilation legs, write W=sum p_a+ and P_T=sum p_aT. The current source
uses

    Q_T = [W sum(p_aT²/p_a+) − P_T²]/2,
    R = exp[−Q_T/(2 Lambda²)]

or its existing reference-shape polynomial times Gaussian. For F_i,

    D_i Q_T = [W sum(m_a² p_ai/(p_a+)²)
               − (sum p_ai) sum(m_a²/p_a+)]/2.

This vanishes for all massless legs and for the tested kinematic subgroup;
it generally does not vanish for the massive-quark dynamical rotations.
The same independently differentiated Ward expression with RC and RZ has
residual **−i(D_i R)C**. Polynomial reference shapes are differentiated as
whole kernels, not treated as plain Gaussians. Finite field membership,
longitudinal cell domains and oscillator projection are separate effects.

For comparison only,

    Q_L = Q_T + W sum(m_a²/p_a+)/2 = (sum p_a^mu)^2/2

is a scalar for on-shell legs; its mass-completion derivative cancels the
displayed drift. This diagnostic does **not** select a replacement regulator,
its normalization or counterterms, and does not establish common-loop or
gauge compatibility. It is not used in the production Hamiltonian/current.

## Actual finite Fock action

The local fermion term is integrated with the unchanged common Gaussian
physical-leg kernel and exact circular-HO/Jacobi measure. Its discrete field
factor is 1/sqrt(2 pi k2); **there is no 2P_N+ conversion for F_i**. All three
quark emitters and source color/flavor maps are retained coherently.

For ground orbitals the independent radial result is

    sqrt(xq xg/xp) b/sqrt(pi) / (1+xp b²/Lambda²)

times the local spin coefficient and field factor. The action scales as
1/P_N+ at fixed modes and transverse width, as required for this generator
component. Absorption is the adjoint of this same labelled piece.

The old basis helper restricted twice-Jz to ±1. Its exact symmetry projector
now accepts all odd integer weights, including empty weights outside the
finite shell. This does not assign total spin or change the existing ±1
bases. Tests compare the extended ranks to independent dense projectors.

## All bare cubic channels: signed crossing and pure-gluon density

The additional derivation is implemented in `m2_cubic_spin_channels.py` and
written out in theory equations `eq:cubic-crossed-good-fields` through
`eq:cubic-gluon-contact-improvement`. It retains the signed Fourier creation
momenta of the source constraints, while every external physical momentum
has positive p+. For physical daughters p1,p2 and parent p0, the signed
fermion field triples (pf,k,pi) are:

| Channel | Signed triple | Field axes (out,in) | Polarization |
| --- | --- | --- | --- |
| q→qg | (p1,p2,p0) | (G,G) | eps2* |
| qbar→qbarg | (−p0,p2,−p1) | (A,A) | eps2* |
| g→q qbar | (p1,−p0,−p2) | (G,A) | eps0 |

Here A=i gamma2 G*. Replace the physical gluon field normalization by
sqrt(2 abs(k+)), but never take absolute values in the constraint inverses.
The antiquark result is transposed into physical outgoing/incoming helicity
order. Its CAR minus already belongs to Tbar=−T^T; no second minus is added.
The pair contact uses G†A, not I. Its physical spin action is Aq C + C Aanti^T
− Ag C. In each channel the independently derived local spin plus density
improvement equals the gauge-restoration Z, with no inverse energy gap.

For g→gg, all-outgoing Fourier order is (−p0,p1,p2), color order is
f^(a0 a1 a2), and polarizations are (eps0,eps1*,eps2*). The source Gauss
constraint includes both ordered gluons. The direct local YM cubic density
equals the source three-gluon numerator even off conservation. Its Gauss
cross term D vanishes on spatial conservation, but its normal derivative
is iZ, and the local spin is 2Z. Thus S+i dD=Z. Each color-stripped piece
is antisymmetric under daughter exchange, restoring Bose symmetry with f.
Many-body creation factors remain in their existing Fock actions.

The additional replay covers 160 combinations: every physical helicity,
both transverse axes, masses 0/.3 GeV for fermionic channels, and the plain
Gaussian/reference-shape degrees 0/1/2. Parameters and per-channel maxima
are in `all_bare_cubic_channels` in the existing saved result.

| Diagnostic over all four channels | Maximum |
| --- | ---: |
| Bare Lorentz Ward residual | 2.04e−12 GeV^(−1/2) |
| Regulated product-rule residual | 2.07e−12 GeV^(−1/2) |
| Independent canonical contact comparison | 3.16e−15 GeV^(−3/2) |
| Independently differenced density first moment | 8.33e−13 GeV^(−3/2) |
| Massive pair-channel smooth regulator drift | .02129549 GeV^(−1/2) |

The all-massless gluon smooth drift vanishes to roundoff. This is not a
claim of finite-domain covariance or higher-order algebra closure. The
source finite-cell M² conversion is independently checked once against
every existing five-/six-body cubic route type; no 2P+ conversion belongs
to the generator. Quark/antiquark Wigner, Lie-algebra, Pauli-Lubanski and
cell/Galerkin tests retain the source physical-helicity order.

All **784 focused tests pass in 29.30 s**. The new channel suite and extended
antiquark tests run together with all the preceding focused suites and
the existing five-/six-body source suites. This is bare-cubic completion,
not physical milestone 1 or a full-repository test run.

The six theory/status checks pass after synchronization. The offline scratch
build has 232 pages; the new equations/table and changed title/status pages
are visually checked under the PDF skill. No final-pass overfull boxes or
literal `??` markers occur. Generic inherited reference/bibliography/package
warnings remain. This is focused QA, not full-note scientific certification
or a requested PDF deliverable.

## Preceding q→qg and retained-fermion numerical evidence

[Saved results](../snapshot/docs/next_level/m2_cubic_lorentz_ward_results.json) contain 32 continuum Ward
cases and 16 actual Fock cases. At the declared point pf=(.7,.13,−.2),
k=(.4,.04,.07) GeV, pi=pf+k and Lambda=.8 GeV:

| Diagnostic | Result |
| --- | ---: |
| Largest bare Ward residual | 1.16e−12 GeV^(−1/2) |
| Largest regulated product-rule residual | 1.72e−12 GeV^(−1/2) |
| Massive m=.3 GeV, plain-Gaussian F1 drift norm | .001497033 GeV^(−1/2) |
| Same F2 drift norm | .008046552 GeV^(−1/2) |
| Largest raw sign/isospin/Jz leakage of the labelled fermion piece | 9.54e−17 |

At K2=9, oscillator cutoff 0, each tested qqq weight has rank 10 and each
destination qqqg weight rank 20. At K2=7, cutoff 1 the ranks are 22 and 40.
The F1 fermion-piece Frobenius norms are .06007–.08733 per g_s at the declared
P_N+=1.9 GeV, b=.3 GeV; the Jz vector identities close at machine precision.
These operator norms are neither probabilities nor an error estimate for
the complete spin/current calculation.

At source `4ad3a926`, **491 focused tests passed in 25.51 s**: 43 new canonical-density tests,
110 preceding cubic-spin tests, the free-spin,
constrained-current, canonical-emission, coupled-Jacobi/pilot and cluster-
interaction suites. Six new neighboring-weight tests are included in the
Jacobi suite; its existing dense-rank tests now cover the wider weights.
No full-repository test or physical qualification is claimed.

At that checkpoint the six existing theory/status checks also passed.
The preceding 230-page offline scratch build succeeded; its new equations and changed
title/status pages are visually checked. No final-pass overfull boxes or
literal `??` markers occur; generic inherited bibliography/reference/package
warnings remain. This is focused layout QA, not a full-note certification or
a requested PDF deliverable.

```sh
PYTHONPATH=src:. python scripts/analyze_m2_cubic_lorentz_ward.py \
  --output docs/next_level/m2_cubic_lorentz_ward_results.json
PYTHONPATH=src:. python -m pytest -q -p no:cacheprovider \
  tests/test_m2_cubic_spin_channels.py \
  tests/test_m2_canonical_cubic_spin.py \
  tests/test_m2_cubic_lorentz_ward.py tests/test_m2_free_spin_generators.py \
  tests/test_m2_constrained_fermion_current.py tests/test_m2_canonical_emission.py \
  tests/test_m2_coupled_longitudinal_jacobi.py tests/test_m2_coupled_pilot.py \
  tests/test_m2_gaussian_cluster_interactions.py \
  tests/test_m2_excluded_fock_sources.py tests/test_m2_six_body_vertices.py
```

## Next construction and recovery

Derive the quartic density moments and higher-order Lorentz commutators;
retain the source constraint momenta, improvements and surface terms.
Complete the common-regulator/Wick and finite-domain partners,
all required Fock transitions, common-order H2 waves/norm/connection transport,
and multiplet/current resolution controls. Only those combined tests can
explain and control the old two-structure discrepancy. Milestones 2–4 remain
downstream/iterative; no physical calibration or nuclear choice is silently made.

The eight pre-edit entry/theory/basis files are saved from `426a8243` in
`computation_handoff/backups/2026-09-09_pre_cubic_spin/theory_source_pickup.tar`.
All other originals, including tests, remain in that Git checkpoint. No
original volume, review mirror or public GitHub release is changed here.
The exact pre-canonical-density source and notes are additionally preserved
in Git checkpoint `e1146be53b08753a9616e98e183d66de03e432d4`.
The exact originals before the all-channel extension remain at `d2784a2e`;
no original volume or archived note was overwritten.

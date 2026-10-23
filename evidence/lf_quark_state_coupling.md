# LF quark/collective state coupling — result, 10 September 2026

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

The canonical normalization audit and the first nonzero-momentum **quark-pair
Fock-wave** coupling test pass. This advances beyond a connected one-body
collective wave, but only through **one bare Yukawa insertion** in a declared
free-particle reference. It is not a bound state, full quantum action, a new
production quark-meson model, or a repaired BLFQ/H2 result. The original
pre-evaluation is retained below, including its then-unverified candidate.

Source: `src/deuteron_wigner/quantum/finite_lf_coupling.py`;
tests: `tests/test_finite_lf_coupling.py`;
replay: `scripts/evaluate_finite_lf_coupling.py`;
record: [numerical results](../snapshot/docs/next_level/lf_quark_state_coupling_results.json).
Originals are at `ae9b7450b7014669e8182696c8e3831de247eb61` and the eight-file
`computation_handoff/backups/2026-09-10_pre_lf_quark_state_coupling.tar.gz`.
No old physics kernels or unrelated files are edited; no public push/export.

## 1. Derived normalization, not a guessed interaction scale

The primary canonical field convention is checked against
[Srivastava–Brodsky, Eqs. 10–11, 24 and Appendix B](https://arxiv.org/html/hep-ph/0011372v2).
That paper's spinors have ubar*u=1; this repository uses 2m. Replacing
u_paper by u_repo/sqrt(2m) converts sqrt(m/p+) to 1/sqrt(2p+).
The finite-cell derivation below is our normalization calculation, not a
choice to complete that paper's dynamics or adopt a published binding model.

With flat continuum brackets delta^3(p-p'), define Delta_p=(2pi)^3/V and
b_cont(p_n)=b_n/sqrt(Delta_p). The quadrature coefficient of each field is

    Delta_p / sqrt((2pi)^3) / sqrt(Delta_p) / sqrt(2p+)
        = 1/sqrt(2p+ V).

The field expansions therefore use u, v, or polarization vectors times
that factor. The normalized good-fermion metric is
sqrt(2)*u_+^dagger*u_+/(2p+)=I. Three factors and the spatial integral give

    <L,R|P-_3|P> = g v(L,R;P) delta_(L+R,P) / sqrt(V),
    delta M^2 = 2 P+ delta P-.

Here v includes 1/sqrt(8 pL+ pR+ pP+) and has GeV^-1/2 units; V is GeV^-3,
so the matrix element is GeV. The continuous density of states is not an
extra multiplier after this conversion. A stripped n-field coefficient
similarly acquires V^(1-n/2); in particular the quartic factor is 1/V.
This algebra does not supply the quartic kernels or their constrained partners.

For this diagnostic, bosons are periodic and fermions antiperiodic only
longitudinally; both are periodic transversely. Integer and half-integer
plus grids are compatible in all four canonical channels. Zero-plus is
outside the test, not assigned a physical prescription. Coordinate-space
wavepacket integration with three superposed external fields independently
agrees with the conserved mode sum. Full canonical tensors, dimensions and
longitudinal boosts pass; the preserved QCD spin/color kernels are unchanged.
Identical gluon counting remains the existing unordered-pair/ordered-half
rule; Bose square roots belong to the occupation action. Quark and antiquark
momentum assignments are distinguishable and must NOT be halved.

## 2. Actual retained quark-pair state, bounded quantum counting

The reference interaction is the already declared bilinear structure

    L_Y = -sum_i g_i qbar (sigma_i + i gamma5 tau.pi_i) q,

so its first LF Hamiltonian vertex has the positive Yukawa sign. This is
additional exploratory soft structure, not the canonical QCD Hamiltonian.
The two multiplets and their eight real scalar/isovector components are
retained explicitly, alongside quarks and antiquarks. The open vertex keeps
all 12 quark and 12 antiquark helicity/flavor/color states. Independent
graded occupation and Jordan–Wigner actions agree, including an occupied
intervening fermion that reverses the sign. Adjoint and Bose factors pass.
Color-singlet and vector-isospin intertwining are checked at the full tensor
level, not inferred from scalar rates.

The zeroth-order generator in this particular experiment is **free**: four
positive two-node collective mass-squared blocks and equal-mass free quarks.
It is not the previous nonlinear potential's full quantum generator.
Scalar self-interactions, gluon dynamics and matching counterterms have
not been appended or assumed absent in the production theory. All values
here are explicitly illustrative; no earlier fitted or bare parameters change.
The preserved canonical QCD channels receive their normalization tests but
are not falsely claimed to be dynamical in the collective-parent run.

The spatial cell has spacings (.25,.20,.20) GeV, volume
24805.021344239853 GeV^-3 and parent (P+,PT)=(.5,0,0) GeV. Both longitudinal
quark/antiquark partitions (.125,.375) and (.375,.125) are retained. At
transverse shell N, both species occupy the declared square integer grid,
giving 2(2N+1)^2 routes at PT=0, each with 144 resolved amplitudes. There is
no division by two. This is a finite numerical momentum domain, not a
physical UV/IR regulator or a model chosen at fixed harmonic resolution.

The coupling is multiplied by lambda f(t), with f=sin^4(pi*t/T), T=12 GeV^-1.
This smooth pulse is an explicit test protocol, not a fitted physical time
dependence. From a coherent normalized parent in a fixed charge block,

    psi_1(r,T) = -i integral_0^T dt f(t)
        exp[-i E_pair(r)(T-t)] V_r exp[-i h_parent t] psi_0.

This is the coefficient of lambda in the **Fock state**, including phases,
not an on-shell width or probability-only amplitude. Every one-Yukawa-order
sector in the declared positive-plus domain is included: the original
parent and all allowed q/anti pairs. Emission/absorption needs an initial
fermion, and an all-creation monomial cannot conserve strictly positive
plus from vacuum. This does NOT determine the constrained zero-mode vacuum.
More sectors and instantaneous terms are required at subsequent orders.
No previously calculated quark loop is inserted as a new bare interaction.

## 3. Results and linked source response

The analytic spectral pulse filter agrees with an independent adaptive time
integral using matrix exponentials: the shell-3 full complex wave difference
is 1.9552e-16 (14,112 pair amplitudes). The time-integral error estimate is
8.28e-13. The whole wave is unchanged by tested node rotations and by a
longitudinal boost with the cell and LF time transformed consistently.
This is not an independent dynamical transverse-boost/covariance certificate.

| Transverse shell | Pair routes | Leading pair norm squared |
| --- | ---: | ---: |
| 0 | 2 | .00152560757831 |
| 1 | 18 | .0316013971193 |
| 2 | 50 | .0421680866669 |
| 3 | 98 | .0421685381519 |
| 5 | 242 | .0421685403254 |
| 8 | 578 | .0421685403271 |

The scan uses the neutral sigma/pi3 seed. Separate normalized pi1 +/- i*pi2
seeds test charges +/-1; each gives .0122606829702 at shell 3, with total
charge residual below 1.85e-17. The neutral residual is exactly zero. An
earlier development seed mixed electric charges: it was useful only as a
linear-operator probe, not a physical state benchmark, and was replaced
before saving the accepted result. No coherent charge-superselection
violation is used to claim a physical quantum state.

These are coefficients of lambda^2 in the pair transition probability.
The state psi_0+lambda psi_1 is NOT unit-normalized to that order: the
second-order parent amplitude is still needed. Do not renormalize this
truncated vector by hand or identify .0421685 with a physical probability
at an uncontrolled coupling. Convergence of these finite-domain values
does not prove the infinite-volume/continuum or higher-order limit. The
omitted reversed-assignment wave norm at shell 3 would be .1452042323,
exactly half the leading transition weight by symmetry. Neither small
retained amplitudes nor an accurate parent norm would detect that loss.

A scalar mass-squared source s (GeV^2) changes the SAME collective mass
blocks by s*[[.2,1],[1,-.3]], equally for the isovectors. This is the
quadratic source -s phi D phi/2, not a QCD quark-mass insertion or EM current.
Its Hamiltonian derivative is D/(2P+); the positive-plus field normalization
does not depend on the collective mass. Native Torch derivatives of a
fixed complex projection of the pair wave with respect to (s,g1,g2) are

    (.00285563817026 GeV^-2, -.00440050288012, .00290697258685).

Independent fourth-order finite differences differ by at most 1.62e-14.
The source matrix does not commute with the mass matrix: this tests actual
mode mixing and shared dynamical response, not a separate observable head.
No electromagnetic current or C/D/E source transport is certified by this
scalar-source derivative. The existing nonstationary source result remains
a separate, preserved control.

## 4. Reassessment and next scientific step

The concrete normalization obstruction is resolved. The useful architecture
now has normalized interaction edges connecting physical momentum sectors,
full graded quark endpoints, a color intertwiner, isovector structure and
shared source derivatives. These are strategic tensor/network roles; there
is no new claim of topological order, confinement or quantum advantage.

The next bounded target is **one electromagnetic insertion linked to this
quark/collective transition**, not an all-order exponentiation of its cubic
block or another generic compression benchmark. Pre-evaluate before coding:

1. Derive the one-Yukawa/one-photon canonical LF source operator from the
   constrained fermion action. Include photon attachments to both quark
   endpoints and to a charged collective field, plus the mixed instantaneous
   photon/Yukawa term. The preserved covariant/LF tree amplitudes provide an
   independent control, not a substitute for normalized state matrix elements.
   Preserve the independent one-photon transverse C structure when carrying
   the full retained source action; D/E start at two photons. A minimal-only
   first diagnostic must be labeled, not mistaken for completed C/D/E matching.
2. Declare the Fourier source and enumerate every intermediate and final
   Fock/momentum route at that joint order. A positive-plus source can create
   pairs from vacuum; a one-body source-current truncation cannot silently
   omit them. Keep source endpoints and kinetic factors explicit. If using
   a safer restricted transfer for the first test, say what it excludes.
3. Compare full complex state/current response with independent time-ordered
   or covariant amplitude calculations; test Ward differences, transverse
   information, source parameter derivatives and negative omitted partners.
   Do not claim the old nucleon component discrepancy is repaired.
4. Then reassess the route back to shared p/n and modular-nuclear dynamics.
   An actual binding action, its matching, multi-sector convergence and
   independent quantum minus/transverse currents remain physical work.

This next question has a concrete canonical route, so no new executive
choice or scientific blocker is declared. Missing physical fit parameters
alone do not block explicit reference experiments. Do not initiate a sigma/
NJL/BLFQ completion program. Retained modes counted once remains approved.

Useful next-step derivation (preliminary, not a completed EM implementation):
for canonical good fields chi=2^(1/4)*psi_+, write
B0(pT)=alphaT.pT+beta*m and BY=beta*(sigma+i gamma5 tau.pi). The constrained
free/Yukawa generator is (B0+g BY)1/(i partial_minus) (`docs/next_level/B0+g BY`, full upstream checkout)/2.
Its linear pair block is

    chi_u^dagger [B0(pqT) BY/pq+ - BY B0(-pantiT)/panti+] chi_v / 2,

times the collective field normalization. The minus sign is the signed
antiquark Fourier momentum, not an absolute-value denominator. A direct
pointwise full-tensor check against the new covariant-spinor leaf agrees;
turn this into parameterized regression coverage when deriving the EM
partner. Gauging both B and the longitudinal inverse produces mixed terms;
gauging only an external free vertex is insufficient. Keep covariant versus
contravariant EM components explicit before using this as an operator formula.

A concrete first domain is a classical spacelike probe with q+=0 and
nonzero commensurate transverse transfer, with both Fourier signs of a real
source and a smooth compact time profile. This is not a real on-shell photon:
do not apply a polarization formula containing 1/q+ at that point. Such a
probe cannot create a strictly positive-plus quark pair from vacuum, so
parent and q/anti sectors suffice at one Yukawa/one source insertion in
this restricted domain. Nonzero-plus sources remain a later sector test.
For a declared final momentum window, include EVERY shifted intermediate
needed by both insertion orders, even outside that window. A hard truncation
of each vertex independently can spoil the gauge identity. An endpoint-zero
pure-gauge pulse supplies a useful whole-wave negative/identity control;
it does not test independent transverse C information by itself. This is
the next experiment's proposed domain, not an already executed EM result.

Recovery verification found eight source originals exactly equal to Git.
The archive also retains one macOS AppleDouble metadata member; it is not
a ninth source document. No archive extraction or original removal occurred.

## Validation at this checkpoint

All **23 new tests** pass with warnings treated as errors. The integrated
continuation/theory set passes **538 tests**, with the same 12 pre-existing
warnings (old docstring escapes and determinant negative controls). The
independent replay agrees field by field at relative tolerance 2e-12 and
absolute tolerance 2e-15; bitwise repeatability is not claimed. The full
historical suite was not rerun or declared green; its previously documented
spinful synthetic-closure failure is not changed by this coupling work.
No physics tolerance was weakened and no source parameters were fitted.

Replay with the established environment, `PYTHONPATH=src:.`, and
`scripts/evaluate_finite_lf_coupling.py`; the focused new regression is
`tests/test_finite_lf_coupling.py`. The canonical TeX/machine-state hash and
archive recovery checks pass separately. Persistent goal active, local only.

## Preserved pre-evaluation (before execution)

The [nonstationary source result](nonstationary_lf_source.md) qualifies
canonical/source transport at a bounded weak order and preserves complete
one-charge wave routes. It is not a full sourced Fock state. Return now to
the preserved canonical quark/color operators, exact graded algebra and
shared state/interaction architecture. No new binding model is selected.

## Concrete normalization question

`canonical_lf_network.cubic_vertex_tensor` supplies resolved quark/gluon
spin/color leaves, and `cubic_word` supplies the ordered monomial. Their
point kernels strip coupling and spatial delta/measure; they have
GeV^-1/2 units, not the GeV units of a discrete P-minus matrix element.
Do not add them to a free Hamiltonian with a guessed scalar multiplier.
The source-current work likewise must retain its endpoint metric and
direct terms when transported into normalized states.

The next executable audit should derive a consistent canonical field/mode
measure and a bounded finite wavepacket (or explicitly declared cell)
matrix element. Reuse the actual source field expansions and verify
dimensions, plus/transverse conservation, quark and antiquark ordering,
color tensors and Bose counting independently. A numerical cell/basis is
an approximation to qualify, not confinement or a QCD matching scale.

One concrete normalization route to verify is a periodic-volume expansion
with discrete canonical brackets. Three field factors contribute V^-3/2
and the spatial Hamiltonian integral contributes V, suggesting g_s v/sqrt(V)
for the stripped cubic leaf; a quartic leaf instead carries 1/V. Derive the
full factors of 2p+ and 2pi from the actual field convention before using
this relation. Boundary conditions must be compatible across quark,
antiquark and gluon channels (including their momentum grids), not selected
independently. Check the same result using normalized wavepacket/quadrature
conventions where possible. This candidate is not yet an executed matrix
element or a choice of physical finite-volume dynamics.

The preserved detailed starting point is
[canonical LF network integration](canonical_lf_network_integration.md),
including the distinction between bare canonical leaves and the complete
partonic tree with its instantaneous partners. Keep that distinction when
forming a quantum state rather than inferring closure from a single vertex.

## Quantum and scientific acceptance before a solve

1. Specify which quark/gluon and collective modes are retained quantum
   variables and which kernels are already matched/integrated. Count them
   once. Existing one-loop reference results are not new bare vertices to
   requantize. A connected one-body wave does not supply vacuum/pair sectors.
2. Transport complete operator tensors and source partners into normalized
   states. Compare independent direct field/mode integration and the graded
   operator action; preserve physical signs and all resolved indices.
3. Identify every sector reached at the proposed interaction/source order
   before restricting the carrier. The recent four-corner example proves
   that exact retained amplitudes need not preserve the full wave. Use an
   explicitly order-qualified extension and report leakage or missing sectors.
4. Only then execute the smallest genuine quark/collective quantum coupling
   and linked response control with an independent reference. A canonical
   scattering leaf or free mass matrix is not a physical p/n Hamiltonian.
   Source-dependent states and operators must share parameters, not acquire
   separate fitted observable heads.
5. Reassess where strategic DNN/tensor connectivity represents actual
   conserved quantum numbers, entangling interactions, gauge paths and
   unresolved matched physics. Do not substitute a generic optimizer or
   a graph drawing for that physical architecture.

The preserved `graded_fock_network.FockRegister` already separates ordered
occupation actions from Jordan--Wigner tensor contraction and warns about
boson boundary products. `canonical_lf_network` keeps bare QCD color and
propagating/instantaneous structures independent of BLFQ/H2. Reuse those
utilities; do not rebuild their old finite model or silently choose a
quark-meson/NJL completion program. Unknown matching parameters may remain
explicit during exploration. Stop only if the audit requires a genuine
new physics/regulator/model choice that cannot be resolved from the stated
plan. Generated p/n, modular nuclei and matched distributions remain the
mission; this is a coupling/normalization pre-evaluation, not their completion.

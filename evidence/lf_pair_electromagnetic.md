# LF quark/collective electromagnetic transition — 10 September 2026

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

The normalized **one-Yukawa/one-EM-source Fock response** now agrees with
independent covariant tree tensors and time integration. It retains all
quark/antiquark helicities, flavors, colors and both collective nodes, the
independent transverse C structure, both photon attachments and the charged
collective attachment. Two different contact contributions are necessary:
the constrained fermion instantaneous term and the collective normalization
endpoint at the Yukawa vertex. Neither a free current nor Ward conservation
alone supplies this response.

This is a controlled **q+=0 classical spacelike source** experiment, not a
physical bound pion/nucleon, full quantum loop current, production potential,
or repair of the old BLFQ/H2 covariance defect. D/E start at two photons;
they are not set to zero in the action. Existing loop results are not used
as new bare interactions. The persistent goal remains active, local only.

Source: `src/deuteron_wigner/quantum/lf_pair_electromagnetic.py`;
tests: `tests/test_lf_pair_electromagnetic.py`;
replay: `scripts/evaluate_lf_pair_electromagnetic.py`;
record: [numerical results](../snapshot/docs/next_level/lf_pair_electromagnetic_results.json).
Originals: `28fd253fb3b09c878d0f874c02ca6f48576f5506` and eight source files
in `computation_handoff/backups/2026-09-10_pre_lf_pair_electromagnetic.tar.gz`.
The preceding [normalization result](lf_quark_state_coupling.md) is preserved.

## Pre-evaluation and physical domain

The prior result supplies g*v/sqrt(V), full graded pair states and a
scalar mass-source derivative. The next question was whether the SAME
canonical interaction has a normalized electromagnetic response, rather
than an independently fitted current. The fermion constraint gives a
concrete route; no new binding model or coefficient choice was needed.

The basic constrained-field construction is established LF physics, checked
against [Srivastava–Brodsky, Sec. 4](https://arxiv.org/html/hep-ph/0011372v2).
The external EM/Yukawa derivative, finite-mode transport and independent
tests below are the present project calculation, not a claim of inventing
canonical LF quantization or a mandate to finish that paper's program.

Use symmetric p±, metric (+---), and an explicit EM convention
D_q=partial+i e Q A, Q=diag(2/3,-1/3). In the preserved external chiral-source
notation this is v=-e Q A, not v=+e Q A. Its O(4) connection becomes
i e Q_phi A, with Q_phi[pi1,pi2]=-i and Q_phi[pi2,pi1]=i. This sign map and
local Yukawa gauge invariance are directly tested. The original density's
positive-v derivative and the physical Hamiltonian insertion have the same
reported vertex after BOTH v=-e A and H=-L sign conversions. No old source
coefficient or quark charge is flipped to obtain agreement.

The source has nonzero commensurate qT and q+=0. It is spacelike, not a real
on-shell photon; no polarization containing 1/q+ is used. A source with
zero plus cannot create strictly positive-plus pairs from vacuum. Thus
parent and q/anti sectors suffice at this joint order, in the declared
final momentum window with all required intermediates. The constrained
zero-mode vacuum and nonzero-plus source-created pair sectors remain open.

## 1. Fermion constraint and actual mixed instantaneous term

For canonical good fields chi=2^(1/4) psi_+, define

    B0 = alphaT.pT + beta*m,
    BY = beta sum_i g_i (sigma_i+i gamma5 tau.pi_i),
    BA = -e alphaT.Q AT,       Pi = p+ - e Q A+,
    h_chi = e Q A- + (B0+BY+BA) Pi^-1 (B0+BY+BA)/2.

The good spin axes are normalized with u_+/sqrt(sqrt(2)*p+), and likewise
for v. The independent linear-BY derivative reproduces the full-spinor
Yukawa leaf. The linear-EM derivative reproduces all four free current
components, including the antiquark normal-ordering transpose and sign.
No absolute value replaces the signed antiquark Fourier denominator.

Let a,b be physical final q/anti momenta, P=a+b-q the initial collective
momentum, Y_A=beta Gamma_A and BA=-alphaT.Q AT with e stripped. The mixed
pair kernel from the actual Schur complement is

    Kf_A = chi_u^dagger [ BA Y_A/a+ - Y_A BA/b+
        + A+ { B0(aT) Q Y_A/(a+)^2
                    + Y_A Q B0(-bT)/(b+)^2 } ] chi_v
        / (2 sqrt(2P+ V)).

The terms proportional to A+ come from gauging Pi^-1. Gauging only B is
insufficient. This equals, independently,

    ubar [ Q slash(A) gamma+ Gamma_A/(2a+)
           - Gamma_A gamma+ Q slash(A)/(2b+) ] v
        / sqrt(8 a+ b+ P+ V),

the two instantaneous pieces of the covariant fermion propagators. There
is no mixed J+ term (source A-); transverse and J- mixed terms are nonzero.
The two node Yukawa couplings remain explicit outside this O(4) tensor.
Kf per dimensionless g and source A (GeV) is dimensionless. The classical
source does not introduce another quantum 1/sqrt(V) factor.

## 2. Collective C current and its normalization endpoint

The actual one-source density gives the resolved collective vertex

    Gamma^mu(p1,p0) = [(p1+p0)^mu I
        + C {q^2 (p1+p0)^mu - (p1^2-p0^2) q^mu}] Q_phi.

C is a symmetric node matrix (GeV^-2); it contains independent transverse
information. At fixed q- = omega and q+=0, set p0-=w, p1-=w+omega. After
the equal positive-plus kinetic factors, write

    U = Gamma(A,w=0)/(2P+),
    R = partial_w Gamma(A,w)/(2P+)
      = A+ (I+q^2 C) Q_phi/P+.

For canonical parent free matrices h0,h1 the first source generator and
field endpoint are

    J_phi = U + (h1 R + R h0)/2 - omega R/2,
    W_phi = R/2.

Both agree independently with derivatives of the original C action, not
only a constructed Ward representative. The instantaneous Yukawa/source
vertex also includes **Kphi=Y_final W_phi**. Endpoint-zero external pulses
do not remove this internal interaction endpoint. A pure-gauge test alone
cannot see every endpoint because its A+=0; the independent A+ test does.

For a fixed source direction, J_phi(omega)=J0+omega J1+omega^2 J2.
In time this is J0*a+i J1*dot(a)-J2*ddot(a). Source derivatives must not
be discarded. D/E contain at least two photons and do not enter this
one-source calculation; existing higher-source results remain intact.

## 3. Independent four-component amplitude and Fock-wave checks

The covariant control contains both fermion attachments and charged
collective exchange with its full node inverse and C vertex. Its LF
counterpart instead uses normalized Hamiltonian vertices, the two pair
energy denominators, the parent denominator, Kf and Kphi. Negative crossed
fermion plus is retained. Internal poles are excluded explicitly, not
assigned an unselected prescription. The initial parent is an energy
eigenvector only in this frequency-domain control; omega is otherwise
arbitrary, since the Yukawa pulse can supply the remaining minus energy.

All 36 component/frequency/charge cases agree: maximum vector difference
2.43e-15. Ward substitution A=q has residual below 3.99e-17 GeV. Raw source
columns are (J-,J+,-Jx,-Jy), so no second transverse metric sign is applied.
This distinction fixed an initial test-contraction error; no physics kernel
or tolerance was altered to make Ward pass. Longitudinal and transverse
kinematic LF boost tests pass for the tree tensor, with the measure/cell
transformed; this does not certify a fixed rectangular cutoff's covariance.
C changes the amplitude without changing its Ward divergence.

The time experiment uses the same free masses, g=(.31,-.19), quark mass
.27 GeV and numerical cell as the preceding reference, with independent
C=[[.14,.037],[.037,-.06]] GeV^-2. These are illustrative values, not a
physical fit. Parent momentum is (.5,.2,0) GeV, source transfer
(q+,qx,qy)=(0,-.2,.2) GeV and T=12 GeV^-1. Both the Yukawa test pulse and
the ordinary source envelope are sin^4(pi*t/T). Neutral and charged seeds
are separate; coherent node mixing is retained within each charge block.

For the real A+ cosine source, BOTH +/-q branches have Fourier coefficient
1/2. All amplitudes in each final square window are retained. At shell 1
the +q branch has 12 pair routes (1,728 complex amplitudes), and -q has
4 (576). Eight attachment-route intermediates for +q lie outside the final
window and are retained. They are intermediate bookkeeping labels, not
extra orthogonal states to double count in a norm. No intermediate is
dropped merely because its momentum lies outside the output window.

The response is the mixed coefficient in

    psi = psi0 + lambda psiY + a psiA + lambda*a psiYA + ... .

It includes photon-after-Y on either fermion, photon-before-Y on the
collective, and both simultaneous contacts. A spectral lower-order/time-
quadrature calculation agrees with an independent initial-value solve
using full Dirac-spinor operators: maximum full-wave difference across
three charges and two source signs is 1.63e-10 GeV^-1. Actual charge sectors
are preserved. The whole wave also passes a longitudinal boost with
transformed cell, source four-vector and LF time.

For theta(t,x)=sin^4(pi*t/T) cos(qT.x), A=partial theta, the endpoint-zero
pure-gauge mixed wave is below 6e-17 across all six runs. Omitting the
fermion contact leaves up to .0321; dropping outside-window intermediates
leaves up to .0337. These gauge coefficients are dimensionless, since the
theta amplitude is dimensionless. For the separate unit +q A+ source and
positive-charge seed, omission errors are:

| Omitted contribution | Wave error (GeV^-1) |
| --- | ---: |
| Quark attachment | .474313 |
| Antiquark attachment | .237157 |
| Collective-before order | .0572484 |
| Fermion instantaneous term | .291352 |
| Collective normalization endpoint | .0818127 |
| Source time derivatives | .00331053 |
| Outside-window intermediates | .364546 |

The full derivative wave, not a selected component or probability, is the
comparison object. Neither psiYA nor its squared norm is a probability;
higher interaction/source orders are required for a unitary finite-source
state. A hard cutoff on each vertex separately fails this source test.

## 4. Shared derivatives and final-window scope

At fixed free masses, the full wave is linear in g_n*(1,C00,C01,C11).
Those feature tensors are generated by the same Hamiltonian calculation,
not learned observable heads. Native Torch derivatives of a fixed complex
wave projection with respect to (g1,g2,C00,C01,C11) are

    (-.00153873250818, .00912632067677,
      -.000254955310897, .000162820101809, .0000742991642601).

The first two have GeV^-1 units and the C derivatives GeV. Independent
centered differences of separately rebuilt responses agree within 6.88e-15.
This exact polynomial control does not supply mass derivatives in the new
feature map; the preceding scalar mass-source work remains separate.

The numerical record reports the expanded final-window scan and quadrature
refinement. Increasing a window adds previously unrepresented output
amplitudes: exact amplitudes inside one window never certify the whole
continuum. The real-source squared derivative norm changes from .0840929
GeV^-2 at shell 1 to .217935 at shell 2 and .226307 at shell 3. This is not
a physical probability, distribution, or proof of regulator/volume matching.
It reaches .226360079329 at shell 4 and .226360136593 at shell 5, with
256 and 400 pair routes across both source signs. Quadrature refinement
changes the shell-5 vector by 1.87e-15 GeV^-1. Since windows are nested,
the newly represented shell-5 wave has norm approximately 2.39e-4 GeV^-1;
small change in the squared total norm is not an equally small vector-tail
certificate. No claim is made for the remaining infinite-window tail.

## Validation scope

All 28 new tests pass with warnings treated as errors (15.87 s). The
integrated focused continuation/theory selection passes 566 tests with
12 pre-existing warnings (157.32 s). No physics tolerance was loosened.
The full historical suite was not rerun or declared green; its previously
documented synthetic F1 tolerance failure remains unchanged. The final
theory/source-hash check is separate from scientific validation. Numerical
quadrature and cancellation diagnostics are not bitwise-repeatability claims.

## Reassessment and next step

The interaction now has a checked canonical source partner, not just a
scalar response. The important architectural objects are normalized
sector-changing vertices, full spin/flavor/color intertwiners, source-linked
node matrices, fermionic signs and explicit intermediate routes. Neither
tree conservation nor this restricted source experiment proves dynamical
covariance of a bound nucleon, confinement or quantum advantage.

Return these tools to the **proton/neutron and modular-nuclear state
construction**. Do not continue an indefinite charged-pion model-completion
campaign. Next pre-evaluate the first color-singlet baryon interaction layer:

1. Reuse the saved p/n exterior-algebra interpolators as benchmark inputs,
   not solved nucleons. Their flavor/spin/color ordering differs from the
   canonical helicity/flavor/color leaf: derive the map and preserve all
   signs, phases, orbital information and allowed spin/isospin couplings.
2. Derive the emission/absorption crossings of the retained scalar/isovector
   vertex from the same good-field constraint, alongside unchanged canonical
   q->qg. Assemble all three emitter terms and all reachable qqq*collective
   and qqqg sectors with color, charge, momentum and identical-fermion
   antisymmetry. A one-orbital scalar-diquark interpolator is not the allowed
   production state space or a justified many-body compression.
3. Compare independent exterior-algebra and graded/tensor actions on the
   complete declared state; retain source partners rather than attach a
   separate fitted observable head. Test the physical interaction graph
   before choosing DNN/tensor/quantum connectivity or compressing sectors.
4. Reassess the shared dynamics/matching requirements before solving or
   fitting bound states. A first-order cloud is not a physical p/n prediction;
   second-order state normalization, instantaneous interactions, unresolved
   matching and nonzero-plus source sectors remain open. Preserve the
   multi-target/GTMD goal and original topology-aware research direction.

This has a concrete canonical path and requires no new executive decision.
No production potential, BLFQ/H2 reconstruction, physical refit or public
push is authorized by this checkpoint. Prior results and failures remain
preserved. Ordinary implementation-test issues (a duplicated metric sign,
an escaped regex and a misplaced test tuple index) were corrected without
weakening physics tests or changing source parameters.

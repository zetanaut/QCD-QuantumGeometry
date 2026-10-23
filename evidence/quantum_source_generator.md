# Quantum source generator and nonzero-mode quantum response — 10 September 2026

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

The preceding [source-network transport](source_network_transport.md) is
qualified at local checkpoint 44b8ff84. The pre-evaluation below is retained;
its regular-time state and stationary LF weak-source tests are now executed.
These are reference controls, not a generated physical pion, nucleon or
complete LF quantization. No source-free model or fitted coefficient changed.

## Result and scientific interpretation

The actual C/D/E density now generates a canonical Hamiltonian and quantum
states at nonzero spatial momentum. It agrees with the original second-order
field equations and with an independent untruncated Schrödinger solution,
including complex phases, source-created pairs and all retained oscillator
amplitudes. Charge and spatial momentum are conserved. This advances beyond
the previous spacelike source kernel: that inverse was **not** a Hamiltonian.

The stationary LF map works through one weak insertion only when the
Hermitian generator, external-state transformations and direct contact are
all included. The actual E vertex exercises its quadratic LF-energy term.
It does not settle arbitrary source frequency, zero-plus constraints,
quantum LF minus/transverse currents or the old BLFQ covariance failure.

The main quantitative caution is that accurate survival amplitudes do not
certify the full state. In the two-node example the largest survival error
is 1.29e-7, but the charged full-vector error within the retained space is
3.21e-4 and the untruncated solution has 1.52e-6 probability beyond it.
Keep this distinction in any future learned/quantum compression test.

## Execution choice recorded before implementation

Use the charged quadratic sector of the already qualified source action,
all node C/D/E matrices, and a nonzero periodic spatial Fourier pair with
both real charged components. Derive its regular-time Z,B,V matrices from
the actual density. A homogeneous time-dependent transverse EM source
preserves momentum/charge but can generate correlated pairs. Compare the
canonical and original second-order evolution, then actual quantum Fock
states/response against the untruncated symplectic Gaussian result. The
finite quantum calculation is an independent control, not a new fitted
pion, binding dynamics or a claim that the homogeneous cell became LF.

For the LF bridge, use a positive nonzero plus-momentum domain and a
stationary source coefficient at one weak insertion. Compare the original
quadratic-frequency inverse with an explicitly Hermitian first-order
generator, transformed endpoints and direct source contact. Test the map
on the actual E contact for a forward pair of nonzero-plus photons, where
the source modifies the LF velocity term. This is a declared source-order
control, not a global finite-source LF quantization or a prescription for
zero-plus modes. Do not invent a fast extra mode by solving the truncated
EFT as an exact new UV theory. Reassess the unresolved domains afterward.

## Why this is the next physics step

The scalar/pion source construction has been derived in a Lagrangian
functional. Its C/E terms contain derivatives of dynamical fields, so
canonical momenta depend on the external source. An interaction/source
kernel cannot simply be relabelled as a quantum Hamiltonian. This matters
before differentiating the Hamiltonian to measure currents or multi-source
responses in a state. It is independent of the still-open physical fit.

A read-only check of the actual implemented density used one O(4) field,
zero field/connection at a point and a pure electric curvature
F_0x=f T, where T rotates the two charged pion components. For E=e,
the charged-field velocity-dependent terms are

\[
 \mathcal L=\tfrac12[(\partial_t\phi)^2-(\nabla\phi)^2]
 +e f^2[(\partial_t\phi_{\rm ch})^2-(\partial_x\phi_{\rm ch})^2].
\]

At fixed other derivatives, the ordinary-time velocity Hessian is
I+2 e f^2 P_ch. With symmetric LF coordinates, hold partial_- fixed
and vary partial_+; the free velocity Hessian is zero while the added
one is e f^2 P_ch. Automatic derivatives of the actual stress density at
e=.7 GeV^-4, f=.12 GeV^2 gave ordinary-time eigenvalues
(1,1,1.02016,1.02016), free LF (0,0,0,0), and sourced LF
(0,0,.01008,.01008). This is a local kinematic pre-evaluation, not a
quantum state, physical parameter choice, instability or new-degree-of-
freedom proof. The weak-source expansion and LF characteristic/constraint
structure must be treated deliberately.

## Bounded investigation before a state solve

1. Derive the source-dependent canonical transform and its contact terms
   from the same action. In a regular time slicing, inversion of the
   source-dependent kinetic matrix induces higher-source Hamiltonian
   terms even when the original density is polynomial. Include endpoints,
   operator ordering and any measure/contact qualifications needed for a
   quantum comparison; do not equate a Lagrangian source vertex to a
   Hamiltonian observable without the derivation. Retain external-source
   time derivatives/frequencies as well: a static energy curvature alone
   does not determine the dynamical Compton/current response.
2. Test an actual nonzero-momentum quantum state/response control where an
   independent Lagrangian or spectral calculation is available. Distinguish
   time-ordered propagation and direct source contacts, and verify the
   order of the source expansion and quantum counting. A regular-time
   reference is an independent control, not a substitute claimed to be LF.
3. Analyze the LF constraint/source map at the required perturbative source
   order, including zero-plus and instantaneous/endpoint contributions.
   Do not invert the source-free singular velocity Hessian numerically or
   interpret its finite-source change as a physical new mode by fiat.
4. Reassess what now permits a genuine canonical/retained quantum state
   construction and which physical/model choices still need resolution.
   Use the preserved canonical LF fields, source partners, graded Fock and
   modular utilities. No production sigma/NJL/BLFQ model is selected and
   no inherited bound-state repair campaign is authorized.

The goal remains jointly generated states and currents, eventually p/n and
modular nuclei with independent covariance/data and GTMD-level validation.
This pre-evaluation has a concrete mathematical investigation available;
it is not yet a blocker requiring a new user choice. Preserve retained
modes counted once and the unresolved matching directions. If the
constraint analysis reveals a genuine unresolved physics decision, stop
and expose it rather than continuing by assumption.

## Executed canonical derivation

Restrict the existing O(4) action to its charged quadratic EM sector. The
two neutral fields decouple here; they are not removed from the nonlinear
theory. Each of n multiplets retains both real charged fields and both
normalized spatial modes sqrt(2/V) cos(kx), sqrt(2/V) sin(kx), k>0.
There are m=4n real Q coordinates, retaining both charges and both momentum
signs after quantization. The spatial volume cancels from this quadratic
action; it is not a new fitting or QCD-matching scale. The constant positive
mass-squared matrix M is declared reference input, not a calculated hadron.

For the spatially uniform source A_x=a(t)T, F_0x=adot T, define
T_ch=[[0,-1],[1,0]], J_sp=-T_ch and

\[
 T=I_n\otimes T_{\rm ch}\otimes I_2,\quad
 X=kI_n\otimes I_2\otimes J_{\rm sp}+aT.
\]

Node matrices below mean their tensor product with I_4. Direct Fourier
integration of the **original implemented density**, independently of this
matrix reduction, gives

\[
 L=\tfrac12\dot Q^TZ\dot Q+\dot Q^TBQ-\tfrac12Q^TVQ,
 \quad Z=I+2\dot a^2 E,\quad B=2\dot a\,CTX,
 \quad V=X^TZX+M\otimes I_4-4\dot a^2D.
\]

Q has GeV^-1/2, P GeV^1/2, a GeV, adot GeV^2; C,D have GeV^-2
and E GeV^-4. Therefore Z is dimensionless, B has GeV and V GeV^2.
Actual-density automatic derivatives independently verify the momentum
and kinetic Hessian. On the positive-Z regular-time domain,

\[
 P=Z\dot Q+BQ,\qquad
 H=\tfrac12(P-BQ)^TZ^{-1}(P-BQ)+\tfrac12Q^TVQ.
\]

Use Weyl ordering. Coefficients depend on time, not Q, in this slice; the
finite quadratic operator has no field-dependent ordering ambiguity.
Its Hessian in (Q,P) order is

\[
 h_{\rm ph}=\begin{pmatrix}
 V+B^TZ^{-1}B&-B^TZ^{-1}\\-Z^{-1}B&Z^{-1}
 \end{pmatrix}.
\]

In particular -L_int with dotQ replaced by the free P omits the B^2 term
and the inverse-kinetic partners. Inverting Z resums source powers of this
*declared finite action*, not unknown higher-order EFT/QCD operators.
No field-dependent path-integral measure or continuum counterterm is being
claimed to vanish. The independent control here is the actual finite
Schrödinger equation and its normalization, including the vacuum phase.

The original equations are independently evolved as

\[
 Z\ddot Q+(\dot Z+B-B^T)\dot Q+(\dot B+V)Q=0,
 \quad \dot Z=4\dot a\ddot aE,
 \quad \dot B=2\ddot aCTX+2\dot a^2CT^2.
\]

The source a(t)=a0 sin^4(pi t/tau), 0<t<tau, vanishes with its first two
derivatives at the endpoints. There P=dotQ, enabling direct comparison
of the two endpoint evolution matrices. Dropping source derivatives is
a separate negative control, not an allowed approximation by default.

## Independent quantum evolution, with full information

The source-free normal frequencies are sqrt(eigenvalues(M)+k^2), each
repeated four times. Quantize Q,P in complete oscillator shells and form
operator products **before** projection, using a two-shell buffer. This
keeps the top-shell quadratic moments and the vacuum energy. Oscillators
are a numerical basis, not an added harmonic confinement interaction.

Evolve the vacuum and the charged one-particle vector
u=e_low tensor (1,i)/sqrt(2) tensor (1,i)/sqrt(2). Its Noether charge is
+1 and its physical momentum is +k. The time-dependent source can supply
energy but no net spatial momentum or charge; occupation number is not
conserved. Pair probabilities mean excitation number greater than the
initial 0 or 1, not a physical pion-production rate.

An independent solution is
psi0=N exp(-Q^TAQ/2), psi1=(b^TQ)psi0. In the source-free normal basis
write h_QQ=K, h_PQ=Lc, h_PP=R. The Schrödinger equations are

\[
 \dot A=i(K-ARA)-(L_c^TA+AL_c),\quad
 \dot{\log N}=-\tfrac i2\operatorname{tr}(RA)-\tfrac12\operatorname{tr}L_c,
 \quad \dot b=-iARb-L_c^Tb.
\]

Initial A=Omega, N=prod(omega/pi)^(1/4), b=sqrt(2Omega)u.
No phase alignment or vacuum-energy subtraction is performed. The exact
oscillator amplitudes follow independently from the Bargmann generating
function

\[
 f_0(z)=v_0e^{z^T\mathcal Rz/2},\quad
 \mathcal R=2\sqrt\Omega(\Omega+A)^{-1}\sqrt\Omega-I,\quad
 f_1(z)=\big[\sqrt{2\Omega}(\Omega+A)^{-1}b\big]^Tz\,f_0(z).
\]

Here v0 is the full complex vacuum overlap; coefficients are amplitudes
divided by sqrt(n!). This supplies every retained amplitude and the norm
outside a shell; the projection is **not** renormalized to hide that tail.
Position-space Hermite expansion, dimensionless covariance, charge,
momentum, survival and the derivative with respect to pulse amplitude
provide additional independent checks.

Saved controls: k=.23 GeV and tau=8 GeV^-1. One node uses M=.12 GeV^2,
C=.4, D=-.1 GeV^-2, E=.7 GeV^-4 and a0=.15 GeV. The two-node source
matrices are noncommuting and remain fully populated; their exact entries
and a0=.10 GeV are in the replay. Minimum sampled kinetic eigenvalue is
1; minimum V eigenvalues are .1264 and .133893 GeV^2. A sampled minimum is
not an all-source global positivity theorem.

| Control | One node, final shell 16 | Two nodes, final shell 8 |
|---|---:|---:|
| Complete basis states | 4,845 | 12,870 |
| Canonical vs original evolution, normalized error | 1.36e-11 | 7.38e-12 |
| Wrong -L_int generator error | .002385 | .001034 |
| Dropped source derivatives error | .13558 | .09463 |
| Charged complex survival error | 2.68e-11 | 1.29e-7 |
| Charged projected full-vector error | 4.96e-6 | 3.21e-4 |
| Charged outside-shell probability deficit | 1.41e-10 | 1.52e-6 |
| Charged dimensionless covariance error | 1.46e-8 | 7.49e-5 |
| Vacuum / charged pair probabilities | .072083 / .092543 | .043390 / .055498 |

Phase/covariance errors use sqrt(omega)Q and P/sqrt(omega), so they do not
add incompatible units. The outside-shell deficit near 1e-10 competes
with ODE/roundoff accuracy; it is not a rigorously resolved tail bound.
The two-node charged norm/charge error is below 9e-10. At a0=.08 GeV,
d<1|U|1>/da0= .03271721-.74721854 i GeV^-1 agrees between finite Fock
and independent Gaussian evolution within 1e-8 GeV^-1. This is dynamic
source response, not static energy curvature or a fitted observable head.

## Stationary LF one-insertion map

Use the symmetric convention p^2=2p+ p- -pT^2, omega=p-. At fixed
strictly positive p+, first canonicalize the inverse by its kinetic
metric 2p+. For generic Hermitian matrices with compatible units,

\[
 K(\omega)=h-\omega I-\epsilon(d\omega^2+b\omega+c),\quad
 \delta h=-h d h-\tfrac12(hb+bh)-c,\quad
 w=-dh-b/2,\quad W=I+\epsilon w.
\]

Then, through first order only,

\[
 K^{-1}=W(h+\epsilon\delta h-\omega I)^{-1}W^\dagger
       +\epsilon d+O(\epsilon^2).
\]

This identity holds with noncommuting matrices. It follows from
omega(h-omega)^-1=h(h-omega)^-1-I, not from assigning a new fast EFT root
physical-particle status. The pole normalization is -partial_omega K;
different perturbed modes require I+epsilon[(omega_n+omega_m)d+b]. The
transformed modes Wv_n satisfy that cross-mode metric to O(epsilon^2).
The direct d contact has no pole but is essential off shell.

Tested generic first-coefficient error is 1.81e-15 in the declared GeV
coordinates. Omitting the generator, endpoint or contact respectively
gives 10.36, 3.52 and .819 in the same response units. Halving epsilon
reduces the finite-map, pole-equation and normalization errors by four,
as expected for an O(epsilon^2) remainder.

For the **actual E action** take photons q1=-q2 with q+ nonzero, q-=0,
qT=0 and transverse equal polarization. Holding p+ and pT fixed gives
T_E=4E(q+)^2 omega^2, including off-diagonal node entries. Thus
h=(M+pT^2 I)/(2p+) and d=4E(q+)^2/(2p+). Both external factors of
(2p+)^-1/2 must also accompany the original unnormalized Green function.
In this example epsilon is the product of source amplitudes (GeV^2), so
d has GeV^-3 and T_E is dimensionless. In the generic test epsilon is
dimensionless and d has GeV^-1; do not mix those conventions. This is a
single two-photon E insertion, **not** the full minimal+C/D/E Compton
response, a finite-source constraint solution or an LF zero-mode rule.

## Reassessment and next scientific step

The canonical/source issue has a concrete solution in the tested domains;
it is not a blocker or a reason to change models. Retained modes are
quantized once; the previously calculated quark loops are not inserted
and then requantized here. No pion data are refitted, no p/n binding is
inferred, and no sigma/NJL/BLFQ production action is selected.

Next qualify the nonstationary LF weak-source map with source momentum
transfer, positive-plus endpoint normalization and direct terms. Then
check the linked one-/two-source orders before joining it to the preserved
graded quark/color and shared-state network. Its pre-evaluation is
[recorded here](nonstationary_lf_source.md). Do not keep extending Gaussian
tests without advancing this LF/state connection; the full-vector control
is now available for future quantum/DNN truncation and compression tests.
Zero-plus crossings, complete non-Abelian LF constraints, nonquadratic
collective/quark quantum dynamics and physical joint matching remain open.
Only a genuine unresolved physics choice should interrupt the persistent
goal; this next bounded investigation has an algebraic path.

## Evidence, originals and context

Source: `src/deuteron_wigner/quantum/quantum_source_generator.py`;
tests: `tests/test_quantum_source_generator.py`; replay:
`scripts/evaluate_quantum_source_generator.py`; saved values:
`docs/next_level/quantum_source_generator_results.json`.
Run with the project's Python environment, PYTHONPATH=src:., single-thread
BLAS, and pytest -p no:cacheprovider. New tests run with warnings as errors.
All 26 new tests pass under that policy; all 491 focused continuation/
theory tests pass with 12 pre-existing warnings. An independent replay
reproduces every saved field exactly. All eight archived originals were
verified byte-for-byte against 44b8ff84. No test tolerance was relaxed.
The focused continuation regression is distinct from the historically
failing full suite; the old spinful synthetic-closure tolerance is unchanged.

Originals: Git `44b8ff8409a1eea70280e0bbfd96fa9061405a35` and
`computation_handoff/backups/2026-09-10_pre_quantum_source_generator.tar.gz`
(eight pre-edit files). No old source kernels/reports, fitted matrices,
historical volumes or public review export were overwritten. New quantum
work and current pickup/theory surfaces are local, not pushed.

Primary methodological context, abstract pages consulted only:
[Grosse-Knetter, Hamiltonian/Lagrangian quantization of effective gauge theories](https://arxiv.org/abs/hep-ph/9308201)
and [Franke et al., Quantization of Field Theory on the Light Front](https://arxiv.org/abs/hep-th/0404031).
These discuss equivalence and LF construction/regularization issues;
neither is claimed to prove this finite-mode implementation or select its
production dynamics. The formulas and numerical comparisons above are
the present explicit derivations/tests, not unverified imported results.

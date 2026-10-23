# Canonical transverse current, projection boundary and first Wick partner

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Scientific source: `85717caebc5efdd151b5a625511a81dccff368ea`, extending
`380f1bad`. Date: 2026-09-07. Local only; no publication-mirror refresh or push.
Selected canonical Galerkin authority and H2-first remain unchanged.

## Outcome

The constrained quark field now supplies tested free and interaction current
kernels. The retained qqq↔qqqg transverse contact and all complete Q5/Q6
free transverse actions execute. The crossed gluon→quark–antiquark current
kernel and the first common-regulator current Wick contraction are derived.
This is **not yet the complete matched physical electromagnetic four-current**.

The main note, section `sec:canonical-transverse-current`, contains the full
equations. The independent primary source is
[Srivastava–Brodsky, Eqs. 3, 21, 24 and Appendix B](https://arxiv.org/html/hep-ph/0011372v2).
The projection and Wick-partner identities below are this continuation's
derivations, checked directly against continuum spinors and the existing
canonical Hamiltonian; they are not a claim that the paper supplies this
finite regulator or its physical counterterms.

## Constrained-field layout

Symmetric light-front components are (+,−,x,y). With physical transverse
upper components and D=∂−igA,

\[
i\sqrt2\,\partial_-\psi_-=
[\alpha_T\cdot(-i\nabla_T+gA_T)+\beta m]\psi_+.
\]

The kernel B0(p)=(α·pT+βm)/(√2 p+) retains signed Fourier modes.
B1(r,ε)=α·ε/(√2 r+) supplies the gluon contact.
For positive-energy quarks u=(1+B0(p))χ√(√2p+);
antiquarks use B0(−p) and χbar=iγ²χ*.
The implementation fixes the existing Lepage–Brodsky spin phases.

Free K+=I, K−=χ†Bf†Biχ, and
Kj=χ†(αjBi+Bf†αj)χ/√2.
At equal positive p+, the transverse kernel is

\[
K_T=\frac{p_{fT}+p_{iT}+ih\,\hat z\times\Delta_T}{2p^+}.
\]

It is diagonal in physical helicity and independent of mass in this
particular kinematics. The charge-stripped antiquark kernel is identical;
its physical charge contains the normal-order minus.

The order-g kernel has K+=0,
K−=χ†(Bf†B1R+B1LBi)χ and
Kj=χ†(αjB1R+B1Lαj)χ/√2.
The ordered order-g² kernel has only K−=χ†B1L B1Rχ.
Color, physical charge, Fock maps and each gluon factor
1/√(2π k2) remain explicit. **Do not insert a 2P+ mass conversion.**
The ordered A_left A_right product has not already been normal ordered.

For quark emission the signed denominators are p_out+ and p_parent+,
with a conjugated creation polarization.
For ordered b†d†A pair creation they are p_q+ and −p_qbar+,
with an unconjugated annihilation polarization and one quark flavor charge.
The pair's action-level CAR map remains separate.
Full instantaneous-propagator Dirac calculations test all these kernels.

The plane-wave contact obeys the canonical vertex Ward identity before
external projection. The pair control explicitly uses opposite daughter
charges:

\[
[V,\rho]_{q\bar q,g}
=Q_q[V(p_q,p_{\bar q}-\Delta)-V(p_q-\Delta,p_{\bar q})]
=2P^+\Delta_T\cdot J_{T,[1]}.
\]

Every allowed K2=9 route is tested, not one selected longitudinal state.

## Exact continuum projection, including the boundary

The finite density is D_oi(s)=∫φ_o*(Q+s)φ_i(Q).
The momentum moment is M_a^j=∫φ_o*(Q+s)Q_a^jφ_i(Q).
**M is not D_N Q_N.** The ket's N+1 ladder neighbors contribute even
though both external states lie in N. Ground-state M00=−sD00/2.
The implementation factors the exact displacement tables; no amplitude
threshold or occupation truncation beyond the declared external shell is used.

The active recoil is s_a=F_ai Δ and physical intrinsic momentum is
κ_i=Σ_a(F^-1)_ia Q_a. The current combines this moment with
((1−x_i)Δ/2+x_i P_mean)D and the physical helicity cross-product term,
then multiplies Q_i/(x_i P+).
It acts on retained qqq/qqqg and complete qqqgg, qqqq-qbar,
qqqggg and qqqq-qbar-g sectors, preserving all raw current tails.
Separate x/y calls avoid allocating both huge output tensors unnecessarily.

For the retained contact, the parent receives the full photon transfer
at the local emission vertex. The new relative mode has m=0 and integral
b/√π for each circular radial mode, with √(x_q x_g/x_parent).
Exact tree rotation and coherent emitter/color sums precede symmetry
projection. Independent six-dimensional polar-wave integration checks it.

The free finite-matrix conservation defect is exactly

\[
[H_{0N},\rho_N]-2P^+\Delta_T\cdot J_{T,N}
=-\Pi_NH_0(1-\Pi_N)\rho\Pi_N
 +\Pi_N\rho(1-\Pi_N)H_0\Pi_N.
\]

An independent N+2 shell evaluates this boundary because free H0 only
changes radial excitation by two. It is generally nonzero.
At N=1, the reduced fixed-spin/Jz block hides the raw-shell defect through
selection. The actual coupled N=2 block exposes it. Neither normalized
charge nor a low-shell accidental zero proves current/gauge completion.
A continuity-only construction cannot determine the missing transverse
part or choose a physical gauging of the regulator.

## First current Wick partner under the same field regulator

The gluon contraction of the ordered quark K−[2] gives, per g² and
quark charge,

\[
S^-_{J,\mathrm{IF}}
=\frac{C_F I_{\rm trace}K_2^2}{2\pi(P^+)^2}
\sum_{k_2=2,4,\ldots,2N_b}\frac1{k_2(p_2-k_2)^2}.
\]

This equals −∂E_IF^−/∂p_q+ at fixed cell and fixed field regulator,
where E_IF^− is precisely the existing instantaneous-fermion Wick
mass-squared term divided by 2P+.
The independent spin/polarization sum checks it directly.
All k2>p2 contributions remain; external Galerkin N is not the field trace.
Normal ordering must subtract the current partner together with the
energy partner. This does not determine finite current counterterms,
complete the other current contractions or supply propagating loops.

## Verification and preservation

161 new direct tests: 60 constrained/crossed spin kernels, 34 retained
free projections, 22 contacts, 21 packed-shell actions and 24 current Wick.
The complete four-file current run passed 137 tests in 54.86 s; the separate
Wick run passed 24 in 1.24 s.
Before the extension, 150 current/emission/good-current/tree regressions
passed in 80.01 s. These are separate invocations, not a summed single run.
Current M2 collection has 1,138 tests; collection is not a pass claim.
The full named M2+Q0–Q2 regression subsequently passed 1171 tests at
85717cae in2199.75s (1138 M2,33 Q,1885 inherited warnings).
The main note renders to 161 scratch pages. All 438 labels are unique and
all explicit references/citations resolve; the new mathematics and status
pages passed visual inspection. No rendered ??, final overfull boxes or
missing glyphs remain. A generic inherited undefined-reference warning
remains despite the direct reference audit. No PDF deliverable was requested.

The new cancellation tests use operand-scaled floating-point bounds:
subtracting nonzero mass-flip amplitudes cancels analytically, and the
largest adjoint check sums hundreds of thousands of raw entries.
No Hamiltonian, current, physical cutoff or existing physics tolerance
was changed to remove an error.

Backup:
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_transverse_shell_extension.tar`,
created from `380f1bad` (901,120 bytes).
It preserves the preceding current kernels/tests and the f69eba99-era
main theory, roadmap, handoff, machine state, AGENTS and document test.
Original shared theory sources and all 22 original volumes remain unchanged.

## Next scientific work

This checkpoint's free-Q5 contact matching and free-minus targets have
since advanced in [the current continuation](m2_current_covariance_result.md).
Its historical path below is not a statement that those new pieces are absent.

1. Project the excluded-sector quark emission and sea-pair current contacts,
   preserving their Bose/Fermi/color maps and both spin endpoints.
2. Carry J_PP, J_PQ, J_QP and J_QQ together through the same H2 wave map
   and coupling order as H and the norm. Do not relabel the standalone
   free or retained contact current as a matched physical current.
3. Complete remaining minus-current/Wick, gauged-regulator and propagating
   loop/counterterm partners under common field and external regulators.
4. Establish shared finite physical conditions and an input/holdout split;
   then nucleon/nuclear spin-one/GTMD matching and controlled uncertainties.
5. Continue many-body quantum compression and measurement resources against
   the corrected exact references. The full-tree circuit is a benchmark,
   not a demonstration of polynomial scaling or advantage.

The current extension does not change the earlier matched model's Pauli
form-factor values or cure their roughly one-third shell variation.
All three physical workstreams remain incomplete. Missing physical inputs
do not stop parameterized operator work.

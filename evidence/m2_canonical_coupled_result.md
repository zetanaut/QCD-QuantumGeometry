# Canonical coupled interactions, partial dynamics and quantum solver

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

6 September 2026. Scientific source
`27266f923bd68bec2f866f9b9ab8c8d7e11ae84e` continues the coupled Jacobi
foundation at `a03598a9`. It advances all three requested workstreams but does **not**
complete physical H2 dynamics, a predictive deuteron state/current, or an
efficient many-body quantum algorithm. The current handoff identifies the
committed source. Older numerical matrices remain byte-for-byte unchanged.

## What is now implemented

- Exact C45/circular-HO phase and width crosswalk, with an explicit omitted
  squeezed-state norm.
- Unequal-fraction bare qq cross-pair matrices in qqq and qqqg, including
  excited global Jacobi modes and both qqqg color coordinates.
- Canonical, physically momentum-conserving q-to-qg emission on the enlarged
  carrier, using LF spinors, canonical field normalization and one coherent
  three-emitter source. This is not a numerical extension of C50/C52.
- Canonical mixed qg current with the transverse Minkowski sign restored.
- Direct and crossed normal-ordered instantaneous-fermion qg contacts.
- A separately identified partial Galerkin Hamiltonian, its baryon density
  and mass-squared-flow continuity current.
- Sparse Lanczos reference and an actual PennyLane controlled-gate
  symmetry-sector graph circuit with simulator energy/gradient optimization.
  It contains neither StatePrep nor a dense QubitUnitary.

## Corrections that change the scientific interpretation of the old source

The old 266-state M2m/M2n/M2o chain and its exact backend tests remain useful
**numerical and algebraic oracles**, not proof of canonical QCD normalization.
Do not import their old C53 values into the corrected coupled branch.

1. **Physical versus scaled momenta.** C47 defines q_i=p_i/sqrt(x_i).
   At cluster transverse momentum zero its scaled inverse gives
   q_q=sqrt(z_g) q_rel and q_g=-sqrt(z_q) q_rel. C50/C52 pass those scaled
   values to physical spinors and polarizations, missing sqrt(z_q) and
   sqrt(z_g). Thus the old physical daughter sum is
   (sqrt(z_g)-sqrt(z_q)) q_rel, generally nonzero. The direct test uses
   k_q=5/2, k_g=2, K=9/2 and q_rel=(0.2,-0.1) GeV.
2. **LF spin and field factors.** The old C45 function is the canonical
   instant-spin Dirac spinor, not a fixed-LF-spin spinor. Both solve the
   Dirac equation; they require a momentum-dependent two-state rotation.
   With ubar u=2m, canonical fermion fields contain u/sqrt(2p+). The gluon
   contains 1/sqrt(2k+). Those factors cannot be inferred from a dimensional
   label on an integrated matrix.
3. **Gluon current metric sign.** SB Eq.(25) is
   J_g^a=f_abc (partial_- A_b,mu) A_c^mu. Transverse metric contraction and
   color-slot antisymmetry give **+f_abc A_b^i partial_- A_c^i** in Euclidean
   transverse components. The canonical one-gluon current is therefore
   +(k'+k)/(2 sqrt(k' k)) F^a, F^a_bc=-i f_abc. At equal momenta it is +F.
   Both C7 qqqg singlets obey (sum_i T_i+F)|singlet>=0. The legacy
   Euclidean-minus C192/C406 descendant instead gives -F and fails this
   independent global-charge test. This is not repaired by one common
   rescaling of the four C410 products.

These statements follow from direct local tests and the original
[SB source, Eqs.(11), (24), (25), Appendix B](https://arxiv.org/html/hep-ph/0011372v2).
The historical contracts are preserved, not silently mass-edited or promoted
to the corrected normalization. No old spectrum has been fitted back.

## Oscillator crosswalk and contact derivation

For a_+^dagger=(a_x^dagger+i a_y^dagger)/sqrt(2), and the conjugate minus
combination, let n=min(n_+,n_-), m=n_+-n_-. In flat momentum measure,

\[
 \phi^{\rm circ}_{n_+,n_-}(p;b)
 =(-1)^n\,\phi^{\rm C45}_{nm}(p;b)/(2\pi).
\]

The C45 coordinate Fourier phase is (-1)^n i^|m|. Consequently the
Fourier-transformed circular mode at the origin is b/sqrt(pi) for
n_+=n_-=n, for **every** n, and zero for m!=0. Cartesian Hermite creation
expansions test the phase independently, including excited radial modes.

For the old qg ground, b_in=sqrt(3)b_out, the circular expansion coefficient
is sqrt(3)/2 * (1/2)^n. A shell cutoff N_exc retains 2n<=N_exc and omits

\[
 {\cal T}_{N_{\rm exc}}=4^{-[\lfloor N_{\rm exc}/2\rfloor+1]}.
\]

The general-width implementation integrates finite Laguerre polynomials
after absorbing both Gaussians. It never renormalizes away the omitted norm.
No whole-266-state finite isometry is asserted.

For a pair (i,j), set X=x_i+x_j and alpha_x=x_i x_j/X. The pair-adapted
relative momentum is Q=(X_j p_i-X_i p_j)/sqrt(x_i x_j X), whose conjugate
coordinate is sqrt(alpha_x)(r_i-r_j). Conserved pair total and spectators
make all other pair-tree coordinates identical at contact. In normalized
physical position measure, the pair contact is

\[
 C_{x'x}^{ij}=\frac{b^2}{\pi}\sqrt{\alpha_{x'}\alpha_x}\,
             R_{x'}^T D R_x.
\]

R maps the default global tree to the pair tree. D is unity between any
two m_pair=0 radial modes with identical spectator occupations, and zero
otherwise. Rectangular output shells explicitly probe finite-parent leakage.
The phase makes these radial entries positive; an alternating sign in D
would double-count the Fourier phase.

For quark pairs the full-nucleon bare mass matrix is

\[
 \frac{V_{qq}^{ij}}{g_s^2}
 =\frac{K_2}{2\pi n^2} C_{x'x}^{ij} T_i\!\cdot T_j,\qquad
 n=(k'_{i,2}-k_{i,2})/2\ne0.
\]

Both ordered current products are included once. There is no extra
1/x_parent or additional 2P+ conversion. Equal share gives b^2/(6pi);
the (9,9,9)->(1,17,9) contact is smaller by sqrt(17)/9. The earlier
fixed-physical-width Gaussian diagnostic is a different regulator away
from equal share.

Independent integration of the physical r_i=r_j surface tests every
matrix element through N_exc=2 in both sectors, including complex phases.
The matrices are Hermitian, S3 covariant, Lz preserving and preserve the
joint color/orbital/longitudinal/spin-flavor sign projector.

**UV qualification:** one pair acting on a ground mode excites every
relative m=0 radial state with the same coefficient. The squared output
norm through radial n=0,...,R grows as R+1. A contact is a distribution;
finite Galerkin matrix elements are valid, but their existence is not
operator-norm convergence or a renormalized continuum limit.

## Canonical emission and mixed/instantaneous contacts

For an emitting parent fraction X=x_q+x_g,

\[
 p_{qT}=\frac{x_q}{X}p_{{\rm parent},T}
          +\sqrt{\frac{x_qx_g}{X}}Q,\qquad
 p_{gT}=\frac{x_g}{X}p_{{\rm parent},T}
          -\sqrt{\frac{x_qx_g}{X}}Q.
\]

Plus and transverse momentum conservation hold on all 1092 K2=27 routes.
Minus energy is not separately conserved at a light-front Hamiltonian
vertex; no false on-shell four-momentum test is imposed.

With normalized LF external spinors, the color-stripped full-nucleon
plane-wave mass kernel (units GeV before HO integration) is

\[
 {\cal V}^{M^2}
 =-\frac{2P_N^+\,
  \bar u(p_q)\gamma\!\cdot\epsilon^*(p_g)u(p_{\rm parent})}
 {2\sqrt{p_q^+p_{\rm parent}^+}\sqrt{4\pi k_g}}.
\]

Here k_g is the **undoubled** positive integer. The source sign is that
of -g_s psibar gamma.T psi A. Full normalized HO projection includes
sqrt(alpha) d^2Q/(2pi)^2 with alpha=x_qx_g/X and the circular C45 phase.
The numerator is a constant plus a linear polynomial in Q; the implementation
uses exact-degree Gaussian quadrature over all R^2, not the old finite
rectangle. Tests verify boosts, scaling, direct radial/angular integration,
canonical unit J_q^+, flavor, total Jz and coherent three-emitter antisymmetry.

The qg cross-current matrix uses the same physical pair contact and
K2/(2pi n^2), multiplied by the positive canonical gluon factor
(k'_{g,2}+k_{g,2})/(2 sqrt(k'_{g,2}k_{g,2})) and T_i.F.
Quark spin/flavor and gluon helicity remain explicit spectator identities.

For the instantaneous fermion, anticommuting gamma+ through the first
transverse gamma converts SB's -1/2 to the usual +1/2 ordering. Retained
direct and crossed color matrices are respectively T^a T^b and T^b T^a.
Per ordered field term the coefficient is

\[
 \frac{K_2}{2\pi\sqrt{k'_{g,2}k_{g,2}}\,r_2}\,
 \frac{\bar u'\gamma\!\cdot\epsilon_L\gamma^+
                   \gamma\!\cdot\epsilon_R u}
      {2\sqrt{p_q^{\prime+}p_q^+}}\; C_{x'x}^{ig}.
\]

The direct r_2=k_{q,2}+k_{g,2} is positive. The crossed
r_2=k_{q,2}-k'_{g,2} may be negative; both are odd and never zero.
No epsilon clipping is permitted. The two ordered color matrices obey
C_direct-C_crossed=-T_i.F. Bosonic-commutator self-induced inertia is not
this contact, and neither contact is V^dagger(E-H0)^-1 V.

## Partial Hamiltonian, current and cutoff measurements

The explicitly named Galerkin pilot is

\[
 H_{\rm pilot}=
 P\{H_0+g_s(V_{\rm emit}+V_{\rm emit}^{\dagger})
       +g_s^2(V_{qq}+V_{qg}+V_{\rm IF,direct}+V_{\rm IF,crossed})\}P.
\]

Every parameter is supplied by the caller. This is a **normal-order-then-
Galerkin-project partial source**, not yet the historical corresponding-
propagating-graph conditional regulator or a complete finite-C43 action.
The missing self contractions, higher-Fock/induced terms, counterterms,
zero modes and physical matching remain named omissions.

Controlled diagnostic: total K2=9 (whole three-quark system, **not** the
old active K9 subsystem with parent nucleon K2=27), b=0.4 GeV,
m_q=0.35 GeV, m_g=0, g_s=0.3, proton I=1/2, Jz=+1/2.
These numbers are exploratory, not fitted inputs.

| N_exc | qqq / qqqg hg=-1 / qqqg hg=+1 ranks | Total | Lowest M² (GeV²) |
|---|---|---:|---:|
| 0 | 10 / 7 / 20 | 37 | 1.42143323030189 |
| 1 | 37 / 67 / 100 | 204 | 1.41947502860488 |
| 2 | 87 / 247 / 320 | 654 | 1.30252682156313 |

The N_exc=2 shift warns against treating the smaller-shell agreement as
convergence. These are eigenvalues of the displayed partial operator,
not nucleon masses. Proton/neutron and opposite-Jz spectra agree in the
declared equal-quark-mass model.

The normalized baryon density has quark charges 1/3 and gluon charge zero.
Its exact oscillator displacement uses delta Q_a=J_x[a,i] Delta_T.
It obeys rho(0)=1 and rho(-Delta)=rho(Delta)^dagger. The longitudinal
mass-squared-flow current is

\[
 J_{M^2,T}=\frac{\Delta_T}{|\Delta_T|^2}[H_{\rm pilot},\rho(\Delta_T)].
\]

At Delta=(0.12,0.16) GeV, continuity residual norms for the three shells
are below 3e-18 GeV². Both diagonal-pair and number-changing emission
blocks contribute. Divide by 2P+ for P-minus flow. This does not determine
the divergence-free transverse current or constitute full electromagnetic
or non-Abelian Ward/Slavnov--Taylor closure.

## Quantum implementation and limitations

The circuit basis already has the required total-sign/color/I/Iz/Jz
quantum numbers. A Hamiltonian-support spanning tree proposes two-level
rotations; controlled Gray-code gates realize each rotation and a relative
phase. The proposal threshold affects only the ansatz graph, never H.
Disconnected components are rejected rather than silently dropped.

Padding has a penalty strictly above an absolute-row-sum spectral bound.
Each complete logical rotation is identity on padding; intermediate
Gray-code swaps may visit padding, so this is not a noise-protection code.
The 37- and 204-state tests use 6 and 8 qubits and 72 and 406 real
parameters. Actual gate-state and independent two-component updates agree
to less than 3e-20 in the named probes; measured energy errors are below
7e-16 GeV² and final padding probability is zero.

This O(D)-parameter prototype is a scaling baseline, not a demonstrated
polynomial-cost many-body ansatz. The sparse classical eigenvector is used
only for validation, never as circuit initialization.

The 37-state complex-angle L-BFGS run from small random angles reached a
1.28e-9 GeV² energy error and squared overlap 0.99999999880, but hit its
700-iteration limit with gradient norm 1.54e-6: **not optimizer convergence**.
A zero-phase real-rotation BFGS trial reported convergence yet remained
2.78e-5 GeV² above the exact ground, demonstrating why optimizer success
alone is insufficient. A tree-only uniform initializer has therefore
been added to populate every proposed subtree without exact-state input.
The uniform trial also hit 700 iterations: energy error 3.35e-9 GeV²,
overlap² 0.99999999775, gradient norm 3.71e-6 and zero final padding.
It does not solve the optimizer limitation; do not erase unsuccessful trials.

Reproduce the uniform trial in the declared quantum environment:

```python
import numpy as np
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_symmetry_graph_solver import (
    SymmetryGraphCircuit, lanczos_ground,
)
p = build_coupled_pilot(
    total_k2=9, cutoff=0, target="PROTON", jz_twice=1,
    b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3,
)
c = SymmetryGraphCircuit(
    p.matrix, root=int(np.argmin(p.matrix.diagonal().real)),
    support_threshold=1e-12,
)
result = c.fit(c.uniform_parameters(), max_iterations=700,
               gradient_tolerance=1e-8)
exact_energy, exact_state = lanczos_ground(p.matrix)
print(result.success, result.message, result.fun - exact_energy)
```

## Historical scientific fork, now resolved as H2-first

The user selected matched H2 first on 2026-09-06. See
[the subsequent matching result](m2_matched_h2_result.md). The alternatives
below record the checkpoint's rationale, not a request for another approval.

The remaining issue is not a missing bookkeeping certificate. At finite
Fock and oscillator cutoff, projecting the fields before contractions and
normal-ordering the full source before Fock projection are different:

\[
 P A A P-P A P A P=P A(1-P)A P.
\]

The old C57 rule selected instantaneous graphs together with their
corresponding propagating support in the old C45/C47 carrier. It cannot be
silently declared identical to this new Galerkin pilot. The corrected
canonical vertex and metric sign also prevent recovering that identity by
one scalar adjustment. This must be settled before a physical matched
Hamiltonian/current or a fitted spectrum is claimed.

The next **direction decision** is the near-term closure target:

1. **Recommended: matched H2 effective theory.** Keep qqq plus qqqg as
   retained states, derive the new corresponding-propagating regulator and
   its self-energy/current partners, and match excluded-sector effects into
   explicit energy-/resolution-dependent operators. Use the existing
   physical-anchor program, not new default fitted values.
2. **Explicit higher-Fock construction.** Enlarge the carrier to qqqgg and
   quark-pair sectors and construct their canonical vertices/currents before
   reducing back to H2. This changes the scope and computational program.

Either route still needs source-qualified renormalization, physical
current closure and microscopic-to-nuclear matching. No finite H2 truncation
alone proves all-orders QCD or all-observable completeness. Neither the
user's three workstreams nor the full project are marked complete.

## Reproduction, files and preservation

Implementation modules and corresponding tests share these suffixes:
`m2_coupled_pair_action`, `m2_canonical_emission`,
`m2_canonical_mixed_current`, `m2_canonical_instantaneous_fermion`,
`m2_coupled_pilot` and `m2_symmetry_graph_solver`.
They reside in `src/deuteron_wigner/quantum/` and `tests/test_*.py`.

The 101 new tests pass. The named regression passes 443 tests (410 M2 and
33 Q0--Q2), with 1,885 inherited warnings in 295.00 seconds. The six graph
tests were rerun after the uniform initializer addition and pass in 2.25
seconds; no tolerance was relaxed. This is not a full-repository test claim.

The declared quantum environment remains valid. Reproduce the named
regression with `tests/test_m2_*.py` plus `test_q0_plhqcd0.py`,
`test_q1_plhqcdstate.py` and `test_q2_plhqcdobs.py`. The document-consistency
tests separately pass (6 tests, 0.05 seconds) after synchronizing the TeX
and machine-index hashes. Test counts are not evidence of physical completeness.

The updated theory note was compiled to a 133-page scratch PDF for equation
and layout QA, not as a requested new deliverable. All 344 unique labels and
76 cited bibliography entries resolve; no rendered `??`, overfull boxes or
missing-glyph warnings remain. The revised status/current-sign pages and
new section 18.7 were visually checked. Generic inherited BibTeX and
undefined-reference rerun warnings persist despite that explicit audit;
the build is not described as warning-free. No additional volume is needed
for this bounded continuation: the derivation belongs in the current main
note, with the original volumes preserved.

Pre-edit pickup/theory/test files are preserved at
`computation_handoff/backups/2026-09-06_pre_canonical_coupled/pickup_theory_and_test.tar`,
from source/document HEAD `9ac2d9a0`. Original theory archives, the legacy
sources/matrices, and unrelated user work are untouched. No push or public
mirror refresh is included.
The backup SHA-256 is
`3d12ef24bb0abbca47aa23bdf3559c685db7c368f34f883017d18d25a3baf059`.

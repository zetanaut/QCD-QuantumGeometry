# M2j permutation-complete C7/H2 embedding result

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-05

Status: M2J_SOURCE_SUPPORTED_INTERACTING_HAMILTONIAN_PARTIAL_H2_READY.

## Scientific result

M2j maps the complete M2i K9 nested-Jacobi member into an exact color/orbit
S3-sign target

\[
{\cal H}_{\mathrm{M2j}}
={\cal H}_{qqq}\oplus{\cal H}_{qqqg}
\]

while retaining the complete C7 color-singlet multiplicity. This removes the
color multiplicity obstruction and supplies the correct algebraic sign
projector on the abstract quark-slot orbit. M2k subsequently shows that the
equal-local-width M2i Jacobi ground state is not closed under cyclic emitter
tree changes. M2j therefore does not yet provide a complete noncolor quark
antisymmetrizer, H2 dynamics, or current.

The implementation is
src/deuteron_wigner/quantum/m2_permutation_complete_h2.py; its direct tests
are tests/test_m2_permutation_complete_h2.py.

## Exact color/orbit sign projector

Let \(R_{\rm orb}(\pi)\) permute the six active/spectator slot assignments and
let \(D_c^{(s)}(\pi)\) be the action of the same permutation on the complete C7
color-singlet basis of sector \(s\). Before operator assembly M2j applies

\[
A_3^{(s)}
=\frac1{3!}\sum_{\pi\in S_3}
\operatorname{sgn}(\pi)\,
R_{\rm orb}(\pi)\otimes D_c^{(s)}(\pi).
\]

The color/orbit image dimensions are

\[
\operatorname{rank}A_3^{(qqq)}=1,\qquad
\operatorname{rank}A_3^{(qqqg)}=2.
\]

The representation, projector, and exchange-eigenvalue residuals are,
respectively, 6.66e-16, 3.33e-16, and 2.78e-16.

All three quark-emission color vectors are retained. Their sum vanishes at
2.22e-16, as color singletness requires, and their span has rank two. In
the deterministic antisymmetric basis, the represented active-emitter
direction is

\[
u_A=\left(\frac{\sqrt3}{2},-\frac12\right)
\]

up to numerical phase noise, and an orthogonal retained direction is

\[
u_\perp=\left(\frac12,\frac{\sqrt3}{2}\right).
\]

The latter is RETAINED_UNSOURCED_NOT_PHYSICAL_ZERO. The M2k audit further
distinguishes the rank-two raw color span from the rank-one covariant
one-body-emitter orbit: the two raw same-orbit color directions are not two
additional independently coupled C53 vertices.

## Finite-\(K\) target basis and exact source embedding

The M2i source has 450 color-reduced directions: two q directions and 448 qg
directions. Each q direction produces one antisymmetric qqq state. Each qg
direction produces both antisymmetric qqqg singlet multiplicities. Therefore

\[
\dim{\cal H}_{qqq}=2,\qquad
\dim{\cal H}_{qqqg}=448\times2=896,\qquad
\dim{\cal H}_{\mathrm{M2j}}=898.
\]

The qqq longitudinal tuple is \((9,9,9)\). Every qqqg tuple is stored in
three-quark-then-gluon order,

\[
(k_{q,2},9,9,k_{g,2}),\qquad
k_{q,2}+k_{g,2}=9,
\]

and closes exactly at nucleon \(K_2=27\). The active qg oscillator mode and
both M2i nested Jacobi ground-mode labels remain explicit. The exact
Talmi--Moshinsky/Jacobi action between different emitter trees is not
represented in the 898-state target.

The source embedding \(E:\mathbb C^{450}\to\mathbb C^{898}\) uses the C7 qqq
bridge phase and \(u_A\) in every qqqg doublet:

\[
E^\dagger E=I_{450}.
\]

Its image projector has rank 450. The direct sum of all \(u_\perp\) channels
has rank 448, is orthogonal to the image, and together the two projectors
resolve the complete 898-dimensional target. The embedding has 898 nonzero
entries and numerical isometry/orthogonality residuals below \(10^{-14}\).

## Maximal source-supported Hamiltonian

M2j separates terms according to what the source actually licenses. H0, the
two C396 mass directions, and the M2i nested-Jacobi identity shift are
Fock-diagonal color scalars and therefore act on both qqqg multiplicities:

\[
H_{\rm univ}^{\rm M2j}
=H_{qqq}^{\rm src}\oplus
\left(H_{qqqg}^{\rm src}\otimes I_2\right).
\]

The represented C117 interaction and C53 number-changing vertex are known
only in the active-emitter source channel. They are lifted isometrically:

\[
H_{\rm rep}^{\rm M2j}=E H_{\rm rep}^{\rm M2i}E^\dagger,
\qquad
H_{\rm rep}^{\rm M2j}P_\perp=0.
\]

This last equation records unavailable action; it is not an assertion that
the physical orthogonal-channel interaction vanishes. The complete retained
operator is

\[
H_{\rm M2j}=H_{\rm univ}^{\rm M2j}+H_{\rm rep}^{\rm M2j},
\qquad
E^\dagger H_{\rm M2j}E=H_{\rm M2i}.
\]

At the existing named exploratory point the target is 898 by 898 with 18,338
nonzero entries. Source reconstruction and source pullback each close at
3.55e-15; Hermiticity closes at 1.11e-16; and represented interaction
leakage into the retained orthogonal channel is 2.78e-17.

This is a genuine interacting target because the represented C53 qqq-to-qqqg
block is nonzero. It is still a partial H2 Hamiltonian. The existing
synthetic C9/H2 numerical matrices are not consumed.

## Current boundary

The M2h/M2i active-quark operator is Fock diagonal and color scalar. M2j
therefore extends it to both qqqg multiplicities with the same direct-sum
rule as the universal Hamiltonian. For the tested transfer pair
\(\Delta_T=(\pm0.2,0)\,\mathrm{GeV}\) and the forward operator:

- source pullback residual: 2.22e-16;
- forward 898-state identity residual: 0;
- transfer-reversal adjoint residual: 1.11e-16.

This is the complete-multiplicity active-quark diagonal current, not the
complete three-quark or Hamiltonian-gauged H2 current.

## Explicitly unavailable physics

The following Hamiltonian content is not supplied by M2j:

- a noncolor-S3-covariant emitter-tree basis;
- the excited Jacobi modes, or symmetry-adapted global ground state, required
  by the exact tree recoupling;
- the full spin--flavor S3 recoupling;
- active-spectator and orthogonal-singlet C117 interactions;
- instantaneous, induced, constrained-field, zero-mode, and
  sector-dependent counterterm/matching terms.

The following current content is also unavailable:

- spectator-quark currents on the fully permuted emitter trees;
- q-to-qg and qg-to-q number-changing current blocks;
- currents obtained by gauging the regulated interaction;
- induced, constrained-field, pair, zero-mode, and counterterm/matching
  contributions.

No physical proton/neutron state, production current, nucleon mass, physical
response rank, fit, or observable prediction is claimed at this increment.

## Verification and exact next derivation

The 15 direct M2j tests pass. The complete named M2 chain passes 130 tests.
Those tests establish the algebra and implementation contracts; the equations
and source ownership above are the scientific evidence.

M2k now supplies the exact emitter-tree coordinate maps and proves two facts.
First, the M2i equal-local-width ground has cyclic overlap \(32/33\), so a
lawful S3 implementation must add the recoupled Jacobi tail or use the
symmetry-adapted global-ground width
\(b_\rho=\sqrt{2/3}\,b_\lambda\). Second, the normalized S3 orbit sum of one
representative canonical emitter is exactly the M2j active-emitter color
direction; the covariant projected emitter rank is one. The next construction
is therefore a symmetry-adapted or enlarged noncolor basis with full
spin--flavor recoupling, not two freely normalized spectator C53 copies. That
same finite Hamiltonian/regulator must then be gauged to produce interaction
and number-changing current blocks. Orthogonal-singlet C117 action and the
remaining H2 counterterms stay explicit open entries until independently
derived.

# M2 H0/H1 boundary audit

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-03
Last updated: 2026-09-04

## Purpose

This note records the result of auditing the frozen microscopic H0/H1 branch
before supplying an `H_{0,K}` to the main-line exploratory operator bundle.
It is a claim-boundary note, not a rejection of the earlier work. The earlier
branch remains valuable as a typed kinematics, basis, term, and solver
validation architecture.

## Audited source locations

- `src/deuteron_wigner/microscopic/h0/resolution.py`
- `src/deuteron_wigner/microscopic/h0/basis.py`
- `src/deuteron_wigner/microscopic/h0/terms.py`
- `src/deuteron_wigner/microscopic/h1/basis.py`
- `src/deuteron_wigner/microscopic/h1/hamiltonian.py`
- `src/deuteron_wigner/microscopic/h1/current.py`
- `src/deuteron_wigner/microscopic/h1/solvers.py`
- `src/deuteron_wigner/microscopic/h1/state.py`
- `src/deuteron_wigner/microscopic/h1/tensor_network.py`

The implementation itself carries the relevant status labels, including
`H0_VALIDATION_ONLY`, `C8_H1_VALIDATION_ONLY`, and
`UNIMPLEMENTED_NOT_ZERO`. Those labels are part of the evidence and must not
be removed when the branch is reused.

## What is established

The H0 layer establishes useful typed identities and validation gates. In the
declared symmetric light-front convention it uses the free invariant-mass
structure

\[
M^2_{\mathrm{free}}=\sum_i\frac{k_{\perp i}^2+m_i^2}{x_i},
\qquad
p^2=2p^+p^- -p_\perp^2.
\]

It keeps the oscillator scale `b` distinct from the Hamiltonian resolution
scale `lambda_H`, records endpoint and zero-mode policies, enforces
fermion-mode and longitudinal-boundary conditions, and provides explicit
color, permutation, charge, baryon-number, `J^z`, and center-of-mass gates.
The reduced canonical vertex supplies a typed qqq-to-qqqg benchmark and its
adjoint relation.

The H1 layer then supplies a nontrivial valence qqq basis tower, an exactly
Hermitian model matrix, an exact/matrix-free small-basis solver, state
tracking, and a toy/model valence current. The primary tower dimensions are
4, 7, and 10.

These are real reusable validation results. They do not by themselves fix the
physical deuteron Hamiltonian or current.

## What is not established

The current main-line C401/C410 spaces have dimensions 1350, 2706, and 4758
for K9, K11, and K13. The H1 valence tower has dimensions 4, 7, and 10, and
the H0 reference construction contains only the small explicitly enumerated
reference states in qqq, qqqg, and qqqq-qbar sectors. There is no declared
isometry, embedding, or basis map from those H0/H1 spaces into the current
C401/C410 direct-sum coordinate spaces.

Consequently, the H1 matrix cannot be used as the `H_{0,K}` required by
`quantum/operator_bundle.py`. Doing so would silently identify different
bases, omit Fock sectors, and misrepresent model terms as the C396/C410
operator directions.

The H1 discrepancy ledger explicitly leaves qqqg, qqqq-qbar, higher
orbitals, zero modes, instantaneous partners, and the basis tail as
`UNIMPLEMENTED_NOT_ZERO`. The valence current is likewise a validation/model
current, not a production current matched to the current light-front/LPS
routes. No physical deuteron sector, spectrum, current, fit, or activation is
therefore authorized by this branch.

## M2 decision

Do not port or relabel the frozen H0/H1 model as a physical or dimension-
matched `H_{0,K}`. Preserve it in place and use it for:

1. convention and exact-block tests;
2. small-basis solver and state-tracking tests;
3. future basis-map design and truncation studies;
4. explicitly named exploratory demonstrations whose basis is kept separate
   from C401/C410.

For the main-line K9 response loop, an `H_{0,K}` must be supplied through a
new explicit interface with all of the following recorded:

- the K-local basis identity and dimension;
- the map from every H0 basis state to the C401/C410 coordinates;
- sector, charge, `J^z`, color, parity, center-of-mass, and zero-mode
  policies;
- units and resolution dependence;
- the omitted-sector and counterterm treatment;
- Hermiticity and commutator tests;
- a claim tier no stronger than the evidence supports.

The interface is now implemented at
`src/deuteron_wigner/microscopic/h0/basis_map.py` and directly tested. M2
now also has an explicitly exploratory map instance in
`src/deuteron_wigner/microscopic/h0/k_local.py`: the C47 intrinsic
`q_rel^2` basis, CM projection, and diagonal functional are used while M2
assembles the sparse HO recurrence. C128 `pperp2` is a recurrence cross-check
only; neither the C7/C8 matrices nor the defective historical C128 numerical
free operator/fractions are used. The mass terms remain external C401/C396
directions. This closes the H0 boundary implementation, but not a
state-to-physical-observable calculation; all omissions and normalization
owners are recorded in
`docs/next_level/m2_h0_basis_map_contract.md`.

The subsequent K9 state-to-current audit closes the next question at a C47
color-intertwiner obstruction, not at an adapter failure. The accepted M2
eigenspace is the full six-dimensional q block with open quark helicity and
open triplet color. More strongly, the complete K9 M2 space is
`2 * 3 + 448 * 3 = 450 * 3`: C47 `q_basis` retains the two q triplets and each
qg tuple uses C47's `U_(3<-3x8)=T^b/sqrt(C_F)` triplet isometry. Its verified
fundamental Casimir makes `Hom_SU(3)(1,H_M2,K9)={0}`. The dimension difference
from the three target helicities rules out only an isomorphism; it is not the
primary obstruction. A nonzero color-singlet deuteron composition cannot land
in the present M2 space. C405 shares the same K9 direct-sum axis but explicitly
leaves the C117 q diagonal block unavailable rather than zero; C114 likewise
has no complete finite-HO current action. Neither a vector selection nor a
`P J P`/trace diagnostic is lawful. The next construction first introduces or
binds an enlarged many-body/hadronic finite-K color-singlet spin-one Hilbert
space, then derives its finite-K current intertwiners. C7/C8 remains separate
without an explicit basis map. The focused invariant-projector boundary test
is `tests/test_m2_state_current_boundary.py`.

## Related records

- `docs/next_level/mainline_quantum_substrate_integration.md`
- `handoff/CURRENT_PROJECT_HANDOFF.md`
- `handoff/ROADMAP.md`
- `src/deuteron_wigner/quantum/operator_bundle.py`

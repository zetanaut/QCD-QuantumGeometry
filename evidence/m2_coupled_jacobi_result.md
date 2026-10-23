# Coupled longitudinal/Jacobi foundation — 6 September 2026

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

The first coupled-basis increment is implemented at scientific checkpoint
`a03598a94ed3d4b5e67751b657a02c3fbd1d877c`. It supplies executable momentum
routing, unequal-share Jacobi coordinates and measures, complete oscillator
shells, and joint quark-permutation/fermion-sign projection.

**Scope:** source-derived kinematics and a finite-shell symmetry construction,
not the enlarged interacting Hamiltonian. C53 routing is not a new emission
amplitude. The regulated pair-action pilot, C45 excited-state phase/width map,
generalized C53 matrix elements and mixed/gluon normalization remain next.
The existing 266-state Hamiltonian, current and exact PennyLane oracle are
unchanged.

Implementation: `src/deuteron_wigner/quantum/m2_coupled_longitudinal_jacobi.py`.
Direct tests: `tests/test_m2_coupled_longitudinal_jacobi.py`.
The main theory derivation is `sec:coupled-jacobi` in
`references/DeuteronWigner_complete_theory_note_current.tex`.

## Executed result

| Object | Result | Scope |
|---|---:|---|
| qqq longitudinal tuples at nucleon K2=27 | 91 | Ordered slots before symmetry reduction |
| qqqg longitudinal tuples | 364 | Positive even gluon; positive odd quarks |
| Emitter-labeled emission routes | 1,092 | Kinematics only; not nonzero amplitude count |
| Worst inverse-Jacobi residual over all tuples | `4.44e-16` | Dimensionless |
| Worst emission transverse-metric residual | `8.88e-16` | Dimensionless |
| New direct tests | 59 passed | Algebra, measure, independent routes and symmetry |
| Affected basis/pair/M2k/M2l regression | 118 passed | No tolerances relaxed |
| Entire M2 plus Q0–Q2 regression | 342 passed | 309 M2 plus 33 preserved quantum tests |

The broad run took 281.74 seconds and retains 1,884 inherited mpmath
deprecation warnings and one sparse-efficiency warning. This is not a
full-repository test claim. The previous 451-test pair/bridge run remains
historical evidence; its 168 bridge tests were not all repeated because this
increment does not modify their source.

Reproduce the broad run from the repository in its declared quantum
environment:

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q -p no:cacheprovider tests/test_m2_*.py \
  tests/test_q0_plhqcd0.py tests/test_q1_plhqcdstate.py tests/test_q2_plhqcdobs.py
```

The verified environment used here is
`/private/tmp/deuteron-quantum.i3XrCg/env/bin/python`, reconstructed from the
repository environment file in the preceding oracle work. This temporary
path is not a permanent installation requirement.

## Coordinate and measure result

At a binary split A|B, with `X_A=sum_A x_i`, `X_B=sum_B x_i` and `X=X_A+X_B`,

\[
 Q_{A|B}=\frac{X_Bp_{A,T}-X_Ap_{B,T}}{\sqrt{X_AX_BX}}.
\]

The rows in tree postorder, followed by total P_T, form J_x with

\[
 J_x\,\mathrm{diag}(x_i)J_x^T=I,\qquad
 J_x^{-1}=\mathrm{diag}(x_i)J_x^T.
\]

Thus free intrinsic mass is `sum_i m_i²/x_i + sum_v Q_v²`. Its already
mass-weighted transverse terms do not get divided by daughter fractions again.

For flat intrinsic measures,

\[
 d\mu_\kappa=\prod_i d^2\kappa_i\,\delta^{(2)}(\sum_i\kappa_i)
       =(\prod_i x_i)\prod_v d^2Q_v,\qquad
 \Psi_x(\kappa)=\Phi_x(Q)/\sqrt{\prod_i x_i}.
\]

Independent exact rational determinants and direct momentum reconstruction
check this measure. Consistent Fourier-measure factors still belong in the
C45 matrix-element adapter; these coordinate factors do not replace field
or Fock normalization.

Merging the emitted gluon and its daughter quark recovers the parent
coordinates and leaves both spectators unchanged. Appending the global
qg relative coordinate gives an orthogonal map preserving the transverse
free mass. All 1,092 routes are tested. No emission strength is inferred.

## Symmetry closure, and what it does not imply

A permutation changes both fractions and coordinates. Its oscillator
rotation depends on the incoming longitudinal tuple. Therefore the new
joint action is not a fixed orbital matrix tensor-multiplied by the
longitudinal permutation.

Complete circular-HO excitation shells at one common global width are
closed under these rotations. The cutoff `N_exc` counts excitations,
excluding zero-point energy. The full sign projector uses longitudinal,
orbital, C7 color and the 64-state spin/flavor tensor together. It preserves
target Iz, I=1/2, Jz, charge and baryon number. It does not choose a state.

For proton I=1/2 and Jz=+1/2 at K2=27:

| N_exc | qqq rank | qqqg, h_g=-1 | qqqg, h_g=+1 | Total rank |
|---|---:|---:|---:|---:|
| 0 | 91 | 244 | 728 | 1,063 |
| 1 | 334 | 2,428 | 3,640 | 6,402 |
| 2 | 789 | 8,980 | 11,648 | 21,417 |

Ranks are traces of the joint sign/isospin/weight projectors, checked
independently against explicit dense projectors at a small resolution.
These are symmetry-allowed directions, not populated C53 source rays,
a new interacting-Hamiltonian contract, or a qubit-resource estimate.
Matrix-free projector application avoids constructing the full dense matrix.

**Finite-shell symmetry closure is established; interaction closure is not.**
A contact or emission operator may still reach outside the shell and
retained Fock sectors.

## The old-model comparison must be explicit

The equal-share qqq ground and all six M2k tree maps are recovered:
`Q_rho=q_rho/sqrt(2/3)`, `Q_lambda=q_lambda`, and common global
`b=0.4 GeV` imply `b_rho=sqrt(2/3)b_lambda`. Both proton/neutron
ground states and both helicities remain in the full sign image.

This is not an isometry of the whole old 266-state domain. In particular,

\[
 Q_{qg}=q_{\rm rel}/\sqrt{x_{\rm parent}}.
\]

At parent fraction 1/3, the old q_rel width 0.4 GeV becomes
`sqrt(3)*0.4 GeV` in Q_qg. It is not the new common width 0.4 GeV.
A lawful old-source map must keep the width-changing oscillator tail or
quantify a projection, and fix the excited-state C45 phase convention.
Do not silently call this mismatch a Hamiltonian correction.

A different-fraction transfer also changes the Gaussian metric. With the
longitudinal kets stripped off, the normalized transverse ground overlap
between (9,9,9) and (1,9,17), evaluated at fixed physical intrinsic momenta,
is `0.6829684165133539`. Independent Cartesian quadrature agrees.
This is not the full overlap of orthogonal longitudinal states, not a
pair-contact matrix element, and not the former fixed-physical-Gaussian
regulator away from equal share.

## Next executable scientific step

1. Derive and test the C45 phase/width comparison and a small
   unequal-fraction qq pair-action pilot on this carrier, including qq pairs
   within qqqg. Retain the original Gaussian diagnostic as a separately
   named regulator control.
2. Generalize the same source-owned C53 bilinear/projector to changing parent
   fractions and excited modes. Sum emitters coherently with one coupling.
   Do not assume the equal-weight spin-scalar cancellation persists.
3. Reconcile both mixed orderings and gluon-square contractions mode by mode.
   The bare qq factor cannot normalize the whole legacy C410 aggregate.
4. Measure action outside a retained shell using explicit larger regulated
   parents and several cutoffs. Then integrate H, rebuild its matched
   density/current, update encoding/padding and repeat exact quantum parity.

The first item is the next bounded result. The others state the continuation,
not prerequisites for every exploratory calculation.

## High-level path to the predictive framework

| Step | Scientific deliverable |
|---|---|
| 1. Finish the coupled operator model | Pair/emission amplitudes, remaining C117 products, explicit basis maps, regulated Hamiltonian and truncation diagnostics |
| 2. Complete dynamics and physical currents | Required Fock/instantaneous/constrained/induced/counterterm content, normalization/matching, conserved one- and many-body currents, and a lawful microscopic-to-nuclear deuteron construction |
| 3. Scale the topology-aware quantum solver | Symmetry-preserving tensor-network/quantum ansatz benchmarked against exact oracles; may advance alongside steps 1–2, without hardware being mandatory |
| 4. Identify and calibrate the physical system | State tracking, shared calibration conditions across resolutions, nuclear mechanisms, uncertainties/covariances and withheld validation |
| 5. Close the common GTMD parent | Quark/gluon off-forward operators, gauge links, soft subtraction, scheme-qualified matching/evolution and required-order identities and sum rules |
| 6. Deliver predictive observables | Correlated TMD/GPD/PDF/Wigner, form-factor and process predictions from that parent, convergence/holdout checks and a reproducible release with explicit coverage |

This is a dependency map, not a promise that the remaining research consists
of six small coding tasks. The operational phenomenology track remains useful
in parallel. Formal all-order identities are not claims of completed all-order
numerics, and “all observables” means those supported by the completed operator
and factorization scope.

## Theory and document verification

The main high-level note now contains the derivation in section 18.6 and its
status chapter distinguishes the new local source from the unchanged
represented-M2 oracle. The six current-theory consistency tests pass.
Tectonic 0.17.0 typeset the updated source to 129 pages; PDF-skill visual
verification checked the status page and all three pages of the new section.
All 331 labels are unique, all 48 checked reference targets and 76 cited
keys resolve, and no rendered question-mark placeholders occur. The final
log has no overfull boxes or missing characters. Inherited underfull-box
and generic undefined-reference/BibTeX-rerun notices remain; a warning-free
build is not claimed. The PDF is a verification intermediate, not a new
requested deliverable or theory volume.

## Preservation and publication

Pre-edit pickup notes, theory source, machine index and theory-consistency test
are preserved from `844f180c` at
`computation_handoff/backups/2026-09-06_pre_coupled_jacobi/pickup_theory_and_test.tar`.
Original theory archives and bibliography remain unchanged.
No public mirror refresh or push is part of this increment.

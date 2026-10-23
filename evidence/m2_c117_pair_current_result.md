# First C117 quark-pair result — 6 September 2026

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

The first bare quark–quark cross-pair kernel is implemented and directly
tested. It establishes **nonzero action outside the current equal-share
longitudinal carrier**. This is an exploratory source-derived result, not a
complete C117 finite-C43 coordinate action, physical spectrum, or replacement
for the represented M2m Hamiltonian.

Implementation: `src/deuteron_wigner/quantum/m2_c117_pair_current.py`.
Direct tests: `tests/test_m2_c117_pair_current.py`.
The main theory note contains the full derivation in `sec:c117-pair-current`.
This result supersedes the preparation-only status of the
[launch brief](https://github.com/uva-spin/DeuteronWigner/blob/b6fbd04239f8cf9573b54aeec75be0b527881012/docs/next_level/c117_full_current_extension_plan.md).

## Ordered sign and canonical quark normalization

The source operator agrees with the instantaneous-current term of
[Srivastava–Brodsky, Eq. (24)](https://arxiv.org/html/hep-ph/0011372v2#S4).
The following finite-cell and pair results are derived here, not supplied by
that reference.

On longitudinal functions, inverse `partial+` is anti-Hermitian, whereas
`R=Q0(i partial+)^-1Q0` is Hermitian. For an operator-valued Hermitian current,
`Rj(x)` instead has a pointwise **Fock** anti-Hermitian adjoint.
For `j(x)=sum_n j_n exp(-i*pi*n*x/L)`, `j_-n=j_n^dagger`,

\[
\int_{-L}^{L}(Rj)(Rj)\,dx
=-2L\sum_{n\ne0}\frac{j_nj_{-n}}{(\pi n/L)^2}.
\]

Thus the source's `-1/2` multiplies a negative ordered square, not a positive
Gram product. Spectral differentiation and coordinate integration agree for
noncommuting complex currents. C114's descriptions now distinguish the two
adjoints and the positive kernel from the ordered square.

The normalized good field `chi=2^(1/4) psi_+` has
`J_q^+=chi^dagger T chi`. Canonical discrete anticommutators, unit transverse
modes and box modes give one factor `1/(2L)` per number-preserving quark
density. An already normalized discrete Fock ket gets no extra box factor.
For distinct quarks, opposite nonzero integer transfers n and -n, and
unchanged spectator, both ordered constituent terms give

\[
I_{ab;cd}=\int d^2r\,\phi_a^*\phi_b^*\phi_c\phi_d,\qquad
\frac{V_{ij}^{P^-}}{g_s^2}
=\frac{L}{2\pi^2n^2}I_{ab;cd}\,T_i\cdot T_j,
\]
\[
\frac{V_{ij}^{M^2}}{g_s^2}
=\frac{K_2}{2\pi n^2}I_{ab;cd}\,T_i\cdot T_j .
\]

One ordered contribution is half these expressions. `2P+=pi*K2/L` cancels L
once; no extra active-subsystem factor is inserted. Units are GeV and GeV²,
respectively. Physical coupling and finite-C43 matching remain unselected.

The four-mode HO integral preserves C45 phases and angular/radial labels.
Coordinate quadrature independently checks nontrivial transitions and width
scaling. A two-fermion Fock-space calculation verifies the cross-pair exchange
sign and two-order multiplicity after subtracting the separate one-body term.
Full C7 qqq/qqqg color matrices pass Casimir, Hermiticity and S3 covariance
tests; individual qg pairs mix the singlet coordinates.

## What this reveals about the old C410 shape

C408's quark self-contraction uses the positive kernel Bqq, while C410 stores
`-Bqq/2`. The canonical **bare quark** mass term is `K2*Bqq/(4*pi)`:
`-K2/(2*pi)` times that stored shape. At K9, the legacy scalar is
`-0.193343782719`; the canonical result is
`+0.276944568622 GeV²` per `g_s²`.

This exposes a source-to-primitive sign/normalization mismatch. The legacy
exploratory Hamiltonian is preserved, not silently flipped. Its unbound
adapter never established a physical coordinate interpretation.

Nor may that quark factor simply multiply the entire aggregate: the mixed
C408 term uses the raw gluon factor `-(k'_g+k_g)`, while canonical C151 field
normalization gives `-(k'_g+k_g)/(2 sqrt(k'_g k_g))`. This ratio varies across
partitions. One scalar does not, by itself, normalize all four products.

## Nonclosure witness at nucleon K2=27

The bounded diagnostic retains all **91 positive APBC quark triples** and a
fixed physical-coordinate Gaussian. Its equal-share member is exactly the
M2l intrinsic global ground: `b_single=b_lambda/sqrt(3)`, contact
`b_lambda²/(6*pi)`. An orthogonal physical Jacobi transform separates the CM
Gaussian. The independent M2l pair-relative width `b_rho/2=b_lambda/sqrt(6)`
gives the same contact.

Keeping that physical Gaussian fixed away from equal share is an explicit
regulator extension, not a statement that unequal-share x-scaled Jacobi
widths stay unchanged. The outgoing modes are valid normalized probes of
action beyond the old equal-share slice.

At `b_lambda=0.4 GeV`:

| Quantity | Result |
|---|---:|
| Longitudinal matrix dimension | 91 |
| Nonzero matrix entries | 2,184 |
| Coordinates reached from `(9,9,9)` | 24 |
| Permutation orbits reached | 4 |
| Contact integral | `0.00848826363157 GeV²` |
| Projected outside-equal-share action norm per `g_s²` | `0.0618654126653 GeV²` |

The four sorted doubled-momentum triples are `(1,9,17)`, `(3,9,15)`,
`(5,9,13)` and `(7,9,11)`. The diagonal pair element vanishes by exact
zero-transfer exclusion, but the outgoing action does not. It preserves
total momentum and the total fermion sign with the existing color and
proton/neutron spin-flavor states, for both helicities.

The reported norm includes projection onto the stated transverse Gaussian;
it is not the norm of the full untruncated contact action.
This is a witness of missing action, not a proof of transverse/Fock closure
of the 91-state diagnostic. It shows that projecting into the unchanged
266-state M2m space would lose real matrix elements. Additional Jacobi,
longitudinal and qqqg completion remains necessary.

## Validation and unchanged boundaries

The 33 new direct tests pass. The initial broad run had 427 passes and three
failures caused solely by C405's old byte hash of the corrected C114
description. That one reviewed source pin was updated, with the guard retained;
the 62-test affected suite then passed. No tolerance was relaxed.
The final combined run is **451 passed in 314.92 seconds**: 250 M2 tests,
33 preserved Q0--Q2 tests and 168 affected bridge tests. This is a named
regression, not a full-repository pass. The 1,884 inherited mpmath deprecation
warnings and one Q2 sparse-efficiency warning remain visible.

Reproduce from the repository root in the declared quantum environment:

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q -p no:cacheprovider tests/test_m2_*.py \
  tests/test_c114_icurrent.py \
  tests/test_c404_c117_i2_longitudinal_color_primitive.py \
  tests/test_c405_c117_i2_current_topology_embedding.py \
  tests/test_c408_c117_i2_weight_routing_closure.py \
  tests/test_c409_c117_i2_derivative_density_reconciliation.py \
  tests/test_c410_c117_i2_retained_aggregation_boundary.py \
  tests/test_c411_c117_i2_finite_c43_adapter.py \
  tests/test_q0_plhqcd0.py tests/test_q1_plhqcdstate.py tests/test_q2_plhqcdobs.py
```

C410/C411 numerical shapes, M2m H, M2n current, M2o encoding and the exact
PennyLane oracle are unchanged. No extra C53 coupling, fit, production current,
variational algorithm, hardware claim or physical activation is introduced.
Complete source-qualified C117 coordinate actions still number zero.

## Theory-note verification

The main theory note was updated and typeset with Tectonic 0.17.0 to 126
pages. PDF-skill visual verification checked the ownership table and new
equations, including the explicit transverse projection in the leakage norm.
All 324 labels are unique; all 47 checked reference targets and 76 cited
keys resolve, and no rendered question-mark placeholders occur. The final
log has no overfull boxes or missing glyphs. Generic undefined-reference and
BibTeX-rerun notices, along with underfull-box notices, persist; a
warning-free build is not claimed. The PDF is a verification intermediate,
not a new deliverable or replacement theory volume. The six current-theory
status/consistency tests also pass.

## Next executable step

1. Derive the coupled longitudinal/Jacobi carrier required by these transfers
   and the same source-owned C53 vertex; retain the former equal-share model
   as a labeled controlled limit.
2. Reconcile the remaining mixed/gluon normalization at the mode-matrix level,
   preserving all color-pair matrices and ordered products.
3. Establish basis maps and closure before assembling H. Recover the old
   identified contribution, not necessarily its total spectrum.
4. Rebuild the matched current and encoding, then repeat exact
   compact-matrix/PennyLane parity.

These derivations do not require selecting physical finite-C43 matching or
a production current by default.

## Backups and publication

The pre-edit source and notes are preserved from `fb5bef05` in
`computation_handoff/backups/2026-09-06_pre_c117_pair_kernel/source_and_notes.tar.gz`.
The dependent C405 audit file and theory-status test are in the adjacent
`additional_audit_files.tar.gz`. Original theory archives remain untouched.
No public-mirror refresh or push has been performed.

# Common-order canonical tree four-current: executed checkpoint

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Scientific source: `f910bef7e30fb1c0707ca91f0ca56870a1a49362`.
The seven minus-current modules and their 130 direct tests complete the
normal-ordered canonical tree-current blocks represented through order g⁴
on the declared H2 parent. This is NOT all order-g⁴ QCD, a complete physical
current, restored rotational covariance or a prediction for a physical nucleon.
H2-first and canonical global-Jacobi Galerkin authority remain resolved.

The living theory note's `sec:fourth-order-minus-current` gives the derivation;
`m2_four_current_quantum_results.json` records actual measurements and bounds.
The earlier plus/transverse benchmark and historical results are unchanged.

## Operator content and matching

For the constrained field B=B[0]+g B[1], the minus spin current contains
B[0]†B[0], g(B[0]†B[1]+B[1]†B[0]), and g²B[1]†B[1].
Both signed constrained denominators are retained. At Delta+=0 they
coincide and produce 1/(r+)². In crossed quark/gluon scattering r+ may
be negative; the square is not an absolute-value prescription.
Physical charge, ordered color, field factors 1/sqrt(2 pi k2), and exact
Fock factors accompany each kernel. No mass-squared conversion is inserted
into the normalized current.

The implementation preserves:

- Free retained and all Q5/Q6 currents with both spin endpoints and N+2
  outside-shell momentum neighbors.
- One-gluon retained, P/Q5 and all four charged Q5/Q6 contacts, including
  crossed antiquark signs and pair creation. Linear parent momentum uses
  N_in+1 ket neighbors; relative coincidence moments use N_out+1 bra neighbors.
- Quadratic retained quark/gluon scattering, Pqqq→Q5gg pair emission,
  Q5 quark/gluon scattering and gg↔sea conversion, and both
  instantaneous-fermion Pqqqg→Q6 current families.
- Exact displaced local/pair coincidence integrals, complete color spaces,
  Bose/Fermi projectors and all raw wave tails. Factorized actions do not
  replace the declared parent by its old source-reachable range.

Set W5=g A1+g² A2+g³ C3, C3=C3_QQ+B5_Q6, W6=g² A6.
The raw current is the degree-four part of

    JPP[0] + g JPP[1] + g² JPP[2]
    + W5_f† (J55[0] + g² J55[2]) W5_i
    + W5_f† (g J5P[1] + g² J5P[2])
    + (g JP5[1] + g² JP5[2]) W5_i
    + W6_f† J66[0] W6_i
    + g (W6_f† J65[1] W5_i + W5_f† J56[1] W6_i)
    + g² (W6_f† J6P[2] + JP6[2] W6_i).

The same N_f^(-1/2) and N_i^(-1/2) finite-model maps normalize both ends.
The tests retain the whole-parent products' nonzero degrees five through
eight before extracting zero through four; they do not mistake the untruncated
product for an order-four result. They check the full kinematic LF boost,
reversed-transfer adjoints, independent Dirac/polar quadrature, charge/flavor,
statistics and omission witnesses. They do not prove dynamical rotations.

## Executed quantum measurements

Parameters: K2=9, retained cutoff 0/dimension 37, b=.4 GeV, mq=.35 GeV,
mg=0, g=.3, matching energy1.4 GeV², P+=1.3 GeV, Delta=(.12,.16) GeV.
One six-qubit controlled-gate state per target/cutoff supplies all 16 complex
helicity elements of J+, J-, Jx, Jy as 32 Hermitian quadratures.
The optimizer uses actual PennyLane energy/adjoint gradients and tree-state
retraction; exact eigenvectors are computed only after optimization.
Execution uses PennyLane's `lightning.qubit` statevector simulator with
analytic expectations (`shots=None`), not a physical quantum device or a
finite-shot noise/measurement-resource benchmark.

| Target | Q cutoff | M² (GeV²) | State residual (GeV²) | Maximum current error |
|---|---:|---:|---:|---:|
| proton | 1 | 1.421430407868028 | 9.4773e-9 | 2.5254e-11 |
| neutron | 1 | 1.421430407868027 | 7.3951e-9 | 2.2183e-11 |
| proton | 2 | 1.421428799077575 | 9.4776e-9 | 2.5253e-11 |

Every element satisfies its own residual/gap-derived bound. At cutoff 1, the largest
bounds are 6.027e-8 and 2.073e-8, respectively; padding is zero. Updates/
evaluations are 56/113 and 57/115. Concurrent wall times were 214.14 and 214.51 s;
these are not isolated performance comparisons or evidence of quantum advantage.
The larger-cutoff proton run also uses56 updates/113 evaluations, zero
padding and an independent maximum bound 6.0262e-8. Its wall time was
1367.61 s, including full current construction; it is not a quantum-only timing.

The full five-structure orientation diagnostic now yields:

| Target | Q cutoff | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| proton | 1 | -.880942275132 | -.312417387351 | .298437873255 |
| neutron | 1 | .881424003284 | .208278512303 | -.220680139623 |
| proton | 2 | -.880939719612 | -.312409773758 | .298429737890 |

Structure residuals are below 2.73e-14. This does NOT remove orientation
dependence. The proton magnetic extractions remain .882387267363 from plus
versus 2.644271817627 from transverse; the neutron gives -3.17995e-7
versus-1.762848324562. Conservation, kinematic boosts and a successful
five-tensor representation do not establish a physical covariant current.
Do not subtract the B terms or fit away this incomplete-model artifact.
At proton cutoff 2 the magnetic extractions remain .882386934506 versus
2.644266373730, with a five-structure residual 1.43e-14. Small changes between
these two excluded cutoffs do not establish retained/longitudinal or ultraviolet
convergence, nor resolve the order-one covariance problem.

## Verification and reproduction

The independent review-mirror run passed 169 tests in 332.66s: all 130
minus-current checks, six theory/status checks and 33 Q0--Q2 checks.
Imports were verified to resolve from the mirror source, not the canonical
package. This run used document export ba7370d2 and scientific source
f910bef7; the later cutoff 2 additions change documentation/results only.
Seven inherited escape-sequence/sparse-efficiency warnings remain.

All 130 new minus tests passed in one invocation in 193.18 s. The current M2
collection is1456 tests; collection is not a full-suite pass.
The separate surrounding constrained/antiquark/free-minus/orientation/Wick,
common transverse, six-body contact, projector and tree-retraction regression
passed 213 tests in 266.77 s at scientific source f910bef7.
The earlier complete named M2+Q0--Q2 result remains 1171 passed at 85717cae.
No complete current-source/full-repository single-run pass is claimed.

The six synchronized theory/status checks passed in 0.09 s. The scratch
theory render has 169 pages and 461 unique labels; all explicit references
and citations resolve. The new equations, numerical table and scope boundary
passed visual inspection. No rendered ??, final overfull box or missing
glyph was found. A generic inherited undefined-reference warning remains
despite that direct audit; it is not suppressed or reported as a clean log.

    python -m pytest tests/test_m2_one_gluon_minus_contact.py \
      tests/test_m2_shell_minus_current.py tests/test_m2_excluded_minus_contact.py \
      tests/test_m2_two_gluon_minus_current.py tests/test_m2_quadratic_minus_fock.py \
      tests/test_m2_quadratic_minus_actions.py tests/test_m2_fourth_order_minus_current.py \
      -q -p no:cacheprovider
    python scripts/benchmark_m2_matched_four_current_quantum.py --target PROTON --cutoff 1
    python scripts/benchmark_m2_matched_four_current_quantum.py --target NEUTRON --cutoff 1
    python scripts/benchmark_m2_matched_four_current_quantum.py --target PROTON --cutoff 2

## Next scientific path and recovery

1. Complete remaining Wick/current, interaction-projector and common field-
   regulator/propagating-loop/current-counterterm partners. Carry the same
   choices through H, norm and every current; preserve unresolved finite terms.
2. Test retained/excluded/longitudinal resolution and dynamical spin/rotation
   covariance. Establish shared physical finite conditions and input/holdout
   separation; an incomplete-current artifact is not a fit target.
3. Complete microscopic nucleon-to-nuclear spin-one/GTMD matching,
   conservation/sum-rule and angular-condition tests, and uncertainties.
   Develop genuine many-body topology compression and scalable measurements
   against controlled exact references. All three physical workstreams remain open.

Pre-edit notes, handoffs and entry points from 4e5fbc3c are preserved in
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_four_current_notes.tar`.
The complete old mirror is recoverable from commit 4e5fbc3c and also from the
local-only 946-MiB `pre_four_current_publication.tar` beside that backup;
the large redundant archive is not added to GitHub.
The original supplied TeX/Bib and 22-volume archive remain unchanged.
No new volume is required: this extends the existing living note's canonical
Hamiltonian/current and quantum chapters. Frozen volumes remain original sources.

# Source-derived free spin generators — 9 September 2026

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

The first constructive part of roadmap milestone 1 is complete: free quark
and massless-gluon Lorentz/Pauli–Lubanski actions, a finite-cell/circular-HO
reference projection, the moving source-Jacobi tangent, and norm-correct
transport identities. **The interacting many-body H2 spin action and spin
qualification are not complete.** Existing H/current kernels, elastic poles,
quantum results and B-coefficient diagnostics are unchanged.

Implementation: `src/deuteron_wigner/quantum/m2_free_spin_generators.py`.
Replay: `scripts/analyze_m2_free_spin_generators.py`; saved results:
`docs/next_level/m2_free_spin_generators_results.json`.
Direct checks: `tests/test_m2_free_spin_generators.py`.
The complete derivation is also in the living theory note,
`sec:source-free-spin-generators`.

Execution: **217 focused tests passed in 69.28 s**, including all 89 new
checks and the existing finite-H2, canonical-emission, spin-resolved-current,
elastic-spin and saved-spin-result regressions. This is not a full-suite run
or evidence of physical covariance. Replay uses the existing scientific
environment with `PYTHONPATH=src:.` and single-thread BLAS:

```text
python scripts/analyze_m2_free_spin_generators.py
python -m pytest -q -p no:cacheprovider tests/test_m2_free_spin_generators.py tests/test_m2_matched_h2_reduction.py tests/test_m2_canonical_emission.py tests/test_m2_spin_resolved_current.py tests/test_m2_cluster_spin_diagnostic.py tests/test_m2_cluster_spin_results.py
```

The six existing theory/status checks also pass after synchronization. The
offline 226-page scratch theory build succeeds; the new three-page derivation,
title and status pages are visually checked. No final-pass overfull boxes or
literal unresolved-reference markers are found. Generic inherited
bibliography/reference/package warnings remain. This is focused layout QA,
not full-document scientific certification or a new PDF deliverable.

## What was derived

Use symmetric p±=(p0±pz)/sqrt(2), positive p+, active U(t)=exp(-itG), and
flat momentum amplitudes. Every free generator has

    G = -i [v·∂ + (∂·v)/2] + A.

The half-density term is absent only for invariant dp+ d²pT/(2p+) amplitudes.
For F1=(Bx−Jy)/sqrt(2), F2=(By+Jx)/sqrt(2), and
p−=(m²+pT²)/(2p+), the velocities are (px,p−,0) and (py,0,p−).
Differentiating the existing canonical LF spinors, rather than assuming a
constituent rotation, gives

    A_F1 = (−py σ3 − m σ2)/(2p+),
    A_F2 = ( px σ3 + m σ1)/(2p+),  A_Jz = σ3/2.

Matrices in the implementation use the actual source order **(down,up)**.
Independent finite Cartesian Lorentz transformations of the source spinors
verify all ten generator names. Transforming the existing gluon polarizations
and restoring A+=0 by a pure-gauge term gives the massless result
A_F1=−py h/p+, A_F2=px h/p+, A_Jz=h, h=diag(-1,+1).
A massive two-helicity gluon is not supplied or silently invented.

For quarks, W0=p·J and W=EJ−p×B cancel all orbital and half-density terms.
They give p·W=0, W²=−3m²/4, and the three intrinsic massive LF spin matrices
σ/2. Massless quarks/gluons instead obey Wμ=h pμ. The differential Lorentz
algebra passes, including [F1,F2]=0: **F1/F2 themselves are not SU(2) spin**.
This distinction and interaction dependence are consistent with the canonical
analysis of [Harindranath, Mukherjee and Ratabole](https://arxiv.org/abs/hep-th/0004192).

## Two different finite effects, neither discarded

For the actual centered finite-cell modes phi_k=exp(iπkx−/L)/sqrt(2L),

    X_nm = −i L (−1)^(k_m−k_n) / [π(k_m−k_n)]  (n≠m), X_nn=0,
    [X,P+] = i(I−B),  B_nm=(−1)^(k_m−k_n).

Independent quadrature checks both PBC and APBC, including nonconsecutive,
reordered modes. B is the cell-face/domain term: x−phi does not preserve the
momentum operator's (anti)periodic domain. Its spectral norm is N for N modes,
not a small roundoff error or an omitted-shell estimate. The replay gives
3,5,9 for N=3,5,9. Boundary-vanishing superpositions can have zero boundary
expectation; there is no claim of operator-norm convergence to canonical CCR.

The explicit free finite reference uses the source circular-HO phases and
projects Fi=−piX+{p−,−i∂pi}/2+Ai only AFTER forming its cubic orbital words.
An N+3 working shell is exact for those words. Independent HO quadrature
checks the result. For N=1, three longitudinal modes and two helicities,
the 18-state reference embeds exactly in the 90-state N=4 reference.
It satisfies [Fi,P+]=−ipi+i B⊗pi to below 4e−16 GeV (Frobenius residual).

For any explicit parent and isometry U, the exact identity is

    [U†AU,U†BU] − U†[A,B]U
      = −U†A(1−UU†)BU + U†B(1−UU†)AU.

The quark example separates projected parent-cell and transverse-leakage
norms 1.855643 and 2.507710; the gluon example gives 0.754489 and 0.984363.
These are matrices that add, not scalar norms to subtract. Reconstruction
residuals are below 3e−16. The parent-cell commutator is not called continuum
Lorentz closure. Complete/cut known spin multiplets separately test the
identity; they are references, not a new nucleon-state assignment.

## Moving fractions and consistent H2 transport

At a source Jacobi point, dP+/dt=P_i and
dx_a/dt=(p_ai−x_a P_i)/P+. The implementation also differentiates the
mass-weighted Jacobi frame. Independent finite Lorentz transformations verify
qqq and qqqg cases on both axes. Even at P_T=0 the saved K2=9 qqq example
has dx/dt=(0.0284339,0.00455179,−0.0329857): a transverse rotation moves
the fractions off their discrete grid points. This is a continuum tangent,
not a chosen interpolation or a completed fixed-K many-body operator.

At one common H2 anchor use U=Omega N^(-1/2). Raw matched products require
N^(-1) between operators. If U depends on momentum, a differential generator
also needs A_eff=U†AU−iU†(v·∂U), with the corresponding orthogonal leakage.
The implemented product rule and moving-frame gauge law are tested on an
explicit norm-matched reference. The required interacting wave/pole/norm
derivatives are **not** manufactured by this identity.

## Next executable scientific target

Derive the interaction-dependent dynamical generators from the same constrained
canonical energy-momentum tensor and common Gaussian source, including the
regulator/Wick and finite-domain partners. Construct the actual many-body
action and then transport it through the common-order H2 waves, both norms
and moving-frame connection. Total W must be built from total P and total
generators; it is not the sum of one-particle W or helicity matrices.
Only then qualify degenerate multiplets and re-evaluate the current with
independent longitudinal/oscillator changes. This work needs no physical fit
or new executive choice at present. No B subtraction, ad hoc Melosh rotation,
covariance restoration, physical nucleon assignment or new quantum run occurred.

Pre-edit source and all notes are recoverable at
`b6fbd04239f8cf9573b54aeec75be0b527881012`. The seven edited entry/theory
files were additionally archived before editing at
`computation_handoff/backups/2026-09-09_pre_free_spin_generators/theory_pickup.tar`.
Original volumes and archived sources are unchanged. This scientific update
is local; the review mirror and GitHub retain the preceding published snapshot.

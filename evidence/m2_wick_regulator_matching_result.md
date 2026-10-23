# Ordered Wick-current and interacting-projector continuation

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Publication update, 8 September 2026: this checkpoint is included in the
user-authorized canonical/review-mirror/GitHub synchronization. The source
marker in `computation_handoff/repo/CURRENT_SOURCE_COMMIT.txt` identifies the
exact release. No-push statements below describe the earlier scientific
completion checkpoint, not a restriction on this authorized publication.

The independent review-mirror validation of document export `c0b504ef`
(scientific source `520cc782`) passes 258 focused theory/Wick/reference/H2/
envelope/isospin tests in 488.15 s, all 15 full-current resolution checks
in 306.82 s, and three scope/input checks in 7.41 s. Imports resolve from
the mirror source, not the canonical package. Six inherited Python
escape-sequence warnings recur in these runs; no full-repository test or
new quantum calculation is claimed. The later publication-note update does
not change numerical code or saved matrices. All 22,270 selected canonical
files and all 45,152 published inventory entries are independently reconciled.

Finite-model completion at local source `520cc782c92663bbe6dd166ddfa2c19cf72aa1fd`, following published
main `ce14fb1206b74239a3f74aa056c19af960ef9f5c`. No public push.

Local source 520cc782 completes the requested finite-model Wick, common-regulator and H2 loop/four-current matching through raw order g4 at Delta+=0 with explicit finite parameters and both endpoint norms. All long calculations and independent Q5-shell-two checks are finished. Q5 shell1-to2 changes the g4 coefficient by 20.7% despite only 0.0101% normalized-total change; retained and longitudinal sensitivity also remain. This is validated finite-model completion, not cutoff convergence, physical renormalization, restored covariance or completion of the entire project. No public push.

## Completion audit: requested finite-parameter scope

The audit covers the user's requested Wick, regulator and loop/current work
for the chosen canonical global-Jacobi H2-first model. It checks implementation
and independent derivations/tests, not merely total-current conservation or
status flags. All primary runs are finished and their original matrices are
preserved. The model parameters are explicit uncalibrated test values.

| Requirement | Completion evidence | Scope retained |
| --- | --- | --- |
| Ordered Wick H/current and subtractions | Unreordered Fock algebra, physical external/internal attachment paths, per-mode H derivative checks, matched subtraction before projection, PP/Q5 H2 returns and both norms | Normal ordering is a scheme, not determination of physical finite terms |
| Common regulator | Same whole physical-leg Gaussian in cubic/quartic vertices, sources, QQ/U65, Wick terms and their independent gauge/projector derivatives; shaped family is inserted before projection and Wick | Auxiliary factorization legs are not photons; source convergence is not loop convergence |
| H2 loop/four-current | Independent H/wave recurrences through g4; free, cubic, quartic-source/pair, Wick and exact P/Q projector contributions; both energy/spin endpoints and norm maps | Delta+=0, gapped spectral envelope, flavor-degenerate actual total-I=1/2 source domain; no full-QCD g4 claim |
| Algebra and symmetry | Per-order Ward, reversal, kinematic boost, charge/H divided difference, independent minus derivatives, exact source-domain isospin cancellation | Transverse continuity does not determine minus or restore dynamical covariance |
| Finite-reference matching | Whole-vertex degree0/1/2 comparison of all current components; degree-limit derivation and two-norm bounds; independent scalar/batched eleven-array byte equality | Fixed finite reference and explicit widths/conditions, not regulator removal |
| Resolution behavior | Independent Q5/Q6 changes, completed full Q5 shell2, retained physical embeddings with both metrics, K9/K11 physical-label probes, nested finite-Q wave/plus bounds and finite-mass attribution | Measured sensitivity remains; no infinite-space error certificate or physical convergence |

Consequently the requested finite-parameter research goal is complete. The
assembled four-current's completion flag refers to exactly this scope.
Partial Hamiltonian, scalar-shape and numerator objects remain partial;
none can silently select a full current. Physical flags remain false.
This is neither an assertion that the entire project is finished nor a
new requirement to fit nature before continuing parameterized development.

### Final Q5-shell-two result

The original 17107.41-s run saves all eleven complex 37-state matrices in
`m2_gaussian_cluster_four_current_Q5_shell2.npz`; its adjacent JSON
contains all component/order comparisons and both-norm bounds. It changes
only Q5 from shell1 to2 at fixed Q6 shell0, K9 and P shell0. Raw Q dimensions
are 22230/130500/792/7200. Parameters: b=.4 GeV, mq=.35 GeV, mg=0,
g=.3, P+=1.3 GeV, Gaussian width=.8 GeV, eight boson/eight fermion modes,
paired normal ordering and finite mu_q²=.011, mu_g²=-.007 GeV² per g²;
Ef=1.39, Ei=1.47 GeV², Delta=(.12,.16) GeV, mean PT=0, proton Jz=+1/2.
The physical-leg endpoint-product, smooth fixed-membership Jacobi, gapped
spectral envelope and total-I=1/2 H2 prescriptions are unchanged.

| Raw order | Full four-component relative change, Q5 shell1 to2 |
| --- | ---: |
| g0, g1 | exactly zero |
| g2 | .0139613676 |
| g3 | .5895431342 |
| g4 | .2069301018 |

At g4 the plus/minus/x/y changes are 17.2394%, 28.2183%, 12.1006% and
12.1489%. The normalized total changes by only .0100520%; this does not
hide or resolve the raw-order dependence. Its absolute Frobenius change
.000913800481 is bounded by .00219136638 after both positive norms.
Against shell0 the g4 change is 126.47%. None establishes convergence.

The raw/normalized Ward norms are 1.88697e-15/1.88628e-15. Independent
H2 coefficient identities and both norms pass in 122.02 s. Four further
checks pass in 1.26 s. They compare every saved matrix/component, the
finite-reference bound, and all five plus coefficients plus both norms
against the independent nested-Q calculation. Plus g0..3 and both norms
match exactly; the full g4 matrix difference is 3.22e-19, not just equal
scalar norms. Raw/normalized plus differences are 3.42e-21/5.36e-21.

Final focused regressions: all 252 tests pass in 448.40 s across the
Gaussian Wick H/current, whole-vertex reference kernel/H2, spectral envelope,
total-isospin H2 and common H2 files. The completion-scope and invalid-input
checks pass separately (three tests, 5.57 s, including one new scope check).
The current M2 collection is 2854 tests in 5.90 s, not full-suite execution.
The expensive full-current reversal/boost checks were completed earlier;
this final audit changes capability metadata and routing messages, not
their numerical kernels, and does not misreport them as newly rerun.

Original scratch outputs, source/document history through `5dc5966d`,
and all earlier pre-edit archives remain. No matrix coefficient, old
report, original theory document or unrelated user file was removed.

### Next scientific path

Use the explicitly bound common-regulated H2 four-current as the new finite-model classical reference for the existing PennyLane/graph interface, retaining both energies/norms, every component and raw coupling order. Continue resolution and dynamical spin/rotation diagnostics without fitting away defects. Physical current selection, calibration data/covariance and finite-condition choices require the theory lead; nuclear spin-one/GTMD matching and topology-aware many-body scaling remain downstream. Do not restart completed benchmark jobs or reopen the resolved canonical/H2-first choice. No physical activation or public push is selected.

The sections below are chronological evidence. Their then-open steps and
then-running jobs are historical, not additional current work queues.

## Ordered quadratic-current Wick algebra

The canonical constrained field gives
`J^-_[2]/g_s² = (alpha_T A_T psi_+)^dag (2 r_L^+ r_R^+)^-1
(alpha_T A_T psi_+)`, with the actual signed Fourier modes retained.
At zero longitudinal transfer, after subtracting the vacuum identity,
the quadratic descendants are quark, antiquark and gluon number operators.
Linear-current vacuum contractions vanish by the single-generator color
trace. Positive longitudinal field modes exclude number-changing quadratic
descendants at this transfer; this does not remove zero modes of the theory.

For the same field envelope as the preceding Hamiltonian Wick calculation,
define
`A_R = I_R K2² / [2 pi (P+)²]`, with bosons `k2=2,...,2Nb` and
fermions `r2=1,3,...,2Nf-1`. The local J-/(2P+) coefficients per g_s² are

- quark, with its physical charge external:
  `w_q(p2) = C_F A_R sum_k 1/[k2 (p2-k2)²]`;
- antiquark, with its physical (negative) charge external:
  `w_anti(p2) = C_F A_R sum_k 1/[k2 (p2+k2)²]`;
- gluon, including explicitly supplied quark-loop flavor charges:
  `w_g(k2) = T_F A_R (sum_f e_f)/k2
  sum_r [(k2-r2)^-2 + (k2+r2)^-2]`.

These are coefficients of the ordered composite, not a chosen
charge-conjugation-symmetrized or physically renormalized current. The
antiquark result is a Pauli-blocking subtraction from the vacuum and has
signed constrained momentum `-(p2+k2)`. The gluon result is stimulated
emission/absorption in the fermion-vacuum contraction. Its bare electric
charge remains zero. Applying a cancellation theorem for a complete
two-gluon/one-photon amplitude to this isolated equal-light-front-time
contraction would incorrectly discard it.

The implementation is `m2_current_wick_self.py`. Independent tests apply
the unreordered constrained field directly to vacuum and one-particle Fock
states, retaining exact creation/annihilation signs and Bose factors, and
subtract vacuum expectation values only afterwards. They also test the
antiquark external-momentum derivative of its ordered instantaneous-fermion
self energy and the internal charged-fermion derivative of the gluon self
energy. Quark and antiquark coefficients must not be equated by assumption.
The connected identity has no dependence on the many-body oscillator cutoff.

Normal ordering removes every displayed contraction with its negative,
in both H and the current. That is a composite subtraction scheme, not a
determination of physical finite mass, vertex or current conditions.
The existing normal-ordered tree-current calculations are unchanged.
The following H2 extension now carries these operators through the actual
coupled-sector wave maps. It is more than a scalar cancellation test.

## Interacting oscillator-projector partner

The existing regular straight-path tangent is now projected on explicit
outer canonical symmetry bases. For inner embeddings `p_i,p_f` and
`Pi_a=p_a p_a^dag`, it satisfies
`Delta_j K_j = Pi_f rho - rho Pi_i` and
`K_j = Pi_f K_j + K_j Pi_i`.
Exact continuum translation matrix elements are used; the path integral is
numerical and its quadrature order is explicit.

For the supplied outer Hamiltonians `H_i,H_f`, define

`B_j = p_f^dag (K_j H_i + H_f K_j) p_i / (2P+)`.

With `h_a=p_a^dag H_a p_a` and `rho_P=p_f^dag rho p_i`,

`2P+ Delta_j B_j =
 h_f rho_P - rho_P h_i
 - p_f^dag (H_f rho - rho H_i) p_i`.

This is the full inner-cutoff boundary for the represented interacting H,
including its cubic and both quartic source families, not a free-H-only
approximation. The smaller pilots identify coordinates; the supplied
parent Hamiltonian determines the inner Hamiltonian in this identity.

The outer oscillator cutoff is still explicit. Unlike quadratic H0,
contact sources have arbitrarily high oscillator tails. An exact identity
for each finite outer parent is not a proof that this outer cutoff has
converged, nor that the field regulator/propagating loops are matched.
No correction is automatically added to an existing H2 snapshot.
The transverse path freedom and the known covariance discrepancy remain.

At K2=9, retained cutoff 0 and outer cutoff 2, b=.4 GeV, mq=.35 GeV,
mg=0, g=.3, P+=1.3 GeV and transfer (.12,.16) GeV, the proton boundary
source norms (free, cubic/g, cross-quartic/g², IF-quartic/g²) are

- same helicity: `4.62e-16, .0694022575484, .0143967528851, .00199511440069`;
- opposite helicity: `2.38e-15, .143939448628, 1.53e-17, 6.38e-19`.

The parent has 654 states per helicity and the inner space has 37.
At outer cutoff 1 the same-helicity ground-shell tangent is exactly zero:
a spin-scalar density cannot change orbital Lz at fixed individual
helicities and fixed total Jz. The opposite-helicity block is nonzero.
This is selection, not cutoff convergence.

## Paired one-body subtraction through the full H2 wave map

The new `WickOneBody` uses one explicit field envelope, coincident trace,
sea flavor count, normal-order choice and two finite kinetic mass-squared
coefficients. The latter multiply **g_s²**, have units GeV² and modify only
the kinetic mass directions; they do not silently change the vertex mass.
The ordered antiquark Hamiltonian IF sum is `sum_k 1/[k(p+k)]`.
Its IG sum equals the quark IG sum, including the exact zero-mode exclusion.
Independent unreordered field and current-square Fock calculations check both.

With the complete raw five-body resolvent `R5`, leading wave `A1=R5 V1`
and summed one-body coefficients `S_P,S_55`, the addition is

`delta C3 = R5 S_55 A1`,
`delta H_eff = g² S_P + g⁴ V1^dag delta C3`,
`delta N = g⁴ (A1^dag delta C3 + delta C3^dag A1) = -d_E delta H_eff`.

Sector sums are understood. The existing non-diagonal free-shell propagation
is retained. The local H2 matrix uses the newly linearized H and the new
norm, without eigenvalue clipping. A Q6 one-body insertion first contributes
at g⁶; it is correctly absent from this g⁴ model, not forgotten.

The new direct minus-current coefficients are `w_P` at g² and
`A1_f^dag w_55 A1_i` at g⁴. They are added to the canonical tree current
evaluated with **C3+delta C3 at both endpoints**. Thus both the free Q5
current and the P↔Q5 contact inherit additional raw backflow. Omitting it
changes the result at the same order. Plus/transverse currents also inherit
the changed waves and norms. All four use both endpoint whiteners.

The finite quark kinetic partner is `e_q mu_q²/[2(p_q+)²]` per g².
There is no bare Abelian gluon kinetic current. Ordered gluon-loop charges
remain explicit and must have the Hamiltonian's trace flavor count.
All four represented Q5/Q6 one-body projections retain every orbital,
color and spin column. The H2 snapshot has a separate interface so bare
minus-current evaluation cannot silently discard the direct partner.

### Executed controls

`scripts/benchmark_m2_paired_wick_h2.py` produces the saved
[finite-model results](../snapshot/docs/next_level/m2_wick_regulator_matching_results.json).
At K2=9, retained cutoff0, b=.4 GeV, mq=.35 GeV, mg=0, g=.3,
matching E=1.4 GeV², P+=1.3 GeV and transverse transfer (.12,.16) GeV,
it compares seven field members under each of three explicit conditions:
ordered, zero-finite normal ordering, and illustrative finite normal ordering.
The longitudinal counts and transverse trace width are varied separately
as well as together; the Gaussian trace is Lambda²/(4 pi).

The cutoff2 calculation retains 37 P coordinates and raw Q dimensions
22,230, 130,500, 43,692 and 364,800, totalling 561,259 parent coordinates.
At longitudinal counts8, increasing Lambda from .4 to1.6 GeV changes the
ordered M² from1.429602384722 to1.552206030031 GeV².
At fixed Lambda=.4 GeV, increasing counts8→16 changes it to1.429507243744.
Zero-finite normal ordering recovers the original M²=1.421428799078 GeV²
and entire H/norm matrices exactly for every field member.
Illustrative finite coefficients mu_q²=.011 and mu_g²=-.007 GeV² survive,
giving1.430342524485 GeV² for every field member; this is **not a fit**.
The same exact cancellation holds for the directly inserted current.

The independent cutoff0 control computes the **full** matched minus
current, including the changed backflow. The larger cutoff2 control records
the **direct** current insertion only; its full-current field is null, not
an inferred full measurement. Both calculations use classical diagonalization,
not PennyLane or hardware. Every raw extra-wave equation and normalized
charge test passes. Exact subtraction cancellation at either cutoff is not
evidence of propagating-loop ultraviolet convergence: the Gaussian trace
does not yet regulate the pre-existing propagating vertices.

### What this does not settle

Normal ordering pairs the known ordered composites. It does **not** supply
all vertices obtained by gauging internal contractions and the regulator.
At zero longitudinal transfer, a longitudinal one-body scalar commutes
with the good density, so that Ward identity does not determine a missing
minus-current loop term. Finite kinetic mass conditions also do not fix
vertex-mass or independent transverse-current conditions. Physical
calibration, dynamical covariance and continuum renormalization remain open.

## Full Q-sector regulator vertices and free-propagation return loops

The straight-path oscillator projector now acts on every packed coordinate
in all four Q5/Q6 sectors, with explicit inner and outer cutoffs, path
quadrature, transfer and charge. It preserves the exact particle-statistics
projection and both helicity endpoints. It satisfies
`Delta.K = Pi_f rho - rho Pi_i` and `K = Pi_f K + K Pi_i`.
The first-shell same-helicity zero is a selection rule, not a missing term
or a convergence plateau. Second-shell radial entries are retained.

For a supplied **coupled** outer Hamiltonian action, the raw inner boundary is

`B_H = Pi_f (K H_i + H_f K) Pi_i / (2P+)`.

The exact finite-parent product rule is checked separately against the
canonical quartic Q5 scattering/conversion, cubic Q5↔Q6, free and one-body
Wick actions. Sea and six-body conversion outputs are nonzero and must not
be replaced with sector-diagonal actions. The longitudinal one-body Wick
scalar has zero inner oscillator-projector boundary because it commutes
with that projector; its separate minus/gauged-contraction partners are
not inferred to vanish.

### Moving-projector terms in the propagator

The embedded propagator is `R(E)=Pi(E-Pi H0 Pi)^-1_Pi Pi`.
It is **not** `Pi(E-H0_outer)^-1 Pi`. The difference has an independent
ground-to-second-shell Schur-complement check for all four sectors, and
the non-diagonal inner-shell equation is checked against the projected
outer free Hamiltonian.

The transverse regulator term is

`B_R^j = (K^j R_i + R_f K^j)/(2P+) + R_f B_H0^j R_i`.

The first two terms come from the moving endpoint projectors. They cannot
be dropped or compressed into the inner Hamiltonian term. With
`Gamma_R^+ = R_f rho R_i` and
`Gamma_R^j = R_f J_parent^j R_i + B_R^j`, the independently checked identity is

`(E_f-E_i) Gamma_R^+ - 2P+ Delta_j Gamma_R^j = rho R_i - R_f rho`.

The energies here are mass squared. This is the zero-longitudinal-transfer
identity; it is not a determination of a complete nonzero-skewness
four-current. For the free parent, the outer shell includes N_inner+2
to retain its full Hamiltonian boundary. Checks cover the entire outer
coordinate set, unequal energies, reverse helicities/transfer, zero
transfer, the energy derivative and the exact pole prohibition.
Neither energy-difference nor transverse-transfer division is introduced.

### Actual canonical Q5 return diagrams

`Q5LoopRegulatorCurrent` uses the actual uncompressed outer cubic and
quartic source references, `V = g V1 + g² V2`. Its regulator coefficients are

`delta J_reg,[n] = sum_(r+s=n) V_r,f^dag B_R V_s,i`, `n=2,3,4`.

Every source keeps its **outer** tail before gauging. Replacing it with
`Pi V` loses a nonzero current. Independent whole-source polynomials
check each coupling order; both endpoint reversals and actual-source
two-energy Ward contractions pass. These coefficients are before both
retained norms. They do not yet add the W55/Q6 loop/backflow regulator
terms to the full H2 snapshot or select a physical transverse current.

Implementation: `m2_shell_projector_current.py`,
`m2_projected_shell_resolvent.py`, and `m2_q5_loop_regulator_current.py`.
The 73 new direct tests passed in three focused runs: 52 packed-projector/
propagator checks (27.28 s), 13 actual coupled-H checks (104.46 s), and
8 actual-source Q5 return-loop checks (42.54 s).

The subsequent combined run passes all **104 tests in345.64 s**: those
73 new tests plus31 preceding retained-projector tests. The current M2
collection contains1605 tests, not a claimed full-suite pass.

The [executed Q-sector results](../snapshot/docs/next_level/m2_qsector_regulator_results.json) preserve
proton outer-cutoff2 and3 and neutron outer-cutoff2 runs, with both helicity
blocks and all three source orders. At K2=9, retained/inner-Q cutoff0,
b=.4 GeV, mq=.35 GeV, mg=0, E_f=1.39 and E_i=1.47 GeV²,
P+=1.3 GeV and Delta=(.12,.16) GeV, the largest contracted Ward residual
is2.334e-18. Omitting the regulator current raises it as high as.002041.
The outer-cutoff3 Q5 spaces have72,630 gluonic and409,500 sea coordinates.

Crucially, the proton spin-flip current norms summed at g=.3 change by
15.30% (x) and20.35% (y) from outer cutoff2 to3. These are raw regulator-
operator norms, before retained whitening, not form-factor changes.
Same-helicity equality at those two cutoffs follows the even/odd orbital
selection: with an inner ground orbital and spin-diagonal K, that block
receives only even outer shells. Opposite helicities receive odd shells.
Neither the unchanged same-helicity values nor roundoff Ward closure
establish convergence. The runs take31.36,170.64 and30.74 s respectively.

Reproduction:

```text
python -m pytest tests/test_m2_shell_projector_current.py \
  tests/test_m2_gauged_shell_resolvent.py \
  tests/test_m2_qsector_interacting_regulator.py \
  tests/test_m2_q5_loop_regulator_current.py \
  tests/test_m2_interacting_projector_current.py \
  tests/test_m2_galerkin_projector_current.py -q -p no:cacheprovider
python scripts/benchmark_m2_q5_regulator_current.py --target PROTON --outer-cutoff 2 --output /tmp/proton2.json
python scripts/benchmark_m2_q5_regulator_current.py --target PROTON --outer-cutoff 3 --output /tmp/proton3.json
python scripts/benchmark_m2_q5_regulator_current.py --target NEUTRON --outer-cutoff 2 --output /tmp/neutron2.json
```

### Remaining scientific path

Next carry the product-rule regulator terms through the interacting
`R5 W55 R5` insertion and the complete Q6 excursion, including source
contacts, five-body backflow and both norm/energy endpoints at order g⁴.
The P-projector/source partners must use the same outer coordinate system.
Then complete common field-regulator and gauged-contraction/propagating-loop
counterterm matching with explicit finite conditions. The present exact
finite-parent identities are not ultraviolet or outer-shell convergence,
physical calibration, dynamical covariance, or a quantum-device calculation.

## Paired-H2 checkpoint verification

The paired-H2 extension adds 50 direct checks (20 one-body, 13 projection,
17 H2/current). Together with the preceding Wick/projector checks, all
121 tests pass in one invocation (269.04 s). The M2 collection now contains
1532 tests; no current full-repository or full-M2 pass is claimed.
The 42 recorded benchmark members pass: 21 full-current cutoff0 controls
in66.27 s and 21 direct-current cutoff2 controls in151.13 s. Across them,
the largest raw extra-wave equation residual is1.194e-16 and the largest
charge error is5.863e-15. The cutoff0 normalized full minus-current
ground expectations for counts8/Lambda=.4 are .354284206224 ordered,
.346287322675 zero-finite subtracted and .348621937411 illustrative-finite.
Their negligible imaginary parts are retained in the machine result.

Reproduce the new checks with:

```text
python -m pytest tests/test_m2_wick_one_body.py \
  tests/test_m2_wick_current_projection.py tests/test_m2_wick_h2_matching.py \
  tests/test_m2_ordered_current_wick.py tests/test_m2_current_wick_self.py \
  tests/test_m2_interacting_projector_current.py \
  tests/test_m2_galerkin_projector_current.py -q -p no:cacheprovider
python scripts/benchmark_m2_paired_wick_h2.py --target PROTON --cutoff 0 --current full
python scripts/benchmark_m2_paired_wick_h2.py --target PROTON --cutoff 2 --current direct
```

Preceding checkpoint:

The new 16 Wick tests and preceding 24 current-Wick tests passed together
(40 tests, 1.45 s). The combined 71-test Wick/projector run passed in
166.88 s: 26 new tests and 45 preceding tests in one invocation. This
tests the stated algebra and finite-parent identities, not physical
renormalization or a complete repository regression.

Reproduction:

```text
python -m pytest tests/test_m2_ordered_current_wick.py \
  tests/test_m2_current_wick_self.py \
  tests/test_m2_interacting_projector_current.py \
  tests/test_m2_galerkin_projector_current.py -q -p no:cacheprovider
```

The underlying constrained-field conventions are from
[Srivastava and Brodsky, Eq. 24 and Appendix B](https://arxiv.org/abs/hep-ph/0011372);
the product-rule gauging principle is discussed by
[Kvinikhidze et al.](https://arxiv.org/abs/0904.4128).
The explicit ordered-current coefficients and noncommuting oscillator
construction above are project derivations, not formulas attributed to
the commuting-cutoff example in the latter paper.

The Q-sector construction above advances this checkpoint's former next
action. Continue its W55/Q6 and P/source partners through full H2, then
common gauged-contraction/propagating-loop/current-counterterm matching.
Do not retain only a free-sector boundary or only a source-image projection
of the current tails.
Retain independent inner, outer, longitudinal and field-trace controls.
Do not relabel a finite-parent identity as continuum renormalization.

Pre-edit source/notes are preserved at
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_remaining_wick_matching_notes.tar`
from ce14fb12, as well as in Git history. Original supplied documents remain
unchanged. No publication is requested for this ongoing research turn.

The immediate pre-edit paired-H2 source/notes are additionally preserved in
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_paired_wick_h2.tar`
from c91af0f1. No original supplied materials were changed or removed.

Before the Q-sector extension, the source/notes from 5252a82d were preserved
in `computation_handoff/backups/2026-09-06_pre_canonical_self/pre_qsector_regulator.tar`.


## Interacting canonical H2 Q-projector transport

The new `m2_interacting_q_loop_regulator.py` completes the Q-only cutoff
derivative of the **normal-ordered canonical** H2 return through g_s^4.
This is an extension of the free-Q5 calculation above, not a claim that
common field/current matching or the persistent goal is complete.

On the full outer Q5+Q6 coordinates set
`S1=(V1,0), S2=(V2,D6P), R=diag(R5,R6),
U=U65+U56, W=diag(W55,0)`. All seven direct Q6 source components enter
D6P. R is the embedded **inner** resolvent; S/U/W keep their unprojected
outer tails. The return coefficients are

```text
Sigma2 = S1† R S1
Sigma3 = S1† R S2 + S2† R S1
Sigma4 = S2† R S2 + S1† R W R S1
       + S1† R U R S2 + S2† R U R S1
       + S1† R U R U R S1
```

The order-three S1†RUR S1 vanishes by sector support. The Q6 part of
Sigma4 equals F6†R6F6 with F6=D6P+U65R5V1, retaining both direct/iterated
cross terms. Neither an interacting-Q resummation nor source compression
is introduced.

For the **Q-projector-only** derivative, replace every R in turn by the
existing B_R, with outgoing factors left and incoming factors right.
Hold the outer S/U/W fixed; this does not assert that their physical
currents vanish. Thus

```text
D_j(R W R) = B_R,j W_i R_i + R_f W_f B_R,j
D_j(R U R U R)
  = B_R,j U_i R_i U_i R_i
  + R_f U_f B_R,j U_i R_i
  + R_f U_f R_f U_f B_R,j
```

The last line and the mixed-source terms retain both five-body backflow
legs. An equivalent prescription first projects X=U,W and includes
`(Pi X Pi)'=(K X_i Pi_i+Pi_f X_f K)/(2P+)` along with R'.
Adding this extra vertex also to the **unprojected** word prescription
double counts a demonstrably nonzero term. This is the project's
noncommuting-oscillator derivation, using the general product-rule strategy
of [Kvinikhidze et al., Gauge invariance in the presence of a cutoff](https://arxiv.org/html/0904.4128v1);
that paper's commuting momentum cutoff is not substituted for the present
oscillator projector.

The norm independently replaces each R by +R², since -d_E R=R². It
recovers A6†A6 **and** A1†B5+B5†A1. The local correction is
`N_f^-1/2 [sum_n g_s^n deltaJ_Q,reg,n] N_i^-1/2`.
Exact whitening follows the existing local H2 convention; it contains
higher-order terms beyond the raw g^4 polynomial and is not an additional
regulator-current vertex.

The production implementation reuses exact raw propagation vectors.
Hermitian factors to the left of an insertion are moved to the bra in
**reversed order**. A separate, uncached, direct Leibniz evaluator checks
this optimization. Only exactly zero sector branches may be skipped;
there are no numerical amplitude thresholds or rank truncations.

Independent tests cover:

- full 37-by-37 return Hamiltonians and both norms against separately
  constructed inner H2 models, at two energies and both helicities;
- the independently matched two-energy density/current backflow;
- every interacting word's finite-outer Ward identity and explicit
  projected-versus-unprojected gauging;
- nonzero second-shell terms, separate Q5/Q6 inner cutoffs, and a negative
  test showing the effect of double-counting an internal boundary;
- low-order reduction to the preceding actual Q5 loop current, reversed
  spin/transfer/energy endpoints, and once-only local normalization.

The Ward tests supply the contracted **actual finite-parent** vertices
`H0_f rho-rho H0_i`, `U_f rho-rho U_i` and `W_f rho-rho W_i`.
They do not define the missing transverse parent current by dividing a
continuity defect by transfer, and do not assume projected contacts
commute with density. The identity for a propagator word is not yet the
identity for the entire retained current: its source partners are needed.

The reproducible `benchmark_m2_interacting_q_loop_regulator.py` uses
three deterministic complex retained probes per endpoint (seed 20260907),
not eigenstates or form factors. Its 3-by-3 probe norms must not be called
full 37-by-37 matrix norms. All intermediate Q5/Q6 rows remain present.
It checks both local H/norms against independent inner models and retains
separate RUR, RWR and RURUR order-four contributions.

**Next:** integrate the already-paired Wick Q5 insertion into this same
regulator wrapper, complete P/source projector motion on the same outer
coordinates, then regulate the propagating vertices and Wick contractions
with one field prescription and derive their gauged-contraction and
finite-current partners. No physical calibration/current or covariance
repair is selected, and no new quantum run or public push is claimed.

Pre-edit notes are preserved from `ef933452` in
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_interacting_q_loop.tar`.
The new scientific source files have no overwritten predecessor.

### Executed interacting-Q controls

Scientific source `c8034005d42d21e01fa5c4bbaca08f699bf95033`. The final 38 direct tests pass in 735.00 s; a separate 43-test H2/Q5/resolvent regression passes in 89.77 s. The 1643-test M2 collection is not a full-suite pass. The initial uncached implementation also passed 28 tests (762.03 s) and 8 radial tests (517.35 s) before exact-vector reuse.

[Saved proton/neutron controls](../snapshot/docs/next_level/m2_interacting_q_loop_regulator_results.json) cover both helicities at outer cutoff 2/inner cutoff 0, with561,222 raw Q coordinates and 37 retained coordinates. The largest independent H/norm probe error is 1.182256e-16. Nonzero RUR, RWR and RURUR terms are retained separately. Proton spin-flip order-four x/y probe norms are 3.03980125e-6/2.91876083e-6; neutron values are 3.02148117e-6/2.90257908e-6. These are deterministic three-probe matrix norms, not full operator norms or form factors.

Proton elapsed 932.32 s using the direct evaluator before exact-vector reuse; neutron elapsed 492.89 s using reused production coefficients. This is not a controlled speed comparison. Both local norms are applied; no new cutoff-convergence, covariance or quantum claim is made.

The current theory/status test original is also preserved in `computation_handoff/backups/2026-09-06_pre_canonical_self/pre_interacting_q_loop_theory_contract.tar`.

## Paired Wick and full retained/Q oscillator transport

The paired Wick insertion is now part of the same Q-word engine: W55 is
replaced by W55+S55, with the identical g² S_PP in the retained Hamiltonian.
H0, the free resolvents and the canonical vertex mass are unchanged.
The extra R5 S55 R5 return and its two regulator insertions occur at g⁴;
Q6 one-body returns begin at g⁶ and are not imported. Although the
longitudinal scalar commutes with density/projector motion, its sandwiched
interaction boundary vanishes while the moving-propagator current does not.

The new `m2_h2_projector_current.py` also supplies the outer-retained
source boundary. For exact nested maps p_f,p_i and a return Σ,

```text
B_P,Σ,j = p_f† (K_P,j Σ_i + Σ_f K_P,j) p_i / (2P+).
```

For each word S_l† C S_r this is evaluated as
B_l,i† C_i A_r,i + A_l,f† C_f B_r,f, divided by 2P+,
where A=S p v and B is the same actual source acting on the K_P-displaced
outer-retained vector. The incoming B uses K_P†. Moving C_f to the bra
reverses its factor order. Q-projector motion remains in the existing Q
engine; adding another projected-source Q vertex double counts it.

Source actions contract the retained symmetry isometries with arbitrary
complex probes before forming raw Q images. They retain all four Q sectors,
both cubic channels, coherent IF/IG and sea terms, all seven direct-Q6
components, and the exact Bose/Fermi normalization. Both gluon-helicity
blocks share a single probe axis. No amplitude/rank threshold or dense
Q-by-Pouter source matrix is introduced. Existing basis-column APIs remain.

The actual outer Hamiltonian must reduce to the Q engine's inner one,
including Wick PP. Both source powers and all Q sectors are checked on the
requested columns; normalized calls check the whitened columns. This is
not full-operator certification on untested directions. Independent parent
bilinears supply each return and its negative energy derivative, replacing
every R with +R² and retaining both backflow derivatives.

All three oscillator corrections are summed before the two independent
local norms: P-H, P-return and Q-return. This is a separately callable
cutoff correction, not the complete physical tree-plus-loop four-current.

For current common-width canonical source kernels, total oscillator
occupation cannot decrease from parent to daughters. Consequently
Pi_Q S (1-Pi_P)=0 when every Q inner cutoff is no greater than the P inner
cutoff, making the P-return boundary zero even when raw outer source tails
are nonzero. Same-helicity ground-P motion first needs outer shell two.
Neither selection zero establishes convergence; field-regulated kernels
must test this rule anew.

Pre-edit source and notes from `08a36a93` are preserved in
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_wick_q_integration.tar`;
the four legacy source builders are additionally preserved in
`pre_source_vector_actions.tar` in that directory. No original was removed.

**Continue persistently:** common field regulation of propagating vertices
and Wick contractions, their gauged-contraction/loop/current counterterms,
and independent retained/excluded/longitudinal/field-regulator resolution
controls. Finite physical conditions remain explicit and unselected;
normal-order subtraction, Ward closure and exact probe agreement are not
physical renormalization. No covariance repair, quantum execution or
publication is selected.

### Executed paired-Wick/P/Q controls

Scientific source `aeb8ee1039aef31d71be61aefa05cf8a00c6c65e`. All 87 distinct new tests pass in
separate runs (20+3 Wick-Q, 49 source-vector, 14 P/source and one radial).
Final P/source: 1415.11 s; radial: 708.18 s.
A separate 173-test legacy regression passed in 150.47 s. The 1730-test
M2 collection is not a full-suite pass.

[Saved proton/neutron controls](../snapshot/docs/next_level/m2_h2_projector_current_results.json)
record 561,222 raw Q rows, both energy/norm endpoints and unequal probe
counts (two outgoing, one incoming). Maximum independent inner H/norm
probe error is 1.1145598e-16; maximum P-return Ward residual is
2.0340081e-19. Every P/Q coefficient order and both raw/two-norm
sums are saved, not promoted to full operator norms or form factors.
The radial test uses P outer2, Q outer3, Q5 inner2/Q6 inner1. Its
finite-parent Ward check is not a continuum N+2 free-current closure.

Six current-theory/status checks pass in 0.09 s. The 176-page scratch render
has 496 unique labels and every explicit reference/citation resolves. The new
equations, numerical scope and status chapter pass visual review, with no
rendered ??, final overfull boxes or missing glyphs. The inherited generic
undefined-reference warning is not suppressed. No PDF deliverable or push
was requested.

## Common Gaussian source and Wick kernel

Scientific source `bbbf2420bded99d968a552d34ac55324aa6484f3`. Exact common Gaussian source integrals now
support a boost-invariant active-cluster family, all 13 source kinds on actual
proton/neutron coordinates, and its mode-resolved ordered Wick H coefficients.
The independent field-leg family is a finite-frame diagnostic; its fixed-Jz
source rejects off-comoving use. Existing bare/current snapshots are unchanged.

All 234 combined checks pass in 41.97 s: 185 new regulator/Wick/source checks
and 49 existing source regressions. Independent polar-wave quadrature, boosts,
physical-leg path derivatives, signed Wick sums, exact H subtraction and
source-shell Parseval controls are included. M2 collection 1915 is not a full-suite pass.

**Scientific boundary:** the cluster Wick trace is `I(p,k)=Lambda² k/(4 pi p)`,
not the old constant trace. Its nonzero mode derivatives cannot be omitted
from a gauged contraction. Finite source-norm convergence is not loop/current
UV convergence. Full regulated PP/W55/U65, gauged contractions and both-norm
H2/loop/current assembly remain open. Physical finite conditions, calibration
and covariance repair remain unselected. No new quantum or physical claim is made.

**Continue persistently:** Apply the same active-cluster kernel to retained PP, W55 and U65; gauge physical field legs and ordered contractions before projection; assemble all H2/current words and both norms through g^4. Then complete longitudinal-envelope partners and independent loop/current/resolution controls with explicit finite conditions.
No unresolved scientific blocker requires user direction at this checkpoint.
Original source/notes from `53df1005` are preserved in
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_common_field_regulator.tar`.
No original was removed. GitHub/review mirror remain at the preceding
authorized publication; there is no new push request.

The full derivation is in the living theory note, `sec:common-Gaussian-cluster`.
For participating positive physical modes `w`, define `S=sum(w)`,
`u=sqrt(w/S)` and `B=sqrt(S/2)(I-u u^T)diag(1/sqrt(w))`.
The correlated ordered-field kernel is `F=exp(-Q/(2 Lambda²))` with
`Q=|B p_T|²=[S sum(p_a²/w_a)-|sum(p_a)|²]/2`.
On a momentum-conserving vertex Q is the sum of the incoming/outgoing active
transverse invariant masses. Spectators carry identity and `B w=0`.
This explicit transverse UV family is not a proof of full BRST/covariance
or derivation from a commonly regulated uneliminated Dirac constraint.

The same quartic kernel on `(p,k;p,k)` integrates to `Lambda² k/(4 pi p)`.
Every IF/IG/A4 summand retains its signed denominator and original color/flavor
factor. Quark/antiquark orderings remain distinct; no propagating `k<p`
restriction is imported. Only the exactly-zero direct IG transfer is excluded.
Exact normal ordering removes the actual weighted H operator; finite kinetic
quark/gluon mass² coefficients remain explicit, with vertex mass unchanged.

At equal external plus modes and unequal transverse endpoints,
`T(p,k;delta)=I(p,k) exp[-(p+k)delta²/(4p Lambda²)]`.
Independent differentiation under the fixed-momentum integral verifies its
external/loop mode and transverse derivatives. At zero transfer
`partial_p I=-I/p` and `partial_k I=I/k`. Thus the quark IF partner contains
`-partial_p[I/(k(p-k))]=I/[k(p-k)²]+I/[p k(p-k)]`.
The second term is missed by a naive weighted canonical minus current.
These smooth derivatives do not gauge finite longitudinal-envelope edges.

A physical cluster leg shifts all correlated Gaussian rows along `B[:,leg]`,
not a unit row. Its straight-path derivative satisfies the integral endpoint
identity and is regular at zero transfer. Charges, canonical-polynomial
currents and external normalization remain separate. Neither that Ward
identity nor exact H subtraction is complete regulated current matching.

The independent radial source control uses `X=x+y`, `t=X b²/Lambda²`,
`z=(1-t)/(1+t)` and polynomial `c+f_x eta_x+f_y eta_y`.
The infinite-shell squared norm is
`(xy/X)[|c|² Lambda²/(4 pi X)+(|f_x|²+|f_y|²)Lambda^4/(8 pi X²)]`.
Actual global-Jacobi norms agree with finite analytic radial sums in all
three cubic channels, widths .3/.8/1.7 GeV and shells 0 through 4.
The independent field-leg family can lower inherited oscillator occupation;
the active-cluster source preserves the bare no-lowering selection instead.

Implementation: `m2_gaussian_field_regulator.py`, `m2_gaussian_field_vertices.py`,
`m2_gaussian_field_sources.py` and `m2_gaussian_cluster_wick.py`. Reproduce
checks with the five `tests/test_m2_gaussian_*.py` files plus
`tests/test_m2_source_vectors.py`. The new objects mark full H2 matching
incomplete; existing matched bare/current objects do not silently accept them.

Documentation verification: six theory/status checks pass. The 179-page
scratch render has 507 unique labels and all explicit references/citations
defined. The new kernel/Wick/current/Parseval equations and status page
passed visual review. No rendered ??, final overfull boxes or missing
glyphs were found. The inherited generic undefined-reference warning
remains unsuppressed; no PDF deliverable is requested.


## Common-regulated H2 Hamiltonian, norms and oscillator transport

Scientific source `6af007c4469032cefdee3c526abcfdd53670375d`. The same active-cluster regulator now
appears in retained PP, all W55/U65 interactions, all thirteen source kinds
and the ordered Wick H/subtraction. Full represented H2 returns and both
energy-dependent norms through g⁴ agree with an independent raw-Q Neumann
recurrence. The finite P/source/Q oscillator-current correction binds
these actual regulated H/source definitions without adding Wick twice.
Complete physical field-current matching remains unfinished.

All 149 distinct new checks pass in separate runs: 91 exact interaction
orbitals, 37 full-H2 controls and 21 Q/P-source current checks.
The Q-current portion passed 12 tests in 202.68 s before an invalid
equal-inner/outer test setup; the corrected outer-shell P/source portion
passed nine in 981.44 s. The cutoff guard was not relaxed.
H2 passed as 32 tests in 235.91 s and five in 92.98 s.
The preceding 234 source/Wick checks pass again (44.85 s);
20 unchanged-default interaction checks pass (27.99 s), and nine selected
H2/Q regressions pass (118.36 s; one overlaps a new H2 validation test).
The focused M2 collection is 2064, not a full-suite pass.
An unscoped repository collection was interrupted after 63 historical
collection errors, including C165 package-root mismatches. Those unrelated
historical gates were not repaired or used to block this scientific work.

[Seven saved controls](../snapshot/docs/next_level/m2_gaussian_cluster_h2_results.json)
vary Gaussian width, Q5/Q6 shells and the field envelope with explicitly
uncalibrated finite terms. At Lambda=.8 GeV, Q5 shell1→2 with Q6=1 changes
the self-energy Frobenius norm from .005729744 to .007259288 GeV².
This is cutoff dependence, not established loop/current convergence.
The report uses full retained matrix norms, not form factors or selected states.

**Continue persistently:** Derive physical-leg and canonical-polynomial currents from the same ordered cluster kernel, including the four-leg pair-current extension and gauged Wick subtraction. Complete finite longitudinal-envelope partners, common H2 loop/current words with both norms, and independent resolution/finite-condition controls. Do not weight old bare currents or gauge auxiliary factorization legs by assumption.
The auxiliary spin-zero color-adjoint current coordinate is not a physical
gluon. Its conserved Hamiltonian factorization does not define the
off-conservation four-leg electromagnetic kernel. Free states beyond an
interaction envelope are not deleted. The two represented sea flavors
must agree with the Wick trace. Physical finite conditions, calibration,
covariance repair, quantum execution and publication remain unselected.
No unresolved scientific blocker requires user direction at this checkpoint.

Original source and notes from `0f583ca9` are preserved in
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_common_cluster_H2.tar` (verified Git archive commit). No original was removed.
GitHub/review mirror remain at the preceding authorized publication;
there is no new push request.

The full derivation is in the living theory note, `sec:common-cluster-H2`.
On the conserved contact slice, t=X b²/Lambda² and z=(1-t)/(1+t);
each m=0 radial endpoint contributes z^n/(1+t).
Independent direct Gaussian integration checks rectangular contacts and boosts.
Retained emission and all five U65 channels use the same physical-leg kernel.

In full raw Q5+Q6, S1=(V1,0), S2=(V2,D6P), R=(E-H0Q)^-1,
U=U65+U56 and W=W55+paired Wick/finite one-body. Independently,
A1=R S1, A2=R(S2+U A1), A3=R(U A2+W A1).
H_eff=H_PP+g² S1†A1+g³(S1†A2+S2†A1)+g⁴(S1†A3+S2†A2).
N=I+g² A1†A1+g³(A1†A2+h.c.)+g⁴(A2†A2+A1†A3+h.c.).
N agrees with I-partial_E H_eff, including both backflow derivatives.
Q6 one-body insertions start at g⁶. No source-rank projection, identification
of kinetic and vertex masses, or isolated higher-order subset is introduced.

The explicitly named plus diagnostic obeys the zero-transfer two-energy
identity q[I-(H_eff(Ef)-H_eff(Ei))/(Ef-Ei)], reducing to qN and then qI
after the two norms at equal energy. It is not a complete physical current.
The P/source correction uses regulated outer-P actions and checks the
actual columns against the Q engine, including after whitening.
Different field widths, Wick conditions, P+ or bare/regulated mixtures
are not the same finite model even when matrix dimensions agree.

Reproduce the seven uncalibrated controls with
`scripts/benchmark_m2_gaussian_cluster_h2.py --output <new-result.json>`.
Fixed test values: K2=9, P shell0, b=.4 GeV, mq=.35 GeV, mg=0, g=.3,
P+=1.3 GeV, E=1.4 GeV², two flavors, normal ordering and finite
mass²/g²=(.011,-.007) GeV². No data were fit.
Proton shell1 self-energy norms at Lambda=.4/.8/1.6 GeV are
.003052468/.005729744/.007095715 GeV². These are complete
37-dimensional retained matrix norms, not eigenstate observables.
Q5=2/Q6=1 retains 230382 raw Q coordinates. Narrowing the field envelope
changes interactions without deleting free states.
The largest H anti-Hermitian Frobenius norm is below 4.176e-16 GeV².

For the next gauged-contraction derivation, retain the electromagnetic
charge routing of both external and contracted fermion legs before taking
the Wick trace. External-mode differentiation alone must not be assumed to
supply the whole current, and a transverse Ward identity does not determine
J-minus. The physical four-leg kernel, not its conserved auxiliary
factorization, owns these derivatives.

Six theory/status checks pass (0.09 s; final repeat 0.07 s). The 181-page
scratch render has 514 unique labels, with all explicit references and
citations defined. The new Hamiltonian/norm equations, numerical scope
and current-status page passed visual review. No rendered ??, final
overfull boxes or missing glyphs were found. The inherited generic
undefined-reference warning is unsuppressed. The PDF was used for
verification, not as a deliverable.


## Physical-leg common-kernel current: canonical spin plus endpoint product

Scientific source `3e29cfe01d5dedab4c346f4e07c3c48a59a84ed9` derives the explicit physical-leg
endpoint-product current from the common ordered Gaussian kernel.
All nine retained/Q5/Q6 cubic route types and unfactorized four-leg
pair-contact orbitals retain canonical spin components and physical recoil.
The retained cubic action is assembled with all emitters, colors, spin
flips and flavor charges on actual proton/neutron spin endpoints.

All 99 new tests pass together in 138.13 s. Independent polar-wave
integration and field derivatives check every current component.
Full shifted Dirac Ward endpoints, bare ladder-moment minus limits,
kinematic boosts, Hermitian reversal, permutation symmetry and charge
linearity pass. All 276 surrounding Gaussian interaction/source/Wick
regressions pass in 36.93 s. The 2163-test M2 collection is not a
full-repository pass.

The four-leg check initially assumed that every charged projected channel
must be nonzero. Equal-mode outgoing q/qbar in its ground relative orbital
instead cancels exactly by charge antisymmetry. The test now records that
selection rule; the Ward and Hermitian tests were already passing.
Nothing was fitted or made nonzero by altering the physics.

**Next, under the active persistent goal:** Gauge and Wick-contract the original physical four-leg monomials, retaining external and internal charged-fermion routing, and pair their current subtraction with the same Hamiltonian subtraction. Bind all physical P/Q5/Q6 contact/source actions, then common H2 loop/current words through g^4 with both norms. Complete finite longitudinal-envelope partners and independent resolution/finite-condition controls. Keep the explicit endpoint-product representative separate from physical finite matching; do not gauge auxiliary factorization legs.
The complete gauged Wick subtraction, full physical Q5/Q6 current action
assembly, longitudinal-envelope currents and common loop/finite matching
remain unfinished. No physical finite conditions, dynamical covariance
repair, quantum execution or public push are selected. This is not an
unresolved blocker requiring user direction.

Original source and notes at `ed332cb6` are preserved in
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_cluster_field_currents.tar` (Git archive commit verified). No original was removed.
GitHub and the public review mirror remain at the preceding authorized
publication; this research checkpoint has not been pushed.

The derivation is in `sec:common-cluster-field-current` of the living note.
For physical leg a use sigma=+1 incoming, -1 outgoing, and its physical
charge e. The base current kinematics obey sum sigma*pT=-DeltaT.
V_a and F_a are the bare M² polynomial and common Gaussian after shifting
that one physical leg by sigma_a*DeltaT. The named prescription is

`J_Lambda^mu = F0 J_can^mu + sum_a e_a V_a K_a^mu/(2P+)`,
`K_aT = integral_0^1 dt partial_(p_aT) F(p_a+sigma_a t DeltaT)`,
`K_a^- = -integral_0^1 dt partial_(p_a+) F(p_a+sigma_a t DeltaT)`,
`K_a^+ = 0`.

The ENDPOINT V_a stays fixed along the path. It is not V(t).
Canonical J_can includes charges and discrete gluon factors and already
uses J/(2P+). DeltaT*K_aT=sigma_a(F_a-F0), hence the full physical-leg
Ward endpoint sum follows from the separately checked canonical Dirac
identity. No inverse DeltaT² or transverse Ward definition of J-minus
is introduced. The bare limit recovers the original canonical spin current
in every component, including all independent ladder-moment minus kernels.

For G=B.T B, S=sum w and fixed K2/P+:
`partial_(p_aT) F = -Lambda^-2 (G[a,:] pT) F`;
`-partial_(p_a+) F = K2/(4 P+ Lambda²)
[sum_b p_bT²/w_b - S p_aT²/w_a²] F`.
An affine cubic V_a produces a cubic minus insertion; exact joint Gaussian
moments precede projection. Physical plus derivatives hold longitudinal
envelope membership fixed, not gauge that envelope by assumption.

The four-leg orbital supports actual retained/five/six-body pair geometry
with two independent relative coordinates and full shared-vertex recoil.
Its canonical coefficient is supplied per ordered monomial; actual
retained qg direct/crossed H and J-minus coefficients are included and
checked. This is not yet every raw quartic/source action or a Wick trace.
An auxiliary spin-zero color-adjoint current coordinate is not a gluon.

Reproduce the new checks with the three
`tests/test_m2_gaussian_cluster_current*.py` / pair-current test files:
- `tests/test_m2_gaussian_cluster_current.py` (54);
- `tests/test_m2_gaussian_cluster_pair_current.py` (33);
- `tests/test_m2_gaussian_cluster_current_actions.py` (12).

The 276-test regression covers Gaussian field regulator/current/sources,
common cluster regulator/Wick and all common-cluster interaction orbitals.
No complete repository or loop/current convergence result is asserted.

Documentation verification: six theory/status checks pass. The 184-page
scratch render has 521 unique source labels, 105 explicit reference targets
and 79 cited entries, all defined. The new equations and their continuation,
front summary and current-status pages passed visual review. No rendered ??,
final overfull boxes or missing glyphs remain. Generic inherited reference
and BibTeX rerun notices remain unsuppressed. No warning-free or full-page
audit is claimed; the render is not a requested PDF deliverable.


## Smooth gauged Wick current and its paired H2 insertion

Scientific source `314bca0ed1d6e8ab870fbeb42182a2303e931c88` derives the smooth connected Wick currents
of the same ordered four-physical-leg Gaussian kernel. External and internal
charged attachments are integrated before contraction, with distinct finite-
transfer paths. Canonical, regulator, ordered, subtraction and finite pieces
remain independently inspectable. Quark/antiquark signed denominators and
each gluon-loop flavor are retained; a gluon is not assigned a bare charge.

The paired current acts on retained P and every supplied raw Q5/Q6
orbital/color/spin coordinate. Its direct H2 contribution is g² W_PP plus
g⁴ A1_f† W55 A1_i, with both existing energy/spin/norm endpoints.
The paired H and third-order backflow are already in H2 and are not added
again. Q6 one-body insertions first return at g⁶.

All 77 new checks and 276 surrounding Gaussian regressions pass together:
353 tests in 128.83 s. Independent original-leg quadrature, unreordered
fermion algebra and full orbital/color/spin tensors test the formulas.
The 2240-test M2 collection is not a full-suite run.

Seven explicitly uncalibrated controls retain exact ordered/subtraction
cancellation. At Lambda=.4/.8/1.6 GeV, the proton normalized direct
unsubtracted Wick-current matrix norms are .025435002/.110063738/.452907974.
At Lambda=.8, increasing Q5 from shell one to two changes this matrix by
7.371205e-6 in Frobenius norm and retains 230382 raw Q coordinates.
These are one contribution to J-minus, not form factors or evidence of
complete loop/current convergence. The subtracted finite contribution still
depends on H2 waves/norms; exact contraction cancellation does not fit its
finite parameters or establish physical renormalization.

**Next, under the active persistent goal:** Bind the remaining physical P/Q5/Q6 contact/source current actions to the same common Gaussian kernel, then assemble all H2 loop/current words through g^4 with both norms and the now-derived smooth Wick subtraction. Complete finite longitudinal-envelope partners and independent resolution/finite-condition controls. Keep physical finite matching distinct from the explicit endpoint-product representative; do not gauge auxiliary factorization legs or add the Wick Hamiltonian twice.
Full gauged contractions remain incomplete because the finite longitudinal
envelope has not been gauged. Other Q contact/source actions and the
complete common-loop current also remain unfinished. No unresolved physics
blocker requiring user direction has been found at this checkpoint.
No physical fit, covariance repair, quantum run or push is selected.

Original source and notes at `03f3c33c` are preserved in
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_gauged_cluster_wick.tar`
(Git archive commit verified). No original was removed. The public review
mirror and GitHub remain at the preceding authorized publication.

The complete formulas and charge table are in
`sec:common-cluster-gauged-Wick` of the living theory note.
Implementation: `src/deuteron_wigner/quantum/m2_gaussian_cluster_wick_current.py`.
Reproduce the 77 direct checks with the two
`tests/test_m2_gaussian_cluster_wick_current.py` and
`tests/test_m2_gaussian_cluster_wick_projection.py` files.
The first has 64 contraction/path checks; the second has 13 full projection
and H2 checks. All passed in the combined 353-test run.
The preceding 99 physical-leg current tests also pass again in 147.36 s.

Reproduce the seven controls with
`scripts/benchmark_m2_gaussian_cluster_wick_current.py --output <new-result.json>`.
The saved result is `docs/next_level/m2_gaussian_cluster_wick_current_results.json`.
K2=9, retained shell0, b=.4 GeV, mq=.35 GeV, mg=0, g=.3, P+=1.3 GeV,
DeltaT=(.12,.16) GeV, Ef=1.39/Ei=1.47 GeV², two explicit flavors and
normal ordering are declared. Finite mass²/g²=(.011,-.007) GeV² are
uncalibrated test values, not physical inputs. No eigenstate is selected.

Documentation verification: six theory/status tests pass. The 186-page
scratch render has 528 unique labels, 106 explicit reference targets and
79 cited entries, all defined. The new equations/charge table, H2 and
sensitivity continuation, front summary and current-status pages passed
visual review. No rendered ??, final overfull boxes or missing glyphs
remain. The inherited generic reference/BibTeX rerun notices are
unsuppressed; no warning-free or full-page audit is claimed. The render
is verification material, not a requested PDF deliverable.


## Complete common-regulated cubic actions and H2 current words

Scientific source `110fd1869b75adc4fde582541870ccc0f91b141b` completes the represented common-regulated
cubic contact-current actions from P to Q5 and Q5 to Q6, and every cubic
H2 return word through g⁴. Canonical/regulator parts remain separate;
all raw color/orbital/spin coordinates, both flavors, normalized creation
statistics and the antiquark charge sign are retained.

The order-g retained current is supplemented by both P↔Q5 endpoints
against A1, A2 and C3 at orders g², g³ and g⁴, and both A6† J65 A1
returns at g⁴. Both existing H2 energy/spin/norm endpoints are used.
Independent full-wave polynomial coefficient extraction verifies every
retained power and excludes its nonzero g⁵/g⁶ terms.
This is the complete cubic contribution, not the full four-current.

All 24 distinct new checks pass in separate runs: eight bare limits
in 15.83 s, eleven regulated/raw-Ward checks in 214.75 s, and five
H2/boost/rectangular-resolution checks in 398.37 s.
All 113 Wick and existing excluded-current regressions pass in 114.41 s.
The 2264-test M2 collection is not a full-suite run or physical validation.

**Next, under the active persistent goal:** Derive and project all remaining regulated quartic PP, Q5Q5, P-to-Q5 and P-to-Q6 currents, including IF/IG/A4 regulator partners and physical 1-to-3 recoil. A zero canonical contact does not imply a zero regulator current. Then combine free, cubic, quartic, Wick and projector pieces in the same H2 loop/current words through g^4 with both norms. Determine the finite longitudinal-envelope partner at the declared transfer from its explicit projector, and complete independent resolution/finite-condition controls. Do not infer J-minus from a transverse Ward identity, gauge auxiliary factorization legs, or add Wick H twice.
Quartic current actions, longitudinal-envelope and complete loop/finite
matching remain open. There is no unresolved blocker requiring user
direction at this checkpoint. No physical fit, covariance repair,
quantum execution or public push is selected.

The source and notes immediately before this change are preserved at
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_cluster_Q_cubic_currents.tar`,
a verified Git archive of `aba2e67a`. No original was removed.
The public review mirror and GitHub remain at the preceding authorized
publication, not this local scientific checkpoint.

Implementation: `src/deuteron_wigner/quantum/m2_gaussian_cluster_excluded_current.py`.
The two new test files are
`tests/test_m2_gaussian_cluster_excluded_current.py` (19) and
`tests/test_m2_gaussian_cluster_cubic_h2_current.py` (5).
The first separately tests quark, antiquark and pair-creation channels;
the second recomputes the Q-wave recurrence and extracts the complete
degree-six formal wave-current polynomial by discrete Fourier coefficients.
Only powers one through four are retained in the H2 cubic current.
The same-transfer adjoint is tested separately from physical
reversed-transfer absorption. Raw rectangular action agrees with the
ground-column restriction of the independently enlarged incoming space.
There is no source-rank or amplitude pruning and no giant Q6-by-Q5 matrix.
The purely gluonic cubic electromagnetic/flavor contact is an exact zero,
not a removed charged channel. The canonical plus cubic contact is also
zero; the free plus current and loop density are different contributions.

Documentation verification: six theory/status checks pass. The 187-page
scratch render has 529 unique labels, 107 explicit reference targets and
79 cited entries, all defined. The new cubic/H2 equations and continuation,
front summary and status pages passed visual review. No rendered ??, final
overfull boxes or missing glyphs remain. Inherited generic reference/BibTeX
rerun notices are unsuppressed; no warning-free or full-page audit is
claimed. The render is verification material, not a requested deliverable.

## Complete physical quartic source currents and H2 returns

Scientific source `981a0e36ec7050e979f0e89eafd173419cc85979` builds all ten represented quartic
P-to-Q5/Q6 source-current families from the same four-physical-leg Gaussian
kernel. The physical 1-to-3 recoil, unit-kernel factorization, separate
valence/created-pair charges, signed IF denominators, normalized Fock
factors and H2 words are derived in theory
`sec:common-cluster-quartic-source-current` and
`eq:cluster-quartic-source-H2-words`.

The three Q5 families are valence IF, IG and sea emission. The seven Q6
families add gluon A4, IF pair, IG pair and IG three-gluon emission.
Canonical source contacts occur only for valence IF and gluon IF pair.
Other charged families retain regulator currents; pure-gluon currents
are exactly zero while their Hamiltonians remain in the waves.

Because every ordering within a family has the same physical fields and a
transverse-constant H, its ordered color sum commutes with the scalar orbital
integration. This is an exact linear factorization before projection, not
weighting an old truncated current or gauging an auxiliary gluon.
The minus current retains independent squared constrained denominators.

All P-to-Q5 source-current returns enter at g3 against A1 and at g4
against A2. All P-to-Q6 source-current returns enter at g4 against A6.
Both endpoints, including reversed-transfer absorption, use the actual
common-H2 energies/spins and both norms. C3 times the quartic source
current first enters at g5 and is explicitly excluded.

Direct tests:

- 56 orbital checks cover all ten source families, independently integrated
  shifted H Ward endpoints, scalar four-field minus/transverse derivatives,
  old canonical bare limits, rectangular shells, physical spectator recoil,
  finite field-envelope membership, both boost laws and path refinement.
- 21 raw-action checks cover all ten families, direct spin/color-slot Ward
  enumeration, both resolved flavors, Bose/Fermi statistics, proton/neutron
  bare limits and nonzero opposite-Jz shell-one IG actions.
- Four H2 checks compare independent Neumann-wave polynomial coefficients,
  nonzero proton/neutron order-five remainders, reverse endpoints, both
  norms and definition guards. Ground-orbital opposite-Jz zeros are tested
  as selection rules, not advertised as nonzero matching controls.

The first two groups pass together (77 in 151.97 s); the H2 group passes
separately (four in 139.69 s). Another 219-test run passes in 131.95 s:
56 new orbital checks plus 163 prior physical-current/Wick regressions.
The final typed-route endpoint diagnostic passes after an explicit
sector-before-route validation correction. M2 collection: 2345, not a
full-suite pass.

The equal-mode outgoing q/antiquark ground-orbital regulator current
cancels by charge-odd exchange symmetry. Its shell-one counterpart and
unequal-mode source currents are nonzero. Separately, at K2=9 the valence
IG six-body H kernel contains k1-k2=0 because all three gluons have mode2.
This is not an absent-current approximation; unequal-mode K11 orbitals
retain the IG term.

Implementation:
`src/deuteron_wigner/quantum/m2_gaussian_cluster_quartic_source_current.py`.
Tests: `tests/test_m2_gaussian_cluster_quartic_source_current.py`,
`tests/test_m2_gaussian_cluster_quartic_source_actions.py`,
`tests/test_m2_gaussian_cluster_quartic_source_h2_current.py`.

**Next:** Derive and project the remaining regulated quartic PP and Q5Q5 physical pair currents, including every IF/IG/A4 ordering and gg-to-quark-antiquark conversion. Retain actual flavor charges, signed constrained denominators and four physical field legs; do not gauge auxiliary factorization legs. Combine these with the completed cubic, quartic-source and Wick contributions, free currents and oscillator-projector transport in all H2 loop/current words through g^4 with both norms. Determine the finite longitudinal-envelope partner at the declared transfer from its explicit projector, then perform independent resolution and explicit finite-condition controls. Do not infer J-minus from a transverse Ward identity or add Wick H twice.
This is the complete normal-ordered quartic source contribution, not the
missing quartic PP/Q5Q5 pair-current action or full matched four-current.
No physical finite calibration, convergence, covariance repair, quantum
execution or publication is inferred. Backup: `computation_handoff/backups/2026-09-06_pre_canonical_self/pre_cluster_quartic_source_currents.tar`, from
`bdb7fd62`. Full persistent goal remains active.

## Physical-pair continuation: physical quartic pair currents and H2 returns

Scientific source `c6c0d63b8d0b9237aed9b96320692712741b5569` implements all
normal-ordered quartic PP/Q5Q5 pair currents on the four physical fields.
The six charged components include number scattering, both constrained
orders, IF/IG conversion and all-four-quark annihilation/recreation.
Pure-gluon currents are zero without deleting their Hamiltonians.
Annihilation helicities are accumulated coherently using injective spin maps.

The physical Hamiltonians reproduce the regulated retained pilot and actual
H2 leading Q5 waves at N5=0,1, including IF/IG conversion. Arbitrary raw
isospin columns need not equal the auxiliary I=1/2-restricted pair operator;
a nonzero domain witness is preserved. No auxiliary gluon is gauged.
Independent raw-slot shifted-H Ward references check every charged component.
Bare limits, charge linearity, boosts, statistics and adjoints pass.

The pair numerator is g2 J_PP + g4 A1_f† J55 A1_i, with both existing norms.
Separate quartic sources supply g3 and g4. Independent degree-eight
full-wave polynomial extraction verifies pair powers and excludes later
terms. Ground-orbital spin-selection zeros are distinguished from a
nonzero shell-one H2 spin-transition regulator current.

All 25 distinct new tests pass in separate runs: 19 action tests in
335.48 s, five H2/rectangular/guard tests in 423.41 s, and one higher-shell
H2 test with two strengthened Hamiltonian reruns in 570.96 s.
All 163 surrounding current/Wick regressions pass in 136.50 s.
M2 collection: 2370 in 5.46 s, not a full-suite execution.

**Next, under the active persistent goal:** Assemble the common-regulated free, cubic, quartic-source, quartic-pair and Wick currents with the existing P/Q oscillator-projector transport in all H2 loop/current words through g^4, with both energies, spin endpoints and norms. Verify the combined finite-parent Ward identity and independent wave recurrence. Determine the finite longitudinal-envelope partner at the declared transfer from its explicit projector, then run independent resolution and explicit finite-condition controls. Do not gauge auxiliary legs or add Wick H twice.
No unresolved blocker requires user direction at this checkpoint.
The full matched four-current remains unavailable. No physical condition,
covariance repair, quantum execution or public push is selected.

Backup: `computation_handoff/backups/2026-09-06_pre_canonical_self/pre_cluster_quartic_pair_currents.tar`,
a verified archive of `27195d75f7364029bc71a99ddde89241e56e50ba`
(2,519,040 bytes). No original was removed. The public mirror and GitHub
remain at the preceding authorized publication, not this local checkpoint.

Implementation: `src/deuteron_wigner/quantum/m2_gaussian_cluster_quartic_pair_actions.py`.
Direct tests: `tests/test_m2_gaussian_cluster_quartic_pair_actions.py`
and `tests/test_m2_gaussian_cluster_quartic_pair_h2_current.py`.
Theory: `sec:common-cluster-quartic-pair-current` and
`eq:cluster-all-quartic-H2-current`.

## Combined-current continuation: exact Gaussian projector products

Scientific source `ca8b914610dfc776f163cf28d63c7e522e7f473b` combines the common-regulated free,
cubic, quartic-source, quartic-pair and paired Wick current numerators
through g4 on the actual H2 waves and both norm maps. Individual families
and ordered/subtraction/finite Wick diagnostics remain inspectable.
The stable total pairs Wick coefficients before projection. Summing huge
separately projected diagnostics had left a spurious 0.046875 bare-limit
residue at Lambda=1e7 GeV; constituent-level pairing removes that error
without loosening the independent old-current limit test.

The finite-parent projector is now bound to the actual endpoint Hamiltonians
and metrics, not just dimensions. It is not enough for a Gaussian interaction:
the first outer shell can give an exact selection zero while later even
shells contain a nonzero boundary. Exact shifted-H Gaussian integrals now
evaluate the continuum products around the finite-rank projector commutator.
No arbitrary outer cutoff, inverse-transfer repair, norm fit or SVD is used.
The method covers constant pair/source and affine cubic orbitals, and agrees
with the independent N+2 free quadratic construction.

All nine cubic and ten quartic source route types pass independent Hamiltonian,
physical-current Ward, reversal and finite-projector tests. Charged spectator
recoil is included: a pure-gluon Hamiltonian can have a nonzero oscillator
boundary even when its direct electromagnetic field current is exactly zero.
The exact kernel is already projected onto the actual full retained P space.
Its free/cubic/quartic Ward identity closes degreewise through g2 for proton
and neutron, same- and opposite-spin endpoints (absolute tolerance 2e-14).
This is not yet an exact Q5/Q6 projector or a complete H2 four-current.

All 50 distinct new tests pass in separate runs: assembly 5 in 153.26 s;
finite-parent binding 3 in 286.23 s; exact pair/free kernel 18 in 5.31 s;
all cubic/quartic source orbitals 19 in 19.53 s; full retained action 5 in
75.09 s. All 163 surrounding current/Wick regressions pass in 136.40 s.
M2 collection: 2420 in 6.09 s, not a full-suite run.

The standalone combined-current Ward diagnostic uses K2=9, P/Q5/Q6 inner
shell0, outer shell1, proton Jz=1/2 at both ends, b=.4 GeV, mq=.35 GeV,
mg=0, P+=1.3 GeV, g=.3, Lambda=.8 GeV, eight boson and fermion field modes,
normal-order subtraction, finite mu_q²=.011 and mu_g²=-.007 GeV² per g²,
Ef=1.39 and Ei=1.47 GeV², mean PT=0, Delta=(.12,.16) GeV and normalized
independent complex probe columns (seeds321/987). It compares an independent
Hamiltonian recurrence against every current coefficient before the norm maps.
Its Ward right-hand side includes (Ef-Ei)(rho_eff-rho_P); omitting that
off-shell term would be an incorrect matching target.

| Coupling power | Physical + finite-parent residual | Physical + exact PP only |
|---|---:|---:|
| 0 | 2.538e-17 | 2.526e-17 |
| 1 | 7.179e-4 | 4.379e-18 |
| 2 | 2.102e-4 | 1.631e-5 |
| 3 | 1.804e-6 | 1.804e-6 |
| 4 | 1.657e-6 | 1.657e-6 |

These are absolute scalar Ward residuals in the declared normalization,
not a full-matrix H2 pass or a convergence claim. Exact PP removes the
first-order defect and the retained part of second order; Q-loop terms
remain unresolved. The first-shell finite-parent correction vanishes by
selection, not because its infinite tail vanishes.

**Next, under the active persistent goal:** Carry the exact shifted-Gaussian projector boundary through every actual Q5/Q6 free, cubic and quartic Hamiltonian block, including charged spectator attachments to neutral-gluon monomials, then pull it through the same H2 waves through g^4 and both norms. Check the full two-energy matrix Ward identity independently. Derive the minus-projector and finite longitudinal-envelope partners independently; a transverse Ward identity does not determine them. Then complete independent resolution and explicit finite-condition controls. Do not add Wick H twice, gauge auxiliary legs, fit a conservation defect or select physical calibration by default.
No unresolved scientific blocker requires user direction at this checkpoint.
No physical finite condition, covariance repair, quantum execution or push.

Backup: `computation_handoff/backups/2026-09-06_pre_canonical_self/pre_combined_cluster_H2_current.tar`,
verified Git archive of `30a62a17b65ffc560c11fe9c03607953c85e1b83`
(2,560,000 bytes). No original was removed. The public mirror remains
at the preceding authorized publication, not this local checkpoint.

Implementation: `m2_gaussian_cluster_h2_current.py`,
`m2_gaussian_cluster_projector_kernel.py` and
`m2_gaussian_cluster_retained_projector.py` under
`src/deuteron_wigner/quantum/`. Reproduce the full residual diagnostic with
`scripts/benchmark_m2_gaussian_cluster_h2_current.py`.
The living theory derives `eq:cluster-exact-projector-boundary`
and `eq:cluster-two-energy-Ward-target`.

## Exact Q-sector and internal-projector continuation

Scientific source `7e615edc7bc150db81bda4d6b2138e9533294806` supplies exact free Q5/Q6 oscillator
boundaries, every cubic P-to-Q5 and Q5-to-Q6 boundary, and their complete
H2 return words through g4. The same physical Hamiltonian and analytic
Gaussian products are used; there is no finite outer-H tail approximation.
Both created-pair flavors, all raw coordinates and normalized statistics
remain resolved. Pure-gluon H has charged-spectator projector attachments.
Its ground/shell-one same-spin zero is not a generic absence: the shell-one
opposite-spin action is nonzero and passes the independent Hamiltonian test.

The first raw P-to-Q5 Ward tests exposed a second projector, not numerical
error. The actual retained basis restricts the three quarks to I=1/2,
whereas the raw Q spaces contain additional isospin. Orbital correction
alone left full-matrix residual norms .0260658412 (qqqgg) and .0618800540
(qqqq-qbar) at K9, inner shell0. These residuals vanish against actual
A1/A2/C3 source-wave bras at roundoff but must not be advertised as zero
on every raw column.

The internal projector is I=1-(S01+S02+S12)/3, with S exchanging flavors
only, not helicities or physical momenta. Its charge commutator contains
S_ab(e_a-e_b)(D_a-D_b). A straight interpolation between the two physical
quark recoil displacements gives D_a-D_b without an inverse-transfer
repair. The off-diagonal isospin tangent is then inserted through the SAME
physical cubic H. This closes the unreduced P-to-Q5 Ward tests without
changing their tolerance, removing raw states, or changing the Hamiltonian.
The baryon-number tangent is exactly zero; EM and flavor tangents remain
explicit. The isospin current is a finite-projector representative, not a
physical finite calibration or a new electromagnetic charge.

All 34 new tests pass: 31 raw/free/cubic/isospin checks together in 356.66 s,
and three H2 matrix/polynomial checks in 360.36 s. Independent flavor-swap
algebra, exact displacement endpoints, reversal, path refinement, raw H
equivalence, full source matrices, both statistics and same-transfer
adjoints are checked. The H2 test reconstructs H coefficients from an
independent Neumann wave recurrence, not the current assembler.
M2 collection: 2454 in 6.15 s, not a full-suite execution.

All 42 preceding exact Gaussian/projector regressions pass in 90.70 s.
The updated standalone benchmark also completes: its scalar residuals at
powers0..4 after exact PP/free-Q/cubic-Q corrections are
2.526e-17, 4.379e-18, 3.905e-18, 1.183e-6 and 1.395e-7.
It preserves the earlier finite-parent and exact-PP-only controls.

At the preceding explicit K9 inner-shell-zero control, after exact PP,
free-Q and cubic-Q corrections, Frobenius norms of the full retained-matrix
two-energy Ward residuals are:

| Power | Proton same spin | Neutron opposite spin |
|---|---:|---:|
| 0 | 8.575e-16 | 0 |
| 1 | 3.094e-16 | 1.240e-16 |
| 2 | 1.388e-16 | 6.605e-18 |
| 3 | 4.955e-5 | 4.152e-19 |
| 4 | 7.226e-6 | 6.888e-19 |

Ef=1.39, Ei=1.47 GeV²; Delta=(.12,.16) GeV; mean PT=0;
Lambda=.8 GeV, b=.4 GeV, P+=1.3 GeV, mq=.35 GeV, mg=0, g=.3,
eight boson/fermion modes, paired normal-order subtraction and explicit
finite mu_q²=.011/mu_g²=-.007 GeV² per g². The neutron transition is
Jz=+1/2 to -1/2; the proton has Jz=+1/2 at both endpoints.
These are matrix norms, not the earlier scalar-probe residuals.
Both norm maps preserve the g0/g1/g2 closure. The neutron higher-order
selection zero is NOT evidence that the still-missing quartic Q-projector
words can be omitted; the proton has a nonzero witness.

**Next, under the active persistent goal:** Derive exact Gaussian projector currents for all quartic P-to-Q5/Q6 sources and Q5Q5 Hamiltonian blocks, retaining charged spectators, both pair flavors and the actual internal-projector domains. Pull them through the same H2 waves and both norms through g4, and close the full two-energy matrix Ward identity including the nonzero proton third/fourth-order defects. Then derive the independent minus-projector and finite longitudinal-envelope partners and complete resolution/explicit finite-condition controls. Do not infer J-minus from transverse continuity, gauge auxiliary legs, add Wick H twice or choose physical calibration by default.
No unresolved blocker requires user direction. The complete four-current
remains unavailable. No physical finite condition, covariance repair,
quantum execution or public push is selected.

Backup: `computation_handoff/backups/2026-09-06_pre_canonical_self/pre_exact_Q_projector_currents.tar`,
verified Git archive of `ea133bf1e97e3d8d3c58c32a3d1e6c373324fe0b`
(2,621,440 bytes). No original was removed. The public mirror remains at
the preceding authorized publication, not this local checkpoint.

Implementation: `m2_gaussian_cluster_Q_projector.py` and
`m2_isospin_projector_current.py` under `src/deuteron_wigner/quantum/`.
Theory: `sec:exact-Q-projector-current`.

## Exact quartic Q projector and order-four transverse closure

Scientific source `17a4998af53ff7cb591d7f98a735720f298a6ec4` supplies all ten quartic P-to-Q5/Q6
projector sources and all nine physical Q5Q5 quartic boundaries.
The source action includes the retained isospin tangent for both qqq and
qqqg parents. All charged spectators, created-pair flavors, physical color
kernels and Fock factors remain explicit. The Hamiltonian is unchanged.

The new H2 words are A1f† B5P + BP5 A1i at g3, and at g4:
A2f† B5P + BP5 A2i + A6f† B6P + BP6 A6i + A1f† B55 A1i.
Both original norms act after common-order selection. An independent
degree-eight full-wave polynomial verifies these powers and exhibits
nonzero higher-order terms which are deliberately excluded.

The physical Q5 pair-annihilation extension matches the original auxiliary
Hamiltonian on actual A1 waves. It is NOT the auxiliary I=1/2 restriction
on arbitrary raw isospin columns. Raw Ward tests use the physical four-leg
H at both endpoints; the auxiliary coordinate is never gauged as a gluon.

The first nonzero neutral-pair test at shell1 failed its nonzero assertion,
not its Ward or adjoint identity. All three neutral pair Hamiltonians are
nonzero there, but the projector current vanishes: a constant pair contact
changes relative radial excitation by two, whereas a charged spectator
translation leaves that pair coordinate unchanged. Separate shell1 zero
controls and shell2 nonzero controls now pass. This differs from neutral
number-changing A4 emission, whose opposite-spin shell1 boundary is nonzero.
A selection zero is not an absent interaction or cutoff convergence.

All 34 new tests pass across separate runs: 15 source tests together with
seven existing isospin regressions in 84.42 s; 15 pair tests in selections
of eight (463.88 s) and seven (103.83 s); two shell0 full-H2 matrix tests
in 320.92 s; the independent wave polynomial in 68.56 s; and the Q-shell1
full-H2 matrix test in 1289.72 s. Six preceding raw P-to-Q5 cubic
regressions pass in 63.17 s. M2 collection: 2488 in 6.23 s, not a full-suite
execution or a physical-accuracy claim.

Full retained-matrix two-energy Ward Frobenius norms:

| Power | Proton Q0, same spin | Neutron Q0, opposite spin | Proton Q1, same spin |
|---|---:|---:|---:|
| 0 | 8.5750e-16 | 0.0000e+0 | 8.5750e-16 |
| 1 | 3.0944e-16 | 1.2403e-16 | 3.0944e-16 |
| 2 | 1.3877e-16 | 6.6050e-18 | 1.2484e-16 |
| 3 | 5.6037e-19 | 4.1523e-19 | 9.7083e-19 |
| 4 | 1.4226e-18 | 6.8878e-19 | 2.5882e-18 |

Both norm maps preserve closure at each power. The independent Hamiltonian
recurrence retains (Ef-Ei)(rho_eff-rho_P). The formerly nonzero proton g3/g4
defects close through the derived quartic terms, without a fit, tolerance
relaxation, inverse-transfer repair or change of Hamiltonian.

Controls: K2=9, P shell0, Q5/Q6 shell0 or both shell1; b=.4 GeV, mq=.35 GeV,
mg=0, P+=1.3 GeV, g=.3, Lambda=.8 GeV, eight boson and fermion modes;
paired normal-order subtraction and explicit finite mu_q²=.011 and
mu_g²=-.007 GeV² per g²; Ef=1.39, Ei=1.47 GeV²; Delta=(.12,.16) GeV,
mean PT=0. Proton endpoints have Jz=+1/2; the neutron transition is +1/2
to -1/2. Closure at two finite Q resolutions is NOT observable or UV
convergence, full four-current matching or physical renormalization.

The updated standalone benchmark also completes, preserving the earlier
finite-parent, PP-only and free/cubic-Q controls. Its all-exact-transverse
scalar residuals at powers0..4 are 2.526e-17, 4.379e-18, 3.905e-18,
2.310e-20 and 4.289e-20.

**Next under the active persistent goal:** Derive the independent minus-projector and finite longitudinal-envelope partners from the same physical/projector kernels, retaining plus-momentum dependence and the explicit finite-cell prescription. Check their boost, reversal and charge identities; do not infer intrinsic J-minus from the transverse Ward contraction or set it to zero by default. Assemble all four components with the common H2 waves and both norms through g4, then complete independent resolution and explicit finite-condition controls. Keep physical calibration explicit; do not add Wick H twice, gauge auxiliary legs, fit a conservation defect or claim restored covariance.
The complete four-current remains unavailable. No unresolved scientific
blocker requires user direction. No physical fit, covariance repair,
new quantum execution or public push is selected.

Backup: `computation_handoff/backups/2026-09-06_pre_canonical_self/pre_exact_quartic_Q_projector.tar`, a verified Git archive of
`a0a061afc4d4a23545a9658dab658f60c3cded41` (2,652,160 bytes).
No original was removed. The public mirror remains at the preceding
authorized publication ce14fb12, not this local checkpoint.

Theory: `sec:exact-quartic-Q-projector-current`.
Implementation: `m2_gaussian_cluster_quartic_Q_projector.py` and `m2_gaussian_cluster_quartic_Q_pair_projector.py` under `src/deuteron_wigner/quantum/`.

## Independent smooth minus and finite-cell spectral envelope

Scientific source `c6d03e4173e28c78394efbaf451662fea68ff635` derives the
smooth oscillator minus tangent from the physical plus-momentum dependence
of normalized Jacobi wavefunctions. It includes their squeeze and measure
normalization, with S_a=(I-w_a w_a^T)/(2P+) and
L_a=(S_a Q-w_a P_T/P+)·grad_Q+tr(S_a). The commutator [Pi,L_a] has exact
support through shell N+2. Complete shells remove the orthogonal frame
rotation, not this symmetric squeeze. The same straight path retains
intermediate P_T=Pbar_T+(t-1/2)Delta_T. K-minus is an off-diagonal tangent
and obeys K-minus(Pbar+P+ beta)=K-minus(Pbar)+beta·K_T. Its intrinsic value
is independently derived, not inferred from transverse continuity.

The exact Gaussian H product retains all interaction tails. Independent
changed-wavefunction derivatives and a direct derivative of projected
physical pair H verify it. The zero-transfer, zero-mean, charge-stripped
ground free-orbital boundary is b²(d-|w_a|²)/(2(P+)²), generally nonzero.

The field envelope has the explicit covariant spectral representative
E[A+]=chi(D+ - e A+) with cuts inside APBC/PBC eigenvalue gaps. Its
first variation is -e (chi(p_m)-chi(p_n))/(p_m-p_n) A+_mn, with the
equal-eigenvalue divided difference zero. Moving cuts within the same gaps
does not change this result. At Delta+=0, even arbitrary transverse/spin
multiplicity connects only equal p+, so this FIELD-envelope minus tangent
is exactly zero. Nonzero-transfer boundary mixing is independently nonzero.
This covers gap-preserving linear response, not spectral crossings or a
global holonomy choice. It does not remove the dynamical zero-mode sector
or set smooth Gaussian/Jacobi or retained-isospin currents to zero.

All 32 new tests pass: 20 smooth-minus tests and 18 existing exact-projector
regressions pass together in 8.39 s; 12 envelope tests pass in 1.07 s.
Three- through six-particle geometry, shells0/1, unequal fractions,
oblique paths, reversal, refinement, boosts, idempotency, exact free-H
support and physical qq/qg/pair-conversion kernels are checked. Envelope
tests independently perturb and diagonalize the Hermitian Fourier operator,
including zero-transfer degeneracies and nonzero boundary mixing.
M2 collection: 2520 in 5.28 s, not full-suite execution.

These are independently callable kernels, not yet full raw P/Q minus
actions or a complete four-current H2 assembly. The preceding full
transverse matrix checks through g4 at Q shells0/1 remain unchanged.

**Next under the active persistent goal:** Carry the smooth minus boundary
through all P/Q free, cubic and quartic H2 words through g4 with both
energies and norms. Derive the retained-isospin minus partner in a compatible
declared finite-cell prescription. Bind the spectral field-envelope result
explicitly; verify the combined four-current with independent H derivatives,
reversal, boosts and matrix identities, then resolution and explicit finite
conditions. No physical calibration, zero-mode completion, covariance repair
or push is selected. No unresolved scientific blocker requires user direction.

Backup: `computation_handoff/backups/2026-09-06_pre_canonical_self/pre_smooth_minus_and_envelope.tar`,
a Git archive of `a41118b2` (1,280,000 bytes; contents listed and verified).
Original documents remain in Git; no original was removed. Public ce14fb12
is unchanged. Theory: `sec:smooth-minus-spectral-envelope`.

## Smooth-minus H2 and total isospin

The independently derived smooth oscillator-minus boundary now executes on
every represented P/Q5/Q6 free, cubic and quartic block and returns through
all H2 words at g4. It retains exact Gaussian Hamiltonian tails, physical
flavor charges, charged spectators, both energies, raw backflow and both
original norms. The Hamiltonian and existing transverse APIs are unchanged.
The five inspectable families are retained, free Q, cubic Q, quartic source
and quartic Q5 pair; see theory `eq:smooth-minus-H2-complete-words`.

All 38 raw action checks pass across three selections, covering targets,
flavor reconstruction, boosts, reversal, statistics and every physical
source/pair family. Four full-H2 minus tests pass together in 765.52 s,
including an independent degree-eight wave polynomial. Its nonzero g5/g6
terms are excluded. The separate full Q-shell1 matrix check passed in
3918.33 s, with five family norms (retained, free Q, cubic Q, quartic
source, quartic Q5 pair) of 0.44766054, 0.05266997, 0.01653439,
0.00020032 and 0.00105993. Reversal and both norms pass at this shell.

All three neutral quartic pairs have NONZERO intrinsic minus boundaries at
shell0 although their transverse boundary vanishes through shell1: the
minus squeeze reaches two oscillator quanta, the translation only one.
A different free neutron qqqggg ground-shell zero at K9 follows from equal
quark radial factors times zero net charge; the proton control is nonzero.
Neither is an absent Hamiltonian or convergence evidence.

The retained-isospin contribution is resolved in the H2 source domain,
without setting an unknown raw kernel to zero. Gauged idempotency gives
I_f K_I + K_I I_i = K_I, hence I_f K_I I_i=0. K_I maps retained I=1/2
to I=3/2. The present flavor-degenerate strong Hamiltonian conserves TOTAL
SU(2), including antiquark generators -tau^T/2 and singlet sea-pair
creation. Every A1/A2/C3/A6 wave has total I=1/2. Consequently
A_f^dag H_QP K_I I_i and its reverse vanish by orthogonality for any
proper projector tangent, including minus. Retained PP partners vanish
likewise; both norms preserve the zeros. This does not cover arbitrary raw
Q vectors or isospin-breaking extensions. The raw isospin-minus kernel is
not supplied, and the existing nonzero raw transverse kernel is retained.

All 22 isospin checks pass in 47.92 s: independent Pauli tensor-product
Casimirs, actual proton/neutron waves at Q shells0/1, random-vector strong
W55/U65 intertwiners, and twelve cubic/quartic raw-complement sources.
Maximum wave residuals are 1.33e-16 (proton Q0), 1.23e-16 (neutron Q0),
2.81e-16 (proton Q1), and 2.52e-16 (neutron Q1). Two additional guards
pass, including rejection rather than projection of an isospin-impure wave.

`assemble_cluster_h2_four_current` explicitly combines physical free,
canonical/contact, Gaussian-regulator, paired Wick and exact oscillator
terms. It binds the gapped spectral field envelope at Delta+=0 and checks
the actual total-isospin domain. All four combined-current tests passed
in 2775.93 s, including proton/neutron reversal, boost, independent-H
Ward identities and both norms. Complete finite-reference/physical
matching flags remain false. M2 collection: 2608 in 5.60 s, not full-suite
execution. The 88 new tests passed across separate runs: 38 raw action,
5 full-H2 minus, 22 isospin, 4 combined current and 19 finite-response/
matrix-diagnostic tests. Scientific source: `1ecc6e97`.

The combined proton test has now passed: the maximum Ward Frobenius
residual over g0..g4 is 8.58e-16 and maximum boost residual is 1.83e-15,
including reversal and both norms. The neutron opposite-spin test also
passed, with maximum Ward residual 1.25e-16 and boost residual 4.11e-16.

The new `scripts/benchmark_m2_gaussian_cluster_four_current.py` evaluates
one named exploratory control per invocation, preserving full matrices
alongside its report. It separates Q5/Q6 shells, field-mode count, Gaussian
width, two finite mass terms and the neutron. Existing outputs cannot be
overwritten. The baseline completed in 439.62 s, with full-matrix Ward
norm 1.88e-15 and both norms positive. Its report and all complex matrices
are preserved in `m2_gaussian_cluster_four_current_baseline.json` and the
adjacent NPZ file. Four further full-matrix controls are now saved:

| Control, relative to baseline | Full raw Ward norm | Main finding |
| --- | ---: | --- |
| Field modes 8 to 16 | 1.88e-15 | All eleven saved matrices bitwise identical after paired subtraction |
| Finite quark coefficient of g², .011 to .021 GeV² | 1.78e-15 | Analytic H/N/current response matches full rebuild |
| Finite gluon coefficient of g², -.007 to .003 GeV² | 2.29e-15 | Nonzero order-four electromagnetic loop response despite zero bare gluon charge |
| Gaussian width .8 to .4 GeV | 1.74e-15 | Order-one and order-three matrix shapes change; not absorbed by the two mass directions |

Files share the baseline prefix and use suffixes `field_tail16`,
`finite_quark_shift`, `finite_gluon_shift`, and `Gaussian_width04`.
Each JSON identifies its adjacent full complex NPZ matrices and explicit
nonphysical parameters. The combined Q5/Q6-shell1 full-current benchmark
is still running. Its result is not implied by the passed minus-only test.

### Finite kinetic parameters and remaining regulator shape

The analytic directions in `m2_gaussian_cluster_finite_response.py` keep
vertex masses, bases, regulator and both energies fixed. They differentiate
the actual C3 backflow, H orders two/four, norm order four, and every raw
current word at orders two/four. Quark and gluon directions both include
loop effects. The inverse-square-root Fréchet derivative moves both norms,
including noncommuting/off-diagonal motion and eigenvalue degeneracy;
negative metrics are rejected, not clipped. See theory equations
`eq:finite-kinetic-H2-tangent` through `eq:finite-kinetic-norm-Frechet`.
The original 13 tests pass in 754.85 s, two full saved-matrix comparisons
in 121.39 s, and two normalized-current comparisons in 572.13 s.
Two additional matrix/least-squares diagnostic checks pass in 1.37 s.

At the benchmark point, raw current derivative norms (+,-,x,y), per GeV²:

- quark: (.00152661, 5.58120946, .00038345, .00031157);
- gluon: (.00033028, .00047255, .00008599, .00007537).

Changing the Gaussian width .8 to .4 GeV changes the full raw matrix by
only 0.3096% because the unchanged free term dominates. Order-resolved
relative changes at g0 through g4 are (0, .272166, .025181, .565988,
.349569). Neither finite mass direction changes g1 or g3. The best real
rescaling of the .4-GeV g1 current toward its .8-GeV matrix is 1.350939,
but leaves 8.4077% relative shape error. This is an independently checked
least-squares diagnostic, NOT a fit applied to H or the current. It is
not a lower bound on every possible observable fit, nor a demand that
all finite-cutoff bare matrices be physical observables.

Thus exact Wick-tail cancellation and Ward closure do not establish
regulator independence. The next parameterized construction must match
momentum-dependent vertex/current operators together, not just adjust
two masses or enforce one scalar conservation condition. A controlled
finite-reference kernel expansion is derived in the theory note as the
next implementation target; it is not yet wired through all H2 words.

Next: finish the running full-current Q-shell1 benchmark, then execute
paired finite-reference vertex/current transport and its independent
resolution controls. Keep physical finite conditions explicit and separate.
Ward closure is not cutoff independence; finite test masses are not
physical matching conditions. No zero-mode completion, covariance repair,
physical calibration, quantum execution or public push is claimed.

Backup: `computation_handoff/backups/2026-09-06_pre_canonical_self/pre_smooth_minus_P_Q_actions.tar`,
a 2,734,080-byte Git archive of `d1362b39`, contents verified. Originals
remain in Git; no original was removed. Theory: `sec:smooth-minus-H2-isospin`.

## Whole-vertex finite-reference shape

Scientific source `e83942eb` implements the derived family
`F_M(Q)=exp(-a Q) S_M((a-a_ref)Q)` for explicitly named finite widths,
with reference width at least the base width. The new degree is independent
of coupling order. `GaussianReferenceShape.integrate` evaluates exact joint
oscillator/Gaussian polynomial moments before projection. No product of
separately projected matrices replaces a full integral.

Canonical physical currents receive the same `S_M`; the base Gaussian's
first derivative receives `S_M-(a-a_ref)/a S_(M-1)`. Independent direct
physical-momentum differentiation verifies all three charged cubic channels,
including minus. Actual retained/Q5/Q6 cubic routes, all ten quartic source
families and three complete pair/projector matrix cases pass. The same
shape is carried by the original shifted H kernel in projector products;
charged spectator boundaries are retained. Existing default H2 models are
unchanged and do not silently receive a partially shaped operator.

The zero-transfer Wick trace is multiplied by `sum(r**m,m=0..M)`, where
`r=1-base_width²/reference_width²`. Finite-transfer external/internal paths
are derived separately, not obtained by differentiating an already
contracted common mean. The analytic finite sums agree with six direct
uncontracted four-leg loop quadratures. Every quark, antiquark and gluon
mode retains its original signed denominator, flavor/color factor and
paired H/current subtraction. See theory equations
`eq:reference-shape-current-weight`, `eq:reference-shape-Wick-trace`, and
`eq:reference-shape-Wick-path-moment`.

All 42 tests in the two new files pass together in 13.44 s. Three new
independent derivative cases are included in 256 surrounding tests passing
in 75.25 s: 45 distinct new tests, not 298 new tests. M2 collection is
2653 in 6.40 s, not full-suite execution. The complex shifted-shell
kernel-matrix errors at degrees 0,1,2,4,8 are
`.2055511, .0529182, .0120795, .000497863, 5.08573e-7`, against the
separately evaluated reference Gaussian. This is not full-loop convergence.

The next H2 integration has a concrete algebraic requirement. On the
conserved quartic slice, expand `E D E^dag` as the correlated triangular
sum `sum_{u+v<=M} delta_a^(u+v)/(u!v!) E^[u] D E^[v]^dag`, with
`E^[u]=(-partial_a)^u E`. Preserve the auxiliary projector in D.
Multiplying two degree-M half expansions would include unwanted terms;
the existing test provides a nonzero witness. This restriction is within
one physical vertex, not a new global cut across the H2 loop products.
The nonconserved current continues to gauge the whole physical vertex,
never an auxiliary leg. This correlated H2 integration is not yet executed.

Backups from exact pre-edit commit `0068f5b0` are
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_reference_shape.tar`
(1,198,080 bytes) and `pre_reference_shape_Wick.tar` in that directory
(61,440 bytes), with contents verified. No originals were removed.
The full Q5/Q6-shell1 four-current benchmark remains running. No physical
calibration, covariance repair, continuum convergence or public push is claimed.

## Correlated reference-shape H2 integration

Scientific source `b29fd27a` implements the triangular auxiliary pair sum
derived above. Its radial half derivatives are analytic, regular through
the radial zero at t=1, and independently checked by Laguerre integration.
They are unweighted derivative halves, not separate degree-M vertices.
The unchanged auxiliary weight, quark statistics and exact total-isospin
projector remain between every pair of factors. Direct physical four-leg
annihilation/conversion operators agree on the actual H2 source domain,
including a shell-one control. No equality on arbitrary auxiliary-isospin
columns is inferred.

One bound `GaussianReferenceShape` now follows the retained H, all cubic
and quartic sources, W55, U65, the Wick H/current pair, independent physical
currents and their exact shifted-projector Hamiltonians. The free H is not
shaped. Mismatched current/H shapes are rejected. The Wick series enters
exactly once, with the same composite subtraction; finite kinetic terms
remain independent. H2 products keep every selected whole-vertex factor
through g4, with no additional diagram-wide shape-degree cut.

All 69 new tests pass together in 124.45 s. Independent checks include
the full H2 recurrence, energy derivative, two-energy charge identity,
both norms, Hermitian pair action, unsubtracted Wick H, six contact types,
and bitwise recovery of the original H2 at degree zero or equal widths.
Surrounding checks pass in two disjoint runs: 318 in 473.46 s and 99 in
58.41 s. M2 collection is 2722 in 8.38 s, not a full-suite execution.

At E=1.4 GeV², Q-shell zero, base width .8 GeV and independently evaluated
reference width 1.2 GeV, the relative full-matrix Frobenius errors are:

| Shape degree | H2 Hamiltonian | Norm |
| --- | ---: | ---: |
| 0 | 3.71306e-4 | 1.45809e-5 |
| 1 | 2.77703e-5 | 6.76174e-7 |
| 2 | 2.37658e-6 | 3.53084e-8 |

These are finite-model improvements, not a universal error bound. The
separately evaluated reference four-current also passes, with Ward
Frobenius residual 1.91358e-15 in 443.48 s. All eleven matrices are saved
in `m2_gaussian_cluster_four_current_Gaussian_reference12.npz`, with the
same-prefix JSON recording parameters. Degree-one/two shaped four-current
runs are still active; H/norm agreement does not predeclare current matching.

The original-Gaussian full Q5/Q6-shell1 run completed in 3179.08 s. Its
raw/normalized Ward residuals are 1.86750e-15 and 1.86678e-15, and all
eleven matrices are preserved in
`m2_gaussian_cluster_four_current_Q5_Q6_shell1.npz` with its parameter JSON.
Against Q-shell zero, g0/g1 current coefficients are bitwise identical;
relative full-matrix changes at g2/g3/g4 are .0260056, .161299 and .905342.
The normalized total changes only .000120152 relative to its Frobenius
norm. This is a direct warning against allowing lower-order-dominated
totals to hide the 90.5% fourth-order resolution effect. It is not a
converged resolution study or restored dynamical covariance.

**Next:** finish the two running shaped-current members, compare every
coupling coefficient and both norm endpoints with the named reference,
verify full shaped-current reversal/boost/Ward identities, then strengthen
the independent resolution and finite-reference loop/current remainder
controls. Physical finite conditions remain explicit and uncalibrated.
No physical fit, second Wick H, auxiliary photon leg or public push.

Pre-edit backup: `computation_handoff/backups/2026-09-06_pre_canonical_self/pre_reference_shape_H2.tar`,
2,744,320 bytes, archive of `938672bd`, contents verified. Original tracked
sources and the benchmark script remain recoverable in Git; no originals
were removed. The public repository remains at its preceding authorized
publication. Theory: `eq:reference-shape-radial-half-derivative` within
`sec:smooth-minus-H2-isospin`.

## First full shaped current and exact moment acceleration

Scientific source `9652ea55` preserves the first full degree-one shaped
current, evaluated with the original scalar moment implementation in
1891.07 s. Its parameter JSON and all eleven complex arrays are
`m2_gaussian_cluster_four_current_reference12_degree1.json` and the
same-prefix NPZ in this directory. The raw/normalized Ward Frobenius
residuals are 1.83470e-15 and 1.83440e-15. Independent checks against the
current Hamiltonian close each coupling coefficient and both norm maps.
The first two tests in the new three-test full-current file pass; its
long reversal/boost test is still running, not recorded as passed.

Against the independently evaluated 1.2-GeV reference, the relative
four-component matrix errors at g1/g2/g3/g4 are:

| Shape degree | g1 | g2 | g3 | g4 |
| --- | ---: | ---: | ---: | ---: |
| 0 | .0705420 | .00919625 | .177019 | .0890472 |
| 1 | .00573539 | .00130692 | .0186147 | .00484471 |

Every nonzero individual component improves. The largest remaining
component error is .0198960 in the g3 minus coefficient. The normalized
total error is only 7.03609e-5; it does not replace the separated errors.
The free matrix and the identically zero g1 plus term are unchanged.
These are finite-reference errors, not physical calibration or convergence
as the field regulator and Fock/oscillator resolutions are removed.

Ten new raw shaped-minus checks cover degrees one/two: direct changed
physical-wavefunction derivatives, reversal, boosts and path refinement.
Four numerical checks retain genuinely nonzero coefficients and
covariances down to 1e-80. They are included in 235 focused tests passing
together in 274.45 s. M2 collection is 2739 in 13.81 s, not full execution.

The Gaussian insertion recurrence now operates on vectors of actual
oscillator matrix entries. Complete rectangular shells are downward
closed under lowering one occupation, so the recurrence has precisely
the same inputs and contractions. Tiny matrices use the original scalar
recurrence to avoid setup overhead. This is only algorithm selection:
no new shell cutoff, coefficient threshold or matching condition enters.
Six physical shaped matrices and three general complex-polynomial
matrices are bitwise identical to the pre-edit scalar implementation.
The six physical comparisons take about 3--50 times less time with
batching; that is a kernel-level measurement, not an end-to-end speed claim.
The full degree-one before/after rerun is still active.

Separate Q5-only and Q6-only shell-one four-current runs are now active,
alongside the joint-shell result already saved. Q6 first enters the raw
current at g4 in this H2 construction; compare raw coefficients before
the separately resummed norm maps. The joint fourth-order change need
not equal the sum of the separate changes, because Q5-to-Q6-to-Q5 words
couple the two resolutions. Measure the nonadditive difference explicitly;
do not assume attribution from the sector labels.

**Next:** finish full-current reversal/boost, degree-two matching and the
end-to-end algorithm comparison, then inspect the separate/joint sector
controls and continue finite-reference loop/current and resolution
completion. All previously started live calculations are preserved.
No physical fit, covariance repair, auxiliary photon leg, second Wick H
or public push. A read-only remote check confirms public main is ce14fb12.

The original scalar implementation is recoverable at `4396be46` in
`src/deuteron_wigner/quantum/m2_gaussian_field_regulator.py`; it is also
contained in the existing `pre_reference_shape_H2.tar` backup. No original
was removed. Theory: `eq:reference-shape-first-full-current-errors`.

## Separate resolution effects and finite-reference error bounds

Scientific source `8487b26c` preserves the separately evaluated Q5-only
and Q6-only shell-one currents, each with its parameter JSON and all
eleven complex matrices in `m2_gaussian_cluster_four_current_Q5_shell1.*`
and `m2_gaussian_cluster_four_current_Q6_shell1.*`. Raw Ward Frobenius
residuals are 1.71959e-15 and 1.70342e-15. Independent H2 coefficient
checks verify each coupling order and both norm maps for both separate
members and the previously saved joint member.

At unchanged finite benchmark parameters, the fourth-order matrix changes
relative to Q5/Q6 shell zero are:

| Changed resolution | Relative g4 matrix change |
| --- | ---: |
| Q5 only | 1.02480261 |
| Q6 only | .0182693936 |
| Q5 and Q6 jointly | .905341856 |
| Mixed difference, J11 - J10 - J01 + J00 | .120799757 |

The mixed difference is bitwise zero through g3 and nonzero in every
component at g4. This is already a raw-current effect, not induced by
normalization. The coupled Q5/Q6 words prevent additive attribution of
matrix effects from scalar percentages. Q5-only changes g2/g3 by
.0260056/.161299; Q6-only leaves all raw coefficients through g3 bitwise
unchanged. These results do not establish resolution convergence.

Exact longitudinal enumeration at K2=9 gives free kinetic-mass floors
2.5725, 4.0425, 3.3075 and 4.7775 GeV² in qqqgg, qqqq-qbar, qqqggg
and qqqq-qbar-g respectively. Positive transverse kinetic energy makes
the free-resolvent gap at least 1.1025 GeV² for both benchmark energies,
uniformly in oscillator cutoff. Thus the observed sensitivity is not a
nearby-pole artifact. The active Q5-shell-two run adds the first
non-diagonal radial propagator entries; parameters are not refitted.
The gap is not a claim at other longitudinal resolutions or energies.

The finite-reference degree limit now has an explicit mathematical
argument: the whole kernels and their first physical-leg derivatives
have an integrable reference-Gaussian majorant. At fixed finite
resolutions and finite widths, dominated convergence precedes projection
and passes through the finite H2 words and off-pole free resolvents.
The two-norm error bound in theory `eq:reference-shape-two-norm-error-bound`
uses the raw current difference and BOTH metric differences. It requires
positive endpoint metrics and neither assumes they commute nor clips
eigenvalues. It applies separately to each raw coupling coefficient
transported with the unchanged full norms. The evaluated total-current
bounds at M0/M1 are .00815856411/.000653473549, versus actual errors
.00782569994/.000639676869. These are finite-reference Frobenius errors,
not physical predictions, interval certificates or cutoff-removal bounds.

All 29 new checks pass across two runs: 25 resolution/normalization checks
in 215.09 s and four shell-uniform gap checks in 2.24 s. Separately,
20 existing shape checks (with derivative-majorant assertions added to
four cases) and the 20 new norm-error checks pass together in 3.07 s.
These counts overlap; they are not 69 distinct new tests.

The full original-versus-batched degree-one comparison has completed:
all eleven arrays are bitwise identical. Runtime was 2211.92 s versus
1891.07 s for the original run, so the measured local kernel speedups
are not an overall speedup claim. The original matrix dataset remains
canonical; the identical rerun is preserved in task scratch storage.

**Next:** finish the active degree-one full-current reversal/boost test,
both degree-two calculations and the Q5-shell-two current control. Compare
each order/component and both norm endpoints with the named reference;
test degree-two numerical equivalence independently. Continue remaining
finite-reference and resolution work without replacing uncertainty by
a small total-current change. Physical conditions remain unselected.
No second Wick Hamiltonian, auxiliary photon leg, covariance repair or push.
The pre-edit script and test sources remain in Git at `7de05458`; earlier
theory originals and the existing pre-reference-shape archives remain intact.

## Loop and finite-condition attribution of Q5 sensitivity

Scientific source `0345d642` saves five resolved plus-current controls and
two full-current finite-mass response controls, with all complex matrices.
The corresponding prefixes are `m2_gaussian_plus_resolution_` and
`m2_gaussian_mass_resolution_` in this directory. The two new benchmark
scripts reproduce their explicitly uncalibrated parameters and separate
the raw coefficients from the two endpoint norm maps.

The plus-current split uses fourteen coherent matrices: in each Q5 sector,
the quartic-source square and QQ/direct-Q6/iterated-Q6 backflow; in each
Q6 sector, direct and iterated squares and their interference. Their sum
reproduces the full-current plus coefficient within 7.0e-19 in Frobenius
norm for baseline, Q5-only, Q6-only and joint shell-one controls. The two
QQ backflow terms are further split into six physical-contact, auxiliary-
projected-pair and Wick-owned-mass submatrices, never added a second time.

At the named Q5-only shell-zero-to-one step, the plus-current change lies
entirely in QQ backflow to numerical precision. The Wick-owned diagonal
is the explicit finite kinetic-mass term after exact paired normal-order
subtraction, not the divergent trace. Its signed Frobenius projection
onto the plus g4 change is .985749315; the remaining subterms are retained.
For the mixed Q5/Q6 grid, the nonadditive plus effect is in the iterated
six-body squares and their Q5 backflow. These are measured selection
patterns for this grid, not permission to remove other words generally.

An independent calculation uses the previously derived analytic quark
and gluon finite-mass responses of the full four-current. At fixed
regulator, basis, energies and vertex mass, raw H/N/current dependence is
affine in these two finite coefficients through H2 order four. The
zero-parameter intercept is a diagnostic, not a selected physical
condition or a replacement of the saved benchmark. Its two norm maps
are rediagonalized, not approximated by a linear whitening response.

| Current coefficient | Q5 shell0-to1 relative change of the mass-independent part |
| --- | ---: |
| g2 | .510396783 |
| g3 | .161299080 |
| g4 | .776623572 |

The finite terms have signed Frobenius projection .974011935 onto the
full g4 resolution change, but that does not make the residual converged.
The absolute g2 change is unchanged: the finite PP term had enlarged
the denominator of its earlier relative diagnostic. These projections
are neither probabilities nor uncertainty/variance fractions. Finite
parameters must not be chosen merely to flatten cutoff dependence.

Independent zero-parameter Hamiltonian and norm rebuilds, every coupling-
order Ward identity, both norm maps and the separate W55 decomposition
verify the attribution. Intercept raw Ward Frobenius residuals are
1.98471e-15 and 2.35017e-15 at Q5 shells zero/one. The analytic finite
plus response agrees with the independent W55 mass submatrices below
8.5e-19 in Frobenius norm.

The Q5-shell-two plus-only control completes in 202.44 s and includes
the first non-diagonal radial propagation. Its raw g4 coefficient changes
.172394435 from shell one; its mass-independent plus coefficient changes
.240106509 relative to that part's shell-one norm. Full shell-two
four-current evaluation remains running. No other component's resolution
behavior is inferred from this plus-only calculation.

All 17 new tests pass across two runs: ten coherent-loop/identity checks
in 391.13 s and seven independent finite-condition checks in 73.76 s.
The long pre-existing degree-one full-current test file also finishes:
all three checks pass in 4157.11 s, including reversal and boosts.
M2 collection: 2785 in 9.02 s, not a full-suite pass.

**Next:** Finish both live degree-two full-current evaluations and their
original/batched comparison, and finish the complete Q5-shell-two current.
Compare each order/component and both norms before extending the resolution
study. Keep the loop and finite-condition diagnostics separate; no finite
mass fit, covariance repair, auxiliary photon leg, second Wick H or public
push. All original full-current datasets and earlier backups are unchanged.
The pre-edit theory/status sources remain in Git at `54ae9dc5`.

## Nested finite-reference residuals and current bounds

The new residual transport compares explicitly nested Q spaces with one
unchanged retained pilot, regulator, finite conditions and longitudinal
resolution. Exact physical-coordinate injections replace any assumption
that equal-sized matrices represent the same states. The reference is
finite, not the infinite oscillator or Fock complement.

For embedded smaller waves, form the actual reference residuals
`r1/2 = V1/2* - (E-H05*) A1/2_hat`,
`r6 = W6P* + U65* A1_hat - (E-H06*) A6_hat`, and
`r3 = W55* A1_hat + U56* A6_hat - (E-H05*) C3_hat`.
Then the exact errors obey the triangular recurrence

```
dA1/2 = R5* r1/2
dA6   = R6* (r6 + U65* dA1)
dC3   = R5* (r3 + W55* dA1 + U56* dA6).
```

Every reference interaction and its actual auxiliary projector is retained.
The local higher-order residual alone misses propagated lower-order errors.
For A1/A2, source truncation, free kinetic leakage and the smaller solve's
residual are also separated coherently. Dividing each complete residual
norm by the actual finite free-resolvent spectral gap bounds its wave
error. Poles are rejected, not clipped.

Each good-current word is bounded using the wave errors and the maximum
sum of absolute physical constituent charges. Both exact telescoping
orders are available; the smaller bound is used. The identity-density
version bounds the full order-four metric, and the inverse-square-root
bound carries the result through both positive endpoint metrics. See theory
`eq:nested-reference-H2-error-transport` and
`eq:nested-reference-good-current-bound`.

The Q5 shell2-to3, Q6-shell-zero control completes in 795.04 s and saves
18 complex retained matrices in
`m2_gaussian_resolution_defect_Q5_shell2_to3.npz`, with its parameters and
scalar bounds in the adjacent JSON. The lower g4 matrix agrees with the
previous independent shell-two plus control within 1.02e-18 in Frobenius
norm. This run predates the optional separate leading-defect output fields;
the error/current formulas are unchanged and that distinction is recorded
in its JSON. No old benchmark is replaced.

| Raw plus coefficient | Shell2-to3 relative change | Absolute change | Evaluated finite-reference bound |
| --- | ---: | ---: | ---: |
| g2 | .327227673 | .00712746836 | .142605492 |
| g3 | 4.02e-16 | 8.37e-20 | .00533396444 |
| g4 | .213002768 | .000854516502 | .0231918558 |

The normalized total changes by only 1.91470e-5 relative to its lower-shell
norm. Its absolute change is 8.71037e-5, bounded by .0456030. The bound is
conservative, not a physical uncertainty certificate; neither the small
total nor the existence of a bound establishes convergence. The g3
selection zero on this step must not be generalized to other components,
shells or kinematics. In particular, the next g4 step is 21.3%, following
17.2% from shell1 to2: those steps are not monotonically shrinking.

At the outgoing energy, the qqqgg C3 local residual norm is .00824383,
whereas its complete propagated residual is .0425304. Its actual wave
error is .0173155, below the gap bound .0261769. This nonzero witness
shows why a local-residual-only error estimate would be insufficient.

Ten new checks pass in 402.98 s, independently reconstructing all H2 waves,
each good-current coefficient and both norms on joint shell0-to1 and
Q5 shell1-to2 controls, including radial propagation, physical-coordinate
injections, charge contraction and pole/input checks. A separate direct
shell-three H2/current rebuild passes in 335.24 s, verifying every saved
reference good-current coefficient and both norms. All eleven new tests
therefore pass across the two runs.
The earlier eight-test run passes in 352.38 s and is superseded, not
counted as eight additional tests.

### Full-goal scope check

| Requirement | Established | Remaining work or claim boundary |
| --- | --- | --- |
| Ordered Wick Hamiltonian/current partners | Represented ordered contractions and paired subtraction at the declared finite prescription, propagated through H2 g4 | Physical finite conditions are not determined by normal ordering |
| Common interaction regulator | Whole physical vertices, sources, interactions and their independent current/projector partners share the explicit kernel | Regulator removal or physical renormalization is not established |
| Common H2 four-current | All represented components through g4, both energies/norms, finite-cell envelope and actual isospin source-domain cancellation; K11 and retained-shell-one independently verified | Full Q5-shell-two matrices now independently verified above; do not infer other components from plus-only results |
| Finite-reference matching | Degree-one/two full-current comparisons, analytic degree-limit argument, both-norm bounds and independent scalar/batched byte equivalence | Finite-reference verification does not establish regulator removal |
| Resolution/error control | Independent Q5/Q6 axes, loop/finite-condition attribution, nested finite-reference good-current bounds, K9/K11 physical-label probes and the full retained-current comparison | No demonstrated cutoff convergence or infinite-space error bound; completed Q5-shell-two control is recorded above |

This earlier scope check is superseded by the completed requirement audit
at the beginning of this note. The full Q5-shell-two control is now verified. This checkpoint does not change physical conditions, introduce a
second Wick Hamiltonian or auxiliary photon leg, repair covariance, or
authorize a public push.

### Retained-only and longitudinal-resolution comparison definitions

Two additional **full four-current** controls have started, without
restarting the three existing jobs:

- `retained_shell1`: K9, P shell1, Q5/Q6 shell1, 204 retained states;
  compare to the saved 37-state `Q5_Q6_shell1` with identical Q spaces.
- `longitudinal_K11`: K11, P/Q5/Q6 shell0, 69 retained states;
  compare to the saved K9 `baseline` using common physical-label probes.

The benchmark preserves its older case parameter dictionaries and H/N
matrices. All physical parameters, energies and explicit uncalibrated
finite conditions remain as previously declared. No old output was replaced.

**Retained coordinates.** The exact occupation-label embedding E restricts
every H2 wave, H and N at each energy. Normalized coordinates need
`T_a = N_large,a^(1/2) E N_small,a^(-1/2)`, not E itself. The two
energy maps differ by Frobenius norm `2.1193615051298834e-5`.
The naive E pullback of the normalized good current differs by
`6.6625729739158464e-9`; the correctly transformed current agrees to
roundoff. This small number is a coordinate artifact, not a physical error
estimate. Source: `m2_gaussian_retained_resolution.py`.

Writing `Q_E=I-E E†`, the low-minus-compressed-high Ward RHS is exactly
`-E† H_f Q_E rho E + E† rho Q_E H_i E`. This follows independently
from the Hamiltonian products and is linear in each H coefficient. It
does not supply a current, a minus component or a conservation repair.
For the stated same-Jz P0-to-P1 control, the spin-preserving density
cannot connect the added Lz=+/-1 states to the ground orbital, so both
density leakage blocks vanish. The contraction difference is therefore
a selection zero. It does not prove equality of the other current
components or establish retained-shell convergence.

**Longitudinal coordinates.** No index-by-index identification of 37- and
69-dimensional matrices is permitted. Three separate seeds are projected
with each actual joint symmetry isometry: qqq; qqqg with gluon helicity
-1; qqqg with helicity +1. All have global-Jacobi ground orbitals and
amplitudes `prod(x_i)^p`, for separately declared p=1 and p=2. Their raw
quark labels are `(0,1,3)`, `(1,1,3)`, `(0,0,3)`, with
`0=u down, 1=u up, 2=d down, 3=d up`, and raw color coordinates 0,1,1.
These are diagnostic families, not a physical wavefunction or selected
mixture. All full current matrices remain preserved.

For the three-column frame U, retain the **full** Gram matrices
`G_a=U† N_a U` and compare `G_f^(-1/2) U† J_raw U G_i^(-1/2)`.
Equivalently pull the full normalized current back with
`N_a^(1/2) U G_a^(-1/2)` at both endpoints. Coupling-order selection
precedes normalization. Source: `m2_gaussian_longitudinal_resolution.py`.

The initial work-in-progress color-zero gluon seeds were symmetry-null:
their norms were around `3e-21`, and normalizing their roundoff remnants
failed two compact-basis rotation checks. The actual color (12) exchange
is diag(+1,-1); identical first/second quarks with symmetric orbit require
color coordinate 1. The corrected, explicitly specified seeds pass
independent random compact-basis rotations. Numerically unresolved seeds
are rejected, not normalized. No Hamiltonian or current was altered by
this correction, and no result was published from those invalid probes.

**K11 gap and field envelope.** At fixed P+, the longitudinal cell length
changes as `L=pi K2/(2P+)`. Fixed mode counts do not fix physical momentum
caps. Here both caps exceed every physical leg; doubling Nb/Nf from 8 to
16 gives array-identical paired finite Wick operators on P and all Q
sectors. Independent exact enumeration bounds the K11 free-resolvent
gap by at least `931/1200 GeV²` at E<=1.47, uniformly in oscillator
cutoff through positivity of transverse kinetic energy. This is not a
statement about arbitrary K or ultraviolet convergence.

The current matrix jobs are still required. The retained/longitudinal
comparison definitions do not complete the full-goal audit above.
Original benchmark source is preserved at `bc1e8560` and in the existing
Git history; the pre-edit archives listed in the handoff remain intact.
There is no public push or selected physical calibration.

Verified method source: `446657de78a06e8b85ed695d6f0c9e7e77bbae1a`.
All 31 distinct new tests pass across runs. The combined run has 30 passes
in 348.50 s and a failed assumption that the retained Ward boundary must
be nonzero. The corrected two boundary tests pass in 272.58 s, checking
the actual selection-zero leakage blocks and an independent nonzero
complex-matrix witness. The 2827-test M2 collection is not a full-suite
pass. The five full-current controls remain running.

### Executed K11 four-current and K9/K11 named-probe comparison

Scientific source: `8c0e6a6674fc40737e3b15fad50e23b30a8a8312`.
The K11 run is now **complete**, in 2826.68 s. All eleven current/H/N
matrices are preserved in `m2_gaussian_cluster_four_current_longitudinal_K11.npz`,
with the adjacent JSON report. Raw and normalized total Ward Frobenius
residuals are `3.338564992045332e-15` and `3.3380309607323546e-15`.
An independent H2 recurrence rebuild verifies every g0--g4 matrix Ward
identity, both Hamiltonians/metrics, the raw common-order sum and both
final norm maps. This is not a new physical condition or calibration.

The saved 37-state K9 and 69-state K11 matrices are compared only through
the named physical-label frames described above. The 48 matrices in
`m2_gaussian_longitudinal_resolution_K9_K11.npz` preserve both probe powers,
frames, H/N endpoints, raw coefficients and raw/normalized totals. The
adjacent report retains absolute and component-resolved differences.

Relative Frobenius grid changes, over all four components and 3x3 entries:

| Quantity | p=1 family | p=2 family |
| --- | ---: | ---: |
| Raw g0 coefficient | 0.009837 | 0.076847 |
| Raw g1 coefficient | 0.160528 | 0.009147 |
| Raw g2 coefficient | 0.011465 | 0.170290 |
| Raw g3 coefficient | 0.677812 | 1.489652 |
| Raw g4 coefficient | 0.477642 | 0.675595 |
| Normalized total, g=.3 | 0.009833 | 0.077498 |

These are finite diagnostics, not physical error bars or a passed
longitudinal-convergence study. The raw total change is also decomposed
coherently as `D=sum_n D_n`, where `D_n=g^n (J_K11,n-J_K9,n)`.
The signed projection `Re tr(D† D_0)/||D||F²` is .983325/.988619.
The five signed projections sum to one and include negative terms;
they are not probabilities or uncertainty fractions. Most of the raw
total displacement here is already present at g0, which contains no
Wick loop or finite mass counterterm. This kinematic sampling effect is
separate from the large g3/g4 relative coefficient changes.

Seven distinct new checks pass across three runs: one K11 full-matrix
test in 54.10 s; five new probe/report tests within 26 tests in 30.79 s;
one coherent-attribution test in 2.22 s. Each saved normalized current
is independently pulled back from its complete 37/69-dimensional
matrix using both full Gram metrics. The 2834-test M2 collection is
not a full-suite pass. Reproduce the probe comparison with
`python -m scripts.benchmark_m2_gaussian_longitudinal_resolution --output NEW.json`;
the command preserves existing outputs.

Four primary full-current jobs remain: both degree-two evaluators,
full Q5 shell2 and retained shell1. K11 is no longer a live job. Originals
remain at `b255129f` and in the earlier archives. No output was overwritten,
physical condition selected or public push made. The full matching
objective remains active, with cutoff dependence unresolved.

### Verified degree-two full-current finite-reference comparison

Scientific source: `5528c317a53708bd9450d8f0b91eef5011d509a8`.
The batched degree-two run has completed in 13138.84 s. Its eleven
complex matrices are preserved in
`m2_gaussian_cluster_four_current_reference12_degree2.npz`, with the
adjacent report identifying the evaluator and the fixed 1.2-GeV reference.
The independently started original scalar run remains active; no
degree-two scalar/batched equivalence or speedup is asserted yet.

Every g0--g4 Ward coefficient, both independent H/N endpoint rebuilds,
the raw order sum and both normalized endpoints pass. Raw and normalized
total Ward Frobenius residuals are `2.0614020975004524e-15` and
`2.06100007552567e-15`.

Relative stacked current coefficient errors against the unchanged
finite reference are:

| Degree | g1 | g2 | g3 | g4 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.0705420 | 0.00919625 | 0.177019 | 0.0890472 |
| 1 | 0.00573539 | 0.00130692 | 0.0186147 | 0.00484471 |
| 2 | 0.000560407 | 0.000191000 | 0.00217063 | 0.000310572 |

Every nonzero higher-order component error is less than 0.15 times its
degree-one value. The maximum is .0024345542220830015 (g3 minus),
versus .019896032284170473 at degree one. The g0 matrices and zero
g1-plus component remain unchanged. All individual errors, not only
these stacked norms, are recorded in the JSON report.

The normalized total relative error is `6.942042171040344e-6`.
Its absolute error `6.311269711961503e-5` lies below the two-norm bound
`6.375321784723073e-5`; the same bound passes for each separately
transported raw coupling coefficient. Both full positive endpoint
metrics remain unchanged. This is a finite-reference bound evaluated
numerically, not an interval certificate, a physical calibration or
cutoff removal.

Eight distinct new tests pass within 30 checks in 28.17 s. The enriched
report is also checked separately by one component-error test (2.15 s)
and six coefficient/norm-bound tests (2.63 s); these are reruns, not seven
additional distinct tests. The 2842-test M2 collection is not a full-suite
pass. Originals remain at `8781f414` and in the earlier archives.
The original scalar M2, full Q5 shell2 and retained shell1 calculations
remain live. Finite-reference degree improvement does not resolve the
independently observed Q-shell and longitudinal sensitivities. The
complete matching goal is active, with no physical selection or push.


## Completed original scalar degree-two cross-check

Scientific source: `12e960157b2fb0e726faed17989ee7e9615cd774`. The independent original scalar run
completes in 17534.292037792038 s. Its report and all eleven matrices
are preserved separately as
`m2_gaussian_cluster_four_current_reference12_degree2_scalar.json` and
`.npz`; neither the original scratch output nor the batched archive is
overwritten. All eleven array payloads are bit-for-bit identical, including
signed zeros, dtype and shape. All common physical metadata agrees.
This checks every g0--g4 current coefficient, raw and normalized totals,
and both H/N endpoints, not merely conservation residuals or norms.

One new explicit byte/metadata comparison passes within 31 tests in
22.62 s, with the expensive previously passed degree-one reversal/boost
test deselected. The 2843-test M2 collection (7.68 s) is not full-suite
execution. The observed scalar/batched wall-time ratio is 1.3345383,
not a controlled performance measurement or a universal speedup claim.

The earlier degree-two finite-reference errors and bounds are unchanged.
Process 58742 is terminal and must not be restarted or polled as live.
Only full Q5 shell two and retained shell one remain live among the
primary calculations. Their component/order and two-norm controls still
precede the full completion audit. Original source/documents remain at
`60d3dd15` and in the earlier backups. No physical calibration, covariance
repair, new Hamiltonian term or public push is selected.

The theory source compiles offline to 215 pages. Changed pages 146, 153
and 163 are visually checked without clipping or overlap; no overfull
boxes or visible unresolved-reference markers remain. Inherited package,
reference and bibliography summary warnings remain. The six current
theory/status tests pass in 0.14 s. This is layout QA of the affected
pages, not a full-page visual or physics certification or a PDF deliverable.


## Completed retained-shell-one full current and exact comparison

Scientific source: `5dc5966d6a0ca7c973026ea4727877e541640fa4`. The 204-state retained-shell-one
run completes in 9645.553950958012 s. Its eleven full complex matrices
are preserved in `m2_gaussian_cluster_four_current_retained_shell1.npz`.
The exact 37-to-204-state comparison saves 27 arrays in
`m2_gaussian_retained_resolution_P0_P1.npz`, with both adjacent JSON reports.
The comparison takes 138.67 s; all original scratch outputs remain unchanged.

Both complete H2 norm maps and every coupling-order Ward coefficient pass
an independent Hamiltonian-wave recurrence check. Raw/normalized total
Ward residuals are 9.29254e-15 and 9.29118e-15. The actual occupation-label
embedding E and separate endpoint metric embeddings T_f/T_i are independently
reconstructed; no compact-column matching or one-metric shortcut is used.

After raw restriction, the minus component changes as follows:

| Order | Absolute Frobenius change | Relative minus change |
| --- | ---: | ---: |
| g0 | .1590693482 | .0208485236 |
| g1 | .00264835137 | .00789224111 |
| g2 | .00134390008 | .00193869086 |
| g3 | 4.02e-19 | 1.33e-15 |
| g4 | 4.21e-18 | 1.99e-15 |

Plus and both transverse components agree to roundoff at every order.
With the correct endpoint maps the normalized minus difference is
.15901352446414083 in Frobenius norm, or 2.0677646% relative to the
lower minus component (1.7491875% relative to the stacked four-current).
A naive raw-E pullback of the larger normalized current introduces a
separate 5.81681e-5 coordinate error; that is not the .159014 operator effect.

The independent retained smooth-minus P-projector coefficients reproduce
the observed g0/g1/g2 difference with full-matrix remainder norms
1.10e-14, 4.64e-16 and 9.20e-16. These are the free/cubic/quartic retained
projector boundaries; Wick and finite kinetic diagonals have no oscillator
boundary. No new term is added or fitted. The largest absolute effect is
already at g0, not an uncancelled Wick loop. Equality of H/N restrictions
does not equate the gauge variations of different oscillator projectors.

The Ward-boundary and difference-residual norms remain 1.3160e-15 and
1.4374e-15. At Delta+=0 their near-zero values do not constrain the minus
component. Near-null reference projections have ill-conditioned relative
ratios, so the report retains their absolute norms and explicitly warns
against interpreting those ratios as resolution errors. No matrix entries
or small coefficients are dropped.

Five new tests pass across three runs: one complete H2/current check
(427.49 s), three coordinate/report checks (46.17 s), and one independent
retained-minus attribution check (221.42 s). The 2848-test collection
(7.03 s) is not full-suite execution. All original source/documents are
recoverable at `b136f407` and in the earlier archives. The canonical
comparison script is `scripts/benchmark_m2_gaussian_retained_resolution.py`;
run it as a module with explicit reports and a fresh output path.

Processes 84213 and 70093 are terminal. The final Q5-shell-two validation
and the requested finite-model completion audit are recorded above. The result does not establish cutoff convergence, physical
renormalization or covariance restoration. The completed finite-model scope does not remove those physical limitations.
No physical calibration or public push is selected.

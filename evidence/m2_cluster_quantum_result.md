# Common-regulated H2 quantum integration: completed finite benchmark

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

8 September 2026, source `e3c0257391c4c2af8738905dfc6c36e6996d295e`.

Publication reconciliation, 9 September 2026: these completed results are
included in the user-authorized review-mirror/GitHub refresh representing
scientific source `6814ef5d`. No numerical kernel, parameter or result is changed
and no new quantum optimization is performed. Original no-push statements below
describe the 8 September scientific checkpoint, not the current publication.
All three deliverables in the
[implementation plan](m2_common_regulated_quantum_plan.md) are complete:
source-bound classical adapter, measured complex-current oracle, and actual
controlled-gate endpoint solves/readout. This advances the common-regulated
finite H2 reference from scientific source `520cc782`; it does not replace
the old M2o contract or bare-current quantum controls.

## What is bound and measured

`m2_cluster_quantum_reference.py` binds the actual two source snapshots and
their compact basis maps. The saved-current loader independently rebuilds
both H/N endpoints before accepting the existing eleven-array export.
The live route requires the full four-current assembly, not a plus-density
substitute. Both enforce the named physical-leg, smooth-minus, gapped spectral
and flavor-degenerate I=1/2 H2 source-domain prescriptions. Ward closure is a
check, not authentication of arbitrary transverse/minus additions.

For each endpoint, W = N^(-1/2) and L = W[H + e(N-I)]W. The current is
W_f [sum(g^n C_n, n=0,...,4)] W_i. The raw coefficients already represent
J^mu/(2P+); neither Wick H nor 2P+ is inserted again. All twenty transported
coefficients and four totals are measured separately. Norms remain full and
fixed during coefficient transport: these are not Taylor coefficients of
the resummed normalized current.

Each 37-state solver uses six qubits. Both exact and variational transition
readouts use seven: a flag keeps the two endpoint coordinate spaces distinct.
The real/imaginary Hermitian off-diagonal blocks measure
v_f^dagger A v_i on (|0>v_f+|1>v_i)/sqrt(2). This deliberate refinement of the
plan uses the same explicit endpoint handling even in the exact oracle.
Rectangular maps are supported when both circuits share a declared register.
No dimension-matching assumption identifies spin fibers.

Exact StatePrep is confined to oracle tests. Variational solves start from
the uniform tree with independent relative phases (seed 4901, sigma .03),
then execute the existing complex tree gates. Geometry is `tree`, step rule
`tree_state_retraction`, metric_rcond is None, probability regularization is
1e-20, support threshold is 1e-12, and the iteration cap is 2000. The support
graph only proposes gates; it does not sparsify the measured H. All tested
graphs are connected. Exact eigensystems are accessed only after both solves,
not as seeds. Readout coherently controls those actual gate preparations;
it does not reload optimized statevectors through StatePrep or dense unitaries.

The analytic runtime was PennyLane/Lightning 0.38.0/0.38.0 on
`lightning.qubit`, NumPy 1.26.4 and SciPy 1.17.1. No shots or hardware run,
efficient many-body encoding, resource advantage, or dependency upgrade is claimed.

## Executed evidence

All files below have prefix `docs/next_level/m2_cluster_quantum_` and paired
JSON/NPZ outputs. Every forward case measures 24 complex operators, solves
both endpoints independently, and passes every current/state-error bound.

| Case suffix | Independently identified member | Larger endpoint residual (GeV^2) |
|---|---|---:|
| baseline | Saved proton +/+ baseline | 7.94714e-9 |
| Q5_shell1 | Saved complete Q5 shell-one current | 7.94660e-9 |
| Q5_shell2 | Saved complete Q5 shell-two current | 7.94705e-9 |
| Gaussian_reference12 | Saved wider Gaussian reference | 7.95488e-9 |
| reference12_degree2 | Saved whole-vertex degree-two matched reference | 7.95460e-9 |
| p_minus | Fresh proton -/- four-current | 7.45972e-9 |
| n_plus | Fresh neutron +/+ four-current | 8.73744e-9 |
| n_minus | Fresh neutron -/- four-current | 8.42677e-9 |
| p_flip | Fresh proton -/+ and independent reversed +/- current | 7.94714e-9 |
| n_flip | Fresh neutron -/+ and independent reversed +/- current | 8.73744e-9 |
| zero_transfer | Fresh proton +/+ at Delta_T=0 | 7.94714e-9 |

Signs are outgoing/incoming twice-helicity. All 22 endpoint solves finish in
56 iterations / 113 objective evaluations with zero padding probability.
Both anchors remain e_f=1.39 and e_i=1.47 GeV^2. The baseline is K2=9,
retained/Q5/Q6 shell zero, b=.4 GeV, m_q=.35 GeV, m_g=0, g=.3, P+=1.3 GeV,
Gaussian width .8 GeV, eight boson/fermion modes and two loop flavors;
finite mass-squared coefficients are .011 and -.007 GeV^2 per g^2.
Electromagnetic charges are (2/3,-1/3), quadrature order eight,
mean P_T=0 and, except the named zero case, Delta_T=(.12,.16) GeV.
Family-specific Q cutoffs/reference shapes are retained verbatim in each report.
These are explicitly uncalibrated test conditions.

Across all eleven cases the largest exact-oracle/action discrepancy is
4.47e-16 and the largest controlled-current discrepancy on the executed
states is 7.78e-16. The stated absolute simulator budget is 8e-13, including
zero operators. The largest per-order Ward Frobenius residual is 1.15e-15.
The two reverse currents were freshly assembled with exchanged anchors/spins
and negative transfer, not defined as adjoints. Their largest raw-order
adjoint discrepancy is 6.74e-16; reversed measured transitions agree with
the conjugate forward values within 1.99e-16.

For independently diagonalized reference ground states, each exact vector
is phase-aligned to its executed gate state. With Rayleigh r below the first
excited eigenvalue, the endpoint distance is bounded by
sqrt(2) min(1, residual/(lambda_1-r)). Every measured raw coefficient/total
obeys ||A||_2 times the sum of the two distance bounds, plus the separate
8e-13 simulator budget. This is a phase-aligned numerical accuracy statement,
not a selection of physical nucleon spin phases or a form-factor extraction.
Degenerate/excited-state misuse is rejected; such cases require subspace work.

## Verification and reproducibility

The combined regression passes **144 tests in 29.60 s**, including all 59
new adapter/readout/replay checks. Four older full-action M2o oracle cases
are deselected, and 1040 inherited mpmath deprecation warnings recur.
Collection finds 2913 M2 tests in 17.36 s; that is not full-suite execution.
All six current theory/status consistency checks pass separately in 0.10 s.
The updated note compiles offline to 219 scratch pages; the changed status
and readout equations were visually checked. No final-pass overfull boxes
or literal unresolved-reference markers were found. Inherited bibliography
fallback/rerun and generic reference warnings remain; this focused layout
check is not full-document scientific certification or a PDF deliverable.

The new tests reconstruct H/N/L and source CSR bases, every normalized order,
the complete baseline matrix map (37 by 37 transitions for all 24 operators),
complex nonorthogonal/orthogonal/phase probes, rectangular maps, padding,
and independently replay every saved gate circuit and all 24 readouts.
Zero/small-coupling controls use g=0,+.001,-.001 with independently rebuilt
snapshots/norms and the same source coefficients; they are recombination
checks, not new full current assemblies. Zero transfer and both spin-flip
reversals above are full live assemblies.

Run the benchmark as a module with a fresh output path, for example:

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.benchmark_m2_cluster_quantum --saved docs/next_level/m2_gaussian_cluster_four_current_baseline.json --output /private/tmp/new_cluster_quantum_baseline.json
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.benchmark_m2_cluster_quantum --target NEUTRON --helicity-out -1 --helicity-in 1 --coupling .3 --transfer .12 .16 --check-reversal --output /private/tmp/new_cluster_quantum_n_flip.json
```

Use the project's quantum environment. The pinned temporary runtime at this
checkpoint is `/private/tmp/deuteron-quantum.i3XrCg/env/bin/python`; it is not
a durable project dependency. The benchmark refuses to overwrite evidence.
Fresh current assembly is done once outside the optimizer. The saved-family
route does not restart the completed long matching calculations.

The first archive replay caught a packaging error: storing scipy CSR objects
directly in NPZ produced pickle-bearing members. This was fixed by explicit
numeric data/indices/indptr/shape arrays. Every other saved array was verified
byte-for-byte unchanged and the basis was reconstructed from the actual
source. No pickle was enabled. Original nine affected exports remain under
`computation_handoff/backups/2026-09-08_pre_quantum_basis_export/`; do not use
them as normal replay inputs. Two later exports were numeric from creation.
Earlier reports omit the `solver_controls` field added later to the script;
the executed controls above and saved parameters/history record those runs.

## Scope and next scientific step

The full current's numerical kernels and original eleven-array current
exports are unchanged. Pre-edit code and theory are recoverable at
`8e0146d0`; the original 22 volumes and bibliography remain untouched.
The living note now supplies the tagged-readout and two-state-error equations
in `sec:common-regulated-quantum-readout`. No new volume is needed for this
bounded continuation. Public GitHub/review-mirror release remains `db7bb7c6`;
this is local development, not a new publication.

Next, analyze the common-regulated current's dynamical spin/rotation and
resolution dependence with the same explicit source phases and both norms.
Separate kinematic identities, measured current-prescription effects,
retained/excluded/longitudinal resolution effects and finite-reference errors;
do not fit a conservation or covariance defect away. The already observed
20.7% Q5 shell1-to2 raw g4 change remains even though its total-current change
is small. Quantum parity does not resolve it.

Then come shared physical finite-condition/current selection and covariance
with the theory lead, microscopic spin-one nuclear/GTMD closure with matching,
evolution and sum rules, and structured topology-aware many-body scaling /
later explicit higher-Fock enlargement. None is silently selected or marked
complete by this benchmark. No scientific blocker arose in this three-part step.

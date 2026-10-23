# Executed exact M2 PennyLane oracle — 5 September 2026

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Scientific source checkpoint: `5b44ca7390059039833bf848b8fcc8402d9c8f18`.
This is an exact analytic simulator validation of the existing exploratory
266-state represented Hamiltonian/current, not a physical deuteron result.
The implementation consumes M2o directly; frozen Q0/Q1/Q2 fixtures are not
used as replacement Hamiltonians or M2 state sinks.

## Result and independent mathematical check

`src/deuteron_wigner/quantum/m2_pennylane_oracle.py` uses nine big-endian
wires, `lightning.qubit`, complex128, exact `qml.StatePrep`, and analytic
`qml.SparseHamiltonian` expectations for the penalized Hamiltonian, six
Hermitian reversal quadratures, and the padding projector. No generic Pauli
expansion, ansatz, optimizer, hardware target, shots, fit, or activation is chosen.

SparseHamiltonian is an observable, not a nonunitary state-update operation.
For `q(v) = ||v||² <v/||v|| | O | v/||v||>` and `q(0)=0`, the adapter measures

```
<a|O|b> = [q(a+b) - q(a-b) - i q(a+i b) + i q(a-i b)] / 4.
```

This polarization identity retains normalization factors and the imaginary
sign, works for nonorthogonal endpoints, and handles identical/phase-related
states. Setting `a` to each computational basis vector reconstructs all 512
action components from actual backend measurements. The implementation does
not multiply a sparse matrix by the returned state and call that independent
quantum action evidence. This O(2^n) verification is deliberately an exact
oracle, not a scalable quantum algorithm or a claim of quantum advantage.

The independent reference uses the original compact Hamiltonian and directed
M2n matrices; its quadratures are derived from adjoints, not read back from
the encoded observable dictionary. Proton and neutron targets are tested
separately at `(0.2,0)` and `(0.12,0.16) GeV`, including both reversals.
Seeded complex non-eigenstates, nonorthogonal/phase-related states, exact
helicity eigenstates, and nonzero spectral transitions are exercised.

Worst absolute errors across the four target/transfer cases:

| Check | Maximum error |
|---|---:|
| State preparation | 0 at stored precision |
| Hamiltonian action | `9.34e-15 GeV²` |
| Six quadrature actions | `1.25e-15` in each operator's units |
| Directed/reversed density/current actions | `1.24e-15` in each operator's units |
| Hamiltonian/density/current expectations | `1.78e-15` in each operator's units |
| Complex transitions | `1.59e-15` in each operator's units |
| Continuity action | `3.19e-16 GeV²` |
| Nonzero spectral current identity | `3.54e-14 GeV` |
| Ground energy | `2.89e-15 GeV²` |
| State and reconstructed-action padding | 0 at stored precision |

The strongest selected Jx transitions have magnitudes about `1.04151 GeV`
on-axis and `0.624905 GeV` obliquely; the test cannot pass using only the
zero diagonal eigenstate currents. At oblique transfer both current actions
are nonzero. Pure-padding states and coherent represented/padding mixtures
also verify the penalty and absence of cross-block matrix elements.
Exact numerical values and versions are in `m2_pennylane_oracle_validation.json`.

## Fresh environment and reproducibility

A clean environment was created directly from the unchanged
`environment_quantum.yml`, including its editable `.[analysis,quantum]`
installation. No dependencies were manually copied and no user-site or
`PYTHONPATH` injection was used. `pip check` passes. The tested platform is
macOS arm64, Python 3.11.16, NumPy 1.26.4, SciPy 1.17.1, SymPy 1.14.0,
mpmath 1.4.1, PennyLane/Lightning 0.38.0, autoray 0.6.12, and LHAPDF 6.5.5.
The isolated environment for this run is
`/private/tmp/deuteron-quantum.i3XrCg/env`; recreate it if temporary storage is cleared.

From the repository root, an independent reproduction is:

```sh
conda env create --prefix /path/to/new/quantum-env --file environment_quantum.yml
conda activate /path/to/new/quantum-env
python -m pip check
env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q -p no:cacheprovider tests/test_m2_*.py \
  tests/test_q0_plhqcd0.py tests/test_q1_plhqcdstate.py tests/test_q2_plhqcdobs.py
env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/validate_m2_pennylane_oracle.py
```

The completed regression is **250 passed in 269.53 seconds**: 217 M2 tests
(including 28 new quantum-oracle cases) plus 33 Q0/Q1/Q2 tests. This is a
named regression, not a full-repository pass. Optional quantum tests skip
when PennyLane is absent; none skipped in this accepted run. The new mpmath
version emits 1,884 upstream `bitcount` deprecation warnings, and an existing
Q2 mutation-boundary test emits one sparse-efficiency warning. These warnings
are retained, not suppressed.

The first fresh-environment run exposed two inherited, platform-specific
thresholds: the M2h third singular value was `6.742e-12` versus a `4e-12`
assertion; the forward-charge residual was `1.665e-15` versus `8e-16`.
Neither changes the science. The two resolved singular values remain
`1.786e-3` and `2.403e-4`, and the diagnostic rank is still two. The null-value
test now uses the real/imag Frobenius roundoff budget
`sqrt(2*m) ||scales|| eps/h` for central differences of O(1) pair currents.
Charge uses a `64 eps` numerical budget for the 5,376-node normalized grid.
Only these two tests changed: no Hamiltonian/current equations, calibration,
normalization prescription, or diagnostic-rank threshold changed.

## Theory-note verification

The revised note compiles with Tectonic 0.17.0 to 123 pages. The updated
quantum derivation is visually checked. All 75 bibliography entries and 45
cross-reference targets resolve; 316 source labels are unique, no rendered
`??` placeholders occur, and no overfull boxes or missing glyphs are reported.
The previously documented generic BibTeX rerun notice persists despite those
checks; underfull-box and engine compatibility notices also remain. A
warning-free publication build is not claimed. The PDF is a verification
intermediate, not a new requested deliverable or replacement theory volume.

## Backup, scope, and next scientific action

The pre-edit theory and pickup documents are preserved in
`computation_handoff/backups/2026-09-05_pre_exact_quantum_oracle/pickup_and_theory.tar.gz`
(SHA-256 `291e5e7f13b15975d436e1f15b3eaaaf8167298df3fb807a7049c2ab77b17b60`).
The two preceding numerical tests are in the adjacent `response_tests.tar.gz`
(SHA-256 `495bde825fbd587169580e8629f9dc61912caf05909f626fc443c007dc5974d9`).
The complete preceding checkpoint remains in Git at `6af331e3`.
Original theory volumes, scientific source matrices, the orphaned C157 files,
and the published review mirror are unchanged. This local checkpoint is not
yet mirrored or pushed; GitHub remains at the preceding publication.

Next, derive full/all-emitter C117 action beyond the represented source image.
Trace the field/Fock-state and finite-cell factors and the already-fixed
`P^- -> M²` conversion, construct the lawful total-sign extension, and verify
source pullback, symmetry, units, and count-once normalization. Do not insert
unavailable terms as zero or create independently normalized C53 copies.
This source-level derivation can proceed without choosing a production
electromagnetic/LPS current, a physical state/fit, or a variational ansatz.
Those choices remain separate decisions. The exact oracle is now the
regression reference for subsequent operator and quantum-algorithm work.

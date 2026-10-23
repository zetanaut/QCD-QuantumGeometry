# Common-regulated H2 quantum reference: plan and completed execution

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

**Completed 8 September 2026:** all three deliverables below now execute.
See [the result and replay record](m2_cluster_quantum_result.md): eleven
forward cases, both independent spin-flip reversals, all 24 complex
operators, and both controlled-gate endpoint solves pass. Six-qubit solves
and seven-qubit tagged readout remain explicitly finite-model diagnostics.
The preflight and original requirements below are retained as history,
not an unimplemented queue. The next scientific work is dynamical
spin/rotation and resolution analysis; physical matching remains separate.

Prepared 8 September 2026 from public/local release
`db7bb7c66f74787ce12c7b5239861f6168dee686`, representing scientific source
`520cc782c92663bbe6dd166ddfa2c19cf72aa1fd`.

## Objective and present boundary

Carry the completed **explicit finite-model** H2 Hamiltonian, both endpoint
norms and four-current through the existing measured quantum interface.
First establish exact encoding/measurement parity, then variational-state
accuracy. This is the next implementable benchmark, not a physical fit,
regulator-removal result or many-body quantum-advantage claim.

No unresolved physics choice blocks this parameterized benchmark. The existing
finite parameters and current prescriptions are named test inputs, not newly
selected production conditions. H2-first and canonical global-Jacobi authority
remain selected. Do not restart any completed matching calculation.

The old `m2_pennylane_contract.py` and `M2PennyLaneOracle` are typed to the
266-state M2o baryon-current construction. The graph circuit accepts a caller's
Hermitian matrix, but `benchmark_m2_matched_four_current_quantum.py` still builds
the older **bare normal-ordered** four-current. Neither is an already-integrated
quantum reference for the new common-regulated current. Preserve both as
historical regression controls; do not relabel their results.

## Scientific objects that must travel together

Use `build_cluster_pilot`, `GaussianClusterH2.snapshot`, and the separately
bound `assemble_cluster_h2_four_current`. Follow theory
`sec:finite-H2-matching-completion` and `m2_wick_regulator_matching_result.md`.

For each endpoint a, retain its reference energy e_a, effective H_a, positive
N_a and W_a = N_a^(-1/2). The solver matrix is

    L_a = W_a [H_a + e_a (N_a - I)] W_a.

It is **not** H_a alone or W_a H_a W_a. These e_a are explicit linearization
anchors, not optimized physical eigenvalues. A fixed-anchor benchmark does
not solve the nonlinear self-consistency or physical-calibration problem.

The supplied directed current has raw coefficients C_n^mu, n=0,...,4, in
the order (+,-,x,y), already divided by 2P+. Its measured normalized form is

    C^mu(g) = sum(n=0,...,4) g^n C_n^mu,
    Jbar^mu = W_f C^mu(g) W_i.

Apply the two separate norm maps after selecting raw order four. Measurements
of W_f C_n^mu W_i keep the full endpoint norms fixed; they are diagnostics of
transported raw coefficients, not Taylor coefficients of the resummed norms.
Do not add a second Wick Hamiltonian or a second factor of 2P+.

Keep the exact retained coordinate order, target, both helicities, P/Q5/Q6/K
resolution, both anchors, coupling, masses, oscillator width, Gaussian shape,
field-mode caps, finite Wick terms, charge/flavor assignment, transfer/mean
momentum, all four prescriptions and quadrature controls explicit. Equal
matrix dimensions are not proof of a basis map. The isospin-minus result
remains restricted to the proved flavor-degenerate I=1/2 H2 source domain.

## Execution order and acceptance

### 1. Bind the new classical reference without rebuilding expensive currents

Start with the saved `m2_gaussian_cluster_four_current_baseline.json/.npz`:
proton, Jz=+1/2 at both endpoints, K2=9, retained/Q5/Q6 shell zero,
e_f=1.39 and e_i=1.47 GeV^2, Delta_T=(.12,.16) GeV and mean P_T=0.
All other parameters come explicitly from that report and its source builder.
There are eleven saved arrays: H/N at each endpoint, raw and normalized total
current, and five four-component raw coefficients.

Implement a small common-regulated adapter rather than copying the old M2o
phase/manifest machinery. Reconstruct the actual declared pilot coordinate
map from the source and compare both saved H/N endpoints with the independently
rebuilt snapshots. Bind the saved currents only to that identified model.
Reject mixed bare/regulated endpoints and incompatible metadata. Treat saved
input as data: no executable pickle, no inferred physical defaults.

Acceptance: exact raw-order reconstruction; independent positive-norm and
linearized-H reconstruction; all-component normalization; original Ward
identity and source/basis agreement. Reuse the saved Q5-shell-one/two and
reference-width/degree-two matrices as separate controls, not averaged models.
Matrix parity alone is a classical adapter result, not quantum execution.

### 2. Measure the new current, including complex transitions

Use the installed analytic `lightning.qubit` simulator and sparse Hermitian
observables. The baseline endpoint has 37 states embedded in 64 amplitudes
(six qubits). Preserve compact big-endian ordering. Keep zero-padded currents
and a separately penalized solver Hamiltonian with padding above a verified
spectral upper bound. Do not rely on positive physical energies to exclude
zero-energy padding from a variational search.

For a directed square matrix A in explicitly aligned coordinates, measure
(A+A^dagger)/2 and (A-A^dagger)/(2i), retaining both real and imaginary parts.
Use complex non-eigenstate probes and nonorthogonal, orthogonal and pure-phase
endpoint pairs. Exact StatePrep/polarization is appropriate for this oracle
only; label it as an exact simulator diagnostic, not a variational solve.

Acceptance: backend-tracked measurements reconstruct all twenty transported
raw coefficient matrices and the four total components against independent
compact calculations. Test actions or enough basis transitions to check the
whole map, not just diagonal eigenstate expectations. Verify padding
isolation/penalty, zero and small coupling, zero transfer, oblique transfer and
an independently assembled reversed transfer with the endpoints exchanged.
Adjoint symmetrization alone is not an independent reversal test. Error
tolerances must be stated per operator/order with absolute budgets for zeros.

The saved baseline supplies only one proton/helicity/transfer pair. Complete
target/spin acceptance requires separately assembled proton and neutron
Jz=+/-1/2 endpoints and helicity-preserving/flipping currents, preserving the
existing explicit phase maps. Do not invent missing blocks from dimensions or
replace a four-component current with the plus-density diagnostic.

### 3. Solve and measure from the actual controlled-gate states

Reuse `SymmetryGraphCircuit` and `quantum_natural_fit` on each required L_a.
Use the existing tree-only complex initializer, with declared seed and controls;
do not seed the optimizer with exact eigenvectors or prepare its state via
StatePrep/dense-unitary insertion. Keep exact reference vectors outside the
solver input path. A graph-support threshold proposes gates only: it must
not sparsify the measured Hamiltonian. Solve every disconnected component
if one occurs; never silently discard it.

For unequal endpoint states, a single-state expectation is not a transition.
Use coherent controlled preparations of the two solved circuits. One explicit
readout is (|0> psi_f + |1> psi_i)/sqrt(2), with flag-block observables

    O_Re = [[0, Jbar], [Jbar^dagger, 0]],
    O_Im = [[0, -i Jbar], [i Jbar^dagger, 0]].

Their expectations give Re/Im(psi_f^dagger Jbar psi_i). This route uses one
additional readout flag (seven qubits for two 37-state endpoints); distinguish
that resource from the six-qubit state solver. Rectangular spin-fiber maps
need explicit embeddings. Do not reload a classical statevector with StatePrep
and describe that measurement as controlled-gate transition execution.

Acceptance: executed residual <=1e-8 GeV^2 at each endpoint, explicit padding
probability, and independent ground-state identification/gap checks. Energy
agreement or optimizer success alone is insufficient. Compare each measured
component/order to the same operator on the executed states first, and then
to the independent model-reference states with phase conventions explicit.
For normalized, phase-aligned endpoint vectors the elementary bound is

    |v_f^dagger A v_i - u_f^dagger A u_i|
        <= ||A||_2 (||v_f-u_f||_2 + ||v_i-u_i||_2).

Use separately justified spectral-separation/residual bounds for those state
errors, or report the direct finite-model distances. Near degeneracy requires
a subspace treatment; a residual does not by itself identify the ground state.
Keep simulator roundoff, variational-state error, finite-reference matching
error and P/Q/K sensitivity separate. A small summed current must not hide a
large raw fourth-order change.

## Historical preflight completed before implementation on 8 September

- Five saved matrix families checked: baseline, Q5_shell1, Q5_shell2,
  Gaussian_reference12 and reference12_degree2. All eleven arrays are present;
  raw-order sums reconstruct bitwise and both-norm current reconstruction has
  zero observed difference using the existing stable whitener.
- All ten endpoint metrics are positive. Each reconstructed 37-state L_a has
  a connected 36-edge proposal tree at threshold 1e-12; the smallest observed
  first spectral gap is 0.5918267 GeV^2. This is a finite-matrix preflight, not
  physical mass determination or a newly executed quantum solve.
- Installed PennyLane/Lightning 0.38.0/0.38.0, NumPy 1.26.4 and SciPy 1.17.1
  are usable. The existing graph/oracle regression passes 34 tests in 5.59 s;
  four expensive full-action oracle cases were deselected. There are 1040
  inherited mpmath `bitcount` deprecation warnings. These tests exercise the
  existing backend and old reference models, not the new adapter above.
- Separately, 36 saved-matrix/norm/resolution checks pass in 2.55 s (five
  expensive independent-rebuild cases deselected), and all six current
  theory/status consistency checks pass in 0.11 s. Together with the backend
  run these are 76 targeted checks, not a full repository test run.
- No new common-regulated quantum adapter, measurement suite or solve is
  claimed complete by this preparation note. No long matching job was restarted.

The pinned local runtime is `/private/tmp/deuteron-quantum.i3XrCg/env/bin/python`
at this checkpoint. It is temporary, not a durable dependency: use the
repository's quantum environment declaration if it is no longer present.
No dependency upgrade is needed for this step. The current upstream
[SparseHamiltonian documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.SparseHamiltonian.html)
describes CSR expectation-value observables, not nonunitary gates; the
[Lightning device documentation](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_qubit/device.html)
describes the simulator. Those moving pages are API context, not evidence that
this project's pinned 0.38 environment has been upgraded or validated on 0.45.

## Original handoff and stop conditions (now executed above)

The next implementation should deliver one small adapter/readout layer, focused
equation/measurement tests, one resumable benchmark with fresh output paths,
and a short result update. Preserve old scripts, saved matrices and original
theory. Avoid new phase numbers, blanket test-count gates and duplicate ledgers.
Do not rerun expensive current construction inside optimizer iterations.

No new theory volume or TeX revision is needed merely to describe this launch:
the mathematical source is already in the living note. Record actual new
derivations/results there when obtained. The source and all pre-preparation
documents remain recoverable at `db7bb7c6` in local and public Git history.
This preparation corrects the stale operational roadmap and adds a pickup
pointer; it does not change scientific kernels or published evidence. Keep
the public review mirror at its identified release until another publication
is requested; local preparation is not an unannounced GitHub release.

Escalate only a genuinely inequivalent physical choice or an unresolved
mathematical contradiction. Debug encoding/solver issues and report numerical
nonconvergence honestly; they do not authorize fitting away covariance or
selecting physical finite terms. After this benchmark, advance dynamical
spin/rotation and resolution analysis, shared physical matching, microscopic
spin-one nuclear/GTMD closure, and structured many-body/topology scaling.

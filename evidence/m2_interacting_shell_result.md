# Interacting complete-shell H2 and linear-storage quantum geometry

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

2026-09-06--07 continuation of the selected canonical Galerkin/H2-first path.
This is a **validated bare finite-model construction**, not renormalized QCD,
physical form factors, or completion of the three physical workstreams.
Scientific source: 2effede59449b9724ee2ec9485bda7be66935702.

## One propagation for the represented interaction and current

The complete normal-ordered quartic five-body action W55 and the leading
six-body excursion now use the same full non-diagonal Galerkin resolvents
as the complete-shell free-Q study. Every raw spin, color and orbital
amplitude is retained. The earlier cutoff0/1 assembled operators remain
independent controls; their numerical results are reproduced there.

The Q5 contacts and cubic Q5/Q6 transitions are now tensor-factorized.
For each exact longitudinal/spin route, an orbital matrix and color matrix
act on packed coordinate blocks. This avoids assembling a many-body sparse
matrix without selecting a rank or deleting small amplitudes. Adjoints use
the conjugate-transposed same factors. Quark antisymmetry, gluon Bose factors,
both IF orderings, all A4 assignments and the coherent current square remain.

The pair-current auxiliary slot is a spin-zero adjoint **current**, not a
physical helicity-zero gluon. Let its exact I=1/2/S3-sign projector be P_a,
and let D be the positive pair-transfer weight. Because [P_a,D]=0,

    E U (U† D U) U† E† = E P_a D P_a E† = E D P_a E†,  U U†=P_a.

The implementation applies P_a explicitly; it does not drop it or
diagonalize a growing dense auxiliary projector. At the second shell its
raw auxiliary dimension is2880. Independent full-tensor projection and
old complete-isometry comparisons check this identity on arbitrary columns,
not only on an already symmetric source image. IG conversion and sea/gg
annihilation are counted once in this square, separately from IF conversion.

For source orders r=1,2, use Ar=R5 Vr and Br=R5 W55 Ar. The single-W
coefficient of g^n, n=4,5,6, is the sum over r+s+2=n of

    H[n]   = Vr† Bs,
    N[n]   = Ar† Bs + Br† As,
    rho[n] = Br,f† rho55 As,i + Ar,f† rho55 Bs,i.

These isolated g5/g6 terms are diagnostics, not all QCD at those orders.
Only n=4 enters the common fourth-order H2 model. With

    A1=R5 V1, A2=R5 V2, F6=W6P+U65 A1,
    A6=R6 F6, B5=R5 U65† A6, C3=R5 W55 A1+B5,

its raw waves are g A1+g² A2+g³ C3 in Q5 and g² A6 in Q6.
Their products are truncated at the stated order. The same waves produce
H_eff, N=I-d_E H_eff, and both spin blocks of the local good current.
Whitening uses that N without eigenvalue clipping. Raw W-induced tails
are not projected back into the older free-source image.

Independent full-Q iterative solves check both the single-W response and
the six-body coupling-order remainder. The six-body solve reproduces the
expected factor-four remainder reduction when g is halved. The combined
H/norm/current and non-diagonal energy-derivative checks also pass.
These are finite-model checks, not physical uncertainty estimates.

## Executed second-shell interaction and quantum result

K2=9, retained cutoff0/dimension37, excluded Q5/Q6 cutoff2, b=.4 GeV,
mq=.35 GeV, mg=0, g=.3, local matching energy E0=1.4 GeV²,
Delta=(.12,.16) GeV. No parameter is refitted.

Raw Q5 dimensions are22230/130500; raw Q6 dimensions are43692/364800,
or561259 including P. The complete leading six-body coefficient norms are

| Quantity | Second shell |
|---|---:|
|H6[4] Frobenius norm, GeV²|.00138714724610529|
|N6[4] Frobenius norm|.000989269491065903|
|electromagnetic rho6[4] norm|.000654787658136931|
|charge/norm disagreement|1.20109e-19|
|six-body probability-coefficient norm|.000330186110536479|
|five-body backflow-coefficient norm|.000668268910920797|
|direct return norm, GeV²|.000723785392573396|
|iterated return norm, GeV²|.000744314265558985|
|coherent cross return norm, GeV²|.000103570421844039|

Matrix norms do not add as scalars. Equal-mode zeros in two K9 direct
components are selection rules checked in the earlier K11 source tests,
not absent interactions. This leading excursion is not a full Q6 resummation.

Actual six-qubit/72-parameter graph gates start from a tree-uniform state
with seeded small phases, never an exact Hamiltonian eigenvector.
Exact eigensolves are performed only after optimization for comparison.

| Target / optimizer | M², GeV² | state residual, GeV² | updates | model F1 | model F2 |
|---|---:|---:|---:|---:|---:|
|proton / dense metric|1.421428799077574|7.87315e-9|63|.882386607914287|3.26593589131e-7|
|neutron / dense metric|1.421428799077573|9.84294e-9|58|1.844099299e-9|-2.13649057809e-7|
|proton / tree geometry|1.421428799077573|8.64731e-9|63|.882386607914229|3.26593589125e-7|

All have zero measured padding probability and meet the1e-8 GeV² executed
state-residual target. The dense proton F1/F2 errors against the identical
finite-model exact state are9.22e-13/1.03e-16; neutron errors are
6.64e-12/2.91e-15. Tree proton errors are9.81e-13/9.69e-17.
They lie below their independently evaluated state-residual/gap bounds.
Those bounds control solver error in this model, not physical theory error.

Relative to the preceding cutoff1 matched calculation, the exact-model
proton/neutron F2 values change by about-31.6%/-32.9%, despite a mass-squared
change of only about-1.61e-6 GeV². This is a concrete warning against using
mass stability as evidence of current or ultraviolet convergence.
The fixed-input comparison is not a refitted cutoff uncertainty band.

## Exact tree metric, with an explicit numerical regularization option

For the complete complex tree circuit, each new child has zero amplitude
before its first rotation. Let w_e be the parent's probability immediately
before edge e, p_v the final vertex probabilities, and S_e the descendant
subtree of the child. The state amplitude at v acquires the sum of all
edge phases on its root path. Direct differentiation gives

    G_theta,theta[e,f] = delta[e,f] w_e/4,
    G_theta,phi = 0,
    G_phi,phi[e,f] = P(S_e intersect S_f)-P(S_e)P(S_f).

With oriented edge/vertex incidence B[e,v]=delta[v,child]-delta[v,parent],

    G_phi,phi^-1 = B diag(1/p_v) B.T.

This is the exact interior Fubini-Study inverse in these coordinates,
not an approximation that drops covariance between branches.
Apply B.T, divide by p, then apply B: O(D) time and storage for this action.
The angle inverse is also diagonal. No dense state Jacobian, metric,
SVD, or eigensystem is needed. Independent dense Jacobians verify chains,
stars and branching trees, including recovery of the imaginary-time tangent.
An action-only32768-state test prohibits dense linear-algebra calls; it is
not a32768-state quantum execution.

At zero vertex probability the coordinate chart is singular. A caller must
explicitly choose either the exact interior inverse or a nonnegative epsilon
for the inverse preconditioner: w->w+epsilon and p->p+epsilon.
Positive epsilon is **not the exact metric**. It is declared optimization
regularization and never changes or discards an executed state amplitude.
The matched proton run above uses epsilon=1e-20; the dense control uses
its separately declared spectral metric cutoff1e-18.

The full tree still has O(D) parameters/gates and uses classical state
information. Removing dense geometry is a useful scaling advance, not
polynomial many-body compression, efficient shot-based metric estimation,
noise protection, or quantum advantage. The relation to natural-gradient
geometry follows [Stokes et al.](https://quantum-journal.org/papers/q-2020-05-25-269/);
the tree inverse above is the project's explicit derivation.

The separate retained-cutoff-1 scaling control now executes 204 states,
8 qubits and 406 parameters. With epsilon=1e-20 it reaches residual
9.38842366441e-9 GeV² in292 updates/844 evaluations (187.94s fit time),
with M²=1.419475028604877 GeV² and zero padding. Its electromagnetic
density is .882078010264720, differing from the independent exact state
by6.33e-15, below its1.07e-7 state-error bound. This is the **bare retained
pilot**, not the matched interacting-Q5/Q6 model in the preceding table.
No exact state seeds either calculation. The single-observable measurement
wrapper now returns a one-entry array consistently across supported QNode
return conventions; direct one- and multiple-observable checks pass.

## Verification scope

This increment adds43 direct tests:26 interacting-shell and17 tree-geometry
tests. The related68-test interaction/optimizer regression passed in131.77s.
After the single-observable wrapper correction,35 quantum/geometry tests
passed in11.44s. Separate complete invocations passed all8 higher-shell QQ
tests in296.41s,3 combined-H2 tests in492.01s and3 independent six-body
tests in510.38s. Identical raw waves are cached only within the QQ tests
to avoid repeating the same solve across H/norm/current assertions.
No assertion, source column, amplitude or physical tolerance is removed.

The M2 collection now contains977 tests; collection is not execution.
The last complete named M2+Q0--Q2 run remains907 at320ae968.
No current1010-test or full-repository pass is claimed.

## Reproduction, preservation, and next scientific work

    python -m pytest tests/test_m2_six_body_factorized.py tests/test_m2_five_body_factorized.py tests/test_m2_interacting_shell_six.py tests/test_m2_shell_canonical_qq.py tests/test_m2_interacting_shell_h2.py tests/test_m2_tree_projective_geometry.py -q
    python scripts/benchmark_m2_interacting_shell_six.py --cutoff 2
    python scripts/benchmark_m2_fourth_order_graph.py --shell-cutoff 2 --target PROTON
    python scripts/benchmark_m2_fourth_order_graph.py --shell-cutoff 2 --target NEUTRON
    python scripts/benchmark_m2_fourth_order_graph.py --shell-cutoff 2 --target PROTON --geometry tree --probability-regularization 1e-20

The separate solver-scaling control is explicitly the bare retained pilot,
not the interacting Q5/Q6 matched model:

    python scripts/benchmark_m2_tree_geometry.py --cutoff 1 --probability-regularization 1e-20

Pre-edit copies from d65de25e are in
computation_handoff/backups/2026-09-06_pre_canonical_self/:
pre_interacting_shell_six.tar, pre_interacting_shell_qq.tar,
pre_shell_graph_benchmark.tar, pre_tree_metric.tar and
pre_single_observable_measurement.tar.
The original22 volumes and C43/C57 sources remain unchanged.
No public push or review-mirror refresh is included.

Next derive the canonical transverse electromagnetic current from the same
constrained quark field, including its interaction-dependent contacts.
Continue common-regulator loop/current/counterterm matching, keeping
field traces distinct from external Galerkin truncation. Add the missing
physical transverse/non-Abelian and endpoint/zero-mode current partners,
then select shared physical conditions and withheld validation. Nuclear
spin-one and GTMD/Wilson-line construction remain subsequent dependencies.
In parallel, test larger exact-reference quantum problems and develop
genuine many-body/topology compression and measurement resources.
Physical calibration is unselected and all three workstreams remain open.

# M2 K-local H0 basis-map contract

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-03
Last updated: 2026-09-04

## Purpose

The existing C7/C8 H0/H1 validation spaces are not dimension-matched to the
C401/C410 coordinate spaces. Before any H0 can enter the main-line response
bundle, the source basis must be mapped explicitly into the K-local target
coordinates. This note defines the minimum contract implemented by
`microscopic/h0/basis_map.py`.

## Map orientation and required evidence

For a source H0 basis of dimension (n_s) and a C401/C410 target space of
dimension (n_t), the embedding (E_K) has shape (n_t\times n_s) and acts as

\[
|\psi\rangle_{\mathrm{target}}=E_K|\psi\rangle_{\mathrm{source}}.
\]

The contract requires:

- both K-local basis identities and both dimensions;
- one nonempty source-sector label for every source basis state;
- source and target units;
- an explicit omitted-sector treatment;
- a named Hermiticity test and at least one named commutator test;
- an explicit claim tier;
- a source certificate before a map can be labeled physical.

The map is not assumed to be an isometry. Its (E_K^\dagger E_K-I) residual
is reported, so a proposed map cannot hide a normalization defect. A source
operator can be embedded as (E_K A_K E_K^\dagger), but only after the source
operator and map are supplied explicitly.

## Current status

`microscopic/h0/k_local.py` now supplies one explicitly exploratory map
instance at K9, K11, and K13. It constructs the C47 intrinsic
`q_rel^2` sparse operator at M2 in the public C47 basis and applies an exact
label permutation into the current C401/C410 coordinate order,

\[
H_{0,K}^{\mathrm{expl}}=E_K Q_{\mathrm{rel},K}^2 E_K^\dagger.
\]

This is not a physical H0 map. It makes no C7/C8 dimension assumption, does
not reuse their 4/7/10-dimensional matrices, and does not select a deuteron
state, production current, fit, or activation.

C47 supplies the x-scaled basis, its normalization, exact CM projection, and
the diagonal `q_rel^2` functional. It does not supply the complete sparse H0
matrix. M2 assembles the radial off-diagonal HO recurrence. The recurrence is
directly cross-checked against the source-qualified C128 `pperp2` layer, but
the historical C128 fractions and numerical free matrix are not inputs to M2.

## Implemented operator and basis order

For fixed `m`, the nonzero intrinsic-HO matrix elements are

\[
\begin{aligned}
\langle n,m|q_{\rm rel}^2|n,m\rangle
 &=b_{\rm HO}^2(2n+|m|+1),\\
\langle n+1,m|q_{\rm rel}^2|n,m\rangle
 &=-b_{\rm HO}^2\sqrt{(n+1)(n+|m|+1)},
\end{aligned}
\]

with the Hermitian partner included. The direct sum is ordered as follows:

- `q` precedes `qg`;
- `q` is quark helicity `(-1,+1)`, then open-triplet color component
  `(0,1,2)`;
- `qg` is longitudinal partition, intrinsic-HO mode, quark helicity, gluon
  helicity, then open-triplet color component;
- the source HO sub-order is C47 shell-major `(2n+|m|), n, m`;
- the target keeps the historical C128 `n, m` sub-order used by C401/C410.

`E_K` maps identical complete labels and is a dimensionless square
permutation. The K9 instance has shape `1350 x 1350`, 1,350 map entries, and
zero isometry residual. The K11 and K13 shapes are `2706 x 2706` and
`4758 x 4758`.

The C47 convention is

\[
q_i=\frac{p_i}{\sqrt{x_i}},\qquad
Q=\sqrt{x_q}q_q+\sqrt{x_g}q_g,\qquad
q_{\rm rel}=\sqrt{x_g}q_q-\sqrt{x_q}q_g.
\]

Therefore \(Q=P_\perp\) and the CM-clean two-body kinetic combination is
\(\sum_i p_i^2/x_i-P_\perp^2=q_{\rm rel}^2\). The M2 matrix represents this
already-intrinsic quantity. It must not acquire another
`1/(x_q*x_g)` factor: that factor belongs to the defective historical C128
numerical free route, not to C47's x-scaled `q_rel` coordinate.

## Units and normalization ownership

| Factor | Owner and treatment |
| --- | --- |
| Kinetic operator | M2 sparse `q_rel^2` recurrence in C47's x-scaled basis; C47 supplies basis/CM/diagonal-functional provenance, not a sparse Hamiltonian matrix |
| Transverse scale | C45/C47 `b_HO` in `GeV`; the matrix carries `b_HO^2` |
| Finite cell | C43/C45 normalized modes; no residual `L` or `P^+` factor remains in this C47-coordinate/M2-recurrence `q_rel^2` term |
| Sparse recurrence and basis map | M2 recurrence is cross-checked against C128 `pperp2` only; the M2 permutation is exactly isometric and introduces no scale factor |
| Quark/gluon mass terms | C401/C396 `D_mu_q_sq` and `D_delta_mu_g_sq`; excluded from H0 and added exactly once by the bundle |
| C117 interaction | C411 owns its separate exploratory `P^-`-to-`M^2` conversion, residual normalization, mixing coefficient, and bundle coefficient |

The historical C128 source and document backups remain unchanged. Its basis
dimension and retained target ordering are used as coordinate metadata, but
its numerical free matrix is not reused. The C401 audit shows that the
historical partition reconstruction shifts `x_q` and affects the qg
transverse-kinetic denominator. Only C128's dimensionless HO
`pperp2` recurrence is cross-checked; M2 imports neither those fractions nor
any evaluated C128 free-matrix entry. This avoids C396 mass double counting
and avoids silently carrying the C128 longitudinal defect into the new kinetic
term.

## Sector support and omissions

The `q` block is exactly zero after the C47 exact CM-ground projection and
`P_perp` subtraction: a one-particle state has no intrinsic relative kinetic
motion. The `qg` block covers every current canonical longitudinal partition,
the retained CM-ground intrinsic-HO modes, both quark and gluon helicities,
and each open-triplet color component. There are no cross-sector matrix
elements in this free-kinetic term.

Higher Fock sectors, constrained zero modes, confinement, interactions, and
counterterms are `UNIMPLEMENTED_NOT_ZERO`. Charge/flavor, a color-singlet
completion, a deuteron `J^z` sector, full parity, and a physical state remain
unselected. The current-matching problem is outside this H0 result.

## K9 parameter-explicit eigenspace increment

The first state calculation uses the explicitly named, nonphysical
`M2_K9_EXPLORATORY_BASELINE_V1` point:

| Input | Value | Owner/status |
| --- | --- | --- |
| `D_mu_q_sq` coefficient | `0.20 GeV^2` | C401/C396 exploratory input |
| `D_delta_mu_g_sq` coefficient | `0.10 GeV^2` | C401/C396 exploratory input |
| C117 residual normalization | `0.50` | C411 exploratory input |
| C117 mixing coefficient | `0.80` | C411 exploratory input |
| C117 bundle coefficient | `0.07` | C411 exploratory input |

These values are a deliberately well-conditioned numerical point, not a fit,
physical matching condition, or parameter recommendation. The declared
one-at-a-time sensitivities are `+0.05 GeV^2` in `D_mu_q_sq`, `-0.05 GeV^2`
in `D_delta_mu_g_sq`, and `+0.03` in the C117 bundle coefficient.

The exact sparse K9 calculation finds a sixfold lowest eigenspace at
`0.194586374083865 GeV^2`, separated by
`0.412129729715685 GeV^2` from the next level. This corrected gap uses the
exact C47-to-target permutation for C117's qg block; the earlier
`0.421163695550323 GeV^2` value interpreted that block in its native source
order. The q-sector eigenvalue is unchanged because the map does not alter
the q block. It is the full q-sector
subspace: average q weight is one, qg weight is below `2e-30`, and the q--qg
block vanishes exactly. Its open labels contain three `J^z=-1/2` and three
`J^z=+1/2` components, with two open-triplet-color components per color. No
single vector is selected: M2 tracks the orthonormal invariant projector.
Across three Krylov seeds and tolerances `1e-10`/`1e-12`, the maximum energy
change is `1.11e-15 GeV^2`, residual is `2.36e-15`, and maximum principal
angle is `3.65e-8` radians.

Sparse, bundle matrix-free, and linear-operator actions agree to
`8.33e-17` on a deterministic probe. Subspace-averaged Hellmann--Feynman and
central finite-difference derivatives agree within `2.64e-11`; all three
projected derivative operators are branch-independent on this degenerate
space. The declared sensitivity shifts are `+0.0500000000000002 GeV^2`,
approximately zero, and `-0.00232012539262810 GeV^2`, respectively. They are
conditioning observations only.

The recovered Q0 codec embeds the six basis vectors in the 2,048-state,
11-qubit K9 register with exact compact round trips and zero padded leakage.
The Q0-encoding/Q1-style sparse-StatePrep echo reproduces the subspace-average
M2 total and the three direction expectations; its largest QNode--sparse
residual is `1.19e-18`. The frozen Q1 and Q2 public APIs remain explicitly
fixture-only, so M2 does not inject this external Hamiltonian into their C144
fixture or observable registries. No light-front/LPS current calculation is
started: the completed representation audit proves that the required
state-to-current composition map and finite-K current are not yet supplied.

## K9 state-to-current representation boundary

The requested interface audit is implemented in
`quantum/m2_state_current_boundary.py`.  Its domain is the basis-independent
six-dimensional projector range

\[
\operatorname{Ran}P_{K9}\subset\mathcal H_{M2,K9}=\mathbb C^{1350},
\qquad
\mathcal H_{M2,K9}=\mathcal H_q^{6}\oplus\mathcal H_{qg}^{1344}.
\]

The projector is the complete q block to numerical residual below `3e-15`.
Its labels are quark helicity `(-1,+1)` times open triplet color `(0,1,2)`;
it has open `J^z=(-1/2,+1/2)`, no target-helicity-zero component, and no
selected charge/flavor, color singlet, orbital/parity, nucleon composition,
or external transfer labels. Its amplitudes are dimensionless compact-basis
coordinates under the C47/M2 finite-basis inner product; the M2 Hamiltonian
has `GeV^2` units. This is not an external deuteron normalization.

C47 makes the color obstruction stronger than the earlier six-to-three
dimension observation. `q_basis` retains two open fundamental-color triples.
For each noncolor qg tuple, C47 retains the triplet output of its explicit
`U_(3<-3x8)=T^b/sqrt(C_F)` isometry. At K9 there are 448 such qg tuples.
The live M2 label permutation preserves those groups, hence

\[
\mathcal H_{M2,K9}=2\,\mathbf3\oplus448\,\mathbf3
=450\,\mathbf3.
\]

The direct source check verifies `U^dagger U=I`, that its image is in C47's
`3 x 8` triplet projector, and that it intertwines all eight product and
fundamental generators. The fundamental Casimir is `4/3`. An invariant vector
would have zero Casimir, so the retained M2 representation has no singlet:

\[
\operatorname{Hom}_{SU(3)}(\mathbf1,\mathcal H_{M2,K9})=0.
\]

Thus a nonzero color-singlet deuteron composition cannot land in the present
M2 space. The six-versus-three result rules out an isomorphism only; it does
not rule out an abstract `C^3 -> C^6` embedding. The color-intertwiner result,
not dimension counting, is the primary obstruction.

In contrast, the light-front adapter requires four spin-one target amplitudes
`(I++, I+0, I+-, I00)` normalized as `J+/(2P+)` in Drell--Yan kinematics,
and the LPS adapter requires a dimensional `(4,3,3)` current in component
order `(+,-,x,y)` and canonical target-spin order `(+1,0,-1)`. A lawful
finite-K passage first requires an enlarged many-body/hadronic color-singlet
space and source-qualified maps and operator

\[
C_{i/f}:\mathbb C^3_{\rm spin\text{-}1}\otimes\mathbf1_{\rm color}
\longrightarrow\mathcal H_{D,K},
\qquad J_{D,K}^\mu:\mathcal H_{D,K}\longrightarrow\mathcal H_{D,K},
\qquad J_{\rm target}^\mu=C_f^\dagger J_{D,K}^\mu C_i.
\]

Neither the color-singlet space `H_D,K`, its composition `C_i/f`, nor its
finite-K current exists in the current source-qualified repository. A
color-singlet `C_i/f` with codomain `H_M2,K9` is necessarily zero by the C47
decomposition above. Selecting a three-dimensional numerical image would be
neither a color-singlet map nor a source-qualified composition.
Although C405 has the same `q(6)+qg(1344)` direct-sum axis, its q diagonal
block is `UNAVAILABLE_NOT_ZERO_FOR_C117_I2`, its qg block is conditional and
incomplete, and C114 reports all four current products unavailable. Those
instantaneous-current topology structures are not a completed external
electromagnetic spin-one current.

Consequently neither `P_f J_K9^mu P_i` nor `(1/6) Tr(P_K9 J_K9^mu P_K9)` is
evaluated. If source-qualified later, these are colored-subsystem diagnostics
only, never deuteron target-current matrix elements. The projector is basis
invariant but cannot supply the missing current or target-spin indices. The
boundary test rotates the six solver vectors by an arbitrary unitary and
verifies that the projector, support facts, and obstruction are unchanged. It
never selects an eigenvector or calls either current adapter. The immediate
construction is first to introduce or bind a
source-qualified finite-K many-body/hadronic color-singlet Hilbert space
carrying spin-one deuteron composition, with initial/final momentum,
charge/flavor, Fock, orbital/parity, and normalization labels. Only afterward
may finite-K current intertwiners be defined. Missing information is not zero.

## Existing self-contained Q0/Q1/Q2 validation environment

`pyproject.toml` now declares `sympy` and `mpmath` with the pinned PennyLane
and Lightning pair in the `quantum` extra; `environment_quantum.yml` declares
Python 3.11 and the same symbolic dependencies. The maintained project-local
environment is exercised without user-site imports:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 deuteron_wigner_q0_plhqcd0/.venv311/bin/python -m pytest -q tests/test_q0_plhqcd0.py tests/test_q1_plhqcdstate.py tests/test_q2_plhqcdobs.py
```

Status: `SELF_CONTAINED_EXISTING_ENVIRONMENT_VALIDATED`.

The three suites pass `15/4/14` in clean processes with Python `3.11.15`,
NumPy `1.26.4`, SciPy `1.17.1`, SymPy `1.14.0`, mpmath `1.3.0`, PennyLane
`0.38.0`, and PennyLane-Lightning `0.38.0`; `site.ENABLE_USER_SITE` is false
and SymPy resolves inside that project-local environment. The frozen Q0/Q1/Q2
scientific APIs and fixtures remain unchanged. SymPy and mpmath were manually
seeded into this existing project-local environment from locally available
pure-Python installations after the package index was unreachable. Therefore
`FRESH_ENVIRONMENT_REBUILD_VERIFIED` is not claimed. A normal online rebuild
remains `conda env create -f environment_quantum.yml`; it installs
`.[analysis,quantum]` from the root declaration and needs no `sys.path.append`
workaround, but that clean reconstruction was not executed in this run.

## Focused evidence

The direct tests establish exact C47 public-label ordering and diagonal
functional; an all-K9-entry C128 `pperp2` recurrence cross-check with exact
raising/lowering radical arguments, orientation, Hermitian partner, and
forbidden selection rules while its free-matrix routes are poisoned;
exact mapped-versus-direct target equality; Hermiticity; five zero
commutators; positive finite qg blocks; zero q block; preservation and
non-use of the historical C128 numeric matrix; and an exact split between H0
and the two C401/C396 mass directions. The K9 target has 2,784 nonzeros and
minimum qg eigenvalue `0.05160763033910279 GeV^2`. The reproducible M2-named
regression is defined as every `tests/test_m2_*.py` file. It passes 130 tests:
five basis-map, 11 K-local H0, six K9 eigenspace, five
invariant-projector/current-boundary, ten color-bridge, eight C53-mixed-state,
ten spectator-composition, ten separately resolved NN-state, and ten
transfer/current tests, plus ten microscopic current-response, 15 active-quark
current/response, 15 resolved-spectator Hamiltonian/current/response tests,
and 15 M2j permutation-complete H2 tests.
The final relevant C47/C128/C401/C411/M2 regression passes 264 tests.

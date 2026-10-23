# Binding architecture preflight: spectrum, color geometry and cluster limits

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. Continuation from 6afc590f. **Necessary-limit screening,
not a selected confining action or a solved physical nucleon.**

## Outcome and direction

The original project requires one information-preserving construction for
protons/neutrons and modular nuclei. The recent source calculations qualify
canonical/retained operator pieces, not their nonperturbative binding law.
This preflight rules out three shortcuts before a fit:

1. A discrete free-quark level is not a physical binding threshold.
2. A mandatory confining connection between every quark in a many-nucleus
   network prevents color-singlet clusters from separating.
3. Different baryon connection drawings cannot be assigned independent
   diagonal energies in the existing quark color Hilbert space. Their states
   overlap. Independent gauge/flux degrees, if used, require an actual
   enlarged quantum-field state and its derived overlap, not an extra label.

A color-additive linear reference passes the simplest two-singlet energy
limit but has a resolved hidden-color contribution and an algebraic
large-distance tail. Projecting onto a single singlet-product expectation
would erase both. This does not identify that tail with nuclear physics or
justify deleting hidden color to make a preferred result.

**Carry forward matrix-valued shared dynamics in a physically defined
quark/gluon/collective carrier.** Use topology as actual transport, field
content and recoupling structure, not independent fitted energies assigned
to drawings. The already approved combination of explicit retained physics
and matched learned residuals remains in force; this does not reopen that
decision or select one production confinement mechanism.

## What was reviewed

- `references/volume_0_algebraic_geometric.tex`: state/mass-operator definition,
  distinct kinematic and dynamical symmetries, matched nuclear hierarchy and
  the distinction between a prediction and an independently fitted function.
- `references/volume_viii_symmetry_adapted_tensor_networks_prediction_compiler.tex`:
  physical state/operator/provenance separation. Its historical RSA--BLFQ
  example is superseded by the user's direction reset, not a current mandate.
- `docs/next_level/quantum_geometric_modular_architecture.md`: the common
  source functional is still a specification; classical gauge links, color
  singlets and a positive finite kernel do not establish binding or LF
  covariance. Its common Gram/state/operator principle is reused here.
- The joint-learning, retained-mode, quark-threshold and latest complete
  baryon-source results. Their numerical reference parameters are not
  physically matched quark masses, confinement or nucleon current kernels.

This is a focused scientific review, not a new full audit of all original
volumes. Earlier successful tools and failed controls remain unchanged.

## 1. Free LF threshold and finite sampling

For positive-plus free quarks, x_i=p_i^+/P^+, sum x_i=1. In intrinsic
transverse variables sum k_iT=0,

    M0² = sum_i (m_i²+k_iT²)/x_i >= (sum_i m_i)².

The bound follows from Cauchy--Schwarz; equality is at k_iT=0 and
x_i=m_i/sum m. An independent check sums the preserved Cartesian
four-vectors, including transverse total momentum and longitudinal boosts.
For the latest illustrative equal mass .27 GeV, the free three-quark
threshold is .81 GeV. The neutron mass is about .940 GeV ([PDG 2025 neutron
entry](https://pdg.lbl.gov/2025/AtomicNuclearProperties/neutron.html)). This
comparison is only a warning against identifying the reference spectrum
with physical binding. It is not a QCD constituent-mass condition.

If an assumed unconfined interaction leaves that free three-quark continuum
as an open asymptotic channel, an isolated ordinary bound pole must lie
below it. Merely fitting a higher mass cannot establish such a bound state.
Confining dynamics change the premise: colored free particles are not the
physical asymptotic states. Dressed propagators, thresholds and matching
must then be derived within that chosen realization. Bound states embedded
in a continuum require separate protection and are not silently assumed.
No spectrum or width of the full canonical QCD theory is inferred here.

The finite anti-periodic longitudinal grid uses odd positive n_i,
x_i=n_i/K2, sum n_i=K2. Exhaustive free minima are:

| K2 | Minimizing odd modes | Free minimum (GeV) |
|---:|---|---:|
| 3 | 1,1,1 | .8100000000 |
| 5 | 1,1,3 | .9222255689 |
| 7 | 1,3,3 | .9222255689 |
| 9 | 3,3,3 | .8100000000 |
| 11 | 3,3,5 | .8336546047 |
| 61 | 19,21,21 | .8109017537 |
| 121 | 39,41,41 | .8102251094 |

A hypothetical .9 GeV level would be below the K2=5 free minimum but above
the continuum threshold. That example is an inequality, not a fabricated
computed eigenstate. Grid refinement is not confinement; the nonmonotonic
grid error must not become a learned physical mass shift. This scan uses no
old fixed-K interaction, oscillator, Gaussian regulator or H2 construction.

## 2. The complete six-static-quark color carrier

Each of six distinct pinned slots carries a fundamental color. The raw
space has dimension 3^6=729. For each partition P=(a,b) into two triples,

    b_P(c1,...,c6) = epsilon(c_a1,c_a2,c_a3)
                     epsilon(c_b1,c_b2,c_b3)/6.

Taking the triple containing slot zero first gives ten partitions. Their
actual column matrix E has

    G=E†E,       spectrum(G)={0 five times, 2 five times},
    G_PP=1,      |G_PQ|=1/3 for P != Q.

The full singlet dimension is five, not ten. No small nonzero physical
direction is truncated: the rank deficiency is exact recoupling redundancy.
The SVD supplies an orthonormal isometry U into the raw carrier. Independent
Gell-Mann actions annihilate every global singlet and verify

    t_i.t_j = (P_ij-I/3)/2,       sum_(i<j) t_i.t_j = -4 I.

For the partition (012)(345), the first triple's color Casimir has spectrum
{0,3,3,3,3}. The other four directions are the complete octet-octet singlet
couplings for that partition, not extra topology states or optional noise.

These color tensors embed isometrically in a fixed-occupancy exterior space
with six distinct orbitals and a canonical slot order. The test does not
solve light-quark spin/flavor/orbital antisymmetry, local quantum Gauss law,
dynamical gauge fields or a physical baryon wavefunction. Global color
singletness must not be relabeled as those stronger properties.

## 3. Independent diagonal connection energies are incompatible

Suppose a network assigns each partition an energy d_P and asserts that
its color tensor is an eigenstate of one Hermitian H:

    H E = E D,       D=diag(d_P).

Multiplying by E† and taking the adjoint gives the necessary condition

    G D = D G,      (d_Q-d_P) G_PQ=0.

All distinct partitions overlap, so every d_P must be equal for this exact
diagonal-eigenstate interpretation. Geometry-dependent unequal values do
not define such an H. This is not cured by treating G as the identity or
whitening states while leaving diagonal labels unchanged.

At two equilateral unit triangles separated by R=4, assign the tempting
within-triple pair-length energy to each of the ten partitions. The
Frobenius norm ||GD-DG|| is **17.44254798**, not roundoff. A best Hermitian
least-squares operator can be constructed, but it does not obey the requested
ten eigenstate equations. Ordinary nonorthogonal-basis calculations remain
valid with independently defined full H_PQ and N_PQ; the no-go is specific
to the claimed independent eigenenergies, not to using recoupling channels.

Genuine gauge-field/flux excitations can enlarge the state space. Then derive
the full matter-and-field Gram matrix and the same operator/source matrix
elements. Static color recoupling alone neither supplies these excitations
nor proves them orthogonal. This is directly relevant to strategic DNN
node placement: matrix elements and field amplitudes are permitted unknowns;
incompatible independent eigenvalue labels are not.

## 4. Cluster separation and same-interaction geometry response

Lengths below are in units of a triangle side a, and energies in kappa*a.
Neither a nor kappa is fitted to QCD. For two parallel aligned equilateral
triangles with center separation R>=2a, the minimum connected spanning-tree
length is 4a+R, whereas two independent three-site trees have length 4a.
An energy proportional to the mandatory connected tree therefore has excess
kappa*R and a nonvanishing separating force. Its singlet/color labels do not
fix that physical cluster failure. A solver contraction tree is harmless
in this respect unless its edges are incorrectly made mandatory physical
confining interactions. Allowing junctions does not remove the cost of a
mandatory long inter-cluster connection.

The second control keeps all five color directions in the illustrative
linear color-pair operator

    V = -(3/2) kappa sum_(i<j) t_i.t_j |r_i-r_j|.

The product of isolated color singlets has expectation 6*kappa*a for every
separation. Full diagonalization does not give exactly that answer:

| R/a | Ground-energy shift / (kappa*a) | Hidden-color weight | Force / kappa |
|---:|---:|---:|---:|
| 2 | -.02245186057 | .00400410580 | -.03611736088 |
| 4 | -.002425851289 | .000172550049 | -.001929799287 |
| 8 | -.0002744160503 | 8.63467372e-6 | -.0001066296368 |
| 16 | -3.24211660e-5 | 4.79315509e-7 | -6.19747096e-6 |
| 32 | -3.93384801e-6 | 2.81854222e-8 | -3.72518051e-7 |
| 64 | -4.84286823e-7 | 1.70803061e-9 | -2.28173176e-8 |

The geometry response is -<psi|dV/dR|psi>; independent displaced-geometry
energies and full raw-color equations agree. It is a static force, **not an
EM current** or a Lorentz-covariant light-quark observable.

The tail can be derived, not just fitted. With a=kappa=1, C_A the first
triple Casimir, S=sum_(i=0)^2 t_i.t_(i+3), d=sqrt(R²+1),

    V = 6 I + (3/2)(d-1) C_A + (3/2)(d-R) S.

For the singlet product b, C_A b=0, <b|S|b>=0 and ||S b||²=1.
The large-R off-diagonal coupling is 3/(4R), the octet gap is 9R/2, giving

    Delta E = -1/(8 R³) + O(R^-4).

The numerical coefficient approaches this independently derived value.
This tail is a property of the declared static pair model. It is not a
matched nuclear interaction, and apparent small hidden-color probability
does not justify dropping its energy or source contribution. Physical
multi-gluon/hadronic forces can exist; the term "color van der Waals" is
not by itself proof of an artifact. Their range and matching must be assessed
in a specified theory, not rejected wholesale or accepted from this control.

## Matching, provenance and next experiment

The following are **necessary filters**, not a production model selection:

| Candidate shortcut | Result | Required alternative |
|---|---|---|
| Bound-state claim from one discrete free threshold | Grid-dependent false certificate | Derived asymptotic spectrum and resolution/matching controls |
| One mandatory confining tree across all quarks | Nonzero confining force between separated singlets | Cluster-compatible physical interaction, distinct from solver connectivity |
| Independent energies for overlapping color partitions | Fails Hermiticity on the actual carrier | Full common operator/Gram matrices, or derived additional gauge-field states |
| Single singlet-product diagonal of a color-pair potential | Misses resolved mixing and algebraic tail | Retain full color dynamics and qualify the induced inter-cluster interaction |

The standard distinction between dynamics, current construction and
relativistic cluster properties remains important. [Keister and Polyzou](https://arxiv.org/abs/1109.6575)
show that even a Poincare-invariant few-body construction can require separate
cluster corrections; we do not adopt or repair their model. [Takahashi et al.](https://arxiv.org/abs/hep-lat/0204011)
provide quenched static three-quark Wilson-loop evidence, not a light-quark
chiral/LF nucleon action or its EM current. A Y-shaped drawing is not enough
to transfer that static result to our p/n dynamics. These are prior-art and
limitation checks, not claims of novelty for color recoupling or string models.

The next bounded question is to **derive physically distinct gauge-path/
field states and their common overlap/operator map in the preserved canonical
quark/gluon carrier**, before introducing trainable connection energies.
Compare path recoupling with genuine added field content, retaining complex
interference and source partners. If an independent flux representation is
needed, specify its matching to the already retained gluons so they are not
counted twice. Start with a limited matrix-element/overlap construction;
do not silently introduce a new lattice/strong-coupling Hamiltonian or launch
a full confinement solve. Stop for a physical realization decision if the
required field content or vacuum cannot be qualified within that scope.

The minimal starting calculation is the first nonzero gauge-field order of
the candidate dressing/path operator in the existing normalized LF modes,
including every same-order vacuum, overlap and operator contribution.
Keep state-dressing paths distinct from the process-dependent Wilson paths
of GTMD operators. A finite gauge-fixed Fock calculation is not automatically
a solution of local Gauss law, and a first-order emission cloud is not a
nonperturbative flux vacuum or a confining state. This is an explicit scope
for testing whether proposed topology states have a physical field image,
not permission for an indefinite extra loop or Fock-space campaign.

For the eventual common binding action, declare the scheme and scale,
source-complete retained subtraction, chiral content and asymptotic spectrum.
Short-distance canonical amplitudes, protected soft channels, nucleon
state/current data and nuclear cluster conditions constrain different parts.
Static energies do not determine transverse magnetic currents; p/n fits do
not determine all independent connected nuclear interactions. Binding,
physical covariance, nuclear prediction and matched GTMD/TMD/GPD remain open.
Unknown coefficients permit exploration; none is silently fitted here.

## Validation and recovery

18 new direct tests pass with warnings as errors (0.50 s). The new tests plus
preserved modular-color, graded-Fock and canonical-LF tests pass **114 tests,
12 pre-existing warnings (5.82 s)**. A strict-warning run over the older
modules fails on their known NumPy determinant warnings; the new module is
warning-clean. Two initially arbitrary absolute roundoff checks were replaced
by matrix-norm-scaled machine-precision checks; no physical tolerance or old
test changed. The previous 618-test source checkpoint remains preserved,
not rerun or relabeled as this preflight's validation. No full-suite-green
claim or rendered-PDF check is made.

Final-source verification passes all 18 new tests plus six theory/hash checks
with warnings as errors: **24 passed (0.43 s)**. Adding those document checks
to the focused modular/graded/canonical selection gives **120 passed, 12
existing warnings (3.55 s)**. The final replay agrees with the saved results;
all eight archived originals match Git 6afc590f byte-for-byte. The new TeX
section has balanced braces/environments and unique labels. These document
checks do not certify every historical derivation or a physical model.

The final focused selection is `tests/test_binding_geometry_preflight.py`,
`tests/test_modular_gauge_network.py`, `tests/test_modular_baryon_algebra.py`,
`tests/test_graded_fock_network.py`, `tests/test_canonical_lf_network.py` and
`tests/test_current_theory_contract.py`, run with pytest -q -p no:cacheprovider.

- Source: `src/deuteron_wigner/quantum/binding_geometry_preflight.py`.
- Tests: `tests/test_binding_geometry_preflight.py`.
- Replay: `scripts/evaluate_binding_geometry_preflight.py` (prints JSON).
- Saved results: `docs/next_level/binding_geometry_preflight_results.json`.
- Originals: Git `6afc590f` and eight files in
  `computation_handoff/backups/2026-09-10_pre_binding_geometry_preflight.tar.gz`.

Use the compatible project runtime with PYTHONPATH=src:., bytecode disabled
and single-threaded BLAS. Current theory, handoffs and roadmaps are updated
locally. Earlier scientific sources, user files and public/review snapshots
remain intact. No production model, old-loop requantization or public push.

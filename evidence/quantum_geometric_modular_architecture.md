# Modular gauge-aware architecture: specification and executed controls

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

9 September 2026. Local continuation from `6c409c26`, under the user's
authorization to advance the physics-first, general-nucleus construction.

## Outcome and architectural decision

We now have executable building blocks for ordered SU(3) transport, a shared
nonlinear invariant functional and its link responses, fermionic proton/neutron
interpolators, variable-composition spin coupling, and a common overlap map
for composite states and operators. A further control distinguishes flexible
spatial interactions from positive cross-time kernels. These results settle
specific network-placement and composition questions. They do **not** yet
generate physical nucleon states or nuclear binding.

Carry forward an **expandable shared interaction/state/operator architecture**,
with exact representation couplers and explicit gauge paths surrounding
learnable invariant functions. Do not choose a valence-only or rigid-diquark
physical model merely because a compact interpolator is convenient. Do not
make a tree approximation to the physical graph by erasing its holonomy.
Direct observable fits remain useful baselines, not the state-generating
foundation. The previous Ward-only and spinful controls explain why a shared
propagator or conservation condition alone is insufficient.

This is a decision about the architecture's responsibilities, not a uniquely
identified soft interaction. The new controls do not select Euclidean lattice
QCD, resume BLFQ repairs, establish physical topological order, or demonstrate
quantum advantage. The smallest honest next physics target is a qualified
**state-generating nucleon interaction with its consistently derived currents**.

## Concrete pilot specification

The physical construction must generate proton and neutron structure, then
compose arbitrary nuclei. Deuteron is the first nuclear benchmark. A/Z and
target spin are variable; nucleus-specific variational states are allowed,
but unrelated per-nucleus or per-observable physical corrections are not.

| Physical location | Fixed structure / information | Learnable freedom and qualification |
| --- | --- | --- |
| Resolved field sites and momentum/orbital fibers | Quark, antiquark and gluon representations; flavor, statistics, spin, endpoints and scale | Expandable scalar coefficient functions and variational amplitudes; no silent removal of sea/gluon sectors |
| Coupling vertices | Color intertwiners, applicable spin/flavor couplers, fermion grading | Degeneracy-channel coefficients; exact labels do not establish dynamical rotations |
| Transport edges and closed paths | Ordered matrix transport, representation, endpoints and process labels | Couplings between permitted path/field structures; preserve relative non-Abelian orientation |
| Spatial interaction blocks | Real/Hermitian, gauge-compatible shared interaction terms | Flexible invariant neural functions; locality, long-distance/chiral content and QCD correspondence still need explicit qualification |
| Time/kinematic blocks | Positive/self-adjoint evolution where that formulation requires it | Restricted sufficient families or independently qualified alternatives; a real scalar network output is not enough |
| Composite nuclear interfaces | Antisymmetry, overlap/Pauli quotient, common state/operator matching, variable composition | Correlated many-body amplitudes and justified induced interactions; no independent-nucleon norm assumption |
| Probe insertions | Shared source dependence, spin/flavor/gauge paths, both endpoints | Consistently generated local and nonlocal operators, not independent fitted heads for each observable |

At a declared nucleon resolution \(\Lambda_h\), the intended schematic object
is a source-dependent effective action or equivalent interaction kernel,

\[
\Gamma_\theta[q,\bar q,A;a,\ldots]
=\Gamma_{\rm qualified\ short}[q,\bar q,A;a,\ldots]
+\sum_r f_{\theta r}(\mathcal I_1,\ldots)\,\mathcal O_r[q,\bar q,A;a,\ldots].
\]

Here \(a\) denotes external probes, \(\mathcal I\) invariant arguments and
\(\mathcal O_r\) explicitly qualified operator structures. This equation is
a **specification**, not an implemented or renormalized physical action.
The state equation, current insertions and matching must come from the same
choice. A Ward identity constrains longitudinal structure; it cannot supply
unidentified transverse magnetic physics. At nuclear resolution
\(\Lambda_N\), matching carries states, norm and all required one-/many-body
operators together. No numerical values of these physical scales, confinement
law, chiral truncation or finite matching coefficients are inferred from the
dimensionless graph experiments below.

The limited alternatives to pre-evaluate next are: an explicitly covariant
continuum shared-kernel realization, versus a gauge-compatible finite
Hamiltonian/network realization with a controlled continuum/light-front
connection. Either must specify short-distance correspondence, soft dynamics,
state normalization and transverse currents before a substantial solve. A
phenomenological observable-only fit is a comparison, not a replacement for
the shared-state objective. Do not make resolving every known prerequisite
of a borrowed method mandatory for our project.

Only the following pieces of this specification are implemented here: finite
classical SU(3) links; one invariant neural-feature functional; finite fermion
interpolators and spin couplers; exact small overlap maps; finite positive
kernels. Antiquark/gluon **dynamical states**, orbital wavefunctions, an
electromagnetic four-current and nuclear interactions are not implemented by
these files. The two hidden layers in the inference control are fixed, not
trained; the control is not yet a variational quantum DNN.

## 1. Ordered gauge transport and information retained by the graph

For a stored edge \((a,b)\), \(U_{ab}\) transports a fundamental column from
\(b\) to \(a\). Under independent local gauge transformations,

\[
U_{ab}\mapsto G_aU_{ab}G_b^\dagger,\qquad
U_{ba}=U_{ab}^\dagger,\qquad
U_P=U_{v_0v_1}\cdots U_{v_{n-1}v_n}.
\]

Thus \(U_P\mapsto G_{v_0}U_PG_{v_n}^\dagger\); normalized traces of closed
paths and epsilon contractions of three transported color columns are
gauge invariant. Neither contraction by itself supplies a baryon wavefunction.
Hermitian path reversal retains the process label. It is **not** the
future-/past-pointing staple change associated with a different process.

On a six-node, seven-edge graph, fixing five spanning-tree links to identity
preserves every Wilson loop when the two remaining link matrices are retained.
Erasing those remaining matrices changes the invariant functional and its
response. More subtly, two configurations have identical separate elementary
loop traces to \(1.83\times10^{-17}\), yet differ in

\[
\frac13\operatorname{Tr}(W_PW_QW_P^{-1}W_Q^{-1})
\]

by **0.02032736**. The shared functional differs by **0.01802141**. The
relative orientation of noncommuting holonomies matters. Separate loop traces
are therefore not generally an information-complete replacement for matrix
transport. This is gauge-path information, not a quantized topological-order
invariant or evidence that a drawing's node coordinates have physical meaning.

The independent native `Path` does not reuse historical `WilsonPathId.inverted`
semantics. That older metadata operation flips FUTURE/PAST while reversing a
path description; downstream eikonal controls depend on that convention.
It must not be treated as an oracle for geometric inverse transport. Those
historical files are preserved, not silently repaired in this continuation;
an explicit process/path adapter is required before combining the interfaces.

## 2. One neural functional owns its responses

With \(\operatorname{Tr}(t_at_b)=\delta_{ab}/2\), define

\[
z_\ell=1-\operatorname{ReTr}U_\ell/3,\quad
f_\theta(z)=\theta^T[h_2(z)-h_2(0)],\quad
F_\theta=\sum_\ell f_\theta(z_\ell),
\]

where \(h_1=\tanh(w_1z+b_1)\), \(h_2=\tanh(W_2h_1+b_2)\).
Four final coefficients are shared across all three loop structures and
all graph configurations. The link response is the actual derivative

\[
j_e^a=\left.\frac{dF_\theta[\ldots,e^{i\epsilon t_a}U_e,\ldots]}
{d\epsilon}\right|_0.
\]

Repeated and reversed occurrences of an edge all contribute. For
\(J_e=\sum_a j_e^at_a\), local invariance gives

\[
\sum_{e=(v,w)}J_e-\sum_{e=(w,v)}U_e^\dagger J_eU_e=0.
\]

Twelve independent local-gauge cases pass. All 56 link/generator derivatives
were compared with independently perturbed function values: maximum errors
are \(3.76\times10^{-8},3.76\times10^{-10},1.32\times10^{-11}\) for steps
\(10^{-3},10^{-4},10^{-5}\). Omitting one edge response gives a nonzero local
Noether residual. These are **classical color-response** checks, not
electromagnetic form factors, the quantum Gauss-law state constraint or
Lorentz/light-front covariance.

Synthetic inference uses ten development configurations, twelve reserved
configurations, four final coefficients and 100 noise realizations with
\(\sigma=10^{-5}\). Adding responses to functional values reduces reserved
response RMSE from \(5.47\times10^{-6}\) to \(7.97\times10^{-7}\); the design
condition number changes from 1482 to 162. However the comparison has
**10 versus 570 measurements**, not equal information/cost. Hidden features
and the truth family are fixed. This establishes conditional closure, not
physical identification, robust model selection or a quantum advantage.

## 3. Fermionic nucleon interpolators and variable-composition assembly

In an orthonormal finite quark-orbital carrier, use normalized interpolators

\[
B^\dagger_{ph}\propto\epsilon_{abc}\epsilon_{st}
u^\dagger_{as}d^\dagger_{bt}u^\dagger_{ch},\qquad
B^\dagger_{nh}\propto\epsilon_{abc}\epsilon_{st}
u^\dagger_{as}d^\dagger_{bt}d^\dagger_{ch}.
\]

Exterior algebra supplies all quark exchange signs. Local color generators
annihilate these vectors; internal spin and isospin are 1/2, with baryon
number one and charges one/zero. This scalar-ud interpolation choice does
**not** assert a physical pointlike diquark or a complete nucleon state.

Exact SU(2) Clebsch–Gordan products assemble arbitrary list lengths and
allowed coupling sequences. Executed cases include A=1, pn spin 0/1,
pnn spin 1/2, ppn spin 3/2 and ppnn spin 0. The largest control has 26,244
sparse occupation amplitudes. Norm, fermionic exchange and independently
applied internal-spin Casimirs agree to numerical roundoff. The ppn spin-3/2
case is intentionally generic, **not** a helium-3 ground-state prediction.
Fixed flavor composition does not generally mean a total-isospin eigenstate.
Internal SU(2) spin is not the complete orbital/Poincaré action.

An identical odd composite creator squares to zero. A separated-orbital
parallel-spin pn product has unit norm, but the same-orbital product has
norm **10/9**. This finite algebra result is not a medium-effect prediction.
It disproves treating overlapping composite modules as independently
normalized elementary nucleon modes.

## 4. Common Gram map: normalization and every operator together

Let \(E\) embed cluster coefficient columns in the full quark occupation
carrier. Compute, rather than assume,

\[
N=E^\dagger E=V_+\Lambda_+V_+^\dagger,\quad
W=V_+\Lambda_+^{-1/2},\quad U=EW,\quad U^\dagger U=1,
\]
\[
O_{\rm cl}=W^\dagger(E^\dagger OE)W,\qquad
\langle O\rangle_c=\frac{c^\dagger E^\dagger OEc}{c^\dagger Nc}.
\]

The plus subscript denotes the retained non-null range. For the exact
Pauli/redundancy controls the removed direction maps to zero in the full
carrier. A tolerance is explicit in code; near-small eigenvalues are not
automatically safe to discard, as the earlier operator-sensitivity study
already demonstrated. General matching between different representations or
scales must also carry induced operators and both endpoints; the present
same-carrier example does not supply that dynamics.

For both same-orbital pn/np orderings at spin 0 and 1, the two-column Gram
matrix has eigenvalues \(0,20/9\) and rank one. Its isometry closes to
\(2.23\times10^{-16}\). Charge is one after the **common** transport, versus
the unqualified Euclidean-coordinate value 10/9. A three-column/rank-two
case with a nontrivial orbital-charge operator agrees with direct full-carrier
evaluation to \(1.67\times10^{-15}\). This is an independently derived inner
product, **not** a fitted rescaling of a failed physical current.

## 5. Neural placement: positivity is a separate requirement

The previously defined gauge-invariant real function produces a symmetric,
positive-entry kernel \(K_{ij}=e^{-f(1-\operatorname{ReTr}U_iU_j^\dagger/3)}\).
It nevertheless has negative eigenvalues. Positive entries and symmetry do
not imply positive semidefiniteness.

| Group sample | Size | Unrestricted neural cross-time minimum eigenvalue | Positive-character kernel | Same neural spatial potential, dressed kernel |
| --- | ---: | ---: | ---: | ---: |
| Seed 81, preflight | 40 | -0.303412 | 0.054249 | 0.053446 |
| Seed 192, fresh | 32 | -0.196792 | 0.051194 | 0.050591 |
| Seed 293, fresh | 64 | -0.503801 | 0.024527 | 0.024417 |
| Seed 394, fresh | 96 | -0.859593 | 0.022624 | 0.022362 |

A sufficient construction for the positive column is

\[
K_{ij}=\exp\left[\sum_{R=3,6,8}a_R
\left(\frac{\operatorname{Re}\chi_R(U_iU_j^\dagger)}{d_R}-1\right)\right],
\qquad a_R\ge0.
\]

Each character kernel is a Gram kernel of representation matrices. Its real
part is real positive semidefinite. Nonnegative sums, Schur products and the
entrywise exponential power series preserve that property; the constant
subtraction contributes a positive overall scalar. This proves sufficiency
for arbitrary finite configurations, beyond the sampled eigenvalues. The
weights (0.5,0.2,0.1) are control inputs, not selected QCD couplings. This
family is sufficient, not necessary for every admissible physical kernel.

For any finite real spatial potential \(V_i\), including the **same** neural
function rejected in the cross-time role,

\[
T=DKD,\qquad D_{ii}=e^{-\Delta\tau V_i/2}
\]

is positive semidefinite. Code subtracts \(V_{\min}\) for numerical stability
and returns it: the unshifted transfer is
\(e^{-\Delta\tau V_{\min}}T_{\rm returned}\). An arbitrary real bounded
spatial potential is not ruled out by failure as a temporal kernel. Negative
neural coefficients are not categorically forbidden.

This is a finite group-kernel statement. Independent local Gauss projection,
fermion dynamics, spatial locality, a continuum limit and correspondence to
unitary light-front evolution still need construction. Kernel invariance
under common left/right multiplication is not arbitrary independent local
gauge invariance of a complete spacetime theory. A positive matrix alone is
not a proof of full reflection positivity, confinement, a nucleon pole or
light-front covariance. Thus the result constrains **where** flexibility
lives without committing us to a lattice-QCD research program.

## Reassessment and next scientific action

Advance from architectural algebra to **dynamical specification and an
uncompressed physical control**. The reusable maps above are sufficiently
defined; generating credible nucleon dynamics is not mechanically determined
by them. The next pre-evaluation must name the finite interaction, scales,
counterterm/matching assumptions, state equation and all current partners.
Prefer a small comparison that exposes a failed premise before a large fit.

1. Derive a bounded source-dependent nucleon interaction family with explicit
   short-distance, chiral/long-distance and transverse-current content.
   Check weak-field/symmetry and spectral/state-norm requirements before
   solving a putative nucleon. Estimate effort and expose any new unresolved
   regulator/continuum prerequisite before adopting that route.
2. Solve the declared uncompressed reference; require generated proton and
   neutron states, independently evaluated current components, symmetry and
   resolution controls. Do not identify interpolators with these states or
   fit away covariance discrepancies.
3. Place the qualified state/operator objects in the symmetry network;
   compare operator-sensitive refinement and classical/quantum cost on the
   same physics. Only then calibrate jointly, preserving a genuinely withheld
   prediction set and correlated uncertainty.
4. Match modular nuclear interactions/operators; qualify deuteron and then
   A=3/4 before broader nuclei. Develop nonlocal GTMD/TMD/GPD insertions,
   their matching/evolution and sum-rule/current limits on the same parent
   construction. None of the graph controls completes these physical steps.

Do not add another generic scalar/Yukawa or graph-polishing campaign. The
consequential remaining uncertainty is **which controlled soft dynamics can
supply a predictive common nucleon/nuclear model**, not missing bookkeeping.
There is no theorem that topology, the group labels or the neural optimizer
will identify this for us. A new physical commitment must be justified by
the above pre-evaluation; the user's authorization supports qualified
continuation, not silently treating an untested ansatz as QCD.

## Reproduction, preservation and scope

- Sources: `src/deuteron_wigner/quantum/modular_gauge_network.py` and
  `src/deuteron_wigner/quantum/modular_baryon_algebra.py`.
- Direct tests: `tests/test_modular_gauge_network.py` and
  `tests/test_modular_baryon_algebra.py`: **35 passed**. Combined regression
  with the five preceding exploration files and theory/handoff checks:
  **171 passed in 18.80 seconds**, with six non-failing SymPy/mpmath
  deprecation warnings. The full repository suite was not run; test counts
  are not evidence of physical completeness.
- Replay: `scripts/evaluate_modular_gauge_architecture.py --output NEW.json`;
  add `--continuation` for Gram/positivity results. Existing output files
  cannot be overwritten by this replay. Run with `PYTHONPATH=src:.`.
- Saved [first results](../snapshot/docs/next_level/quantum_geometric_modular_architecture_results.json)
  and [continuation results](../snapshot/docs/next_level/quantum_geometric_modular_gram_positivity_results.json)
  include seeds, parameters and limitations. First replay took about 0.60 s;
  this excludes development and is not a scaling estimate.
- Pre-edit source/document checkpoint: `6c409c266342d1ae5f557c3224f7b8556ea66ae1`.
  Six entry/roadmap/index originals are in
  `computation_handoff/backups/2026-09-09_pre_modular_gauge_architecture.tar.gz`;
  the pre-edit main TeX is separately in
  `computation_handoff/backups/2026-09-09_pre_modular_theory_note.tar.gz`.
  All seven originals were compared byte-for-byte with that Git checkpoint.
- The main theory note receives the governing scope/direction and a focused
  mathematical update. Its historical body and detailed later BLFQ addendum
  are **not** represented as a newly line-by-line re-audited complete theory.
  Original volumes, earlier numerical implementations and reports, unrelated
  user files and the public/review snapshot remain unchanged. No push.

## Established foundations and originality boundary

The group, exterior-algebra, Gram-map and positivity identities are established
mathematics. Our result is their explicit, tested placement in this project's
shared neural/state/operator and general-nucleus architecture, not a claim to
have invented symmetry networks, Wilson loops or gauge-invariant learning.

Relevant primary foundations are [Singh, Pfeifer and Vidal on symmetry tensor
decomposition](https://arxiv.org/abs/0907.2994), [Zohar and Burrello on locally
gauge-invariant PEPS](https://arxiv.org/abs/1511.08426), and [Favoni et al. on
lattice gauge-equivariant networks](https://arxiv.org/abs/2012.12901). Their
results do not by themselves prove our light-front dynamics or architectural
universality. [Kvinikhidze and Blankleider](https://arxiv.org/abs/nucl-th/9901001)
provide an established common-equation/current foundation, not an implemented
current here. [Kanwar and Wagman](https://arxiv.org/abs/2103.02602) discuss why
Euclidean transfer positivity and real-time unitarity cannot be interchanged
by an unqualified discretization; we use that as a limitation/control guide,
not as selection of their lattice action. No physical novelty or quantum
advantage is established by these finite experiments.

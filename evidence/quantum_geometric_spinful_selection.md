# Spinful architecture selection: shared dynamics, transverse information and inference

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

9 September 2026. Exploratory architecture evidence, not a nucleon prediction.

## Decision supported by the experiments

**Recommend putting the learnable physical freedom in a restricted, explicitly
declared shared interaction/kernel, with consistently generated states and
operators.** Use symmetry-compatible tensor/neural/quantum representations for
those objects; do not replace their dynamical connection with independent
observable fits or a propagator plus current conservation alone.

This is an architectural recommendation, not selection of the physical
interaction, a proof of global optimality, or authorization for an unbounded
new model-development campaign. The Yukawa example below is a completed
control, not the project's replacement nucleon model. No BLFQ repair was run.

| Construction level | Evidence in this investigation | High-level conclusion |
| --- | --- | --- |
| Direct covariant Dirac/Pauli functions | Algebraically admissible current structures; no independently generated state or interaction | Useful observable parametrization/control, insufficient by itself for our shared-state GTMD objective |
| Propagator plus Ward-constrained vertex | Same propagator, pole and charge admit arbitrary transverse Pauli response; 48 off-shell checks pass | Conservation does not determine magnetic physics; transverse information needs its own physical origin |
| Common local interaction, amplitudes and current | Independently computed covariant loop numerator and LF overlap agree; shared-coupling synthetic inference has reserved-point closure | A workable small example of the intended dependency direction, not a completed nonperturbative nucleon construction |

The first row is an algebraic comparison, not a separately trained competing
model. None of the three rows is a complete competing nucleon fit.

**What is now settled at this level:** covariance must be checked independently
of conservation; magnetic/spin-orbital information cannot be removed merely
because charge or state-based diagnostics look satisfactory; shared dynamics
can generate a consistent spinful current without component-specific repair;
and calibration must probe more than charge and a single response channel.

**What remains undecided:** the physically appropriate soft interaction,
degrees of freedom and matching scale; nonperturbative state/current
completion; useful many-body network connectivity; any foundational physical
topological mechanism; and comparative classical/quantum cost.

## Scope, provenance and original project connection

The pre-evaluation and fixed synthetic design were recorded in the
current handoff (`handoff/CURRENT_PROJECT_HANDOFF.md`, full upstream checkout) before the grid
and inference replay. The controlled action is a charged Dirac field coupled
locally to a neutral scalar, through order g², with equal internal/external
fermion mass M=m=1, positive scalar mass μ, and κ=g²/(16π²). Masses and momenta
below are in these units. This is a perturbatively dressed elementary fermion,
not a confined three-quark nucleon or a bound deuteron. No phenomenological
internal/external mass split is introduced.

The [original architecture](../snapshot/references/algebraic_geometric_next_level_model_note_revised.tex)
already requires the direction interaction → state and matched operator →
correlator → observables. Its network edges carry physical sector, spin,
orbital, color/flavor and scheme labels; its nodes represent physical maps.
This experiment tests that direction with a cheap spinful control. It does
not change the established information-preserving nucleon-to-nuclear design
into a campaign to finish a borrowed approximation.

The scalar-fermion LF numerators and overlap reference are established
physics, credited to [Brodsky, Hwang, Ma and Schmidt, section 4, equations
41–51](https://arxiv.org/abs/hep-th/0003082). The relevant PDF equations were
also visually checked. Their unscaled light-front convention is converted
to our symmetric convention; their more general mass choices are not adopted.
The [Ball–Chiu vertex construction](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.22.2542)
supplies the established Ward-identity framework for the separate freedom
witness. Our contribution here is the implemented comparison, independent
controls and resulting project-specific design constraints, not discovery
of Yukawa physics or of transverse-current freedom.

## Independent spinful positive control

Use metric (+---), p±=(p⁰±p³)/√2, q=p′−p, Q²=−q² and spinors with
ubar u=2m. Define σμν=i[γμ,γν]/2. Extract F1,F2 from

\[
J^\mu_{h'h}=\bar u_{h'}(p')
\left[\gamma^\mu F_1(Q^2)+\frac{i\sigma^{\mu\nu}q_\nu}{2m}F_2(Q^2)\right]u_h(p).
\]

The computed current is **not built by inserting fitted F1,F2** into this
formula. First integrate the full loop numerator, then decompose its spinor
matrix elements. With u=α+β, α=uv, β=u(1−v), r=αp′+βp,
Δ₀=u²+(1−u)μ² and Δ=Δ₀+αβQ², the finite on-shell matrix representative is

\[
\Gamma_R^\mu=\gamma^\mu+\kappa\int_0^1du\int_0^1dv\,u
\left[\frac{(\not p'-\not r+1)\gamma^\mu(\not p-\not r+1)}{\Delta}
-\gamma^\mu\left(\frac{(2-u)^2}{\Delta_0}+\log\frac{\Delta}{\Delta_0}\right)\right].
\]

The common on-shell charge subtraction cancels divergent γμ terms before
numerical integration; it is not a separate counterterm for each component.
This representative is used only between on-shell spinors and is not an
off-shell Ward-complete vertex. A separate Gordon reduction gives

\[
F_1=1+\kappa\int du\,dv\,u\left[(2-u)^2
\left(\frac1\Delta-\frac1{\Delta_0}\right)
-\frac{\alpha\beta Q^2}{\Delta}-\log\frac\Delta{\Delta_0}\right],\qquad
F_2=2\kappa\int du\,dv\,\frac{u^2(2-u)}\Delta.
\]

The third route integrates the LF wave amplitudes directly over x,k⊥,φ.
At g=1, rows are constituent helicity and columns parent helicity:

\[
\psi=-\frac{\sqrt{1-x}}{k_\perp^2+(1-x)^2+x\mu^2}
\begin{pmatrix}1+x&k_x-ik_y\\-(k_x+ik_y)&1+x\end{pmatrix}.
\]

Restore g at each amplitude. The outgoing argument is k′⊥=k⊥+(1−x)q⊥.
The diagonal overlap is charge-subtracted before radial integration; the
helicity-flip overlap gives F2 with factor −2m/(qx−iqy). The measure is
dx d²k⊥/(16π³). At Q²=0 use the analytic shift derivative for F2, not 0/0.
The amplitudes independently match the local scalar Dirac vertex divided
by its LF energy denominator; their angular weights are Lz=0,±1.

The replay covers μ=(0.35,0.8,1.6), Q²=(0.05,0.25,1,4), and three oblique
boosts with rotated transfer directions: **36 current/frame cases**. Each
extracts F1,F2 separately from plus, minus, transverse x/y and all components,
using the helicity matrix and checking the two-structure residual. Positive
loop and LF quadratures are independently refined (48→80 and 64→96).

| Numerical diagnostic | Largest observed absolute error/change |
| --- | ---: |
| Component-extracted F1/F2 versus invariant integrals | 1.67×10⁻¹⁵ |
| Direct LF overlap versus invariant integrals | 1.11×10⁻¹⁶ |
| On-shell current-conservation residual | 2.91×10⁻¹⁵ |
| Loop-current quadrature refinement | 1.39×10⁻¹⁵ |
| LF quadrature refinement | 3.99×10⁻¹⁷ |

These are smooth finite-control numerical errors, not physical uncertainties
or accuracy of a nucleon model. The independent exact anchor μ=m=1 gives
F2(0)=κ; both integral routes give 0.01 at κ=0.01. At μ=0.5,Q²=1,
(F1,F2)=(0.9970305460,0.0137367350).

**Important boundary:** all four components here come from the covariant
loop. The independently derived LF overlap supplies the plus current only.
We have not constructed all LF minus/transverse time orderings, instantaneous
terms or zero-mode contributions, nor repaired the old BLFQ current.

## Tests that distinguish success from imposed consistency

### Same propagator and conservation, different magnetic response

For s=p², define A(s)=1+a(s−1), B(s)=A(s)+c(s−1)² and
S⁻¹(p)=A(s) p-slash−B(s), with a=0.13,c=0.07. Its mass-one pole and residue
are unchanged. The nonsingular finite-difference vertex is

\[
\Gamma^\mu_{\rm BC}=
\frac{A(s')+A(s)}2\gamma^\mu+(p'+p)^\mu
\left[\frac a2(\not p'+\not p)-\{a+c(s'+s-2)\}\right].
\]

Adding λ iσμνqν/2 changes neither
qμΓμ=S⁻¹(p′)−S⁻¹(p) nor the propagator. Sixteen off-shell momentum pairs
at each λ=−0.03,0,0.03 give a maximum Ward residual 1.25×10⁻¹⁵, while the
bare vertex's maximum defect is 3.51. On shell F1=1 and F2=λ in all three
cases. The equal-virtuality limit is also tested.

This is a covariant mathematical non-uniqueness witness, not a claim that
these polynomial functions define a globally physical propagator. Learning
the propagator and enforcing the Ward identity cannot by itself identify
the transverse response. A common action must include any allowed transverse
interactions or matched coefficients; even that action still needs physical
specification and calibration.

### Conservation without covariance must fail the validation

Add the deliberately preferred-frame structure

\[
\delta\Gamma^\mu=0.02\frac{Q^2}{1+Q^2}
\left[n^\mu-q^\mu\frac{n\cdot q}{q^2}\right]\mathbf1,
\qquad n=(1,0,0,0).
\]

It is conserved and vanishes in the zero-transfer limit, but keeps a fixed
preferred vector as the external momenta change. Conservation remains below
8×10⁻¹⁶ while plus/transverse extracted form factors differ by as much as
0.00919, with nonzero two-structure fit residuals. The test therefore detects
a failure that a conservation-only loss would miss. This is an injected
negative fixture, not an attempted fit or explanation of the old discrepancy.

### Spin-orbital coherence and state normalization are separate obligations

Deleting both Lz=±1 off-diagonal wave amplitudes preserves the imposed charge
normalization but sets F2 to zero at every tested transfer. The full value at
Q²=1 is 0.0137367. Exact Jz labels alone do not preserve the dynamical weights
and interference of the allowed components. Combine this result with the
[scalar response audit](quantum_geometric_scalar_geometry.md): compression
must be judged on operators as well as state distance or probability.

The unsubtracted two-body norm is ultraviolet divergent. At μ=0.5,κ=0.01,
transverse cutoffs 2,8,32,128 give probabilities
0.02642,0.04097,0.05488,0.06875, respectively; the asymptotic increment for
a fourfold cutoff increase is κ log 4. Finite charge-renormalized form factors
do **not** provide a normalized continuum quantum state for a circuit.
Any state-based pilot needs an explicit finite resolution and consistent
state/operator reduction. This result is a warning, not a newly selected
renormalization research queue.

## Shared-interaction inference: what the measurements identify

Use three distinct neutral scalar fields of masses (0.35,0.8,1.6) with
synthetic shared κa=(0.006,0.004,0.003). Their orthogonal field sectors have
amplitudes proportional to ga; their loop corrections add through g². The
same three parameters generate both F1 and F2, without independent channel
coefficients. This is a deliberately fixed, known control family.

Development Q²=(0.1,0.3,0.8,1.6); reserved Q²=(0.18,0.55,1.2,2.7).
Run 100 fixed-seed Gaussian-noise realizations, σ=2×10⁻⁵ per datum, and
nonnegative least squares for Dirac-only and joint Dirac/Pauli data: 200 fits.
Both comparisons use identical F1 noise; the joint fit adds four F2 data.
They are **not equal-measurement-budget comparisons**. F1 is the Dirac form
factor, not the Sachs electric form factor GE=F1−Q²F2/(4m²).

| Conditional inference diagnostic | F1 only | Joint F1,F2 |
| --- | ---: | ---: |
| Linear response condition number | 129,735 | 743 |
| Reserved F1 root-mean-square error | 3.80×10⁻⁵ | 1.41×10⁻⁵ |
| Reserved F2 root-mean-square error | 1.60×10⁻³ | 1.63×10⁻⁵ |
| Third coupling root-mean-square error (true value 0.003) | 0.00734 | 0.00232 |

Charge alone has zero coupling-response rank because F1(0)=1 is fixed.
Both transfer-dependent designs have full algebraic rank and exact noiseless
closure, but F1-only inversion is poorly conditioned. Joint data improve
reserved magnetic-response error by about 98-fold in this fixed design;
the individual couplings nevertheless remain substantially uncertain.

This is within-family synthetic prediction at unused transfer points, not
prediction of a wholly withheld observable family: F2 data enter the joint
fit. It does not test model misspecification, real-data systematics, unknown
masses, higher-order terms, long-range dynamics or out-of-family truth.
Do not quote these errors as physical confidence intervals. The high-level
lesson is to design joint, complementary calibration and reserved tests,
not to infer that a successful toy fit identifies the physical Hamiltonian.

## Direction after this result

**Advance the shared-interaction architecture; stop generic control polishing.**
The scalar and spinful reference controls now answer the bounded selection
questions. More repetitions of them cannot identify the missing nucleon
dynamics or prove a topological mechanism.

The next useful step is a bounded **physical pilot specification**, evaluated
against the original project rather than an inherited model roadmap:

1. Declare the first nucleon-scale degrees of freedom, resolution and a small
   number of symmetry-allowed soft interaction hypotheses. Identify where
   long-distance, chiral and short-distance matching information enters.
   Keep the matched nucleon-to-deuteron hierarchy available; a direct six-quark
   solution is not made an automatic prerequisite.
2. For each candidate, state how the same interaction generates amplitudes,
   transverse current content and eventually nonlocal operators. The
   [gauging-of-equations method](https://arxiv.org/abs/nucl-th/9901001) is an
   established consistency foundation, not an implemented solver here and
   not a substitute for specifying transverse dynamics.
3. Define the symmetry-labeled state/operator network and physical partitions,
   keeping spin-orbital and sector coherence. Compare representations only
   against a controlled uncompressed reference. Quantum geometry belongs to
   that actual ansatz; a convenient optimizer tree does not define physics.
4. Set independent covariance/charge/limit controls, complementary calibration
   channels, genuinely reserved predictions and explicit cost/review bounds
   before a physical fit or nonperturbative solver is launched.

This is a specification/selection step, not automatic permission to implement
all candidate dynamics. A genuine choice about physical scope should be
presented with its assumptions and likely cost before a substantial commitment.

For foundational topology, first identify a physical connection, invariant
or interference mechanism and an observable that distinguishes it. The
original note already separates provenance topology from dynamics, and
nonzero gauge curvature generally makes Wilson transport depend on path
geometry, not just homotopy. The present neutral-scalar control contains no
nontrivial gauge-link topology. It cannot validate that part of the project;
neither its absence here nor a solver-tree success settles its usefulness.

## Reproduction, preservation and publication boundary

- [Implementation](../snapshot/src/deuteron_wigner/quantum/spinful_selection_controls.py)
- [Replay script](../snapshot/scripts/evaluate_spinful_selection.py)
- [Numerical results](../snapshot/docs/next_level/quantum_geometric_spinful_selection_results.json)
- [Focused tests](../snapshot/tests/test_spinful_selection_controls.py)

Run `PYTHONPATH=src:. python scripts/evaluate_spinful_selection.py --output NEW.json`
from the canonical repository with NumPy/SciPy installed. Output creation is
exclusive to avoid overwriting a saved run. The saved complete numerical
replay took 8.31 seconds locally; this excludes development and verification
and is not a production-cost or quantum-speed estimate.

The 26 new tests cover spinors and gamma algebra, direct vertex amplitudes,
component/frame extraction, charge, the analytic magnetic anchor, independent
quadrature refinement, reversal Hermiticity, transverse freedom, injected
covariance failure, cutoff dependence and synthetic closure. Passing negative
controls means detecting the intended defect, not validating that defective
construction. All **136 combined focused checks pass**: 26 new spinful,
104 prior scalar and six theory/handoff checks, in 17.88 seconds. No
full-repository or physical-claim pass is implied.

Pre-edit checkpoint: `6f3b5bc4108fbbec35654c965b4e56ad6cfc6502`.
The six original entry/status files are preserved in the
pre-experiment backup (`computation_handoff/backups/2026-09-09_pre_spinful_selection.tar.gz`, full upstream checkout).
All six originals were verified byte-for-byte against that Git checkpoint.
Prior scalar/BLFQ numerical sources and results, the main TeX, all original
volumes and the public review mirror are unchanged. The main TeX still
predates the direction reset; this linked result and current handoff carry
the new findings pending a deliberate consolidation. No public push,
production model, neural/quantum training or quantum-advantage claim is included.

# Quantum-geometric exploration: first state/current reference

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

9 September 2026. **Pilot selection and numerical feasibility completed;
network, learned dynamics and physical nucleon qualification not completed.**

## Decision and pre-evaluation

The question was which small example can test our proposed *joint* state and
operator representation without first requiring a new research campaign.
This advances the original information-preserving construction by giving it
a falsifiable reference. It does not choose a production Hamiltonian.

| Candidate | Useful evidence | Hidden cost or limitation | Decision |
| --- | --- | --- | --- |
| Prescribed Dirac/Pauli form factors | Spinor conventions and observable adapter | Covariance inserted into the answer; no independently generated composite state/current | Keep as an adapter test, not the primary control |
| Covariant constant-vertex two-scalar composite | Nontrivial wavefunction, independently evaluated triangle, and a known pair-current contribution | Spin zero; input bound-state mass; no realistic soft QCD dynamics | Select as the first reference only |
| Spin-half fermion/boson or Yukawa reference | Closer to nucleon spin and current structures | Regulator, counterterm and current-completion choices may become the project | Defer until a bounded independent spin reference is specified |

The scalar reference follows the constant-vertex construction and triangle
reduction of [Choi and Ji, hep-ph/9906225](https://arxiv.org/abs/hep-ph/9906225),
especially equations (12)--(14) and (20). Their model is not a Hamiltonian
solution of QCD. We specialize the struck-constituent triangle to equal masses
and one unit charge. Their nonsymmetric light-front component convention is
translated to ours below. The original PDF equations were inspected visually
as well as read in extracted text.

The more involved fermion reference remains plausible, not discredited:
[Karmanov, Mathiot and Smirnov](https://arxiv.org/abs/hep-th/0612163) explicitly
analyze regulator-dependent orientation structures and their removal in a
Yukawa calculation. That is a reason to check its prerequisites before making
it our first dependency, not a claim that these issues are universally insoluble.

**Originality boundary:** the control is borrowed and credited. The proposed
joint symmetry/geometry-aware representation, compression and shared soft
dynamics are our research questions; this calculation neither implements
them nor establishes their novelty. A literature control is an instrument
for evaluating that proposal, not a new mission to complete that literature.

## Exact specification of this control

Two distinguishable scalar constituents have mass m; the active one has unit
charge, the spectator is neutral. A constant Bethe--Salpeter vertex describes
a spin-zero composite with input mass 0<M<2m. This is a contact-kernel/triangle
reference, not an arbitrary fitted elastic form factor. It can be viewed as
the corresponding contact-ladder bound-state ansatz with its pole specified;
we do not compute a binding coupling or solve a mass eigenvalue problem here.
No dynamical photon loops or QCD gauge fields are included. The finite elastic
triangle does not imply that an unsubtracted contact binding equation is UV
finite. Only the state and one-photon current specified here are qualified.

Use the project's symmetric convention p^±=(p^0±p^3)/sqrt(2), so
p²=2p⁺p⁻−pT². Define q=P'−P and q²=−Q². The spin-zero current has
J^μ=(P'+P)^μ F(Q²). Every numerical input below uses m=1, M meaning M/m,
and Q² meaning Q²/m². No GeV values or experimental masses are selected.

The covariant triangle has spectator denominator D0=k²−m²+i0, active
denominators D1=(P−k)²−m²+i0 and D2=(P'−k)²−m²+i0, and numerator
(P+P'−2k)^μ. Feynman parameters a,b,c multiply D0,D1,D2, with a+b+c=1.
After the shift k→ell+bP+cP', the odd term integrates to zero; b↔c symmetry
leaves a(P+P')^μ. Thus, with the overall vertex strength fixed ONCE by charge,

\[
 F(Q^2)=\frac{I(Q^2)}{I_0},\qquad
 I(Q^2)=\int_0^1 db\int_0^{1-b}dc\,
 \frac{a}{1-M^2a(1-a)+Q^2bc},\quad a=1-b-c,
\]
\[
 I_0=I(0)=\int_0^1 dx\,\frac{x(1-x)}{A(x)},\qquad
 A(x)=1-M^2x(1-x)>0.
\]

The dimensional I has units m^-2; its ratio is dimensionless. The covariant
route integrates the b,c simplex. An independent adaptive one-dimensional
integral fixes I0 for ALL routes, so a failed numerical normalization cannot
be hidden by renormalizing each computed curve.

For active longitudinal fraction x, the normalized light-front wavefunction is

\[
 \psi(x,\mathbf k)=\frac{4\pi}{\sqrt{I_0}}\,
 \frac{\sqrt{x(1-x)}}{\mathbf k^2+A(x)},\qquad
 \int_0^1\frac{dx\,d^2\mathbf k}{16\pi^3}|\psi|^2=1.
\]

At q⁺=0, the plus-current form factor is the overlap

\[
 F_{\rm DY}(Q^2)=\int_0^1\frac{dx\,d^2\mathbf k}{16\pi^3}
 \psi^*(x,\mathbf k+(1-x)\mathbf q)\psi(x,\mathbf k).
\]

We evaluate x, radial momentum and azimuth directly; neither the angular
integral nor a Feynman-parameter reduction is reused from the covariant
route. The map r=λt/(1−t), 0<t<1, covers the infinite transverse domain.
Changing λ tests numerical mapping, not a change of physical cutoff.

The same model gives a finite radius

\[
 m^2\langle r^2\rangle=-6\frac{dF}{d(Q^2/m^2)}\bigg|_0
 =\frac{1}{I_0}\int_0^1dx\,
 \frac{x(1-x)^3}{A(x)^2}.
\]

In contrast, psi falls as 1/kT², so its unregulated transverse second moment
diverges logarithmically. This reference cannot qualify arbitrary moments,
TMD evolution or ultraviolet QCD matching just because F and its radius exist.

## Second frame and a meaningful negative control

Choose qT=0, P'⁺=βP⁺ with 0<β≤1. On-shell kinematics fixes
Q²=M²(1−β)²/β and P'⁻=M²/(2βP⁺). Write y=k⁺/P⁺ for the SPECTATOR fraction,
s=kT² and tau=2P⁺k⁻. In m=1 units the denominators become

\[
 D_0=y\tau-(s+1),\quad
 D_1=(1-y)(M^2-\tau)-(s+1),\quad
 D_2=(\beta-y)(M^2/\beta-\tau)-(s+1).
\]

The plus current has two contour intervals. At 0<y<β use the D0 pole
tau0=(s+1)/y; at β<y<1 use the D1 pole tau1=M²−(s+1)/(1−y), with the
opposite contour orientation. After the transverse angle integral,

\[
 F_{\rm L}=\frac{V+N}{(1+\beta)I_0},\qquad
 V=\int_0^\beta dy\int_0^\infty ds\,
 \frac{1+\beta-2y}{y D_1(\tau_0)D_2(\tau_0)},
\]
\[
 N=\int_\beta^1 dy\int_0^\infty ds\,
 \frac{1+\beta-2y}{(1-y)D_0(\tau_1)D_2(\tau_1)}.
\]

These are signed CURRENT contributions, not Fock-sector probabilities.
The implementation uses algebraically reduced nonsingular forms and the
independent radial map s=λ²t/(1−t). At β=1 the pair interval vanishes.
For β<1, dropping N is the deliberately incomplete negative control.
Correct charge at Q²=0 therefore does not make the omitted-pair curve valid.

During the initial implementation comparison, this independent check caught
a missing 1/β on the final-state mass term in the valence denominator. Restoring
the on-shell expression above removed the discrepancy; no parameter, current
normalization or reference value was fitted. The regression tests cover it.

This compares **J⁺ in two different frames** with the covariant reference.
We have not separately computed J⁻/JT, tested spin-half angular conditions,
or restored the old nucleon's plus/transverse covariance. Reconstructing those
components from the covariant F would not constitute such an independent test.

## Executed numerical evidence

[Replay data](../snapshot/docs/next_level/quantum_geometric_scalar_pilot_results.json) cover M/m=1,1.5,1.9
and Q²/m²=0,0.25,1,4, at quadrature orders 32,64,128; the direct overlap uses
twice that azimuth order. The target was absolute F error below 1e-8 in this
declared grid, with order refinement and radial maps λ=0.5,1,2. Threshold
M→2m and arbitrary large transfers are outside the verified numerical domain.

| Check | Measured maximum |
| --- | ---: |
| Direct DY overlap versus covariant integral, order 128 | 1.05e-14 absolute F |
| Longitudinal valence+pair versus covariant integral, order 128 | 7.22e-15 absolute F |
| Order 64→128 change across all three routes | 9.33e-15 absolute F |
| Changed infinite-domain radial maps | 4.19e-14 absolute F |
| Radius integral versus second-order forward derivative | 3.35e-10 in m²⟨r²⟩ |
| Largest relative error when pair current is omitted | 53.65% |

Representative point M/m=1.5, Q²/m²=1:

- Covariant and DY F: **0.919522367732**.
- Longitudinal valence: **0.804740062780**.
- Longitudinal pair: **0.114782304952**.
- The sum agrees; valence alone is about **12.48%** low.

Charge normalization and positive decreasing spacelike F pass on the grid.
The complete numerical replay took about 0.27 seconds excluding imports on
this host. These are floating-point consistency/refinement measurements,
not certified error bounds or physical uncertainty estimates. The model's
spin, interaction and operator restrictions dominate its applicability.
Verification also passed 34 focused scalar-reference tests plus six existing
theory/handoff checks. The full repository suite was not rerun; the unchanged
BLFQ kernels were not requalified by this isolated reference test.

## Result-driven reassessment and next bounded action

**Advance the control to representation work.** An accessible state/current
reference and an actual missing-current failure are now available. The pilot
selection and feasibility question is answered within its original bounded
estimate. There is no scientific reason to restart BLFQ regulator repairs to
perform the next representation experiment.

**Keep the control narrow.** The scalar contact interaction is NOT selected
as the missing nucleon soft dynamics. A positive real spin-zero wavefunction
cannot demonstrate nontrivial physical Berry topology, spin multiplets,
fermion statistics or local color/gauge constraints. A rank split between x
and radial momentum measures coordinate correlation, not automatically
particle entanglement. Do not conceal these limitations by using qubit labels.

**Next proposed bounded question (exploration 2, feeding 3):** can a small
joint state/operator representation preserve this reference's norm, elastic
overlaps and the separately identified pair-current contribution under
resolution changes and a first rank reduction? First derive the common
quadrature weights, representation maps and operator action; keep an
uncompressed reference. The pair contribution must remain an explicit
transition/current object, not an independently fitted output or a discarded
low-probability sector. Do not paste the exact form factor into the network
and call its covariance a learned result.

Before that step, compare an uncompressed complex representation with a
minimal x/radial rank decomposition; specify how shifted momenta and the
nonvalence interval are evaluated and what phases the representation can
retain. Use the present toy's exact axial structure without pretending it
implements non-Abelian intertwiners. Numerical grid error and rank error
must be measured separately. No quantum hardware or DNN training is needed
to decide this question.

Provisional pilot targets, to be reviewed before implementation: 1e-8
reference discretization error, 1e-4 absolute F error and 1e-3 state L2 error
after compression. A state-norm bound alone does not bound every current;
test the currents themselves. Use the evaluated grid for development and
reserve Q²/m²=0.0625,0.5625,2.25 and M/m=1.7 for interpolation/representation
validation. These are unexecuted proposed holdouts, not experimental data.
There is no fitted dataset, trainable interaction or calibrated physical
parameter in the present result.

Estimate one small implementation/review cycle, with uncertainty about the
operator representation. If the pair-current map requires a new dynamical
model, re-scope before building it. If compression is ineffective, retain the
full reference and reconsider the partition; do not relax observable accuracy
or remove the pair term. Reassess after the first actual accuracy/cost curve.
A spin-half control is still required before nucleon spin claims, but an
open-ended spin/current completion campaign is not a prerequisite for this
bounded scalar representation test.

## Reuse, replay and preservation

Use [the new isolated reference module](../snapshot/src/deuteron_wigner/quantum/scalar_composite_reference.py),
[replay script](../snapshot/scripts/evaluate_scalar_composite_reference.py) and
[direct tests](../snapshot/tests/test_scalar_composite_reference.py). With the normal
project Python environment and PYTHONPATH=src, run the replay script to print
JSON; its optional --output writes a new file and refuses to overwrite one.

Existing complex-state/current readout and full-state tree routines are
potential adapters, not dependencies of this continuum reference. The old
tree metric requires its original full-carrier parametrization and must not
be reused as the metric of a compressed tensor ansatz without derivation.
The old color/permutation tables have no direct scalar-domain role. Preserve
the earlier free-nucleon and regulator failures as separate controls; the
new negative control does not diagnose their magnitude or cause.

The original six pickup documents are preserved at Git checkpoint
8f9e396794deeb3df2ed0a409090a99d301d7885 and in
the pre-pilot archive (`computation_handoff/backups/2026-09-09_pre_quantum_geometric_pilot.tar.gz`, full upstream checkout).
No existing scientific kernel, saved BLFQ array or original theory source
was modified. This is a local exploratory addendum: the main TeX, review
mirror and public GitHub are not updated or represented as current by it.

# Quantum pair thresholds, shared continuum and linked source response

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. Continuation from `8c0c8524`. This is an actual
nonzero-momentum one-quark-loop calculation, not another tree Ward test or
a spacetime reinterpretation of the homogeneous quantum cell. Independent
spinor phase-space, LF phase-space, covariant traces, complex loop boundary
and dispersive evaluations agree. The unchanged tree reference contains a
heavy scalar mode for which a narrow, weakly corrected pole is unjustified.

This result informs the shared quantum/network architecture; it does **not**
select a production sigma model, confine quarks, calculate physical hadron
widths, or supply a complete quantum electromagnetic vertex or renormalized
action. The user-approved retained-modes-counted-once interpretation stays
in force. The cut must not be added as an independent interaction beside a
solver that already generates exactly that retained quark fluctuation.

## Scientific question and calculation

The last tree comparison left the collective modes as real poles. Their
coupling to retained quarks opens a continuum once s>4M². Its absorptive
part is finite and fixed by the actual quark vertices; unknown local finite
counterterms cannot remove a nonanalytic branch cut. This is a useful
quantum matching test before selecting a subtraction prescription or
enlarging the state solver. Neutral-scalar fermion form factors and the
earlier NJL gap/axial reference are preserved, not repeated or adopted as
the production path.

Take Nc=3, Nf=2 and the declared free-quark mass M=.2843478671 GeV from
the tree reference. It is not a newly determined quantum mass or the small
Lagrangian mass source .004 GeV. The continuum threshold is
2M=.5686957343 GeV. It belongs to this **unconfined reference**, not an
observed hadronic threshold. It cannot be relabeled as a physical continuum
of asymptotic colored particles in QCD.

We retain 20 bilinears with open Dirac/flavor/color indices:
S=qbar q, P^a=qbar i gamma5 tau^a q, all four components of each
A_q^{a,mu}=qbar gamma^mu gamma5 tau^a/2 q, and V_Q^mu=qbar gamma^mu Q q.
The order is S,P1..3,A1[0..3],A2,A3,V_Q[0..3]. Each pair-creation
amplitude has 12 quark x 12 antiquark states. With overall current/source
couplings stripped, define the Hermitian cut matrix

\[
 \mathcal A_{IJ}(q)=\frac12\int d\Phi_2(q;p,\bar p)
   \sum_{h,f,c}\mathcal M_I(p,\bar p)\mathcal M_J(p,\bar p)^*.
\]

It is Im of the current correlator and minus Im of the induced collective
inverse-propagator term. It has units GeV². This optical-theorem convention
is explicit; it differs by a factor pi from a spectral density defined as
Im Pi/pi. The phase-space measure is

\[
 d\Phi_2=\frac{\beta\,d\cos\theta\,d\varphi}{32\pi^2}
 =\frac{dx\,d\varphi}{16\pi^2},\quad
 \beta=\sqrt{1-4M^2/s},\quad
 x=\frac{1+\beta\cos\theta}{2},\quad
 k_{T,\mathrm{rel}}^2=sx(1-x)-M^2.
\]

The first numerical integral boosts rest-frame momenta. The independent
LF integral constructs positive plus momenta xq+,(1-x)q+ and their on-shell
minus components directly. Both retain canonical quark/antiquark spinors
and all color/flavor phases. The third calculation is an independently
reduced covariant trace. Instantaneous terms have no propagating two-body
cut; that does not remove them from uncut loops or local/source partners.

## Source identities, not a diagonal-rate approximation

Independent traces give

\[
 \mathcal A_{SS}=\frac{N_c s\beta^3}{4\pi},\qquad
 \mathcal A_{P^aP^b}=\delta_{ab}\frac{N_c s\beta}{4\pi},
\]

\[
 \mathcal A_{A_q^{a,\mu}A_q^{b,\nu}}=\delta_{ab}\left[
 \frac{N_c\beta^3}{24\pi}(q^\mu q^\nu-sg^{\mu\nu})+
 \frac{N_cM^2\beta}{4\pi s}q^\mu q^\nu\right],
\]

\[
 \mathcal A_{P^a A_q^{b,\mu}}=
 i\delta_{ab}\frac{M q^\mu}{s}\mathcal A_{P^aP^a},\qquad
 \mathcal A_{V_Q^\mu V_Q^\nu}=
 \frac{N_c(5/9)\beta}{12\pi}(1+2M^2/s)
 (q^\mu q^\nu-sg^{\mu\nu}).
\]

The vector contraction vanishes. The free-quark subdiagram satisfies
q_mu M_{A_q^{a,mu}}=-i M M_{P^a}, including its phase. This is **not**
the full interacting axial Ward identity/PCAC with the small current mass:
the collective Noether current and source partners must also be assembled
there. Positive scalar/vector/axial source spectra do not justify imposing
positive spectral density on a colored quark or gluon propagator.

Numerically the 20x20 spectral matrix has its expected positive-semidefinite
Gram structure. The vector null direction and three quark axial relations
are preserved. Deleting P--A cross terms keeps every diagonal rate and
positivity but breaks the axial source relation. A scalar mass insertion
also differentiates the phase-space threshold/measure, not just spinor
vertices: at fixed s, dA_SS/dM=-3Nc M beta/pi and
dA_PP/dM=-Nc M/(pi beta). Independent finite differences of the LF integral
verify both away from threshold. This is a zero-momentum mass insertion
in the free-quark loop, not an already matched physical scalar current.

## Common continuum and nonlocal quantum matching information

For either collective channel X, the two multiplets have

\[
 -\operatorname{Im}\Delta K_{X,ij}(s+i0)
 =g_i g_j\mathcal A_X(s).
\]

This is a shared rank-one quark-pair cut in multiplet space, not two
independent widths. The direction orthogonal to g is dark for **this
particular cut**; it is not declared decoupled from all dynamics. Diagonal
replacement destroys that property. The state/source network should keep
the common resolved pair continuum and interference, rather than assign
independent decay/source heads to each collective node. This is a concrete
physics constraint on network connectivity, not a topology/quantum-advantage
claim or a theorem that a particular optimizer is best.

The UV-finite twice-subtracted nonlocal inverse term is independently
computed from the cut and Feynman parameters:

\[
 \widehat{\Delta K}_X(z)=
 -\frac{z^2}{\pi}\int_{4M^2}^{\infty}
 \frac{\mathcal A_X(s)}{s^2(s-z)}ds,
 \qquad z<4M^2.
\]

For B(z)=-int_0^1 dx log[1-x(1-x)z/M²]/(16pi²),

\[
 \widehat{\Delta K}_P(z)=-4N_c z B(z),\qquad
 \widehat{\Delta K}_S(z)=-4N_c[(z-4M^2)B(z)+4M^2zB'(0)].
\]

These are unit-Yukawa two-point pieces; multiply by g_i g_j for the mode
matrix. The definition subtracts value and slope, **not** an instruction
that physical local mass/kinetic counterterms vanish. Independent S/P
subtractions do not themselves specify a chirally/source-complete
renormalized action. Finite local matching, its source derivatives, gluon
and collective loops, and the interacting vacuum remain open.

The beta-variable dispersive integral and x-variable Feynman integral agree
to 8.9e-15 relative at the recorded spacelike/below-threshold points.
Above threshold, a direct complex log integral at s+i epsilon reproduces
the cut sign/normalization; epsilon is a numerical boundary approach, not
a physical width. Near-root subdivisions resolve the logarithms; the new
tests also pass with numerical warnings treated as errors. A split of the
dispersion quadrature reconstructs the full result, but is not silently
identified with a physical Wilsonian/Fock partition.

## What changed scientifically

Rest and LF spinor cut matrices agree with the independent traces within
5.5e-16 relative; resolution and four-component source boost checks pass.
The [full replay](../snapshot/docs/next_level/quantum_pair_threshold_results.json) records the source
identities, phase-space negatives and nonlocal terms. At the original tree
couplings (3,1), projected leading quark-pair Gamma/M values are:

| Tree mode | Mass (GeV) | Leading Gamma/M | Interpretation |
|---|---:|---:|---|
| Scalar light | .576184 | .002893 | Barely above the reference pair threshold |
| Scalar heavy | .913303 | .809723 | A narrow/small-width assumption is unjustified |
| Pion light | .139570 | 0 | Below this quark-pair threshold only |
| Pion heavy | .526289 | 0 | Below this quark-pair threshold only |

These are leading **diagnostics**, not solved complex poles, physical
resonance widths, or proof that closed modes have no other decay channels.
The heavy value cannot be advertised as a reliable narrow-width prediction.

The allowed direction g -> g+eta*(-v2,v1)/|v| preserves the old tree mass
and, to numerical accuracy, its aligned light-pion coupling. At
eta=3.54423 the heavy-scalar projection on this cut vanishes, with
g=(.998775,3.925183). This is an unfitted sensitivity witness, not a selected
repair. The light scalar's Gamma/M becomes .01625. Meanwhile the norm of
the twice-subtracted pion piece relative to the chosen tree inverse at
Q²=1 GeV² grows from .778 to 1.182. Unknown local matching coefficients
are not included in that comparison, so it is neither a full loop-error
bound nor a physical instability prediction. Suppressing one width is not
enough to establish a controlled shared quantum model.

## Reassessment and next bounded experiment

The limitation of the old narrow-pole picture is now explicit. The safe
continuation is to preserve the shared continuum and source interference,
not erase the cut, fit an artifact away, or declare this unconfined
reference to be physical QCD. No new executive choice is needed for the
next parameterized matching experiment, and no production realization is
selected by the coupling sensitivity. The persistent goal remains active.

Next pre-evaluate the **one-quark-loop charged-collective EM triangle with
its linked inverse-propagator derivative**, using the protected light mode
below the pair threshold and spacelike photon transfer. Derive the
charge-subtracted, nonlocal transverse response from the common loop, not
from the two-point Ward contraction. Keep both photon attachments, actual
flavor charges, mode mixing and source normalization. Compare independent
parameter/trace or LF controls where available, then determine the retained
loop contribution that must be subtracted from any previously tree-fitted
local C coefficient. Do not immediately refit C or claim full quantum
matching; other loops, local source partners and a production action remain.
This adds missing transverse quantum information without repeating the
neutral-scalar fermion form-factor test or launching a sigma/NJL program.
Pre-evaluate cost/order/domain and reassess if the calculation requires an
uncontrolled resummation or a new physical model/renormalization decision.

The subsequent physical sequence remains joint quantum state/interaction
matching, generated p/n states and currents with withheld covariance/data,
modular nuclei, and GTMD/TMD/GPD matching/evolution.

## Reproduce and recover

Source `src/deuteron_wigner/quantum/quantum_pair_threshold.py`; direct tests
`tests/test_quantum_pair_threshold.py` (28); replay
`scripts/evaluate_quantum_pair_threshold.py` prints JSON and writes nothing.
Use the established Python environment, `PYTHONPATH=src:.`, disabled bytecode
and one BLAS/OpenMP thread, with pytest `-p no:cacheprovider`. The saved
JSON exactly reproduces. All 28 new tests pass with warnings treated as
errors; the combined focused science/theory run passes 367 tests with 12
pre-existing warnings. No full-suite claim or change to the historical
synthetic F1 tolerance failure. All seven pre-edit documents match their
backup byte-for-byte; macOS metadata is retained in the archive too.
Seven original documents are preserved at `8c0c8524` and
`computation_handoff/backups/2026-09-10_pre_quantum_pair_threshold.tar.gz`.
Earlier sources, untracked user files and public/review copies are unchanged.

The unitarity-cut mechanism is established, not a claimed invention:
[Cutkosky, Singularities and Discontinuities of Feynman Amplitudes (1960)](https://doi.org/10.1063/1.1703676).
The source phases, two-flavor factors, LF phase-space normalization and
subtracted integrals above are derived and independently checked in the
project's own conventions; this does not adopt another model-completion
agenda or infer all-orders matching from a one-loop result.

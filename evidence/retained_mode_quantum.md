# Retained-mode quantum pilot: explicit and eliminated responses

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. Developed from `320e28e7`, local `main`.

## Result and reassessment

The user has selected the **retained-mode action interpretation**: quantum
fluctuations of retained degrees are calculated once, while already
excluded physics belongs in matched coefficients. Do not reopen this
decision or reinterpret an already quantum-effective functional as a bare
Hamiltonian.

The first bounded quantum test passes. The full eight-component reference
is quantized in a finite cell, not replaced by its classical stationary
point. Its quantum energy and scalar/pseudoscalar responses agree with
exact energy-dependent elimination, including source transport, direct
eliminated-sector response and the induced normalization. The internal
chiral spurion identity holds independently. Old tree-fitted parameters
give materially different quantum responses; they have **not** been
rematched or silently treated as still physically calibrated.

This advances the shared-action/state/source construction. It does not
select the two-multiplet sigma reference as production dynamics, restore
nucleon covariance or generate physical pion/proton/neutron predictions.
No numerical or physics blocker remains within this bounded experiment.
The next target is a limited nonzero-momentum retained quark/collective
and gauged-source calculation, with canonical LF amplitude matching and
the existing shared fit directions—not a fit of this cell to hadron data.

## 1. What is quantized, and what the scale means

Use all components of the preceding two real O(4) multiplets:
phi_i=(sigma_i,pi_i^1,pi_i^2,pi_i^3), i=1,2. For a spatially homogeneous
ordinary-time cell of volume Vc, define

\[
 H=\sum_{i,A}\frac{\Pi_{iA}^2}{2V_c}+V_c\left[
 \sum_i\frac{\lambda_i}{4}(\phi_i^2-a_i^2)^2
 +\frac{\kappa}{2}(\phi_1-\phi_2)^2
 -\sum_i c_i(m\sigma_i+p^a\pi_i^a)\right],
 \qquad [\phi_{iA},\Pi_{jB}]=i\delta_{ij}\delta_{AB}.
\]

Fields are GeV, canonical momenta GeV^-1, Vc GeV^-3, lambda dimensionless,
a_i^2 and kappa GeV^2, c_i GeV^2, and H GeV. The code removes only the
field-independent constant sum_i Vc lambda_i a_i^4/4. It does not remove
zero-point energy. The constant is fixed for these source variations;
parameter-dependent vacuum-energy conventions need consistent treatment
in any later parameter fit.

**This is ordinary-time quantum mechanics, not a light-front zero-mode
quantization.** Spatial nonzero modes, quarks and gluons are not quantized
inside this pilot. The canonical LF/QCD tensor and ordered-path utilities
remain separate and regression tested. Likewise the preceding C/D
source-curvature operators remain available; their interacting quantum
transverse EM current is not computed by these scalar/pseudoscalar tests.

Vc=1000 GeV^-3 corresponds to a 1.97327 fm cell side and 2pi/L=.628319 GeV
for its first omitted periodic spatial wavenumber. The latter is **not**
an inferred QCD renormalization/matching scale. The omitted modes have
not been integrated out to determine the coefficients. We reuse the prior
tree-calibrated parameter point only as an explicitly defined reference.
The user-approved Wilsonian interpretation is implemented as a counting
rule in this finite quantum test, not claimed as completed QCD matching.

### Numerical representation is not a confining model

Use complete oscillator shells sum n_iA <= N, with the same numerical
frequency for all four components of each multiplet. The oscillator
frequency specifies a basis; its potential is added and subtracted
algebraically, leaving the Hamiltonian above. It is not a physical HO
confinement interaction, fixed-K partonic truncation, or import of H2.
Complete shells preserve internal O(4) rotations; the mass spurion is the
explicit symmetry breaking, not an arbitrary componentwise cutoff.
The canonical commutator above defines the untruncated cell. Projected
coordinates have finite-boundary commutator defects; internal rotation
closure must not be mistaken for an exact finite Heisenberg algebra.

The dimension is binomial(N+8,8): 3,003 at N=6, 12,870 at N=8, 43,758 at
N=10 and 125,970 at N=12. The modest extension to N=12 took seconds per
case and remained within the pre-evaluated session cost; it was needed
to distinguish basis sensitivity from quantum response. Two intermediate
shells beyond N are used when multiplying the quartic polynomial. The
implementation computes P V(phi) P, **not** V(P phi P); multiplying
already-projected coordinates can erase real vacuum moments.

## 2. Quantum sources and independent identities

For external quark bilinears B_A=(qbar q,qbar i gamma5 tau^a q), the source
operator is O_A=Vc sum_i g_i phi_iA and H(B)=H+B_A O_A. B is GeV^3,
O GeV^-2. The zero-source quark mass insertion is m+sum_i g_i<sigma_i>.
This is not a calculated QCD propagator mass or a constituent-mass fit.
The unchanged resolved 24x24 Dirac/flavor/color bilinears from the previous
module supply the open quark endpoints; no spin/color projection or
canonical color-vector replacement is made. At B=0 parity and isospin
give the S/P channel coefficients from the corresponding susceptibilities.

With H|0>=E0|0>, v_A=(O_A-<O_A>)|0>, the real symmetric source operators
give a connected Euclidean response

\[
 \chi_{AB}(\nu)=\frac{2}{V_c}\operatorname{Re}
 \langle v_A|(H-E_0+i\nu)^{-1}|v_B\rangle,
 \qquad \chi_{AB}(0)=-\frac1{V_c}\frac{\partial^2 E_0}{\partial B_A\partial B_B}.
\]

The ground pole is removed by centering the source and adding
eta |0><0|, eta=1 GeV, which is inert on the orthogonal complement.
This is not a fitted subtraction. In
the Gaussian limit, it gives exactly g^T(K+nu^2 I)^-1 g; both the
frequency and volume normalization are independently checked. A coupled
harmonic reference also checks the vacuum shift and zero-point energy.
Nonlinear source derivatives agree with independent ground-eigenvalue
finite differences. Positive spectral response and its decrease with
Euclidean frequency are checked for these Hermitian collective sources,
not imposed on arbitrary colored kernels or matched differences.

For mass/pseudoscalar spurions, the operators are instead Vc c_i phi_iA
(dimensionless). Global chiral rotation gives

\[
 E_0(m,\mathbf p)=E_0(\sqrt{m^2+\mathbf p^2},0),\qquad
 \chi_{c\pi^a,c\pi^b}(0)=\delta_{ab}\frac{\sum_i c_i\langle\sigma_i\rangle}{m}.
\]

This static susceptibility has units GeV^2. Both the finite rotation and
generator commutator are tested independently; all three pion directions
agree. The replay's Ward errors stay below 1.7e-13 GeV^2. A finite-cutoff
internal Ward identity is not a Lorentz/component-covariance test or a
physical pion-pole/decay-constant determination.

## 3. Elimination changes representation, not quantum counting

Exact finite-matrix projection is established mathematics, not an
originality claim or selection of a nuclear shell-model workflow; see
[Feshbach, *Unified theory of nuclear reactions*](https://doi.org/10.1016/0003-4916(58)90007-1).
Here it is applied to this new nonlinear quantum reference, without the
old BLFQ/H2 model, a coupling expansion or independent fitted currents.

For the small N=3 cell (165 states), retain the 35 states whose second
multiplet is in its numerical oscillator vacuum. This is an algebraic
representation choice, **not** a physical high-energy/Wilsonian split.
Writing Q=1-P,

\[
 H_{\rm eff}(E)=H_{PP}+H_{PQ}(E-H_{QQ})^{-1}H_{QP},\quad
 W(E)=P+Q(E-H_{QQ})^{-1}H_{QP},
\]
\[
 N(E)=W^\dagger W=I-\partial_E H_{\rm eff}.
\]

An independent bracketed nonlinear eigenvalue solve gives
1.1548747598305147 GeV versus the full 1.1548747598305167 GeV; neither
is a hadron mass. The full wave is reconstructed to 3.8e-15. Its projected
ordinary norm is .9210406741, while phi^dagger N phi=1. Source derivatives
act on every H block and its resolvent. At a fixed pole,

\[
 \partial_a H_{\rm eff}=W^\dagger(\partial_a H)W,
 \qquad E_a=\frac{\phi^\dagger H_{{\rm eff},a}\phi}{\phi^\dagger N\phi}.
\]

The scalar source mean is 48.35690865 GeV^-2. A bare projected operator
with ordinary projected normalization gives 37.48112101 instead. Source
derivatives and the normalization cannot be selected independently.

For A=H-E0+eta |0><0|+i nu, eta=1 GeV, and a real centered source vector v,
its Schur complement S=A_PP-A_PQ A_QQ^-1 A_QP gives

\[
 v^T A^{-1}v=v_Q^T A_{QQ}^{-1}v_Q+
 (v_P^T-v_Q^TA_{QQ}^{-1}A_{QP})
 S^{-1}(v_P-A_{PQ}A_{QQ}^{-1}v_Q).
\]

The direct Q response and both transformed endpoints are necessary.
At nu=0,.2,.5 GeV the full and eliminated susceptibilities agree within
3.6e-14 GeV^-2. Dropping the direct term leaves norm defects 11.5784,
10.3036 and 6.6764 GeV^-2. The second derivative also agrees through an
independent reduced-pole finite-difference calculation. Adding the old
eliminated tree exchange to the explicitly quantized modes adds its
susceptibility again: a deliberate double-counting negative fails.

The full state is retained. Its small-cell node Schmidt spectrum has
largest probability .93893 and nonzero additional components. This is a
cutoff-dependent state-information witness, not converged physical
entanglement, QCD topological order or a quantum speedup claim. No DNN
topology or production quantum ansatz is selected by these numbers.

## 4. Resolution, quantum effect and calibration boundary

At Vc=1000 GeV^-3, the S/P quark-source susceptibilities (GeV^-2) are:

| Basis frequency | N | Scalar | Pseudoscalar |
|---|---:|---:|---:|
| .50 GeV | 6 | 85.9432 | 91.0154 |
| .50 GeV | 10 | 87.8893 | 93.2295 |
| .50 GeV | 12 | 88.0072 | 93.3676 |
| .65 GeV | 10 | 87.9846 | 93.3438 |
| .65 GeV | 12 | 88.0283 | 93.3910 |
| Original stationary tree | — | 17.2841 | 477.3439 |

At frequency .65, N=10 to 12 changes the responses by about .05%; the
two N=12 frequency choices differ by about .025%. These are observed
resolution checks, not rigorous error bounds or continuum-QCD convergence.
An initial broader scan at frequency .20 was less converged at the same
N; frequency .35 approached the same region more slowly. The basis was
changed to improve numerical resolution, without changing physical H.

The quantum scalar expectations are about (.0146869,.0095059) GeV versus
the classical (.0760959,.0520600) GeV; the corresponding quark mass
insertion is .0575667 versus .2843479 GeV. The large differences cannot
be attributed to the small final basis spread. They are finite-cell
quantum results at unchanged parameters, not evidence about a physical
quark mass or the accuracy of the full intended framework.

Volume sensitivity is substantial: the N=10, frequency .50 runs at
Vc=500 and 2000 GeV^-3 give S/P responses (38.5422,38.7618) and
(126.9785,239.3830). These two ancillary runs are not independently
converged volume limits. Do not refit physical pion/nucleon data to this
homogeneous cell or interpret the effect as a need for a sigma-model
repair campaign. It demonstrates why coefficients must be matched for
the declared quantum modes and domain. A passing symmetry identity does
not preserve the values of a previous tree calibration.

## 5. Next bounded target and preserved boundaries

Advance the retained-mode interpretation. Keep quantum state amplitudes,
shared interaction coefficients and their source partners together, with
their different geometries. The next pre-evaluation should specify a
small **nonzero-momentum retained quark/collective interaction** and its
gauged response, using the preserved canonical LF amplitudes, open
Dirac/flavor/color structures, ordered paths and protected collective
channels. Define source-dependent retained subtraction and the spatial/
temporal matching domain before claiming physical coefficients. Test a
limited independent state/current comparison before enlarging the model.

This is not automatic selection of a production potential, regulator,
fixed Fock truncation, LF zero-mode prescription or solver. Full quantum
transverse currents, causal momentum dependence, QCD matching, generated
p/n states, component/frame covariance, modular nuclei and GTMD/TMD/GPD
prediction remain open. Earlier canonical, source-curvature, nonlinear
tree, b1/PDF and quantum-network utilities and their failures remain intact.
The present quantum calculation tests first and second source responses;
higher connected quantum bilinear vertices are not yet evaluated.
No new physical direction decision is required to retain the interpretation
the user just approved; do not ask for it again.

## Replay, tests and recovery

Source: `src/deuteron_wigner/quantum/retained_mode_quantum.py`.
Tests: `tests/test_retained_mode_quantum.py` (24 direct tests).
Replay: `scripts/evaluate_retained_mode_quantum.py`; default runs all ten
resolution/volume cases and the independent elimination comparison.
Saved output: `retained_mode_quantum_results.json`. A smaller four-case
replay is used in unit tests; it is not substituted for the saved full run.

Use the existing compatible Python 3.12 environment with one BLAS thread:

```sh
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /private/tmp/deuteron-canonical.zxUoKe/env/bin/python scripts/evaluate_retained_mode_quantum.py
```

The first harmonic test needed N=8 instead of N=6 for the declared error,
and the source-derivative check needed a smaller finite-difference step.
Resolution/step were refined; no physical or identity tolerance was relaxed.
The complete polynomial, harmonic normalization, nonlinear finite
differences, generator identities, independent reduced eigenvalue and
response reconstruction test different parts of the calculation.

Validation: **321 focused tests**, including the 24 new tests and 297
preserved science/theory checks. The full ten-case replay matches a fresh
run exactly. Twelve existing dependency/docstring warnings remain; this
is not a full-suite claim. The previously reproduced synthetic F1 tolerance
failure on unmodified f509938c is untouched. The new TeX delimiters,
unique label and current source/hash pointers are verified; no PDF rebuild
or full historical-note audit is claimed.

The seven pre-edit pickup/theory documents are preserved in
`computation_handoff/backups/2026-09-10_pre_retained_mode_quantum.tar.gz`
and Git at `320e28e7`; all seven archive members match that commit exactly.
Prior source modules and unrelated user work are
unchanged. This is local only; no public push or review-mirror export.

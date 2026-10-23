# Collective-to-quark matching and the quantum interpretation boundary

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. Developed from `9c282a2c` on local `main`.

## Outcome and direction review

Two sequential steps are now implemented and independently checked:

1. Resolved collective exchange, including its one-/two-source interaction
   insertions, transports into the scalar/isovector-pseudoscalar quark
   channels. Functional retained subtraction and source-dependent changes
   of collective coordinates preserve the result when all endpoints move.
2. Nonlinear tree elimination generates connected six- and eight-quark
   coefficients as well as four-quark exchange. The full implicit tree
   response is retained and differentiated without a pairwise cutoff.

These advance the information-preserving shared-interaction architecture;
they neither determine QCD coefficients nor produce a physical nucleon.
All Dirac/flavor/color indices of this S/P block survive. It is not a
complete channel basis for QCD. The preserved canonical color-vector/LF
exchange is separate, unchanged and regression tested, not projected into
S/P or declared matched by agreeing matrix dimensions.

**Physics review reached:** a quantum state calculation must specify what
the learned functional means: an action for retained quantum modes, or a
quantum-effective functional whose vertices already incorporate their
fluctuations. This distinction changes the matching and current equations.
The bounded tree reference cannot decide it. Unknown Yukawas or other fit
coefficients alone are NOT a reason to stop exploratory work. The review
is about the quantum approximation and avoiding a new inherited model
commitment, not missing numerical input, permission or repository checks.

## Established ancestry and actual contribution

Gaussian completion, partial bosonization, inverse-response differentiation
and tree elimination are established mathematics, not an originality claim.
Partial bosonization does not itself determine a Yukawa coupling or a QCD
effective potential; those require an additional dynamical matching
calculation. See the explicitly separate transformation and dynamics in
[Gies, Schmieden and Zambelli, section 2.1](https://link.springer.com/article/10.1140/epjc/s10052-024-13689-3).
Their FRG/electroweak model and pointlike S/P truncation are not adopted.

Attaching an external source to all interacting pieces, with subtraction
against overcounting, is also established; compare
[Kvinikhidze and Blankleider](https://arxiv.org/abs/nucl-th/9901001).
We test the specific collective interaction insertions here, not that
paper's complete three-body equations or an external-quark radiative
amplitude. The project contribution is the explicit connection of our
preserved source operators to resolved quark tensors, nonlinear terms,
joint-learning sensitivities and the architecture decision below.

## 1. Fields, signs and open quark information

Use the preceding calibrated two-multiplet reference, not a new fit:

\[
 \phi_i=(\sigma_i,\pi_i^1,\pi_i^2,\pi_i^3),\quad
 B_A=(\bar q q,\bar q i\gamma_5\tau^a q),\quad
 Y_{iA,B}=g_i\delta_{AB},\quad i=1,2.
\]

Each bilinear has open **4 Dirac x 2 flavor x 3 color** indices. Its
matrices are the actual derivatives of the existing mass matrix and obey
Dirac-adjoint Hermiticity. The Pauli matrices are tau, not tau/2; source
generator normalization remains with the original source module. Color is
identity in this auxiliary block, not an erasure of canonical color-vector
exchange. The bilinears are Grassmann-even classical functional variables;
numerical commuting probes below test their coefficient tensors. They are
not measured condensates, quark wavefunctions, probabilities or quantum
expectation values. In particular, replacing B by its state mean before a
nonlinear evaluation is not justified.

The original mass convention is **m+g phi**. For x=phi-v, the Euclidean
interaction in the action is +B^T Y^T x. Thus its stationary source is
J=-Y B, not +Y B. Keep the vacuum constant and linear +(g.v) B_scalar
mass term separately. This fixes the otherwise easily missed sign of
the induced cubic-in-bilinears coefficient.

At spacelike Q^2 the two zero-source exchange weights are

\[
 w_S=g^T(H_\sigma+Q^2I)^{-1}g,\qquad
 w_P=g^T(H_\pi+Q^2I)^{-1}g.
\]

The open tensor is sum_A w_A (B_A)_ab (B_A)_cd, with w=(wS,wP,wP,wP),
and the eliminated Euclidean action is -B^T W B/2. No identical-fermion
projection, Fierz channel deletion, LF-time-local Hamiltonian extraction,
or normal-ordering/loop prescription is supplied by this tensor alone.

## 2. Source-complete Gaussian transport at the tested orders

Use K=H_pi-p^2 I for the four explicitly spacelike routes
p, p+q1, p+q2, p+q1+q2. The Minkowski metric is (+---); the overall
propagator i is stripped. Sources are Taylor amplitudes a,b in GeV with
dimensionless directions epsilon1,2. The four-route matrix is only a
Taylor jet sufficient for the specified mixed insertion, not a complete
finite-grid interacting field theory at nonzero sources:

\[
 K(a,b)=K_0+aK_a+bK_b+abK_{ab},\quad
 K_a=-\Gamma_a,\quad K_{ab}=-S_{ab}.
\]

Off-diagonal blocks contain the actual preceding current

\[
 \Gamma^\mu=(p'+p)^\mu I+
 \{q^2(p'+p)^\mu-(p'^2-p^2)q^\mu\}C
\]

and the fourth action derivative for S, including the minimal 2g^{mu nu}
contact and the C/D source-curvature terms. Both Fourier signs of each
real source are included. With R=K0^-1 and fixed Yukawa endpoints Y,

\[
 W=Y^TRY,\quad W_a=-Y^TRK_aRY,
\]
\[
 W_{ab}=Y^T(RK_aRK_bR+RK_bRK_aR-RK_{ab}R)Y.
\]

The two propagation orders AND contact are necessary. Direct Torch
differentiation of the inverse agrees with the analytic mixed derivative
to 5.7e-14 in the declared route. Interaction Ward relations agree to
1.5e-14 (one source) and 1.1e-14 (two sources); deleting the contact leaves
a 4.3171 defect in the contracted two-source route. The latter is an
algebraic dimensional response, not a physical dimensionless error bar.
External quark-leg attachments are still needed for a full radiative
amplitude. Nonlinear six-/eight-quark current tensors at general momenta
and a complete source-gauged many-body kernel are not implemented here.

The normalized Gaussian integral Z[B]/Z[0] gives the same stationary
quadratic source functional exactly. Its B-independent determinant cancels
in that ratio; this is NOT permission to discard background-dependent
fluctuations of the nonlinear theory. A source-dependent basis change
K'=U^T K U, Y'=U^T Y leaves W and its derivatives unchanged. Freezing Y'
while varying U fails the direct second-derivative comparison.

### Retained matching

\[
 W_{\rm res}=Y_f^T K_f^{-1}Y_f-Y_r^T K_r^{-1}Y_r,
 \qquad \partial^n W_{\rm res}=\partial^n W_f-\partial^n W_r.
\]

The test supplies a separately declared synthetic retained functional
with the same external coordinates; it is not inferred QCD matching.
Reconstruction passes at zero, first and mixed second source order.
Inverting K_full-K_retained is NOT this subtraction. Adding the full
response beside its retained part double counts. For a nonlinear theory,
subtract full source-dependent functionals, including induced higher
interactions, not only their two-point coefficients.

## 3. Nonlinear elimination and what pairwise compression loses

The full shifted homogeneous potential is exactly quartic:

\[
 V(v+x)-V(v)=\tfrac12 H_{ab}x_ax_b+
 \tfrac16 T_{abc}x_ax_bx_c+\tfrac1{24}U_{abcd}x_ax_bx_cx_d.
\]

H, T, U are the second, third and fourth derivatives of the **actual
eight-field potential**. Analytic tensors are independently checked by
Torch differentiation. With R=H^-1, L=RY and
P_{a,AB}=T_{abc}L_{bA}L_{cB}, stationary elimination gives

\[
 \mathcal W(B)=-\tfrac12 W_{2,AB}B_AB_B+
 \tfrac16 W_{3,ABC}B_AB_BB_C+
 \tfrac1{24}W_{4,ABCD}B_AB_BB_CB_D+O(B^5),
\]
\[
 W_2=Y^TRY,\qquad W_{3,ABC}=-T_{abc}L_{aA}L_{bB}L_{cC},
\]
\[
 W_{4,ABCD}=U_{abcd}L_{aA}L_{bB}L_{cC}L_{dD}
 -P_{a,AB}R_{ab}P_{b,CD}
 -P_{a,AC}R_{ab}P_{b,BD}
 -P_{a,AD}R_{ab}P_{b,BC}.
\]

The three cubic-exchange pairings are essential. W2,W3,W4 have units
GeV^-2, GeV^-5, GeV^-8; B is GeV^3 and the action density is GeV^4.
The SSS and S P_a P_a cubic coefficients are nonzero, odd-pseudoscalar
terms vanish, and all three isovectors remain. These are four-/six-/eight-
quark operator coefficients, **not automatically two-/three-/four-nucleon
forces**. That identification requires resolved fermion contractions,
matching, composition and state solutions.

An independent stationary root solves V'(v+x)+YB=0 with a verified positive
local Hessian. The calculation retains the whole near-vacuum branch,
not a finite polynomial in B. Its linked exact tree derivatives are

\[
 \mathcal W_B=Y^Tx_*,\qquad
 \mathcal W_{BB}=-Y^T[V''(v+x_*)]^{-1}Y.
\]

Both agree with finite differences of the independently solved value or
gradient. No quantum state was replaced by a mean density to obtain them.
For B=s(-1,.2,-.1,.3) GeV^3, the measured relative action-density errors are:

| s | Through four-quark | Through six-quark | Through eight-quark |
|---|---:|---:|---:|
| 1e-5 | 1.315% | .02144% | .0003393% |
| 1e-4 | 12.78% | 2.092% | .3328% |
| 3e-4 | 36.06% | 17.77% | 8.549% |
| 1e-3 | 99.52% | 163.61% | 265.26% |

Near zero the absolute omitted terms scale as s^3,s^4,s^5, respectively.
At the largest probe more polynomial terms make the approximation worse,
while the full local stationary branch remains stable. This is an expansion
failure, not a new physical phase or evidence about actual nucleon density.
The local zero-momentum expansion does not remove the earlier requirement
to protect light poles at finite momentum. Keeping explicit collective
variables or the full nonlocal/nonlinear functional avoids imposing a
pairwise truncation; it does not by itself solve quantum fluctuations.

## 4. Pion information does not identify all quark interactions

In this inverse-calibrated family v is aligned with the light pion mode e.
Changing g by 1.2 times a unit vector perpendicular to v preserves the
entire zero-source collective potential, the light pion coupling g.e and
the tree quark mass 0.2843478671 GeV. It changes the scalar exchange at
Q^2=.1 GeV^2 from 14.33920 to 17.80257 GeV^-2 (about 24.2%), and the
pseudoscalar exchange from 79.38293 to 77.67297 GeV^-2. The tree mass is
an exploratory output, not a measured constituent-mass calibration.

Likewise the prior C/D null directions keep the fitted light radius and
polarizability unchanged, yet change the quark-endpoint one-/two-source
responses. The replay witnesses norms .0151421 GeV^-3 and .0518767 GeV^-4.
Physical matching must constrain these directions or retain their joint
uncertainty. A coordinate change with all endpoints moved is different
from changing these physical coefficients at a fixed basis. No default
zero, separately fitted nucleon head or missing-information probability
is introduced.

## 5. Physics decision before the next quantum state solve

The tree stationary result is not generally the logarithm of the full
quantum path integral. For the nonlinear reference V''(v+x_*(B)) depends
on B. Its finite local log-determinant ratio is .5050 at s=1e-4 and 3.3316
at s=1e-3, while a Gaussian control has zero change. **These are not loop
energies, a spacetime trace, a controlled relative quantum correction or
an experimental prediction.** They establish that the earlier constant-
determinant cancellation cannot be reused for the nonlinear theory.
Tree elimination and the fluctuation determinant are distinct operations;
see [Dittmaier, Schuhmacher and Stahlhofen](https://arxiv.org/html/2102.12020v2).
Their heavy-field/local expansion is not adopted for the protected pion.

The outstanding choice is **which modes have already been integrated out**:

- A **scale-dependent action for retained modes** is matched after excluded
  physics has been integrated out, and retained quark/gluon/collective
  degrees are quantized once. Shared source terms enter that quantization.
  Its parameters can be jointly learned, but the scale, regulator, retained
  content and matching prescription must be declared.
- A **quantum-effective vertex functional** already includes a specified
  set of fluctuations. Its state/source equations follow from the
  appropriate effective-functional construction. Reusing it as an ordinary
  bare Hamiltonian and quantizing the same modes again is not equivalent.

This distinction is established independently of solver choice; see
[Berges, Tetradis and Wetterich, section 2.1 point 9](https://arxiv.org/html/hep-ph/0005122v1).
An exact transformation can relate suitable formulations, but a finite
ansatz and its fitted coefficients are not automatically interchangeable.
These formulations can coexist at distinct, explicitly related levels;
the question is the role of the NEXT learned interaction, not a permanent
either/or prohibition or a reopening of joint state/interaction learning.
No FRG, DSE, NJL, sigma-model or BLFQ completion program follows from it.

**Recommendation for review:** a bounded scale-qualified retained-mode
pilot is the closer fit to the user's canonical LF/Fock plus quantum-state
learning objective. Keep canonical quark/gluon and ordered gauge-path
content; use symmetry-labeled collective variables as an explicit,
matched representation and a flexible shared residual, not a physical
promotion of this two-multiplet quartic potential. Learn states and shared
interactions as already directed. The alternative is learning a specified
quantum-effective functional, with a different no-double-counting rule.
This recommendation is not a selection of numerical cutoff, production
interaction, instantaneous Hamiltonian or solver approximation.

After deciding that interpretation, the next bounded experiment should
compare an explicit and eliminated **quantum** response at one declared
scale/order, retaining source partners and canonical amplitude matching.
Only after its first independent state/current test should a p/n pilot
start. Then proceed to component/frame covariance and withheld observables,
modular deuteron and A=3/4 composition, and GTMD/TMD/GPD matching/evolution.
General nuclei and all-orders completion remain goals, not current results.

The persistent continuation reaches its stated physics-review boundary
here. Do not automatically launch a new nonperturbative model program.

## Reproduction and recovery

Implementation: `src/deuteron_wigner/quantum/collective_quark_matching.py`.
Direct tests: `tests/test_collective_quark_matching.py`.
Replay: `scripts/evaluate_collective_quark_matching.py`; saved numbers in
`collective_quark_matching_results.json`. Use the compatible existing
Python 3.12 environment recorded in the preceding source result, with
`PYTHONPATH=src:.`, no bytecode/cache writes and one BLAS thread.

```sh
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /private/tmp/deuteron-canonical.zxUoKe/env/bin/python scripts/evaluate_collective_quark_matching.py
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /private/tmp/deuteron-canonical.zxUoKe/env/bin/python -m pytest -q -p no:cacheprovider tests/test_collective_quark_matching.py
```

The new module has **22 direct tests**. Together with the preceding focused
source/chiral/causal/momentum/joint-learning, deuteron-information,
canonical-LF/graded-Fock/modular-gauge/baryon and current-theory regression,
the run covers **297 tests**. The saved replay matches a fresh calculation
exactly. All seven archive members were compared byte-for-byte with
9c282a2c. This is not a new full-suite claim: the prior synthetic F1
tolerance failure reproduced on unmodified f509938c remains documented,
not repaired or hidden. Twelve existing dependency/docstring warnings
remain in the broader focused run. No PDF rebuild or historical full-note
re-audit was requested or performed.

The replay and tests distinguish analytic/source identities, numerical
stationarity, calibration, identifiability and physical interpretation.
The first test run exposed a float32 probe passed to a float64 inverse;
the probe was corrected to float64, not an identity tolerance loosened.
The replay sensitivity check was changed from an arbitrary .1 lower bound
to the actual reproducible .0518767 witness, not a physical acceptance cut.
The six-quark sign was checked against the original full potential plus
quark mass term. No previous scientific tolerance or failure was changed.

Original seven pickup/theory files are preserved in
`computation_handoff/backups/2026-09-10_pre_collective_quark_matching.tar.gz`
and in Git at `9c282a2c`. Earlier modules, failures, source archives,
BLFQ/H2 branch and unrelated untracked user work remain untouched.
This update is local only: no public push or review-mirror export.

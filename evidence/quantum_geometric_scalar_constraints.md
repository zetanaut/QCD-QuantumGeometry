# Constraint-preserving scalar state/current representation

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

9 September 2026. Bounded exploratory experiment complete. **Advance to a
small geometry/response audit, not a learned physical model.**

The analytic endpoint lift, with residual rank five chosen before new
validation, preserves three known core conditions exactly and passes all
16 new kinematic checks. It retains 570 coefficients, versus 3,249 for the
degree-56 dense representation. It does not accelerate the cheap analytic
reference or establish physical covariance, particle entanglement or topology.

## Question and pre-evaluation

The [preceding compression experiment](quantum_geometric_scalar_compression.md)
passed finite elastic accuracy targets but changed a known ultraviolet
coefficient by 0.0012765. Its unmodified source and results are preserved.
This experiment asks whether exact known conditions can coexist with joint
state/current compression without separate observable corrections.

Use the same constant-vertex, equal-mass scalar control described in the
[reference derivation](quantum_geometric_scalar_pilot.md). The composite mass
is an input; no binding dynamics is learned. The compactified core is

\[
s=k_T^2/m^2,\quad t=\frac{s}{1+s},\quad
h(a,s)=(1-t)g(a,t),\qquad g(a,t)=\frac1{a+(1-a)t},
\]

with dimensionless virtuality \(a\in[0.0975,2]\). Its known conditions are

\[
g(a,0)=1/a,\qquad g(a,1)=1,\qquad g(1,t)=1.
\]

A UV-only ansatz would enforce only the second condition. Instead compare
an analytic lift with an affine boundary projection, both acting on the same
saved polynomial core. These are elementary numerical constructions, not a
claim of new physics or novel topology. Keep the original continuous charge
normalization and explicit valence/pair integrals; no per-curve normalization
or appended exact pair form factor. Bound the investigation to one
implementation/review cycle, reassessing for unstable cancellation, new
physical prerequisites or observable-specific patches.

## Two constructions and what they guarantee

Define

\[
L(a,t)=\frac{1-t}{a}+t,\qquad
w(a,t)=\frac{(1-a)^2}{a}t(1-t),\qquad
g_L=L-w f_\chi.
\]

Writing \(d=a+(1-a)t\), direct multiplication gives \(Ld=1+w\).
Thus \(L-wg=g\): the exact residual is the original core itself. Any finite
residual, including complex coefficients, preserves all three conditions.
The double zero at \(a=1\) additionally fixes
\(\partial_a g_L(1,t)=-(1-t)\). This derivative is not an extra fitted datum.

The absolute representation error is
\(g_L-g=-w(f_\chi-g)\). On this domain, \(0\le w\le2.08848\).
For the exact residual, the subtraction condition ratio is
\((L+wg)/g=1+2w\le5.17696\); it is not a severe cancellation problem here.
These statements concern this fixed domain, not a uniform bound as the
binding threshold \(a\to0\) is approached.

For the alternative, let

\[
H_t f=f(a,t)-(1-t)f(a,0)-t f(a,1),\qquad
g_P=L+H_t[f_\chi(a,t)-f_\chi(1,t)].
\]

This is an idempotent affine boundary projection, not an orthogonal
projection in the physical state metric. Since \(g(1,t)=1\), it recovers
the exact core and satisfies
\(\|g_P-g\|_\infty\le4\|f_\chi-g\|_\infty\).
The implementation subtracts endpoint lines in coefficient space and caches
the 57 coefficients at \(a=1\); that cache is included in resource counts.

Both constructions enforce representation identities, **not** a proof that
arbitrary learned residuals define a conserved physical current. Changing the
propagator or vertex as dynamics requires a compatible current derivation.

## Selection frozen before validation

Keep the degree-56 interpolant and test residual ranks 1, 2, 3, 4, 5, 6, 8,
12 on the existing 12 development points:
\(M/m=1,1.5,1.9\), \(Q^2/m^2=0,0.25,1,4\).
Require state L2 error at most \(10^{-3}\); absolute norm, each current,
separate valence/pair and frame-difference errors at most \(10^{-4}\);
boundary error below \(10^{-12}\). Select the first passing tested rank for
each construction, then the smaller coefficient count, with maximum
development current error as tie-breaker.

| Construction | First passing tested residual rank | Stored coefficients | Max. development state L2 error | Max. development current error |
| --- | ---: | ---: | ---: | ---: |
| Analytic lift | 5 | 570 | 2.664e-4 | 5.898e-6 |
| Affine projection | 8 | 969 | 8.020e-6 | 1.317e-5 |

Rank seven was not tested. The selected lift uses the same 570 stored real
coefficients as the old raw rank-five core. Its **whole separated function
has at most seven channels**: five residual channels and two fixed endpoint
channels. Residual rank is not whole-function rank, Schmidt rank or a
many-body bond-dimension result. The 5.7x saving refers to coefficient storage
against this degree-56 dense baseline, not an optimized global minimum cost.

Before construction/rank selection, reserve a NEW grid:
\(M/m=1.2,1.4,1.6,1.8\) crossed with
\(Q^2/m^2=0.16,0.81,1.69,3.24\). The choice above was recorded in the
handoff before evaluating that grid. The previous experiment's 16 reserved
points are now seen regression data. No observable samples train the core;
the rank selection uses development errors only.

## New validation and controls

The frozen lift passes all 16 new points. These are absolute dimensionless
errors against the reference, not fitted uncertainty estimates:

| Quantity | New raw rank-five comparison | Selected lift |
| --- | ---: | ---: |
| State L2 error | 5.307e-4 | 1.345e-4 |
| Squared-norm error | 3.007e-5 | 1.423e-5 |
| Transverse-frame overlap current | 5.370e-5 | 3.264e-5 |
| Longitudinal-frame total current | 2.990e-5 | 1.394e-5 |
| Separate valence term | 2.891e-5 | 1.397e-5 |
| Separate pair term | 7.901e-6 | 1.542e-6 |
| Difference between frames | 3.082e-5 | 2.269e-5 |

The three boundary checks give zero for the lift; its algebra establishes
the continuous identities, not just their sampled values. The old raw core
has maximum sampled UV, zero-momentum and unit-virtuality errors of
0.0012765, 0.0027693 and 0.0003830 respectively. The projection satisfies
these conditions to rounding accuracy in its development checks.

All old regression points also pass. Current quadrature orders 64 to 96
change the selected result by at most 2.23e-15 across all 44 kinematic points.
The degree-56 dense constrained reference has state error below 3.31e-14
and each current error below 5.33e-15 on development plus new validation.
Degree-24/40/56 dense comparisons separate interpolation from rank error.
These finite-domain checks do not prove uniform observable accuracy over
unseen masses, transfers, current components or distribution operators.

The direct controls retain failures that matter:

- Lift rank four passes development current targets but fails the state
  target (1.471e-3). Accurate curves alone do not retain the required state.
- A zero residual satisfies all three exact conditions yet produces a
  current error exceeding 0.01. Exact constraints alone are not dynamics.
- At \(M/m=1.5,Q^2/m^2=1\), omitting the pair term still loses over 10%.
- Matched complex basis/operator transport changes currents by at most
  1.12e-16; changing only the state gives an error of 0.06293.
- Tests forbid hidden calls to the exact core inside represented currents
  and retain the original normalization rather than renormalizing the
  compressed state to conceal its error.

Median times for 12 current evaluations, three repetitions on this host:
analytic reference 0.110 s; raw rank five 0.919 s; selected lift 0.960 s;
constrained dense 0.958 s. Thus there is **no demonstrated speed gain**.
Storage is 4,560 coefficient bytes for the lift versus 25,992 for degree-56
dense. Timings exclude construction/imports; temporary arrays and total
process memory are not included in coefficient counts. Even degree-24 dense
passes these finite targets, so do not present degree-56 compression as a
best-possible resource comparison. The complete replay took about 45 s.

## Reassessment and next bounded question

**Advance the representation/solver question, keep the physical-model claim
unchanged.** Exact scalar conditions need not be sacrificed to compression.
The current scalar representation now suffices as a small controlled input
for testing geometry; repeatedly optimizing this cheap analytic toy is not
the project's scientific objective.

Next pre-evaluate a bounded tangent/metric and operator-response audit of
the constrained low-rank ansatz. Derive its geometry from its actual state
map, separating low-rank factor redundancy, global phase and physically
distinct directions. Check derivatives and transported current responses
against independent finite differences on a small finite carrier. Distinguish
normalizing a state for projective geometry from the fixed continuous
normalization of the reference current. Do not reuse the full-tree metric
inverse by assumption, equate a small tangent norm with a small operator
response, or interpret trial coefficient variations as new matched dynamics.

Bound that to one further review cycle. If a proposed geometry test requires
new physical current completion, narrow it to fixed-reference representation
geometry. A later spin/gauge and shared-soft-dynamics pilot remains necessary;
no nucleon fit, production model, BLFQ repair or hardware run is selected.

## Reproduction, verification and recovery

- [Replay and all numerical rows](../snapshot/docs/next_level/quantum_geometric_scalar_constraints_results.json)
- [Replay program](../snapshot/scripts/evaluate_scalar_constraints.py)
- [Constrained representation](../snapshot/src/deuteron_wigner/quantum/scalar_constrained_representation.py)
- [Direct tests](../snapshot/tests/test_scalar_constrained_representation.py)
- Current roadmap (`handoff/QUANTUM_GEOMETRIC_EXPLORATION_ROADMAP.md`, full upstream checkout)

From the repository, use `PYTHONPATH=src:.`, Python with NumPy/SciPy, and
`python scripts/evaluate_scalar_constraints.py --output NEW_RESULT.json`.
Output creation is exclusive to preserve earlier evidence. The optional
`--development-only` mode does not evaluate new validation points.

Verification: 26 new tests plus 29 earlier representation, 34 scalar-reference
and six theory/handoff checks pass together (95 total). The full repository
suite was not rerun. These tests check the stated algebra and numerical
controls; their count is not evidence of physical completion.

Pre-edit checkpoint: `952d6cb642badacb253cc39617025871d4988628`.
The six entry/state/roadmap originals are in the
backup archive (`computation_handoff/backups/2026-09-09_pre_scalar_constraints.tar.gz`, full upstream checkout).
All six archived files were verified byte-for-byte against that checkpoint.
Earlier scalar source/results and the BLFQ branch are unchanged. The main
TeX, original volumes, public GitHub and review mirror are not updated by
this isolated experiment; the current handoff indexes this result separately.

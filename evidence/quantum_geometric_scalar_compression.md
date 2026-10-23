# Joint scalar state/current representation: first compression experiment

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

9 September 2026. **Bounded numerical experiment complete.** Rank five meets
the declared finite-grid accuracy targets, including reserved kinematics.
It reduces coefficient storage, but shows no meaningful speed advantage over
the dense interpolant and does not enforce the exact ultraviolet coefficient.
The next question is constraint-preserving compression, not a learned nucleon
Hamiltonian or quantum advantage.

## Pre-evaluation and scope adjustment

The preceding [pilot](quantum_geometric_scalar_pilot.md) supplied a known
state/current reference. This experiment asks whether a common reduced
representation can retain the state AND both contributions to its current.

Three options were compared before implementation:

- A wavefunction-only x/radial SVD would leave the nonvalence current outside
  the representation. Adding its exact form factor afterward would not test
  joint compression.
- Independent fits to the observable curves would discard the shared operator
  structure and could conceal the known frame failure.
- A shared propagator-core representation covers both the state and the
  pair kernel, using only the already specified model. This was selected.

This refines the proposed x/radial split to **virtuality/radial** coefficients.
Neither coordinate split is a physical particle-entanglement decomposition.
No bound-state Hamiltonian, new interaction or fitted current is introduced.
This approximates the fixed reference integrals; it does not derive a new
conserved interacting current for arbitrary trainable coefficients. Learning
a different physical core would require its own current/matching derivation.
Interpolation and [SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
are established numerical methods. The question is their suitability for our
joint construction, not their novelty. Off-grid evaluations use the retained
[Chebyshev polynomial](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebval.html),
not a fresh query to the exact form factor.

## Common state/operator map

Use the previous m=1 convention and define

\[
 h(a,s)=\frac1{s+a}=(1-t)g(a,t),\qquad
 t=\frac{s}{1+s},\qquad g(a,t)=\frac1{a+(1-a)t}.
\]

Here s=kT², and a denotes a positive inverse-propagator mass term, not a
particle label. For a bound-state amplitude, a=A_M(x)=1−M²x(1−x). The
longitudinal pair term also needs B_Q(z)=1+Q²z(1−z). Consequently the tested
envelope M≤1.9, Q²≤4 requires the common domain
0.0975≤a≤2, 0≤t≤1. Both endpoints and shifted transverse momenta remain
inside that domain. The implementation rejects virtuality extrapolation.
Only masses 1,1.5,1.7,1.9 and the stated transfer grids have numerical evidence;
the domain formula is not a convergence certificate everywhere in the envelope.

An uncompressed tensor-product interpolant represents

\[
 g(a,t)\simeq \sum_{i,j=0}^{d} C_{ij}T_i(\widehat a)T_j(2t-1),
 \qquad C_\chi=U_\chi\Sigma_\chi V_\chi^\dagger.
\]

The coefficient matrix is obtained from Chebyshev--Lobatto samples of g,
not from any measured or generated form-factor data. The reduced object
actually stores UχΣχ and Vχ†, not a hidden full matrix. Evaluation first
contracts the virtuality axis, then evaluates the remaining radial polynomial.
Complex coefficients are supported; the reference coefficients are real.
The coefficient-space SVD norm is not a state-error or current-error bound.

With Gauss nodes x_i,t_j and weights w_i,w_j, the whitened state array is

\[
 b_{ij}=\sqrt{\frac{w_iw_j x_i(1-x_i)}{I_0}}\,
 g(A_M(x_i),t_j).
\]

This includes the dx d²k/(16π³) wavefunction measure: the factors (1−t)²
in |psi|² cancel ds/dt. Its squared Frobenius norm approximates the continuous
state norm. No post-compression normalization is imposed. I0 is the SAME
continuous charge normalization as in the previous reference for every route.

The transverse plus-current overlap is evaluated directly as

\[
 F_{\rm DY,\chi}=\frac1{I_0}\int dx\,ds\,x(1-x)
 \left\langle h_\chi(A_M(x),s')^*h_\chi(A_M(x),s)\right\rangle_\phi,
\]

where s'=s+(1−x)²Q²+2sqrt(s)(1−x)Q cos(phi). No analytic form-factor
value is used in this evaluation.

For the longitudinal frame, Q²=M²(1−β)²/β. Changing the contour intervals
to 0<z<1 gives the common-core contractions

\[
 V_\chi=\int dz\,ds\,(1+\beta-2y)y\,
 h_\chi(A_M(z),s)^*h_\chi(A_M(y),s),\quad y=\beta z,
\]
\[
 N_\chi=\int dz\,ds\,(1+\beta-2y)(1-y)\,
 h_\chi(B_Q(z),s)^*h_\chi(A_M(y),s),\quad
 y=\beta+(1-\beta)z.
\]

F_L,χ=(Vχ+Nχ)/[(1+β)I0], with Nχ=0 at Q²=0. This follows algebraically
from the two contour residues already established in the pilot. B_Q is an
off-shell current factor, **not a normalized extra-sector wavefunction**.
The pair term is separately evaluated using the compressed core on both
factors. Its signed contribution is not a sector probability. Changing χ
changes it as well as the valence term; it is not an exact answer pasted onto
a compressed state.

The displayed conjugations specify the numerical bilinear representation;
they do not define an arbitrary complex covariant Bethe--Salpeter model.
For a change of basis v→U_out v and u→U_in u, the sampled operator is
transported as O→U_out O U_in†. A varying complex rephasing test preserves
each current with this transport. Changing state phases alone deliberately
fails. This tests phase bookkeeping, not gauge dynamics or Berry topology.

## Selection, validation and error separation

Targets were retained from the pilot: 1e-8 reference discretization accuracy,
1e-3 state L2 error and 1e-4 absolute compressed-current error. The latter was
also applied to norm drift, separate valence/pair terms and frame difference.

Degrees 24,40,56 were compared on the development grid. Degree 24 already
misses the reference target without any rank truncation. Degree 56 was used
for the rank comparison, isolating interpolation error from rank error.
Orders 64 and 96 were then compared independently for the current integrals.

The **12 development points** use M=1,1.5,1.9 and Q²=0,0.25,1,4. Among
χ=1,2,3,4,5,6,8,12, the first rank satisfying all development targets was
**χ=5**. That choice was frozen before evaluating the **16 reserved points**:
the three new transfers 0.0625,0.5625,2.25 for the three development masses,
plus all seven transfers at M=1.7. No rank was retuned after validation.
These are kinematic holdouts of a known common analytic function, not unseen
physical dynamics or experimental prediction tests.

| Maximum error | Development, rank 5 | Reserved points, rank 5 |
| --- | ---: | ---: |
| State L2 | 4.15e-4 | 4.15e-4 |
| Norm drift | 2.95e-5 | 2.95e-5 |
| Transverse overlap F | 5.67e-5 | 3.62e-5 |
| Longitudinal total F | 2.95e-5 | 2.92e-5 |
| Separate valence contribution | 2.95e-5 | 2.89e-5 |
| Separate pair contribution | 5.81e-6 | 7.20e-6 |
| Difference between the two frame evaluations | 4.21e-5 | 2.97e-5 |

Rank four fails, reaching 2.21e-3 state error and 6.13e-4 overlap error.
Rank eight reduces the development overlap error to about 1.31e-7. Thus rank
refinement changes measurable accuracy, not merely a software pass flag.
The largest 64→96 current-quadrature change at the selected rank is 1.34e-13.

The omitted-pair negative control remains wrong by about 12.48% at M=1.5,
Q²=1. The deliberately unmatched basis rephasing changes a current term by
about 0.06293; matching state/operator transport reduces that discrepancy
below 2.3e-16. Neither control is repaired by rescaling a curve.

## Cost and an additional physical-boundary warning

Degree 56 stores 57²=3,249 real coefficients. Rank five stores 570:
**5.7 times less coefficient storage**, 25,992→4,560 bytes. This is not a
5.7-fold reduction in total process memory: grids, current evaluation arrays,
and temporary polynomials remain. Building the factors still constructs a
dense interpolant and performs a classical SVD; no scalable preparation is shown.

Three steady-state timing repeats for all 12 development points gave median
times of about **0.105 s for the exact formula, 0.919 s for dense interpolation,
and 0.906 s for rank five**. Construction/imports were excluded from these
timings. The small dense-versus-reduced difference is not convincing evidence
of acceleration; the reference formula is about nine times faster. Most
evaluation cost remains in radial polynomial and current quadrature work.
Do not replace the cheap exact control with the compressed code for speed.

After reviewing the finite-grid success, an additional check tested the
known asymptotic condition lim(s→∞)s h(a,s)=g(a,1)=1 over 121 virtualities.
The dense interpolant preserves this to 1.34e-15, but rank five permits a
maximum coefficient drift of **0.0012765, about 0.128%**. The 1/s falloff
is retained, but its exact coefficient is not. This was not hidden by
renormalization and was not used to retune the chosen rank. Finite elastic
accuracy is therefore not a certificate of exact ultraviolet information.
The already-known divergent transverse second moment remains outside scope.

## Result-driven direction decision

**Retain:** the joint core construction, explicit pair-current map,
complex-bilinear transport, independent references and rank/error curves.
They show a useful compression mechanism at declared finite accuracy.

**Do not infer:** learned soft physics, exact covariance after compression,
physical entanglement, meaningful topological order, quantum speedup,
nucleon spin qualification or readiness for a full DNN fit.

**Next bounded experiment:** test an explicitly constraint-preserving
representation before introducing learned dynamics. A concrete candidate is

\[
 g(a,t)=1+(1-t)r(a,t),\qquad
 r(a,t)=\frac{1-a}{a+(1-a)t},
\]

and compression of r, keeping the boundary term exactly. This is a proposed
next experiment, NOT yet implemented. Compare it with the saved unconstrained
rank curve at equal current accuracy, using the same explicit pair operator.
Do not call this ultraviolet QCD matching: it preserves one known condition
of this control. The present holdouts have now been seen; reserve a new
kinematic set before selecting any revised representation.

Pre-evaluate whether that factorization is preferable to a constrained
projection and whether either affects numerical conditioning. Inventory the
other known core conditions, such as g(1,t)=1, rather than assuming that one
enforced boundary supplies every required identity. Use one small
implementation/review cycle, with no expanding dynamics dependency. Reassess
the accuracy/cost and exact-constraint evidence before moving to optimization
geometry or a carefully bounded spin-bearing control. If preserving a known
identity requires an observable-specific patch, change the representation.
Do not pursue hardware optimization for this analytically cheap toy.

## Evidence and preservation

[Numerical results](../snapshot/docs/next_level/quantum_geometric_scalar_compression_results.json),
[isolated implementation](../snapshot/src/deuteron_wigner/quantum/scalar_joint_representation.py),
[replay](../snapshot/scripts/evaluate_scalar_joint_representation.py) and
[direct tests](../snapshot/tests/test_scalar_joint_representation.py) preserve the
experiment. Run the replay with the project Python environment and
PYTHONPATH=src; --output creates a new JSON file and refuses to overwrite one.
The main study took about 32 seconds excluding imports on this host; the
boundary check was recorded afterward and is also included in the replay.
Verification passed 29 new representation tests, 34 existing scalar-reference
tests and six current-theory/handoff checks. The full repository suite was
not rerun and the old physical branch was not requalified by these checks.

The six original pickup documents are preserved at checkpoint a7880031 and
in the pre-experiment archive (`computation_handoff/backups/2026-09-09_pre_scalar_compression.tar.gz`, full upstream checkout).
The earlier scalar reference and all BLFQ kernels/results remain unchanged.
This is a local exploratory addendum. The main TeX, review mirror and public
GitHub have not been updated by this experiment.

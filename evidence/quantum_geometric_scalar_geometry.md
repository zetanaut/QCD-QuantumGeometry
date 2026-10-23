# Scalar geometry and joint current-response audit

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

9 September 2026. Bounded audit complete. **Close the scalar-control stage;
carry operator-sensitive directions into the next spinful pilot design.**

The actual constrained-state geometry and both current-leg derivatives pass
independent checks. However, a state-only numerical metric is not sufficient
to qualify the shared current representation. A constructed, exactly
rank-preserving change leaves the reference state almost unchanged while
moving the pair current beyond the preceding absolute accuracy target.
This is a representation/conditioning result, not a new physical covariance
failure, nucleon prediction or proof against quantum natural gradients.

## Scope and pre-evaluation

Start from [the fixed degree-56, residual-rank-five lift](quantum_geometric_scalar_constraints.md),
not a newly learned Hamiltonian or current. Its scalar core is
\(g_C(a,t)=L(a,t)-w(a,t)f_C(a,t)\), with
\(L=(1-t)/a+t\) and \(w=(1-a)^2t(1-t)/a\).
The three exact core conditions remain imposed for every tested perturbation.
The mass is still an input and the original continuous charge normalization
is unchanged. No observable fit, optimizer, hardware run or BLFQ repair is made.

Compare raw factor coordinates with a nonredundant coefficient-tangent chart.
Derive the state metric rather than importing the old full-tree inverse;
differentiate state and both current endpoints; test complex directions,
normalization, coordinate changes, numerical refinement and current responses
along weak state directions. This is one implementation/review cycle. The
potential mismatch in virtuality coverage was identified before testing:
the bound state uses \(a=1-M^2x(1-x)\le1\), while the pair kernel additionally
uses \(a=1+Q^2z(1-z)\ge1\), in the existing units \(m=1\).

Projective geometry is established machinery, not our novelty claim; see
[Stokes et al., Quantum Natural Gradient](https://quantum-journal.org/papers/q-2020-05-25-269/).
The contribution here is the explicit audit of its pullback and limitations
for our shared constrained representation. Topology or physical symmetry
does not follow from this numerical geometry.

## Geometry actually differentiated

For \(C=LR\) with \(L\in\mathbb C^{57\times5}\),
\(R\in\mathbb C^{5\times57}\),

\[
\delta C=\delta L R+L\delta R,\qquad
(\delta L,\delta R)=(LX,-XR)\Longrightarrow\delta C=0.
\]

There are 570 complex raw factor coordinates and 25 complex factor-gauge
directions, leaving 545 complex coefficient tangents. The stored reference
itself remains real. These are parameter counts, not physical entanglement
or observable-identifiability counts.

Let \(U,V\) be orthonormal bases for the rank-five left/right singular spaces,
with complements \(U_\perp,V_\perp\). Use the Frobenius-orthonormal tangent
blocks \(UMV^T\), \(U_\perp KV^T\), \(UNV_\perp^T\).
This removes factor redundancy without treating weak state sensitivity as
gauge. The first 285 chart directions keep the right space fixed, so finite
affine changes in that subspace preserve rank at most five exactly. General
affine tangent paths need not retain rank beyond first order and are used
only for derivative tests, not as optimizer steps.

For the quadrature-weighted unnormalized state \(b(C)\), set
\(n=b^\dagger b\), \(\psi=b/\sqrt n\), \(J=\partial b/\partial z\). Then

\[
A=\frac{(I-\psi\psi^\dagger)J}{\sqrt n},\qquad
ds^2=\delta z^\dagger A^\dagger A\,\delta z.
\]

At the real reference, the real-coordinate derivatives are \([A,iA]\), so
the quantum geometric tensor has blocks
\(\left(\begin{smallmatrix}G&iG\\-iG&G\end{smallmatrix}\right)\),
\(G=A^TA\); its real metric is \(\operatorname{diag}(G,G)\).
The imaginary cross blocks are retained, not discarded as zero physics.
An external global phase \(\delta b=ib\) has zero horizontal tangent.
It is **not** an extra free phase of the constrained core: multiplying that
core by a phase would violate its fixed nonzero boundaries. Do not subtract
another complex coordinate from the 545 merely because states are rays.

In the continuous polynomial class, a tangent that vanishes on the open
bound-state domain must vanish identically. Likewise a horizontal-null
tangent would require a constant multiple of the entire core; its fixed
boundaries exclude that nonzero multiple. Thus very small sampled singular
values are not proved exact continuum null directions. The issue is unstable
continuation/sensitivity at finite precision, with additional finite-grid
aliasing—not mathematically independent polynomial values on two intervals.

## Shared current derivative and normalization

For each current integral, differentiate
\(F=I_0^{-1}\int W\,\overline{h_o}h_i\):

\[
\delta F=I_0^{-1}\int W
\left(\overline{\delta h_o}h_i+\overline{h_o}\delta h_i\right).
\]

Keep transverse-frame overlap, longitudinal valence and longitudinal pair
terms separately, including the pair kernel's extra virtuality range.
The coefficient gradient is \(G_o:\overline{\delta C}+G_i:\delta C\);
for real chart directions it gives \(G_o+G_i\), and for imaginary directions
\(i(G_i-G_o)\). No exact pair answer is appended.

Projectively normalizing a state does not authorize changing the reference
current. A separately considered ratio obeys
\(\delta(F/n)=\delta F/n-F\delta n/n^2\).
At zero transfer, the represented raw charge equals \(n\), not exactly one.
The executed norm-sensitive probe gives raw charge derivative 0.402036; its
normalized-ratio derivative is below 1.45e-15. Omitting the norm term would
leave a false 0.402034 response. The actual current is not renormalized.

## Numerical checks and decisive counterexample

Raw factor-gauge cancellation and a finite invertible factor change preserve
\(C\) within 1.79e-15 and 4.94e-16 respectively. For a complex unit coefficient
direction, central differences at steps 1e-3, 1e-4 and 1e-5 agree with raw
state, horizontal-state and current derivatives to at worst 4.97e-12,
1.65e-11 and 1.88e-11. Both-leg and normalization omissions fail direct tests.
On an explicitly identified eight-direction subspace, the metric has
condition number 5.83; a coordinate transformation preserves the natural
state-space direction to relative 5.69e-16. The corresponding Euclidean
parameter-gradient direction changes by relative 1.61. The full ill-
conditioned metric is not inverted, and no optimizer is run.

The initial 32/48-point state grids underresolve tangents: a degree-56
virtuality polynomial becomes degree 112 in x. At relative singular-value
cutoff 1e-8, the full single-state numerical ranks are 211, 317, 369, 369,
369 for orders 32, 48, 64, 96, 128. These are cutoff-dependent sampled ranks,
not physical parameter counts. A coarse weak witness changes its horizontal
norm from 4.22e-10 at order 48 to 2.47e-9 after refinement. This sampling
effect is retained in the report, not called gauge.

Use order 128 for the resolved audit. A single-state witness weak at M=1.5
is not weak at M=1.8; that is limited coverage, not a universal statement.
As a cross-check, stack the normalized-state Jacobians for M=1,1.5,1.9 with
equal weight. Within the 285 fixed-right directions, project the M=1.5,
Q²=1 pair response into the subspace whose singular values are at most
1e-8 times the largest. Normalize this projected coefficient direction.
This deliberately constructs a diagnostic witness; it is not blind model
validation or an inferred physical uncertainty. The cutoff is a declared
numerical diagnostic, not a selected optimizer regularization.

For this ensemble witness:

| Diagnostic | Executed value |
| --- | ---: |
| Three-state horizontal tangent norm | 1.2741e-9 |
| M=1.5 raw state tangent norm | 1.0354e-9 |
| M=1.5, Q²=1 overlap derivative | 6.8143e-10 |
| Same-point longitudinal valence derivative | 6.7620e-10 |
| Same-point pair derivative | 8.2102e-4 |
| Additional M=1.2 / M=1.8 horizontal norms | 1.6699e-9 / 1.3448e-9 |

State orders 96/128/160 and current orders 64/96/128 retain this result.
Changing the diagnostic cutoff to 1e-6 or 1e-10 still yields pair responses
8.35e-4 or 5.42e-4. The conclusion is not tied to one numerical rank count.
At Q²=0 the pair response is zero, so charge alone cannot expose the gap.

A finite step of 0.25 along the fixed-right witness changes the M=1.5 state
by only **2.5885e-10**, while its Q²=1 pair current changes by
**2.0526e-4**, exceeding the preceding 1e-4 absolute-current tolerance.
The overlap/valence changes are about 1.7e-10. Rank remains at most five
(sixth singular value 1.83e-17) and all three core constraints remain exact.
The original reference is not changed; this is a separate deliberate
negative control, not a physically allowed new solution with a matched current.

Adding the three dimensionless current responses to the state-tangent norm
detects the witness (diagnostic norm about 8.21e-4 versus state norm 9.68e-10
at M=1.5). Such an unweighted joint response norm is **not** automatically a
physical quantum metric; its content and scales would require justification
in a solver or inference problem. No augmented optimizer is selected here.

## Reassessment and next scientific step

**Close this scalar audit with a design constraint:** do not identify weak
valence-state metric directions with disposable parameter gauge. A common
representation must retain and test the domains probed by its operators.
State-only convergence, fixed boundaries and small bond/factor rank do not
certify current accuracy. This does not invalidate the preceding frozen
representation's successful tests; it limits what training can safely infer.

The next bounded step is pre-evaluation of a small **spinful shared-dynamics
pilot**. Compare a few explicit candidate constructions before implementation:
physical degrees of freedom and learnable kernel/vertex content; exact versus
dynamical symmetries; how the same construction supplies its current; the
operator-sensitive state/sector carrier; and an affordable independent
positive/negative covariance control. Identify unresolved physical inputs
and prior art without inheriting a published method's unfinished repair queue.
Recommend a pilot and review its scientific/cost implications before a full
fit or substantial new model dependency. The scalar control can remain a
regression fixture; no further scalar precision or optimizer campaign is needed.

## Evidence and recovery

[Full numerical report](../snapshot/docs/next_level/quantum_geometric_scalar_geometry_results.json),
[replay](../snapshot/scripts/evaluate_scalar_geometry.py),
[implementation](../snapshot/src/deuteron_wigner/quantum/scalar_representation_geometry.py),
[direct tests](../snapshot/tests/test_scalar_representation_geometry.py), and
updated roadmap (`handoff/QUANTUM_GEOMETRIC_EXPLORATION_ROADMAP.md`, full upstream checkout).
Replay from the repository with `PYTHONPATH=src:.` and
`python scripts/evaluate_scalar_geometry.py --output NEW_RESULT.json` in a
NumPy/SciPy environment. Output creation is exclusive. The recorded replay
took about 24 seconds; it is not a speed-advantage benchmark.

All 110 focused tests pass: 15 new geometry/response checks plus the preceding
95 scalar, representation and theory/handoff checks. The full repository
suite was not rerun. Test counts are not a physical-quality claim.

Pre-edit source: `3fba8ce2a9eacfe50b9e75540b53cdadcb09fd1a`.
Six original entry/state/roadmap files are saved in
the backup (`computation_handoff/backups/2026-09-09_pre_scalar_geometry.tar.gz`, full upstream checkout).
All six archived files match the pre-edit source byte-for-byte.
Earlier scalar source/results and all BLFQ kernels remain unchanged. The
main TeX, original volumes, review mirror and public GitHub are unchanged;
this bounded result is indexed separately in the current handoff.

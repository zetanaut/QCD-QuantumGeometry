# Gluon interaction screening: force, surface independence and shared responses

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

11 September 2026; continuation from 8ff38e84. **One necessary candidate test,
not a selected binding action, quantum vacuum or physical nucleon result.**

## Result and decision

A smooth, finite-range, scalarized two-point field-strength surface functional
produces an attractive static force and decaying separated-surface coupling.
Nevertheless it fails an exact same-boundary condition when used without
connector dependence and higher ordered correlations. An arbitrary closed
surface changes its Wilson factor; full SU(3) connection products do not.
Reject this **unrestricted two-point surface closure as a complete interaction**.
Do not reject all two-point approximations or claim that QCD cannot confine.

This matters for our proposed quantum/geometric network: learned gluonic
correlation strengths cannot replace the connection and its linked operator
hierarchy. Separate face traces also lose essential ordered information.
The existing matrix-valued connection and full baryon-color tools pass the
new controls and remain the organizing representation. Their success is
algebraic; it does not yet determine the source-free dynamics or its vacuum.

The origin and limitations of field-correlator expansions are established
prior art, not our invention. In particular, an exact Wilson expansion is
surface-independent while individual cumulant terms need not be; a truncation
on a prescribed minimal surface can be a useful approximation. That is not
an exact unrestricted functional. We do not adopt the cited gluelump solution
program, its quenched/large-Nc reductions, or its physical parameters.
[Simonov and Shevchenko, sections 2 and 4](https://onlinelibrary.wiley.com/doi/10.1155/2009/873051)

## Candidate and exact scalar checks

Use Euclidean coordinates and six independent bivector components
I=(mu,nu), mu<nu. Write the candidate transported/color-contracted covariance
as C_IJ(z)=delta_IJ d exp(-z²/ell²), with strength d>=0 in GeV^4 and length
ell>0 in GeV^-1. Color normalization and g² are absorbed into d. The
approximation deliberately drops connector dependence and all contributions
beyond its scalar quadratic surface action:

\[
 S_2[S]=\frac d2\sum_I\int_S d\sigma_I(x)\int_S d\sigma_I(y)
 e^{-(x-y)^2/\ell^2},\qquad W_2[S]=e^{-S_2[S]}.
\]

There is no extra 1/2 in the mu<nu surface convention. Positivity of this
smooth kernel is not proof that it is the curvature covariance of any gauge
connection, a reflection-positive quantum theory or a physical large-distance
spectral correlator. The Gaussian profile is a finite test, not a fitted
long-distance vacuum model or new QCD regulator.

For a rectangle R by T,

\[
 I(L)=\sqrt\pi\ell L\,\mathrm{erf}(L/\ell)
       -\ell^2(1-e^{-L^2/\ell^2}),\qquad S_2(R,T)=\frac d2 I(R)I(T),
\]
\[
 V_2(R)=\lim_{T\to\infty}\frac{S_2(R,T)}T
       =\frac{d\sqrt\pi\ell}{2}I(R),\qquad
 \frac{dV_2}{dR}=\frac{\pi d\ell^2}{2}\mathrm{erf}(R/\ell).
\]

The force is -dV/dR; its attractive magnitude tends to pi*d*ell²/2. This is
a soft static reference, not a light-quark mass or a proof of confinement in
full QCD with pair creation. For two identical parallel rectangles separated
normally by r, their cross term decays exactly as exp(-r²/ell²). This is a
surface-functional cluster check, **not a calculated two-nucleon force**.

Now integrate over all six outward faces of a closed box with side lengths
L_i. Direct four-dimensional surface quadrature agrees with the independent
separable expression

\[
 S_2[\partial V]=d\sum_{i=1}^3
 I(L_j)I(L_k)(1-e^{-L_i^2/\ell^2}),\qquad \{j,k\}=\{1,2,3\}\setminus\{i\}.
\]

It is strictly positive for nonzero sides and d>0. In a regular topologically
trivial region, adding an auxiliary closed surface to a Stokes representation
does not change its boundary observable. The exact non-Abelian statement
requires ordered, commonly transported field insertions; it is not the
Abelian assertion that an untransported color flux simply integrates to zero.
Our candidate omits precisely that structure. A distant added bubble has a
vanishing cross term with the original sheet, but retains its positive self
action, so it changes W_2 by a nontrivial factor. A thin connecting neck can
make the surface connected without removing this limit. No topology defect
or inserted magnetic source is part of this control.

For d=1 GeV^4, ell=1 GeV^-1 (illustrative only):

| Cube side / ell | Closed S2 | Spurious factor exp(-S2), required to be 1 |
|---:|---:|---:|
| .25 | .0006954694 | .9993047723 |
| .5 | .0382492813 | .9624729845 |
| 1 | 1.4075365074 | .2447454698 |
| 2 | 19.0997925223 | 5.0707e-9 |

The largest quadrature discrepancy is 1.07e-14, much smaller than the
violation. One geometry-independent normalization cannot remove it for all
bubbles; setting d=0 also removes the proposed static force. Choosing one
preferred surface instead is an additional approximation with a restricted
claim and controlled errors, not a demonstrated restoration of the identity.

## Exact non-Abelian control and the baryon-color interface

The twelve cube edges carry generic SU(3) matrices constructed with the
existing fundamental generators. They are fixed test connections, not a
quantum ensemble or a new lattice Hamiltonian. Replace each edge of the
bottom-square boundary by a three-edge detour over a side. With prefix P from
the common base to the replaced edge, old edge U and detour D,

\[
 L=PUD^{-1}P^{-1},\qquad W_{\rm old}=L W_{\rm new}.
\]

Four replacements express the bottom loop through four sides and the top.
All cancellations hold at the matrix level, for arbitrary link values and
local gauge transformations. The residual matrix error is 2.27e-15. Replacing
each factor by its separate normalized trace instead gives an error .269966
in the reference loop. Scalarization is not an information-preserving map.

The same identity is carried through the full 27-component epsilon-epsilon
three-line baryon color contraction, using the saved baryon seed. Its complex
amplitude is .2075926292 + .04072837568 i; the two factorizations differ by
3.31e-16. These are color amplitudes only, with no solved quark spin/orbital
wavefunction, proton/neutron mass, sea distribution or nuclear binding.

All twelve edges and eight generators are differentiated. The 96 connection
tangents agree with the direct boundary derivative within 1.59e-15, with an
independent finite-difference check. In the illustrative interior edge (0,4),
the true boundary derivative is zero. If connector derivatives are frozen,
the computed tangent has norm .895726. The omitted terms matter even when
the unsourced factorization is exact. This is a **color-connection response**,
not an electromagnetic current, mass derivative or bilocal PDF insertion.

## Reassessment: retain connection-based shared dynamics

Advance the representation; stop expanding or fitting the failed two-point
closure. This study supports using the canonical gauge connection and its
non-Abelian self-interactions to organize the dynamical model, with learned
residuals and quantum-state correlations defined there. In a curvature
description, higher correlators and connector variations are linked by gauge
identities; independent fitted moments are not a substitute. Even Gaussian
statistics of a connection do not generally imply Gaussian statistics of its
nonlinear curvature. No minimum sufficient finite cumulant order is proved.

**Next bounded task:** specify a connection-based source-free residual operator
family with the canonical quark/gluon action and common mass, metric/current
and bilocal sources. Compare its zero-source stability and chiral/sea-quark
content before a nonperturbative baryon solve. Preserve full Dirac/color
nodes and source derivatives, full-minus-retained subtraction and separated
singlet channels. Unknown coefficients allow a transparent parameterized
test; they are not a reason to invent a physical fit. Do not substitute another
independent-density model, reopen the broad approved SU(3) direction, build a
new lattice/strong-coupling program by default, or repair this candidate via
an indefinite cumulant campaign. A genuinely new physical choice must be
exposed before substantial commitment.

The planned full mass/metric/bilocal matching of this candidate was not
performed because it failed the earlier necessary condition. The exact color
source test must not be relabeled as that matching. Physical p/n binding,
local quantum Gauss law/vacuum, current covariance, nuclear forces and
predictive TMD/GPD/GTMDs remain open. This result narrows the interaction
choice; it does not complete the predictive goal or prove physical topological
order. No other team's binding program has been adopted.

## Reproduction and recovery

- Implementation: `src/deuteron_wigner/quantum/gluon_surface_consistency.py`.
- Direct tests: `tests/test_gluon_surface_consistency.py` (29 warning-strict).
- Six theory/handoff consistency tests also pass warning-strict (35 together).
  New TeX braces/environments and all document-label/bibliography-key uniqueness
  checks pass; no PDF rendering or full historical theory re-audit.
- Focused integration: 98 passed with six existing NumPy determinant warnings
  in three unchanged gauge-network tests. This includes the preceding nuclear
  matching and LF path-state tests, not a full repository or warning-strict
  integration certification.
- Replay: `scripts/evaluate_gluon_surface_consistency.py`; calculated output
  in `gluon_surface_consistency_results.json`.
- Preserved before editing: eight entry/roadmap/theory documents at
  `8ff38e8456aeb81c52d25f9a91e568f5e12fcc4d` and
  `computation_handoff/backups/2026-09-11_pre_gluon_surface_consistency.tar.gz`.
  All eight members are byte-identical to that commit; archive SHA-256
  `b8d2f8ed289f1f7bd350dfd6b78252b06f1aec13ae6248c6ba4b1bc967741003`.
- Prior scientific implementations/results are unchanged. Local only; no
  public push, review export, new theory volume or PDF rendering.

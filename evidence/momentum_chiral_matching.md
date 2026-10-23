# Momentum-dependent matching and nonlinear shared dynamics

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. Bounded operator/generating-functional qualification,
not a production interaction or new bound-state prediction.

Source: src/deuteron_wigner/quantum/momentum_chiral_matching.py.
Replay: scripts/evaluate_momentum_chiral_matching.py and
[numerical results](../snapshot/docs/next_level/momentum_chiral_matching_results.json).

## Result and revised direction

The shared-dynamics construction now has a momentum-resolved color-current
term connected to the canonical distinguishable-quark Born amplitude,
exact spin/color/flavor recoupling, ordered color transport, and a
nonlinear invariant response with its complete first/second/source
derivatives. Actual PyTorch differentiation agrees with independent
equations and finite variations; momentum coefficients feed those same
responses rather than independent observable heads.

Three results govern the next implementation:

1. Keeping only scalar/pseudoscalar Dirac channels cannot reproduce the
   full canonical vector-exchange tensor. Chiral/pion success is not a
   license to discard the remaining structures.
2. A nonlinear kernel can pass a chiral identity while omitting its
   required learned-curvature term. A full derivative check is necessary.
3. For nonlinear dynamics, subtract retained physics at the generating-
   functional level, then differentiate. Subtracting the momentum
   coefficient and applying the nonlinear map afterwards misses mixed
   retained/residual terms, even when both constructions are invariant.

These results qualify the architecture, not its physical coefficients.
The next bounded question is the causal/spectral realization and
independent scalar/vector/axial matching information needed to select a
momentum-dependent state/current reference. A spacelike interpolation
alone is not a valid production Hamiltonian or bound-state kernel.
Do not assume positive spectral density for a colored propagator, or
analytically continue this diagnostic rational basis by default.

## Canonical amplitude and channel information

Use the existing canonical LF spinors, symmetric LF convention,
physical transverse gluons and SU(3) generators. No old Gaussian, HO
basis, fixed-K cell or H2 elimination is imported.

For distinguishable on-shell quarks, q=p1'-p1=-(p2'-p2), t=q^2<0,
and J_i^mu=ubar_i' gamma^mu u_i, the amplitude is

\[
{\cal M}^{(2)}
 ={g_s^2\over t}J_1^\mu J_{2\mu}
       \sum_a t^a\otimes t^a .
\]

It is a 36 x 36 spin/color matrix with ubar u=2m and an overall
S-matrix phase stripped. It is dimensionless, not a finite LF
mass-squared matrix. Identical-quark exchange/antisymmetrization belongs
to a separate assembly step, not this distinguishable t-channel check.

The independent LF evaluation uses

\[
{\cal M}_{\rm prop}/g_s^2
 =-{1\over t}\sum_h(J_1\cdot\epsilon_h)
                  (\epsilon_h^*\cdot J_2),\qquad
{\cal M}_{\rm inst}/g_s^2=-{J_1^+J_2^+\over(q^+)^2},
\]

with the same open color factor. Both signed exchanged q+ domains are
retained. Dropping the instantaneous term fails. Among 24 new momentum
cases and their boosted partners, one q+ is about -1.81e-4 GeV:
the separate pieces are millions of times larger than their sum.
The raw split/covariant relative difference reaches 4.79e-10,
consistent with floating-point conditioning (at most 4.37 epsilon
times the sum-of-norms condition estimate).

Algebraically combine the common 1/(q+)^2 terms before evaluation:

\[
{{\cal M}_{\rm combined}\over g_s^2}
={J_1^+\,\boldsymbol q_T\cdot\boldsymbol J_{2T}
 +J_2^+\,\boldsymbol q_T\cdot\boldsymbol J_{1T}
 -2q^-J_1^+J_2^+\over q^+t}
-{\boldsymbol J_{1T}\cdot\boldsymbol J_{2T}\over t}.
\]

This independent LF form agrees within 4.99e-13 relative; the spin-summed
amplitude norm is boost-invariant within 3.56e-15. The zero-q+ domain
still requires a separate limit. This is a numerical bare-parton control,
not resolution of the historical nucleon covariance discrepancy.

For all four open Dirac indices,

\[
\sum_\mu(\gamma^\mu)_{ab}(\gamma_\mu)_{cd}
=1_{ad}1_{cb}-(\gamma_5)_{ad}(\gamma_5)_{cb}
-\tfrac12(\gamma^\mu)_{ad}(\gamma_\mu)_{cb}
-\tfrac12(\gamma^\mu\gamma_5)_{ad}(\gamma_\mu\gamma_5)_{cb}.
\]

The independent 16-matrix trace projection reconstructs this exactly.
The tensor-bilinear coefficients vanish for this particular Born
vector tensor, not for every possible production interaction.
Our pseudoscalar matrix is gamma5; using i gamma5 changes the associated
coefficient convention. These are tensor permutations; a Grassmann
operator reordering also carries its fermion sign.
For the standard recoupling framework see
[Nieves and Pal](https://arxiv.org/abs/hep-ph/0306087).

The exact crossed color and flavor identities are

\[
\sum_a t^a_{ij}t^a_{kl}
 ={4\over9}\delta_{il}\delta_{kj}
   -{1\over3}\sum_a t^a_{il}t^a_{kj},\qquad
\delta_{fg}\delta_{hi}
 =\tfrac12\delta_{fi}\delta_{hg}
  +\tfrac12\sum_a\tau^a_{fi}\tau^a_{hg}.
\]

No color octet or isovector channel is discarded. In the declared
Hilbert–Schmidt tensor metric, the best Dirac scalar/pseudoscalar-only
projection has relative residual 1/sqrt(2)=.707107, and retaining only
the crossed color singlet has residual 1/3. These are fixed-basis
open-index information losses, **not percentages of error on physical
observables**. Restricting to a particular on-shell observable can hide
such losses; it does not establish equality of the full interaction.

## Momentum representation and honest matching scope

At zero external gauge field, use the spacelike scalar coefficient

\[
d_\theta(Q^2)={g_s^2\over Q^2}
 +{1\over\mu^2}\sum_{n=0}^{N-1}\theta_n
 {x^n\over(1+x)^{n+2}},\qquad x=Q^2/\mu^2.
\]

It multiplies the complete color-vector tensor, not a pion-only
projection: in the displayed Minkowski amplitude the coefficient is
-d(Q^2), since t=-Q^2. The basis correction falls as 1/Q^4 and does not alter
the leading 1/Q^2 coefficient. This is a **diagnostic spacelike
parameterization**, not a selected set of physical mediator poles,
QCD running coupling or causal continuation. The replay uses
g_s^2=1.2, mu=.8 GeV, theta=(.3,-.2,.1), domain
.01<=Q^2<=100 GeV^2; all are explicit illustrative inputs.
Out-of-domain evaluation is rejected, not assigned zero.

At raw Born order, if the declared retained amplitude is
r g_s^2/Q^2, matching supplies the remainder
(1-r)g_s^2/Q^2. Tests use r=0,.4,1. If r=1, the additional canonical
Born coefficient is zero at that order. This does **not** identify
unknown nonperturbative soft coefficients as physical zeros.
No authenticated higher-order or nonperturbative QCD target is invented.
On-shell Born agreement does not uniquely determine an off-shell kernel,
its gauge-path completion or radiative transverse structure. The Wilson
completion is a covariant effective-operator construction, not proof of
all non-Abelian Slavnov--Taylor identities or equivalence to QCD beyond
the matched order.

For finite Euclidean Fourier quadrature,

\[
F_{xy}=\sum_q w_q\,d_\theta(q_E^2)e^{iq_E\cdot(x-y)}.
\]

Coordinates are GeV^-1, w_q includes d^4q/(2pi)^4 in GeV^4,
d is GeV^-2 and F is GeV^2. Symmetric modes give a real reciprocal F.
The six-mode replay is a finite representation check, not a converged
continuum transform. Unsampled modes are unknown physical information,
not physical zeros; this finite sum does not establish long-distance
cluster decay. There is no finite-LF cell or state-normalization map here.

## Shared functional with strategic nonlinear placement

Let S_xy be a coordinate-space quark propagator of dimension GeV^3.
Retain Dirac x flavor x color ordering, 4 x 2 x 3, in every block.
W_xy transports color from y to x along an existing ordered path,
with W_yx=W_xy^dagger. Local gauge transformations act as
S_xy -> G_x S_xy G_y^dagger and W_xy -> g_x W_xy g_y^dagger.

Define, with Euclidean gamma matrices,

\[
{\cal C}_W(S)=\sum_{\mu,a,b}{\rm Ad}(W)_{ab}
 (\gamma_\mu\otimes1_f\otimes t^a)S
 (\gamma_\mu\otimes1_f\otimes t^b),\qquad
v_{xy}=F_{xy}{\rm Tr}[{\cal C}_{W_{xy}}(S_{xy})S_{yx}],
\]
\[
\Phi[S,W]=\tfrac12\sum_{xy}w_xw_y\,u(v_{xy}),\qquad
u(v)=v+{\lambda v^2\over2\mu^8}.
\]

Here w_x is a d^4x quadrature weight (GeV^-4), v has dimension
GeV^8 and Phi is dimensionless. The learned/invariant nonlinear map
sits on identified pair structures, not on an independent observable
head or a square of the total action. The finite disconnected-block
limit is additive; this is not a proof of physical continuum clustering.
More ordered invariants/channels can be needed to retain full information;
one trace invariant is not asserted to be a universal ansatz.

At lambda=0, trivial W and the vector exchange coefficient, this is
the familiar single-exchange/rainbow-ladder functional structure. It is
used as a reference, **not adopted as the production solver**.
The nonlinear term exercises the additional response terms required
by a flexible invariant neural functional; it is not demonstrated to
be a microscopic 2PI truncation, a renormalized QCD action or a positive,
causal quantum theory. It is a solver-independent Green-function
construction, not the LF quantum-state parameterization itself.

A fast exact color contraction is independently checked against all
4 x 8 generator terms:

\[
\sum_{ab}{\rm Ad}(W)_{ab}t^a S t^b
 =\tfrac12 W\,{\rm tr}_c(W^\dagger S)-\tfrac16 S.
\]

No color trace replaces the full open block: the second term and all
Dirac/flavor indices remain. The functional is real on the tested
gamma5-Hermitian propagator slice; positivity/unitarity does not follow.

Define the self-energy by
delta Phi=sum w_x w_y Tr[Sigma_xy delta S_yx]. Then

\[
\Sigma_{xy}=u'(v_{xy})F_{xy}{\cal C}_{W_{xy}}(S_{xy}),\qquad
K[\delta S]=-\delta\Sigma[\delta S],
\]
\[
\delta\Sigma_{xy}=
u'F_{xy}{\cal C}_{W_{xy}}(\delta S_{xy})
+u''\,\delta v_{xy}F_{xy}{\cal C}_{W_{xy}}(S_{xy}),
\]
\[
\delta v_{xy}=F_{xy}{\rm Tr}[
 {\cal C}_W(\delta S_{xy})S_{yx}
 +{\cal C}_W(S_{xy})\delta S_{yx}].
\]

Sigma and K[delta S] have dimension GeV^5 as coordinate kernels.
In an orthonormal quadrature representation sqrt(w_x w_y) Sigma_xy
has dimension GeV, as required for an inverse propagator. No stationary
solution of this new functional is claimed in this step.
The common self-energy/kernel consistency principle is established;
see [Munczek](https://arxiv.org/abs/hep-th/9411239).

For a local nonsinglet axial generator A_x=gamma5 tau^a alpha_x/2,
the current tensor anticommutes with A and

\[
K[A_xS_{xy}+S_{xy}A_y]=A_x\Sigma_{xy}+\Sigma_{xy}A_y.
\]

The corresponding vector/EM identity is
K[Q_x S_xy-S_xy Q_y]=-Q_x Sigma_xy+Sigma_xy Q_y.
These are consistency identities of the interaction functional,
not a computed full quark/nucleon EM vertex. Because the local
color-vector bilinears are flavor neutral, their explicit EM/axial
source response comes through the propagators and the underlying
source-dependent state equation; no arbitrary Abelian path is appended.

For an external color-path source, differentiate every occurrence of
W and W^dagger, including reversed edges. Its direct Phi derivative,
Sigma derivative and mixed state/source derivative agree between
ordered-path tangents, finite variations and PyTorch. This is an
external color-path response, **not the proton EM current**.
The broader requirement to differentiate all retained factors is
consistent with the [gauging-of-equations construction](https://arxiv.org/abs/nucl-th/9901001);
that reference's three-body solver is not imported.

## Two failures invisible to a symmetry-only check

First, axial variations give delta v=0. Dropping u'' from K can therefore
still pass the axial consistency identity. In the replay, the full
and incomplete chiral residuals are 4.11e-16 and 2.01e-16 in maximum
entry norm, but omitting u'' changes the radial kernel response by
70.99%. The full first/second variations agree with PyTorch to
5.73e-16 and 3.43e-15; independent finite differences agree to
7.26e-13 and 1.44e-12. Percentages are finite-control response changes,
not model error estimates for data.

Second, for nonlinear u,

\[
\Phi_{\rm residual}=\Phi_{\rm full}-\Phi_{\rm retained}
\ne\Phi[F_{\rm full}-F_{\rm retained}].
\]

In the simple same-u decomposition v_full=v_ret+v_res, the missed
term is

\[
\Phi_{\rm full}-\Phi_{\rm ret}-\Phi[v_{\rm res}]
={\lambda\over2\mu^8}\sum_{xy}w_xw_y v_{{\rm ret},xy}v_{{\rm res},xy}.
\]

Subtracting only the scalar coefficient before the nonlinear map misses
this mixed contribution. The replay's naive subtraction changes the
residual self-energy by 62.47%, kernel response by 57.82% and color-path
source by 69.78%. Both constructions remain invariant; conservation
alone does not detect this error. The implemented functional difference
recombines Phi, Sigma, K and source responses exactly up to roundoff.
Physical full/retained functionals need not share this simple same-u
shape; they must share external fields, units, scheme and source
definitions before matching.

## Next physical boundary

Keep the exact Born channel content, necessary LF instantaneous partner,
ordered gauge transport, nonlinear derivative terms and functional-level
subtraction. These are portable architecture constraints, not a
production choice of d_theta or u.

Next pre-evaluate a causal/spectral realization and a minimal independent
matching set that probes scalar/radial as well as vector/axial/current
information. Establish which color-singlet response positivity conditions
apply; do not impose an unproved positive spectral ansatz on colored
propagators. Decide the short-distance scheme, degrees of freedom and
retained source amplitudes before an interacting state/current solve.
Preserve the generated pion pole and its residue when that full solve is
available, with subtraction of the corresponding retained contribution.
The preceding contact reference's pion is not inserted into this new
functional as if it had been generated here.

Confinement, physical coefficients, full QCD/meson-loop matching,
analytically continued pion amplitude/norm, LF state/Hamiltonian map,
interacting nucleon currents/covariance and modular nuclear predictions
remain open. No g4 matching completion follows from a nonlinear term
whose leading counting could start at g4. No new phase hierarchy or
inherited NJL/DSE/BLFQ repair agenda is introduced.

## Reproduction and preservation

Use the previously qualified runtime; no dependencies changed:

    PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      /private/tmp/deuteron-canonical.zxUoKe/env/bin/python \
      scripts/evaluate_momentum_chiral_matching.py

The new test file has 28 checks, including independent recoupling and LF
amplitudes, local color/chiral/vector identities, full functional
derivatives, actual momentum-parameter/state mixed gradients, path-source
derivatives, nonlinear subtraction, Hermiticity, finite block additivity,
negative controls, and a serializable replay.

All **195 focused tests passed in 6.49 s**, with 12 warnings in preserved
code/dependencies (legacy docstring escapes and NumPy determinant warnings).
This includes the 28 new checks, preserved science checks and six
theory/status checks. Exact regression command in the same runtime:

    PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      /private/tmp/deuteron-canonical.zxUoKe/env/bin/python -m pytest -q -p no:cacheprovider \
      tests/test_momentum_chiral_matching.py tests/test_shared_chiral_dynamics.py \
      tests/test_chiral_soft_matching.py tests/test_nonlocal_source.py \
      tests/test_joint_state_dynamics.py tests/test_deuteron_information.py \
      tests/test_canonical_lf_network.py tests/test_graded_fock_network.py \
      tests/test_modular_gauge_network.py tests/test_modular_baryon_algebra.py \
      tests/test_current_theory_contract.py

The seven original entry/theory/roadmap files are preserved at 0875965d
and in computation_handoff/backups/2026-09-10_pre_momentum_chiral_matching.tar.gz.
All seven archive members match that commit byte-for-byte. The new TeX
section has balanced braces/environments; the theory/status hash checks
pass. These are source checks, not a rendered-document inspection.
All new source, tests and numerical results are separate. Prior physical
implementations and results remain unchanged. The main TeX receives a
focused mathematical/status update, not a full historical re-audit.
No PDF, public push or review-mirror export; the older documented
full-suite synthetic-fit tolerance failure is not altered.

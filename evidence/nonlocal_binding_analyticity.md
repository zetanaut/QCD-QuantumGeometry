# Binding-candidate review: nonlocal mass poles are not a quantum Hamiltonian

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

11 September 2026. Developed from 1cb997e2. Local only.

## Result and decision

A specific possible shortcut in the nonlocal-residual proposal is **not
qualified for direct use as our retained-mode kinetic input**. A covariant
Gaussian-smeared chiral quark interaction can eliminate real quark poles in
a constant background, but its same quadratic kernel has complex-frequency
and complex LF-energy poles. Their residues survive in the full Dirac
matrix. Removing real poles is therefore not enough to supply stable
self-adjoint quantum evolution or a physical confinement mechanism.

This is a necessary candidate-realization check, not a nucleon calculation.
It does not prove that complex gauge-fixed colored propagators are forbidden,
that all nonlocal actions fail, or that locality must be violated. It rejects
the **unqualified substitution** into the existing positive-metric state
carrier. No nonlocal NJL, diquark truncation, altered contour, indefinite-
metric auxiliary state space or production confinement model is adopted.

The existing canonical interactions, Fock states and source operators are
unchanged. The next binding candidate must specify the interaction on those
states and its matched source dependence, or explicitly justify a different
physical-state realization. A fitted colored propagator cannot substitute
for that action. No further pole-deletion/analytic-continuation repair
campaign follows automatically from this failed shortcut.

## The actual candidate sector and its limits

At the Euclidean action level, a concrete nonlocal chiral interaction has
covariantly smeared fields

\[
 \chi_{L,R}(x)=\int d^4y\,f_\Lambda(x-y)
 U_c(x,y)W_{L,R}(x,y)q_{L,R}(y),
\]
\[
 S_Y=g\int d^4x\,[\bar\chi_L\Sigma\chi_R+
                         \bar\chi_R\Sigma^\dagger\chi_L].
\]

Color and left/right external-source transport follow their prescribed
paths. Under local flavor rotations, Sigma transforms as L Sigma R-dagger;
all S/P components must be retained. Reverse transport and conjugate kernels
define the barred fields. The full proposal would also require canonical
quark/gluon dynamics, a qualified Sigma action, spurion sources and retained
subtraction. These are not fixed by writing S_Y. In particular, this study
does not select its scalar potential, solve a gap equation or show baryon
binding. It examines a necessary quadratic sector **conditional on** a
constant Sigma=v I background. Its source derivatives must include every
smearing link and field response; no complete new source implementation is
claimed here.

With f_E(p_E^2)=exp[-p_E^2/(2 Lambda^2)], the background mass term is

\[
 M_E(z)=m_c+M_0e^{-z/\Lambda^2},\quad M_0=gv>0,
 \qquad M(s)=m_c+M_0e^{s/\Lambda^2},\quad s=p^2=-z.
\]

The Minkowski metric is +---; masses and momenta are in GeV. The test treats
this as the declared **global analytic ansatz**, not a continuation inferred
uniquely from finite spacelike samples. If it is intended only on a bounded
Euclidean domain, this pole analysis tests that particular extension, not
every possible real-time completion. M0 and Lambda are illustrative, not
matched QCD parameters; mc is a current-mass source, not M0. A classical
background is not an already calculated quantum dressing. A fitted full
dressing cannot be requantized beside the same retained fluctuations without
matching/subtraction.
The invariant four-dimensional smearing here is a separate candidate, not
the old BLFQ/H2 Gaussian regulator. Its Euclidean definition is not already
a qualified real-time action or a Minkowski smearing distribution.

Nonlocal chiral quark models and pole-removal interpretations are established
prior art: [Rezaeian, Walet and Birse](https://arxiv.org/abs/hep-ph/0408233)
use a nonlocal NJL/diquark construction, which we do not adopt.
[Radzhabov and Shang, section 2](https://link.springer.com/article/10.1140/epjc/s10052-025-14592-1)
explicitly study Gaussian mass poles and the associated continuation
problems. Their pole-modification prescription is not used. Our contribution
here is the necessary compatibility test against this project's retained
Dirac/LF state interpretation, not a novel confinement theory.

## Exact chiral-limit classification

The inverse quark matrix and its scalar denominator are

\[
 Q(p)=\not p-M(s),\qquad d(s)=s-M(s)^2,\qquad
 \det Q=d(s)^2.
\]

For mc=0 every integer Lambert branch gives a pole:

\[
 s_n=-{\Lambda^2\over2}W_n(-2r^2),\quad r=M_0/\Lambda,
 \qquad Z_n={1\over d'(s_n)}={1\over1-2s_n/\Lambda^2}.
\]

Independently, positive real roots solve y exp(-2y)=r^2, y=s/Lambda^2.
The left side has its unique maximum 1/(2e) at y=1/2. Thus:

- 0<r<1/sqrt(2e): two positive real roots; their scalar invariant residues
  have opposite signs. Other branches still contain complex roots.
- r=1/sqrt(2e)=0.4288819424803534: the two roots coalesce. d=d'=0 and
  d''=-2/Lambda^2, so a simple-pole residue is not defined.
- r>1/sqrt(2e): no real invariant roots, but the leading pair is complex
  conjugate. The real-pole condition has not produced a pole-free propagator.

There are no negative real roots because d(s)<0 for s<0. The reported
finite branch checks do not truncate the infinite analytic branch set.
The scalar residue sign is recorded, not imposed as a positivity condition
on a gauge-fixed colored quark propagator.

At Lambda=.8 GeV the two leading positive-real-part frequency roots at
zero spatial momentum are:

| M0/Lambda | Invariant roots s (GeV^2) | Frequencies (GeV) |
|---|---|---|
| .30 | .0721727370, .8680857316 | .268649841, .931711185 |
| .45 | .299447943 +/- .139565048 i | .561169622 +/- .124351927 i |
| .60 | .174569663 +/- .357183125 i | .534850523 +/- .333909297 i |

Negative-frequency partners also exist. An infinitesimal real-axis boundary
prescription cannot move the isolated upper-half-frequency poles away.
This is not confusion with the ordinary negative-frequency Feynman pole.
These are first-sheet poles of the declared meromorphic function, not
resonance poles reached across a physical continuum cut. The naive
Hamiltonian retarded candidate is Q(omega+i0,k)^-1 gamma^0; multiplication
by the constant invertible gamma^0 cannot cancel those poles.
Two independent complex real/imaginary root equations agree with the
Lambert result, and direct argument-principle contour integrals count the
enclosed roots without rounding their answers to integers.

## Full Dirac and light-front consequences

At fixed real p+>0, the symmetric LF convention gives

\[
 p^-_n={s_n+p_T^2\over2p^+},\qquad
 \underset{p^-=p^-_n}{\operatorname{Res}}Q^{-1}
 ={\not p_n+M(s_n)\over2p^+d'(s_n)}.
\]

For r=.45, p+=.8 GeV and pT=(.2,-.1) GeV,
p-=(.218404964 +/- .087228155 i) GeV. The inverse Dirac matrix has two
vanishing singular values and two nonzero ones. Its numerator does not
cancel the poles. Keeping the full spin matrix therefore does not cure
this analytic issue. Longitudinal LF boosts rescale p+ and p- inversely
but do not make the energies real.

With mc=.004 GeV and M0=.36 GeV, independent numerical roots give
s=.2987033463 +/- .1496429043 i GeV^2. The upper LF pole is
.2179395914+.0935268152 i GeV. The full Dirac singular values are
(1.469922289,1.469922289,1.55e-16,1.04e-16) GeV and the LF residue norm is
2.754112351. Direct matrix inversion approaching the pole reproduces that
residue, with relative errors 5.77e-5 and 5.77e-6 for successive tenfold
smaller offsets. This is not solely a massless-limit accident.

For a self-adjoint Hamiltonian on a positive-metric carrier, its resolvent
is analytic off the real spectrum and obeys
norm[(w-H)^-1] <= 1/abs(Im w). Likewise a positive nonorthogonal Gram matrix
N and Hermitian H have real generalized eigenvalues after restriction to
the non-null image. Thus ordinary changes of path/state coordinates cannot
turn these complex kinetic poles into stable retained eigenmodes. The local
Dirac Hamiltonian is checked independently as a real-spectrum control.

This statement has a strict interpretation boundary. Gauge-fixed colored
Green functions need not themselves be positive physical-state resolvents.
[Hayashi and Kondo](https://arxiv.org/abs/2105.07487) show, under their stated
complex-singularity assumptions, that Lorentz symmetry and locality can
survive while positivity and temperedness fail; an indefinite-metric
interpretation is possible. We do not apply a scalar positivity theorem
to QCD color, or claim that our global exponential satisfies every assumption
of that reconstruction. A separately justified physical-sector and real-time
prescription could change the interpretation, but it is not supplied by
Gaussian smearing, a Ward identity, or our present state carrier.

## Reproduction, numerical correction and scope

- Source: `src/deuteron_wigner/quantum/nonlocal_binding_analyticity.py`.
- Direct tests: `tests/test_nonlocal_binding_analyticity.py`.
- Replay: `scripts/evaluate_nonlocal_binding_analyticity.py`, printing the
  saved `nonlocal_binding_analyticity_results.json` without writing files.
- Eight originals, including the bibliography: 1cb997e2 and
  `computation_handoff/backups/2026-09-11_pre_nonlocal_binding_analyticity.tar.gz`.

The first unsubdivided contour quadrature missed the empty upper-frequency
count by 1.13e-4 because a real pole lies close outside its lower edge.
Subdividing each unchanged edge into four panels resolves the cancellation;
96/144-point panel checks agree with zero within 5.1e-15. The same unchanged
rectangle counts one upper pole for r=.45 and .60. No tolerance was loosened,
pole deleted, contour moved or count rounded. Ten branch equations have
relative residual below 4.1e-15; the other invariant rectangle counts two
roots to roundoff at both resolutions. These checks verify this candidate's
analytic structure, not a physical confidence interval or new bound state.

22 new tests pass with warnings as errors (0.24 s). The focused code
integration with preserved causal, chiral, canonical and spin/path controls
passes 134 tests with six existing docstring-escape warnings (2.94 s).
All eight archived originals match 1cb997e2 byte-for-byte. The new theory
section and three source references are synchronized; a full historical
suite or manuscript audit is not claimed.
The final run including the six document checks passes 140 tests with the
same six warnings (3.13 s). Replay verification allows floating-point
roundoff, not changed physics: a bitwise comparison failed only in contour
diagnostics, whose largest change was 2.23e-16. The roots and reported
Dirac-residue results reproduce; no saved result was rewritten to hide it.
The new theory section has balanced structure and four unique labels.
Archive SHA-256:
`c1577415e441845ebd28e3ffb95b85968bc454a1c6f3cff28fa3acd7397234dd`.

**Reassessment:** reject the naive Gaussian pole-removal substitution as a
direct self-adjoint retained-mode input. Do not launch the cited models'
continuation or diquark programs. Shared binding remains open: proceed only
with an actual source-free interaction and physical-state realization, with
canonical structures, retained subtraction and source partners intact.
This result narrows a candidate route; it neither rules out the broader
nonlocal-residual architecture nor qualifies a replacement production action.

# Causal response realization and independent matching information

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. Completed bounded pre-evaluation and executable response
controls. This is **not** a causal completion of the nonlinear quark
functional, a physical spectral fit, or a new pion/nucleon solution.

Implementation: src/deuteron_wigner/quantum/causal_spectral_matching.py.
Replay: scripts/evaluate_causal_spectral_matching.py and
[numerical results](../snapshot/docs/next_level/causal_spectral_matching_results.json).

## Outcome and direction

There is a constructive finite causal representation for compatible
color-singlet **responses**, and it preserves full source matrices. It does
not justify imposing positive spectral weights on the colored interaction
kernel or its matched remainder. It also does not uniquely determine the
shared dynamics from pion poles and spacelike two-point information.

The new decisive findings are:

- A nonzero positive additive spectral measure changes the leading UV
  moment. It cannot supply only the previous diagnostic's 1/Q^4 correction
  while leaving an already fixed 1/Q^2 coefficient unchanged.
- Full and retained responses can both be positive while their causal
  difference is not. Positive-clipping the remainder changes matching.
- Two positive nine-pole responses agree on four spacelike values and two
  UV moments to 5.56e-17, but differ by 88.32% in one resolved spectral bin.
  This percentage is a constructed ambiguity, **not a QCD uncertainty**.
- Even the *entire* zero-source two-point response does not fix a scalar
  insertion. In a shared Gaussian example, two source couplings give that
  same response but differ by 77.16% in its scalar derivative.
- A genuine field-coordinate redundancy is different: transforming both
  the action and every source preserves responses. Holding sources fixed
  produces errors of 8.16%, 41.88% and 64.92% in three controls.

**Advance with a source-complete common-action construction; do not launch
an unconstrained spectral fit or promote the previous spacelike ansatz.**
The immediate next experiment should connect a finite causal realization
to the *same* action's chiral state equation and independent scalar/current
responses, with explicit physical source normalization and a small set of
scheme-qualified targets. The response representation here is a diagnostic
constraint and possible computational representation, not an independent
observable head or a replacement for canonical quark/gluon dynamics.

Spectral dispersion, Gaussian realization and field-redefinition invariance
are established tools, not claimed inventions. The project-specific work is
their joint application to our full-source, retained-subtracted construction,
the explicit failure controls, and the resulting placement/matching
requirements for flexible shared dynamics. No novelty or quantum advantage
is inferred from these controls.

## 1. Which object is positive?

For compatible Hermitian, gauge-invariant source operators in a positive-
metric physical Hilbert space, the positive-frequency spectral matrix has
the Gram form rho_ab=sum_n <0|O_a|n><n|O_b|0> delta(s-s_n). It is positive
semidefinite. The statement applies within the appropriate quantum-number,
Lorentz/tensor and conjugation conventions, not to every scalar coefficient
of every tensor decomposition. Do not mix scalar, vector and axial channels
in an arbitrary positive matrix as if symmetry allowed their cross terms.

Colored, gauge-fixed propagators are different. For a concrete primary
analysis of signed gluon spectral structure and its assumptions, see
[Cyrol et al., Reconstructing the gluon](https://arxiv.org/abs/1804.00945).
No positivity theorem or confinement prescription for colored quark/gluon
propagators is selected from this paper. A causal signed representation is
not automatically an admissible microscopic colored theory either.

Our finite normalized reference is

\[
C(z)=\sum_n {R_n\over s_n-z},\qquad s_n\geq0,\quad R_n=R_n^\dagger,
\qquad C_E(Q^2)=C(-Q^2).
\]

s,z,Q^2 are GeV^2; R is dimensionless and C is GeV^-2. Physical source
factors restore current-specific dimensions. `physical_response` requires
R_n positive semidefinite; `signed_remainder` does not. These labels mark
spectral positivity classes, **not complete local-QFT validation**.

For Im(omega)>0, C_R(omega,k)=C(omega^2-k^2), using exp(-i omega t).
All finite poles lie on the real invariant axis; the upper-frequency-half-
plane is analytic. With E_n=sqrt(s_n+k^2)>0,

\[
C_R(t,k)=\theta(t)\sum_nR_n{\sin(E_nt)\over E_n},\qquad
C_E(\tau,k)=\sum_nR_n{e^{-E_n|\tau|}\over2E_n}.
\]

Independent time integration agrees with the frequency resolvent. The
reflection matrix K_(ia,jb)=C_E(t_i+t_j)_ab for t_i>0 is a Gram matrix when
R_n is positive semidefinite. A signed example with residues (-.1,1.1) at
s=(.25,1) has **positive C_E(Q^2) for every Q^2>=0**, yet its tested
reflection matrix has eigenvalue -0.0093142. Positive spacelike values alone
therefore do not suffice.

For real compatible source residues, the finite representation has an
explicit free-field quadratic realization C=B^T(M^2-zI)^-1B. This supplies
a sufficient linear response realization, not the nonlinear interaction,
crossing/locality for arbitrary source matrices, multipoint unitarity,
confinement, physical continuum or interacting LF covariance. In particular,
none of these follows for the previous quark functional by substituting a
spectral scalar coefficient into it. General complex matrices still require
their physical conjugation/crossing and quantum-number restrictions.

## 2. Short-distance budget, subtraction and retained physics

In the **finite, unsubtracted** reference,

\[
C_E(Q^2)={M_0\over Q^2}-{M_1\over Q^4}+O(Q^{-6}),\qquad
M_j=\sum_ns_n^jR_n.
\]

If an additive soft measure is positive, its M0 is positive semidefinite;
M0=0 forces every positive residue to vanish. Thus a nonzero positive
addition cannot preserve a saturated M0. One must redistribute existing
strength, use a signed *remainder*, change the UV assumptions, or justify a
different representation. This is a simple spectral moment theorem, not a
new QCD no-go result. The previous fixed-coupling Born coefficient is not the
running, subtracted, renormalized QCD UV correlator. Divergent continuum
moments cannot be replaced by these finite moments without a scheme.

For physical current correlators with necessary subtractions,

\[
C_E(Q^2)=\sum_{j=0}^{N-1}c_j(Q^2-Q_0^2)^j
+[-(Q^2-Q_0^2)]^N\int ds\,
{\rho(s)\over(s+Q_0^2)^N(s+Q^2)}.
\]

The c_j are separately supplied renormalized contact/subtraction constants;
they are not positive spectral weights or default zeros. The implementation
tests the remainder against independently reconstructed finite Taylor
contacts for N=1,2,3. It does not claim a renormalized QCD dispersion integral.

For example, full residues (.6,.4) at s=(.25,1.2), and a retained residue
1 at s=.25, are separately positive. Their residual is

\[
C_{\rm res}(z)=-{.4\over .25-z}+{.4\over1.2-z}.
\]

It is causal and has zero M0, yet C_res(-.5)=-.298039 GeV^-2 and the
reflection control has eigenvalue -0.378623. Exact full=retained+residual
recombination holds. This residual is not an additional positive-norm
particle sector. For nonlinear dynamics, the preceding **functional**
subtraction remains mandatory: Gamma_res=Gamma_full-Gamma_ret, with common
fields, scheme, source definitions and variations. A difference of response
matrices here is a control, not a replacement for that full matching.

## 3. What data determine, and what they do not

The nine-pole experiment fixes s in [.25,4] GeV^2, four spacelike values at
Q^2=(.2,.6,1.4,3), and M0/M1. The resulting 6x9 linear map has a three-
dimensional nullspace. Two strictly positive residue vectors in it have
identical supplied information. At z=2.125+.025i GeV^2 their Lorentz-smeared
spectral strengths differ by 88.32%; their Euclidean-time responses differ
by only 3.95e-5 in the reported finite sampled norm.

Three independently selected spectral bins resolve this **known finite
family** in noiseless arithmetic; residue recovery error is 1.77e-13. The
unit-row-normalized condition number is still 2.05e4, versus 3.77e8 for
seven spacelike points plus the same two moments. This is improved but still
ill-conditioned, not a stable physical fit or a universal count of needed
measurements. Real covariance, finite resolution, threshold/continuum
uncertainty and model enlargement must be included. Complete exact data on
an analytic continuum are a different uniqueness problem. See
[Fei, Yeh and Gull](https://arxiv.org/abs/2010.04572) for a primary discussion
of analytic continuation and positivity-compatible families; their
reconstruction algorithm is not adopted here.

Even complete diagonal spectra omit mixed-source information: changing the
relative sign of two same-quantum-number source overlaps at one pole leaves
both diagonal responses unchanged but alters their off-diagonal response.
The reference tests preserve that interference rather than assigning
independent positive scalar fits to all observables.

## 4. Representation freedom versus missing physical source couplings

Take real M^2>0 and source matrix B in a shared quadratic action, with
A(z)=M^2-zI and C=B^TA^-1B. For an invertible analytic field change T(z),

\[
A'=T^TAT,\qquad B'=T^TB,\qquad B'^T A'^{-1}B'=B^TA^{-1}B.
\]

Use transpose, not Hermitian conjugation of the complex argument. Our
T=I+(.7 z/GeV^2+.3h)N has N^2=0 and det T=1, so its inverse is explicit
and the normalized Gaussian measure brings no missing Jacobian. Internal
propagators change by 7.29%--83.69%, but properly transported responses
agree within 6.20e-16 relative. Freezing B instead produces the failures
listed above. With M^2(h)=M^2+hC_s, both first and second scalar-source
derivatives also agree, including actual PyTorch differentiation of T(h).

At h=0, replacing C_s by C_s+.25 GeV^2 I leaves the entire two-point
response unchanged while changing dC/dh by 77.16% at z=-.4 GeV^2. The
identity

\[
\partial_h C=-B^TRC_sRB,\qquad
\partial_h^2C=2B^TRC_sRC_sRB,\qquad R=(M^2-zI)^{-1}
\]

is checked independently. These h couplings are illustrative scalar
sources, not a quark-mass calibration. The difference is missing *physical
source information*, unlike the redundant T coordinate. General nonlinear
field changes require the full Jacobian, source partners and consistent
order/scheme treatment; the exact Gaussian example does not excuse their
omission. See [Criado and Perez-Victoria](https://arxiv.org/abs/1811.09413).

## 5. Pion pole and minimal independent matching categories

The earlier generated reference pole is transported, **not newly solved**.
For normalized source pair (P, div A/mu),

\[
a={1\over\mu^2}\left(G_\pi,{f_\pi m_\pi^2\over\mu}\right),\qquad
R_\pi=aa^T,\qquad a_A={2m\over\mu}a_P.
\]

PCAC fixes the relative PP/PA/AA pole residues, not just its mass.
With mu=.8 GeV and the preserved finite-mass solution, 35% retention leaves
65% of the **whole** residue matrix in the residual (error 8.5e-22).
Restoring mu^4 gives correlator units GeV^2. This supplies neither Ward
contact polynomials nor transverse axial/EM currents. Protecting this
transported pole does not insert it into the new quark functional as a
generated solution. Finite-mass PCAC is used; the separate chiral-limit
source normalization must not be obtained by dividing by m=0.

There is no universal finite minimum matching set before the shared action,
retained content and parameter rank are specified. The necessary categories
are nevertheless clear:

| Information | What it constrains | Boundary to preserve |
|---|---|---|
| Canonical channel amplitudes, scale/scheme and perturbative short-distance terms | QCD correspondence and retained subtraction | Born matching does not fix off-shell/radiative completion or all UV moments |
| Generated pion pole, f_pi and pseudoscalar residue/PCAC | Chiral pole, norm and linked longitudinal sources | Separate contact constants and transverse response; no imported pole as a solve |
| Scalar susceptibility and/or scalar three-point form factor, with mass dependence | Radial curvature and scalar-source coupling | Scheme, disconnected contributions and contact terms; not another pion-only target |
| Vector and transverse axial spectral/correlator information with covariance | Response strength and channel dependence invisible to the pion pole | Include tensor projections, continuum/threshold and source normalization |
| Mixed-source insertions and resolved momentum dependence | Interference, higher functional derivatives and weak parameter directions | Only symmetry-allowed mixed channels; a two-point fit is not full dynamics |
| Later p/n four-current and cross-frame withheld observations | Physical state/current closure and predictive ability | Not enforced by rescaling or independent per-observable heads |

Concrete source leads verified during this step are the
[updated ALEPH vector/axial spectral analysis](https://arxiv.org/abs/1312.1501),
whose update corrects unfolded-bin correlations, and a
[primary lattice pion scalar-form-factor calculation](https://arxiv.org/abs/2503.20687),
which includes disconnected contributions. The
[FLAG review](https://arxiv.org/abs/2411.04268) is a route to underlying
scheme-qualified mass/decay-constant calculations, not a substitute for
their primary inputs. These are **identified sources, not imported data or
selected calibration**. Do not combine their normalizations, flavor content,
isospin corrections, finite-volume effects or uncertainties by assumption.
No unavailable coefficient or covariance was filled with zero.

## Next bounded implementation

Construct a **small source-complete causal common-action candidate** using
the preserved canonical quark/gluon and ordered-path structures, keeping
nonlinear flexibility on identified invariant interactions. Make temporal
realization explicit: a Hermitian enlarged realization and a compatible
nonlocal elimination must carry the same sources; a spectral scalar
coefficient alone does not establish this for the nonlinear action. The
candidate must generate its own dressing/pion response, not attach this
reference pole or independently fitted spectra. Compare scalar and
transverse/mixed responses of that same construction before p/n fitting.

The immediate deliverable is the concrete action/source realization and
its first discriminating matched-response experiment, not another generic
positivity/graph demonstration or a full spectral-reconstruction campaign.
Select a finite, explicit parameterization and matching window for that
experiment, audit which source data can actually enter with consistent
units/scheme, and assess remaining rank and sensitivity. If realizing it
requires a new many-body/renormalization program, report that cost and
direction choice before making it a project prerequisite. Causality of
two-point controls, chiral covariance and complete derivatives are necessary
checks, not sufficient proofs of the resulting physical theory.

No production model has been selected; BLFQ/H2 remains preserved, not
resumed. Nucleon component/frame covariance, nuclear many-body matching and
GTMD/TMD/GPD predictions remain later physical tests. This result narrows
how shared dynamics must be represented and constrained; it does not move
those observables into the completed column.

## Reproduction and preservation

Use the existing qualified runtime (no dependency changes):

    PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      /private/tmp/deuteron-canonical.zxUoKe/env/bin/python \
      scripts/evaluate_causal_spectral_matching.py

The new 27 tests cover analytic/time-domain and reflection controls, UV
moments, subtraction, PCAC residue transport, positive-spectrum nullspaces,
independent matching probes, exact field/source congruence and actual
first/second Torch derivatives. All **222 focused tests passed in 5.98 s**,
including 27 new checks, 189 preserved science checks and six theory/status
checks, with 12 existing warnings (legacy docstring escapes and NumPy
determinant warnings). This is not a new complete-suite execution.

    PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      /private/tmp/deuteron-canonical.zxUoKe/env/bin/python -m pytest -q -p no:cacheprovider \
      tests/test_causal_spectral_matching.py tests/test_momentum_chiral_matching.py \
      tests/test_shared_chiral_dynamics.py tests/test_chiral_soft_matching.py \
      tests/test_nonlocal_source.py tests/test_joint_state_dynamics.py \
      tests/test_deuteron_information.py tests/test_canonical_lf_network.py \
      tests/test_graded_fock_network.py tests/test_modular_gauge_network.py \
      tests/test_modular_baryon_algebra.py tests/test_current_theory_contract.py

The seven original entry/theory/roadmap files are preserved at 0ad7a54c and
in computation_handoff/backups/2026-09-10_pre_causal_spectral_matching.tar.gz.
All seven originals match that commit byte-for-byte; one macOS metadata
sidecar is also preserved. New TeX braces/environments and source hashes
are checked, without rendering a PDF.
Earlier implementations/results are unchanged. The main TeX receives a
focused mathematical update, not a full historical re-audit or PDF build.
No public push or review-mirror export is included.

# Transverse source matching: radius, two-photon response and remaining freedom

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. **The common action can carry the missing charge-radius
structure without violating current conservation. Radius matching alone
does not determine the independent two-photon response.** This bounded
experiment is complete; it does not select a production sigma model.

Implementation: src/deuteron_wigner/quantum/transverse_source_matching.py.
Replay: scripts/evaluate_transverse_source_matching.py and
[numerical results](../snapshot/docs/next_level/transverse_source_matching_results.json).

## Result and implication for the project

Two local, chiral/gauge-covariant source operators are added to the preceding
common-action reference. Actual action derivatives reproduce the transverse
four-current, its required two-source contacts and the real Compton
structure response. Local non-Abelian connection transformations and
Lorentz transformations pass; omitting the commutator or contact terms fails.
Both operators vanish at zero external sources: no new potential, vacuum,
scalar radius, quantum quark dressing or composite state is manufactured.

The charge-radius calibration fixes c=1.8588598 GeV^-2. The explicitly
restricted d=0 hypothesis then gives alpha_pi=14.9351 x 10^-4 fm^3,
versus COMPASS 2.0(0.6)_stat(0.7)_sys x 10^-4 fm^3 under alpha=-beta.
This fails the independent comparison at the declared tree order.
Allowing the second invariant gives d=-0.8049674 GeV^-2 and reproduces both
inputs. **This is a two-input calibration, not an independent prediction.**
No significance, loop-error bound or all-orders exclusion is claimed.

The original three mixing/potential choices give the same fitted light-mode
responses. Nevertheless, the six entries of the two symmetric coefficient
matrices have information rank two, leaving four unfixed directions in the
fixed, nondegenerate mode basis. An explicit change in an unfixed direction
leaves both light observables invariant but changes a heavy/light off-shell
current witness. More accurate fitting of the same two numbers cannot
identify these missing interactions. The witness is not a physical decay
prediction for the unqualified heavy collective mode.

**Architectural consequence:** learn coefficients of shared invariant
operators and generate their source partners by differentiation. Keep
source dynamics distinct from source-free state dynamics, and transport
unidentified coefficients/uncertainty rather than zero-filling them.
The negative d is an interaction coefficient, not a negative probability;
physical two-point spectral positivity must not be imposed on every local
source coefficient. Full amplitude/dispersion/UV admissibility is a separate,
uncompleted requirement. These tests reuse the existing Torch capability;
they do not demonstrate new quantum advantage or an optimized DNN geometry.

## Action, conventions and established ancestry

Use the preceding two real O(4) multiplets phi_i=(sigma_i,pi_i^a), i=1,2.
For the external flavor connection A_mu, D_mu=partial_mu+A_mu and

\[
\mathcal F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu+[A_\mu,A_\nu],
\]
\[
\Delta\mathcal L=-C_{ij}(D_\mu\phi_i)^T\mathcal F^{\mu\nu}D_\nu\phi_j
 +D_{ij}\phi_i^T\mathcal F_{\mu\nu}\mathcal F^{\mu\nu}\phi_j.
\]

C,D are real symmetric matrices, GeV^-2; fields are GeV, connections GeV,
curvature GeV^2, and L is GeV^4. All Lorentz contractions use (+---).
This flavor curvature is **not the canonical QCD color field strength**.
Canonical spin/color/Fock and ordered color-path implementations remain
unchanged. Using a connection curvature is geometric structure, not a
proof of topological order or a topological invariant of the nuclear state.

These are established external-source derivative-operator types, related
to the radius/two-source structures of chiral effective actions. The
underlying external-source/Ward construction is due to
[Gasser and Leutwyler](https://doi.org/10.1016/0003-4916(84)90242-2);
the distinction between polynomial low-energy coefficients, loops and
nonperturbative matching is also explicit in the primary analysis of
[Mateu and Portoles](https://arxiv.org/abs/0706.1039).
We do not adopt that paper's resonance model or identify C,D with a
renormalized L9/L10 scheme by notation. This experiment's contribution is
their resolved shared-action implementation, source-complete checks and
information comparison within this project's architecture, not discovery
of the general operator types.

Under a local real orthogonal transformation R(x),
phi'=R phi, A'_mu=R A_mu R^T-(partial_mu R)R^T, and F'=R F R^T.
This proves invariance of both terms. Tests independently transform fields,
their first derivatives, connections and connection derivatives, including
the noncommuting derivative terms. Abelian EM checks alone would miss the
incorrect omission of [A_mu,A_nu].

The approximation is a weak-external-source tree derivative expansion.
There are no new zero-source propagator poles or kinetic terms; contact
responses are local at this order. This does not establish a UV completion,
strong-background hyperbolicity, full quantum causality, chiral loops,
or a completed QCD-matched collective sector. No indefinite derivative
resummation or resonance prescription is inferred.

## One-source current and the required two-source partners

For normalized light mode e, define c=e^T C e and d=e^T D e. With
q=p'-p, t=q^2, the action gives the electromagnetic correction

\[
\Delta\Gamma^\mu(p',p)=c\{t(p'+p)^\mu-(p'^2-p^2)q^\mu\},\qquad
q_\mu\Delta\Gamma^\mu=0.
\]

The formula is checked against three actual field/source derivatives of
the action, not used as its own oracle. It is transverse even off shell.
Dropping the second term and using an on-shell form-factor multiplier
off shell fails. At equal external mass,

\[
F_V(t)=1+ct+O(t^2,\text{loops}),\qquad
\langle r_V^2\rangle=6(\hbar c)^2 c.
\]

The D term has no one-source vertex. It starts with two sources. For two
incoming photons q1,q2 and p'=p+q1+q2, the actual contact S obeys

\[
q_{1\nu}S^{\nu\mu}
=\Delta\Gamma^\mu(p',p+q_1)-\Delta\Gamma^\mu(p'-q_1,p).
\]

The numerical check contracts the fourth action derivative with q1 and an
independent second polarization. Keeping the corrected one-source vertex
but omitting this contact fails. The non-Abelian construction also retains
vector/axial and mixed source partners; it is not an EM-only output head.
The full general Compton amplitude requires pole/Born terms as well as
contacts. The contact by itself is not represented as that full amplitude.

## Independent real-photon reduction and polarizability

For real transverse photons, the C one-source correction contracted with
physical polarization vanishes even for an off-shell intermediate pion.
Thus the corresponding mixed Born corrections vanish in this particular
real-photon comparison. Independently, integration by parts and
[D_mu,D_nu]=J3 F_mu_nu give, modulo total derivatives and the source Maxwell
divergence term,

\[
\Delta\mathcal L\simeq-(c+2d)e_{\rm em}^2\pi^+\pi^-F_{\rm em}^{\mu\nu}
F^{\rm em}_{\mu\nu}.
\]

The charge-stripped structure amplitude for incoming k, outgoing k' is

\[
\Delta T=-4(c+2d)[(k\cdot k')(\epsilon\cdot\epsilon'^*)
 -(k\cdot\epsilon'^*)(k'\cdot\epsilon)].
\]

Direct fourth action derivatives agree with this independent reduction.
Restoring e_em^2=4 pi alpha_em and using the relativistic-to-nonrelativistic
normalization pi_rel=exp(-imt)psi/sqrt(2m), with
L_NR=2 pi (alpha_pi E^2+beta_pi B^2) psi^dagger psi, yields

\[
\alpha_\pi=-\beta_\pi=\frac{2\alpha_{\rm em}}{m_\pi}(c+2d)(\hbar c)^3.
\]

Here alpha,beta are fm^3. This normalization is checked separately from the
matrix differentiation. No anomalous radiative weak vector amplitude or
full pion decay calculation is supplied by these parity-even operators.

## Inputs, matching and predictivity boundaries

Use the [PDG radius .659(4) fm](https://pdglive.lbl.gov/DataBlock.action?node=S008CR)
and [charged-pion mass .13957039 GeV](https://pdglive.lbl.gov/DataBlock.action?node=S008M).
The radius was withheld in the preceding experiment; it is now openly a
calibration input. The additional independent source is the
[COMPASS primary result](https://arxiv.org/abs/1405.6377), which assumes
alpha=-beta and includes radiative, chiral-loop and nuclear corrections
in extracting its quoted summary. We use that summary, not raw event data
or an automatically combined likelihood. No precision statistical rejection
or renormalized QCD coefficient extraction is made.

The replay retunes a separately labeled charged-mass variant of the old
potential at each of its three mixing choices. Its .515 fm^2 scalar shape
is still the approximate Nf=2+1 isospin benchmark from the preceding note;
it is not promoted to a precision charged-pion matching condition. Source
operators leave that variant's source-free scalar result unchanged. The
original .1348 GeV isospin calculation is preserved. Substituting that mass
in the restricted polarizability conversion gives 15.4636 x 10^-4 fm^3,
so the qualitative failure is not a charged/isospin mass-convention effect.

The deliberately restricted d=0 trial is not a default assignment of
unknown physics to zero. It tests whether the radius operator alone is
enough; the answer is no. Joint matching instead determines

\[
c=\frac{r_V^2}{6(\hbar c)^2},\qquad
d=\frac12\left[\frac{m_\pi\alpha_\pi}
 {2\alpha_{\rm em}(\hbar c)^3}-c\right].
\]

One-input-at-a-time uncertainty scans are recorded, not a joint posterior.
The linear form-factor diagnostic at Q^2=.02,.05,.1 GeV^2 is
(.962823,.907057,.814114); these are truncated calibrated outputs, not
validated withheld predictions. The size of c Q^2 reaches .186 in that
window; higher-order and loop errors are not bounded here. Do not extend
the polynomial to a timelike spectral model or high-energy QCD prediction.

In the fixed two-mode basis, radius measures e^T C e and polarizability
measures e^T(C+2D)e. The corresponding 2x6 information matrix has rank two.
For h perpendicular to e, terms proportional to eh^T+he^T or hh^T in either
matrix leave these light projections unchanged. They are not automatically
field-coordinate redundancies: a fixed-basis transition probe changes.
Conversely a genuine simultaneous orthogonal change of fields, C,D and
mode vectors leaves the entire source action unchanged, as tested.

For a declared same-basis retained action, subtract Delta L_full-Delta L_ret
before source variations. Because this particular coefficient basis is
linear, the result is C_res=C_full-C_ret and D_res=D_full-D_ret. Both action
and first/second source responses recombine. Subtracting only the radius
coefficient while retaining the full D double counts a two-source term.
This is a functional algebra control, not an inferred QCD subtraction map.
Nonlinear field-dependent coefficients require full functional subtraction.

## Reassessment and next bounded question

Retain the two invariant source-operator classes and their derivatives.
Reject radius-only inference, arbitrary current heads and default closure
of unidentifiable matrix directions. The next question is the
**full-minus-retained matching map between these collective source responses
and the quark-resolved shared state/current network**. Pre-evaluate the
smallest amplitude/source comparison that identifies shared versus
independent coefficients and prevents double counting. Do not silently
transfer fitted pion coefficients to nucleon heads or start more potential
fitting, a resonance/loop repair campaign, or BLFQ/H2 reconstruction.

Source-free nonperturbative quark dynamics and QCD matching still need to
generate physical p/n states and currents. This experiment has not solved
those, interacting nucleon covariance, arbitrary-nucleus matching or
GTMD/TMD/GPD predictions. Its role is to specify necessary source dynamics
and the independent information the original quantum/geometric construction
must preserve.

## Verification and preservation

Use the existing runtime, without dependency changes:

    PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      /private/tmp/deuteron-canonical.zxUoKe/env/bin/python \
      scripts/evaluate_transverse_source_matching.py

The same environment runs tests/test_transverse_source_matching.py plus
the exact regression list in [the preceding result](common_action_pilot.md#verification-and-preservation),
including tests/test_current_theory_contract.py. **275 focused tests pass
in 7.05 s**: 24 new source checks, 245 preserved science checks and six
theory/status checks, with the same 12 existing warnings. Fresh replay
matches all saved numerical/decision fields exactly. This is not a
full-suite pass; the older synthetic-fit tolerance failure remains recorded
and untouched. New TeX braces/environments, unique labels, source hashes and
index pointers are verified. Originals are at 2b80df33 and in
computation_handoff/backups/2026-09-10_pre_transverse_source_matching.tar.gz.
All seven archive members match that checkpoint byte-for-byte.
Earlier implementations/results are not edited. Main TeX receives a focused
update, not a full historical re-audit or PDF build. No public push.

# Nonzero-momentum quark interaction and four-current matching

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. Local continuation from `f35d8493`. Two sequential
experiments pass: the resolved scalar/pion exchange current, then its
combination with the canonical QCD color-vector Born current. All four
components agree with independent light-front propagator constructions and
transform correctly with the actual external spin bases. This is a tree
amplitude qualification, **not a physical nucleon, quantum-matched action,
or repair of the old BLFQ covariance violation**.

## What is retained and what is not inferred

The user-selected interpretation remains an action whose retained modes are
quantized once. This test uses its local tree S/P block with both collective
multiplets and canonical quark/gluon amplitudes, without integrating a
spacetime loop or introducing a finite Fock/momentum projector. The prior
homogeneous quantum-cell response is **not** substituted for a Lorentz
propagator. The former tree masses/couplings supply numerical controls,
not quantum-rematched coefficients or newly fitted hadron parameters.

The direct labeled two-quark amplitude has 12 states per endpoint:
spin(-,+) x flavor(u,d) x color(0,1,2). The current is four 144x144 complex
matrices. No spin/flavor/color averaging replaces the amplitude. Outgoing
exchange and direct-minus-exchange antisymmetrization are checked separately.
These quarks are external partonic controls, not asymptotic physical colored
particles or a baryon wavefunction. No probabilities, cross sections or
observable fits are inferred from their raw matrix norms.

The numerical domain is the explicitly listed nonzero spacelike exchanges,
off-pole quark routes and nonzero signed internal plus momenta in
[the replay](../snapshot/docs/next_level/nonzero_momentum_current_results.json). The main four cases
span exchanged Q² about .106--.290 GeV². The separate negative-plus test is
a wider kinematic algebra control, not an extension of a fitted EFT window.
There is no physical UV split, matching cutoff or continuum quantum error
estimate. In particular, neither these arbitrary test momenta nor the old
cell's 2pi/L is a QCD matching scale.

Local color-singlet S/P bilinears need no color Wilson line between their
own coincident endpoints. The ordered nonlocal path and complete canonical
color machinery remain preserved for the broader action; no local test
licenses deleting those paths, replacing gluon exchange by S/P channels,
or treating the reference as a complete QCD operator basis.

## Shared amplitude and attached current

Use metric (+---), symmetric p±=(p0±p3)/sqrt(2), ubar u=2m. Suppress overall
S-matrix i and the electric coupling e. With B=(1,i gamma5 tau^a),

\[
 R_X(k)=(H_X-k^2 I)^{-1},\qquad
 W_X(k)=g_X^T R_X(k)g_X,\qquad X=S,P.
\]

The tree exchange amplitude is W_S times the scalar bilinear product plus
W_P times the sum of all three pseudoscalar products, with explicit color
identities. This sign follows local L_Y=-qbar g(phi)q and the usual scalar
propagator. The canonical vector amplitude is separately
g_s² J_1^alpha J_{2,alpha} (t_a tensor t_a)/k². It is not a Fierz-truncated
S/P replacement, and its sign follows the preserved canonical convention.

For incoming photon q, let r=p1'-p1, s=p2'-p2 and q=r+s. On quark line 1
the two external attachments are

\[
 W_A(s)\,\bar u_1'\left[
 B_A S(p_1+q)\gamma^\mu Q+
 \gamma^\mu Q S(p_1'-q)B_A\right]u_1\,
 (\bar u_2'B_Au_2),\quad Q=\mathrm{diag}(2/3,-1/3).
\]

Add the interchanged line-2 expression with W_A(r). The current on charged
pion exchange is obtained from the same previously derived source action:

\[
 \Gamma^\mu(r,-s)=(r-s)^\mu I+
 C\left[q^2(r-s)^\mu-(r^2-s^2)q^\mu\right],\qquad
 I^\mu=g^T R_P(r)\Gamma^\mu(r,-s)R_P(-s)g.
\]

Write b_ja=ubar_j' i gamma5 tau^a u_j. The internal contribution is
i(b_11 tensor b_22-b_12 tensor b_21) I^mu. Its contraction is the same
flavor tensor times W_P(r)-W_P(s), cancelling the external-line contraction.
Neutral scalar and pi3 attachments commute with electric charge. This
derivation is also checked numerically using the full external matrices;
the current is not manufactured by solving q.J=0.

The independent transverse C changes the current without changing that
contraction. The D invariant begins at two external photons and is absent
from this one-photon calculation; its coefficients are **not set to zero**.
The replay uses C=[[1.7,.35],[.35,-.2]] GeV^-2 explicitly as an unfitted
non-diagonal test, not the earlier pion-calibrated projection.

For QCD color-vector exchange, attach the photon to both ends of each
quark line. The gluon is electrically neutral. The radiating quark line is
conserved in its gluon index only after those attachments are summed and
the complete fermion propagator is used. No EM current is attached to an
auxiliary color leg. The color trace distinguishes this tensor from S/P;
neither is discarded when forming the combined current.

## Independent LF and frame/component controls

The covariant propagator uses slash(k) directly. The independent LF
calculation instead sums canonical u or crossed v spinors:

\[
 S(k)=\frac{\slashed{k}_{\rm on}+m}{k^2-m^2}
       +\frac{\gamma^+}{2k^+},\qquad
 k^-_{\rm on}=\frac{k_T^2+m^2}{2k^+}.
\]

For k+>0 the numerator is sum u ubar; for k+<0 it is -sum v(-k_on)
vbar(-k_on). Negative internal plus is not silently discarded. The explicit
pair-route test has minimum k+=-1.29926 GeV with future external quarks.
The code excludes poles and exactly k+=0; it does not supply a loop
zero-mode/endpoint prescription by extrapolation.

The LF version also independently diagonalizes each collective Hessian and
uses its signed LF energy denominators:
R(k)=-sum_n e_n e_n^T/[2k+(k--(M_n²+k_T²)/(2k+))]. This reconstructs
the matrix inverse H-k² for both signs of k+. It is not a complete
multi-particle time-ordered Hamiltonian or nonperturbative reduction.

Independently, each QCD exchange uses the two physical A+=0 gluon
polarizations plus -J1+ J2+/(k+)². Their sum matches J1.J2/k² when the
complete quark line currents are used. Both instantaneous quark and gluon
parts are required. This is an amplitude-level LF decomposition, **not**
an instantaneous potential or a constructed nonperturbative LF Hamiltonian.

Under Lorentz boost Lambda, the current is compared component by component
after transforming all momenta and transporting the LF spin bases with
U(p)=ubar_LF(Lambda p) S(Lambda) u_LF(p)/(2m). These U are independently
checked to be unitary. The tensor comparison is

\[
 J'^\mu=(U_1'\otimes U_2')\Lambda^\mu{}_{\nu}J^\nu
           (U_1\otimes U_2)^\dagger.
\]

This tests more than a conserved contraction or spin-summed norm. Reversal
gives the conjugate-transposed current, and the antisymmetrized amplitude
also remains conserved. No nucleon magnetic extraction is performed.

## Executed results and representation matching

Four noncollinear kinematic cases give maximum combined LF/covariant
relative difference 8.73e-16 and normalized Ward defect 3.97e-16.
Three longitudinal/transverse/oblique boosts give maximum full-component
relative difference 2.33e-14. In the first case:

- Dropping the collective current leaves Ward matrix norm 17.4535 instead
  of roughly 4e-13 for the complete current.
- Dropping instantaneous quark propagation changes S/P current by 9.70%
  and canonical vector current by 4.55% in matrix norm.
- Dropping instantaneous gluon exchange changes the canonical current by
  135%. Across the four controls the defect is 126--174%.
- Removing C changes the current norm by 4.89419 GeV^-1 while its Ward
  contraction stays below 1.3e-14. Conservation does not determine C.

These percentages are diagnostic tensor differences, not errors on a
measured form factor. They are not explanations or repairs of the old
nucleon covariance discrepancy.

An explicitly identified comparison retains the lowest scalar and pion
mass eigenmodes; it is not a momentum/UV cutoff or a physical instruction
to discard the other modes. The matched remainder is the full induced
amplitude/current minus that retained amplitude/current. At zero source,
light plus heavy exchange reconstructs the full response. At one source,
heavy-light transitions are also required: keeping only the heavy diagonal
misses a current coefficient with norm .209020 GeV^-3 in the control.
The full-minus-retained current has norm 4459.31 GeV^-1 and conserved
contraction 1.25e-13. Source subtraction must follow the functional and
endpoint identification, not an equality of matrix dimensions or a
source-free coefficient subtraction.

## Reassessment and next work

The nonzero-momentum tree current is now qualified for the retained local
reference and canonical vector block, without rescaling, a momentum
regulator, finite Fock truncation or BLFQ/H2. Preserve it as an independent
control of future approximations. The result does **not** make local
tree-calibrated coefficients quantum matched, define the full flexible
residual action, or solve the proton/neutron.

Continue the persistent effort with a pre-evaluated, limited **nonzero-
momentum quantum matching correction** for the same source-defined retained
action. Declare retained/excluded quantum content, subtraction conditions,
order/domain and all linked source partners before calculations. Reuse the
current tree control as the zero-loop limit and require an independent
covariant/LF comparison where it is actually available. No standalone
observable heads, physical hadron fit to the homogeneous cell, automatic
NJL/sigma-model completion agenda, or claim that Ward closure proves
nonperturbative covariance. Unknown coefficients alone do not stop the
parameterized investigation. Reassess before a growing loop/renormalization
campaign or a state solve that would implicitly select the production model.

Then quantum-level joint state/interaction matching, generated p/n states
and shared currents, withheld component/frame covariance and data, modular
nuclei and GTMD/TMD/GPD matching/evolution remain the physical sequence.
The persistent goal remains active; no new physical blocker is established
by these two successful tree experiments.

## Reproduce, validate, recover

Source: `src/deuteron_wigner/quantum/nonzero_momentum_current.py`.
Tests: `tests/test_nonzero_momentum_current.py` (18 new checks).
Replay: `scripts/evaluate_nonzero_momentum_current.py`, printing JSON only.
Use the established environment with `PYTHONPATH=src:.`,
`PYTHONDONTWRITEBYTECODE=1`, and one BLAS/OpenMP thread. Run pytest with
`-p no:cacheprovider`. Saved [results](../snapshot/docs/next_level/nonzero_momentum_current_results.json)
are the full replay, not a small-resolution surrogate, and reproduce exactly.
The 18 new checks and preserved focused science/theory regression pass
together: 339 tests, 12 pre-existing warnings. No full-suite claim or change
to the historical synthetic F1 tolerance failure is made. All seven backup
documents match their pre-edit commit byte-for-byte; macOS archive metadata
is also retained.

Seven pre-edit entry/theory documents are preserved at `f35d8493` and
`computation_handoff/backups/2026-09-10_pre_nonzero_momentum_current.tar.gz`.
Earlier sources and user-owned untracked files are untouched. No public
push or review-mirror export; no PDF requested or generated.

The implementation uses established field-theory mechanisms, not a claim
of inventing them. Primary context: [Kvinikhidze and Blankleider, gauging
equations](https://arxiv.org/abs/nucl-th/9901001) for common attachments and
subtraction; [Ji et al., interpolating QED](https://doi.org/10.1103/PhysRevD.98.036017)
for canonical instantaneous fermion propagation; [Marinho et al., LF
Ward--Takahashi identity](https://arxiv.org/abs/0805.0707) for the distinct
role of pair and instantaneous terms. Their bound-state/truncation programs
are not adopted. All displayed tree identities are independently derived
and tested here in the project's conventions.

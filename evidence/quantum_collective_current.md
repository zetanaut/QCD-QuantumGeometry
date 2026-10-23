# Quantum charged-collective current and conditional slope matching

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. Continuation from `46751f3b`. The one-free-quark-loop
charged collective electromagnetic triangle now has an independently
derived transverse shape, its linked inverse-propagator normalization,
direct Dirac-trace/four-component checks, and an independent LF plus-current
overlap. At the old test point its slope is about 82% of the earlier
radius-squared calibration. That contribution must not be added beside the
unchanged tree-fitted local coefficient.

No local C or D was refitted, no quantum Hamiltonian or existing physical
kernel was modified, and no production model or hadron prediction is
selected. This is one free-quark loop in the retained-action reference,
not confinement, a quantum pion bound-state solve, full quantum matching,
or a sigma/NJL completion campaign. In particular, the independent LF
calculation covers **plus only**, not quantum LF minus/transverse components.

## Content, order and finite convention

Use the same local i gamma5 Yukawa coupling, Nc=3 and equal quark masses
M=.2843478671 GeV, with quark charges Q_u=2/3, Q_d=-1/3. External charged
collective kinematics use the old light tree mode, s=.019479893765 GeV²,
below 4M². The mode coupling squared is (e_pi.g)²=9.24554827581. These are
test inputs, not newly matched quark/pion masses or couplings. The full
two-multiplet loop vertex is g_i g_j times the unit-coupling result; its
off-diagonal mode/source terms are not discarded.

The calculation includes both photon attachments on the closed quark loop.
Charged vertices have flavor matrices sqrt(2) tau+ and sqrt(2) tau-.
Their flavor traces give Q_u and Q_d after the common factor two; the
antiquark loop is equivalently Q_d Gamma(-p',-p), providing its sign.
The neutral equal-charge pair cancels. Electric e is stripped throughout.

The finite vertex uses the same zero-invariant wave subtraction as the
preceding twice-subtracted pion inverse term

\[
 \widehat{\Delta K}_P(s)=-4N_c s B(s),\qquad
 B(s)=-\frac1{16\pi^2}\int_0^1dx\log[1-x(1-x)s/M^2].
\]

This is a declared analytic subtraction convention for the loop, **not**
a choice of physical local matching coefficients or proof of a complete
chiral renormalization scheme. Its finite local/source counterparts and
other loops remain to be matched. The calculation is evaluated at prescribed
tree-pole kinematics and consistently through one loop, not by partially
resumming a pole denominator or the heavy scalar's broad continuum.

## Actual triangle, not a Ward-constructed vertex

Let q=p'-p, alpha=u v, beta=u(1-v), gamma=1-u, and
r=alpha p'+beta p. For a=p'-r, b=p-r, c=-r,

\[
 \Delta=M^2-\alpha\gamma p'^2-\beta\gamma p^2-\alpha\beta q^2,
 \quad v^\mu=\tfrac12c^\mu-a^\mu-b^\mu,
\]

\[
 w^\mu=(a.b)c^\mu-(b.c)a^\mu-(a.c)b^\mu+M^2(a+b-c)^\mu.
\]

Here v^mu is a numerator coefficient, not a velocity or the parameter v.
The common unit-charge/unit-Yukawa subtracted loop is

\[
 \widehat\Gamma^\mu(p',p)=\frac{N_c}{\pi^2}
 \int_0^1 du\int_0^1 dv\,u\left[
 v^\mu\log\frac{\Delta}{M^2}+\frac{w^\mu}{2\Delta}
                    -\frac{(p'+p-r)^\mu}{2}\right].
\]

It has units GeV. The numerator comes from the actual trace
Tr[(/k+/p'+M) gamma^mu (/k+/p+M)(-/k+M)] after cycling the gamma5 pair.
One implementation reduces the invariant products; an independent one
uses the original matrices, evaluates +/- loop directions, and contracts
the quadratic tensor with the Minkowski metric. Terms depending on the
d-dimensional numerator only through a local wave coefficient are removed
by the same subtraction, not used as an adjustable transverse term.

Off-shell tests verify

\[
 q_\mu\widehat\Gamma^\mu(p',p)
 =\widehat{\Delta K}_P(p^2)-\widehat{\Delta K}_P(p'^2),\qquad
 \widehat\Gamma^\mu(p,p)=2p^\mu\delta Z_P(s),
\]

\[
 \delta Z_P(s)=-\widehat{\Delta K}'_P(s)
             =4N_c[B(s)+s B'(s)].
\]

The inverse is evaluated independently using the prior two-point integral,
including finite differences of its derivative. This fixes normalization,
not the finite-transfer transverse shape. Omitting the antiquark attachment
fails the charged Ward difference. The direct trace transforms correctly
in all four components under a boost, and the subthreshold vertex satisfies
reversal. These are covariant loop checks, not independent LF calculations
of all four components.

For equal external virtualities s and t=-Q², write
Gamma^mu=(p'+p)^mu f(s,t). The one-loop charge/pole-normalized nonlocal
shape is deltaF=f(s,t)-deltaZ_P(s). The mode's result is (e_pi.g)² deltaF.
Keeping f without the linked normalization would give an erroneous charge
shift .0585611 at the test point.

## Independent LF calculation retains the spin/orbital information

In the Drell--Yan frame, build the canonical pair amplitude from
ubar i gamma5 v, with both quark and antiquark spin labels retained:

\[
 N(x,k_T)=\sqrt{x(1-x)}\,\bar u i\gamma_5 v
 =i\begin{pmatrix}k_x+ik_y&M\\-M&k_x-ik_y\end{pmatrix},\qquad
 \Psi=-\frac{\sqrt2 N}{k_T^2+M^2-x(1-x)s}.
\]

The source-free cloud probability diverges logarithmically; no normalized
continuum Fock state is claimed. The overlap subtracts the charge norm
**before** transverse integration, uses recoils (1-x)q_T and -x q_T with
weights Q_u and -Q_d, and retains both orbital and mass terms in
Tr[N(k')†N(k)]=2(M²+k'.k). Algebraically pairing opposite azimuths keeps
the UV subtraction stable.

The first LF calculation integrates x,k_T²,phi directly. A second integrates
the transverse plane analytically and numerically integrates x and a new
parameter v: with d=M²-x(1-x)s and delta=v(1-v)a² for each recoil,

\[
 \delta F_{\rm LF}=\frac{N_c}{4\pi^2}\int dx\,dv
 \sum_{a=(1-x)q_T,-xq_T}Q_a\left[
 -\log(1+\delta/d)-\frac{\delta(d+M^2)}{d(d+\delta)}\right].
\]

This parameterization is independent of the covariant triangle parameters.
All three answers agree over Q²=0,.02,.05,.1,.5 GeV² to below 2e-17
absolute in the unit-coupling deltaF. The .5 point is only a mathematical
large-transfer control, not the previously used low-Q² matching window.
The direct LF radial/angular resolution change is below 3e-17. Deleting
the orbital numerator changes the .1 result by 66.6% despite preserving
zero-transfer charge subtraction. A normalized charge alone cannot certify
the retained wavefunction/current information.

The independently differentiated slope is

\[
 c_{\rm loop}^{(g=1)}=\left.\frac{d\delta F}{dt}\right|_0
 =\frac{N_c}{24\pi^2}\int_0^1dx\,
 [\tfrac23(1-x)^2+\tfrac13x^2]
 \left[\frac2d+\frac{M^2}{d^2}\right].
\]

As the external s tends to zero it becomes Nc/(24pi²M²). Finite-transfer
derivatives, the canonical spinor numerator and this limiting expression
are separately tested.

## Conditional matching result and its limits

At the old light-mode point, the loop slope is 1.52229275 GeV^-2,
or a radius-squared contribution .35564964 fm². The previously reused
radius input .659 fm corresponds to 1.85885981 GeV^-2. Thus the loop
accounts for 81.9% of that slope **within this reference calculation**;
this is neither a QCD prediction nor a loop-error estimate.

If this retained loop were the only new explicit contribution, the
conditional relation would be

\[
 c_{\rm target}=c_{\rm local}^{R}+c_{\rm loop}
                 +c_{\rm other\ omitted},\qquad
 c_{\rm local}^{R}=.33656706\ {\rm GeV}^{-2}
 \quad\text{only when the other term is omitted for this diagnostic.}
\]

Here c_local^R is the **renormalized on-shell local slope**. It includes
the field/source normalization convention; it is not an instruction to
replace the bare C matrix by this number while forgetting wave/mode mixing.
At Q²=.1 the old linear tree F is .814114. Adding the loop beside the
unchanged fitted coefficient gives .678266 and double counts its slope;
the conditional subtraction gives .830496, retaining the loop's curvature.
Neither curve is a physical form-factor prediction.

No C/D parameter files were changed. Radius matching alone cannot update
the earlier joint radius/polarizability fit: the quantum two-photon contact,
Born subtraction, inverse/triangle/source partners and potential-linked
contributions must also be treated consistently before changing D or claiming
the old two-photon comparison remains calibrated. Large loop slope compared
with a target slope is not, by itself, proof that all higher loops are large;
their control and the complete action remain unestablished.

If retained quarks remain explicit in a quantum solver, the same loop must
not also be inserted independently into its Hamiltonian/current. These
functions are matching/control calculations. Neither this loop nor an
arbitrary residual C/D choice supplies generated p/n states or the modular
nuclear/GTMD endpoint.

## Reassessment and next action

The independent transverse quantum-current step passes, with 22 direct
tests (also with warnings treated as errors) and 389 focused science/theory
tests (12 existing warnings). The saved replay reproduces exactly; all seven
archived originals match 46751f3b byte for byte. A useful next step remains:
pre-evaluate the minimal **two-source quantum matching** needed to update
the joint C/D conditions, deriving the quark-loop two-photon response with
its linked inverse/triangle/Born and potential/source partners. Do not
silently equate a lone quark box with the complete Compton amplitude, refit
C alone, or infer transverse contact information from Ward identities.
Keep the shared continuum and all source interference. Review cost and
scope before a full unselected quark-meson completion/resummation program;
no such program or production model is authorized by this result.

The persistent effort remains active. Joint quantum state/interaction
matching, generated p/n currents/states with withheld covariance and data,
modular nuclei, and GTMD/TMD/GPD matching/evolution remain ahead.

## Reproduce, recover and prior art

Source `src/deuteron_wigner/quantum/quantum_collective_current.py`, tests
`tests/test_quantum_collective_current.py`, replay
`scripts/evaluate_quantum_collective_current.py` (prints JSON only), and
[saved results](../snapshot/docs/next_level/quantum_collective_current_results.json). Use the existing
environment with `PYTHONPATH=src:.`, no bytecode, one BLAS/OpenMP thread and
pytest `-p no:cacheprovider`. No full-suite claim is made.
Seven original notes are preserved at `46751f3b` and
`computation_handoff/backups/2026-09-10_pre_quantum_collective_current.tar.gz`.
Prior sources, user files and public/review copies remain untouched.

The triangle, pole normalization and LF overlap are established mechanisms,
not claims of a new model or theorem. For primary context on three-point
Ward normalization see [Quark-hadron duality and hadron properties from
correlators of pseudoscalar and axial currents](https://arxiv.org/abs/hep-ph/0602217).
Its OPE/duality program is not adopted. The displayed pointlike-reference
integrals, flavor phases, matching convention and independent numerical
controls are derived here in the project's symmetric LF convention.

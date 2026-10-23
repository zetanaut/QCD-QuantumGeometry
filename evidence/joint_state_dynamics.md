# Joint state/dynamics learning and preserved deuteron information

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. Developed from f40d4dd4. Local checkpoint; not a public
export. This advances the user's selection of **both learned states and shared
effective interactions**, standard matching where possible, and a
wavefunction/PDF-to-b1 validation. It does not select a production nucleon model.

## Outcome

The new PennyLane/PyTorch mechanism learns separate state and dynamical
parameters and propagates derivatives through the same source-dependent
Hamiltonian and currents. Exact finite controls expose two different failure
modes: a good data fit need not solve its own state equation, and even a
correct state equation need not identify phase-sensitive dynamics. An
independent matching amplitude resolves the latter in the tested example.

The preserved AV18 S/D radial shapes and supplied nucleon PDFs now give a
separate, executable quantum information-transfer benchmark. Its b1 values
are withheld from state training. Removing relative S/D coherence can preserve
unpolarized forward information while changing b1.

This is not a nucleon mass, form-factor or QCD Wilson-coefficient fit, nor
proof of covariance of an interacting bound-state current. No old HO,
fixed-K, Gaussian-regulated or H2 Hamiltonian was imported. The canonical
LF/Fock, ordered SU(3), composite-algebra and older physics results remain
unchanged and available.

## What the two learned objects mean

For a target T, use a normalized state \(\Psi_{T,\theta_T}\) in a declared,
symmetry-qualified carrier. Parameters \(\phi\) belong to a **shared** source
functional, not separate per-target or per-observable output heads:

\[
 \Gamma_{\rm eff}[\Phi,A;\mu,\phi]
 =\Gamma_{\rm retained}[\Phi,A;\mu]
 +\sum_a c_a(\mu,\phi)\,{\cal O}_a[\Phi,A;\mu].
\]

Here \(\Phi\) denotes the selected resolved fields. This is a specification,
not an implemented complete QCD effective action. Each operator needs a
declared dimension, flavor/color/Lorentz representation, ordering,
resolution, source prescription and matching domain. Quark/gluon, composite
and residual effective operators need not have identical representations.
They must nevertheless obey the applicable gauge, discrete, chiral/spurion,
Poincare and conservation identities. Encoding omitted physics is not a
license to violate those identities.

At an explicitly defined finite resolution, constrain
\[
 (H_\phi-E_T)\Psi_{T,\theta_T}=0,\qquad
 J^\mu_\phi=\left.\frac{\delta\Gamma_{\rm eff}}{\delta A_\mu}\right|_{A=0},
 \qquad
 K^{\mu\nu}_\phi=\left.\frac{\delta^2\Gamma_{\rm eff}}
 {\delta A_\mu\delta A_\nu}\right|_{A=0}.
\]

Source and sign conventions must also be declared when converting an action
derivative to a Hamiltonian current. Source differentiation of kernels alone
does not include every external-leg, normalization or state-response term in
a physical matrix element. A nonlocal-in-time functional requires a causal
Hamiltonian realization, not merely relabeling its value as an LF energy.
An eigenstate residual does not select the ground state; the controls check
that branch separately by independent diagonalization.

Do not minimize energy with respect to unconstrained dynamical coefficients:
that can simply make the interaction arbitrarily attractive. Match shared
coefficients, impose physical conditions, and solve the state equations;
use uncertainty-aware observable fits in the same joint objective.

The pure-state metric is
\[
 g^{\rm state}_{ij}=\Re\langle\partial_i\Psi|
 (1-|\Psi\rangle\langle\Psi|)|\partial_j\Psi\rangle.
\]
The coefficient metric from linear matching is \(A^T\Sigma^{-1}A\), on the
identified subspace. These are different objects. Neither assigning the
state metric to arbitrary effective coefficients nor asserting that the full
joint information metric is automatically block diagonal is justified.
The implementation measures these metrics; it does not yet implement a
natural-gradient optimizer.

## Standard matching, with omissions visible

With common external states, normalizations, regulator, scheme, scale and
order, match amplitudes and the required source insertions:
\[
 {\cal M}^{\rm full}_r-{\cal M}^{\rm retained}_r
   =\sum_a A_{ra}c_a+\text{declared remainder}.
\]

The retained subtraction is essential to avoid counting explicitly resolved
physics again as an effective interaction. Correlated residuals are whitened;
complex amplitudes use stacked real/imaginary parts and the corresponding
covariance. Singular values identify constrained combinations and unresolved
directions. A least-norm representative is not evidence that an unidentified
physical coefficient vanishes.

This follows the standard EFT distinction between matching, power counting
and low-energy calculations; see [Manohar's EFT lectures](https://arxiv.org/abs/1804.05863).
Our finite synthetic matching test supplies its own exact amplitudes. It does
not supply real QCD matching data or justify a perturbative calculation in a
nonperturbative soft domain. Where perturbative matching fails, independent
nonperturbative information or explicitly phenomenological coefficients and
uncertainties are required. Fitting a form factor is not, by itself, matching
every current or bilocal operator.

## Nonlocal interactions must carry their source response

For a fixed path C from y to x, the Abelian EM factor multiplying an ordered
color transporter is
\[
 W_C[A]=\exp\!\left(iQ\int_C dz^\mu A_\mu(z)\right),\quad
 I_C^\mu(q)=\int_C dz^\mu e^{iq\cdot z},\quad
 V_C^\mu=iQ I_C^\mu,\quad V_C^{\mu\nu}=V_C^\mu(q)V_C^\nu(k).
\]

The new implementation evaluates each straight segment analytically with a
regular sinc limit at \(q\cdot \Delta z=0\). It obeys
\[
 q_\mu V_C^\mu=Q(e^{iq\cdot x}-e^{iq\cdot y}).
\]
Different paths with the same endpoints can therefore differ by a transverse
vertex. The tests verify that difference is nonzero, as well as both source
derivatives, the two-source endpoint identity, reversal and Lorentz transport
of the whole prescribed path. Holding a preferred path fixed while boosting
only external momenta would not be this covariance test.

These are **Abelian insertions**; the SU(3) matrix transporter is not replaced
by a scalar trace. Non-Abelian insertions and ordered color products retain
their own algebra. One- and two-probe kernel derivatives are not a completed
interacting nucleon EM four-current, an axial current, or a GTMD soft factor.

## Controlled joint inference: actual results

Replay: [joint results](../snapshot/docs/next_level/joint_state_dynamics_results.json).
The four-dimensional Hermitian family is \(H=H_0+\sum_{a=0}^2\phi_a V_a\),
with \(H_0={\rm diag}(0,.7,1.3,2)\), nearest-neighbor \(V_0=-.4\),
\(V_1={\rm diag}(.1,-.2,.3,-.1)\), and
\((V_2)_{03}=+.2i,\ (V_2)_{30}=-.2i\).
Truth is \((.85,-.45,.55)\). All quantities here are dimensionless fixtures.
The imaginary loop is an identifiability device, not selected physical
time-reversal-breaking nucleon dynamics.

Every hopping is gauged by \(H_{ij}(a)=H_{ij}(0)e^{ia_{ij}}\).
Its differentiated edge currents satisfy exact local unitary covariance
and \(\sum_e B_{ve}J_e=i[P_v,H]\). This finite graph identity is not a
continuum four-current.

Training uses three retained-subtracted matching amplitudes (rank two), the
ground energy and two real diagonal measurements. Complex coherences and
current matrices are withheld. The target is generated by an independent
exact eigensolve, which is not called inside the optimization loss.

| Test | Result and consequence |
|---|---|
| Three Torch starts, one PennyLane start, with state equation | State-equation norm ≤2.07e-10; withheld current-matrix error ≤7.44e-10 on the positive branch. |
| Omit state equation | Training measurements and energy fit at about 1e-10, but residual 0.1770 and current-matrix error 0.1643. |
| Conjugate starting branch, with state equation | Equally good training fit, \(\phi_2=-.55\), current-matrix error 0.3111. Eigenvalues/real diagonal probes cannot select the sign. |
| Add independent phase-sensitive amplitude matching | Matching rank becomes three; starts of either sign recover +.55 and current error <7.41e-11. |

The extra amplitude is the fixture's \(\Im\langle0|H|3\rangle/.2\), not an
invented experimental measurement. Error norms refer to a fixed declared
carrier basis. The conclusion is structural identifiability, not a physical
confidence interval or successful extrapolation to an unknown interaction.

The complex graph state uses actual excitation-preserving PennyLane gates.
Independent Torch matrices reproduce its state and full gradients. The
small reference uses **D wires for D carrier states**; it is not a scalable
many-body encoding. A separate direct test carries gradients through both
hidden/output layers of a Torch coefficient network and a PennyLane
state/current expectation. No fully trained physical DNN or quantum advantage
is claimed.

## Preserved AV18/PDF → b1 benchmark

Replay: [deuteron results](../snapshot/docs/next_level/deuteron_information_results.json).
The unchanged AV18 reader, LF wavefunction, PDF provider and collinear
convolution supply the real table, Melosh rotations, LF measure and observable.
Use AV18 deut.wfk, its fixed PCHIP radial shapes on 0–15 fm⁻¹,
NNPDF31_nnlo_as_0118_1000 member 0 through actual LHAPDF 6.5.5,
Q²=5 GeV², isospin-rotated neutron, x_N and the existing per-nucleon convention.
Input hashes are recorded in the replay. LHAPDF returns xf; the existing
provider divides by x ([API documentation](https://www.lhapdf.org/classLHAPDF_1_1PDF.html)).

Normalize the two supplied radial shapes separately to form \(|S\rangle\)
and \(|D\rangle\). The target coefficients are
\((0.9707804885,0.2399692548)\). The finite-table interpolant total norm is
1.0002283947; its explicit normalization defines this benchmark. It is not
a fitted current rescaling. The table's original Simpson norm and the
interpolant norm are separately reported.

Linearity of the wavefunction and quadraticity of its density give a b1
operator on this two-channel carrier:
\[
 B_{SS}=b_1[S],\quad B_{DD}=b_1[D],\quad
 B_{SD}=b_1[(S+D)/\sqrt2]-(B_{SS}+B_{DD})/2,\qquad
 b_1[c]=c^\dagger Bc .
\]

The real, time-even spin-summed kernel has real interference in these signed
radial conventions. The direct existing real-wave calculation verifies the
polarization identity at seven x points to 6.3e-16. The quantum fit uses the
**supplied full S/D mixing density**, not b1 training values; this is
information transfer, not discovering the deuteron wavefunction from data.

- Both backends give withheld b1 error 4.60e-12 relative to the same
  quadrature and identical derivatives to numerical precision.
- Reversing the relative D sign preserves S/D probabilities and unpolarized
  smearing but changes b1 by up to 5.862e-4. This is much larger than the
  final tested quadrature change, 1.055e-8.
- Orders (64,24,2), (128,32,2), (256,48,2) were compared. The final
  unpolarized norm is 0.9999940; tensor norm is 1.58e-15. Quadrature changes
  are diagnostics, not a rigorous remainder bound.
- An LO impulse formula using this NNLO PDF is a numerical information
  benchmark, **not an order-consistent phenomenological prediction**.
  No PDF ensemble, nuclear model uncertainty, experimental fit, generated
  nucleon PDF, new Hamiltonian or new b1 mechanism is supplied.

## Physical candidate specification and next experiment

The architecture choice is now **joint state and shared effective dynamics**;
do not ask the user to choose between them again. The next unresolved object
is a physically qualified source functional, not another backend.

A useful candidate representation for investigation is a retained canonical
q/qbar/g sector plus a hierarchy of source-gauged nonlocal residual
correlations. Transport quark fields to common anchors before coupling
color/spin/flavor tensors; build coefficients from declared scalar invariants
and ordered holonomies. For example a building block is
\[
 {\cal B}_{\Gamma f}(x,y;C)
 =\bar q_f(x)\,\Gamma\,U_C^{\rm color}(x,y)W_C^{\rm EM}(x,y)\,q_f(y).
\]
Matched multi-bilinear products then encode unresolved correlations, while
the retained canonical field operators still generate explicit dynamics.
This is an operator organization, **not** a proof that any chosen finite
bilinear/four-quark set binds a nucleon. Under chiral rotations, complete
flavor/Dirac multiplets and explicit mass/charge spurions are required; a
single scalar attraction is not an adequate chiral specification. A
finite path/feature dictionary is not complete merely because it is gauge
invariant. Neural functions may parameterize scalar kernels after those
restrictions, not replace the restrictions.

The narrow next physical experiment should compare only two resolutions of
the **same** low-energy content: an explicitly resolved soft correlation
versus its matched residual-operator representation. Choose the common
amplitudes and one-current insertions first. In a perturbatively controlled
domain subtract the retained canonical contribution; in the genuinely soft
domain use qualified nonperturbative or phenomenological conditions with
explicit uncertainty. Do not add an explicit pion channel and fit its entire
contribution again into an unconstrained quark kernel.

Before a substantial nucleon fit this requires:

1. A common wavepacket/LF measure and units; UV, longitudinal/zero-mode,
   renormalization and matching-scale prescriptions. The canonical cubic
   coefficient is not itself a dimension-matched Hamiltonian matrix.
2. A bounded, chirally qualified correlation/operator basis and its
   first-order source partners, with a specified causal realization.
   Compare explicit versus residual content through the same amplitudes;
   quantify truncation and test source insertions, not just masses.
3. Independent p/n state/current solutions: charge, ground-branch and norm,
   axial/chiral limits where applicable, and agreement of observable
   extraction across current components and frames. A conservation pass
   alone cannot qualify the magnetic response.
4. A reserved multi-observable check and correlated uncertainty propagation,
   then modular nuclear matching/cluster limits and GTMD/TMD/GPD operators,
   evolution and sum rules. Single-nucleon data do not fix every many-body
   coefficient.

**Reassessment:** advance from backend qualification to this physical kernel
and matching comparison. No more generic graph/optimizer tests are needed as
the immediate scientific milestone. Current evidence does not select a
unique physical kernel, prove confinement, renormalization or full covariance,
or justify a large physical fit. These are open research tasks, not a reason
to reopen the old BLFQ/H2 repair queue.

## Reproduction and recovery

Install the optional joint-learning extra for the quantum/learning parts.
This run used Python 3.12.14, NumPy 1.26.4, SciPy 1.13.1, PyTorch 2.6.0,
PennyLane/Lightning 0.38.0 and autoray 0.6.12. The compatible pinned versions
were retained, not replaced with latest releases. Torch installation in the
temporary environment changed SymPy from 1.14.0 to its required 1.13.1;
the existing/global environments were not modified.
[PennyLane's Torch interface](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html)
documents hybrid differentiation; actual compatibility was verified in the
stated installed versions.

From the canonical checkout with PYTHONPATH=src:.:

    python scripts/evaluate_joint_state_dynamics.py --output docs/next_level/joint_state_dynamics_results.json
    python scripts/evaluate_deuteron_information.py --build-reference --output docs/next_level/deuteron_information_results.json
    python scripts/evaluate_deuteron_information.py --fit-reference docs/next_level/deuteron_information_results.json --output docs/next_level/deuteron_information_results.json

The build-reference command used /Users/dustin/miniforge3/bin/python
(Python 3.9.23, NumPy 1.26.3, SciPy 1.13.0, LHAPDF 6.5.5); it does not
require Torch. Both learning commands used
/private/tmp/deuteron-canonical.zxUoKe/env/bin/python. These local executable
paths are recorded for replay, not portable prerequisites. Raw AV18/PDF
inputs must be available to rebuild the independent reference; the committed
numerical reference allows the quantum comparison without those bindings.

All 22 new direct tests pass. The combined canonical/graded/modular regression
and theory-index run passes 124 tests (12 existing-code/runtime warnings).
This is a focused suite,
not a claim about the entire repository. The pre-existing broader-suite
synthetic-fit tolerance failure recorded at the preceding checkpoint is not
repaired or hidden by this work. Existing NumPy determinant and invalid-escape
warnings remain in older code. No PDF build or new historical line-by-line
theory re-audit was requested or performed.

The eight entry/status/theory/config originals are preserved and verified
byte-for-byte against the parent in
computation_handoff/backups/2026-09-10_pre_joint_state_dynamics.tar.gz
and in parent f40d4dd4. No previous script, physics result, volume source,
raw wavefunction or PDF table was overwritten. No new theory volume is
needed for this focused result; its mathematics is incorporated in the
main theory note and linked here. Public GitHub and the review mirror remain
at their prior published checkpoints.

# Stable connection, chiral fields and quark sea: constructive result

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

11 September 2026; developed from `5254a42815fa5379647bdbd46fcebd0caa42fca6`.

## Outcome

There is a working **exploratory shared-action class** that passes the requested
finite stability, non-singlet chiral, connection-dependent sea and linked-source
tests. It need not use the failed independent two-point surface closure, a
vanishing dielectric kinetic term, or a pole-removing quark mass ansatz.
This answers a necessary existence/implementation question; it does **not**
establish a renormalized QCD vacuum, confinement, physical p/n states or their
covariant currents. It is not a production quark-meson/dielectric model selection.

The actual advance over the earlier chiral controls is a noncommuting SU(3)
connection-dependent determinant, coupled sea backreaction, both canonical
gauge-energy terms, and exact retained-fermion/source counting. A second,
Hermitian operator control handles arbitrary real pion textures without
assuming their Euclidean determinant defines a positive physical measure.
Its relationship to a physical Euclidean regulator is not established here.

## Construction and sufficient bounds

The proposed local effective-action family retains the canonical connection
and quark tensors. A reference non-singlet chiral multiplet is
`phi=(sigma,pi1,pi2,pi3)`, `rho=phi.phi/2`. In physical units phi has mass
dimension one, rho two, a two, and lambda, y, kappa are dimensionless. A
schematic Minkowski action is

\[
\mathcal L=\bar q[i\slashed D-m-y(\sigma+i\gamma_5\tau\cdot\pi)]q
 -\tfrac14\kappa(\rho)F^a_{\mu\nu}[A]F^{a\mu\nu}[A]
 +\tfrac12(\partial\phi)^2-U(\rho),\qquad
 U=a\rho+\tfrac\lambda2\rho^2,\quad\lambda>0.
\]

This is an effective-action **test family**, not a derivation of new elementary
QCD scalars. Its collective-field identity/measure, operator basis and matching
coefficients must be supplied before production use. Existing two-multiplet
chiral, full Fierz and canonical Dirac utilities remain authoritative in their
scopes. One O(4) example does not discard their other channels. Known quark-meson
vacuum-loop work motivates retaining the sea, not adopting its thermodynamic
model or parameters: [Skokov et al.](https://arxiv.org/abs/1005.3166).

Use a bounded invariant map, not an unconstrained linear kinetic coefficient:

\[
 z(\rho)=\frac{\rho}{\Lambda^2+\rho},\qquad
 \kappa(\rho)=e^{\eta[z(\rho)-z(\rho_*)]},\qquad
 0<\kappa_{\min}\le\kappa\le\kappa_{\max}<\infty.
\]

The reference coefficient is exactly one. Since `0<=z<1`, the bounds are the
ordered pair `exp(-eta*z*)`, `exp(eta*(1-z*))`. For the illustrative `eta=1`,
`Lambda=1`, `rho*=.5`, they are **0.71653131 and 1.94773404**. An unconstrained
`kappa=1-rho` becomes negative and permits arbitrarily negative gauge energy;
the bounded class excludes this without a zero floor. It multiplies the full
non-Abelian `F[A]^2`, including canonical self-interactions, not an independent
scalar gluon-density register. It is not a classical confinement mechanism.

The conjugate electric momentum and energy are

\[
 \Pi_i^a=\kappa E_i^a,\qquad
 \mathcal H_g=\frac{\Pi^2}{2\kappa}+\frac{\kappa B^2}{2}
 \ge\frac12\min(\kappa_{\max}^{-1},\kappa_{\min})(\Pi^2+B^2).
\]

Gauss law uses `(D_i Pi_i)^a=rho_q^a`, not `D_i E_i`. At fixed canonical Pi,
`dH_g/drho=kappa'/2*(B²-Pi²/kappa²)`. Keeping only the magnetic derivative fails
the direct energy/source test. These are analytical continuum canonical
identities; the tests evaluate their finite coefficient/field algebra, not
the complete constrained LF Hamiltonian. Local Gauss projection is still needed.

## What the finite calculation actually evaluates

The existing eight-node/twelve-link cube supplies generic noncommuting SU(3)
matrices, with **node x Dirac-spin x color x flavor = 8 x 4 x 3 x 2**. It is
an algebraic carrier, not a newly adopted lattice regulator. Its three spatial
edge directions do not constitute a four-dimensional QCD ensemble or fix
doublers, continuum normalization, the singlet anomaly, or LF zero modes.

With Hermitian Euclidean gamma matrices and `U_ab:b->a`, set
`(D0)_ab=h gamma_axis U_ab`, `(D0)_ba=-(D0)_ab†`. Then `D0†=-D0` and
`{D0,gamma5}=0`. For a homogeneous chiral field and zero chemical potential,

\[
 D=D_0\otimes1_2+(m+y\sigma)1+i y\gamma_5\tau\cdot\pi,
 \quad K=D_0^\dagger D_0+[(m+y\sigma)^2+y^2\pi^2]1,
 \quad D^\dagger D=K\otimes1_2,\quad\det D=\det K>0
\]

on the nonsingular domain. The last equality uses the paired chiral spectrum
and two degenerate flavors, and is checked against an independent full complex
LU determinant, not inferred from a modulus alone. **There is no extra Nf=2
factor multiplying `log det K`.** There is also no extra copy of the earlier
Pauli–Villars chiral determinant: these are different controlled realizations
of a sea contribution, not additive copies of the same quark loop.

The dimensionless reference action is

\[
 S_{\rm ref}=\beta\kappa(\rho)\sum_{f=1}^6
 (1-\mathrm{Re}\,\mathrm{tr}W_f/3)+8U(\rho)-w\log\det K.
\]

All oriented transports and their source derivatives are retained. The face
sum is a gauge-invariant control for the full-connection action, not a physical
choice of gauge discretization or a proof of confinement. Here
`a=.25, lambda=.6, beta=.7, w=.04, m=.06, y=1, h=.4`. `w` is a declared
reference-cell loop normalization, not a fitted sea-flavor multiplicity.
No values are physical predictions or fits. Compact links bound D0; at finite
matrix dimension n, `-w log det K >= -w n log[C(1+|phi|²)]` for a suitable
positive C. Quartic U therefore dominates at infinity. Determinant zeros give
infinite positive effective action, not a negative runaway. The homogeneous
finite integral is normalizable; no continuum renormalization follows.

Nine widely separated initial points converge on the same two-coordinate
slice minimum, **sigma=1.0011529223, theta=-0.5577742790**. The gradient norm is
`5.27e-10`, and the slice Hessian eigenvalues are **0.14251717, 11.92785626**.
Omitting the sea gives **sigma=0, theta=-0.5598682269**. Thus sea backreaction
changes the chiral branch and the connection; it cannot be absorbed into a
constant. Neither minimization varies the other eleven links, and neither
is a full vacuum solution. These slice energies have different sea content;
their absolute difference is not a vacuum-energy prediction.

## Chiral and source controls

At m=0, the homogeneous action is radial in O(4). Twelve chiral orientations
agree to `8.89e-16`; the stationary radius at fixed theta=.43 is 1.011803,
with finite-difference pion curvature `1.34e-7`. The exact identity is
`H_pi_pi=(dS/dsigma)/sigma`. At nonzero m, the linked identity is

\[
 \sigma S_{\pi_a\pi_a}=S_\sigma+2wym\,\mathrm{Tr}K^{-1}.
\]

This is a non-singlet mean-field potential identity, not a complete axial
current/pion pole calculation. A finite-volume m=0 orientation average still
has **<sigma>=0**; a stationary radial minimum is not a demonstrated physical
chiral condensate. No U(1)_A anomaly/topological susceptibility is supplied.
That distinction is physically essential; see [Creutz](https://arxiv.org/abs/0901.0150).

All mass/link variations derive from the same determinant:

\[
 S_s=-w\mathrm{Tr}(K^{-1}K_s),\qquad
 S_{st}=w\mathrm{Tr}(K^{-1}K_tK^{-1}K_s-K^{-1}K_{st}).
\]

Gradient and Hessian finite-difference errors at the reference check are
`7.01e-8` and `9.86e-8`. The sea link curvature is **0.00067024**; omitting
the genuine second source derivative gives **0.00651261**. Keeping only a
response-response product is incorrect. All eight color generators are
checked, as are local gauge covariance, nonuniform positive scalar masses,
full spin/color/flavor bilocal insertions, and `delta D^-1=-D^-1(delta D)D^-1`.
These are color-link/mass/algebraic bilocal responses, **not** an already
matched electromagnetic, axial, metric, TMD or GTMD operator family.

For arbitrary spatially varying real pion fields there is a separate
canonical operator route. The tested instantaneous one-quark matrix uses
`h_ab=-i h alpha_axis U_ab`, its adjoint on the reverse edge, and onsite
`beta(m+y sigma)+i y beta gamma5 tau.pi`. It is Hermitian and globally
non-singlet chiral covariant at m=0, without requiring a positive Euclidean
pion-texture measure. With `R=max_node |phi_node|`,

\[
 \|h\|\le3h_{\rm hop}+m+yR,\quad
 H_{q,\mathrm{Fock}}\ge-192(3h_{\rm hop}+m+yR),\quad
 H\ge\lambda R^4/8-4|a|R^2-192(3h_{\rm hop}+m+yR).
\]

The last bound includes the summed eight-site U and may include additional
nonnegative scalar-gradient, electric and magnetic energies. Its RHS is bounded
below and tends to positive infinity. It bounds all occupations before any
physical gauge/charge restriction; it does not identify a baryon sector or
a renormalized vacuum energy. Positive scalar kinetic energy and uniformly
positive gauge kinetic forms provide a route to a semibounded canonical
finite-mode quantization; no gauge-field Hilbert-space solve or physical
Euclidean-to-real-time/LF reconstruction was performed. This control prevents
a homogeneous determinant positivity assumption from becoming an unsupported
constraint on every future pion configuration.

## Exact retained-fermion matching and modular limit

Partition the **same source-dependent full matrix** into P,Q blocks:

\[
 D_{\rm eff}=D_{PP}-D_{PQ}D_{QQ}^{-1}D_{QP},\qquad
 \det D=\det D_{QQ}\det D_{\rm eff},\qquad
 (D^{-1})_{PP}=D_{\rm eff}^{-1}.
\]

The retained solver gets D_eff and its induced source partners. Only det D_QQ
belongs to the integrated residual influence. Including det D plus retained
loops again double counts. The actual 192-component connection/chiral matrix,
with 96 retained complete spin/color/flavor coordinates, matches determinant,
first and second mixed link/mass responses to `0`, `1.07e-14`, `4.46e-16`.
The incorrect full-plus-retained first response has an excess **15.10597**.
This implements Gaussian fermionic matching on the actual candidate; it is
not the full interacting gluon, collective-field or LF/Fock matching problem.

Disconnected module determinants add in the action to `7.11e-15`, and cross
source response is exactly zero. This is the modular algebra requirement,
not a predicted two-nucleon cluster limit or nuclear force. Connected nuclear
correlations still require shared dynamics, wavefunctions and exchange; A,
x and Q² dependence cannot be supplied by arbitrary vacuum-density heads.

## Reassessment and complete path forward

**Advance this class as an exploratory candidate.** The requested finite test
works; the physical model remains open. Avoid choosing a production scalar
potential, gauge gate, regulator or inherited chiral model by software default.
The exact full connection and Dirac/color/flavor tensors are fixed structure;
learned residual functions may vary inside bounds, with an explicit operator
basis and common derivatives. A bounded gate alone is not a complete dynamics
basis. Any neural potential must preserve the proved coercivity rather than
permit an unbounded negative learned term. Existing geometric/PennyLane/Torch
tools supply the solver, not missing physical matching.

1. **Next: physical embedding and field-space qualification.** Specify the
   collective-field identity/measure and retained subtraction on the canonical
   LF carrier, keeping quark and gluon source partners. Test local Gauss law,
   inhomogeneous variations, singlet/non-singlet chiral structure, regulator
   dependence and the physical-state Hamiltonian domain. The current graph is
   not promoted to the physical regulator. A local effective action is a
   tractable candidate, not an excuse to omit needed nonlocal/many-body terms.
   If physical embedding fails, compare a different matched invariant family;
   do not undertake indefinite lattice/BLFQ/anomaly repair by default.
2. **Generated p/n and currents together.** Use a full canonical-color,
   symmetry-compatible variational state with residual invariant capacity.
   Compute masses, charges, pion/axial constraints and independent current
   components before a large fit. Require component/frame agreement and
   controlled enlargement; never rescale the old factor-three discrepancy.
3. **Modular nuclear binding and operator validation.** Retain all color and
   exchange channels, cluster-separable connected interactions and variable
   A/Z/spin. Recover deuteron wavefunction information and the preserved b1
   benchmark, then test another nucleus using shared couplings.
4. **Joint inference and predictive distributions.** Derive currents and
   bilocals from the same action; include proper collinear/TMD evolution,
   rapidity/soft factors and matching. Fit p/n/nuclear observables jointly,
   reserve held-out TMD/GPD/GTMD tests and propagate data, truncation, matching
   and model uncertainty. Binding, scalar anomaly, momentum fractions and
   x-dependent sea/gluon structure remain distinct observables.

The next step has a decisive physical deliverable, not another generic
algorithm comparison. Refinement and component-covariance failures require
reassessment early. None of this roadmap asserts all-orders completion.

## Reproduction and preservation

Implementation: `src/deuteron_wigner/quantum/stable_connection_sea.py`.
Replay: `scripts/evaluate_stable_connection_sea.py`; computed output:
`docs/next_level/stable_connection_sea_results.json`.

```sh
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /private/tmp/deuteron-canonical.zxUoKe/env/bin/python -m pytest \
  tests/test_stable_connection_sea.py -q -p no:cacheprovider -W error
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /private/tmp/deuteron-canonical.zxUoKe/env/bin/python -W error \
  scripts/evaluate_stable_connection_sea.py
```

41 new tests pass with warnings treated as errors. The focused integration
run passes 245 tests across this class, prior surface/connection, chiral,
shared-source, parton and LF path utilities, with six pre-existing NumPy
determinant warnings in three unchanged gauge-network tests. This is not a
warning-strict full-suite run. All six current-document consistency tests also
pass (47 direct-plus-document tests, warning-strict); current hashes and paths,
705 unique TeX labels, 94 unique bibliography keys and the new section's eight
balanced environments were checked. No PDF was compiled. Initial link-tangent
finite-difference failures were the predicted O(h²) error (~1.3e-9), not a
changed physical convention; halving the step verifies the factor-four
convergence, and Richardson extrapolation gives <4e-12 without relaxing the
underlying derivative identity. No physics tolerances were fitted to data.

Eight original authority files are preserved byte-for-byte from `5254a428` in
`computation_handoff/backups/2026-09-11_pre_stable_connection_sea.tar.gz`, SHA256
`d81212fa3711e152e6e84a23a50815793fe7c3f46a78510b14ebbf771fa1face`.
The main theory, bibliography, machine index, entry points and both roadmaps
are updated locally. Historical reports, original theory volumes/archives,
the dated public review mirror and unrelated user work remain intact. No PDF,
public export or push was requested or performed; this is not a full re-audit
of every historical equation in the long manuscript.

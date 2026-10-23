# Source/network transport and information-preserving connectivity

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. The bounded experiment is complete. Equal-virtuality
kernels can hide nonzero mode-transition directions; a mode invisible at
one source insertion can return at higher order. Actual transport preserves
field normalization, all declared node/source matrices and mixed
interaction/source derivatives. These are construction constraints, not a
physical fit or generated hadron.

## Preserved pre-evaluation

The following pre-evaluation was committed before implementation. The preceding
[joint source/matching checkpoint](joint_source_matching.md) is complete
at its declared order. This note does not report an implemented network,
a new quantum state, a physical fit or QCD finite matching.

## Scientific question and mission connection

Which full-mode source directions are accessible through the common
interaction's propagators/endpoints, and which disappear if matching uses
only restricted kinematics or diagonal-mode projections? We must preserve
the independent C/D/E source structures and their heavy-light transitions
when connecting the shared action to the quantum/geometric inference
network. This is about physical coupling/matching information, not drawing
an optimizer tree or claiming topology generates missing dynamics.

Use the already qualified, source-dependent finite-momentum exchange
machinery as an algebra control. Do not choose a new production potential,
fit the reference as a physical pion, or replace canonical QCD color-vector
and graded fermion content with scalar/pion channels. Do not quantize a
previously calculated loop again as a bare retained interaction.

## Preliminary, read-only algebra check

For a positive real symmetric n-mode inverse with simple eigenvalues h_i,
one common endpoint vector g and spacelike squared virtuality x, write

\[
 v(x)=(H+xI)^{-1}g,\qquad S_E(x,y)=v(x)^TEv(y).
\]

This is a kernel-insertion coefficient, not a measured observable or a
complete four-quark Compton amplitude. A three-mode numerical check used
`H=diag(.2,.7,1.4) GeV^2`, `g=(1,.8,.6)`, 30 logarithmically spaced
`x=.01..5 GeV^2`, and all six real symmetric E coordinates. The diagonal
response `S_E(x,x)` had rank five; independent x,y selected every seventh
grid point gave rank six. The smallest diagonal singular value was
`3.38e-15`, versus the fifth `9.46e-4`; the independent-route smallest was
`9.68e-4`. These are an unfitted algebra probe, not statistical precision,
a physical range selection or a demonstrated full-source result.

Proposed explanation to prove and qualify: for distinct h_i and nonzero
g_i, diagonal products decompose into n double-pole functions and n-1
independent simple-pole differences, so diagonal data span at most 2n-1
directions rather than n(n+1)/2. Independent left/right virtualities may
recover the complete symmetric source on the reachable subspace. Exact
degeneracy or a missing endpoint projection changes that subspace; nearly
degenerate/weakly coupled modes affect conditioning rather than imposing
an exact physical zero. A graph's appearance alone does not decide this.

## Bounded implementation and independent controls

1. Prove the accessible-subspace criterion and diagonal-rank statement,
   including degenerate/zero-coupling counterexamples and conditioning.
   Preserve hidden directions explicitly; no least-norm physical zeros.
2. Transport C,D,E through the actual finite-momentum source-dependent
   inverse. Include both source orderings and contact terms. Verify that
   the abstract insertion coefficient is realized by the action-derived
   source response, with all declared node indices and quark endpoints.
3. Differentiate the same functional jointly in interactions and sources;
   compare analytic inverse variations with automatic and numerical
   derivatives. Qualify full-minus-retained subtraction at the functional
   level, not by subtracting Hessians or attaching a new observable head.
4. Reassess how these physical connectivity/identifiability constraints
   should inform joint quantum-state/interaction inference. Do not claim
   off-shell matching information is experimental data, a finite exchange
   inverse is a normalized quantum nucleon, or a synthetic rank test solves
   the physical joint fit. Use the result to choose the next actual state/
   source experiment, not launch an indefinite reference-model completion.

Already-preserved implementations to reuse: `collective_quark_matching.py`
for routed source jets and open bilinears; `joint_source_matching.py` for
the new action and conditional families; `joint_state_dynamics.py` for
the distinct state geometry, source family and matching inference; the
canonical LF, graded Fock and modular nuclear utilities for physical
carriers. No missing experimental response is fabricated. The overall
goal remains generated p/n and modular nuclear states/currents with
withheld covariance/data tests and GTMD/TMD/GPD matching/evolution.

## 1. Exact rank and the response-graph cycle ambiguity

Work in a canonical constant-kinetic-metric basis, with H positive real
symmetric, distinct poles h_i, a single endpoint g_i nonzero at each pole,
and a constant real symmetric source matrix E. This finite inverse is not
a nonlocal quantum propagator ansatz. For v_i(x)=g_i/(x+h_i),

\[
 S_E(x,x)=\sum_i\frac{g_i^2E_{ii}}{(x+h_i)^2}
 +\sum_{i<j}\frac{2g_ig_jE_{ij}}{h_j-h_i}
 \left(\frac1{x+h_i}-\frac1{x+h_j}\right).
\]

The n double poles are independent. The off-diagonal terms span n-1
simple-pole differences with zero total residue. The exact diagonal rank
is **2n-1**, whereas independent x,y can span all **n(n+1)/2** symmetric
coordinates. For n distinct virtualities the Cauchy matrix
V_ai=g_i/(x_a+h_i) is invertible, so the full matrix S=V E V^T determines E.
Finite sampling, noise and conditioning still matter; this is not an
inference guarantee for actual data.

The simple-pole coefficients are oriented edge flows on the complete graph
of pole modes. Diagonal data see their vertex divergences. Divergence-free
cycle flows are invisible, with dimension (n-1)(n-2)/2. This is an
**algebraic response-graph cycle space**, not topological order,
confinement or evidence that QCD is a graph model.

For a directed cycle i->j choose the symmetric dimensionless direction
delta E_ij=(h_j-h_i)/(2 g_i g_j mu_ref^2). Its simple-pole differences
telescope. The reference mu_ref^2=1 GeV^2 only fixes normalization of the
direction; it is not a matching scale. A physical E coefficient still has
GeV^-4 units. The saved triangle is

\[
 \delta E=\begin{pmatrix}
 0&0.3125&-1\\
 0.3125&0&0.7291666667\\
 -1&0.7291666667&0
 \end{pmatrix},\qquad
 h=(0.2,0.7,1.4)\ {\rm GeV}^2,\quad g=(1,0.8,0.6).
\]

Its maximum diagonal response is 1.42e-15 in source-shape kernel units;
a selected independent-route response is -0.1889134. Removing one edge
gives a diagonal response 4.052685: replacing the physical source graph by
a spanning tree changes the model, although a solver can still use a tree
encoding that retains the missing operator information. Tests at n=1,2,3,4
give diagonal ranks 1,3,5,7 and independent ranks 1,3,6,10. Node rotations
preserve singular values in a Frobenius-orthonormal source basis. These are
not physical errors without a data response/covariance; the rank formula
does not constrain arbitrary nonlocal vertices or continua with cuts.
This count is within a fixed generator/operator convention, not a count
of independent physical EFT constants. A broader field redefinition can
redistribute terms only with its induced vertices and endpoints retained;
it does not justify simply deleting a source edge from the fixed family.

## 2. Accessibility depends on source order and generator family

For H and endpoint matrix G, the source-free resolvent-reachable space is
span{G,HG,...,H^(n-1)G}. Exact degeneracy or a missing endpoint projection
can reduce it. Multiple endpoint channels may recover absent information.
Weak couplings and near-degenerate poles can instead make a full-rank
inverse ill-conditioned; they are not exact zeros. All cases are tested.

Source-free accessibility limits a **single insertion** through those
endpoints; it is not a certificate for deleting other modes. In the
declared finite linear family, the all-source space must be invariant
under H and every supplied source matrix. Their word span with G constructs
this space. Symmetry then also decouples the orthogonal complement, so
projection preserves the Gaussian response in its invertible/analytic
domain. This does not cover unknown source terms, nonlinear interactions,
further Fock sectors or the full QCD/vector/axial algebra. Numerical rank
thresholds are resolution choices, not inferred physical zeros.

For H=diag(.2,.6,1.2) GeV^2, G=(1,0,0) and E linking 1--2 and 2--3,
source-free accessibility has dimension one but the H/E word space has
dimension three. A network restricted to the populated source-free mode
would miss allowed return paths.

## 3. Actual momentum-routed action and coupled derivatives

The earlier two-mode code is an independent control. The new transport
accepts arbitrary node count and constant positive kinetic matrix Z.
At p0,p0+q1,p0+q2,p0+q1+q2, all spacelike, assemble

\[
 K_0(p)=H-p^2Z,\quad K_a=-\Gamma_a,\quad K_{ab}=-T_{ab},\qquad
 W(a,b)=Y^T(K_0+aK_a+bK_b+abK_{ab})^{-1}Y.
\]

The minimal one-/two-source terms contain Z. C,D,E contact coefficients
come from actual fourth derivatives of the preceding O(4) action, then
multiply trainable full node matrices. Both source Fourier signs,
propagation orderings and contacts are retained. This is a charged-sector
Taylor jet through a*b, not a full finite-source quantum theory.
For R=K0^-1, independent inverse differentiation gives

\[
 W_{ab}=Y^T(RK_aRK_bR+RK_bRK_aR-RK_{ab}R)Y.
\]

The eliminated Ward identity has the required endpoint differences.
Omitting the contact fails. The E-dependent part equals the reduced stress
kinematic factor times g^T R(p1) E R(p0) g; the saved difference is
8.1e-19 GeV^-4. The full two-source response is 1.0407091 GeV^-4 and
its E contribution is -1.6989415e-5 GeV^-4 at the unfitted test point.

The same Torch functional is differentiated in H, endpoint coupling g and
E, and in both sources. Reversing derivative order agrees with analytic
inverse variations and independent finite differences. Propagation and
endpoint feedback therefore remain in a joint fit. These coordinates have
different units; no shared gradient-size threshold or invented physical
information metric is used. Full-minus-retained subtraction acts on W
with its source partners, not on Hessians before inversion.

For phi=T chi, transport H'=T^T H T, Z'=T^T Z T, G'=T^T G and every
M'=T^T M T. Canonicalization uses S=Z^(-1/2) and transports the same
objects. Responses agree within 1.8e-15. Omitting Z changes the two-source
kernel by 0.2445042 GeV^-4; freezing E after transporting H/G also fails.
This covers constant positive Z, not energy-dependent quantum pole matching.

Open S/P coefficients lift through the existing 24x24 Dirac/flavor/color
bilinears as W_AB B^A_ij B^B_kl, retaining all 24^4 endpoints and charged
flavor phases. Direct contractions and charge commutators are independent
checks. Canonical color-vector tensors remain separate and unchanged.
This is not their replacement, identical-fermion antisymmetrization, or
the full external-quark radiative amplitude.

## 4. A hidden mode returns in the actual four-photon response

Take H=diag(.2,.7) GeV^2, g=(1,0), E=[[0,.6],[.6,0]] GeV^-4. One E
insertion vanishes for every virtuality. The coefficient quadratic in E at
four-photon order is nevertheless nonzero. For all photons incoming,
p1=p0+sum q_i, and every ordered pair subset A with complement B,

\[
 W^{(4)}_{E^2}=
 \sum_{|A|=2}
 g^TR(p_1)\,T_E(p_1,p_A;q_B)\,R(p_A)\,
 T_E(p_A,p_0;q_A)\,R(p_0)g,\qquad
 p_A=p_0+\sum_{i\in A}q_i .
\]

There are six ordered pairings. E has no zero-/one-photon vertex and its
higher contacts are linear in E, so these are precisely the quadratic-E
terms at this source order. **This is not the complete four-photon kernel**,
which has other linear and mixed terms.

The expression agrees with an independent fourth source derivative of a
16-route inverse built from actual action-derived two-source coefficients.
It is -5.2156884e-9 GeV^-6; this off-shell coefficient has no positivity
interpretation. All longitudinal contractions vanish. Permutations, gauge
shifts and Lorentz boosts agree. Deleting the second mode yields zero and
loses the response. No loop, physical measurement or general many-body
compression theorem is claimed.

## Reassessment and next scientific step

Retain source transitions/cycle information not identified by restricted
matching, source-order-dependent return paths, normalization and all source
partners. A tree used by an optimizer must not silently restrict the
physical operator family to a tree. Source freedom and experimental-
response uncertainty from the preceding conditional matching stay explicit.

The [next source-generator pre-evaluation](quantum_source_generator.md)
identifies the immediate bridge: derivative source interactions modify
canonical momenta, so their generator and Hamiltonian contact terms must
be derived, not copied from the Lagrangian vertices. A local LF velocity
Hessian check motivates this analysis; it is not a new state or a defect
in the covariant source identities already tested here.

The next step should advance an **actual nonzero-momentum quantum state
and its linked response**, not keep extending Gaussian source algebra.
Pre-evaluate the preserved canonical/retained action's available state
construction, quantum counting, source partners, domain and an independent
control before choosing an experiment. The spacelike H here is a
fluctuation inverse, not a bound-state Hamiltonian. The homogeneous cell
is not an LF state; old tree parameters are not quantum matched; S/P
channels neither replace canonical QCD nor supply confinement. Do not
silently select production sigma/NJL/BLFQ dynamics or launch an inherited
resummation/bound-state repair program. Expose any genuine physics choice.
The persistent goal remains active toward joint states/interactions,
generated p/n, independent covariance/data, modular nuclei and distributions.

## Validation and recovery

- Source: src/deuteron_wigner/quantum/source_network_transport.py.
- Tests: tests/test_source_network_transport.py — 27 including exact replay.
- Replay: scripts/evaluate_source_network_transport.py and
  docs/next_level/source_network_transport_results.json.
- Originals: 01810eebdb4c936c8c34e7b67502e03395bbb6d2 and
  computation_handoff/backups/2026-09-10_pre_source_network_transport.tar.gz;
  eight files, including this note's pre-evaluation.

All 27 direct tests pass with warnings treated as errors. The combined
465-test science/theory regression passes (12 existing warnings). Eight
archived originals match 01810eeb byte for byte. The new
theory environments balance; final hash checks are in the current handoff.
This is not a claim that the entire historical suite passes. Earlier
source/results remain unchanged. No physical coefficients fitted,
production selection, PDF rebuild or public push.

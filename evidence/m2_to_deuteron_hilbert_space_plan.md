# M2-to-deuteron Hilbert-space construction plan

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-05

Scientific lane: exact representation algebra followed by an exploratory
state/current construction. Nothing in this note selects a physical state,
renormalized parameter, production current, fit, response rank, or activation.

## Result established during preparation

The modern M2 space is not itself a deuteron space. At each represented
resolution it factors as

\[
  \mathcal H_{\mathrm{M2},K}
  \simeq \mathcal M_K\otimes V_3,
\]

where `V_3` is the fundamental color representation and `M_K` is the
color-stripped multiplicity space. The live dimensions are

| resolution | `dim(H_M2,K)` | `dim(M_K)` | color modules |
|---|---:|---:|---:|
| K9 | 1,350 | 450 | 450 copies of `3` |
| K11 | 2,706 | 902 | 902 copies of `3` |
| K13 | 4,758 | 1,586 | 1,586 copies of `3` |

After every operator is placed in the same target basis, direct sparse checks
show that the M2 free kinetic operator, both C401/C396 mass directions, the
C411 first-C117 action, and the C53 canonical `q <-> qg` vertex all obey

\[
  O_K = \bar O_K\otimes I_3
\]

in module-major order. Cross-color blocks vanish and the three diagonal color
blocks agree to at worst `3.6e-15` in the audited operators. They therefore
descend without approximation to `M_K`.

The exact completion adds an anti-fundamental spectator color and uses

\[
 |1\rangle_{3\otimes\bar 3}
 =\frac{1}{\sqrt3}\sum_{c=1}^{3}|c\rangle|\bar c\rangle.
\]

For every multiplicity label `alpha`,

\[
 C_{\mathrm{col},K}|\alpha\rangle
 =\frac{1}{\sqrt3}\sum_c
 |\alpha,c\rangle_{\mathrm{M2}}|\bar c\rangle_{\mathrm{spec}},
 \qquad C_{\mathrm{col},K}^\dagger C_{\mathrm{col},K}=I.
\]

Consequently,

\[
 C_{\mathrm{col},K}^\dagger
   (O_K\otimes I_{\bar3})
 C_{\mathrm{col},K}
 =\bar O_K.
\]

The basis qualification is essential. C53 is natively stored in C47
shell-major order. C117 owns the C128/C112 q sector followed by the C47 qg
sector. The M2 H0 target instead retains the C401/C410 ordering, including the
C128 transverse sub-order. C53 and the qg part of C117 therefore require the
exact C47-to-target permutation before they can be combined with H0. Equal
dimensions do not establish a common basis. Interpreting raw C117 as though it
were already in target order gives a false K9 `Jz`-commutator residual
`0.9075188466863634`; the mapped operator has exactly zero residual.

This construction is implemented in
`src/deuteron_wigner/quantum/m2_color_singlet_bridge.py`. It rejects an
operator if it mixes colors or gives unequal actions to the three color
components.

## Result established by the first dynamical launch

`src/deuteron_wigner/quantum/m2_c53_mixed_singlet_state.py` now performs the
first parameter-explicit dynamical calculation in the reduced K9 singlet
space. It maps both C53 and C117 to the M2 target order, independently replays
the reduced C52/C53 emission route, and solves the lowest state separately in
the exact `Jz=+/-1/2` sectors. The named point
`M2_C53_MIXED_K9_EXPLORATORY_BASELINE_V1` uses the existing nonphysical H0
baseline and an explicit dimensionless C53 coupling `g_s=0.05`. The C53
matrix already has `GeV^2` units; neither this coupling nor its diagnostic
C52 mass and plus-momentum inputs are a physical fit.

At this point the 450-dimensional reduced space has q/qg dimensions 2/448.
The C53 operator descends from 624 full nonzeros to 208 reduced nonzeros. The
independently replayed reduced route agrees exactly with the sparse descent,
the full-product color route agrees to `8.9e-16`, and the complete Hamiltonian
is exactly Hermitian and `Jz` conserving. Its lowest helicity pair has

\[
 E_{-1/2}=0.05007934225260527\ \mathrm{GeV}^2,
 \qquad
 E_{+1/2}=0.05007934225260526\ \mathrm{GeV}^2,
\]

with q/qg weights `0.918006079222261/0.081993920777739` on each branch and a
gap of `0.5566367615 GeV^2` after the pair. Sparse, direct matrix-free, and
`LinearOperator` actions agree to `6.3e-17`. The zero-coupling limit returns
the two color-completed q helicity states at
`0.1945863740838668 GeV^2`, with zero qg weight and a corrected gap
`0.412129729715685 GeV^2`.

Hellmann--Feynman and central finite-difference derivatives for both C396
directions, C117, and C53 agree within `2.6e-10` at step `1e-6`. C117 already
has a q-sector slope `-0.07733751308761731` at zero C53 coupling; generated qg
support changes that slope to `-0.1278632853757897`. Thus C53 changes the
C117 response but is not required to make the first C117 response nonzero.
All of these numbers remain exploratory state-construction diagnostics.

## Result established by the spectator-composition launch

`src/deuteron_wigner/quantum/m2_nucleon_spectator_composition.py` now adds the
missing spectator spin/flavor tensors without choosing a physical mixture.
It exports four separate K9 members: scalar and axial spectator families for
the proton and neutron. The scalar member has
`(S_spec,I_spec,L_spec)=(0,0,0)`; the axial member has `(1,1,0)`. Both use the
exact anti-triplet color tensor already matched to C7. Their ordered two-quark
spin and flavor tensors have the required exchange symmetry, so color
antisymmetry times S-wave spin--flavor symmetry gives total spectator-quark
exchange eigenvalue `-1`.

The axial members use explicit Condon--Shortley couplings

\[
 |\tfrac12,+\tfrac12\rangle
 =\sqrt{\tfrac23}|1,+1\rangle|\tfrac12,-\tfrac12\rangle
 -\sqrt{\tfrac13}|1,0\rangle|\tfrac12,+\tfrac12\rangle,
\]

with the corresponding negative-projection partner, in both spin and
isospin. Every retained component has exact proton/neutron charge, baryon
number one, target `Jz`, and target `Iz`. The abstract spin- and isospin-half
Casimir residuals are zero for the scalar member and `2.8e-16` for the axial
member. This treats the exact C53 `Jz` doublet as an abstract spin-half carrier
for recoupling; it does not establish full light-front rotational covariance.

Each state is normalized, the two target helicities are orthogonal, and the
unpolarized doublet reduces after tracing the added labels to one half of the
original C53 pair projector. This last statement is the correct invariant
reduction: an axial state at fixed target helicity is entangled with spectator
spin and does not reduce to one pure active-helicity vector. The inherited
energy and qg weight are unchanged because the spectator Hamiltonian is only
an identity extension. No spectator kinetic energy, interaction, scalar/axial
mixture, physical nucleon, deuteron, or current has been selected.

## Result established by the first separately resolved NN-state launch

`src/deuteron_wigner/quantum/m2_nn_deuteron_state.py` now binds each typed
proton/neutron spectator family, separately, to a finite Volume-IV light-front
NN spin-one amplitude. The internal QCD label remains K9 while the nuclear
quadrature owns the independent resolution
`NN_AV18_R1_NK24_NC14_NP16_KMAX2GEV`; the two labels are explicit in
`H_D^(r_D;K9)::NN_ONLY` and cannot be substituted for one another.

The named wave-function member is the repository AV18 momentum table
`data/raw/av18/deut.wfk`, SHA-256
`6bfe39d084e692c75a1bbbb921e13a7941c6ee5b7de7b71a24e010da9ab35523`.
Its source norm is `1.0000053605388606` and its source D-state probability is
`0.05759854074095002`. The implementation makes the units conversion explicit,

\[
 k_{\mathrm{fm}^{-1}}=k_{\mathrm{GeV}}/(\hbar c),\qquad
 (u,w)_{\mathrm{GeV}^{-3/2}}=(u,w)_{\mathrm{fm}^{3/2}}/(\hbar c)^{3/2},
\]

and evaluates the existing flat-normalized Melosh-rotated light-front
amplitude on a `24 x 14 x 16 = 5376` node spherical quadrature through
`2 GeV`. One common factor `0.9998736629799589` normalizes all three target
helicities; it is not tuned by helicity or observable. The resulting spin-one
Gram residual is `7.8e-16`.

The S and D amplitudes remain coherent arrays, reconstruct the full amplitude
to `1.93e-17`, and give discrete probabilities
`0.9424128212803198` and `0.05758717871968042`. Integrated S--D interference
vanishes at about `1e-16` by angular orthogonality, while the local signed
interference reaches `1.481158104020323e-4` and is therefore retained rather
than discarded. The nuclear isospin factor is explicitly

\[
 |I=0,I_z=0\rangle=\frac{|pn\rangle-|np\rangle}{\sqrt2},
\]

with exact exchange and `I^2` residuals, baryon number two, and charge one.
Both scalar and axial nucleon-family members satisfy these ledgers and inherit
the internal qg weight `0.08199392077773901`; no mixture is selected.

This is the first lawful exploratory NN spin-one state on a separately owned
nuclear resolution. It is not an eigenvector of a constructed nuclear
Hamiltonian and has no derived deuteron mass. At that increment nonnucleonic
sectors were named omitted interfaces and the transfer/current layer was still
unconstructed; the next result records the diagnostic continuation.

## Result established by the first transfer/current launch

`src/deuteron_wigner/quantum/m2_nn_transfer_current.py` now constructs
distinct initial and final momentum fibers in the symmetric zero-skewness
frame and evaluates a finite-grid one-body plus-current diagnostic. All
off-forward S/D overlaps consume the single recoil authority
`VOLUME_IV_SPECTATOR_PRESERVING_XI0_RECOIL_V1`, now factored in
`light_front.py`: the active nucleon receives the full physical `Delta_T`, the
physical spectator receives exactly zero, the intrinsic momenta close in both
fibers, and the affine Jacobian is one.

The named first current member is deliberately pointlike and nonphysical:
`F1p=1`, `F1n=F2p=F2n=0`. At `Delta_T=(0.2,0) GeV`, `Q^2=0.04 GeV^2`,
caller-supplied kinematic reference mass `1.87561294257 GeV`, and
`P^+=2 GeV`, the scalar-family target matrix is

\[
 I^+\simeq
 \begin{pmatrix}
  0.61907812 & 0.00185957 & -0.04319585\\
 -0.00185957 & 0.57897165 &  0.00185957\\
 -0.04319585 &-0.00185957 &  0.61907812
 \end{pmatrix},
\]

with imaginary parts below `7e-18`. Charge at `Q^2=0` closes to `7.8e-16`,
the full `I(-Delta)=I(Delta)^dagger` residual is `1.2e-16`, parity and the
`I_0+ = -I_+0` relation close below `2e-17`, and the SS/SD/DS/DD components
reconstruct exactly. Reversal swaps SD with DS, as required; the off-diagonal
S--D current pieces are nonzero and retained.

The Carlson--Ji angular-condition residual is
`1.4975815808637716e-4`, or relative violation
`1.2028327047759704e-4`. It is reported, not repaired. This is evidence of the
one-body/S--D light-front truncation and the still-missing exchange, pair,
zero-mode, interaction, pion, Delta, short-distance, and nonnucleonic
transition currents. The frame mass is explicitly a caller kinematic
reference, not a deuteron mass eigenvalue. No Hamiltonian-consistent or
production current has been selected.

## Exact relation to the older C7 nucleon color space

Realize the spectator `3bar` as the antisymmetric color pair

\[
 D_{bc,\bar a}=\frac{\epsilon_{abc}}{\sqrt2},
 \qquad
 D^\dagger D=I_{\bar3}.
\]

The total-generator relation

\[
 (T^A\otimes I+I\otimes T^A)D=D\,\bar T^A
\]

holds exactly in the C7 generator convention. Completing the M2 `q` sector
with this spectator produces the unique C7 `qqq` singlet up to one global
phase. Completing the C47 triplet-projected `qg` sector produces one specific
direction in C7's two-dimensional `qqqg` singlet space. In the current C7
canonical basis its coordinates are, up to numerical roundoff,

\[
  \left(-\frac{\sqrt3}{2},\frac12\right).
\]

This is the normalized C7 active-quark-emitter direction with the same global
phase as the completed `qqq` state. The orthogonal C7 `qqqg` singlet remains
`UNMAPPED_NOT_ZERO`. Thus C7 is no longer wholly disconnected from M2: its
color map is fixed, while its spin, flavor, longitudinal, orbital, and
normalization maps remain to be constructed.

## What this does and does not solve

The completion removes the SU(3) obstruction identified by
`m2_state_current_boundary.py`; the first NN launch supplies one separately
resolved exploratory spin-one state, and the transfer launch supplies its
first finite-resolution one-body current diagnostic. These still do not
produce a physical deuteron eigenstate or a complete state/current family.

In particular, it does not yet own:

- a dynamically selected spectator Hamiltonian or scalar/axial mixture;
- a physical proton/neutron mass or full light-front rotational completion;
- spectator longitudinal momentum and internal orbital dynamics beyond the
  named exploratory S-wave composition;
- a nuclear Hamiltonian whose eigenvector and eigenvalue define the deuteron;
- nonnucleonic `NNpi`, `DeltaDelta`, six-quark, hidden-color, and other sectors;
- a Hamiltonian-consistent source-qualified nucleon and deuteron current;
- a derived two-body transfer partition and exchange, pair, zero-mode,
  interaction, pion, Delta, short-distance, or nonnucleonic current terms;
- a production light-front/LPS current selection.

The M2 and nuclear resolutions must remain distinct. K9/K11/K13 label the
internal finite QCD basis. Volume IV defines an independent nuclear resolution
datum `r_D` containing its own basis/grid, Fock content, Hamiltonian and
operator orders, regulator, and scheme. The implemented NN-state API therefore
carries both labels,

\[
 \mathcal H_D^{(r_D;K)},
\]

instead of silently identifying a nuclear grid with an internal M2 `K`.

## Chosen construction route

The project should follow Volume IV's hierarchical route rather than jump to
an unqualified six-quark deuteron:

\[
 \text{M2 quark--gluon nucleon space}
 \xrightarrow{\;C_{\mathrm{col},K}\;}
 \text{color-singlet nucleon space}
 \xrightarrow{\;\mathcal Z_{qg\to\mathrm{had}}\;}
 \mathcal H_D^{(r_D;K)}
 \xrightarrow{\;J_D^\mu\;}
 \mathcal H_D^{(r_D;K)}.
\]

The first deuteron model space should be the controlled nucleonic sector

\[
 \mathcal H_{D,0}^{(r_D;K)}=\mathcal H_{NN}^{(r_D;K)},
 \qquad B=2,\ Q=+1,\ I=0,\ J^P=1^+,
\]

with `NNpi`, `DeltaDelta`, six-quark, hidden-color, and other sectors retained
as explicit omitted interfaces rather than zeros. This implemented
exploratory state is the smallest space
that can use the existing normalized spin-one light-front deuteron amplitude
and the exact Volume IV recoil map without claiming that the NN truncation is
complete.

The initial and final fibers must remain distinct:

\[
 J_D^\mu(P_f,P_i):
 \mathcal H_{D,P_i}^{(r_D;K)}
 \longrightarrow
 \mathcal H_{D,P_f}^{(r_D;K)},
 \qquad \Delta^+=0.
\]

For the one-body pilot, the active nucleon receives the full physical
transverse transfer while the deuteron wave-function arguments shift by
`(1-y) Delta_T / 2`. This must consume the existing recoil authority rather
than reimplement its signs.

## Critical dynamical point

The unmixed M2 bundle is block diagonal in q and qg, and its K9 ground pair is
the reduced q block. The gluon-mass direction therefore has zero derivative
there. C117 is different: after the required basis map, its q block is a
nonzero scalar diagonal action, so its first-order derivative is already
nonzero in the unmixed pair. C53 creates qg support and changes that response.

The source-derived C53 canonical vertex is now the tested first exploratory
mixing operator. It is Hermitian after adjoining absorption, preserves exact
`Jz`, and passes color descent and two independent application routes. Its
coefficient remains explicit and nonphysical until the coupling and
renormalization conditions are fixed. The old C7/H1 Hamiltonian must not be
imported wholesale by dimension or name; only explicitly mapped components
may enter.

## Completed spectator-composition increment

The C53-mixed color-singlet helicity pair is complete at the exploratory
contract level. The next completed code change attaches the nucleon quantum
numbers before any nuclear composition:

1. introduce a typed nucleon-composition tensor that owns spectator-pair spin,
   flavor, charge, baryon number, orbital projection, phase, and normalization;
2. couple those spectator labels to the existing active q/qg helicity pair and
   require exact `Jz`, charge, baryon, flavor, color, and normalization ledgers;
3. do not provide a scalar-diquark or axial-diquark default—those are distinct
   exploratory members unless derived from the microscopic state;
4. retain the C7 orthogonal qqqg singlet and any absent spectator structures as
   `UNMAPPED_NOT_ZERO`, not as zero amplitudes;
5. expose the resulting proton and neutron helicity doublets as inputs to the
   Volume IV light-front composition without identifying internal K with the
   nuclear resolution.

That typed increment is complete, and the code has now formed the first
separately resolved exploratory `NN` spin-one composition. The current
intertwiner remains the next construction.

## Completed acceptance checks for the C53 launch

- The C53 block descends by the same exact singlet projection as H0/C396/C117.
- Sparse, reduced-singlet, and matrix-free actions agree.
- Hermiticity and color-singlet generator residuals are below declared
  numerical tolerances.
- The `q`/`qg` state weights sum to one and change continuously as the explicit
  exploratory C53 coefficient tends to zero.
- The zero-coupling limit returns the accepted two-helicity color-completed M2
  ground space; it must not reintroduce the threefold open-color degeneracy.
- Hellmann--Feynman and finite-difference derivatives agree for H0, both mass
  directions, C53, and C117 at the subspace level.
- C117's existing q-sector response and its qg-induced change are separately
  reported; neither is mislabeled as a physical coefficient.
- No individual vector is selected inside a degenerate helicity multiplet.
- C7's orthogonal `qqqg` singlet remains `UNMAPPED_NOT_ZERO`; missing spectator
  structures, omitted nuclear sectors, and unmatched current terms remain
  `UNAVAILABLE_NOT_ZERO`.
- Every result remains marked exploratory and nonphysical.

## Completed acceptance checks for the spectator-composition increment

- Proton and neutron charge, baryon number, flavor, color, and `Jz` ledgers are
  exact in every included q and qg component.
- The spectator-pair spin/flavor/orbital basis and all Clebsch--Gordan phases
  are explicit, normalized, and independently checkable.
- Scalar- and axial-spectator choices remain named alternatives unless a
  microscopic relation fixes their mixture.
- The unpolarized constructed doublets reduce to one half of the
  color-completed M2 helicity-pair projector when spectator labels are traced
  out; fixed-helicity axial reductions retain their required entanglement.
- Missing C7 singlet directions and higher spectator/Fock structures remain
  explicit omitted interfaces rather than numerical zeros.
- No object is called a physical deuteron: the now-completed Volume-IV NN
  composition is explicitly exploratory and remains upstream of a nuclear
  Hamiltonian, mass eigenvalue, transfer fibers, and current.

## Completed increment: the first separately resolved NN spin-one space

The typed color-singlet proton and neutron doublets are now bound to the
existing
Volume IV light-front two-nucleon amplitude:

\[
 |D,\Lambda\rangle_{NN}
 =\sum_{\lambda_p,\lambda_n}
  \int[d\Gamma_2]\,
  \Psi^{\mathrm{LF}}_{\Lambda;\lambda_p\lambda_n}
  |p,\lambda_p\rangle|n,\lambda_n\rangle.
\]

The calculation evaluates the scalar and axial spectator members separately
and requires a caller to name the channel. It uses one explicitly named
wave-function member,
the exact `S`/`D` interference, Melosh convention, and normalization adapter.
The first member is AV18; CD-Bonn, Norfolk, and analytic fixtures remain
alternative future members, not amplitudes to combine. The implementation
reuses the tested spin mathematics in `light_front.py`; the validation-only
hard-coded numerical reports in `nuclear/n0/core.py` are not physical
state/current evidence.

## Completed increment: finite-resolution transfer and one-body current

The first current is an explicitly scoped one-body diagnostic on this space.
It constructs distinct initial and final momentum fibers and consumes the
existing Volume-IV recoil authority, including the `(1-y) Delta_T / 2`
wave-function shift, without duplicating its signs. It preserves charge at
`Q^2=0`, Hermiticity/reversal, and parity while explicitly reporting the
nonzero spin-one angular-condition residual. Two-body, exchange, zero-mode,
and interaction currents remain separate missing terms.

## Completed increment: current response to the microscopic directions

At the same named transfer/current point, propagate controlled variations of
the C396 mass directions, mapped C117 action, and C53 mixing coefficient
through the normalized helicity pair, typed nucleon family, NN state, and
finite-resolution current. Compare central finite differences with any lawful
state-derivative route and report the response matrix and singular spectrum as
diagnostics only. Keep the pointlike current fixed for this derivative audit;
a source-qualified nucleon form-factor member and the LF-versus-LPS production
choice are separate later upgrades. The parallel C117 normalization derivation
remains open and no exploratory sensitivity is a physical matching
coefficient.

This increment is now implemented in
`src/deuteron_wigner/quantum/m2_nn_current_response.py`. The four parameter
directions are `(D_mu_q_sq, D_delta_mu_g_sq, c117, g_C53)`. Gauge-aligned
central differences at step `1e-6` give nonzero internal-state tangent norms
for every direction. After scaling the columns by the declared diagnostic
increments `(0.05 GeV^2, 0.05 GeV^2, 0.03, 0.01)`, the internal tangent
singular values are

```text
0.08078722386, 0.02426013412, 0.01423771167, 0.004734462258
```

and its diagnostic rank is four. The state is therefore moving in four
resolved directions. The two exact-`Jz` branches have common energy response,
to the shown precision,

```text
(1.0692774173, 0.3153320613, -0.1278632854, -5.3063333483)
```

and qg-weight response

```text
(-0.0689762606, -0.3741507989, 0.0638487212, 2.4427030062).
```

Nevertheless, all eight independently recomputed endpoint currents are
identical. The current amplitude Jacobian for `(I11,I10,I1m1,I00)` is the
exact `4 x 4` zero matrix, its four singular values are zero, and the angular-
condition derivative is zero. This follows from the implemented dependency:
the pointlike current consumes the fixed nuclear wave grid, transfer, and
nucleon current member and acts as the identity on the normalized internal
M2 factor. The baseline angular residual remains nonzero at
`1.4975815808637716e-4`; it is insensitive to these internal variations only
because the present current has no internal operator content.

Thus diagnostic current rank zero is an interface-blindness result, not a
physical irrelevance statement. It does not set exchange, pair, zero-mode,
interaction, pion, Delta, short-distance, nonnucleonic, or microscopic
transition currents to zero.

## Completed increment: active-quark internal current and first non-null response

M2h now supplies the smallest source-derived current supported by the retained
M2 basis. At zero skewness and Wilson order zero, the normalized canonical
quark plus current is diagonal in retained Fock number. Its q block is the
intrinsic identity. For each qg longitudinal partition it uses the exact C45
oscillator overlap with the C47/Volume-XI recoil

\[
 \delta q_{{\rm rel},T}=\sqrt{x_g/x_q}\,\Delta_T.
\]

The 450-dimensional K9 matrix has exact forward charge, adjoint transfer
reversal, explicit finite-shell leakage, and no threshold pruning. Canonical
number-changing, induced, constrained, zero-mode, and operator-matching terms
remain separate unavailable operators rather than physical zeros.

At `Delta_T=(0.2,0) GeV`, projection onto the C53 helicity pair gives a
nonidentity 2-by-2 current. Repeating the four-direction response analysis
produces scaled internal-current singular values

```text
0.001785981969, 0.000240331244, 1.42e-12, 1.21e-16
```

and diagnostic rank two. The direct endpoint derivative agrees with the
state-tangent derivative within `2.9e-10`. The M2g pointlike current remains
the exact rank-zero identity control.

The current two-quark spectator completion has no resolved momentum or
Hamiltonian. A separately labeled exploratory lift therefore combines the
active current with exact flavor charges and a pointlike spectator-charge
identity. It closes proton, neutron, and deuteron forward charge and passes
reversal, parity, and coherent S/D diagnostics, but it is not complete
microscopic-to-nucleon matching. Through the AV18 NN convolution it also has
diagnostic rank two above the declared finite-difference/quadrature floor. The
angular-condition residual remains nonzero and acquires a nonzero microscopic
derivative; it is not repaired.

The implementation and derivation are
`src/deuteron_wigner/quantum/m2_active_quark_current.py`,
`src/deuteron_wigner/quantum/m2_internal_current_response.py`, and
`docs/next_level/m2_active_quark_current_result.md`.

## Result established by the resolved-spectator launch

M2i now gives the two-quark spectator an explicit finite-`K` momentum and
projected free-Hamiltonian embedding. The complete K9 active cluster and two
exchange-related spectator-quark modes `(9,9)` close at nucleon `K2=27`,
so the three valence quarks have exact equal longitudinal shares in the
`qqq` sector. Spectator-pair and cluster-relative Jacobi coordinates are
retained in normalized C45 ground modes at the common K9 scale
`b=0.4 GeV`.

The nested free invariant mass is

\[
 M_N^2=\frac{M_A^2}{y_A}
 +\frac{M_S^2}{y_S}+q_\lambda^2,
 \qquad
 M_S^2=\sum_{i=1}^2\frac{m_{s_i}^2}{z_i}+q_\rho^2.
\]

At the named exploratory point its projected action is
`3 M_A^2 + 1.60 GeV^2`; the K9 helicity pair remains an exact eigenpair to
`1.1e-15 GeV^2`. This is a regulator member, not a physical nucleon mass.
The earlier `2.88 GeV^2` value applied the daughter-fraction weights to the
already normalized transverse variables twice and is superseded.

The active and both spectator diagonal quark currents now consume the same
longitudinal fractions, C45 modes, basis scale, and Volume-XI recoil map. At
`Delta_T=(0.2,0) GeV` the active-cluster and spectator-quark spatial factors
are `0.8824969025845955` and `0.9105103613800342`. Forward charge,
transfer reversal, parity, and S/D reconstruction close without a pointlike
spectator identity. The scalar and axial families remain separate and both
retain diagnostic response rank two. The nonzero angular defects are reported,
not repaired.

The implementation and derivation are
`src/deuteron_wigner/quantum/m2_resolved_spectator_current.py`,
`src/deuteron_wigner/quantum/m2_resolved_spectator_response.py`, and
`docs/next_level/m2_resolved_spectator_current_result.md`.

## Completed increment: M2j color/orbit-sign C7/H2 embedding

M2j constructs the exact S3 sign projector on the abstract six-element
quark-slot orbit and the complete C7 singlet spaces before assembling target
operators. The color/orbit sign multiplicities are one for `qqq` and two for
`qqqg`. The complete
450-dimensional M2i source is embedded isometrically into a target with
dimensions

\[
 \dim\mathcal H_{qqq}=2,\qquad
 \dim\mathcal H_{qqqg}=448\times2=896,\qquad
 \dim\mathcal H_{\mathrm{M2j}}=898.
\]

The source image has rank 450. The retained orthogonal `qqqg` singlet channel
has rank 448, is exactly disjoint from the source image, and is explicitly
marked unsourced rather than physically zero. Every longitudinal tuple closes
at nucleon `K2=27` in three-quark-then-gluon order.

H0, the C396 mass directions, and the M2i nested-Jacobi shift are
Fock-diagonal color scalars and extend to both singlets. The represented C117
and C53 interactions extend only through the exact active-emitter isometry.
The resulting 898-dimensional Hamiltonian is genuinely interacting and pulls
back to M2i at `2.7e-15`, but it remains a partial H2 Hamiltonian because the
noncolor S3 basis and orthogonal-channel interactions are not derived. The
source active-quark current is likewise extended as a color scalar to both
singlets; forward charge and transfer reversal close at numerical precision.

The implementation and derivation are
`src/deuteron_wigner/quantum/m2_permutation_complete_h2.py` and
`docs/next_level/m2_permutation_complete_h2_result.md`. The synthetic
historical C9/H2 numerical Hamiltonian is not consumed.

## Completed increment: M2k emitter-tree and C53 orbit closure

M2k derives all six equal-share Jacobi tree maps. They preserve the exact
free-mass metric `diag(3/2,1)` and become orthogonal in the global coordinate
`Q_rho=q_rho/sqrt(y_S)`. The M2i equal-local-width ground has cyclic overlap
exactly `32/33`, so the 898-state M2j color/orbit target is not a complete
noncolor S3 basis. Exact one-ground-mode closure requires
`b_rho=sqrt(2/3)b_lambda=0.32659863237109044 GeV`; retaining the M2i widths
instead requires the recoupled excited Jacobi tail.

The same audit constructs the canonical C53 orbit sum. Although the three raw
same-orbit color insertions span rank two, the properly covariant projected
emitter branches have rank one. The normalized orbit sum is exactly M2j's
active-emitter direction, with residual `2.2e-16`, and has no source in the
retained orthogonal singlet. Two independently normalized spectator-emitter
C53 copies are therefore neither required nor licensed.

The implementation and derivation are
`src/deuteron_wigner/quantum/m2_emitter_tree_recoupling.py` and
`docs/next_level/m2_emitter_tree_recoupling_result.md`.

## Completed increment: M2l global-ground spin--flavor S3 completion

M2l selects the separately identified symmetry-adapted global ground and
constructs the exact 64-state two-flavor three-quark spin--flavor carrier. At
fixed proton/neutron isospin and helicity, the symmetric `S=I=1/2` state is
unique. Its spectator-pair decomposition has scalar and axial probabilities
exactly `1/2` and no mixed `(S,I)=(0,1),(1,0)` component. This is a derived
property of the common-orbital symmetry limit, not a physical mixture fit.

The exact qqqg color-times-spin--flavor sign projector has rank 20. Applying
the source-owned C53 kernel to all three quark slots proves that its spin-scalar
part cancels by color neutrality and all 80 nonzero spin channels lie in the
total sign sector. Tracing spin--flavor gives color density `I_2/2`: the M2j
active and formerly orthogonal bare-color coordinates each have weight `1/2`.
They are the entangled components of one canonical source and cannot receive
independent couplings.

The implementation and derivation are
`src/deuteron_wigner/quantum/m2_spin_flavor_s3_completion.py` and
`docs/next_level/m2_spin_flavor_s3_completion_result.md`.

## Completed increment: M2m total-sign H2 Hamiltonian

M2m represents the source-supported global-ground Hamiltonian on the complete
fixed-target, fixed-quark-spin total-sign carrier. The C53 ray alone would
give a 162-state truncation, but 104 additional carrier directions are
required by the rank-two fixed-spin sectors. The resulting per-target space
has dimension 266, split into two exact 133-state `Jz` blocks. H0, C396, and
represented C117 are spin/color scalar and preserve all 80 C53-reached
external modes with zero leakage. C53 projects into the complete carrier at
`1.7e-15` residual and is inserted once with no second color coupling.

At the named exploratory point the lowest pair has
`M^2=1.789591467572781 GeV^2`, qqqg weight
`0.056666562870487`, and proton/neutron spectral agreement below `7e-16
GeV^2`. The zero-C53 state is pure qqq. This is the exact represented
source-supported H2 matrix, not the complete physical H2 Hamiltonian. See
`src/deuteron_wigner/quantum/m2_total_sign_h2_hamiltonian.py` and
`docs/next_level/m2_total_sign_h2_hamiltonian_result.md`.

## Completed increment: M2n Hamiltonian-continuity current

M2n gauges the represented M2m system with a normalized baryon-number U(1)
probe, assigning each quark charge `1/3` and the gluon zero. The density uses
the same active qg oscillator, global lambda ground, and symmetry-adapted rho
ground. It is the identity at zero transfer and obeys adjoint transfer
reversal.

The longitudinal minimum-norm current is fixed by

\[
 \Delta_T\cdot\mathbf J_T(\Delta)
 =H\rho(\Delta)-\rho(\Delta)H.
\]

Continuity closes at `2.8e-17 GeV^2`. Gauging C53 produces a nonzero, strictly
qqq-to-qqqg current block with maximum magnitude `0.31026505994186515 GeV` at
`Delta_T=(0.2,0) GeV`. This is an Abelianized finite-basis benchmark, not a
complete electromagnetic or non-Abelian current. See
`src/deuteron_wigner/quantum/m2_hamiltonian_gauged_current.py` and
`docs/next_level/m2_hamiltonian_gauged_current_result.md`.

## Completed increment: M2o PennyLane implementation contract

M2o preserves the 266-state order in a 512-state, nine-qubit big-endian
encoding. Either Hamiltonian `Jz` block fits in eight qubits. Because zero
padding would create false variational states below the positive physical
spectrum, the contract adds a `37.20183718786931 GeV^2` padding penalty above
the certified Gershgorin upper bound while leaving the physical block exactly
unchanged. Directed density/current pairs are exported as six Hermitian
cosine/sine quadratures. Compact/padded actions, state round trips, physical
projectors, penalty isolation, and encoded eigenpairs close numerically.

The system is now ready for PennyLane implementation through a sparse exact
oracle. No ansatz, optimizer, hardware target, physical parameter fit, or
activation has been selected. See
`src/deuteron_wigner/quantum/m2_pennylane_contract.py` and
`docs/next_level/m2_pennylane_contract_result.md`.

## Stop boundary reached: PennyLane implementation is the next task

The requested scientific preparation is complete. The next action is to
implement the nine-qubit `lightning.qubit` sparse oracle and verify it against
the compact SciPy matrices. Choosing a variational ansatz/optimizer is a
separate computational decision; selecting a physical electromagnetic
current, completing the missing H2/current terms, fitting parameters, or
activating physical predictions remains a later physics/executive decision.

## Primary theory anchors

- `references/volume_iv_matched_spin1_nuclear_dynamics.tex`, especially the
  nuclear Hilbert-space, normalized NN state, recoil, matched-operator, and
  current sections.
- `references/volume_vii_concrete_microscopic_nucleon_hamiltonian.tex` and
  `references/volume_ix_dynamical_gluon_fock_sectors.tex` for the nucleon Fock
  and color structure.
- `references/volume_xi_microscopic_nonzero_transfer_gtmds.tex` for the active
  and spectator recoil and common microscopic overlap.
- `references/DeuteronWigner_complete_theory_note_current.tex` for current
  convention and claim boundaries.
- Carbonell et al., *Explicitly Covariant Light-Front Dynamics and Relativistic
  Few-Body Systems*, arXiv:nucl-th/9804029, for the warning that a two-component
  S/D realization is a truncation rather than the most general covariant
  light-front deuteron state.
- Cooke and Miller, *Deuteron binding energies and form factors from light
  front field theory*, arXiv:nucl-th/0112037, for rotational breaking and the
  angular condition in truncated light-front deuteron calculations.

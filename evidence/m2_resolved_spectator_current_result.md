# M2i resolved spectator Hamiltonian and matched diagonal one-body current

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

## Result

M2i replaces M2h's momentum-independent spectator-charge identity with the
smallest exchange-symmetric finite-`K` spectator member that preserves the
complete K9 active subsystem. It supplies:

1. exact longitudinal modes for the active cluster and both spectator quarks;
2. normalized transverse Jacobi wave functions for the spectator pair and the
   active-cluster--spectator-cluster separation;
3. a projected free invariant-mass Hamiltonian embedding; and
4. active and both spectator diagonal quark currents evaluated with the same
   basis scale, longitudinal fractions, and zero-skewness recoil convention.

The implementation is in
`src/deuteron_wigner/quantum/m2_resolved_spectator_current.py` and
`src/deuteron_wigner/quantum/m2_resolved_spectator_response.py`. The
focused regression is
`tests/test_m2_resolved_spectator_current.py`.

This is an exploratory nested-cluster regulator member. It is not the complete
antisymmetrized C7/H2 `qqq + qqqg` basis, contains no spectator or
active--spectator interaction, does not select a scalar/axial mixture, and is
not a physical nucleon solution or production current.

## Finite-`K` longitudinal embedding

The active subsystem retains its full C47 K9 resolution,

\[
 K_{A,2}=9,
 \qquad K_A=\frac92.
\]

The first exchange-symmetric equal-share lift assigns each spectator quark the
same allowed antiperiodic mode,

\[
 k_{s_1,2}=k_{s_2,2}=9,
 \qquad K_{N,2}=27,
 \qquad K_N=\frac{27}{2}.
\]

Thus the cluster and constituent fractions are exact:

\[
 y_A=\frac{K_{A,2}}{K_{N,2}}=\frac13,
 \qquad y_S=\frac23,
 \qquad x_{s_1}=x_{s_2}=\frac13,
 \qquad z_1=z_2=\frac12.
\]

In the `qqq` sector the three quark modes are `(9,9,9)` in
doubled notation. In every retained `qqqg` partition, the active K9 quark
and gluon modes still sum to 9 and the two spectator modes add 18, so every
sector closes exactly at 27. The M2 active-cluster fractions remain internal
fractions; the corresponding nucleon fractions are multiplied by \(y_A\).

This lift is deliberately not identified with the older C7 reference
resolution. Its color tensors have already been matched to C7, but a complete
momentum/permutation basis map is a later step.

## Nested Jacobi coordinates and projected Hamiltonian

Let \(\kappa_A\) and \(\kappa_S=-\kappa_A\) be the cluster intrinsic
momenta. Inside the spectator cluster let

\[
 \ell_1=\kappa_{s_1}-z_1\kappa_S,
 \qquad \ell_2=-\ell_1.
\]

The normalized Jacobi variables are

\[
 q_\lambda=\frac{\kappa_A}{\sqrt{y_Ay_S}},
 \qquad
 q_\rho=\frac{\ell_1}{\sqrt{z_1z_2}}.
\]

M2i retains the C45 ground mode \((n,m)=(0,0)\) in both coordinates and uses
the same oscillator scale as K9,

\[
 b_\rho=b_\lambda=b_{K9}=0.4\ {\rm GeV},
 \qquad
 \langle q_\rho^2\rangle
 =\langle q_\lambda^2\rangle=b^2=0.16\ {\rm GeV}^2.
\]

The Volume-VII free invariant-mass identity factorizes exactly as

\[
 M_N^2
 =\frac{M_A^2}{y_A}
 +\frac{M_S^2}{y_S}
 +q_\lambda^2,
 \qquad
 M_S^2
 =\sum_{i=1}^2\frac{m_{s_i}^2}{z_i}+q_\rho^2.
\]

The normalized transverse coordinate contributes once at the level where it
is introduced. The earlier M2i draft divided both normalized Jacobi energies
by daughter fractions a second time; M2k's recoupling audit exposed that
error, and the implementation and reference values have been corrected.

At the named exploratory point, the spectator quark mass-squared is tied to
the same C396 coordinate as the represented active quark,
\(m_{s_i}^2=\mu_q^2=0.20\ {\rm GeV}^2\). Projection onto the two Jacobi ground
modes gives

\[
 M_S^2=0.96\ {\rm GeV}^2,
 \qquad
 M_{N,\mathrm{M2i}}^2=3M_A^2+1.60\ {\rm GeV}^2.
\]

The two K9 helicity states remain exact eigenvectors of this projected
embedding, with equal diagnostic eigenvalues

\[
 1.750238026757816\ {\rm GeV}^2,
\]

and residual \(1.1\times10^{-15}\ {\rm GeV}^2\). This number is not a nucleon
mass prediction: the projected free member omits interactions, basis tails,
full antisymmetrization, and renormalization.

## Recoil and matched diagonal current

For an active-quark hit, the active cluster receives the physical transfer and
the spectator cluster is preserved. Besides M2h's internal shift,

\[
 \delta q_{qg}=\sqrt{x_g/x_q}\,\Delta_T,
\]

the external Jacobi coordinate shifts by

\[
 \delta q_\lambda^{(A)}
 =\sqrt{\frac{y_S}{y_A}}\,\Delta_T
 =\sqrt2\,\Delta_T.
\]

For a spectator-quark hit, the active cluster is physically unchanged, so the
signed external shift is

\[
 \delta q_\lambda^{(s)}
 =-\sqrt{\frac{y_A}{y_S}}\,\Delta_T
 =-\frac{\Delta_T}{\sqrt2}.
\]

The internal spectator-pair shifts are

\[
 \delta q_\rho^{(s_1)}
 =+\sqrt{\frac{z_2}{z_1}}\,\Delta_T,
 \qquad
 \delta q_\rho^{(s_2)}
 =-\sqrt{\frac{z_1}{z_2}}\,\Delta_T.
\]

The opposite signs are immaterial for the retained S-wave overlap but are
stored explicitly because they matter as soon as orbital spectator modes are
added.

Writing \(J_A\) for the M2h active-subsystem current, the represented nucleon
one-body current is

\[
 J_N^{(1)}(\Delta_T)
 =e_a F_\lambda^{(A)}(\Delta_T)J_A(\Delta_T)
 +\sum_{i=1}^2 e_{s_i}
 F_\lambda^{(s_i)}(\Delta_T)
 F_\rho^{(s_i)}(\Delta_T)\,\mathbf 1_A.
\]

All overlaps are direct C45 displaced-mode integrals. For the ground mode,

\[
 F_0(s)=\exp\!\left[-\frac{s_T^2}{4b^2}\right].
\]

At \(\Delta_T=(0.2,0)\ {\rm GeV}\),

\[
 F_\lambda^{(A)}=0.8824969025845955,
 \qquad
 F_\lambda^{(s)}F_\rho^{(s)}=0.9105103613800342.
\]

Both factors become one exactly at zero transfer. Summing active and spectator
flavor charges therefore closes proton, neutron, and deuteron charge without a
pointlike spectator identity.

## Scalar-family numerical diagnostic

At the existing AV18 nuclear grid and \(\Delta_T=(0.2,0)\ {\rm GeV}\), the
scalar-family spin-one current is

```text
[[ 0.557270249971,  0.001592566104, -0.038878558610],
 [-0.001592566104,  0.521161566948,  0.001592566104],
 [-0.038878558610, -0.001592566104,  0.557270249971]]
```

up to imaginary parts below \(5.2\times10^{-18}\). Forward deuteron charge
closes to \(1.1\times10^{-14}\), transfer reversal to
\(2.3\times10^{-16}\), parity to \(3.4\times10^{-16}\), and coherent
SS/SD/DS/DD reconstruction exactly. The separately accumulated active and
spectator mechanisms reconstruct within \(9.0\times10^{-14}\), the
5,376-node summation floor.

The angular-condition residual is

\[
 1.58142029548\times10^{-4},
\]

so resolving the spectator changes the current but does not restore
rotational covariance by construction.

## Four-direction response

M2i consumes the same eight independently recomputed endpoint states as M2h.
The spectator contribution is proportional to the active-space identity and
is parameter independent at this increment. Therefore the nonzero response is
exactly the M2h response multiplied by \(F_\lambda^{(A)}\). The scaled
pair-current singular values are

```text
1.576123555671e-3, 2.120915781035e-4,
1.249291625128e-12, 1.066950609342e-16
```

and the scaled spin-one singular values are

```text
3.097029627188e-4, 1.996799669294e-5,
8.040800812107e-11, 3.957274885306e-13.
```

Both diagnostic ranks remain two at the declared relative threshold. The
angular-condition derivative becomes

```text
( 4.30639066e-5, -4.03411303e-5,
 -7.44405971e-5,  8.98946356e-4 ).
```

The M2g rank-zero identity-current response and M2h active-current response
remain exact controls.

## Axial-family check

No scalar/axial mixture is selected. Running the axial family separately gives
the same pair-level rank and spin-one diagnostic rank two. Its baseline
angular residual is \(1.27026207988\times10^{-4}\), demonstrating that the
two spin/isospin recouplings remain distinguishable after the same spatial and
current embedding.

## Scientific boundary and next step

M2i closes the momentum-independent spectator gap for the represented
factorized S-wave member and completes its diagonal three-quark impulse
current. M2j has since mapped this member into the exact S3-antisymmetrized
898-state `qqq + qqqg` target with both C7/H2 gluonic singlets and the
maximal source-supported partial interaction/current. The next increment is
the emitter-tree transverse Jacobi recoupling, the two spectator-emitter C53
vertices, orthogonal-singlet interactions, and current blocks obtained by
gauging the same Hamiltonian and regulator. Scalar and axial families remain
separate until a physical Hamiltonian or fit determines their mixing.

The first-C117 finite-cell normalization derivation and the M3 topology-aware
encoding remain valid parallel work streams.

# M2n Hamiltonian-continuity current

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).
## Scope and probe

M2n implements the first current generated from the M2m Hamiltonian rather
than appended as an unrelated operator. The first executable gauge benchmark
is normalized baryon-number U(1): each quark carries probe charge \(1/3\)
and the gluon carries zero. This is common to proton and neutron and avoids
inventing unresolved flavor-dependent electromagnetic matching.

The implementation is
`src/deuteron_wigner/quantum/m2_hamiltonian_gauged_current.py`.

## Matched finite-regulator density

The density uses the same K9 active qg oscillator, global lambda ground, and
symmetry-adapted rho ground as M2m. For transfer \(\Delta_T\), an active-quark
hit carries

\[
 \delta q_\lambda^{(A)}=\sqrt{y_S/y_A}\,\Delta_T,
 \qquad
 \delta q_{qg}=\sqrt{x_g/x_q}\,\Delta_T,
\]

while either spectator hit carries

\[
 \delta q_\lambda^{(S)}=-\sqrt{y_A/y_S}\,\Delta_T,
 \qquad
 \delta q_\rho^{(S)}=\pm\sqrt{z_2/z_1}\,\Delta_T.
\]

The projected density is the one-third active contribution plus the two
one-third spectator contributions. At
\(\Delta_T=(0.2,0)\ {\rm GeV}\), the active-lambda and total-spectator ground
overlaps both equal

\[
 0.8824969025845955.
\]

The equality is a consequence of the S3-adapted widths. At zero transfer the
complete 266-dimensional density is the identity within
\(3\times10^{-14}\). The directed-transfer relation
\(\rho(-\Delta)=\rho(\Delta)^\dagger\) closes at
\(2.3\times10^{-16}\).

## Finite Hamiltonian continuity equation

For nonzero transverse transfer, M2n selects the longitudinal minimum-norm
solution

\[
 \Delta_x J_x(\Delta)+\Delta_yJ_y(\Delta)
 =H\rho(\Delta)-\rho(\Delta)H,
\]

or

\[
 J_i(\Delta)=\frac{\Delta_i}{\Delta_T^2}
 [H,\rho(\Delta)].
\]

This fixes only the longitudinal part. A divergence-free transverse current
cannot be inferred from continuity and remains open.

Splitting \(H=H_{\rm diag}+H_{C53}\) gives a fock-diagonal current and a
strictly qqq--qqqg current. At the named \(g_{C53}=0.05\) point and
\(\Delta_T=(0.2,0)\ {\rm GeV}\), the latter has
maximum matrix element

\[
 \max|J_{C53,x}|=0.31026505994186515\ {\rm GeV}.
\]

The fock-diagonal contribution has zero number-changing block, the C53
contribution has zero fock-diagonal block, and their sum closes continuity at
\(2.8\times10^{-17}\ {\rm GeV}^2\). Current reversal obeys
\(J_i(-\Delta)=J_i(\Delta)^\dagger\) within
\(4.5\times10^{-15}\ {\rm GeV}\).

## Controlled limits and nontrivial current checks

At fixed regulator and density, \(H_{C53}=g_{C53}V_{C53}\) implies

\[
 J_{C53,i}=g_{C53}\frac{\Delta_i}{\Delta_T^2}[V_{C53},\rho].
\]

The explicit zero-C53 limit therefore has exactly zero number-changing
current. The general constructor now accepts this limit and arbitrarily
small couplings while enforcing continuity and the same block decomposition.
The original unconditional lower bound of `1e-12 GeV` incorrectly rejected
these cases. The nonzero-current requirement remains a named-point test.

The 14 current tests include `g_C53=0` and `g_C53=+/-1e-14` regressions.
Small-coupling blocks are divided by the coupling before comparison, which
detects incorrect numerical zeroing. The oblique transfer `(0.12,0.16) GeV`
and its reversal exercise both current components on a seeded complex state.
An independent spectral check verifies

\[
 \langle m|J_i|n\rangle
 =\frac{\Delta_i}{\Delta_T^2}(E_m-E_n)\langle m|\rho|n\rangle.
\]

Here the energies are eigenvalues of the represented mass-squared matrix.
Nonzero transition elements are checked explicitly; diagonal eigenstate
expectations vanish by this identity and cannot validate the full interface.
These are compact SciPy checks. PennyLane parity remains the next action.

## Boundary

This is an exact Abelianized finite-basis benchmark for the represented M2m
Hamiltonian. It is not a complete electromagnetic current or a full QCD
Slavnov--Taylor closure. Flavor-dependent electromagnetic structure,
divergence-free transverse pieces, instantaneous and pair currents,
constrained, induced, zero-mode, and counterterm currents remain separate
open terms.

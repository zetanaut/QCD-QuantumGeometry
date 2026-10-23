# M2m source-supported total-sign H2 Hamiltonian

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).
## Scope

M2m is the first Hamiltonian represented directly on the M2l
color--spin--flavor sign carrier. It uses the symmetry-adapted global Jacobi
ground

\[
 b_\rho=\sqrt{\frac23}\,b_\lambda
 =0.32659863237109044\ {\rm GeV},\qquad b_\lambda=0.4\ {\rm GeV},
\]

and the named nonphysical K9 parameter point. It is exact for the represented
H0, C396, first-C117, and C53 source-supported terms. It is not the complete
physical H2 Hamiltonian.

The implementation is
`src/deuteron_wigner/quantum/m2_total_sign_h2_hamiltonian.py`.

## Corrected global-ground mass embedding

The active q/qg cluster has \(y_A=1/3\); the two-quark spectator cluster has
\(y_S=2/3\) and internal shares \(z_1=z_2=1/2\). With the normalized Jacobi
variables,

\[
 M_S^2=\frac{m_q^2}{z_1}+\frac{m_q^2}{z_2}+b_\rho^2,
 \qquad
 M_N^2=\frac{M_A^2}{y_A}+\frac{M_S^2}{y_S}+b_\lambda^2.
\]

At \(m_q^2=0.20\ {\rm GeV}^2\),

\[
 M_S^2=0.9066666666666667\ {\rm GeV}^2,
 \qquad
 M_N^2=3M_A^2+1.52\ {\rm GeV}^2.
\]

The shift differs from M2i's \(1.60\ {\rm GeV}^2\) because M2m uses the
separately identified S3-symmetric width rather than silently reusing the
local equal-width regulator.

## Why one C53 ray per mode is insufficient

C53 reaches 80 external mode labels
\(\alpha=(k_q,k_g,n,m,\lambda_g)\). At fixed target isospin projection and
quark spin projection, the total-sign carrier has rank

\[
 r(2S_q^z)=
 \begin{cases}
 1,&|2S_q^z|=3,\\
 2,&|2S_q^z|=1.
 \end{cases}
\]

Across both nucleon-helicity sectors this gives 56 rank-one and 104 rank-two
mode carriers. Retaining only the normalized C53 ray would give dimension
162 and omit 104 allowed directions. M2m instead retains the complete
fixed-weight carrier:

\[
 \dim\mathcal H_{\rm M2m}=2+264=266,
 \qquad
 \dim\mathcal H_{J^z=\pm1/2}=133.
\]

The C53 source projects into this carrier with residual
\(1.7\times10^{-15}\). Its norm squared is
\(90.5487172721218\) in each proton/neutron helicity sector.

## Represented diagonal action and C53 block

In the K9 source, H0, both C396 directions, and represented C117 are exactly
scalar in the final active-quark helicity. Their external-mode operator has
zero leakage between the 80 total-\(J^z\)-admissible C53 modes and the other
144 K9 active-subsystem modes. On carrier bases \(U_{\alpha H}\), the qqqg
matrix is

\[
 \langle\alpha,H,a|H_{\rm diag}|\beta,H',b\rangle
 =O_{\alpha\beta}\,
 \langle U_{\alpha H,a}|U_{\beta H',b}\rangle.
\]

The C53 transition is inserted once,

\[
 \langle\alpha,H,a|H_{C53}|N,H\rangle
 =\frac{g_{C53}}{y_A}
 \langle U_{\alpha H,a}|\Phi_{\alpha N,H}\rangle,
\]

with no independent coupling for the second bare-color coordinate.
Hermiticity closes below \(10^{-30}\), and the total-\(J^z\) commutator is
zero at stored precision.

## Named-point spectrum

For the exploratory coupling \(g_{C53}=0.05\), the proton and neutron spectra
are isospin partners. The lowest pair is

\[
 M_{\rm eig}^2=1.789591467572781\ {\rm GeV}^2,
 \qquad P_{qqq}=0.943333437129513,
 \qquad P_{qqqg}=0.056666562870487.
\]

The two helicities are degenerate within \(7\times10^{-16}\ {\rm GeV}^2\),
and the next state in either block is more than
\(1.55\ {\rm GeV}^2\) higher. In the zero-C53 limit the lowest states are
pure qqq states at \(2.1037591222516006\ {\rm GeV}^2\). These are regulator
diagnostics, not a fitted nucleon mass.

## Boundary

M2m does not derive the full spectator/all-emitter C117 interaction,
instantaneous or constrained-field terms, induced interactions, zero modes,
counterterms, or spin-dependent symmetry breaking. Those absences are not
zeros. The next object is a current generated from this identical finite
Hamiltonian and regulator.

# M2l global-ground spin--flavor S3 completion

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

## Scope

M2l closes the missing three-quark spin--flavor permutation carrier for the
smallest lawful continuation of M2k. It chooses the symmetry-adapted Jacobi
ground

\[
 b_\rho=\sqrt{\frac23}\,b_\lambda
 =0.32659863237109044\ {\rm GeV},
 \qquad b_\lambda=0.4\ {\rm GeV},
\]

for which the orbital state is an exact S3 scalar. This is an identified
finite-K exploratory regulator member, not a reinterpretation of the M2i
equal-local-width member and not a physical nucleon fit.

The implementation is
`src/deuteron_wigner/quantum/m2_spin_flavor_s3_completion.py`.

## Exact spin--flavor carrier

Use the ordered one-quark basis

\[
 \{|u,-\rangle,|u,+\rangle,|d,-\rangle,|d,+\rangle\}
\]

and its 64-dimensional three-quark tensor cube. Slot permutations act by

\[
 R_{sf}(\pi)|a_1a_2a_3\rangle
 =|a_{\pi^{-1}(1)}a_{\pi^{-1}(2)}a_{\pi^{-1}(3)}\rangle
\]

in the code's frozen tensor-axis convention. The representation law closes
exactly. The symmetric projector is

\[
 P_{[3]}=\frac1{6}\sum_{\pi\in S_3}R_{sf}(\pi).
\]

For each fixed proton/neutron isospin projection and helicity, M2l also
projects with

\[
 P_{S=1/2}=\frac{15/4-\mathbf S^2}{3},\qquad
 P_{I=1/2}=\frac{15/4-\mathbf I^2}{3}.
\]

The common image of these projectors is one-dimensional. It gives four
orthonormal states

\[
 |p,\!\uparrow\rangle,\ |p,\!\downarrow\rangle,
 |n,\!\uparrow\rangle,\ |n,\!\downarrow\rangle
\]

with normalization, permutation, spin, and isospin residuals below
\(4\times10^{-16}\).

This is the two-flavor SU(4) restriction of the familiar SU(6)-symmetric
quark-model ground-state construction. It is a controlled symmetry limit;
spin-dependent dynamics may break that limit while preserving total Fermi
statistics.

## Scalar and axial spectator relation

For the spectator pair attached to any chosen active slot, define the exact
pair-spin and pair-isospin projectors. The symmetric nucleon state obeys

\[
 \langle P_{S_{12}=0}P_{I_{12}=0}\rangle=\frac12,
 \qquad
 \langle P_{S_{12}=1}P_{I_{12}=1}\rangle=\frac12,
\]

and the mixed \((S,I)=(0,1),(1,0)\) probabilities vanish. Thus the earlier
M2 scalar and axial families are the two orthogonal pair-coupling views of
one symmetric ground state, with equal coherent weights in this particular
common-orbital symmetry limit. This result does not select a physical
scalar/axial probability after hyperfine, mass, or orbital symmetry breaking.

## Total antisymmetrizer in qqqg

Let \(D_c^{(8)}(\pi)\) be the exact two-dimensional C7 qqqg color-singlet
representation. Because the global orbital ground is symmetric, the full
quark antisymmetrizer is

\[
 \mathcal A_{qqqg}
 =\frac1{6}\sum_{\pi\in S_3}{\rm sgn}(\pi)
 D_c^{(8)}(\pi)\otimes R_{sf}(\pi).
\]

It is Hermitian and idempotent, with rank 20 in the
\(2\times64\)-dimensional color-times-spin--flavor space. The rank is not a
new list of twenty couplings; it is the complete sign carrier before fixing
flavor, total spin projection, and a source operator.

## Canonical C53 source

For each emitter-local C53 qg mode label
\(\alpha=(k_q,k_g,n,m,\lambda_g)\), let \(K_\alpha\) be the exact 2-by-2
quark-helicity kernel after removing only the known
\(\sqrt{C_F}=\sqrt{4/3}\) color norm. Let

\[
 c_i^A=\langle C^A_{qqqg}|t_i^a|C_{qqq}\rangle
\]

be the two C7 color coordinates for emission from quark slot \(i\). The
full source image is

\[
 |\Phi_{\alpha N}\rangle
 =\sum_{i=1}^3 c_i\otimes K_\alpha^{(i)}|N\rangle.
\]

The local qg mode is carried covariantly with its emitter; the M2k global
ground condition supplies the unit-overlap three-quark tree recoupling. M2l
does not identify unrelated qg modes across different \(\alpha\).

The exact color-neutrality identity

\[
 \sum_i c_i=0
\]

removes the spin-scalar part of every kernel. Writing

\[
 K_\alpha=\frac12\operatorname{tr}K_\alpha\,\mathbf1
 +\widetilde K_\alpha,
\]

gives

\[
 |\Phi_{\alpha N}\rangle
 =\sum_i c_i\otimes\widetilde K_\alpha^{(i)}|N\rangle.
\]

For the source-owned K9 C53 kernel there are 104 nonzero colorless matrix
entries grouped into 80 nonzero spin kernels. All 80 source channels survive
for every proton/neutron helicity state. The maximum scalar-cancellation and
traceless-reconstruction residuals are \(1.6\times10^{-15}\).

Every surviving source vector satisfies

\[
 \mathcal A_{qqqg}|\Phi_{\alpha N}\rangle
 =|\Phi_{\alpha N}\rangle
\]

with residual below \(1.7\times10^{-15}\), remains in the exact
\(I=1/2\) sector, and closes total \(J^z\) exactly after the qg orbital and
gluon-helicity labels are included.

## Resolution of the second-color-coordinate question

For any nonzero source channel, trace spin--flavor from the normalized source
state. M2l finds

\[
 \rho_c
 =\operatorname{Tr}_{sf}|\widehat\Phi\rangle
  \langle\widehat\Phi|
 =\frac12\mathbf1_2
\]

with residual below \(10^{-15}\). Consequently, in the M2j color basis

\[
 \langle u_A|\rho_c|u_A\rangle
 =\langle u_\perp|\rho_c|u_\perp\rangle
 =\frac12.
\]

This does not contradict M2k. M2k proved that a color-only covariant seed has
one orbit direction and does not license a second coupling. M2l shows that,
after the required mixed spin--flavor carrier is included, both bare color
coordinates occur as a maximally entangled pair. The coordinate formerly
called the orthogonal singlet is therefore populated, but it is not an
independent vertex and cannot receive an independently chosen coefficient.

## Claim boundary

M2l establishes:

- the exact global-ground spin--flavor S3 action;
- unique symmetric proton/neutron \(S=I=1/2\) source states;
- the derived equal scalar/axial pair weights in this symmetry limit;
- the complete qqqg color-times-spin--flavor sign projector; and
- the total-antisymmetric source image of the canonical C53 vertex.

It does not yet establish:

- a complete H2 Hamiltonian on the rank-20 qqqg sign carrier;
- arbitrary four-body qqqg Jacobi recoupling outside the source image;
- spin-dependent SU(4)-breaking interactions;
- the Hamiltonian-gauged number-changing or interaction current;
- constrained, instantaneous, zero-mode, or counterterm blocks;
- a physical nucleon or deuteron state; or
- a production observable.

## Next exact construction

Represent the source-supported H2 Hamiltonian on this total-sign carrier,
with the qqq and qqqg diagonal operators symmetrized over identical quarks and
the canonical C53 block inserted once. Then gauge that identical finite
Hamiltonian and regulator so that the q-to-qg and qg-to-q current is derived
from the Hamiltonian rather than appended independently.

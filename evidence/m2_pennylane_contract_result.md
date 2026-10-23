# M2o PennyLane implementation contract

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).
## Result

M2o freezes the mathematical inputs needed to implement M2m and M2n in
PennyLane without changing the physics or reusing the old C144 fixture as if
it were the new Hamiltonian.

The implementation contract is
`src/deuteron_wigner/quantum/m2_pennylane_contract.py`.

## Encoding

The full compact order is preserved exactly:

1. the qqq \(J^z=-1/2,+1/2\) states;
2. the qqqg total-sign carrier states in deterministic external-mode and
   carrier order.

The direct big-endian computational encoding requires

\[
 d_{\rm compact}=266,
 \qquad d_{\rm padded}=512,
 \qquad n_{\rm qubits}=9.
\]

For Hamiltonian-only work, either exact \(J^z\) block has

\[
 d_{J^z}=133,
 \qquad d_{\rm padded}=256,
 \qquad n_{\rm qubits}=8.
\]

The full nine-qubit encoding is retained for finite-transfer operators that
connect the two helicity blocks.

## Padding penalty

Plain zero padding is unsafe for variational minimization because the named
physical-subspace eigenvalues are positive; a variational state could fall
into a zero-energy padding state. M2o computes Gershgorin bounds

\[
 -9.595600232954302\le\lambda(H)\le13.803118477457504
 \quad({\rm GeV}^2)
\]

and assigns the padding projector the conservative penalty

\[
 \Lambda_{\rm pad}=37.20183718786931\ {\rm GeV}^2.
\]

This lies \(23.398718710411806\ {\rm GeV}^2\) above the certified upper bound,
commutes with the physical projector, and leaves every compact matrix element
and physical eigenpair unchanged.

## Hermitian observable interface

A directed finite-transfer operator is not Hermitian at fixed \(\Delta\), but
M2n proves \(O(-\Delta)=O(\Delta)^\dagger\). M2o exports

\[
 O_c=\frac{O(\Delta)+O(-\Delta)}2,
 \qquad
 O_s=\frac{O(\Delta)-O(-\Delta)}{2i},
 \qquad O(\Delta)=O_c+iO_s.
\]

The density and both current components therefore produce six Hermitian
sparse observables. Hermiticity and directed-operator reconstruction close at
\(4.5\times10^{-15}\). Compact and padded actions agree below
\(3\times10^{-17}\), exact ground-state padding leakage is zero, and the
encoded eigenstate residual is below \(2.2\times10^{-15}\ {\rm GeV}^2\).

## PennyLane launch contract

The first implementation should use:

- `lightning.qubit` with analytic shots;
- `qml.SparseHamiltonian` for the penalized Hamiltonian and Hermitian
  observables;
- `qml.StatePrep` only as an exact oracle route;
- compact SciPy states, actions, expectations, and transitions as the
  independent comparison;
- the eight-qubit block encoding only for a separately selected variational
  eigensolver.

A generic full Pauli decomposition is unnecessary. No ansatz, optimizer,
hardware target, shot budget, physical fit, or activation has been selected.
Those are implementation or executive choices after the exact sparse oracle
is reproduced.

Acceptance must exercise all six quadratures on seeded complex non-eigenstate
probes and include nonzero transition matrix elements. For this current,
`<m|J_i|n> = Delta_i/Delta_T^2 (E_m-E_n) <m|rho|n>`, so exact eigenstate
diagonal expectations vanish. The original `(0.2,0) GeV` transfer also gives
`J_y=0`. Include the oblique `(0.12,0.16) GeV` transfer and its reversal, at
the same `Q^2=0.04 GeV^2`, and compare both transverse actions with compact
SciPy. Check state round trips, padding isolation, and Hamiltonian action in
addition to observable expectations. The M2n zero/small-C53 and oblique-current
regressions now pass in SciPy; no new PennyLane execution is claimed here.

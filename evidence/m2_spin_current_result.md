# Spin-resolved good currents and matched seven-qubit interference

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-06. **Exploratory finite canonical model**, not calibrated
nucleon form factors, magnetic moments, or a complete physical four-current.
Both accepted direction choices remain resolved: canonical global-Jacobi
Galerkin authority, with matched H2 before later explicit enlargement.

## Canonical spin and parity conventions

The local quark J+/(2P+) at Delta+=0 now acts between both Jz=+/-1/2
fibers. Its raw helicity-conserving quark action is projected with the
actual outgoing and incoming longitudinal/orbital/color/spin-flavor
isometries; array dimensions alone never define that identification.
The previous same-helicity API delegates to this more general action.

Transverse parity Px reflects x, leaves y and light-front time unchanged,
and reverses Jz. The canonical quark spinor obeys

    S(Px) = -i Sigma_x gamma0,
    S(Px) u_h(p) = -i u_-h(Px p).

Three quarks supply phase i; a transverse gluon flips helicity without an
additional phase in this convention. Circular oscillator occupations swap
their +/- halves with (-1)^N. The resulting symmetry-basis map is unitary,
squares to -I on this odd-fermion carrier, and intertwines H0, emission,
cross-pair and instantaneous-fermion matrices separately.

For an explicitly supplied positive intrinsic parity, the partner is
v_down=i Px v_up. It is not an independently phase-ambiguous eigenvector
and does not automatically select a production nucleon state. The
independent spinor calculation and current reversal tests fix the signs.
The general light-front discrete-symmetry and spin-flip framework is
described by [Brodsky, Gardner and Hwang (2006)](https://arxiv.org/abs/hep-ph/0601037);
their light-front normalization is not silently imported into ours.

## Two-spin current, effective norm, and form-factor diagnostic

Using symmetric-frame external spinors with transverse momenta +/-Delta/2,
mass M>0 and P+>0 explicitly supplied, the two independent normalized
Dirac/Pauli structures in helicity order (+,-) are

    J+/(2P+) = F1 I + F2 K + residual,
    K = [[0, -Delta_x+i Delta_y],
         [Delta_x+i Delta_y, 0]]/(2M).

A complex least-squares decomposition retains any incompatible residual.
Zero transfer is rejected for Pauli extraction: F2(0) requires a limit.
The examples use sqrt(the finite-model eigenvalue) as M, not a physical
nucleon mass. Agreement with two structures at this kinematic slice does
not establish full Lorentz covariance, transverse-current or zero-mode closure.

For actual qqqgg and qqqq-qbar reference sectors, their raw local currents
now connect the two spin fibers with all source-reachable coordinates
retained. The cross-parent rectangular operator is

    O_eff,fi = Omega_f(Ef)† O_parent,fi Omega_i(Ei),
    O_Euclidean,fi = N_f(Ef)^-1/2 O_eff,fi N_i(Ei)^-1/2,
    w_i = N_i(Ei)^1/2 v_i.

Sector labels, not block positions, identify the excluded spaces.
Two-energy reversal and full reconstructed-parent matrix elements are
tested. Zero-transfer baryon charge is exactly I in the whitened register.
At the self-consistent energy, w is an eigenstate of the energy-local
Hermitian reduction. Using unnormalized retained amplitudes or leaving the
current untransported would invalidate these statements.

## Numerical result

All rows use K2=9, b=.4 GeV, mq=.35 GeV, mg=0, gs=.3 and
Delta=(.12,.16) GeV. N_exc is the retained oscillator cutoff; matched
rows use both actual free-Q sectors at Q cutoff 1.

| Model | Target | Retained states per helicity | F1 | F2 |
|---|---|---:|---:|---:|
| Bare N_exc=0 | proton | 37 | .882387443412 | zero |
| Bare N_exc=0 | neutron | 37 | 3.08201858e-8 | zero |
| Bare N_exc=1 | proton | 204 | .882078010265 | .001385080788 |
| Bare N_exc=1 | neutron | 204 | 9.83885951e-8 | -.0009260704033 |
| Matched N_exc=0 | proton | 37 | .882386814623 | 4.6386150755e-7 |
| Matched N_exc=0 | neutron | 37 | 4.56910183e-10 | -3.0937926284e-7 |

The matched M² is 1.421430998167897 GeV², versus 1.419475028604878
GeV² for the larger bare retained space. These rows represent different
finite truncations, not an uncertainty band or a physical convergence test.
The induced spin-flip signal is resolved even though the bare 37-state
spin-flip current matrix is exactly zero. The two-structure residual is
below 9e-16 for the bare examples and 4e-16 for the matched examples.
Transfer reversal and equal-magnitude rotations agree within roundoff in
the bare tests. The matched tests additionally exercise independent
incoming/outgoing energies and full-parent reconstruction.

## Executed quantum measurement

Combine the two normalized helicity states into

    |chi>=(|up> + (Delta_y-i Delta_x)/|Delta| |down>)/sqrt(2),
    C=(rho+rho†)/2, S=(rho-rho†)/(2i).

Then <C>=F1 and <S>=|Delta| F2/(2M) for the verified two-structure
matrix. Actual PennyLane lightning.qubit StatePrep/SparseHamiltonian
circuits agree with independent compact matrix expectations. The 408-state
bare pair uses nine qubits; the 74-state matched pair uses seven qubits,
versus nine for its two 165-state reference parents.

For matched proton/neutron the measured sine quadratures are
3.89067938657e-8 and -2.59494590734e-8. Agreement with direct expectations
is within 3e-21 for these rows; padding probability is zero. This is an
exact-state oracle, not variational preparation, a scalable ansatz,
hardware feasibility, or quantum advantage.

## Verification, reproduction and continuation

    python scripts/benchmark_m2_spin_current.py --quantum
    python scripts/benchmark_m2_spin_current.py --matched --quantum
    python -m pytest tests/test_m2_spin_resolved_current.py tests/test_m2_matched_spin_current.py -q

36 new direct spin tests are implemented. The focused integration with
the existing good current, excluded matching and reference transport
passes 91 tests in 144.10 seconds. A preceding named M2+Q run, before the
matched-spin extension, passed 687 tests; it is not presented as coverage
of that later extension.

Next: evaluate variational state residuals and current errors, not only
energy, while completing common-regulator higher-order exit/QQ and
physical-current partners. Physical finite calibration remains unselected.
Nuclear two-body matching, the spin-one current and GTMD/Wilson-line/soft
completion, and topology-factorized solver scaling remain open.

Original source files are recoverable from
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_spin_current_source.tar`
and `pre_spin_matching_source.tar` in the same directory.

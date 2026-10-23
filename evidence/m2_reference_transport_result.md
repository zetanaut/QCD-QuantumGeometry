# Finite-reference interaction, norm and current transport

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-06. Scope: **exact matching between explicitly supplied finite
parents**, not continuum QCD renormalization or a physical calibration.
This follows the accepted canonical Galerkin direction and the calculated
Wick/normal-order subtraction; no regulator choice is reopened.

## New mathematical construction

Two parents, labelled L and R, share the same retained physical space but
may have different excluded spaces. A mandatory unitary T maps the retained
coordinates of L into those of R. Equal array dimensions do not imply T=I;
the actual benchmark uses I because both sources bind the same pilot object.
With their separately derived effective Hamiltonians and wave operators:

    Delta H(E) = T† H_eff,R(E) T - H_eff,L(E),
    Delta N(E) = - d Delta H(E)/dE,
    Delta O(Ef,Ei) = T† Omega_R(Ef)† O_R Omega_R(Ei) T
                     - Omega_L(Ef)† O_L Omega_L(Ei).

Both parent operators are supplied explicitly. Correcting H, N and O
together reproduces the reference matrix family exactly, including complex
transition operators. At zero transfer, a conserved charge q obeys
Delta rho_q(E,E)=q Delta N(E), not zero. This is a finite-reference
renormalization trajectory; the reference is not thereby made physical.

The two independent energy derivatives of Delta O use

    Omega'(E) = -Q(E-H_QQ)^-2 H_QP.

They act separately on its bra and ket wave operators. Reversal interchanges
the endpoints and takes the adjoint. These derivatives are not replaced by
one fitted scalar current renormalization.

For an energy-local approximation at E0, write C0=Delta H(E0),
C1=Delta H'(E0). The lower parent obeys the generalized eigenproblem

    A psi = E B psi,
    A = H_L + P(C0-E0 C1)P†,
    B = I - P C1 P†.

Its Schur complement is precisely
H_eff,L(E)+C0+(E-E0)C1-E. The implementation rejects a nonpositive B,
rather than clipping the metric. A conservative spectral-norm remainder
bound is the sum of the two parents' pole-distance Taylor bounds.
An approximate generalized-parent current is **not** invented by this
Hamiltonian routine; exact two-energy operator transport remains separately
available, and exact identities are not claimed for independently truncated
Hamiltonian and current approximations.

The matched mass-flow current retains its off-shell endpoint terms:

    Omega_f†[H,rho]Omega_i =
      (Ef-Ei) rho_eff
      +(H_eff(Ef)-Ef) P†rho Omega_i
      -Omega_f†rho P (H_eff(Ei)-Ei).

Independent tests check the full expression before and after transport.
It must not be replaced off shell by a commutator of two retained matrices.

## Actual excluded-Fock experiment

The common 37-state proton pilot is K2=9, N_exc=0, b=.4 GeV, mq=.35
GeV, mg=0, gs=.3. The lower/reference parents use Q oscillator cutoffs
0/1 and dimensions 92/165, with actual coherent qqqgg and qqqq-qbar
cubic sources and free Q dynamics. Their ground M² values are
1.421431890118926 and 1.421430998167897 GeV².

At E0=1.4 GeV²:

| Diagnostic | Value |
|---|---:|
| Frobenius norm Delta H | .00471476964285 GeV² |
| Frobenius norm Delta N | .00213448568576 |
| Constant-only matching energy error | 9.3066157e-9 GeV² |
| Energy-local matching with metric error | 9.8670627e-11 GeV² |
| Operator Taylor remainder at reference energy | 1.4098747e-7 GeV² |
| Conservative remainder bound | 6.2249595e-7 GeV² |

Moving E0 to 1.42 GeV² reduces the generalized energy error to
4.50e-13 GeV². This is a controlled local approximation to the finite
reference, not evidence of an exact energy-independent all-states operator.

For this nested free-Q enlargement, Delta H is negative semidefinite below
the Q poles and Delta N is positive semidefinite. At the reference ground
energy their nonzero extreme eigenvalues are -.00141572683916 GeV²
and .000652248113955. Missing norm and current pieces are small but resolved:
using the reference-normalized retained state with the lower current gives
charge .999999561082792. Its required correction is 4.38917208592e-7;
the finite-transfer electromagnetic correction is 2.86271027977e-7 at
(.12,.16) GeV.

Even a generous six-direction span (identity, quark/gluon kinetic mass
directions, cubic emission, combined cross-pair contact, instantaneous-
fermion contact) leaves a .00174393835980 GeV² Frobenius residual in
Delta H at E0=1.4. This is an operator representation test, not a physical
fit. Those six structures cannot silently replace the complete induced
matrix or current family.

## Reproduction, scope and next target

    python -m pytest tests/test_m2_reference_transport.py -q
    python scripts/benchmark_m2_reference_transport.py

19 new direct tests pass in 20.54 seconds, including complex-coordinate
maps, metric positivity, independent energy derivatives, remainder bounds,
actual Fock-sector transport, charge failure control, and the off-shell
current identity. Neither old pilot/oracle matrices nor the original
formal volumes were changed.

The next current target is the complete retained spin-resolved good-current
matrix between both nucleon helicities, with an explicit parity/phase map
and independent Dirac/Pauli diagnostic. In parallel the larger reference
must acquire higher-order instantaneous exit/QQ and matched-current partners.
The finite free-Q example is not the continuum, and calibrating it to
itself does not supply physical finite renormalization data. Endpoint/
zero-mode completion, physical matching, nuclear/GTMD construction and
topology-factorized quantum scaling remain open. All three workstreams
continue; no physical activation, push or publication is included.

The pre-edit note/roadmap backup from c75929d8 is
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_reference_transport_notes.tar`.

# Canonical instantaneous qqq to qqqgg exits

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-06. Scope: **both normal-ordered canonical order-gs²
contacts for this transition**, not the complete higher-order Hamiltonian.
The sea-pair channel, QQ interactions, six-body exits, common-regulator
loop completion and physical finite matching are still open.

## Source and normalization

The fourth and fifth interaction terms in
[Srivastava--Brodsky Eq.24](https://arxiv.org/html/hep-ph/0011372v2)
give the instantaneous-fermion (IF) and mixed quark/adjoint-current (IG)
pair contacts. No propagating cubic denominator replaces their constrained
field inverse. Normal ordering precedes external Galerkin projection.

Let p be the incoming doubled quark mode, k1,k2 the outgoing doubled
gluon modes, and p'=p-k1-k2>0. Fermions have odd modes, gluons even.
With outgoing gluon colors a,b and quark helicities h',h, define

    S(h',h;a,b) = ubar_h' gamma.eps_a* gamma+ gamma.eps_b* u_h
                  /sqrt((2p_q'+)(2p_q+)),
    C_12 = U5† sum_ab [(T^a T^b)_q U3] tensor |a,b>,
    C_21 = U5† sum_ab [(T^b T^a)_q U3] tensor |a,b>.

C_12 and C_21 are six-component qqqgg singlet vectors from the unique qqq singlet;
the complete color tensor is independently reconstructed. The spin factor
is independent of mass/momenta, vanishes for h' != h and same-helicity
gluons, and is -2 for the allowed ordered opposite-helicity assignment
in the canonical polarization convention.

Both distinguished outgoing-gluon assignments are included:

    V_IF/g² = K2/[2 pi sqrt(k1 k2)] *
              [ S_12 C_12/(p-k2) + S_21 C_21/(p-k1) ].

The constrained denominators are positive odd integers; neither is a
propagating energy denominator. The original -g²/2 gamma+ gamma.A
expression, four external field norms and finite cell ell=pi K2/P+
independently reproduce this coefficient for different masses and P+.

For the mixed current, j_g=f A_perp d_- A_perp has the pair coefficient
-i f^(c a b)(k1-k2) eps_a* dot eps_b*. The two current orderings and their
opposite Fourier transfers give +g²/(k1+k2)². Consequently

    V_IG/g² = -K2/[pi sqrt(k1 k2)] * (k1-k2)/(k1+k2)²
              * (eps_1* dot eps_2*) delta_h'h (C_12-C_21).

This uses -i f^(c a b)T^c=-(T^aT^b-T^bT^a), checked against explicit
structure constants. The pair transfer is positive, so no Q0 inverse is
needed here. IG vanishes at k1=k2 and in the Abelian color-commutator
limit. Its full field normalization and Bose exchange are independently
tested. IF and IG must be combined as amplitudes, not as probabilities.

## Jacobi measure and particle statistics

The outgoing five-body frame maps orthogonally into the two inherited
three-quark Jacobi coordinates plus two local splitting coordinates.
The normalized measure is sqrt(x_q x_g1 x_g2/X_parent). Each new m=0
canonical oscillator contributes b/sqrt(pi) for any radial n.
The two constant-kernel integrals therefore supply b²/pi, with inherited
oscillator labels preserved. This is not a single two-body pair contact.

The distinguished two-gluon amplitude is divided by sqrt(2!) on the
normalized ordered-slot Fock wavefunction. The action keeps all six color
singlets, exact Iz/Jz, three-quark antisymmetry and two-gluon Bose symmetry.
The independent tests include the enlarged 204-state retained shell.

## Resolved numerical result and matching order

At the shared proton K2=9, retained N_exc=0, b=.4 GeV, mq=.35 GeV,
mg=0, gs=.3 example, the source maps 10 qqq columns into 630/4950
packed qqqgg rows for Q cutoffs0/1. With this incoming ground shell the
new relative modes are m=0, so the source has identical norm at those two
cutoffs; this is a selection rule, not UV convergence.

| Source diagnostic per gs² | Value (GeV²) |
|---|---:|
| IF Frobenius norm | .0107972950701 |
| IG Frobenius norm | .00166764779287 |
| Coherent total norm | .0110769981938 |

The interference 2 Re<V_IF,V_IG> is 3.33725899327e-6 GeV^4.
At E=1.4 GeV², the combined free-Q return coefficient
||V2† R0 V2||F is 2.22045747860e-5 GeV²; multiplying by gs^4 gives
1.79857055767e-7 GeV². The cross block with the existing coherent cubic
source has ||V2† R0 V1||F=.000345053816445 GeV², or
9.31645304401e-6 GeV² after gs^3. These are operator norms, not masses.

The coupling expansion is explicit:

    (g V1+g² V2)† R0 (g V1+g² V2)
      = g² V1†R0V1 + g³(V1†R0V2+V2†R0V1) + g^4 V2†R0V2.

V1 acts on retained qqqg columns; V2 on qqq columns. The cross term is
therefore a new qqq/qqqg effective transition. It must not be relabeled
as a cubic gs² correction or dropped when sources are combined.
Energy derivatives and current insertions must be transported with the
same coherent polynomial. The old cubic-only reference and published
quantum oracles are unchanged by this standalone source increment.

## Verification and next action

    python -m pytest tests/test_m2_instantaneous_pair_exit.py -q
    python scripts/benchmark_m2_instantaneous_pair.py

20 direct tests pass in 4.49 seconds. Next derive the canonical sea-pair
exit, combine the order-resolved sources with the existing free-Q
reference without losing cross terms, then transport norm and currents.
QQ interactions, six-body exits and full regulator/physical matching
remain necessary; no complete order-g^4 QCD claim is made.

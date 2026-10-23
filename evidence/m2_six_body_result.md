# Six-particle cubic paths and coherent H2 matching

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Scope: canonical finite-model operator development, not calibrated QCD.
The H2-first and corrected global-Jacobi authority choices are unchanged.
The first sections preserve the cubic-only milestone at26f71d0f.
The subsequent direct-quartic and combined-H2 continuation below
supersedes its missing-W6P launch target.

## Complete representation and actual cubic action

The explicit sectors are qqqggg and qqqq-qbar-g. SU(3) Dynkin-label
tensor-product recursion gives22 and8 singlets independently of the
numerical source span. The explicit orthonormal bases span all these
singlets and satisfy all eight total-generator constraints. Full raw
color dimensions are13824 and1944.

Six positive doubled modes are odd for fermions and even for gluons.
The sector minima areK2=9 and7. AtK2=9 there are1 and6 longitudinal
tuples; at11 there are6 and21. Five global relative Jacobi coordinates
diagonalize the same normalized mass metric. At N_exc<=1,

    H6^0 = b²(5+N_exc) + sum_i m_i²/x_i.

Packed coordinates preserve exact Iz/Jz, full color and spin/flavor
multiplicities, quark antisymmetry and three-gluon Bose symmetry.
Their local same-Jz electromagnetic/net-flavor good density uses the
same individual-parton recoil and antiquark signs as the earlier sectors.
This is not a full four-current.

The actual canonical cubic U65 contains:

- qqqgg -> qqqggg: quark emission and three-gluon splitting;
- qqqgg -> qqqq-qbar-g: conversion of one gluon into a quark pair;
- qqqq-qbar -> qqqq-qbar-g: all four quark emissions and antiquark emission.

Each split retains the four incoming Jacobi coordinates and adds the
fifth. Finite-cell normalization and sqrt(xa xb/(xa+xb)) measure are
unchanged. Antiquark color is -T^T; its stripped spinor numerator is
-vbar_in gamma.eps* v_out. Independent charge-conjugation/CAR calculations
check the reversed vector endpoints and absence of an extra sign.

With a fixed old-gluon annihilator label, normalized factors are
sqrt(3) S3 for splitting two gluons into three and for appending a
quark-emitted gluon to two existing gluons; pair conversion is
2 sqrt(2) A4. Independent occupation-space ladder operators verify
the sqrt(3), 1/2 vertex and sqrt(2) annihilation factors; the four-quark
creation factor is inherited from the independently checked CAR action.
Do not sum both old-gluon labels again after using those factors.

The sparse distinguished kernels are sandwiched by complete statistics
projectors. The reverse uses precisely their adjoints, on the same
packed coordinates. Both source sectors reach sea+g coherently.

## Coherent excursion and the required five-body return

Write V=H5P/g, U=H65/g, W6=H6P/g² and Rj=(E-Hj^0)^-1.
Then

    A1 = R5 V,
    F6 = W6 + U R5 V,
    A6 = R6 F6,                 actual six-body wave g² A6,
    B5 = R5 U† R6 F6,           actual five-body backflow g³ B5,
    Sigma6[4] = F6† R6 F6,
    N6[4] = A6† A6 + A1† B5 + B5† A1 = -d_E Sigma6[4],
    rho64 (`docs/next_level/Ef,Ei`, full upstream checkout) = A6f† rho66 A6i
                      + B5f† rho55 A1i + A1f† rho55 B5i.

These coefficients carry g^4 exactly once. Direct/iterated amplitudes
interfere. The energy dependence of F6 creates both backflow endpoints.
Keeping only six-body probability misses them. An independent finite
block resolvent confirms the H/norm/current coefficient and the
next coupling-order remainder; zero charge, nonzero charges,
two-energy reversal and independent internal basis changes also pass.

The generic finite-matrix routine requires W6 explicitly and never
fills a missing physical operator with zero. The canonical U65 is now
implemented; the canonical direct quartic W6P is still missing.
The following calculation is deliberately cubic-path-only, not a
complete leading six-body result.

## Actual cubic-path diagnostic

K2=9, retained cutoff0, Q5/Q6 cutoff1, b=.4GeV, mq=.35GeV, mg=0,
gs=.3, proton, E=1.4GeV², Delta=(.12,.16)GeV.
Raw Q6 dimensions are8052 and69600 (cutoff0:792 and7200).
The two coherently summed U R5 V source norms are
.0342333511606 and .0899058969926GeV².

| Coefficient quantity | Value |
|---|---:|
| Sigma6[4] Frobenius norm, GeV² | 4.77942555927e-4 |
| N6[4] norm | 5.57154600132e-4 |
| six-body probability contribution norm | 1.22086606473e-4 |
| five-body backflow contribution norm | 4.35204608459e-4 |
| electromagnetic rho6[4] norm | 3.70758564497e-4 |
| charge/norm error | 6.23e-20 |

The two norm contributions are matrices; their displayed norms do not
add arithmetically. This is a finite-model diagnostic, not a physical
Fock probability, a new eigenvalue or an uncertainty bound. The old
matched parent remains unchanged.

## Verification and continuation

All37 new six-body tests pass in11.89s:9 color/Jacobi,11 matching,
5 spaces/current,7 vertices and5 actual-action/CCR tests.
The preceding canonical QQ source passed823 named M2+Q tests in1025.56s;
these are two separate invocations, not one full860-test execution.

    python -m pytest tests/test_m2_six_body_*.py -q
    python scripts/benchmark_m2_six_body_cubic.py

Next derive direct canonical IF, current-square and A^4 qqqg -> Q6
sources. Include the fixed-spectator valence emissions and the
gluon-to-three-gluon / gluon-to-gluon-plus-pair channels. Then combine
them as W6+U R5 V, keeping the backflow and matched currents above.
Common-regulator loop/current closure, physical finite conditions,
nuclear/GTMD identification and topology-factorized quantum scaling
remain ahead; none of the three physical workstreams is complete.
No old source file or oracle was overwritten by this increment.

## Direct quartic completion of the represented leading six-body excursion

All seven normal-ordered canonical qqqg -> Q6 source components now
execute. Their double-split contact is

    sqrt(xa xb xc/(xa+xb+xc)) (b²/pi) R_out^T C,

where C retains all three incoming Jacobi modes and evaluates the two
new relative coordinates at coincidence. It is not a single pair
contact. Both cutoffs0/1 give identical direct-source norms for the
retained all-ground state by exact oscillator selection; that is not
evidence of ultraviolet convergence.

The components and normalized factors are:

- Valence-quark IF two-gluon emission and quark/adjoint-current pair
  emission, with the old gluon an explicit spectator. The ordered kernels
  have the same canonical IF/IG factors as the earlier qqq source,
  but complete22-by2 color maps and normalized factor sqrt(6)/2 S3.
- Valence-number times sea-pair current, with the old gluon preserved in
  outgoing slot5: 2 A4, coherent up/down creation, complete8-by2 colors.
- A^4 gluon-to-three-gluon source: all24 field assignments times
  K2/[8 pi sqrt(6 kin k1 k2 k3)]. The Hamiltonian1/4 and normalized
  outgoing1/sqrt(3!) are included exactly once.
- IF gluon-to-gluon-plus-pair source: if kout is the surviving gluon and
  lanti the antiquark mode, the distinguished spin/color coefficient is

      2 K2/[2 pi sqrt(kout kin)] *
       [Sout,in C12/(kin-lanti) + Sin,out C21/(-kout-lanti)].

  The leading2 is normalized four-quark creation. Outgoing gluon
  polarization is conjugated; incoming is not. Both constrained momenta
  are signed odd integers. The second, negative denominator is retained.
- IG gluon-to-gluon-plus-pair: normalized2 A4 times

      2 K2/[pi (kq+kanti)²] Spair
        (kin+kout)/(2 sqrt(kin kout)) G,

  with G the projected T_pair.F_gluon and C12-C21=-G.
- IG gluon-to-three-gluon: for one distinguished surviving gluon and
  the other two a created pair, normalized sqrt(6)/2 S3 times

      2 K2/[pi (k2+k3)²] (kin+k1)/(2 sqrt(kin k1))
       [-i (k2-k3)/(2 sqrt(k2 k3)) eps2*.eps3*] C(F f).

  A fully independent expansion of all24 differentiated-field assignments
  in -1/2 (1/i d J)(1/i d J) reproduces this source atK2=11, including
  its sign and all Fock factors.

Independent one-to-three boson occupation algebra verifies A^4, while
actual ladder operators verify the two-created-boson factor on an
existing gluon. Canonical spinors and full finite-cell fields verify the
crossed IF and IG pair sources. Color closure, Jacobi measure, exact
Iz/Jz and statistics all pass. AtK2=9 the valence IG and pure-gluon IG
pair sources vanish because their pair modes are equal. They are
nonzero on the next grid; no missing interaction is set to zero.

At the standard cutoff1 point the individual direct source norms are:

| Component | Frobenius norm per gs², GeV² |
|---|---:|
| valence IF | .00928132521592427 |
| valence IG | 0 (equal-mode selection atK2=9) |
| valence sea | .0493940818367116 |
| gluon A^4 | .00859738752376137 |
| gluon IF pair | .00771413544608548 |
| gluon IG pair | .0552734684907149 |
| gluon IG three-gluon | 0 (equal-mode selection atK2=9) |

The coherent direct source norms in qqqggg / sea+g are
.0126514058506349 / .0708498123933061 GeV².
Combining W6P+U65 R5 V1 gives:

| Complete represented leading six-body coefficient | Value |
|---|---:|
| direct / iterated / interference return norms, GeV² | 2.84778118582e-4 / 4.77942555927e-4 / 5.84850491371e-5 |
| total H4 norm, GeV² | 7.20104830916e-4 |
| total N4 norm | 6.12956690899e-4 |
| electromagnetic rho4 norm | 4.08916592103e-4 |
| charge/norm error | 7.97e-20 |
| six-body probability coefficient norm | 1.78763041981e-4 |
| five-body backflow norm coefficient | 4.37691618395e-4 |
| first-order mass-squared response on preceding state, GeV² | -1.83402372503e-9 |

This completes that represented leading six-body path, not full order-g^4
QCD, renormalization or an interacting-Q resummation. An independent raw
matrix-free Q5/Q6 resolvent confirms the H/norm/current expansion and
its next-order remainder. The previous free-Q parent remains unchanged.

## One order-consistent H2 model and spin-resolved quantum measurement

The new combined snapshot uses the previous five-body V1/V2 returns,
the canonical QQ insertion and the leading six-body correction together,
truncated consistently through g^4. It does NOT import the isolated
g^5/g^6 QQ coefficients into a purportedly complete higher-order model.
Let A1=R5 V1, A2=R5 V2, C3=R5 W55 R5 V1+B5 and A6=R6 F6.
The local current/norm uses

    rho = rhoPP + g² A1f† rho55 A1i
        + g³ (A1f† rho55 A2i + A2f† rho55 A1i)
        + g⁴ (A2f† rho55 A2i + A1f† rho55 C3i
              + C3f† rho55 A1i + A6f† rho66 A6i).

Zero transfer gives q N. All raw tails are retained. Separate same-/opposite-
Jz fibers share the actual six-body density; the latter is not replaced
by a diagonal current. Exact Lz/Iz/spin blocks accelerate the symmetry
operations without discarding amplitudes and agree with the original
unpacked transformation on arbitrary complex probes.

At E0, the energy-local orthonormal model is

    N = I - d_E Heff(E0),
    Hloc = N^(-1/2) [Heff(E0)+E0(N-I)] N^(-1/2),
    rholoc = Nf^(-1/2) rho(Ef,Ei) Ni^(-1/2).

Whitening reconstructs only the small correction to I, avoiding precision
loss from nearly degenerate norm eigenvectors. For delta an eigenvalue of
N-I, the correction is -delta/[sqrt(1+delta)(1+sqrt(1+delta))].
Nonpositive metrics are rejected, never clipped. This is an exact
coordinate conversion of the energy-linearized truncated model,
not an all-orders QCD resummation.

At E0=1.4GeV² both proton and neutron controlled-gate circuits use
six qubits and72 parameters, with no StatePrep or exact-state seed.
Their executed residuals are7.74622e-9 /9.54606e-9 GeV² after58 updates
and117 evaluations each. Actual fit times are4.18 /3.63s; source/snapshot
construction took84.84 /76.40s, so the fit time is not total runtime.
Both energies are1.42143040786803GeV² within roundoff in the declared
isospin limit. Their minimum norm eigenvalues are1.000000018861607.

| Target | measured F1 | measured F2 | absolute F1/F2 errors against same exact model |
|---|---:|---:|---:|
| proton | .882386789716565 | 4.77647025206e-7 | 2.16449e-12 /8.84037e-16 |
| neutron | 6.34344449266e-10 | -3.18630364770e-7 | 6.41035e-12 /1.88495e-15 |

The corresponding evaluated finite-reference state-error bounds are
2.42533e-8 /1.40350e-11 and4.07484e-9 /1.76774e-11. These are not
physical uncertainty bands. Padding is zero. Hamiltonian parity errors
are below1.97e-14 GeV², and metric parity errors below4.85e-15.
The retained positive-parity phase is explicit; the exact eigensolve
occurs only AFTER optimization. Dense classical geometry and an O(D)
graph ansatz remain a small-model baseline, not quantum scalability.

Focused evidence:22 quartic tests pass;8 integrated six-body tests pass;
the extended spaces/current suite passes10 tests, including5 new
spin/block tests; the combined-H2 suite passes its first10 tests, with
2 additional whitening checks passed separately. The18-test combined
space/six-body rerun passes after exact-block acceleration. The last
complete named regression before this increment was860 tests at26f71d0f,
1885 inherited warnings,951.38s. The new complete named regression at
320ae968 passes907 tests (874 M2,33 Q),1885 inherited warnings,1113.67s,
in one invocation. This is not a full-repository test claim.

    python scripts/benchmark_m2_canonical_six_body.py
    python scripts/benchmark_m2_fourth_order_graph.py --target PROTON
    python scripts/benchmark_m2_fourth_order_graph.py --target NEUTRON

The pre-edit main note, pickup, six-body space/current and result note
at9a6cb24d are preserved in
computation_handoff/backups/2026-09-06_pre_canonical_self/pre_six_body_direct_notes.tar.
Next advance common-regulator resolution/loop/current matching, including
the non-diagonal free resolvent needed beyond the first oscillator shell.
Do not infer ultraviolet convergence from equal direct-source norms at
cutoffs0/1. Shared physical finite conditions, complete four-currents,
nuclear/GTMD matching and topology-factorized quantum scaling remain open.

# Canonical sea-pair source and coherent cubic/quartic H2 matching

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-06. **Explicit finite-model result**, not complete order-gs^4
QCD, physical nucleon form factors, or a calibrated deuteron prediction.
This advances the chosen canonical Galerkin/H2-first construction without
reopening either direction choice.

## Instantaneous sea-pair source

The normal-ordered quark-current product in Srivastava--Brodsky Eq.24
contains valence-number and pair-creation currents. For an incoming
valence quark p, final valence p', and created quark/antiquark k,l,
p=p'+k+l in positive odd doubled modes. The pair transfer k+l is positive
and even, so this source never uses a zero-mode inverse.

With the three- and five-body color-singlet isometries U3,U5, define

    C_qbarq = U5† sum_a [(T^a)_emitter U3] tensor T^a_new_pair,
    S_qbarq = ubar_hq gamma+ v_hanti / sqrt((2p_q+)(2p_anti+)).

The complete outgoing color carrier has three singlets. Canonical charge-
conjugated spinors give S=-1 for opposite physical pair helicities and
zero for equal helicities. The valence good current is helicity diagonal.
Including both current orders, the finite cell and all four field norms,

    V_sea/g² = 2 K2/[pi (k+l)²] delta_h'h S_qbarq C_qbarq.

This formula is independently checked at different masses and P+ using
full external spinors. The same double-splitting Jacobi map used for the
two-gluon source preserves the two inherited coordinates and supplies
sqrt(x_valence x_newq x_anti/X_parent) times b²/pi.

The coherent flavor source is u ubar+d dbar. The normalized creation
factor is **2 A4** after the inherited three-quark antisymmetry. Independent
occupation-bit CAR calculations verify the factor and normal-order sign;
it is not inferred only from reprojecting an already antisymmetric output.
Tests also exchange the newly created quark with an inherited quark.

At the shared K2=9, N_exc(P)=0, b=.4 GeV, mq=.35 GeV, mg=0 point,
the sea source maps 10 retained qqq columns into 4500/31500 packed rows
at Q cutoffs0/1. Its norm per gs² is .0468709290598 GeV² at either
cutoff. The equality follows from the new m=0 mode selection and is not
evidence of ultraviolet convergence.

## Preserve coupling orders before source compression

Both actual excluded channels, qqqgg and qqqq-qbar, now have

    V_QP(g)=g V1 + g² V2,

where V1 comes from retained qqqg columns and V2 from retained qqq columns.
For qqqgg, V2 includes the coherent instantaneous-fermion and mixed-
adjoint-current amplitudes. For the sea channel it is the quark-current
source above. Neither is an extra gs-linear vertex.

Within each exactly degenerate free-energy subspace, compression spans the
UNION of the individual order images before multiplying by g. Each image is
independently reconstructed with a reported roundoff residual. This retains
new quartic directions even at tiny coupling or g=0; equal dimensions never
replace the explicit source-column and regulator identity checks.

The resulting parent keeps the caller's original H_PP and free H_QQ:

    H_parent = [[H_PP, V†], [V, H0_QQ]],
    Sigma(E,g)=g² Sigma2+g³ Sigma3+g^4 Sigma4,
    Sigma2=V1†R0V1,
    Sigma3=V1†R0V2+V2†R0V1,
    Sigma4=V2†R0V2.

The gs³ term is a resolved retained qqq/qqqg transition, not zero.
The induced norm is N=I-partial_E Sigma, with the sign of the derivative
retained. For the number-preserving local good current at Delta+=0,

    rho_eff(Ef,Ei)=rho_PP
      + sum_(r,s=1,2) g^(r+s) Vr† Rf rho_QQ Ri Vs.

Both energy endpoints are independent. Every coefficient is tested against
the full finite-parent wave-operator result and transfer reversal.
At zero transfer, each nonzero-order charge coefficient is q times the
corresponding norm coefficient. Omitting the gs³ current while retaining
its Hamiltonian partner would fail that identity.

The same generic finite-reference transport and spin-current interfaces
apply to this parent; no old cubic-only oracle is overwritten.

## Numerical and quantum result

The standard gs=.3, Q-cutoff1 proton parent grows from 165 to **184**
source-reachable states, while the matched H2 register remains 37 states.
At Q cutoff0 the parent grows from92 to111, with
M²=1.421431301525935 GeV² and Q probability7.82258782480e-7.
Proton/neutron masses agree at both cutoffs in the declared isospin limit.
Their cutoff0 Pauli amplitudes vanish to roundoff; cutoff1 permits the
nonzero orbital/spin-flip matching shown below.
Its self-consistent M² is 1.421430409460543 GeV² and Q probability is
1.22227353213e-6. Independent parent eigenvalues and fully uncompressed
Q/P equations agree to roundoff (residual1.86e-15 GeV²).
The shift from the cubic-only reference is -5.88707354e-7 GeV².

At E=1.4 GeV² the Frobenius norms of Sigma2,Sigma3,Sigma4 are
.0860178081710, .000951747259447 and .000233369726763 GeV².
These are coefficients before the respective coupling powers, not
three independent physical mass shifts.

At Delta=(.12,.16) GeV and the self-consistent energy:

| Target | F1 | F2 |
|---|---:|---:|
| proton | .882386786096 | 4.87859814425e-7 |
| neutron | 4.31027656894e-10 | -3.25474063902e-7 |

The parity-aligned two-spin structures close to roundoff. These diagnostic
form factors use the finite-model mass and incomplete physical currents.

The energy-local E0=1.4 reference is separately solved by the actual
six-qubit/72-parameter graph circuit, with no exact-state seed or StatePrep.
The proton run reaches residual7.56804e-9 GeV² in 64 updates (3.95 s).
Its energy1.421430409715187 GeV² agrees with the exact SAME energy-local
model. Measured F1=.882386790456146 and F2=4.76701765737e-7 have
errors9.70e-13 and7.01e-16, below evaluated spectral bounds2.37e-8 and
1.37e-11. Do not compare the energy-local and self-consistent currents
as if they were numerically identical. Full classical geometry remains
a small-model research tool, not a quantum-scaling claim.

The separate neutron run reaches residual8.35018e-9 GeV² in62 updates,
126 evaluations (44.97 s under concurrent load), with zero padding.
Its measured F1=6.72633275662e-10 and F2=-3.18030080601e-7
differ from the identical energy-local exact reference by4.21e-12 and
2.21e-15, below the evaluated bounds3.56e-9 and1.54e-11.
The stopping rule is the executed residual; the separately reported
requested gradient tolerance is not met and is not relabeled as success.

## Reproduction and remaining physical boundary

    python scripts/benchmark_m2_order_resolved.py --spin
    python scripts/benchmark_m2_graph_convergence.py --natural --iterations 600 --metric-rcond 1e-18 --quartic --spin-current
    python scripts/benchmark_m2_graph_convergence.py --natural --iterations 600 --metric-rcond 1e-18 --quartic --spin-current --target NEUTRON
    python -m pytest tests/test_m2_instantaneous_sea_exit.py tests/test_m2_order_resolved_reference.py -q

11 new sea-source tests and 13 order-resolved tests are supplied. The
preceding 74-test focused integration passed before adding the final
zero-coupling test; the latest broad regression is recorded in the handoff.
Source backups are `pre_sea_pair_source.tar` and
`pre_order_resolved_benchmark.tar` under
`computation_handoff/backups/2026-09-06_pre_canonical_self/`.

Next complete interactions WITHIN Q and the required six-body paths,
then common-regulator loop/current and physical finite matching.
At order g^4, for example, V1†R0 W_QQ R0V1 is not generated by the
free-Q parent. Returning repeatedly through P is not a substitute.
Full transverse/non-Abelian, endpoint/zero-mode and renormalized currents,
physical calibration, nuclear/GTMD matching and topology-factorized quantum
scaling remain open. All three workstreams remain physically incomplete.

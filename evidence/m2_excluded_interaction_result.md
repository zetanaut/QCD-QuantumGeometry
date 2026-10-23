# Canonical fermion interactions inside Q: Hamiltonian, norm and current

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-06. **Validated finite-model operator increment**, not full
order-gs^4 QCD, physical renormalization, or a calibrated observable.
The selected canonical/H2-first direction is unchanged.

## What is derived

The normal-ordered quark-current square supplies, within the represented
five-particle Q sectors:

- all distinct qq scattering pairs in qqqgg;
- all distinct qq and q-antiquark scattering pairs in qqqq-qbar;
- annihilation/recreation of any of the four quarks with the antiquark,
  with coherent u/d pair recreation.

The complete six/three color-singlet carriers remain resolved. Ordered-slot
spin/flavor labels, exact longitudinal modes, four Jacobi oscillators and
the joint Iz/Jz coordinates are retained. Scattering is spin/flavor diagonal;
annihilation requires same-flavor opposite-physical-helicity pairs.
The mixed gluon-current, instantaneous-fermion and pure-gluon QQ pieces,
cross-channel gg <-> quark-antiquark contacts and six-body paths are
**not included** in this increment.

For fermion slots i,j, an adapted pair-first Jacobi tree gives

    D_ij = b²/pi sqrt(alpha_out alpha_in) R_out^T C R_in,
    alpha = x_i x_j/(x_i+x_j).

C evaluates the first oscillator at transverse coincidence and preserves
the other three coordinates. Independent six-dimensional physical-coordinate
quadrature verifies the whole cutoff1 block; it does not reuse this tree.

Both current orders give the scattering coefficient

    H_sc/g² = W_sc = 2 K2/[pi (Delta k_i,2)²] D_ij (sum_a T_i^a T_j^a).

The coefficient W used below already has g² factored out. The antiquark
generator is -T^T, derived from normal ordering d d†. The external
vbar gamma+ v / sqrt(2p' 2p) is positive helicity identity, not a negative
norm. Zero longitudinal transfer is removed exactly by Q0, including
transverse rearrangements at unchanged longitudinal labels.
Each distinct pair is summed once, with no additional factorial.

Annihilation instead uses the positive TOTAL pair mode s=k_q2+k_anti2:

    H_ann/g² = W_ann = 2 K2/(pi s²) D_q,anti
               U† [sum_a |T^a><T^a|]_q,anti U S_out S_in*,
    sum_a |T^a><T^a| = (I_9 - |I_3/sqrt(3)><I_3/sqrt(3)|)/2.

Here S=-1 for each allowed canonical pair spin. The denominator never
uses Q0. In particular the same incoming/outgoing longitudinal tuple
has a nonzero annihilation block. Independent SU(3) Fierz and
occupation-bit CAR tests verify the positive pair-current square and
the absence of an extra four-quark creation factor.

The returned matrices are the coefficients W_sc and W_ann;
their physical interaction contribution carries g² exactly once.

## A single QQ insertion changes the current support

Write H_QQ=H0_QQ+g²W. Cubic Q5-to-Q5 transitions are absent by
particle-number counting; the missing cubic paths go to four/six particles.
The present calculation adds the selected W **perturbatively**, without
overwriting or relabeling the previous free-Q parent:

    R(E) = (E-H0_QQ)^-1,
    A1(E) = R V1,                 order-g wave coefficient,
    A3(E) = R W R V1,             order-g³ single-W coefficient,
    H4(E) = V1† R W R V1,
    N4(E) = A1† A3 + A3† A1 = -d_E H4,
    rho4(Ef,Ei) = A3f† rho_QQ A1i + A1f† rho_QQ A3i.

H4,N4,rho4 are coefficients: multiply them by g^4 exactly once.
The quartic exit V2 already supplies other g^4 contributions through the
separate coherent reference. Mixing V2 with W first enters at g^5;
none of these statements supplies all g^4 QCD.

If U is the old free-source isometry, A1 lies in its image, so replacing
A3 by UU†A3 leaves H4 and N4 unchanged. But it loses

    (A3f_perp)† rho_QQ A1i + A1f† rho_QQ A3i_perp

already at order g^4. At zero transfer rho_QQ=q I and this term is zero
by orthogonality. Charge normalization therefore does not detect the
finite-transfer information loss.

The implementation uses raw Q tails for currents and requires each W
to bind the exact raw space object. Equal dimensions are not a basis map.
Independent raw interacting-Q resolvents test the first-insertion
Hamiltonian and current; both endpoint reversal and four separate
charge/norm identities are checked. An individual perturbative norm
coefficient is not required to be positive.

## Numerical witness

Same diagnostic point: K2=9, retained cutoff0, Q cutoff1, b=.4 GeV,
mq=.35 GeV, mg=0, gs=.3, proton, E=1.4 GeV², Delta=(.12,.16) GeV.

| Quantity | Value |
|---|---:|
| Raw qqqgg / sea dimension | 4950 / 31500 |
| QQ coefficient Frobenius norms, GeV² | .218346563039 / 2.07470028559 |
| H4 coefficient Frobenius norm, GeV² | 1.36080685390e-4 |
| N4 coefficient norm | 1.04928286837e-4 |
| electromagnetic rho4 coefficient norm | 7.39697355187e-5 |
| current coefficient lost by old source projection | 3.53769621653e-6 |
| lost current norm after gs^4 weighting | 2.86553393539e-8 |
| A3 wave leakage, qqqgg / sea | 2.96678464391e-4 / 3.48306071290e-4 |
| zero-transfer electromagnetic norm/charge error | 7.67e-21 |

The first-order energy response on the norm-correct, self-consistent
previous finite-reference state is +6.83839647792e-11 GeV². This small
state expectation does not bound the operator correction or omitted
physical terms. It is a first-order insertion response, not a new
nonperturbative eigenvalue or physical mass.

## Verification and continuation

16 scattering/insertion tests and7 annihilation tests are supplied.
The initial scattering/insertion suite passed in two eight-test runs;
after exact-space binding, the relevant rerun plus annihilation suite
passed16 tests. The final joint rerun passes all23 tests in37.60 seconds.
Reproduce with:

    python -m pytest tests/test_m2_excluded_fermion_scattering.py tests/test_m2_excluded_annihilation.py -q
    python scripts/benchmark_m2_excluded_interaction.py
    python scripts/benchmark_m2_excluded_interaction.py --scattering-only

The two earlier source modules are unchanged; these are new separate
files. The preceding pickup/main-note sources are archived in
computation_handoff/backups/2026-09-06_pre_canonical_self/pre_order_resolved_notes.tar
at434d89cf, while the sea/order-resolved increment is preserved atf919976a.

## Subsequent extension: complete represented instantaneous-gluon current square

The above fermion-only result is preserved at 01ef56a1. The subsequent
extension supplies the remaining number-current qg/gg scattering and a
coherent pair-current square across both Q channels. This completes the
normal-ordered instantaneous-gluon current-square QQ contribution on
the represented I=1/2, quark-antisymmetric five-body source sector.
It does NOT complete the canonical QQ Hamiltonian: instantaneous-fermion,
A^4, counterterm/current matching and six-body partners remain separate.

For each gluon in a scattering pair multiply its color generator F^a by
(k_out+k_in)/(2 sqrt(k_out k_in)). Equal modes give positive adjoint
charge +F^a. Independent differentiated-field contractions and the
complete-singlet Casimir identity sum_(i<j) T_i.T_j=-5 I_6 verify this sign.
Each distinct qg or gg pair occurs once; an independent two-boson CCR
calculation verifies the pair count after removing the quadratic Wick term.

For the pair channels introduce an auxiliary qqq/color-adjoint CURRENT
coordinate. It has spin projection zero because it represents J+ under
transverse rotations; **it is not a physical helicity-zero gluon and has
no propagator or added Fock-sector probability**. Its complete finite-shell
I=1/2 and S3-sign isometry is built explicitly. A positive pair-transfer
weight D=2K2/[pi (k1+k2)^2] acts on that coordinate.

The two outgoing maps are

    B_sea = 2 A4 [S_pair T^a] times the one-new-coordinate contact,
    B_gg  = [-i (k1-k2)/(2 sqrt(k1 k2)) eps1*.eps2* f^(a b c)]
             /sqrt(2!) times the same contact.

The contact contains sqrt(x1 x2/(x1+x2)) b/sqrt(pi), with all inherited
Jacobi modes retained. Both gluon creation assignments occur before the
normalized Bose factor. Field contractions fix the relative i/sign.
An independent pair-creation CCR/Wick calculation verifies the 1/sqrt(2!)
factor, separately from the number-current CCR test.

On the represented symmetry sector the pair-current interaction is

    W_pair(s,t)=B_s D B_t†,     s,t in {qqqgg, qqqq-qbar}.

The sea diagonal agrees with the independent sum over all four quark
annihilators on both cubic and quartic source images at cutoffs0/1.
The gg diagonal supplies adjoint-pair annihilation; the off-diagonal
entries supply the mixed IG gg <-> quark-antiquark conversion with its
fixed interference. Do not add the separate sea-annihilation operator
again. The sum is a positive pair-current square; its cross contribution
alone need not be positive.

The same raw-tail H4/N4/two-energy-current insertion equations now sum
incoming sectors coherently before constructing each outgoing tail.
An independent low-rank interacting-Q resolvent verifies this construction,
including the off-diagonal sector terms and two independent current
energies. Zero-transfer charge, norm derivative and reversal still close.

At the same cutoff1 diagnostic point the current auxiliary rank is100,
with B_gg/B_sea norms .237883215487 /1.17029784925 GeV.
The qg and gg scattering coefficient norms are .777264696225 /
.828566950166 GeV². The pair cross-channel H4 coefficient norm is
1.23000844093e-6 GeV² and is not zero.
For the full represented current square:

| Quantity | Value |
|---|---:|
| H4 coefficient norm, GeV² | 2.79017865231e-4 |
| N4 coefficient norm | 2.22282395757e-4 |
| electromagnetic rho4 coefficient norm | 1.44219563560e-4 |
| lost current coefficient under old source projection | 6.59794719586e-6 |
| lost current after gs^4 weighting | 5.34433722864e-8 |
| raw A3 leakage, gg / sea | 1.00001443606e-3 /3.48445038969e-4 |
| charge/norm error | 1.02e-20 |
| first-order M² response on previous state, GeV² | +6.71057592562e-11 |

These remain single-insertion finite-model quantities, not physical
predictions or a resummed interacting-Q spectrum. The 20 new tests
(9 scattering,11 pair-current) pass: the first19 plus preceding23 passed
in a42-test integration, and the final independent pair CCR test passed
separately. Reproduce with:

    python scripts/benchmark_m2_excluded_current_square.py
    python -m pytest tests/test_m2_excluded_current_scattering.py tests/test_m2_excluded_pair_current.py -q

The prior pickup, main note and this result note at f925ceb4 are preserved
in pre_full_QQ_current_notes.tar under the same backup directory.

The instantaneous-fermion/A^4 continuation below supersedes those QQ targets.
Next derive the six-body paths using
the same normalization and coordinate authority, and apply a common
loop regulator and physical matching conditions to H/norm/current
together. Keep the new current-support warning in that implementation.
Physical calibration, full transverse/non-Abelian/zero-mode currents,
nuclear and GTMD matching, and scalable topology-aware quantum methods
remain open. None of the three requested physical workstreams is complete.

## Canonical quartic QQ completion on the represented source sector

The next increment adds qg instantaneous-fermion scattering, normal-ordered
A^4 gg scattering, and IF gg <-> sea conversion. Together with the above
current square these are all canonical normal-ordered quartic QQ source
actions in this represented I=1/2, quark-sign/Bose five-particle model.
This is not a universal operator on other isospin sectors, renormalized
QCD, a six-body completion, or an interacting-Q resummation.

For IF qg scattering, the ordered colors are Ta Tb and Tb Ta. Their
difference is -Tq.Fg. The contact coefficient is

    K2/[2 pi sqrt(k_g,in k_g,out)] D
      [Cdirect Sdirect/(k_q+k_g,in) + Ccross Scross/(k_q-k_g,out)].

The constrained momenta are signed odd integers, never zero. Negative
denominators are not clipped, replaced by absolute values or removed by Q0.
The independent canonical-spinor calculation fixes each helicity factor.

For A^4, distinguish legs 0,1 outgoing and 2,3 incoming. Sum all24
assignments to f^(a b c) f^(a d e) A_b^i A_c^j A_d^i A_e^j,
with conjugate outgoing polarizations. The resulting color/spin tensor
multiplies

    K2/[16 pi sqrt(k1,out k2,out k1,in k2,in)] D.

This contains the Hamiltonian source factor1/4 and the normalized two-gluon
Fock factor1/2. An independent SU(2), two-polarization occupation-basis
calculation normal-orders the polynomial with actual square-root ladder
amplitudes and verifies these factors. The SU(3) projection is complete.
This normal-ordered term includes no quadratic Wick self contribution.

For gg -> q anti-q, distinguished IF conversion is

    K2/[2 pi sqrt(k1 k2)]
      [S(q,anti,1,2) Cdirect/(k2-l_anti)
       +S(q,anti,2,1) Creverse/(k1-l_anti)].

Cdirect and Creverse are the complete sea/gg color projections of Ta Tb
and Tb Ta. They obey Cdirect-Creverse=i Csea Cgg† with the earlier cubic
color phases. The spin factors use ubar gamma.eps1 gamma+ gamma.eps2 v
with the same normalized canonical spinors. The cross-species transverse
contact preserves the three spectators and is independently checked by
six-dimensional coordinate quadrature. On normalized states the forward
map is sqrt(2) A4 M, where A4 is the four-quark antisymmetrizer and M
the distinguished kernel; the reverse is its adjoint. The independently
constructed IG conversion agrees with Bsea D Bgg† on both source orders
at cutoffs0/1. It is a cross-check, NOT an additional interaction.

Write V=g V1+g² V2 and W for the complete canonical QQ coefficient.
For r=1,2 define Ar=R Vr and Br=R W R Vr. A single W insertion gives

    H[n] = sum_(r+s+2=n) Vr† Bs,
    N[n] = sum_(r+s+2=n) (Ar† Bs + Br† As) = -d_E H[n],
    rhon (`docs/next_level/Ef,Ei`, full upstream checkout) = sum_(r+s+2=n)
       (Br(Ef)† rho As(Ei) + Ar(Ef)† rho Bs(Ei)),   n=4,5,6.

All Q channels are summed coherently. The weights g^n are applied once.
The g^5 interference is nonzero. These are the complete SINGLE-INSERTION
coefficients from the two given source orders, not all contributions of
QCD through g^6. In particular repeated W and six-body paths are absent.

At the same cutoff1 diagnostic point:

| Single-insertion coefficient | g^4 | g^5 | g^6 |
|---|---:|---:|---:|
| H Frobenius norm, GeV² | 2.82007670012e-4 | 4.42606690423e-6 | 4.17305566446e-7 |
| N norm | 2.34830067004e-4 | 3.58680916953e-6 | 2.81219128901e-7 |
| electromagnetic current norm | 1.52232269858e-4 | 2.53839463643e-6 | 2.13989852281e-7 |
| zero-transfer charge/norm error | 1.20e-20 | 3.52e-22 | 3.19e-23 |

The weighted sum has H norm2.28428746773e-6 GeV² and first-order
mass-squared response +5.75468240454e-11 GeV² on the preceding
norm-correct self-consistent state. Neither is a new eigenvalue or
a bound on omitted physical effects. The leading current coefficient
lost by the old source projection is6.99064664954e-6.

Verification adds30 tests: IF5, A^4 6, pair conversion8, coherent
canonical insertion11. The full named M2+Q0--Q2 regression passes
823 tests (790 M2,33 Q),1885 inherited warnings,1025.56s. This is not
a full-repository regression or evidence of physical completeness.

    python scripts/benchmark_m2_canonical_qq.py
    python -m pytest tests/test_m2_*.py tests/test_q0_*.py tests/test_q1_*.py tests/test_q2_*.py -q

The old quadrature helper is preserved at db4a88b7 in
pre_IF_A4_QQ_tests.tar in the same backup directory. No old oracle is
overwritten. Continue with coherent six-body excursions, their
five-body backflow and matched current/norm partners, then common-regulator
and shared physical matching. The original three workstreams remain open.

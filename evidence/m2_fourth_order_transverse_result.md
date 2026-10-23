# Common-order H2 transverse current and the noncommuting cutoff partner

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Historical checkpoint: the interacting minus continuation and all-component
measurements are now in [the current result](m2_four_current_result.md).
Statements below about then-unavailable B2/B3 and next work describe the
recorded source revisions; they are not the current queue.

The canonical tree-sector transverse current is now matched through order
g⁴ on the declared interacting QQ/Q6 H2 parent. This is not full order-g⁴
QCD, a complete four-current, a physical fit or restored rotational covariance.
The source increment is `125dcea9`; the separately callable free-projector
partner is source `4309412e30e291564c49a73eedb8c5afb5d77c33`.
H2-first and canonical global-Jacobi authority
remain the selected directions.

## Antiquark sign and six-body contact

The antiquark one-gluon bilinear is evaluated with signed crossed momenta
(-p_in,-p_out), charge-conjugate good-spin axes and the external spin transpose.
Both the physical antiquark charge and its anti-fundamental color generator
already contain a minus sign. Normal ordering gives only one CAR minus.
Factoring out both charge and color therefore leaves an extra minus in the
crossed spin kernel. Absolute-value constraint momenta would be wrong.
Independent full-Dirac instantaneous insertions and Hamiltonian-vertex Ward
identities test this factorization.

The factorized Q5→Q6 transverse contact retains four routes:

| Source and action | Destination | Explicit Fock factor |
|---|---|---|
| qqqgg, quark emission | qqqggg | √3 |
| qqqgg, gluon to quark/antiquark pair | qqqq-qbar-g | 2√2 |
| qqqq-qbar, any quark emission | qqqq-qbar-g | 1 |
| qqqq-qbar, antiquark emission | qqqq-qbar-g | 1 |

The last two include the full fermion projectors; factors are not substitutes
for antisymmetrization. Complete 22/8-color spaces and gluon statistics remain.
The local relative integral is b/√π for every m=0 radial mode, multiplied by
√[xy/(x+y)]. The photon shifts the parent Jacobi coordinate before the exact
five-coordinate rotation. Physical absorption is the reversed-transfer
emission adjoint. No raw amplitude or color channel is threshold-pruned.

Independent ten-dimensional polar-wave quadrature, zero-transfer Hamiltonian
vertex derivatives, boosts, flavor, statistics, adjoints and a complete
second-shell raw-vector action test the operator.

## Common-order current, including raw backflow

Use the same two snapshots and norms as `FourthOrderH2`. Write

    W5 = g A1 + g² A2 + g³ C3,    C3 = C3_QQ + B5_Q6,
    W6 = g² A6.

For each transverse component the *raw* effective current is the polynomial
through degree four of

    JPP[0] + g JPP[1]
    + W5_f† J55[0] W5_i
    + g (W5_f† J5P[1] + JP5[1] W5_i)
    + W6_f† J66[0] W6_i
    + g (W6_f† J65[1] W5_i + W5_f† J56[1] W6_i).

In particular J[4] contains A2†J55A2, A1†J55C3 and its other endpoint,
both P/Q5 contact terms involving C3, A6†J66A6, and both Q5/Q6 contacts
involving A1/A6. There is no canonical J_T[2] bad-field square: that term
belongs to J-minus. This zero is algebraic, not an omitted unknown term.
The normalized operator is N_f^(-1/2) J_raw N_i^(-1/2), using the declared
finite-model norms. This whitening is an explicitly defined model map;
it is not a claim to have all perturbative coefficients at higher orders.

The tests compare every degree-zero-to-four coefficient with independently
evaluated whole-parent wave products, first retaining and extracting their
nonzero degree-five/six terms. Distinct energy and target-spin endpoints,
all reversed-transfer adjoints and omission witnesses are checked.

## Executed quantum measurements

Common bare parameters: K2=9, retained cutoff0/dimension37, b=.4 GeV,
mq=.35 GeV, mg=0, g=.3, matching energy1.4 GeV², P+=1.3 GeV and
Delta=(.12,.16) GeV. Each six-qubit controlled-gate state is optimized
without an exact-state seed. The exact eigensolve occurs afterward.
All twelve complex plus/x/y helicity matrix elements are measured as
24 Hermitian quadratures with the same controlled gates.

| Target | Q cutoff | M² (GeV²) | State residual (GeV²) | G_M from plus | G_M from transverse |
|---|---:|---:|---:|---:|---:|
| proton | 1 | 1.421430407868028 | 9.4773e-9 | .882387267363 | 2.644271817627 |
| neutron | 1 | 1.421430407868027 | 7.3951e-9 | -3.179945432e-7 | -1.762848324562 |
| proton | 2 | 1.421428799077575 | 9.4775e-9 | .882386934506 | 2.644266373730 |

Updates/evaluations are56/113,57/115,56/113, respectively; padding is zero.
Maximum current errors are3.62e-12,4.93e-12,3.62e-12. Every current error
satisfies its own residual/gap-derived state bound; the largest such bounds
are2.97e-8,5.64e-9,2.97e-8. Total runtimes are108.56,109.75,589.58s;
these are not isolated hardware performance comparisons.

The corresponding B1 diagnostics are-.880942275132,+.881424003284,
-.880939719612. B2/B3 remain unknown because the interacting minus current
is not yet matched. Orientation-structure residuals are below2.8e-14
and elastic continuity defects below3.9e-15. **Neither removes the order-one
magnetic disagreement.** These numbers supersede neither the separate
self-consistent free-Q5 controls nor the bare204/654-state solver controls.

## A regular partner for the oscillator projector

The oscillator projector Pi_N does not commute with H0. The
[gauging-equations paper](https://arxiv.org/abs/0904.4128) motivates attaching
the photon to regulator factors as well as dynamics, but its commuting
momentum-cutoff expression cannot be substituted for this projector.
The construction here is a separately declared exploratory straight-path
representative, not a uniquely physical current choice.

Let D(Delta) translate momenta by w_a Delta and let T_j be its derivative at
zero. Since Pi_N has finite oscillator rank, [Pi_N,T_j] has exact support
through shellN+1. Define

    C_j = integral_0^1 D((1-t)Delta) [Pi,T_j] D(t Delta) dt,
    K_j = Pi C_j Q + Q C_j Pi,        Q=1-Pi.

Differentiating D((1-t)Delta) Pi D(t Delta) gives

    Delta.K = [Pi,D],    K = Pi K + K Pi,
    K(-Delta)† = K(Delta).

The off-diagonal projection is the first-order idempotent tangent. Its
discarded diagonal pieces have zero divergence; this is explicit path
freedom, not amplitude pruning. The complementary projector has current -K.
No division by Delta or by an energy difference is used. Continuum displaced
overlaps sandwich the finite commutator; increasing the *reported* outer
shell does not change already reported matrix elements.

Gauging the two projector legs of Pi H0 Pi, then restricting external states,
gives the free regulator contribution

    J_boundary^j = (Pi K_j Q H0 Pi + Pi H0 Q K_j Pi)/(2P+).

Its exact divergence is

    2P+ Delta.J_boundary = -Pi H0 Q D Pi + Pi D Q H0 Pi
                        = [H0_N,rho_N] - 2P+ Delta.J_canonical_N.

Only shellN+2 is needed for the quadratic H0. Mass constants commute with
Pi and contribute zero. The free Hamiltonian remains unchanged.
This partner is not automatically added to any current or H2 snapshot.
Interaction vertices, field-contraction cutoffs, loops and full-current
matching require their own consistent partners.

The ground-shell orbital boundary current vanishes exactly and cannot cure
the already demonstrated g=0 magnetic discrepancy. Regular current
conservation is necessary, not sufficient for Lorentz covariance.

## Reproduction and next work

New common-current tests:59 passed together in87.49s (28 antiquark,
18 six-body contact,13 matching). The21 new projector checks passed
separately in141.25s. A subsequent combined run passed all80 tests
in217.24s. Test counts are coverage, not
scientific evidence by themselves.

    python -m pytest tests/test_m2_antiquark_contact_current.py \
      tests/test_m2_six_body_transverse_contact.py \
      tests/test_m2_fourth_order_transverse_current.py \
      tests/test_m2_galerkin_projector_current.py -q -p no:cacheprovider
    python scripts/benchmark_m2_matched_transverse_quantum.py --target PROTON --cutoff 2
    python scripts/benchmark_m2_matched_transverse_quantum.py --target NEUTRON --cutoff 1

Next derive/project interaction-dependent minus and two-gluon current blocks
with their Wick partners; propagate the same regulator/loop/counterterm
choices through H, norm and every current. Test retained/excluded/longitudinal
resolution and spin/rotation, not just charge or continuity. Shared physical
finite conditions and input/holdout separation are unselected. Nuclear
spin-one/GTMD matching and genuine many-body quantum compression remain.
All three physical workstreams are incomplete; parameterized operator work
can continue without reopening the selected directions.

Pre-edit docs/Bib/status/test copies from `babbe4af` are in
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_fourth_order_transverse_and_projector.tar`.
Original shared theory materials and all22 source volumes are unchanged.
No public push or publication-mirror refresh is part of this increment.

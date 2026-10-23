# Canonical current matching and the covariance warning

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Historical source: `88fe087e658115712fa792e12eebba3e18a1ed00`, following
`3b8bcf79` (matched current) and `493a9e21` (quantum retraction).
This is finite-model/operator progress. None of the three physical
workstreams is complete. H2-first and canonical global-Jacobi authority
remain selected; no historical C43/C57 reproduction is required.

## Completed current increment

The excluded qqqg→qqqgg and qqqg→qqqq-qbar transverse contacts now use the
canonical constrained-field kernels, all coherent emitters, complete color
maps, sqrt(2) Bose and 2 A4 fermion factors. The relative local oscillator
integral is b/sqrt(pi) for every m=0 radial mode, multiplied by the exact
sqrt(x y/(x+y)) Jacobi factor. Parent recoil precedes the exact tree rotation.
Current spin endpoints are independent: a scalar-Hamiltonian fixed-Jz
gather must not silently erase vector-current blocks.

Independent eight-dimensional polar-wave quadrature and zero-transfer
derivatives of the separately implemented Hamiltonian vertex test the
orbital, field, color and Fock normalization. Reversed-transfer adjoints
supply absorption; it is not twice one endpoint.

For the explicitly **free-Q5** parent, write
W_Q(E)=g A1(E)+g² A2(E), A_a=R_Q(E)V_a. The complete supplied transverse
operator is

    J_eff = J_PP[0] + g J_PP[1]
            + W_f† J_QQ[0] W_i
            + g W_f† J_QP[1] + g J_PQ[1] W_i,
    J_local = N_f^(-1/2) J_eff N_i^(-1/2).

All coefficients through g⁴ and both energy/norm endpoints remain separate.
Whole-parent tests cover both nucleons, both helicities, cutoffs 0/1 and
distinct incoming/outgoing energies. A cutoff-2 omission witness detects
lost contact-current matrix elements. No current-amplitude threshold is used.
This does NOT yet transfer the transverse current to the interacting
QQ/Q6 common-order H2 model.

## What the covariance test actually found

For identically phased elastic external spinors at Delta+=0 and mean P_T=0,
the transverse Dirac and Pauli tensors coincide. Thus the transverse current
tests G_M=F1+F2, while J+ gives two apparent coefficients F1', F2'.
Conservation alone does not require these determinations to agree.

At K2=9, retained cutoff 0/37 states, b=.4 GeV, mq=.35 GeV, mg=0,
P+=1.3 GeV and Delta=(.12,.16) GeV, the separately solved free-Q5 reference gives:

| Target/control | Q cutoff | g | M² (GeV²) | F1'+F2' | transverse G_M |
|---|---:|---:|---:|---:|---:|
| proton | 1 | .3 | 1.421430409460545 | .882387273956286 | 2.644271748729026 |
| proton | 2 | .3 | 1.421428801867490 | .882386937907952 | 2.644266305326975 |
| neutron | 1 | .3 | 1.421430409460544 | -3.25043036245e-7 | -1.762848275401543 |
| free proton | 1 | 0 | 1.422500000000001 | .882496902584596 | 2.647490707753790 |

Raw-parent residuals are below 4.2e-15 GeV²; elastic continuity defects
are below 1.8e-17. The discrepancy is NOT repaired by enlarging Q from 1
to 2, nor introduced by the new contact matching. Omitting PQ/QP contacts
changes proton transverse G_M by about 1.68e-7, much less than the
order-one discrepancy. Small implementation residuals are not covariance.

The g=0 lowest state is a finite free-Galerkin pseudostate, not a physical
bound nucleon. Its equal-share S3 spin-flavor control has

    M² = 9 mq² + 2 b²,  D = exp(-Delta²/(2b²)),
    proton: F1'=D, F2'=0, G_M,T=3D,
    neutron: F1'=0, F2'=0, G_M,T=-2D.

These are analytic **negative covariance controls**, not physical targets.
A consistent unitary change of state AND current coordinates cannot change
the matrix elements. Changing only the spin wavefunction or rescaling a
current would introduce a new model and is not an authorized repair.

## Explicit orientation decomposition, not subtraction-as-renormalization

Use the five spin-half tensors of Carbonell et al., Eqs. (6.25),(6.33),
[primary review](https://arxiv.org/html/nucl-th/9804029v1).
The B1 basis is their stated orthogonality convention; it is not a theorem
that identifies a unique physical finite counterterm.
For eta=Delta²/(4M²):

    B1 = (G_M,+ - G_M,T)/2,
    F1_sep = F1' - 2 eta B1/(1+eta),
    F2_sep = F2' - 2 B1/(1+eta).

The electric combination F1'-eta F2' is unchanged. At the g=.3 cutoff-1
points B1 is -.880942237386370 for proton and +.881423975179254 for neutron.
The separated coefficients are (.894695344466674,1.749576404262352) and
(-.0123152887978973,-1.750532986603646). These remain **diagnostics**, not
physical form factors. All out-of-basis residuals are returned.

With only J+ and J_T, B2/B3 are unresolved and returned as None, never zero.
The exact free J- projection now retains <D kappa_j kappa_k> through N+2,
including spin flips, signed antiquark conventions and both target-spin
endpoints. Independent full-Dirac/polar-wave quadrature verifies it.
On the separate g=0 control all four current components resolve all five
coefficients:

| Target | B1 | B2 | B3 |
|---|---:|---:|---:|
| proton | -.882496902584597 | -.316730440465799 | .304322751150023 |
| neutron | .882496902584596 | .211153626977197 | -.223561316292974 |

The five-structure residual is below 6e-16. This demonstrates the diagnostic
decomposition, **not rotational-covariance restoration**. The interacting
minus current has not been supplied by the free control.

## Quantum scaling result

The same complex controlled-gate tree now has an explicit state-retraction
option: derive the horizontal tangent from the executed quantum gradient,
normalize psi+step*tangent, and recover tree angles by reverse subtree
probability accumulation. The implementation uses O(D) storage/time, no
dense Jacobian/metric, no exact eigenstate and no StatePrep. The original
parameter-addition update remains the default comparison.

Separate bare retained-pilot controls, not the interacting QQ/Q6 H2:

| D / qubits | updates / evaluations | residual (GeV²) | M² (GeV²) | current error / bound |
|---|---:|---:|---:|---:|
| 204 / 8 | 206 / 413 | 9.75310e-9 | 1.419475028604880 | 1.46e-12 / 1.11e-7 |
| 654 / 10 | 126 / 253 | 9.80381e-9 | 1.302526821563128 | 1.28e-12 / 6.72e-8 |

Both have zero padding. Fit times were 131.99 s and 336.19 s in these
executions; concurrent work means these are not isolated timing studies.
The 204-state original update separately required 292 updates/844 evaluations.
The older 654-state parameter-addition run was terminated after more than
107 minutes once the alternative completed. It produced no final state or
convergence result and is not counted as either converged or failed.
Full O(D) gate/parameter counts and classical state information remain:
this is not polynomial many-body compression, hardware evidence or advantage.

## Verification, reproduction and preservation

74 current/contact/matching/orientation tests passed together in 113.35 s.
30 minus/orientation tests passed separately in 17.10 s (15 overlap with
the 74); 44 tree/metric/optimizer tests passed separately in 9.80 s.
There are 89 new current tests and 19 new retraction tests.
A subsequent combined constrained/free-minus/transverse/covariance/tree
regression passed178 tests in58.85s at88fe087e. The six existing theory
consistency tests passed in0.08s after synchronization.
The complete named M2+Q0–Q2 run at preceding source 85717cae passed 1171
tests in 2199.75 s (1138 M2,33 Q,1885 inherited warnings). This is not a
new full-repository or all-current-source single-run claim.
The current M2 collection is1246 tests. The main note has a164-page
scratch render with448 unique labels and all explicit references/citations
resolved. New equations, tables, introductory status and current conclusion
passed visual review; there are no rendered ??, final overfull boxes or
missing glyphs. A generic inherited undefined-reference warning remains
despite the direct audit. No PDF deliverable was requested.

Reproduce with:

    python scripts/benchmark_m2_transverse_current.py --target PROTON --cutoff 1 --coupling .3
    python scripts/benchmark_m2_transverse_current.py --target NEUTRON --cutoff 1 --coupling .3
    python scripts/benchmark_m2_free_covariance.py
    python scripts/benchmark_m2_tree_geometry.py --cutoff 2 --probability-regularization 1e-20 --step-rule tree_state_retraction

Pre-edit theory/status, bibliography and solver files from 2f046f65 are in
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_current_covariance_and_retraction.tar`.
Original shared materials and the 22-volume archive are unchanged.
GitHub and the public review mirror have not been refreshed or pushed.

## Next scientific dependencies

The first target below is now implemented in
[the common-order current continuation](m2_fourth_order_transverse_result.md);
its covariance warning remains. The list records this checkpoint's path.

1. Extend canonical transverse contacts to Q5↔Q6, including antiquark
   emission/CAR signs; combine Q6 free currents and QQ/Q6 backflow in the
   same common-order g⁴ H2 map. Do not substitute the free-Q5 reference.
2. Finish interaction-dependent minus, ordered two-gluon and Wick/current
   partners; gauge the chosen Galerkin regulator and complete common-regulator
   loop/counterterm transport. Test orientation and spin covariance in
   addition to continuity, with retained and longitudinal resolution varied.
3. Establish shared physical finite conditions and input/holdout separation
   without fitting away an incomplete-current artifact. Then microscopic
   nucleon/nuclear spin-one/GTMD matching and uncertainty-controlled predictions.
4. Develop genuine topology-factorized many-body compression and measurement
   resources against the corrected exact references.

The observed covariance warning blocks a physical claim, not parameterized
operator development. No physical calibration or production current is selected.

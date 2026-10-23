# Common-regulated H2: elastic spin and resolution diagnosis

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Publication note, 9 September 2026: this result is included in the
user-authorized canonical/main and public-review refresh. The source marker
identifies its exact document export. Scientific source `6814ef5d` and every
numerical result remain unchanged; the no-push sentence at the end records
the original 8 September development checkpoint, not current publication.

## Outcome and scope

The same finite H2 current can now be examined at a self-consistent elastic
pole, with source-fixed relative spin phases, both endpoint norms, every
current family and every raw order through g4. The eight independent baseline
blocks (four proton, four neutron), fresh axis rotation and transverse boost
all pass their algebraic/kinematic checks. The two-structure spin-half current
test does **not** pass. Current conservation and successful quantum readout
must not be confused with restored dynamical covariance.

This is a classical analysis of the existing numerical physics kernels at
`520cc782` and quantum integration at `e3c02573`. No Hamiltonian or current
kernel, finite coefficient, physical input or quantum optimizer is changed.
The additive code and complete numerical evidence are committed at `6814ef5d`.
Original code and documents remain recoverable at `0e162ce1`. The original
six edited theory/pickup/test files are also preserved in the verified archive
`computation_handoff/backups/2026-09-08_pre_elastic_spin_notes.tar`. The original
22 volumes, bibliography and supplied theory archive are unchanged; this
continuation belongs in the living high-level note, not a new volume.

## Why the elastic calculation is new

The preceding quantum measurements used linearization anchors 1.39 and
1.47 GeV². Those are legitimate directed transition benchmarks, but not a
common on-shell mass. We now solve the lowest finite branch of

`H_eff(e*) u = e* u`, `u†N(e*)u=1`, `N=I-∂e H_eff`, `v=N^(1/2)u`.

The explicit bracket (1.3,1.5) GeV² is below all represented free Q poles.
It selects a local isolated finite-model root, not a calibrated nucleon mass
or a proof of a continuum bound state. An independently solved generalized
linearized eigenproblem and a finite-difference H derivative cross-check the
root and metric. Current amplitudes obey

`A_fi = u_f† C_fi u_i = v_f† W_f C_fi W_i v_i`, `W=N^(-1/2)`.

Both Jz fibers are solved independently. Their H, N and linearized-H matrices
intertwine under the actual source transverse-parity map T. The explicitly
positive-parity diagnostic uses `u_minus = i T u_plus`; equal dimensions and
arbitrary eigensolver signs do not identify spin fibers. This phase convention
does not prove total spin 1/2. The current's old on-shell five-tensor diagnostic
is used only with the new same-pole amplitudes.

The distinction is consistent with the on-shell/off-energy-shell discussion
in [Carbonell et al., section 6.3](https://arxiv.org/html/nucl-th/9804029v1).
The orientation-tensor separation is a convention, not a unique physical
subtraction. Both the two- and five-tensor residuals remain in the saved data.

## Baseline results

Parameters remain explicit: K2=9, retained P shell0, Q5/Q6 shell0, b=.4 GeV,
mq=.35 GeV, mg=0, g=.3, Pplus=1.3 GeV, Gaussian width .8 GeV, eight boson and
eight fermion modes, paired two-flavor Wick subtraction, finite quark/gluon
mass-squared coefficients .011/-.007 GeV² per g², electromagnetic charges,
the physical-leg endpoint-product, smooth fixed-membership and gapped spectral
prescriptions. Transfer is (.12,.16) GeV, mean transverse momentum zero.
No value was selected by fitting the following discrepancy.

| Quantity | Proton test flavor | Neutron test flavor |
|---|---:|---:|
| Elastic energy, GeV² | 1.4305035729036712 | 1.430503572903667 |
| Two-structure residual | .173199361558052 | .178296272184940 |
| Residual / current packet norm | .124192959008568 | .722905413914347 |
| Five-structure residual | 9.884e-16 | 5.629e-16 |
| B1 | -.881183679414918 | .881590653168025 |
| B2 | -.321046385916511 | .214031112080730 |
| B3 | .313574576149697 | -.226356555699205 |
| GM from plus (diagnostic) | .882404068833785 | 1.4329254e-8 |
| GM from transverse (diagnostic) | 2.644771427663623 | -1.763181292006795 |
| Elastic continuity norm | 1.559e-17 | 2.453e-18 |

These are incompatible diagnostic extractions, **not physical form factors**.
The residual norm is Euclidean/Frobenius in this fixed frame, not a Lorentz
scalar. Its ratio for the nearly neutral neutron packet is correspondingly
large. The five-tensor condition number is about 41.35. Pole residuals are
below 3.3e-15 GeV². Coefficient imaginary parts are roundoff.

The analysis saves each of 13 contribution families by five raw orders.
Their coefficient and residual vectors sum exactly to the complete result
at fixed kinematics; the norms of residuals are not added as scalars. In this
declared split, B1 comes from the one-body operator evaluated in the interacting
states with the correct norms. Canonical, Gaussian-field, Wick and oscillator
projector terms alter minus-sensitive coefficients but do not cancel B1.
This is attribution for this finite current representative, not an exhaustive
scan of admissible transverse-current prescriptions.

An intentionally incorrect Euclidean renormalization of raw u would change
the full packet by only 7.946e-7 / 1.405e-7 for proton/neutron. This demonstrates
that the dominant warning is not that small normalization error; the incorrect
normalization is not accepted as an alternative.

## Independent kinematic controls

The proton spin-flip current is freshly assembled at axis transfer (.2,0)
and at the original transfer with mean momentum (.13,-.26) GeV. Every full
matrix, family and raw order satisfies the source-phase z-rotation and
transverse-boost laws. These are kinematic tests, not the dynamical transverse
rotations needed to qualify a spin multiplet.
The largest entrywise family/order differences are 2.483e-16 for rotation
and 3.497e-18 for boost (stacked norms 7.952e-16 and 1.234e-17).

Parity/reversal/z-rotation completion of pm and mm from pp/mp is checked against
all four independently assembled baseline blocks for both flavors, family by
family and order by order, and against independent external Dirac tensors at
three transfers. This map is used for the Q5 sensitivity study only with an
explicit label: two independent blocks plus two kinematically completed blocks.
It is not valid by default for the older unequal-anchor transitions.

## Free-state and P/K resolution controls

At g=0 the independent free-only carrier contains the exact H0 and source
coordinates. It does not construct or silently zero missing interacting
coefficients. Q returns and Wick/contact terms have positive coupling order;
the exact g0 current is physical one-body plus retained-projector transport.
All four raw g0 matrices agree with the fresh interacting exports.

| Free proton control | Dimension per Jz | Ground rank | Energy, GeV² | B1 | Two-structure residual |
|---|---:|---:|---:|---:|---:|
| K9/P0 | 37 | 1 | 1.4225 | -.882496903 | .173663404 |
| K9/P1 | 204 | 1 | 1.4225 | -.882496903 | .173948077 |
| K9/P2 | 654 | 1 | 1.305371871 | -.796953463 | .162906495 |
| K11/P0 | 69 | 3 | 1.487833333 | not selected | not fitted |

The relation `B1 = -Aplus(++)` persists through P0/P1/P2. A changing absolute
overlap is not covariance restoration. The K9/P0 neutron has residual
.178712403 and B1=+.882496903. The proton free projector changes B3 by
.006016762 at P0 (P1: .003686194; P2: .008421208), not B1. The neutron ground
projector amplitude vanishes to roundoff.

The K11 rank-three ground subspace and every current block on it are saved.
No arbitrary vector is chosen to manufacture a pair of nucleon helicities.
These free Galerkin states are pseudostates, not bound physical nucleons;
this warning alone neither diagnoses a canonical-current bug nor uniquely
determines a Melosh factor or finite counterterm.

## Finite-reference and Q sensitivity

The prior unequal-anchor full-operator P/Q/K and degree-one/two finite-reference
records remain valid and unchanged. In particular, Q5 shell1-to2 changes the
raw g4 coefficient by 20.7% despite a .0101% normalized-total change. They are
not relabeled as new elastic state/form-factor convergence calculations.

For fixed external tensors S and fixed states/coordinates, an amplitude error
bound gives `||δc|| <= ||S^+|| ||δA||` and `||δr|| <= ||δA||`. Changed states,
norms, pole energies or source domains require additional terms. The new
same-coordinate Q comparison explicitly telescopes the amplitude change into
raw numerator, both norm maps and normalized-state contributions. A separate
term records the change of the external mass-dependent tensor basis. This
telescoping convention is exact but not a unique physical decomposition.

The new proton Q5 shell0-to1 comparison holds K9/P0/Q6-shell0 and all
prescriptions/parameters fixed, solving the pole again at each resolution.
The shell-one pp/mp calculations finish in 1764.68/1759.42 s. Their remaining
two helicity blocks are explicitly kinematically completed, not independently
rebuilt. The elastic energy changes by -5.5190819e-7 GeV² to
1.4305030209954839 GeV². The two-structure residual changes only from
.173199361558052 to .173199009410862; B1 becomes -.881181255653547.
Five-structure and continuity residuals remain roundoff. Small contact B1
contributions are now nonzero but do not cancel the dominant one-body term.

The full packet change has norm 7.9685860e-7. The three exact telescoping
pieces have norms 3.7045275e-7 (raw numerator), 3.0710303e-7 (both norms) and
8.3888059e-7 (normalized state change). These are vector contributions with
cancellations, **not** additive scalar error bars. The B1 change 2.4237614e-6
is the sum of 6.7676077e-7, -9.2625224e-8 and 1.8396259e-6 in the baseline
mass tensor basis; the remaining mass-basis B1 change is roundoff.
The external mass-basis changes of B2/B3 are separately 4.4826231e-7 and
-2.8092326e-7. All family/raw-order arrays reconstruct the exact difference.

This is one same-pole Q sensitivity step, not Q convergence, an elastic
Q5-shell-two calculation or physical-current prescription selection. Data:
`m2_cluster_spin_Q5_shell1_analysis.json/.npz` and
`m2_cluster_spin_Q5_resolution.json/.npz`. Every calculation launched for this
diagnostic is now terminal; no old benchmark needs to be restarted.

## Reproduction and saved evidence

The scripts are additive; previous benchmark outputs are not overwritten.
Use the existing project environment with `PYTHONPATH=src:.` and one BLAS
thread. Fresh full-current blocks take minutes; replay requires seconds.

```text
python -m scripts.benchmark_m2_cluster_spin --target PROTON --out 1 --incoming 1 --delta .12 .16 --output NEW.json
python -m scripts.benchmark_m2_cluster_free_spin --K2 9 --cutoff 0 --target PROTON --output NEW.json
python -m scripts.analyze_m2_cluster_spin --blocks PP.json PM.json MP.json MM.json --output NEW.json
python -m scripts.analyze_m2_cluster_spin --blocks PP.json MP.json --symmetry-complete --output NEW.json
python -m scripts.analyze_m2_cluster_spin_resolution --old OLD_PP.json OLD_MP.json --new NEW_PP.json NEW_MP.json --output NEW.json
```

The saved families under this directory are `m2_cluster_spin_{p,n}_{pp,pm,mp,mm}`,
the `p_mp_axis` and `p_mp_boost` controls, `m2_cluster_free_spin_*`, and the
proton/neutron analysis reports. JSON records explicit parameters and adjacent
numeric-only NPZ paths. NPZ preserves full raw family/order matrices, both
H/N/W/L matrices and states, and the source parity map. The free degeneracy
archive retains the complete ground subspace. No pickle is enabled.

## Verification

The final focused regression passes 192 tests in 76.46 s, including all
53 new pole/phase/symmetry/free-limit/replay/Q-attribution tests. Independent
synthetic complex inputs verify the attribution algebra with scale-aware
roundoff checks; the actual saved packet also satisfies the separate strict
absolute check. M2 collection finds 2966 tests in 22.89 s; this is not a
full-repository test run. No hardware or new quantum execution is claimed.
All six theory/status checks pass in 0.12 s. The living TeX compiles offline
to 224 scratch pages; changed summary, equations, table, Q-attribution and
status pages are visually checked. No final-pass overfull boxes or literal
unresolved-reference markers are found. Inherited package/bibliography/rerun
warnings remain. This is focused layout QA, not certification of every earlier
physics statement, and no PDF deliverable was requested.

## Next constructive scientific step

Derive the light-front dynamical rotation / Pauli-Lubanski spin action from
the same canonical global-Jacobi dynamics and project/eliminate it consistently
with H2. First establish its free-source action and projection leakage, then
the interaction-dependent partners and metric-correct spin algebra/multiplets.
Retain complete degenerate subspaces. Test the current against those states
before making a physical nucleon or rotational-covariance claim.
For exact parent generators X,Y, the finite-source commutator differs from
the projected parent by `-P X Q Y P + P Y Q X P`. The acceptance is to identify
and control these terms and longitudinal-domain boundaries, not to require
exact continuum algebra after discarding leakage. This is an unimplemented
next construction, not a result of the present spin fit.

Do not impose an unexplained Melosh rotation, fit away B coefficients, reopen
the selected canonical/H2-first direction, or restart completed matching or
quantum benchmarks. The present mismatch does not block that parameterized
construction. Physical current/finite-condition selection and calibration
covariance remain later decisions with the theory lead; deuteron spin-one/GTMD
matching and genuine many-body topology/quantum scaling remain downstream.
No new public push is made; the review mirror release remains `db7bb7c6`.

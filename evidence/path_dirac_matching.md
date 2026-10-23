# Canonical Dirac dynamics versus spin-blind path coordinates

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

11 September 2026. Bounded canonical-correspondence step of the shared
binding/matching review, pre-evaluated from `27bdf01b`. Eight current
project/theory/bibliography files are preserved in
`computation_handoff/backups/2026-09-11_pre_path_dirac_matching.tar.gz`.
The archive contains eight files total, including the bibliography.

## Scientific result and decision

The ordinary color Wilson paths qualified previously are valid state
operators in their stated scope, but **not a complete Dirac interaction
representation**. Even allowing arbitrary complex coefficients for every
retained gluon mode, a spin-blind path image misses 9.4116% of the canonical
first-order emission-state correction at the reference point. A spin-diagonal
image still misses 3.0438%. These percentages are norms of a finite
operator/state correction, not errors in proton observables.

Keeping the full spin matrices removes the mismatch in the original small
domain. However, a four-path interpolation that is complete on two spatial
modes misses 68.8169% after adding a third mode. This is not a robust
physical state basis by default. The chosen finite paths require large
cancelling coefficients (up to 136.9 at m=.27) even in the passing domain.

**Use the already qualified canonical Dirac vertex at interaction nodes.**
Let geometric coordinates organize state correlations and explicitly
qualified residual structures. Do not ask scalar path weights to relearn
known spin dynamics, promote the Pauli coordinates to unrelated physical
couplings, or choose a production confinement interaction from this test.
No old kernel or saved p/n state is changed. This is an action/state
correspondence test, not another Wilson-order expansion or BLFQ repair.

The distinction between color transport and spin-field coupling is
established physics, not a novelty claim. For related prior art, the QED
worldline construction of [Gies and Hämmerling](https://arxiv.org/abs/hep-th/0505072)
adds the spin information through a Pauli-term spin factor. We use our
preserved canonical LF QCD vertex directly, not their worldline algorithm
or an assumed nonperturbative spin factor. The project's result here is the
explicit finite-state mismatch, operator consequence and architecture test.

## 1. Complete normalized canonical target

Use the preceding LF cell, gluon spatial modes, full colors/helicities and
positive quark recoil. The emitter momentum is `(1.875,.2,-.2) GeV`; gluons
are `(.25,.2,0)` and `(.5,0,.2) GeV`. Both parent helicities are independent
columns of the tested map. Spectators fix the baryon color interpretation
but are not a new bound state or complete p/n wavefunction.

For `a=p-k`, `m=(k,hg)`, define the gs/color-stripped, cell-normalized
canonical P-minus vertex, with dimensions GeV,

\[
 V_m^{h'h}={-\bar u(a,h')\,\not\!\epsilon_{h_g}^{*}(k)\,u(p,h)
            \over\sqrt{8a^+k^+p^+V_{\rm cell}}}.
\]

Its open color action is `V_m tensor t_a`. Three calculations agree:
the preserved canonical leaf, an independent Cartesian Dirac-component
contraction, and the independently evaluated constrained good-field form

\[
 V_m={1\over\sqrt{2k^+V_{\rm cell}}}
 \left[-\epsilon_m^{-*}I+
 {1\over2}G^\dagger\left(
 {B_0(a)\alpha_T\cdot\epsilon_{mT}^{*}\over a^+}
 +{\alpha_T\cdot\epsilon_{mT}^{*}B_0(p)\over p^+}\right)G\right],
 \quad B_0(q)=\alpha_T\cdot q_T+\beta m_q.
\]

`G` contains the constant normalized good-spinor columns in order (-,+):
`(0,1,0,-1)/sqrt(2)` and `(1,0,1,0)/sqrt(2)`. The constraint route uses no
full Dirac spinor or cubic helper. It retains epsilon-minus and both
momentum denominators, not just a transverse scalar path coupling. At
m=.27 GeV its full-map error is `1.815e-18 GeV`; the Cartesian route gives
`2.233e-18 GeV`. Complete raw color tensors agree with the earlier normalized
canonical cubic tensor, including its generator order.

The free gap is derived from actual daughter energies:

\[
 \Delta_m={|k_T-xp_T|^2+x^2m_q^2\over2p^+x(1-x)},\quad x=k^+/p^+,
 \qquad \chi_m=-V_m/\Delta_m.
\]

Here chi is the finite first-order stationary emission correction, not a
solved physical nucleon state or a new bare interaction. The reference gaps
are `(.0739651282,.0739651282,.0984630303,.0984630303) GeV`.

## 2. Geometric and spin ranks are separate

The first-order image of ordinary paths is `L_Pm=i F_Pm I_spin`. Its best
possible completion, even with arbitrary mode coefficients, is

\[
 \chi_m^{\rm blind}={\operatorname{tr}\chi_m\over2}I,
 \qquad
 V_m=\sum_{\alpha=0}^3v_{m\alpha}\sigma_\alpha,
 \quad v_{m\alpha}={1\over2}\operatorname{tr}(\sigma_\alpha^\dagger V_m).
\]

This is a statement about the FULL map from both parent helicities, not a
separate fit for each spin state. Pauli coefficients are basis coordinates
of the fixed Dirac operator; they are not four freely chosen physical
couplings. Under changes of incoming/outgoing spin bases, transform both
the target and its coordinate image. Resetting the transformed image to
an identity by hand would change its meaning.

The diagnostic spin-aware image uses `(U_P-I)*sigma_alpha` at first field
order, so its bare term is zero. No second-order normalized decorated
Wilson state or source completion is inferred. Four declared closed paths
span the four complex mode/helicity profiles in this test; the previous
three paths have only rank three. At m=.27:

| Tangent coordinates | Complex map rank | Relative chi error | Positive variational deficit, GeV |
| --- | ---: | ---: | ---: |
| Four paths, identity spin tensor | 4 | .0941156 | 1.01754e-5 |
| Four paths, identity and sigma3 | 8 | .0304378 | 1.10511e-6 |
| Four paths, all Pauli tensors | 16 | below 2e-14 | below 3e-31 |

The sixteen singular values lie well above the numerical null threshold;
the largest-to-smallest ratio is about 69.65. All values and coefficients
are saved. No nonzero target direction is deleted. Duplicating a path does
not increase rank or reduce the spin-blind error.

The failure is not only a finite quark-mass helicity-flip effect. At m=0,
the off-diagonal spin terms vanish to roundoff, but sigma3 remains nonzero
and the largest spin-blind image still misses 9.0013% of chi. At m=.54,
the error is 10.4817%, and the diagonal-spin restriction misses 5.8661%.

The original three-path, spin-complete image misses 73.9361% of chi. Adding
the fourth path cures this particular finite interpolation, but increasing
to three spatial gluon modes gives a 24-dimensional map while this image
still has rank16; the residual grows to 68.8169%. This is why the passing
small-domain interpolation is not selected as a binding-state architecture.
The preserved canonical node, with its actual free propagation factor,
supplies `-V/Delta` on the enlarged domain without fitting path weights.
That reuse is not an efficiency benchmark or a one-parameter nucleon solve.

## 3. Common energy and spin-operator consequences

For the same bare reference, let `C0` denote its COMMON unknown expectation
of all order-gs^2 interaction/self/counterterm terms. In the spin-averaged
color-singlet metric, the normalized trial energy coefficient is

\[
 E_2[\chi]-C_0={C_F\over2}\sum_m
 \operatorname{tr}[\chi_m^\dagger\Delta_m\chi_m+
                   \chi_m^\dagger V_m+V_m^\dagger\chi_m],\quad C_F=4/3.
\]

Completing the square gives the positive, parameter-independent comparison

\[
 E_2[\chi_L]-E_2[\chi_*]
 ={C_F\over2}\sum_m\Delta_m\|\chi_{Lm}-\chi_{*m}\|_F^2,
 \qquad\chi_*=-V/\Delta.
\]

The code minimizes that quadratic form in the actual weighted image;
independent small Hermitian reference eigenvalues verify its weak-coupling
limit. The common C0 cancels in the comparison, **not in the physical
theory**. The value `E2[chi*]-C0=-.001044084666 GeV` is therefore not a
mass prediction, full self-energy or confinement evidence. No omitted
counterterm, canonical quartic, source or other-emitter term is set to zero.

A normalized reduced quark-spin diagnostic follows from the same amplitudes:

\[
 \delta\rho=C_F\sum_m\left[
 \chi_m\rho\chi_m^\dagger-\tfrac12\{\chi_m^\dagger\chi_m,\rho\}\right].
\]

It is trace preserving at this order. This uses the common Hermitian
normalization convention, not a Markov/noise model or complete QCD axial
observable. Opposite-input averaged retention along a prepared spin axis
has the invariant loss coefficient

\[
 \ell_a=-\tfrac12\operatorname{tr}[\sigma_a\delta\rho(\sigma_a)]
 ={C_F\over4}\sum_m\|[\sigma_a,\chi_m]\|_F^2.
\]

A common order-two unitary spin-frame rotation does not change these
same-axis losses. An independently normalized finite isometry reproduces
the coefficient. For full chi at m=.27 GeV,
`ell_(x,y,z)=(.000225132635,.000225132635,.0000248467652)`; every spin-blind
image gives zero. At m=0 the longitudinal loss vanishes but the transverse
losses remain `.000238347152`. Thus charge/norm preservation and a small
energy deficit do not certify the spin information needed for observables.
These are constituent LF spin diagnostics, not nucleon magnetic or axial
form factors, and do not repair the old covariance discrepancy.

## Validation and recovery

18 new tests pass with warnings as errors (0.77 s), covering all map columns,
three quark masses, independent good constraints, full color normalization,
weighted stationarity, weak-coupling eigenvalues, spin operators, spin-basis
changes, longitudinal AND transverse LF kinematic boosts, duplicates, zero
target and mode enlargement. Dynamical rotations/local Gauss law remain open.
One newly written absolute 2e-18 trace tolerance encountered 2.31e-18
roundoff; the trace/Hermiticity/spin-algebra checks now use a norm-scaled
16*machine-epsilon bound. No old test or physical threshold was loosened.

The source is `src/deuteron_wigner/quantum/path_dirac_matching.py`; run
`scripts/evaluate_path_dirac_matching.py` with the qualified local Python
runtime and `PYTHONPATH=src:.`. The script prints but does not write files.
Its saved output is `docs/next_level/path_dirac_matching_results.json`.
Final integration passes 157 focused tests with 12 existing warnings
(6.26 s), including preceding path/color/graded/canonical tools and the
theory contract. The replay is reproduced with roundoff-sized diagnostics
separately bounded. All eight archived files match 27bdf01b byte-for-byte.
The new TeX section has balanced structure and six unique labels; the added
spin-factor reference was checked against arXiv and its journal record.
This is not a full historical-suite or whole-manuscript audit, and no PDF
or public review export was produced. Final hashes are in the handoff/index.

## Reassessment and next action

The first short-distance-correspondence condition in the binding review
is now explicit: **preserve the complete canonical spin/color tensor at
interaction nodes**, using geometric transport as a separate structure.
An equivalent geometric encoding of spin is not excluded: it must reproduce
that complete map. The negative result concerns the explicitly spin-blind
or spin-diagonal restrictions, not quantum geometry in general.
This resolves the spin-blind-coordinate limitation without new physics or
an inherited method-repair campaign. The scalar residual coefficient
functions must multiply qualified Dirac/flavor/color multiplets, not stand
in for them. Pauli image completeness at one LF resolution does not
establish a covariant or chirally complete residual action.

Continue the original shared-residual question from
`docs/next_level/joint_state_dynamics.md`: compare a declared explicitly
resolved soft correlation with its consistently source-gauged residual
representation through common amplitudes and currents, subtracting retained
content once. Reuse the full canonical nodes and the already qualified
source partners. Before any binding fit, make the candidate's chiral/Dirac
content, causal realization, asymptotic/cluster channels and matching domain
explicit. Do not extend path interpolation, fit away the spin mismatch,
assume a new confining vacuum or identify the two-O(4) reference as production.
If a physical realization must be selected, surface that decision. Binding,
physical p/n/current covariance and nuclear/GTMD matching remain open.

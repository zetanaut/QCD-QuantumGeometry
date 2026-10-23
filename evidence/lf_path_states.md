# Actual LF path-state images and common operators

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

11 September 2026. Pre-evaluated from the binding/color-geometry checkpoint
at Git `67ec5a08`. This is a **second-field-order state construction**, not
a selected interaction or physical proton/neutron solution. Seven original
current/theory documents are preserved in
`computation_handoff/backups/2026-09-10_pre_lf_path_states.tar.gz`.

## Outcome and architectural decision

Geometric paths can produce additional, distinct quantum states using the
**already retained gluons**. They need neither fictitious orthogonal topology
labels nor a duplicate flux Hilbert space. The resulting states overlap;
their full complex Gram and operator matrices must be carried together.
Path ordering produces important two-gluon color correlations. A norm test
at this order does not see their deletion; a quadratic field operator does.

This advances a physical-image component of the proposed network, not its
binding dynamics. A path dressing is a state operator, **not a Hamiltonian**.
Its free kinetic cost is positive in this example; this says nothing about
the sign of the omitted interacting QCD contributions. No confinement law,
nonperturbative vacuum, local Gauss-law solution or nuclear force follows.
The paths are continuously shape dependent, not quantized topological sectors.

The implementation uses the preserved canonical LF polarizations, finite
field normalization, SU(3) generators and exterior-algebra reference. Wilson
ordering and state overlaps are established structures, not a novelty claim.
The project-specific result is their explicit recoil/color/Fock image and
common operator test. As an external scope check, the static three-quark
Wilson-loop work explicitly distinguishes a path-created state from the
ground state and its excited-state contamination
([Takahashi et al., section 3](https://arxiv.org/html/hep-lat/0204011v1)).
We do not import its quenched lattice action, static potential or algorithm.

## 1. Declared state carrier and quantum counting

Coordinates and momentum use the preserved symmetric convention,
`r=(x-minus,x,y)` at fixed `x-plus=0`, and `k=(k-plus,kx,ky)`.
The cell has lengths `(8*pi,10*pi,10*pi) GeV^-1`, hence spacings
`(.25,.2,.2) GeV`. Fermions are antiperiodic longitudinally, bosons periodic;
all modes have positive plus momentum. The two retained gluon spatial modes
are `(.25,.2,0)` and `(.5,0,.2) GeV`, with BOTH helicities and all eight
colors: 32 distinct gluon oscillators. All normalized occupations with zero,
one or two gluons are retained: 561 configurations, each with 27 raw quark
color coordinates, or 15,147 complex coordinates per field-order wave.

Three quarks have momenta `(1.875,.2,-.2)`, `(.375,0,.2)`,
`(.625,-.2,0) GeV` and the illustrative mass `.27 GeV`. Their three distinct
spin/flavor labels may be `u-`, `d-`, `u+`. The epsilon/sqrt(6) color seed is
globally singlet, but these fixed labels are **not** the project's complete
spin/isospin proton doublet. The engine does not replace that saved doublet.
Every color tensor embeds in the actual exterior quark space with its exact
fermion signs; the distinguished slot is not a distinguishable physical
quark species. Orthogonal spin/flavor labels make this small test embedding
unambiguous. General coherent spin/flavor wavepackets and multi-anchor
baryon dressings remain a later composition, not an assumed tensor product.

Only the distinguished slot is dressed. Define a recoil/color/boson operator
`R_(m,a)` that creates a normalized gluon `(k_m,h_m,a)`, acts with `t_a`
on that quark's color and shifts its momentum `p -> p-k_m`. Its adjoint
annihilates that gluon with the reverse shift. The usual sqrt(n+1) factors
are retained. Every configuration conserves the **same total plus and
transverse momentum**. Positive recoil is checked for every repeated and
distinct two-gluon pair, not just for nonzero output amplitudes. The engine
rejects a parent that would reach nonpositive plus at this order.

The recoil follows by Fourier transforming a translated path `x+r(s)`:
the field's creation phase `exp(+ik.x)` shifts the quark from P to P-k.
The normalized good-component spin basis is momentum independent:

\[
 {\sqrt2\,u_+^\dagger(p',h')u_+(p,h)\over2\sqrt{p'^+p^+}}
 =\delta_{h'h}.
\]

That off-diagonal-momentum identity, not an adjustable quark normalization,
is checked independently with preserved full Dirac spinors. This constructs
a number-preserving quark-line dressing. It is not a QCD interaction vertex
or a claim that dynamical pair creation is absent from QCD.

## 2. Canonical field and ordered state coefficients

For `D=partial-i gs t.A`, transport from the path's source to destination is
`P exp(+i gs integral A_mu dr^mu)`. This polyline lists vertices in traversal
order, **not** the old graph transport-index convention. At fixed x-plus,
`A+=0` removes the longitudinal line contribution and the transverse metric
gives the minus sign in

\[
 f_m(s)={-\dot r_T(s)\cdot\epsilon_{h_m,T}^{*}\over
                 \sqrt{2k_m^+V}}\,
 e^{i[k_m^+r^-(s)-k_{mT}\cdot r_T(s)]},\qquad
 B(s)=\sum_{m,a}\{f_m(s)R_{ma}+f_m^*(s)R_{ma}^\dagger\}.
\]

`B(s)` is a Hermitian dimensionless path generator. The path and finite
gluon mode list are declared inputs, not a physical cutoff/matching claim.
The dressing acts on the transverse perturbative Fock vacuum:

\[
 |\Psi_P\rangle=|b;0\rangle
  +g_s|\psi_{P1}\rangle+g_s^2|\psi_{P2}\rangle+O(g_s^3),
 \quad U_P=1+ig_s\int B-g_s^2\int_{s>t}B(s)B(t)+\cdots.
\]

Define `F_m=int f_m`, `I_mn=int_(s>t) f_m(s)f_n(t)` and
`J_m=int_(s>t) f_m*(s)f_m(t)`. With `j=(m,a)`, `l=(n,b)` and
normalized boson occupations,

\[
\begin{aligned}
 \psi_{1;j}&=iF_m t_a b,\\
 \psi_{2;0}&=-\sum_mJ_m\sum_a t_a^2b
             =-C_F\sum_mJ_m\,b,\quad C_F=4/3,\\
 \psi_{2;jl}&=-[I_{mn}t_at_b+I_{nm}t_bt_a]b\quad(j<l),\\
 \psi_{2;jj}&=-\sqrt2 I_{mm}t_a^2b.
\end{aligned}
\]

The momentum carried by the quark is p, p-k_m or p-k_m-k_n respectively.
Neither the imaginary vacuum term nor the two-gluon term is dropped.
The identities `I+I^T=F F^T` and `2 Re J_m=|F_m|^2` are derived by dividing
the integration square into its two ordered halves. Consequently
`2 Re <b|psi_2;0> + ||psi_1||^2=0`. Bare plus one-gluon states alone are
not normalized at order gs^2. Normal ordering the exponential without its
contractions changes this state operator.

For the LF-detour loop in the saved replay:

- one-gluon coefficient norm squared: `0.000644434203068733`;
- vacuum gs^2 amplitude: `-0.000322217101534367 - 0.000221134359783610 i`;
- two-gluon gs^2 coefficient norm: `0.002964536284790385`.

These are expansion coefficients, not all-order probabilities. Replacing
path ordering by the exponential of the integrated field changes the
second-order wave by `0.00295745328128`, including a two-gluon error
`0.00294917437699`. A scalar phase correction cannot remove the latter.
Reversal obeys the operator-adjoint identity, including its second-order
terms; it does not flip a process label. Concatenation includes all
cross-path contractions, not just a sum of the two individual waves.

## 3. Common Gram and operators

For any represented operator O, the same state image supplies

\[
 O^{[n]}_{PQ}=\sum_{r=0}^{n}
 \langle\psi_{Pr}|O|\psi_{Q,n-r}\rangle,\quad n\leq2,
 \qquad G^{[n]}_{PQ}=O^{[n]}_{PQ}\big|_{O=I}.
\]

For this single-anchor seed,
`G2_PQ=C_F[sum F_P* F_Q - sum J_P* - sum J_Q]`.
The full complex off-diagonal phases survive. Four tested path columns
are independent in the declared polynomial state image, but strongly
overlapping. Their singular values at the illustrative gs=.6 are
`(1.99992060,.01439089,.01047103,.00190378)`. Repeating a path gives the
same state, Gram row and operator row, not a new topology eigenstate.
The full polynomial Gram is positive by construction; its gs^4 pieces
are **not** a qualified fourth-order physical result. No eigenvalue
threshold deletes small but nonzero path information.

The free P-minus operator includes the actual recoiled quark and gluons.
It is not a string-length energy. Subtracting the seed energy E0 times the
Gram gives the positive-semidefinite matrix

\[
 (H_0^{[2]}-E_0G^{[2]})_{PQ}
   =C_F\sum_m F_{Pm}^*F_{Qm}\Delta_m,\qquad
 \Delta_m={|k_{mT}-x_mp_T|^2+x_m^2m_q^2\over
                  2p^+x_m(1-x_m)},\quad x_m=k_m^+/p^+.
\]

The LF-detour diagonal is `5.874627378768e-5 GeV`, the coefficient of gs^2,
with E0=`.2816266666666667 GeV`. Independent Cartesian four-vector sums
verify P-minus. **Canonical QCD cubic interference, quartic/self terms,
matched residuals and currents are not included in this free operator**;
they are not set physically to zero. The free cost neither proves a
confining tension nor rules out attraction in a qualified interacting action.

Deleting the two-gluon wave changes neither G through order two nor its
diagonal normalization. But the Hermitian quadratic probe `B(F_other)^2`
loses `5.508831640618e-8` in its gs^2 expectation. Its vacuum/two-gluon
matrix elements and one-gluon terms come from the same oscillator/color
operator; it is not an independent fitted head or an EM current.
This is why the architecture must preserve operator-sensitive directions,
not certify its representation using normalization or energy alone.

## 4. Independent controls and resolution warning

An independent sparse normalized-occupation calculation evolves the two
Dyson coefficient waves along the path. It constructs the field directly
from its plane waves and does not use the ordered-integral routine.
The full-wave error is `6.274302244762e-12` at tolerances `(2e-10,2e-12)`
with 188 right-hand-side evaluations. The coarser `(2e-8,2e-10)` result
differs by `1.727281544917e-11` (140 evaluations). Outer Gauss order 16 to
32 changes the wave by `8.291432169047e-19`. The two routes share the
preserved generators/polarizations, which also have direct phase, field-
factor, global SU(3), full two-gluon, exterior-sign and good-spinor tests.

Adding the spatial mode `(.25,.2,.2)` with both helicities preserves the
old creation coefficients to roundoff but adds one-gluon information:
the norm coefficient grows to `0.000936991735165476`. The old zero-gluon
gs^2 amplitude MUST change by
`-0.000146278766048371 + 0.000302379514034575 i`.
Its real change cancels the added one-gluon norm. Keeping the old vacuum
coefficient unchanged would violate order-two normalization. This is a
finite-domain comparison, not a continuum or flux-vacuum convergence claim.

19 final-source tests pass with warnings as errors (2.84 s). The focused
integration including prior color/graded/canonical tools, the binding
preflight and theory contract passes 139 tests with 12 existing warnings
(6.13 s). Replay values reproduce with roundoff-aware comparison. All seven
archived source originals match 67ec5a08 byte-for-byte; one additional
macOS metadata member is preserved. New TeX structure and six labels pass.
The bibliography is unchanged. This is not a full historical suite or
whole-manuscript re-audit; no PDF rebuild is claimed.
No old physics test, kernel or tolerance was changed. A newly written
bitwise cross-domain equality was made norm-scaled after the independent
replay exposed a 4.9e-18 floating-point layout difference. No physical
information was dropped or condition relaxed.

## Reassessment and next bounded question

**Advance the physical state representation; do not promote it to a model.**
The cheap path-state question has an affirmative but limited answer:
real gluon correlations can support a nonorthogonal quantum-geometric
representation. The global color, recoil, normalization and operator maps
are explicit. No extra flux register is required for these correlations.

Return now to the **shared binding/matching realization review**. Use these
states as candidate variational coordinates, not a substitute for an action.
The next pre-evaluation should identify which proposed shared soft operators
can act on this carrier with (i) canonical/short-distance correspondence and
full-minus-retained subtraction, (ii) consistently derived transverse as
well as longitudinal source responses, (iii) nonperturbative binding and
the correct physical asymptotic/cluster channels, and (iv) chiral and
dynamical covariance qualification. Distinguish qualified structures,
parameterized uncertainties and genuine physical assumptions. Compare the
smallest decisive candidate test before any fit; do not assign independent
path energies, declare a flux vacuum, restart BLFQ/H2 or extend this Wilson
series indefinitely. If passing those conditions requires an unselected
physical realization, surface that decision instead of a new repair campaign.

The finite A+=0 carrier does not implement arbitrary local gauge
transformations or constrained zero modes. A pure longitudinal path is
trivial **only in this reference**, not a proof that all gauge links can
be removed. A future/past GTMD staple and its rapidity/soft subtraction are
different operator data. The canonical modes are quantized once; no
previous loop is reused as a second bare mode. Main theory/roadmaps are
synchronized, older results and the bibliography retained, local only.

## Reproduce

From the canonical checkout with the qualified local Python runtime:

```text
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /private/tmp/deuteron-canonical.zxUoKe/env/bin/python scripts/evaluate_lf_path_states.py
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /private/tmp/deuteron-canonical.zxUoKe/env/bin/python -m pytest -q -p no:cacheprovider \
 -W error tests/test_lf_path_states.py
```

The replay prints results only; the preserved output is
`docs/next_level/lf_path_states_results.json`. No public push/review export,
production parameter fit or unrelated user-file edit was made.

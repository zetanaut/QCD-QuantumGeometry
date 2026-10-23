# Canonical LF physics in the new operator network

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. Developed from `f509938c2100e5352f1ce13b7ba1e43093dc61ad`.

## Outcome and governing boundary

The reusable canonical field algebra now has a model-independent entry point.
Quark emission, antiquark emission, gluon-to-pair conversion and gluon splitting
have explicit momentum/spin/color tensors and graded Fock-operator actions.
The free quark/antiquark electromagnetic four-current uses the same compiler.
A companion complete **bare qg → qg tree** independently agrees between
covariant and LF propagating-plus-instantaneous contractions, including
external-gluon Ward tests and transformed frames.

This is useful physical infrastructure, **not a selected nucleon Hamiltonian**.
No oscillator basis, fixed longitudinal K, Gaussian regulator, H2/Feshbach
sector elimination, old pole/ground-state ansatz or physical calibration is
imported into the new calculation. The package no longer eagerly imports the
old model when this entry point is loaded; its legacy public APIs remain
available on demand. Existing BLFQ/H2 results and their failures are preserved.

Do not infer that BLFQ as a whole has been disproved by our specific candidate's
failure. The user has rejected making reconstruction/repair of that candidate
the current project route. Established canonical QCD remains a foundation.

## What was reused, and what is new

- [Canonical field utilities](../snapshot/src/deuteron_wigner/quantum/canonical_lf_fields.py):
  exact extraction of ten existing functions, plus their gamma/projector
  definitions. Function syntax trees match the parent revision unchanged.
  Old imports re-export the same function objects. No phase convention was
  silently replaced, including the physical antiquark labels.
- [Graded Fock compiler](../snapshot/src/deuteron_wigner/quantum/graded_fock_network.py):
  new ordered ladder words, independent occupation/sign evaluation and
  Jordan–Wigner tensor-product compilation. Sparse state actions do not
  allocate the formal full register. The prior p/n exterior algebra supplies
  an independent composite-creation reference.
- [Canonical network kernels](../snapshot/src/deuteron_wigner/quantum/canonical_lf_network.py):
  explicit momentum-space cubic tensors, reused free EM current and the
  complete qg tree comparison with signed internal plus momenta.
- [Legacy constrained-current algebra](../snapshot/src/deuteron_wigner/quantum/m2_constrained_fermion_current.py):
  only its gamma import changed. This continuation uses its free four-current;
  its existing interaction-current formulas are retained, not claimed to be
  a complete matched interacting current for the new model.

The canonical theory and instantaneous terms are established physics, not an
original discovery. Their source is the canonical light-cone-gauge formulation
of [Srivastava and Brodsky](https://arxiv.org/html/hep-ph/0011372v2), especially
the interaction Hamiltonian and constrained fields. Our contribution here is
the project-specific separation, representation and independent verification
within the emerging modular architecture. No new physical topological order,
trained state or quantum advantage is demonstrated.

## Conventions and mathematical interfaces

Use p±=(p⁰±p³)/√2, metric (+−−−), A+=0 and ubar u=2m. A physical mode is
(p+,px,py), p+>0; p−=(m²+pT²)/(2p+). The exact global conversion remains
M²=2P+P−−PT². No finite-cell conversion factor is consumed in this interface.

The cubic leaf is the flat-mode P− coefficient

\[
v_{L,R;P}=\frac{N_{L,R;P}\,C_{L,R;P}}{\sqrt{8p_L^+p_R^+p_P^+}},
\qquad p_P^+=p_L^++p_R^+,\quad p_{P\perp}=p_{L\perp}+p_{R\perp}.
\]

It strips gs and the spatial delta/measure. Its units are GeV^−1/2, **not**
GeV or GeV²; a declared wavepacket/quadrature/cell map must supply the measure
before addition to a free-energy matrix. On-shell minus conservation is not
imposed at an individual LF Hamiltonian vertex. Resolved axes use helicities
(−,+), with color inside each spin axis:

| Channel | Tensor axes (left, right, parent) | Color factor |
|---|---|---|
| q → qg | 6 × 16 × 6 | t[a,left,parent] |
| qbar → qbar g | 6 × 16 × 6 | −t[a,parent,left] |
| g → q qbar | 6 × 6 × 16 | t[parent,left,right] |
| g → gg | 16 × 16 × 16 | f[parent,left,right] |

All use the modular SU(3) generator ordering, not an assumed identification
with a differently ordered historical Gell-Mann basis. The antiquark uses
the conjugate representation. The gluon coefficient is symmetric under the
joint exchange of daughter momenta/helicities/colors. Enumerate unordered
daughter pairs once, or supply 1/2 for a fully ordered global enumeration;
the local tensor does not silently choose that global domain.

Each cubic word is a†L a†R aP, acting right-to-left. Its adjoint conjugates
the coefficient and reverses/daggers every ladder. Fermion signs come from
all preceding fermion axes, not from boson axes. Boson factors are √n and
√(n+1). The free EM kernel is the physical flavor charge times the
number-preserving bilinear divided by 2√(pout+ pin+); the antiquark includes
the normal-ordering charge sign. All four components (+,−,x,y) are retained.
An occupied spectator between transfer modes tests the otherwise easy-to-miss
fermion sign.

Finite local boson capacity is explicit. At dimension d,

\[
[a_d,a_d^\dagger]=I-d|d-1\rangle\langle d-1|.
\]

Thus products of projected ladders need not equal projection of a complete
untruncated word. The cap-two negative gives −2 at its boundary, versus +1
when the intermediate excursion is retained before final projection.
Normal ordering/complete intermediates cannot be replaced by a claim that a
finite register obeys the exact boson commutator. These capacities test the
compiler; they do not select a nuclear Fock truncation.

## Complete partonic process and decisive controls

For a Cartesian off-shell fermion momentum r with signed r+≠0,

\[
\frac{\not r+m}{r^2-m^2}
=\frac{\not r_{\rm on}+m}{r^2-m^2}+\frac{\gamma^+}{2r^+}.
\]

The on-shell momentum retains r+ and rT. Negative r+ is not replaced by its
absolute value. For conserved currents J,C and exchanged q,

\[
\frac{J\cdot C}{q^2}
=-\frac{\sum_h(J\cdot\epsilon_h)(\epsilon_h^*\cdot C)}{q^2}
-\frac{J^+C^+}{(q^+)^2}.
\]

The qg tree contains s/u fermion exchange and t gluon exchange with fixed
color factors t_b t_a, t_a t_b and [t_b,t_a]. Both fermion instantaneous
pieces and the gluon instantaneous piece are retained. The result is a
96×96 external spin/color amplitude, stripping gs² and an overall S-matrix
phase. The covariant reference directly contracts the covariant numerators;
it is not defined as the sum of the tested LF pieces. External polarization
replacement tests both gluon Ward identities. Frame comparisons independently
transport Dirac spinors and polarization bases, allowing the gluon gauge term
whose decoupling requires the complete process.

The [numerical replay](../snapshot/docs/next_level/canonical_lf_network_integration_results.json) contains
12 fixed-grid and 12 fresh-momentum points, plus 24 boost/rotation comparisons:

| Control | Result |
|---|---:|
| Largest LF/covariant relative amplitude difference | 6.09 × 10^−14 |
| Largest dimensionless external-gluon Ward residual | 2.50 × 10^−15 |
| Largest transported-frame relative difference | 4.88 × 10^−14 |
| Smallest amplitude error after deleting all instantaneous pieces | 0.522 |
| Smallest Ward error after deleting the three-gluon graph | 0.0155 |
| Cases retaining negative crossed-fermion plus momentum | 5 |

The omission errors are Frobenius-norm diagnostics on these chosen partonic
points, not probabilities or estimates of nucleon truncation errors. Zero-plus
exchange and internal poles are explicitly outside this elementary test
domain; they raise errors instead of becoming invented zeros. No prescription
for the unresolved distributional/zero-mode domains is selected.

The four cubic tensors have 136,136,136,324 nonzero words at their fixture
momenta. Independent color equivariance residuals are below 2.7×10^−16;
direct occupation and tensor actions agree with spin/color contraction below
1.1×10^−16. The formal registers have 28 or 48 modes, but sparse one-parent
tests do **not** establish scalability on arbitrary highly entangled states.

## Verification, preservation and reproduction

The 61 new tests pass. A broader run passes 835 of 836 tests in 56.92 seconds,
including the new tests and legacy mode/emission/excluded-source/current,
cubic/quartic spin and tree controls, plus preceding architecture controls.
Its one failure is the unchanged synthetic F1-only inference tolerance:
3.8122371077386915×10^−9 against 10^−9. The identical failure/value reproduces
from an isolated archive of **unmodified parent f509938c**, with the same
runtime. It is not introduced by the canonical extraction; the earlier
reported all-pass result is environment-specific. No tolerance, old fit,
physics kernel or historical result file was changed to hide it.

Runtime: Python 3.12.14, NumPy 1.26.4, SciPy 1.13.1, SymPy 1.14.0,
pytest 8.4.2. The former temporary Python environment is broken; system Python
3.9 can run the new tests but cannot collect legacy kw_only dataclasses.
A new temporary environment is `/private/tmp/deuteron-canonical.zxUoKe/env`.
It can be recreated with these pinned versions; temporary paths are not
permanent project dependencies. No global installation was changed.
After the documentation update, the six current-theory/handoff checks and
all 61 new tests pass together (67 tests). No full historical theory re-audit
or full repository-suite execution is claimed.

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /private/tmp/deuteron-canonical.zxUoKe/env/bin/python \
  scripts/evaluate_canonical_lf_network.py \
  --output docs/next_level/canonical_lf_network_integration_results.json
```

Pre-edit source and notes are in
the 12-file archive (`computation_handoff/backups/2026-09-10_pre_canonical_network_integration.tar.gz`, full upstream checkout),
with the original package initializer (`computation_handoff/backups/2026-09-10_pre_lazy_quantum_imports.tar.gz`, full upstream checkout)
separately preserved. All 13 members match f509938c byte-for-byte. Original
theory volumes, old numerical products and unrelated untracked work remain
untouched. Local work only: no public push or review-mirror export.

## Reassessment and next physical question

**Advance the portable infrastructure; do not reopen the BLFQ/H2 repair route.**
The canonical ingredients survive independent non-Abelian tree consistency
tests without that model. Necessary instantaneous and exchange terms must
remain explicit in later state/operator representations. The old proton
component mismatch is **not solved** by this partonic result.

This step does not yet assemble the whole scattering amplitude automatically
from the compiled Fock words, supply every quartic/local Hamiltonian channel,
or supply a renormalized interacting EM current. Nor does a momentum-space
global-color tensor implement local Gauss constraints on a dynamical gauge
network by itself. Keep those distinctions visible when connecting the
ordered transport and composite modules.

The next bounded question is the **shared source-dependent soft interaction
and matching map that can generate proton/neutron states**. Prepare a limited
comparison of physically specified interaction families, explicitly retaining
soft/chiral content and operator insertions, instead of selecting an old
confinement potential or fitting independent observable heads. Specify the
resolution/measure, how canonical short-distance terms enter without double
counting, normalization, symmetry breaking and transverse-current freedom.
The decisive next deliverable is a small state-and-current reference with an
independent covariance/limit test, not a generic graph experiment or an
unbounded all-orders repair program. A substantial physical model commitment
requires its own evidence and direction review.

After that: operator-sensitive network compression/joint inference;
matched p/n-to-deuteron then A=3/4 modular dynamics; nonlocal operators and
matching/evolution for TMD/GPD/GTMD predictions; classical/quantum resource
qualification on the same controlled model. Canonical syntax, gauge symmetry,
topology and quantum encoding alone do not complete any of those steps.

# Canonical constrained-field Wick self terms

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-06. Lane: **derived contraction-regulated construction**.
The user selected the corrected canonical global-Jacobi Galerkin branch
as development authority. C43/C57 remains preserved comparison history;
reproducing its different operation order is not a prerequisite.

## What is now established

The five vacuum-subtracted one-body Wick descendants of the canonical
order-gs² quartic Hamiltonian are derived and implemented: quark self
terms from instantaneous fermion and gluon products, and gluon self
terms from the fermion trace, adjoint current, and local four-gluon vertex.
The original partial-pilot matrices are unchanged. These are neither
propagating V†(E-H0)^-1 V terms nor a fitted physical mass correction.

Two independent checks protect the signs: direct fermion occupation-bit
CAR matrices reproduce the antiquark subtraction, while explicit SU(2)
boson Fock matrices reproduce both adjoint-current and quartic tadpoles.
SU(3) uses C_F=4/3, C_A=3, T_F=1/2. Independent Dirac inverse/spinor
and doubly-transverse gluon identities check the constrained propagators.

The source is the nonzero-mode canonical Hamiltonian in
[Srivastava--Brodsky, Eq.24](https://arxiv.org/html/hep-ph/0011372v2).
The following discrete formulas are our derivation with the project's
symmetric light-front and unit discrete field conventions. The continuum
quark expressions also agree in structure with
[Li et al., Appendix B, Eqs.101--104](https://arxiv.org/html/2504.07162v1)
after conversion from their different light-front convention; their
conditional truncation rule and scalar mass cancellation are not imported.

## Regulator and operation order

Let ell_-=2L, delta=pi/ell_-, P+=K2*delta, and p+=p2*delta.
Positive bosons have k2=2,...,2N_b; fermions have r2=1,...,2N_f-1.
Every external field mode must be in that field envelope. There is no
restriction k2<p2 on the Wick trace: negative instantaneous denominators
are retained. Q0 excludes only an exactly zero integer transfer in the
instantaneous-gluon inverse; the other term at the same internal mode
must not be dropped.

The independent coincident transverse contraction is

    I_R = integral d²k_T/(2pi)² R(k_T), in GeV².

A translation-invariant Gaussian test trace gives I_R=Lambda²/(4pi).
This is **not** supplied by the external many-body oscillator cutoff.
It regularizes these contractions; it does not assert that the earlier
bare projected cubic vertices already have a Gaussian field regulator.
Wick expansion, vacuum subtraction, and Galerkin projection have distinct
roles. No equality with a support-before-normal-ordering construction is
assumed.

## Exact discrete one-body kernels

All sums below run over the finite positive field sets, with no extra
longitudinal step weight: the cell factors have already been included.
For a quark with odd p=p2 define

    F_qf(p) = sum_k 1/[k(p-k)],
    F_qg(p) = sum_r [ 1_{r!=p}/(p-r)² - 1/(p+r)² ].

Then its full-system mass-squared correction per gs² is

    S_q(p) = C_F K2 I_R/pi * [F_qf(p)+F_qg(p)].

The first coefficient follows directly from the retained ordered
instantaneous-fermion contact K2/[2pi*k*(p-k)] and the helicity sum 2.
In the second expression, the negative pair term follows from fermionic
normal ordering after subtracting the vacuum identity. The independent
finite-Fock calculation uses (1/2) sum_{n!=0} J_n J_-n/(2n)².
At fixed odd p=2a+1 the convergent infinite longitudinal sums are

    F_qf -> [sum_{j=1}^a 1/(2j-1)-log(2)]/p,
    F_qg -> (1/2) sum_{j=1}^a 1/j².

The transverse UV trace is still divergent if its cutoff is removed.
In particular p=1 gives (-log(2),0), not a positive self probability.

For an even external gluon p=p2 define color/flavor-weighted kernels

    F_gf(p) = T_F n_f/p * sum_r [1/(p-r)-1/(p+r)],

    F_gg(p) = C_A sum_k {
      1_{k!=p}(p+k)²/[4pk(p-k)²] + (p-k)²/[4pk(p+k)²] },

    F_g4(p) = C_A/(2p) * sum_k 1/k.

Its mass-squared correction per gs² is

    S_g(p) = K2 I_R/pi * [F_gf(p)+F_gg(p)+F_g4(p)].

The two adjoint terms are respectively the number-current and Bose-pair
contractions. For the quartic term, the six single Wick contractions give
2(d_T-1) C_A C_R A², where d_T=2 and
C_R=I_R/(2pi) sum_k 1/k. The subsequent one-gluon A² matrix element and
2P+ conversion fix the factor above. The fermion trace counts an explicitly
supplied number of flavors, not a hidden flavor-multiplicity default.

The coupled operator is the sum over the three quarks and, in qqqg, the
gluon, projected with the same longitudinal/orbital/color/spin-flavor
isometries as the pilot. It preserves the exact total-sign, I and Jz
construction. There is no quark-gluon mixed one-body trace: its remaining
single color generator/structure-constant trace vanishes. Vacuum constants
are subtracted, not used as hadron masses.

## Normal ordering is an explicit subtraction, not a physical fit

The earlier canonical pilot used normal-ordered composite products.
The calculation now makes its excluded Wick terms explicit. A consistent
normal-order subtraction for the displayed one-body terms is

    C_R = -gs² S_R
          + delta(m_q²) sum_{i=1}^3 1/x_i
          + delta(m_g²) 1_{qqqg}/x_g .

Both finite mass-squared coefficients are mandatory caller inputs and
may be signed. They change kinetic masses only; no vertex-mass relation
or physical current condition is inferred. Setting both to zero is used
only to test that H_pilot+gs² S_R+C_R=H_pilot.

The full S_R is not, in general, in the span of those two kinetic mass
directions. Thus cancelling the explicit normal-order trace is an
operator-valued composite-operator subtraction, not the claim that two
scalar physical mass counterterms renormalize this truncated theory.
Its remaining propagating UV dependence is not removed by this identity.

At zero skewness the implemented local good densities preserve longitudinal
fractions. They commute with S_R and the two kinetic mass directions before
and after the symmetry projection. No extra divergence-fixed transverse
mass-flow term comes from these diagonal contributions at Delta+=0.
This does not fix the full transverse current, longitudinal nonzero-skewness
vertex, endpoint/zero-mode contributions, or non-Abelian Ward identities.

## Executed numerical example

Use the established 37-state proton pilot K2=9, N_exc=0, b=.4 GeV,
mq=.35 GeV, mg=0, gs=.3; its lowest M² is 1.421433230301892 GeV².
At N_b=N_f=12, n_f=2, and Gaussian trace width .8 GeV:

| Quantity | Value |
|---|---:|
| Quark instantaneous-fermion norm per gs² | 1.26773544729 GeV² |
| Quark instantaneous-gluon norm per gs² | .945837773290 GeV² |
| Gluon instantaneous-fermion norm per gs² | .738665340290 GeV² |
| Gluon instantaneous-gluon norm per gs² | 1.47183462741 GeV² |
| Gluon local A⁴ norm per gs² | .726985908534 GeV² |
| Total self norm per gs² | 1.31112068436 GeV² |
| Residual outside both kinetic mass directions | .748774936428 GeV² |
| Unsubtracted lowest M² | 1.45390799266 GeV² |
| Good-current commutator norm | 7.4e-17 GeV² |

Changing only the trace width to .4 or 1.6 GeV gives unsubtracted M²
1.42955192181 or 1.55133221374 GeV². Explicit zero-finite normal-order
subtraction restores the original matrix to below 2e-33 GeV² roundoff
in this run. These are regulator diagnostics, not physical masses,
uncertainty bands, or evidence of cutoff-independent QCD.

## Reproduction and next science

    python -m pytest tests/test_m2_canonical_quark_self.py tests/test_m2_canonical_gluon_self.py tests/test_m2_constrained_propagators.py -q
    python scripts/benchmark_m2_canonical_self.py

68 new direct tests pass in 4.98 seconds, including the 204-state
N_exc=1 projection/current check. No source behavior of the older
266-state oracle or the canonical pilot matrix is changed.

Next complete the finite-reference loop/observable transport and study
its cutoff dependence under this explicit normal-order convention; then
derive higher-order instantaneous exit/QQ and physical current partners.
The common-regulator continuum loop limit, endpoint/zero-mode matching,
physical finite mass/residue/vertex/current conditions, nuclear/GTMD
identification, and topology-factorized quantum scaling remain open.
No workstream is declared physically complete and no production state
or current is activated. Missing physical inputs do not stop parameterized
development.

Pre-edit pickup/theory/test backup from 40d25b68:
`computation_handoff/backups/2026-09-06_pre_canonical_self/pickup_theory_and_test.tar`,
SHA-256 `c13971ba96a21876d09adb2971e9fd0855339b3cd2ff21bb1bbc126da31a302a`.
The same directory's `source_comments.tar` preserves the source descriptions
before updating their self-term status. Original volumes and C43/C57
sources are unchanged. No push or public-mirror refresh is included.

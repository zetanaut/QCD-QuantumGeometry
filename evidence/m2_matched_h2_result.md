# Matched H2: finite-parent reduction and current benchmark

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Scientific source: a4946f0c. User direction selected 2026-09-06:
**matched H2 first**, with explicit higher-Fock enlargement still on the
completion path. H2 means qqq plus qqqg, not a complete all-orders theory.

## What this increment establishes

The implementation eliminates an explicitly supplied finite complement,
preserving the energy-dependent interaction, reconstruction map, norm
metric and two-energy operator/current matrix elements. The real H2
benchmark retains the N_exc=0, 37-state carrier inside N_exc=1/2 parents
of 204/654 states, at common total K2=9, b=.4 GeV, m_q=.35 GeV, m_g=0,
g_s=.3, proton I=Jz=1/2.

**The eliminated states here are higher oscillator modes in the SAME
qqq+qqqg sectors. They are not qqqgg or sea-quark states.** This validates
the matching machinery and quantifies oscillator-truncation effects.
It does not supply the missing physical higher-Fock kernels or renormalization.

## Exact equations and normalization

Let U embed retained coordinates into the finite parent, and V span its
orthogonal complement. P=UU†, Q=VV†; H is the parent M² operator.
For real E outside the spectrum of H_QQ,

\[
 R_Q(E)=(E-H_{QQ})^{-1},\quad
 H_{\rm eff}(E)=H_{PP}+H_{PQ}R_Q(E)H_{QP},\quad
 \Omega(E)=U+VR_Q(E)H_{QP}.
\]

The reduced eigenproblem is nonlinear: H_eff(E) phi=E phi.
No full-parent eigenstate is used as input to the root solver. It brackets
the lowest root below all Q poles; excluded intruders/poles are rejected,
not clipped or replaced by invented denominators.

The reconstructed state is Psi=Omega(E)phi. Crucially,

\[
 N(E)=\Omega^\dagger(E)\Omega(E)
     =I-\partial_E H_{\rm eff}(E)
     =I+H_{PQ}(E-H_{QQ})^{-2}H_{QP},\quad
 \phi^\dagger N(E)\phi=1.
\]

The minus in the energy derivative matters. Euclidean normalization of
phi alone throws away the eliminated probability. Every operator needs

\[
 O_{\rm eff}(E_f,E_i)=\Omega^\dagger(E_f)O\Omega(E_i).
\]

All PP, PQ, QP and QQ terms contribute. At equal energy the unit charge
becomes N(E), not I; at different energies the two-energy overlap is needed.

For the parent baryon flow Delta.J=[H,rho], define
D_a=H_eff(E_a)-E_a I. Then the exact off-shell identity is

\[
 \Delta\cdot J_{\rm eff}
 =(E_f-E_i)\rho_{\rm eff}
 +D_f U^\dagger\rho\Omega_i-\Omega_f^\dagger\rho U D_i.
\]

The last two terms vanish between on-shell reduced states. Replacing this
by a guessed commutator of one energy-dependent H_eff and rho_eff is not
valid. The tests include nonzero transitions, not only zero diagonal flow.

This is the finite-matrix projection/resolvent construction, consistent
with the light-front effective-Hamiltonian program discussed by
[Pauli](https://arxiv.org/abs/hep-th/9809005). Exactness here is relative to
the **supplied partial finite parent**, not full QCD.

## Executed physical-basis benchmark

The nested map uses actual longitudinal, circular-oscillator, color and
spin/flavor labels followed by the symmetry isometries. It does not assume
the compact eigenvectors defining two different shells share an order.
Every H component restricts correctly; total H discrepancies are below
3e-14 GeV². The original 37-state bare energy is 1.42143323030189 GeV².

| Finite parent | Matched 37-state energy (GeV²) | Excluded norm probability | Bare projected density | Matched/reference density |
|---|---:|---:|---:|---:|
| N_exc=1, 204 states | 1.41947502860488 | 0.00184816854 | 0.88238811151 | 0.88207810865 |
| N_exc=2, 654 states | 1.30252682156313 | 0.21114000857 | 0.88240746394 | 0.79709360545 |

The densities are normalized baryon-density ground-state expectations at
Delta_T=(.12,.16) GeV. “Bare projected” uses the Euclidean-renormalized
projection of the FULL parent ground state, not a different fitted state.
Thus even knowing that projected state does not fix the missing operator.

Exact-root energies agree with independent parent diagonalization within
2e-15 GeV²; reconstructed-state residuals are below 6e-15 GeV².
Without the induced norm, the 654-state example's unit-charge expectation
would be .78885999143. The matched charge is one. These values are exploratory,
not physical form factors, nucleon masses, or continuum convergence evidence.

## Energy-local Hermitian approximation

For E0 away from Q poles, set S=H_eff'(E0), N=I-S and
A=H_eff(E0)-E0 S. Linearizing the nonlinear equation gives A phi=E N phi.
In orthonormal coordinates,

\[
 H_{\rm lin}=N^{-1/2} A N^{-1/2}
 =W^\dagger H W,\qquad W=\Omega(E_0)N^{-1/2},\quad W^\dagger W=I.
\]

This is a finite-parent Rayleigh-Ritz model and has a variational upper
bound. Each operator must use W† O W. It is not one exact operator for
all transitions. In particular, its projected current minus the simple
commutator equals W†H(1-WW†)rho W-W†rho(1-WW†)HW; this leakage is tested.

With E0 equal to the bare 37-state energy, the linearized ground errors
are 6.73e-9 GeV² for the 204-state parent and .01073913891 GeV² for 654.
For the latter, four energy-local updates, using only the reduced model,
produce 1.31326596047, 1.30258424188, 1.30252682314, 1.30252682156 GeV².
The corresponding final lifted-state residual is 8.17e-10 GeV², larger
than the exact nonlinear solve's residual. Energy agreement alone does not
certify the full state or current.

For delta=E-E0 and d=dist(E0,spectrum(H_QQ)), |delta|<d, the Taylor
remainder obeys ||remainder|| <= ||H_QP||² |delta|²/[d²(d-|delta|)].
Near poles the approximation must be refined, not extrapolated.

Direct 654->37 and staged 654->204->37 eliminations agree: embedding
4.36e-15, effective H 2.15e-14 GeV², wave map 3.13e-14, norm 1.50e-14.
Staged norms must compose as Omega_inner† N_outer Omega_inner; the inner
norm alone loses the outer elimination's probability.

## Physical matching still to derive

The next source calculation is the *excluded-Fock* contribution. If H0
is Fock-diagonal, and H=H0+g_s V+g_s² W+counterterms, the leading excluded
term is g_s² P V Q(E-Q H0 Q)^-1 Q V P. Here Q must exclude the retained
qqqg states already propagated in H2; adding their self-energy again would
double count. The oscillator benchmark has P H0 Q != 0 at N_exc=2, so its
correction must NOT be relabeled a pure order-g_s² loop.

For a retained qqqg state, leading cubic exits include qqqgg (quark
emission and gluon splitting) and qqqq qbar (gluon conversion). Their
coherent amplitudes, color/spin factors, Bose/Fermi symmetry, regulator,
instantaneous/self-contraction partners and matched currents still need
construction. This can be done as an intermediate-state sum without first
solving a larger explicit physical eigenproblem.

The old C57 conditional instantaneous prescription is not automatically
the normal-ordered Galerkin parent. The isolated q+qg example in
[Li et al., Appendix B](https://arxiv.org/html/2504.07162v1) absorbs a
sector-identity instantaneous term into its mass counterterm. That proof
does not establish that our coupled three-quark contact operators are
sector identities or can all be removed by one mass fit. We retain their
distinct operator content until the matching calculation decides it.

H2-first is the selected construction order, not selection of a physical
state/current/calibration default and not cancellation of the H3/explicit-sea
roadmap. Quantum graph scaling continues against these exact references;
this dense-complement benchmark makes no runtime or quantum-advantage claim.

## Reproduction and preservation

Implementation: src/deuteron_wigner/quantum/m2_matched_h2_reduction.py.
Tests: tests/test_m2_matched_h2_reduction.py (21 direct tests).
The reduction + existing pilot + graph focused run passes 35 tests, 7.57 s.
The full named M2 + Q0--Q2 regression passes 464 tests (431 M2, 33 Q),
with 1,885 inherited warnings in 286.70 seconds. This is not a full-repository
test claim. The six separate theory-consistency tests pass in 0.06 seconds
after final hash synchronization.

The main theory note's section 18.7.6 and status page were rebuilt and
visually checked in a 135-page scratch PDF. All 350 unique labels and
76 cited entries resolve, with no rendered `??`, missing glyphs or overfull
boxes. The inherited generic rerun/undefined-reference warnings remain
despite this explicit audit. No additional formalism volume is needed
for this bounded increment; original volumes are preserved.

A minimal reproduction is:

~~~python
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_matched_h2_reduction import (
    FiniteParentReduction, nested_pilot_isometry,
)
args = dict(total_k2=9, target="PROTON", jz_twice=1, b_GeV=.4,
            quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)
small = build_coupled_pilot(cutoff=0, **args)
parent = build_coupled_pilot(cutoff=2, **args)
r = FiniteParentReduction(parent.matrix, nested_pilot_isometry(small, parent),
                         scope="higher oscillator shells ONLY, same H2 sectors")
answer = r.lowest_state()
print(answer.energy_GeV2, answer.excluded_probability,
      answer.parent_residual_GeV2)
~~~

Pre-edit notes/theory/test backup, from ddbeb935:
computation_handoff/backups/2026-09-06_pre_matched_h2/pickup_theory_and_test.tar.
SHA-256 e3aa6b36ca04165d077ad5a1bc8ff61cdff0a08a7c840080e0f1cc733a7e42d7.
Original volumes and legacy source matrices remain unchanged. No push.

# Matched H2 with actual higher-Fock sources

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-06. Scientific source: `6547c259`.
Lane: **bare finite-regulator exploratory calculation**, not a renormalized
nucleon/deuteron prediction. H2-first remains selected. The three requested
workstreams are not complete.

## Executed advance

The canonical color-singlet, total-sign `qqq + qqqg` pilot now has actual
cubic exits into `qqqgg` and `qqqq-qbar`. This is not the higher-oscillator-
only complement used by the preceding matching benchmark.

- Quark emission and gluon splitting feed qqqgg coherently.
- Gluon conversion creates the flavor singlet u ubar + d dbar coherently.
- Complete color-singlet multiplicities are six and three.
- Fractions, global Jacobi modes, color, helicity, flavor and particle
  statistics remain resolved.
- The induced norm, self energy and two-energy current match an explicit
  finite parent, including reconstruction in the full uncompressed space.
- Separate proton/neutron electromagnetic good currents and net up/down
  densities now accompany the baryon current.
- Actual controlled-gate PennyLane execution works on the matched model.

Old numerical Hamiltonians, formal volumes and original archives are
unchanged. No missing instantaneous/self-contraction/counterterm term is
declared physically zero. The small finite shifts below do not bound all
omitted higher-Fock or ultraviolet effects.

## Derivation and source ownership

Use symmetric LF coordinates and the canonical interaction of
[Srivastava–Brodsky, Eq. (24) and Appendices A/B](https://arxiv.org/html/hep-ph/0011372v2).
With ubar u=2m and field factors 1/sqrt(2p+), the distinguished-daughter
mass-squared kernel is

\[
 V/g_s={2P^+{\cal N}\over
 \sqrt{\ell_-(2p_0^+)(2p_1^+)(2p_2^+)}},\qquad
 \ell_-={\pi K_2\over P^+}.
\]

For daughter fractions x,y and X=x+y, the physical inverse is
`p1T=(x/X)p0T+sqrt(xy/X)Q` and `p2T=(y/X)p0T-sqrt(xy/X)Q`.
The quark numerator is `-ubar1 gamma.eps2* u0`. The antiquark spinor
`v_h=i gamma2 u_h*` carries physical antiparticle helicity h; its
good-component spin eigenvalue has the opposite sign. The pair numerator
is `-ubar1 gamma.eps0 v2`. Direct Dirac, completeness and good-spin
checks are independent of the source-action assembly.

For gluon splitting use all-outgoing `k0=-p0,k1=p1,k2=p2` and
`e0=eps0,e1=eps1*,e2=eps2*`. With real f(a,b,c) factored into color,

\[
 {\cal N}_{ggg}=-i\{(k_0-k_1)\cdot e_2(e_0\cdot e_1)
 +(k_1-k_2)\cdot e_0(e_1\cdot e_2)
 +(k_2-k_0)\cdot e_1(e_2\cdot e_0)\}.
\]

The derivative phase matters relative to quark emission; all six field
contractions reproduce it. Daughter exchange reverses both f and the
kinematic numerator. On-shell minus conservation is not imposed at a
Hamiltonian vertex.

The HO integral includes `sqrt(xy/X) d²Q/(2pi)²` and the C45 circular
phase `(-1)^min(n_plus,n_minus)`. The kernel is constant plus linear Q,
so exact-degree Gaussian quadrature suffices; direct radial/angular
integration independently checks it. The required orbital m is
`(h0-h1)/2-h2` for quark emission, `h0-h1-h2` for gluon splitting and
`h0-(h1+h2)/2` for pair conversion, using doubled fermion helicities.
The gluon-splitting ground HO projection is an exact angular zero.

Normalized ordered-slot wave functions require

\[
 V_{qqqgg}=\sqrt2\,S_2 V_{q\to qg}+V_{g\to gg}/\sqrt2,\qquad
 V_{qqqq\bar q}=2A_4 V_{g\to q\bar q}.
\]

S2 symmetrizes the gluons, A4 antisymmetrizes all four quarks including
spin/flavor. Independent occupation states check the creation factors.
Adjacent permutation generators prove full exchange symmetry; I+
annihilation on the proton source proves total I=1/2, not just fixed Iz.
Color source norms reproduce CF=4/3, CA=3 and TR=1/2. Singlet generator
residuals are below 8e-15.

These finite-cell/Jacobi/statistics formulas are the project's explicit
derivation, not numerical tables imported from the source paper. Full
equations are also in the main theory note, Sections 18.7.7–18.7.8
(use labels `sec:excluded-fock-h2` and `sec:excluded-fock-decision` if numbering changes).

## Actual finite-parent matching

\[
 H_{\rm ref}=\begin{pmatrix}H_{\rm pilot}&g_sV^\dagger\\
                           g_sV&H_{0,Q}\end{pmatrix},
 \quad Q=qqqgg\oplus qqqq\bar q .
\]

Only the retained qqqg columns have these cubic exits: retained qqq to
qqqg propagation is not counted twice. The self energy is
`g_s² V†(E-H0,Q)^(-1)V`, with negative energy derivative and positive
induced norm correction. The exact parent solve resums returns through P,
not the missing interacting QQ block or complete higher-order QCD.

At Q cutoff 0 or 1, projected H0 is diagonal:
`sum m_i²/x_i + b²(4+N_exc)`. This shortcut is invalid at cutoff 2,
where free radial off-diagonal elements occur; the API rejects that use.
An SVD within exactly degenerate free-energy blocks compresses only the
source-reachable range. Exact rational mass coefficients define the
groups, not floating energy bins. Full source reconstruction errors are
below 2e-15. Current images outside the range are not assumed zero:
the range is H-invariant, so compression preserves the current matrix
elements and commutators for these reference states.

Fixed example: total nucleon K2=9, retained N_exc=0, I=Jz=1/2,
b=.4 GeV, mq=.35 GeV, mg=0, gs=.3. This is not the old active K9
subsystem inside K2=27.

| Q cutoff | packed qqqgg / sea | reachable qqqgg / sea | parent size | matched M² (GeV²) | Q probability |
|---|---:|---:|---:|---:|---:|
| 0 | 630 / 4,500 | 25 / 30 | 92 | 1.421431890118925 | 6.016351967e-7 |
| 1 | 4,950 / 31,500 | 68 / 60 | 165 | 1.421430998167897 | 1.041558235e-6 |

The bare 37-state partial energy is 1.42143323030189 GeV².
Independent eigenvalues agree within 3e-15 GeV² and full reconstructed
state residuals are below 2e-15 GeV². These are finite-reference identities,
not physical convergence errors.

At Q cutoff 1 and E=1.4 GeV², the quark/gluon-emission interference
contribution to the per-gs² self energy has Frobenius norm
0.0120867477832 GeV². Adding incoherent probabilities misses it.

## Matched flavor-resolved good currents

At Delta+=0, positive longitudinal modes give a canonical local J+ diagonal
in Fock number. Its normalized density strips the common 2P+, with actual
recoil `delta Q_a=J_x[a,i] Delta_T`. Quark/antiquark baryon charges are
+1/3 and -1/3; electric charges for u,d,ubar,dbar are +2/3,-1/3,-2/3,+1/3.
The antiquark sign comes from CAR, not a negative vbar gamma+ v.
Net up/down currents remain separate:
`rho_B=(rho_u+rho_d)/3` and `rho_EM=(2rho_u-rho_d)/3`.

Every operator is matched as `Omega(Ef)† O Omega(Ei)`. Equal-energy
charge equals target charge times the induced norm metric, not the
Euclidean identity. The minimally gauged mass-flow current retains
number-changing P-Q terms and satisfies the preceding two-energy
off-shell identity. This is not a full physical current four-vector,
spin-one form-factor prescription, or non-Abelian Ward closure.
The [good-current overlap in an explicit LF Fock calculation](https://www.sciencedirect.com/science/article/pii/S0370269322000247)
supports the source convention, not the renormalization of our finite model.

At transfer (.12,.16) GeV:

| Q cutoff | matched baryon density | bare density on Euclidean-normalized P |
|---|---:|---:|
| 0 | .882387105478064 | .882387199559474 |
| 1 | .882386815079574 | .882387016672472 |

At Q cutoff 1, electromagnetic values are .882386814622664 (proton)
and 4.569101830e-10 (neutron). Proton net-up/down values are
1.764773629702238 and .882386815536484; neutron values exchange.
These near-symmetric values belong to this tiny equal-mass model,
not physical neutron predictions. Zero-transfer proton/neutron charges
are exactly 1/0, net up/down 2/1 and 1/2, and baryon number 1.

## Quantum continuation

The E0=1.4 GeV² energy-local Hermitian model uses 37 directions, six
qubits and 72 graph parameters. Actual controlled rotations/phases,
not an exact-eigenstate StatePrep seed, pass gate/state/energy/padding checks.
Tree-uniform initialization with small independently seeded phases gives:

- L-BFGS-B stops by relative objective reduction at iteration 1,248/2,200.
- Energy 1.421431000297159 GeV².
- Error against the exact linearized model 1.89975e-9 GeV².
- Overlap squared .999999997875492; parameter-gradient norm 8.53431e-8.
- State-equation residual 4.39253e-5 GeV²; final padding norm zero.

The requested 1e-8 gradient tolerance was not met. Objective-based optimizer
success is not a state-residual certificate. The linearized model differs
from the exact parent by approximately 2.30e-10 GeV². No scalable
tensor-network or hardware advantage is claimed.

## Genuine direction decision

The new branch normal-orders then Galerkin-projects on global Jacobi modes.
The historical C57 primary regulator selects corresponding propagating
support before normal ordering on the older C45/C47 field construction.
It explicitly rejects the former operation order as its primary scheme.
They cannot be identified by a scalar, equal dimensions or one spectrum:

\[
 PAA P-PA P A P=PA(1-P)AP .
\]

Bare cubic radial tails are constant in n for m=0 and grow as sqrt(n+1)
for |m|=1. Summed squared source norms thus grow linearly/quadratically
with radial cutoff. This is a source-norm witness, not a calculation of
renormalized loop divergence; the two finite examples do not establish UV closure.

Mass, charge and a gauged continuity identity cannot select all finite
terms either. Add `lambda P_qqqg+c(lambda)P_qqq` to the same finite
parent and solve c to preserve its lowest mass. The projectors preserve
its internal quantum numbers. All members retain unit proton charge
and their own consistently gauged finite continuity:

| unselected lambda (GeV²) | compensating c (GeV²) | same M² (GeV²) | EM density |
|---:|---:|---:|---:|
| -.50 | .001351469716102 | 1.42143099816790 | .881934633711259 |
| 0 | 0 | 1.42143099816790 | .882386814622664 |
| +.50 | -.000382241503847 | 1.42143099816790 | .882451377029293 |

These are identifiability witnesses, not selected counterterms, priors,
or physical uncertainty estimates.

**Recommendation for the theory lead:** make the corrected canonical
coupled-basis branch the authoritative development regulator and derive
its regulated constrained-field/self-contraction/counterterm/current
partners. Preserve C43/C57 as comparison material rather than requiring
the new branch to reproduce the old numerical chain. Alternatively,
require C43/C57 compatibility and derive a full operator conversion first.
This is a physics-direction choice, not another file-edit permission.

After that choice: regulated instantaneous/self partners; shared
mass/residue/vertex conditions with finite-transfer holdouts; resolution
tests; physical transverse/spin currents and nuclear/GTMD matching;
topology-factorized solver scaling against corrected references.
No need to repeat the user's H2-versus-explicit-first choice.

## Reproduction, checks and preservation

```text
python -m pytest tests/test_m2_excluded_fock_sources.py tests/test_m2_excluded_fock_matching.py tests/test_m2_canonical_good_current.py -q
python scripts/benchmark_m2_excluded_fock.py
python scripts/benchmark_m2_excluded_fock.py --quantum
```

Use the project's verified Python environment and one BLAS/OpenMP thread.
107 new direct tests pass in 58.69 seconds. The entire named M2 plus
Q0–Q2 regression passes **571 tests** (538 M2, 33 Q), 1,885 inherited
warnings, 347.85 seconds. This is not a full-repository test claim.
Document synchronization and rendered-equation QA are recorded in the
current handoff; test counts do not prove physical completeness.

Pre-edit pickup/theory/test backup from `1fe3531c`:
`computation_handoff/backups/2026-09-06_pre_excluded_fock/pickup_theory_and_test.tar`,
SHA-256 `db13fb5140cb16c9c65913356762148e58919103d39397496e14740b4a0e2a1f`.
Original archives and unrelated user work are preserved. No push or public
mirror refresh was performed.

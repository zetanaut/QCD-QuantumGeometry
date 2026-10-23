# LF proton/neutron interaction layer: first complete emission

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. Exploratory finite reference, **not a physical bound
nucleon, production interaction, covariance repair or nuclear prediction**.

## Question and outcome

Can the normalized canonical/retained interaction tensors actually act on
the preserved p/n quark states without losing spin, fermion exchange,
emitter interference, collective nodes or gluon color? Yes, for the complete
declared first-emission window. The sparse exterior action agrees with an
independent labelled-particle antisymmetric tensor and the existing graded
tensor compiler. This connects portable operators to the modular state
architecture; it does not supply the missing binding interaction.

The original p/n interpolators and all preceding operators remain unchanged.
One-orbital inputs are benchmarks, not production scalar-diquark states.
Unequal-momentum packets, fixed-total-momentum projections and overlapping
six-quark products are also tested. The action accepts general correlated
quark exterior states; only the factorial independent reference is limited
to at most four quarks. No old oscillator, fixed-K model, Gaussian regulator,
H2 reduction, fitted loop or independent observable head is imported.

## Canonical spin is not LF helicity

For positive mass and physical LF momentum p=(p+,px,py), construct the
rotationless-boost Dirac spinors

    U_can = sqrt(E+m) [ I ; sigma.p/(E+m) ],
    E = (p+ + (m²+pT²)/(2p+))/sqrt(2).

Canonical columns are (up,down). The preserved LF columns are (-,+).
The basis map follows from the actual spinors, not an assumed composite
Melosh prescription:

    U_can = U_LF R,
    R = U_LF† U_can/(2E) = Ubar_LF U_can/(2m),
    R = [ -pR, a ; a, pL ] / sqrt(a²+pT²),
    a=m+sqrt(2)p+, pR=px+i py, pL=px-i py.

R is unitary. Even at zero transverse momentum the opposite column orders
must be permuted. At p=(.375,.2,0) GeV and m=.27 GeV,

    R = [[-.242441475059, .970166032786],
         [ .970166032786, .242441475059]].

The old mode ordering is flavor/spin/color. The new ordering is
momentum/helicity/flavor/color. Every one-particle map is lifted by an
exterior product, including sorting signs. Replacing this map with a label
permutation passes charge and color checks but changes the tested emission
vector by .0124851 GeV. This is a basis error, not a missing interaction to fit.
The map does not construct interacting nucleon spin or prove rotational
covariance of a finite many-body approximation. Mass-zero canonical rest
spin is outside this particular map.

## Same-action emission and source contact

Let a+k=p in plus/transverse components. The retained scalar/isovector
operator has Gamma_A=(1,i gamma5 tau1,i gamma5 tau2,i gamma5 tau3), with
identity in color. Its discrete positive Hamiltonian coefficient is

    V[n,A] = g_n ubar(a) Gamma_A u(p) / sqrt(8 a+ k+ p+ V_cell).

No minus-energy conservation is imposed on a Hamiltonian vertex. Both
collective nodes and all flavor/helicity/color entries remain. An independent
derivative of h_chi=B (1/p+) B/2 gives

    V[n,A] = g_n chi_a† [ B0(a) X_A/a+ + X_A B0(p)/p+ ] chi_p
             / (2 sqrt(2 k+ V_cell)),
    X_A=beta Gamma_A, B0=alphaT.pT+beta m.

Both quark Fourier denominators are positive. The emission is not obtained
by retaining the negative antiquark denominator of the preceding pair
creation formula. The independent leaf difference is 1.15e-14 GeV^-1/2
before the common coupling/cell factor.

The gluon coefficient is the unchanged canonical q->qg tensor times
gs/sqrt(V_cell), with a flavor identity inserted in the verified axis order.
All two physical helicities and eight adjoint colors are explicit. Normalized
Hamiltonian coefficients have GeV units; delta M²=2 P_total+ delta P-minus.
The reference uses the preceding AP-longitudinal-fermion/periodic-boson cell.

The crossed one-Yukawa/one-EM **fermion contact** is also derived, with
a+k=p+q, q+=0 and nonzero qT. With B_gamma=-alphaT.Q A_T, it is

    K_A = chi_a† { B_gamma X_A/a+ + X_A B_gamma/p+
          + A+ [ B0(a) Q X_A/(a+)² + X_A Q B0(p)/(p+)² ] } chi_p
          / (2 sqrt(2 k+ V_cell)).

Its independent instantaneous-propagator expression is

    ubar(a) [ Q slash(A) gamma+ Gamma_A/(2a+)
              + Gamma_A gamma+ Q slash(A)/(2p+) ] u(p)
              / sqrt(8 a+ k+ p+ V_cell).

All four source-component matrices agree within 1.37e-16. The derivative
dH/dA is dimensionless; with A in GeV, K is in GeV. Mixed J+ is zero here;
mixed transverse and J-minus are not. This contact is **not** the complete
sourced baryon state: all sequential quark/spectator/collective insertions,
the outgoing collective normalization endpoint and the gluon-channel source
partners must still be assembled and independently checked. C/D/E are
preserved in their original action; their absence from a source-free
first-emission coefficient is not an assertion that their coefficients vanish.

Canonical LF constraints and instantaneous interactions are established
field theory; see [Srivastava and Brodsky, Section 4](https://arxiv.org/html/hep-ph/0011372v2).
The retained S/P interaction is the project's explicitly qualified reference,
not canonical QCD or a newly selected production sigma model. The current
contribution is its checked state/architecture connection, not a novelty
claim for spin-basis transport or fermionic second quantization.

## Complete emitter and state action

    V = sum_(p,a,k,n,A,indices) V[n,A] b_a† a_(k,n,A)† b_p
        + sum_(p,a,k,lambda,c,indices) Vg b_a† a_(k,lambda,c)† b_p.

Every occupied quark is acted on. Output amplitudes with the same physical
state are summed before taking a norm. In the independent reference a
normalized N-fermion exterior coefficient is expanded into N! ordered
components with parity/sqrt(N!); the vertex acts on every slot, then the
complete antisymmetric tensor is projected back. No sign algorithm from
the one-body implementation is reused in that reference.

For both p/n and both benchmark spin inputs at shell one, the complete
emission norm squared is .00261243504238 GeV². The largest tensor difference
is 9.46e-18 GeV, charge error 4.51e-18 GeV and total-color error 5.57e-18 GeV.
Color includes the emitted gluon's adjoint generator. These are global
color/charge tests, not a demonstration of local quantum Gauss-law closure.

One labelled emitter projected back gives one third of the complete
antisymmetric amplitude. Omitting the other two changes it by .0340747 GeV.
Adding three independent emitter probabilities would give only one third
of the correct norm squared. Route/slot labels are not additional physical
sectors. The exact absorption adjoint, momentum conservation and the whole
channel's Jordan--Wigner tensor action also pass.

An unequal-momentum packet is lifted isometrically; projection onto one
quark in orbital p and two in q has its unrenormalized binomial weight,
exact total p+2q and all fermion correlations. Overlapping p/n products
retain their nonunit composite norm and Pauli exchange. The six-quark
emission obeys the exterior product rule. This is modular algebra, not a
bound deuteron or a claim that nucleons are elementary fermion modes.

During development a NumPy integer mode index narrowed the occupation-bit
arithmetic beyond 64 modes. The adapter now explicitly uses Python integer
indices. Full-vector comparison caught this failure when a zero output had
trivially passed charge/color. A beyond-64-mode nonzero regression remains;
no physics kernel or test tolerance was changed to force agreement.

## First-order time evolution and finite-window evidence

The reference evolves the coefficient of one sin^4(pi x+/T) interaction:

    psi1(T) = -i integral_0^T dt f(t) exp[-i Hout (T-t)] V exp[-i Hin t] psi0.

Hout retains all three quarks and the two-node collective mass matrix, or
the free gluon. Distinct input energies are propagated separately and summed
coherently. Spectral five-harmonic integration agrees with independent
adaptive time integration using matrix exponentials: the small-window
full-wave difference is 5.79e-17. Longitudinal boosts agree only after
transforming cell lengths, LF time and the same physical input state; a new
canonical-spin seed at the boosted momentum is not silently substituted.

The replay uses m_q=.27 GeV, g=(.31,-.19), gs=.43, T=12 GeV^-1 and the
preceding illustrative mass matrices/cell. None is fitted to nucleon data.
For three collinear p=(.375,0,0) GeV input quarks:

| Active daughter transverse shell | Channel-labelled momentum routes | Collective norm² coefficient | Gluon norm² coefficient |
|---|---:|---:|---:|
| 0 | 2 | .0004488251374 | .001122422156 |
| 1 | 18 | .002645245976 | .022008621061 |
| 2 | 50 | .002653105409 | .022814961636 |
| 3 | 98 | .002653114483 | .022815088751 |

The two channel labels share spatial routes but are orthogonal species.
Spin/color/node components are additional resolved amplitudes, not additional
momentum routes. These dimensionless norms are **first-order coefficient
norms, not all-order probabilities**. Increasing the window keeps all
old full amplitudes and adds new ones; shell 2->3 adds a wave of norm about
3.69e-4. That does not bound the remaining infinite tail. No volume,
longitudinal/continuum matching, higher-order normalization or binding follows.

## Reassessment and next experiment

Advance to the linked **baryon-wave source** experiment, not a larger free
emission scan or a fitted nucleon solve. The physical interaction graph now
has three fermionic emitter slots (with no preferred diquark), fundamental
color/helicity/flavor nodes, both collective nodes and adjoint gluon edges.
This supplies concrete architecture information beyond an optimizer tree.
It does not yet select optimal DNN connectivity, a nonperturbative potential,
compression rank, quantum advantage or physical topological order.

Next pre-evaluate one emission plus a q+=0 classical electromagnetic source
on these arbitrary exterior states. Include photon action on every quark
before/after emission, charged collective propagation and its endpoint,
the crossed fermion contact, the canonical gluon-channel mixed partner and
all source-shifted intermediate modes. Compare complete complex waves with
an independent time-order/covariant representation and pure-gauge pulses;
apply the same exterior map to source operators, not separate fitted heads.
Do not infer that the pair-source calculation alone supplies this result.

After that source checkpoint, reassess the shared action, second-order
normalization and unresolved matching before nonperturbative p/n inference.
The broader objective remains generated p/n, modular nuclei and matched
GTMD/TMD/GPD predictions. No central physics blocker or executive decision
has appeared at this bounded crossing/state step. No production dynamics,
old-loop requantization or BLFQ/H2 reconstruction is selected.

## Files and recovery

Validation: 26 new tests pass with warnings as errors (41.89 s), with a
separate passing check of the subsequently added explicit spin-map formula.
The integrated focused continuation/theory set passes 592 tests with 12
pre-existing warnings (325.88 s). Replay was rerun from the final numerical
source; all seven archived originals match Git 0788bd1c byte-for-byte.
No physics tolerance was weakened. Hash consistency and TeX structure are
document integrity checks, not independent physical validation or a rendered
PDF claim. Tiny cancellation errors are not bitwise-repeatability guarantees.
The final numerical and document sources also pass all 26 new tests and six
theory/hash checks together with warnings as errors: 32 passed in 55.38 s.

- Source: `src/deuteron_wigner/quantum/lf_baryon_emission.py`.
- Tests: `tests/test_lf_baryon_emission.py`.
- Replay: `scripts/evaluate_lf_baryon_emission.py`; results in
  `docs/next_level/lf_baryon_emission_results.json`.
- Seven original current/theory notes: Git `0788bd1c` and
  `computation_handoff/backups/2026-09-10_pre_lf_baryon_emission.tar.gz`.

Use the compatible runtime recorded in the preceding coupling result, with
`PYTHONPATH=src:.`, single-threaded BLAS and bytecode disabled. No public push
or review-mirror export. The complete historical suite is not declared green;
its previously documented synthetic F1 tolerance failure remains untouched.

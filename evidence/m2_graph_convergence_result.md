# Converged controlled-gate H2 solver and matched spin-current measurements

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-06. Scope: **small finite-model quantum/classical benchmark**.
No exact-state seed, StatePrep, or dense-unitary insertion is used in the
graph circuit. Dense classical state geometry is used for preconditioning;
this is not scalable many-body quantum computation or quantum advantage.

## Failure identified and resolved at the declared tolerance

The earlier 37-state matched calculation stopped after 1,248 L-BFGS-B
iterations with a 4.39e-5 GeV² state residual, despite small energy error.
Removing its relative function tolerance alone left a 3.21e-5 residual.
Centering the Hamiltonian INSIDE the measured observable helped precision
but still left 2.12e-5 after 5,000 iterations. A subtraction after measuring
the total energy would not recover the lost precision.

Energy convergence is quadratic in state error, and graph angles attached
to tiny amplitudes have very small gradients. The remedy tested here is the
full projective-state metric, following the general
[quantum natural-gradient construction](https://quantum-journal.org/papers/q-2020-05-25-269/).
The implemented geometry is independently derived for this gate tree:

    J_a = partial_a psi,
    B_a = (I-|psi><psi|)J_a,
    G_ab = Re(B_a† B_b),
    d theta/d tau = -(1/2) G^+ grad_theta <H>.

The step parameter tau has units GeV^-2 for H=M².

The complex state/Jacobian follow the same two-component rotation and phase
updates as the actual controlled gates. Finite differences of EXECUTED
states validate every column on small complex graphs. In a resolved full
tree tangent, B delta theta equals -(H-<H>)psi up to phase.

The metric is factored through the real/imaginary Jacobian, rather than
forming a Gram matrix first and squaring its numerical condition number.
An explicit relative metric eigenvalue cutoff determines unresolved tangent
directions, never dropped Hamiltonian/Fock states. Actual PennyLane energies
and adjoint gradients drive every update. The line search uses the centered
energy; near an eigenstate, a roundoff-sized rise is allowed only when an
independently executed state residual decreases. Success requires that
residual tolerance, not an optimizer message or small angular gradient.

## Actual matched 37-state benchmark

Same proton K2=9, retained N_exc=0, b=.4 GeV, mq=.35 GeV, mg=0,
gs=.3, with both free-Q sectors at cutoff 1. The Hamiltonian is the
energy-local Hermitian reduction at E0=1.4 GeV². The initializer is the
tree-only uniform distribution plus fixed small random phases (seed4901).
The independent eigensolve is performed AFTER optimization.

| Explicit controls | Updates | State residual (GeV²) | Time |
|---|---:|---:|---:|
| metric rcond1e-18, residual target1e-8 | 58 | 7.69114e-9 | 3.62 s |
| metric rcond1e-22, residual target1e-10 | 71 | 8.32933e-11 | 4.42 s |

Both use six qubits and 72 graph parameters. The energy is
1.421430998397409 GeV², agreeing with the independent finite-model
eigenvalue to roundoff. Padding is zero. Reported overlaps infinitesimally
above one are floating-point error, not physical probabilities.
An earlier full-Gram rcond1e-14 run plateaued at 3.36e-8; that failed
run is not promoted to convergence.

## State preparation is verified by observables, not energy alone

At Delta=(.12,.16) GeV the same graph circuit measures the energy-local
matched F1 and positive-intrinsic-parity Pauli diagnostic. The spin-flip
transition is reduced onto the up-helicity fiber using the explicit map
down=i T up:

    A2 = -i 2M/(Delta_x+i Delta_y) T† rho_down,up,
    measured F2 = <(A2+A2†)/2>.

Thus this six-qubit measurement is not a diagonal-only current mistaken
for spin flip. The operator contains the actual opposite-helicity
transition and the full excluded-sector norm/current matching.

The rcond1e-18 run gives F1=.882386818764499 and
F2=4.53008903850e-7. Absolute errors against the exact SAME energy-local
model are 2.01e-12 and 4.84e-16. They lie below the numerically evaluated
spectral bounds 2.41e-8 and 1.39e-11, using

    |Delta <O>| <= 2 ||O|| r/(E1-<H>),  when <H><E1.

Here r is the eigenpair residual and E1 the first excited reference energy;
this is an exact-arithmetic bound evaluated numerically, not an interval
arithmetic certificate or physical uncertainty. The finite-model gap is
.586874018864149 GeV². These E0=1.4 currents differ slightly from the
self-consistent-energy spin-current oracle; compare like with like.

## Reproduction and next quantum target

    python scripts/benchmark_m2_graph_convergence.py --natural --iterations 600 --metric-rcond 1e-18 --spin-current
    python scripts/benchmark_m2_graph_convergence.py --natural --iterations 600 --metric-rcond 1e-22 --residual-tolerance 1e-10
    python -m pytest tests/test_m2_graph_natural_gradient.py tests/test_m2_symmetry_graph_solver.py -q

The 18 graph tests pass, including the actual matched37 optimization and
current-error bound. Twelve tests are new relative to the original six.
The original solver and bibliography are preserved in
`computation_handoff/backups/2026-09-06_pre_canonical_self/pre_solver_and_bib.tar`.

Next replace full O(D) state parameterization and dense O(D²) geometry
with symmetry-/topology-factorized proposals and scalable metric estimates,
benchmarking residuals and observables against these exact finite references.
No production optimizer, hardware target or shot budget is selected.

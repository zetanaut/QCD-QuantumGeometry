# Complete-shell free propagation and a genuine resolution study

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

2026-09-06 continuation of the user-selected canonical Galerkin/H2-first
direction. This is a **bare finite-model** result, not physical calibration,
full QCD, or ultraviolet convergence. The preceding complete order-four
interaction benchmark remains separately identified at320ae968.

## What changed mathematically

For d=n_parton-1 global Jacobi vectors and one common width b, the free
mass operator is the Galerkin projection

    H0_N = P_N [sum_a Q_a^2 + sum_i m_i^2/x_i] P_N,
    R_N(E) = (E-H0_N)^(-1).

It is NOT a harmonic-oscillator confining Hamiltonian. In the established
circular momentum phases, its orbital diagonal is b²(d+sum occupations),
and each radial pair-raising entry is

    +b² sqrt((n_a,+ +1)(n_a,- +1)).

The latter first appears at N_exc=2. Merely raising the old diagonal-energy
helper's cutoff would calculate the wrong operator. That helper now rejects
higher shells explicitly; the new full operator carries every raw column.
R_N is the inverse of the projected operator, not the projection of the
continuum inverse.

Within the ground radial channel at cutoff2, the orbital matrix is
b² [[d,sqrt(d)],[sqrt(d),d+2]], with eigenvalues
b²(d+1 ±sqrt(d+1)). More generally the radial Jacobi matrix has diagonal
b²(2n+d) and off-diagonal b² sqrt((n+1)(n+d)).
Independent generalized-Laguerre quadrature verifies its ground resolvent
at cutoffs2/4 for d=4,5. Independent raw creation ladders verify the
third-shell angular blocks. Full sparse tensor products and species/tree
permutations supply other checks.

Only exact orbital-Lz blocks are diagonalized. Spin/flavor and all color
multiplicities remain, with no source/amplitude threshold. Both Q5 and Q6
packed spaces now support complete integer shells; old diagonal-only
matching APIs cannot silently consume the extended spaces.

## Currents and source orders stay together

The new local good density factors the same exact projected continuum
displacement by particle and by exact incoming/outgoing spin/Lz blocks.
It is P_N rho(Delta) P_N, not an exponential of truncated ladder matrices.
Full-tensor comparisons include both helicity endpoints and both Q5/Q6
families. No Wilson-line or transverse physical current is supplied by
this local J+ calculation.

The separate raw Q5 reference binds the actual canonical cubic and
quartic sources, V(g)=g V1+g² V2, to the new R_N. It preserves all
g²/g³/g⁴ Hamiltonian, norm and two-energy-current coefficients:

    Sigma = V† R_N V,
    N = I + (R_N V)† R_N V = I-d_E H_eff,
    rho_eff(Ef,Ei) = rho_PP + (R_N(Ef)V)† rho_QQ R_N(Ei)V.

The old free-Q compressed parent is reproduced at cutoffs0/1. At larger
cutoffs an independent matrix-free full-Q resolvent checks the matching.
The canonical QQ/Q6 terms from the separate order-four benchmark are NOT
silently reused at an incompatible cutoff. The subsequent interacting-shell
extension is documented in [the current result](m2_interacting_shell_result.md);
the five-cutoff free-Q study here remains a separate reference.

Exact-block five-body permutations replace allocation of forbidden spin
rows. Full uncompressed transformations verify the equality. A legacy
normalization test exposed MRRR eigenvector orthogonality loss for a
near-identity norm after harmless source-roundoff changes. Its symmetric
inverse square root now uses divide-and-conquer eigenvectors, with the
same positive-norm requirement and unchanged conservation tolerances.
This alters no physical norm eigenvalue and performs no clipping.

## Executed fixed-input resolution study

K2=9; retained N_exc=0, dimension37; b=.4 GeV; mq=.35 GeV; mg=0;
g=.3; proton; Delta=(.12,.16) GeV. No parameter is refitted.
Each mass solves the energy-dependent H2 equation and is normalized with
the matched metric. Parent residuals below are on the raw reconstructed
P+Q states, not just the compact eigenproblem.

| Excluded N_exc | raw Q5 dimensions, gg / sea | M² (GeV²) | excluded probability | matched electromagnetic density |
|---|---:|---:|---:|---:|
|0|630 /4500|1.421431301525934|7.82258782480e-7|.882387059492464|
|1|4950 /31500|1.421430409460545|1.22227353213e-6|.882386786096472|
|2|22230 /130500|1.421428801867490|1.64193982776e-6|.882386604535556|
|3|72630 /409500|1.421427448619435|2.12881494026e-6|.882386223271088|
|4|196380 /1075500|1.421425705806639|2.53251287865e-6|.882386043861259|

The last raw parent dimension is1,271,917. Reconstructed residuals are
1.45e-15,1.34e-15,2.49e-15,1.42e-15,1.53e-15 GeV².
The lowest free-Q energies are3.2125,3.2125,3.014729123600034,
3.014729123600034,2.915094604274836 GeV².
These numbers are finite-model diagnostics, not nucleon predictions.

At E=1.4 GeV² the weighted self-energy Frobenius norms are
.00338717913,.00774164562,.01017274370,.01701603006,.01902184155 GeV².
An intentionally wrong diagonal-only control agrees to roundoff at0/1,
but differs by .000298648497,.001279786572,.001678636936 GeV² at2/3/4.
The change is material at the operator level even though this particular
weakly mixed retained state has a small mass response.

The instantaneous-source norms are equal at0/1 and again at2/3 because
the new coincident radial modes enter in even shells. At4 they grow again:
the gg quartic norms at0/2/4 are .01107699819,.01918592367,.02713299346;
the sea norms are .04687092906,.07882955637,.10969805569 (GeV² per g²).
This is a selection pattern, NOT evidence of ultraviolet convergence.
Neither shrinking resolution errors nor renormalized cutoff independence
has been established by this sequence.

## Reproduction, preservation, and next step

    python -m pytest tests/test_m2_shell_free_propagation.py tests/test_m2_shell_current_reference.py -q
    python scripts/benchmark_m2_shell_resolution.py --cutoffs 0 1 2 3 4

The new increment has60 direct tests. All178 named propagation, current,
source, old free-Q matching and six-body-space tests pass in144.64s.
The36 additional matched-H2, matched-spin-current and graph-metric
tests pass in70.66s after the normalization correction. These are two
separate invocations; the earlier55-test normalization subset also passes.
The last full named M2+Q regression before this increment is907 tests at
320ae968, not a claim of a full current967-test or repository-wide run.

Pre-edit copies at9ec781f3:

- computation_handoff/backups/2026-09-06_pre_canonical_self/pre_higher_shell_propagation.tar
- computation_handoff/backups/2026-09-06_pre_canonical_self/pre_free_reference_whitening.tar

The interacting Q5/Q6 and raw current/backflow extension is now available
in the subsequent result above. Next test shared regulator
transport and physical finite matching. Wick field traces are still distinct
inputs and are not identified with this external shell cutoff by fiat.
Physical calibration remains unselected. Nuclear/GTMD construction and
topology-factorized quantum scaling remain ahead; no workstream is complete.

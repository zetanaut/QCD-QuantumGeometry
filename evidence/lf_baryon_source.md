# Complete mixed LF baryon source wave at the declared order

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. One canonical/retained emission and one classical
q+=0 electromagnetic insertion. **Free-reference quark states, not bound
protons/neutrons, a physical form factor or a production model.**

**Envelope audit resolved:** an extra ordinary-source Fourier coefficient
was removed before committing this checkpoint. Both development solvers had
shared the incorrect input, so agreement alone did not certify the requested
pulse. All ordinary results below were regenerated with the corrected sin^4
envelope. A direct trigonometric pulse/first-/second-derivative regression
now checks that input independently; all 26 new tests pass with warnings as
errors. The pure-gauge branch was unaffected. The earlier ordinary numbers
are superseded, not accepted scientific evidence.

## Scientific question and result

The preceding emission layer and crossed fermion contact did not supply
the complete baryon source response. The photon can act on the emitter,
either spectator, or the emitted charged collective field, with both time
orders, constrained contacts and a collective normalization endpoint.
This complete mixed wave now has independently checked source ingredients
and agrees with a separate full-Dirac, physical-basis initial-value solve.
Whole-wave pure-gauge cancellation passes; deleting needed intermediate
states or individual required terms fails.

An important exception to a gauge-only diagnostic is now explicit: removing
the entire spectator response can still pass pure-gauge cancellation while
changing an ordinary source wave. A separately gauge-invariant physical
contribution is not optional. Conservation alone cannot qualify a later
state/network compression or a chosen set of current measurements.

## Canonical gluon contact and collective endpoint

The physical EM convention is D_EM=partial+i e Q A. The preserved canonical
color convention is D_color=partial-i gs t.Ag, with Ag+=0. Therefore

    B_EM=-alphaT.Q A_T,       B_g=+alphaT.t Ag_T,
    h_chi=eQA- - gs t Ag- + B (p+ - e Q A+)^-1 B / 2.

Differentiating the latter at one gs and one e gives

    K_g = chi_a† { B_EM B_g/a+ + B_g B_EM/p+
           + A+ [ B0(a) Q B_g/(a+)² + B_g Q B0(p)/(p+)² ] } chi_p
           / (2 sqrt(2 k+ V_cell)),

where a+k=p+q, q+=0, and the quantum gluon's coupling is stripped. Both
physical gluon helicities and all eight colors remain. The independent
instantaneous-propagator formula uses Gamma_g=-slash(eps_g*) t:

    ubar(a) [ Q slash(A) gamma+ Gamma_g/(2a+)
              + Gamma_g gamma+ Q slash(A)/(2p+) ] u(p)
              / sqrt(8 a+ k+ p+ V_cell).

The quark/collective contact is the preceding same-action crossing,
unchanged. No gg or quark quartic source is silently inserted at this order.
The gluon's constrained longitudinal polarization is kept in the full-Dirac
test; it drops out of this instantaneous product through gamma+ gamma+=0,
not an ad hoc polarization deletion. The gluon contact satisfies the complete
fundamental/adjoint color-intertwiner identity, not only a scalar color factor.
Canonical LF constraints and instantaneous terms are established field
theory; see [Srivastava and Brodsky, Section 4](https://arxiv.org/html/hep-ph/0011372v2).
The retained S/P sector remains a qualified project reference, not canonical
QCD or a selected sigma-model production action.

For the collective source, the independent covariant vertex is

    Gamma_phi^mu = [(k1+k0)^mu I
           + C { q² (k1+k0)^mu - (k1²-k0²) q^mu }] Q_phi.

At fixed q+=0 and omega=q-, its normalized off-shell energy polynomial
obeys

    Gamma_phi.A/(2k+) = J + D_out W + W D_in,
    D_in=w-h_in, D_out=w+omega-h_out,
    W=A+ (I+q² C) Q_phi/(2k+).

This identity follows by differentiating the actual energy polynomial,
independently of the explicit canonical J0+omega J1+omega² J2 formula.
It also passes reversed-source Hermiticity. With an on-shell outgoing
collective field, the pole graph becomes

    J D_in^-1 V + W V.

Thus the **outgoing endpoint is +W V**, not the preceding incoming-endpoint
matrix order V W, and not zero. Its units are consistent: W is dimensionless
when A is in GeV; differentiating W with respect to A has GeV^-1 units.
The canonical J includes the source's first/second time derivatives.
An endpoint-zero source pulse does not remove the internal emission endpoint.
The independent C matrix survives; D/E start at two photons and are not
set to zero in the action.

## Order-complete exterior-state evolution

Let lambda multiply the canonical/retained first-emission operator only as
an order-counting parameter; gs and the two Yukawas remain distinct inputs.
Let a be the classical source amplitude. Expand

    psi = psi00 + lambda psi10 + a psi01 + lambda a psi11 + ... .

The initial sector I contains the supplied quark exterior state. A is the
source-first quark sector; Y contains one collective field or gluon; F is
the projected mixed final sector. The actual evolution equations are

    i dpsi00/dt = H_I psi00,
    i dpsi01/dt = H_A psi01 + J_AI(t) psi00,
    i dpsi10/dt = H_Y psi10 + f(t) V_YI psi00,
    i dpsi11/dt = H_F psi11 + f(t) V_FA psi01 + J_FY(t) psi10
                              + f(t) [K_f(t)+K_out(t)] psi00.

Here f=sin^4(pi t/T). J_FY includes the emitter, all spectators and the
charged collective source. The gluon is electromagnetically neutral;
its emitting quark and mixed contact are not. K_out=W_FY V_YI. Fermionic
operator words act on every occupied mode, retain sorting signs and sum
amplitudes before taking norms. Route labels are not extra physical sectors.

The spectral implementation diagonalizes only the free collective node
blocks and integrates the lower-order jets analytically before an outer
Gauss quadrature. The independent implementation uses full-Dirac current
and contact expressions, the covariant collective source map, ordered-list
fermion signs, and a sparse physical-basis initial-value solve. It does
not reuse the spectral integration or block diagonalization.
Both routes reuse the same previously independently qualified canonical
q->qg emission leaf; the new source/contact, retained emission, CAR and
time-evolution comparisons are independent as described above.

The input is explicitly provided; no nucleon state is selected by default.
The benchmark reuses the preceding canonical-spin exterior map. Arbitrary
correlated fixed-quark-number inputs are supported, and a three-quark
unequal-momentum packet/projection is tested without renormalizing its
original .375 squared weight. q+=0 cannot supply the positive plus momentum
needed for a new pair from vacuum at this order; this does not solve the
constrained zero mode or nonzero-plus source-created sectors.

## Domain and independent tests

Final quark and boson transverse momenta are projected into the declared
square. Source-first momenta and emission-first intermediates are generated
before that projection. If the final shell is N and the source has integer
transverse shift delta, the auxiliary emission enumeration extends to
N+max(abs(delta)); all original and source-shifted incoming modes are included.
This suffices for one source shift, not arbitrary source order. Unrepresented
final momenta remain outside the claim.

The three-quark p=(.375,.2,0) GeV, q=(0,-.2,.2) GeV shell-one benchmark
has 21 input, 72 source-first, 92,160 emission-first and 82,944 final complex
coordinates. The mixed sector has 6,480 physical quark-mask/boson-momentum
blocks with 8 or 16 resolved components each. There are 5,040 auxiliary
emission-first blocks outside the final square; this count includes all
enumerated blocks, not a claim that every block contributes to every source.
Removing those intermediates gives a nonzero false pure-gauge wave.

The physical C source and crossed S/P/gluon contacts are tested in all four
components. Complete p/n wave checks include opposite transfer signs,
mixed source polarizations, total charge and momentum, independent Dirac
formulae, refined time integration and quadrature. Both signs of a real
cosine source have Fourier amplitude 1/2. They carry different final total
momenta, so their squared norms add; they must not be added as the same
momentum amplitude or reported as two probabilities.

The ordinary mixed wave has GeV^-1 units per source amplitude; its squared
norm is a GeV^-2 derivative coefficient, not an all-order transition
probability. A pure-gauge theta amplitude is dimensionless. No independent
observable head, source rescaling or physical fit is used. The saved replay
records both p/n and source signs, ODE refinement, omitted-source witnesses
and the full shell-one to shell-two wave comparison. The norm of newly
represented output is separate from agreement within the old window and
does not bound the remaining infinite tail.

## Reassessment: return to shared binding and matching

### Numerical witnesses from the saved replay

For the A+ cosine source with Fourier coefficient 1/2 on each branch:

| Reference input | Transfer sign | Mixed squared-wave coefficient (GeV^-2) |
|---|---:|---:|
| proton, canonical spin up | + | .137740356483 |
| proton, canonical spin up | - | .0458030826936 |
| neutron, canonical spin up | + | .0958708570025 |
| neutron, canonical spin up | - | .0327018413952 |

These are not form factors or probabilities. Directional asymmetry is not
removed by rescaling; the initial transverse momentum and final square are
not symmetric between these two transfer choices. The refined independent
Dirac ODE error in the positive proton branch is 1.43e-11 GeV^-1, improved
from 1.33e-9 under a hundred-fold tighter ODE tolerance. The 64->96-node
quadrature change is 5.15e-16 GeV^-1.

Pure-gauge whole-wave norms are 7.01e-17 and 2.49e-17 for the two branches.
Omitting outside emission-first intermediate blocks gives a false .0338640
wave in the positive branch; that particular omission is invisible in the
negative branch. Omitting the fermion contact gives .0303777 and .0163677.
Thus even a useful negative control need not be sensitive at every point.

In the ordinary positive proton branch, omitted before/after-quark,
collective, fermion-contact, collective-endpoint, outside-intermediate and
spectator contributions change the wave by .0685166, .423098, .0961882,
.261863, .0433669, .329556 and .0979732 GeV^-1, respectively.
The spectator-omitted pure-gauge norm remains 4.40e-17: its missing ordinary
response cannot be diagnosed by gauge cancellation alone.

The shell-one to shell-two comparison is particularly important. All old
amplitudes agree to 5.12e-16 GeV^-1, yet the newly represented wave has norm
**.3945342 GeV^-1**, larger than the old wave norm .3711339 GeV^-1.
The total squared coefficient grows from .1377404 to .2933976 GeV^-2
(a factor of 2.13008).
The larger final carrier has 281,088 components; it is not declared
converged. An exact finite-window current or a converged state-energy test
must not be used to certify the source-sensitive output space. Preserve
this cost/information witness in the binding/network pre-evaluation, not
as a reason to silently discard the extra amplitudes or fit their absence.

### Direction after these controls

This completes the bounded first-emission/one-source construction, not a
nonperturbative baryon model. The architecture now has explicit fermionic
state assembly, canonical/retained emission, spectator interference and
linked electromagnetic partners. These are useful physical constraints on
the neural/tensor/quantum representation, beyond optimizer geometry.

The next task is **a physics pre-evaluation of shared nonperturbative binding
and standard matching**, not automatic continuation to a higher-order loop,
larger Fock-space or source-control campaign. Re-read the original project
construction and current quantum-geometric specification; identify which
state-generating operator/functionals are established, which flexible soft
directions must be learned, and which retained contributions are subtracted
at the common action/source level. Specify the relevant matching domain,
physical parameters, independent p/n observables and decisive controls
before implementing a candidate solve.

The free reference mass .27 GeV, gs=.43, two Yukawas and collective mass/C
matrices are illustrative inputs, not matched nucleon dynamics. The two-O(4)
reference is not promoted to production because these checks pass. There
is no demonstrated confinement, interacting composite spin qualification,
fully matched quantum current, physical p/n covariance repair, continuum
limit, quantum advantage or nuclear/GTMD prediction. Topology and an ansatz
cannot supply those missing dynamics. Preserve all earlier work and failures;
do not restart the frozen BLFQ/H2 model or select production physics silently.

## Implementation, documents and recovery

The corrected source passes all 26 new direct tests with warnings treated as
errors (245.92 s). Its complete replay was regenerated, including both p/n
source signs, time/quadrature refinement and window enlargement. No physics
tolerance was weakened. Seven archived originals match Git e21d891d
byte-for-byte. The new TeX section has balanced braces/environments and
unique labels; document integrity checks do not certify historical physics.
The final corrected-source integration run passes **618 tests with 12
pre-existing warnings in 503.66 s**. This includes the new 26 tests and the
preserved continuation/theory tests listed below, not the full historical
repository suite. The earlier 617-test development run used the superseded
ordinary envelope and is not the final-source acceptance evidence.
The complete historical suite is not declared green: the previously
documented synthetic F1 fit-tolerance failure remains unchanged and outside
this focused selection. Six current theory/document-integrity checks pass
separately with warnings as errors (0.57 s).

Use the compatible runtime from the preceding coupling result, with
`PYTHONPATH=src:.`, single-threaded BLAS and bytecode disabled. The exact
focused integration selection is:

```sh
python -m pytest -q -p no:cacheprovider \
  tests/test_lf_baryon_source.py \
  tests/test_lf_baryon_emission.py \
  tests/test_lf_pair_electromagnetic.py \
  tests/test_finite_lf_coupling.py \
  tests/test_nonstationary_lf_source.py \
  tests/test_quantum_source_generator.py \
  tests/test_source_network_transport.py \
  tests/test_joint_source_matching.py \
  tests/test_quantum_collective_compton.py \
  tests/test_quantum_collective_current.py \
  tests/test_quantum_pair_threshold.py \
  tests/test_nonzero_momentum_current.py \
  tests/test_retained_mode_quantum.py \
  tests/test_collective_quark_matching.py \
  tests/test_transverse_source_matching.py \
  tests/test_common_action_pilot.py \
  tests/test_causal_spectral_matching.py \
  tests/test_momentum_chiral_matching.py \
  tests/test_shared_chiral_dynamics.py \
  tests/test_chiral_soft_matching.py \
  tests/test_nonlocal_source.py \
  tests/test_joint_state_dynamics.py \
  tests/test_deuteron_information.py \
  tests/test_canonical_lf_network.py \
  tests/test_graded_fock_network.py \
  tests/test_modular_gauge_network.py \
  tests/test_modular_baryon_algebra.py \
  tests/test_current_theory_contract.py
```

- Source: `src/deuteron_wigner/quantum/lf_baryon_source.py`.
- Tests: `tests/test_lf_baryon_source.py`.
- Replay: `scripts/evaluate_lf_baryon_source.py`, with saved results in
  `docs/next_level/lf_baryon_source_results.json`.
- Originals: Git `e21d891d` and seven current/theory notes in
  `computation_handoff/backups/2026-09-10_pre_lf_baryon_source.tar.gz`.

The main TeX and current pickup/roadmap/machine index carry this checkpoint.
Three unsupported slash-macro uses in the TeX are corrected to standard
LaTeX notation without changing their equations; originals are backed up.
No PDF rendering or full historical re-audit is claimed. No public push or
review-mirror export. Unrelated user files and old source kernels are intact.

"""Physical retained-shell comparison with separate H2 endpoint metrics.

Only the retained oscillator cutoff changes here. Q5/Q6 coordinates and
all physical/regulator/finite-condition parameters must stay fixed. This
is not a comparison of different longitudinal grids or a selected state.
"""
from dataclasses import dataclass
import numpy as np

from .m2_gaussian_cluster_h2 import require_same_cluster_definition
from .m2_matched_h2_reduction import nested_pilot_isometry


@dataclass(frozen=True)
class RetainedMetricInjection:
    raw: np.ndarray
    normalized: np.ndarray
    energy_GeV2: float


def retained_ward_boundary(hamiltonian_out, hamiltonian_in, density, raw_injection):
    """Low-minus-compressed-high Ward RHS at fixed Q spaces.

    For Pi=E Edag and Q=I-Pi, this is
    -Edag Hf Q rho E + Edag rho Q Hi E.
    Hlow=Edag Hhigh E and rholow=Edag rhohigh E must be independently
    established; the exact same raw E binds both energy endpoints.
    Matched plus currents then restrict exactly and their unequal-energy
    terms cancel in the difference. This predicts a transverse contraction
    difference; it is NOT a current prescription, a repair or a minus
    component. It is linear in H, so also applies to each coupling power.
    """
    hf, hi, rho, e = (np.asarray(a, complex) for a in
                      (hamiltonian_out, hamiltonian_in, density, raw_injection))
    if (e.ndim != 2 or not all(e.shape) or e.shape[1] > e.shape[0]
            or any(a.shape != (e.shape[0], e.shape[0]) for a in (hf, hi, rho))
            or not all(np.all(np.isfinite(a)) for a in (hf, hi, rho, e))):
        raise ValueError('finite high-space matrices and a tall raw isometry required')
    if not np.allclose(e.conj().T@e, np.eye(e.shape[1]), atol=5e-13, rtol=0.):
        raise ValueError('raw occupation-label map must be an isometry')
    q = np.eye(len(e))-e@e.conj().T
    return -e.conj().T@hf@q@rho@e+e.conj().T@rho@q@hi@e


def retained_metric_injection(lower, larger):
    """T=N_large^(1/2) E N_small^(-1/2), after actual wave/H/N checks.

    Tdag T=I follows from Nsmall=Edag Nlarge E, NOT from commuting E
    through whitening. Every reference wave column must restrict through
    the same physical occupation-label embedding. Both energy endpoints
    need their own T. A changed Q space is a separate resolution effect.
    """
    a, b = lower.owner, larger.owner
    if lower.energy_GeV2 != larger.energy_GeV2:
        raise ValueError('equal energy for each retained-space endpoint map required')
    if not require_same_cluster_definition(a.pilot, b.pilot):
        raise ValueError('explicit identical common-cluster definitions required')
    for left, right in ((a.spaces5, b.spaces5), (a.spaces6, b.spaces6)):
        if left.keys() != right.keys():
            raise ValueError('identical Q-sector spaces required')
        for n, small in left.items():
            large = right[n]
            for key in ('sector', 'total_k2', 'cutoff', 'target', 'jz_twice', 'ncolor'):
                if getattr(small, key) != getattr(large, key):
                    raise ValueError('fixed Q coordinates required; mismatch on '+key)
            if (small.longitudinal != large.longitudinal or small.orbitals != large.orbitals
                    or not np.array_equal(small.spins, large.spins)
                    or not np.array_equal(small.coordinates, large.coordinates)):
                raise ValueError('fixed ordered physical Q coordinates required')
    e = nested_pilot_isometry(a.pilot, b.pilot)
    for low_waves, high_waves in zip((*lower.five_wave_coefficients, lower.six_wave_coefficient),
                                    (*larger.five_wave_coefficients, larger.six_wave_coefficient)):
        for n in low_waves:
            np.testing.assert_allclose(high_waves[n]@e, low_waves[n], atol=3e-14, rtol=3e-12)
    for low, high in ((lower.effective_hamiltonian, larger.effective_hamiltonian),
                      (lower.metric, larger.metric)):
        np.testing.assert_allclose(e.conj().T@high@e, low, atol=3e-14, rtol=3e-12)
    # Solve for N_large^(1/2) E N_small^(-1/2), retaining the complete
    # positive metric. No diagonal-only approximation or eigenvalue clipping.
    t = np.linalg.solve(larger.whitener, e@lower.whitener)
    np.testing.assert_allclose(t.conj().T@t, np.eye(t.shape[1]), atol=4e-13, rtol=0.)
    return RetainedMetricInjection(e, t, lower.energy_GeV2)

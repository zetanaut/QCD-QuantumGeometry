"""Tree-level constrained-field identities in symmetric light-front units.

These algebraic decompositions fix instantaneous signs and denominators.
They are not a continuum loop prescription at p+=0, nor a complete Ward
identity for the truncated interacting Hamiltonian.
"""
import numpy as np

from deuteron_wigner.bridge.modes.core import GAMMA, GAMMA_PLUS

METRIC = np.diag([1., -1., -1., -1.])
NULL = np.array([1., 0., 0., -1.]) / np.sqrt(2)


def _four(value):
    p = np.asarray(value, float)
    if p.shape != (4,) or not np.all(np.isfinite(p)):
        raise ValueError("finite Cartesian four-vector (E,px,py,pz) required")
    return p


def slash(momentum):
    p = _four(momentum)
    return sum(g * v for g, v in zip(GAMMA, METRIC @ p))


def fermion_propagator_parts(momentum, mass_GeV):
    """Return (covariant, propagating on-shell numerator, instantaneous).

    Common factor i suppressed:
    (slash p+m)/(p²-m²) =
      (slash p_on+m)/(p²-m²) + gamma+/(2p+).
    The identity also holds for negative p+; replacing p+ by abs(p+) fails.
    """
    p = _four(momentum)
    mass = float(mass_GeV)
    pp = float(NULL @ METRIC @ p)
    den = float(p @ METRIC @ p - mass**2)
    if not np.isfinite(mass) or mass < 0 or pp == 0 or den == 0:
        raise ValueError("nonnegative mass, nonzero p+ and off-shell momentum required")
    pm_on = (p[1]**2 + p[2]**2 + mass**2) / (2 * pp)
    on = np.array([(pp + pm_on) / np.sqrt(2), p[1], p[2], (pp - pm_on) / np.sqrt(2)])
    covariant = (slash(p) + mass * np.eye(4)) / den
    propagating = (slash(on) + mass * np.eye(4)) / den
    instantaneous = GAMMA_PLUS / (2 * pp)
    return covariant, propagating, instantaneous


def gluon_propagator_parts(momentum):
    """Return (two-term axial, doubly transverse, instantaneous) numerators.

    Denominator p² and common i are suppressed. D_DT + p²*n*n/(p+)²
    equals the two-term axial numerator. For conserved endpoint currents,
    division by p² then gives the covariant -g exchange. At zero p+ a
    prescription is needed; this routine intentionally does not supply one.
    """
    p = _four(momentum)
    pp = float(NULL @ METRIC @ p)
    if pp == 0:
        raise ValueError("nonzero p+ required")
    p2 = float(p @ METRIC @ p)
    axial = -METRIC + (np.outer(NULL, p) + np.outer(p, NULL)) / pp
    instantaneous = p2 * np.outer(NULL, NULL) / pp**2
    return axial, axial - instantaneous, instantaneous

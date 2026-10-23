"""Independent scalar-composite controls, NOT nucleon or QCD dynamics.

Constant Bethe--Salpeter vertex, equal constituent masses m=1, 0<M/m<2,
unit charge on the active scalar and a neutral distinguishable spectator.
Inputs q2 mean Q^2/m^2 >= 0, not Minkowski q.q. The composite mass is an
input; this module does not solve a binding Hamiltonian. All routes use ONE
charge normalization. No form-factor fit or component-specific rescaling.

Reference: Choi and Ji, arXiv:hep-ph/9906225, Eqs. (12)--(14), (20).
Our derivation and scope: docs/next_level/quantum_geometric_scalar_pilot.md.
The three numerical routes have different integration variables: covariant
Feynman parameters, direct transverse overlap, and longitudinal LF residues.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss


def _domain(mass_ratio, q2=0.0):
    if not np.isfinite(mass_ratio) or not 0.0 < mass_ratio < 2.0:
        raise ValueError("Require 0 < M/m < 2; threshold is outside this control.")
    if not np.isfinite(q2) or q2 < 0.0:
        raise ValueError("q2 denotes finite spacelike Q^2/m^2 >= 0.")


@lru_cache(maxsize=12)
def _unit_rule(order):
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 8:
        raise ValueError("Quadrature order must be an integer >= 8.")
    z, w = leggauss(order)
    z, w = (z + 1.0) / 2.0, w / 2.0
    z.setflags(write=False)
    w.setflags(write=False)
    return z, w


@lru_cache(maxsize=32)
def charge_integral(mass_ratio):
    """I0 in units m^-2, by adaptive 1D quadrature; shared by every route."""
    _domain(mass_ratio)
    value, error = quad(
        lambda x: x * (1 - x) / (1 - mass_ratio**2 * x * (1 - x)),
        0, 1, epsabs=2e-13, epsrel=2e-13,
    )
    if error > 1e-11 * max(1.0, abs(value)):
        raise ArithmeticError("Charge integral unresolved; refine before using this mass.")
    return value


def triangle_form_factor(mass_ratio, q2, order=96):
    """Covariant triangle on the b,c simplex, with spectator a=1-b-c."""
    _domain(mass_ratio, q2)
    z, w = _unit_rule(order)
    b = z[:, None]
    c = (1 - b) * z[None, :]
    a = 1 - b - c
    denominator = 1 - mass_ratio**2 * a * (1 - a) + q2 * b * c
    integral = np.sum(w[:, None] * w[None, :] * (1 - b) * a / denominator)
    return float(integral / charge_integral(mass_ratio))


def wavefunction(mass_ratio, x, k_squared):
    """psi with measure dx d^2k/(16 pi^3); momenta in units of m.

    x is the ACTIVE constituent's fraction; the spectator has fraction 1-x.
    Equal masses make A(x) symmetric. The large-k tail is 1/k^2, so the
    unregulated transverse second moment is NOT finite.
    """
    _domain(mass_ratio)
    x, k_squared = np.broadcast_arrays(x, k_squared)
    if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(k_squared))
            or np.any(x < 0) or np.any(x > 1) or np.any(k_squared < 0)):
        raise ValueError("Require 0 <= x <= 1 and finite k_squared >= 0.")
    normalization = 4 * np.pi / np.sqrt(charge_integral(mass_ratio))
    return normalization * np.sqrt(x * (1 - x)) / (
        k_squared + 1 - mass_ratio**2 * x * (1 - x)
    )


def overlap_form_factor(mass_ratio, q2, order=96, radial_scale=1.0):
    """q+=0 J+ from direct x,r,phi integration; no angular analytic reduction.

    r=radial_scale*t/(1-t) covers [0,infinity); no finite UV cutoff.
    The fixed 2*order trapezoidal azimuth rule is refined with x and r.
    This uses the continuous charge normalization, NOT a discrete renorm.
    """
    _domain(mass_ratio, q2)
    if not np.isfinite(radial_scale) or radial_scale <= 0:
        raise ValueError("radial_scale must be positive and finite.")
    z, w = _unit_rule(order)
    r = radial_scale * z / (1 - z)
    radial_weight = w * r * radial_scale / (1 - z)**2
    cosine = np.cos(2 * np.pi * np.arange(2 * order) / (2 * order))
    integral = 0.0
    # Bounded memory: no order^3 tensor is retained.
    for x, wx in zip(z, w):
        shift = (1 - x) * np.sqrt(q2)
        a = 1 - mass_ratio**2 * x * (1 - x)
        first = r**2 + a
        second = r[:, None]**2 + shift**2 + a + 2 * r[:, None] * shift * cosine
        angular_mean = np.mean(1 / second, axis=1)
        integral += wx * x * (1 - x) * np.sum(radial_weight * angular_mean / first)
    return float(2 * integral / charge_integral(mass_ratio))


@dataclass(frozen=True)
class LongitudinalParts:
    beta: float
    valence: float
    pair: float

    @property
    def total(self):
        return self.valence + self.pair


def longitudinal_form_factor(mass_ratio, q2, order=96, radial_scale=1.0):
    """qT=0, P'+=beta P+: J+ with both k^- contour intervals kept.

    Returns signed valence and pair CURRENT contributions, not probabilities.
    Uses s=kT^2=radial_scale^2*t/(1-t), independently of the overlap map.
    q2=M^2(1-beta)^2/beta. Only J+ is evaluated: not an all-component test.
    """
    _domain(mass_ratio, q2)
    if not np.isfinite(radial_scale) or radial_scale <= 0:
        raise ValueError("radial_scale must be positive and finite.")
    z, w = _unit_rule(order)
    eta = q2 / mass_ratio**2
    beta = 1 / (1 + eta / 2 + np.sqrt(eta + eta**2 / 4))
    s = radial_scale**2 * z[None, :] / (1 - z[None, :])
    ws = w[None, :] * radial_scale**2 / (1 - z[None, :])**2
    y = beta * z[:, None]  # spectator fraction, 0<y<beta
    first = s + 1 - mass_ratio**2 * y * (1 - y)
    # P'^-=M^2/(2 beta P^+): the final-state mass term carries 1/beta.
    second = beta * (s + 1) - mass_ratio**2 * y * (beta - y) / beta
    valence = np.sum(beta * w[:, None] * ws * (1 + beta - 2*y) * y / (first * second))
    pair = 0.0
    if q2 > 0:
        y = beta + (1 - beta) * z[:, None]  # beta<y<1
        first = s + 1 - mass_ratio**2 * y * (1 - y)
        second = s + 1 + mass_ratio**2 * (y - beta) * (1 - y) / beta
        # The interval Jacobian (1-beta) cancels the residue's denominator.
        pair = np.sum(w[:, None] * ws * (1 + beta - 2*y) * (1 - y) / (first * second))
    normalization = (1 + beta) * charge_integral(mass_ratio)
    return LongitudinalParts(float(beta), float(valence / normalization), float(pair / normalization))


def charge_radius_squared(mass_ratio):
    """-6 dF/d(Q^2), in m^-2; analytic derivative of the reference integral."""
    _domain(mass_ratio)
    value, _ = quad(
        lambda x: x * (1 - x)**3 / (1 - mass_ratio**2 * x * (1 - x))**2,
        0, 1, epsabs=2e-13, epsrel=2e-13,
    )
    return value / charge_integral(mass_ratio)

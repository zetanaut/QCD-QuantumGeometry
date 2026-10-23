"""Joint scalar state/current compression, not a learned physical model.

One tail-factored propagator core g(a,t)=1/[a+(1-a)t], t=s/(1+s),
h(a,s)=(1-t)g(a,t), supplies the bound-state AND pair-current factors.
The core domain follows from M/m<=1.9 and Q^2/m^2<=4. SVD compresses
Chebyshev coefficient axes (virtuality, radial coordinate), not physical
particle entanglement. All currents keep the original continuous I0.
See docs/next_level/quantum_geometric_scalar_compression.md for derivation.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.chebyshev import chebval, chebvander

from .scalar_composite_reference import _domain, _unit_rule, charge_integral


A_MIN = 1 - 1.9**2 / 4
A_MAX = 1 + 4.0 / 4


def _a_coordinate(a):
    if not np.isfinite(a) or not A_MIN - 1e-14 <= a <= A_MAX + 1e-14:
        raise ValueError("Virtuality is outside the declared scalar core interval.")
    return 2 * (a - A_MIN) / (A_MAX - A_MIN) - 1


@dataclass(frozen=True)
class RadialSlice:
    a: float
    coefficients: object = None

    def g(self, t):
        t = np.asarray(t)
        if not np.all(np.isfinite(t)) or np.any(t < 0) or np.any(t > 1):
            raise ValueError("Compactified radial coordinate must lie in [0,1].")
        if self.coefficients is None:
            return 1 / (self.a + (1 - self.a) * t)
        return chebval(2*t - 1, self.coefficients)

    def h(self, s):
        s = np.asarray(s)
        if not np.all(np.isfinite(s)) or np.any(s < 0):
            raise ValueError("Radial momentum squared must be finite and nonnegative.")
        # 1/(1+s), rather than subtracting t from 1, retains the UV tail.
        inverse = 1 / (1 + s)
        return inverse * self.g(s * inverse)


class ExactCore:
    """Cheap analytic baseline; no numerical coefficient table."""
    stored_coefficients = 0
    stored_bytes = 0

    def bind(self, a):
        _a_coordinate(a)
        return RadialSlice(a)


class PolynomialCore:
    """A dense interpolant or actual low-rank factors; not both in memory.

    Complex coefficients are supported. For the real scalar reference all
    factors are real. Basis-phase controls are separate from physical phases.
    """

    def __init__(self, coefficients, rank=None):
        c = np.array(coefficients, copy=True)
        if c.ndim != 2 or min(c.shape) < 2 or not np.all(np.isfinite(c)):
            raise ValueError("A finite two-dimensional coefficient matrix is required.")
        self.degree = tuple(n-1 for n in c.shape)
        self.rank = rank
        if rank is None:
            self.coefficients = c
            self.left = self.right = None
            arrays = (c,)
        else:
            if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= min(c.shape):
                raise ValueError("Rank must be an integer within the coefficient matrix dimensions.")
            u, singular, vh = np.linalg.svd(c, full_matrices=False)
            self.left = np.array(u[:, :rank] * singular[:rank], copy=True)
            self.right = np.array(vh[:rank], copy=True)
            self.coefficients = None
            arrays = (self.left, self.right)
        for a in arrays:
            a.setflags(write=False)
        self.stored_coefficients = sum(a.size for a in arrays)
        self.stored_bytes = sum(a.nbytes for a in arrays)

    def bind(self, a):
        z = _a_coordinate(a)
        if self.rank is None:
            radial = chebval(z, self.coefficients)
        else:
            radial = chebval(z, self.left) @ self.right
        return RadialSlice(a, radial)


def build_core(degree=48, rank=None):
    """Interpolate the known core, with no mass/transfer form-factor samples.

    Chebyshev--Lobatto collocation is solved along both axes; we compress its
    coefficients, not a weighted state Schmidt matrix. The SVD norm alone is
    therefore not an observable-error bound.
    """
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 8:
        raise ValueError("Interpolation degree must be an integer >= 8.")
    nodes = np.cos(np.pi * np.arange(degree+1) / degree)
    a = A_MIN + (nodes + 1) * (A_MAX - A_MIN) / 2
    t = (nodes + 1) / 2
    samples = 1 / (a[:, None] + (1 - a[:, None]) * t[None, :])
    vander = chebvander(nodes, degree)
    coefficients = np.linalg.solve(vander, np.linalg.solve(vander, samples).T).T
    return PolynomialCore(coefficients, rank)


def _pilot_domain(mass_ratio, q2=0):
    _domain(mass_ratio, q2)
    if mass_ratio > 1.9 or q2 > 4:
        raise ValueError("This pilot is scoped to M/m<=1.9 and Q^2/m^2<=4.")


def state_samples(core, mass_ratio, order=64):
    """Whitened x,t samples, including the exact dx d^2k/(16pi^3) measure.

    No discrete normalization is imposed. Frobenius norm is the quadrature
    approximation to physical L2 in this axial scalar subspace.
    """
    _pilot_domain(mass_ratio)
    x, wx = _unit_rule(order)
    t, wt = _unit_rule(order)
    rows = [core.bind(1-mass_ratio**2*z*(1-z)).g(t) for z in x]
    weights = np.sqrt(wx[:, None] * wt[None, :] * x[:, None] * (1-x[:, None]) / charge_integral(mass_ratio))
    return weights * np.asarray(rows)


def _bilinear(outgoing, incoming, weights, a_out, s_out, a_in, s_in,
              phase, transport_operator):
    if phase is not None:
        theta_out, theta_in = phase(a_out, s_out), phase(a_in, s_in)
        outgoing = outgoing * np.exp(1j*theta_out)
        incoming = incoming * np.exp(1j*theta_in)
        # O' = U_out O U_in^dagger. This is basis transport, not new physics.
        if transport_operator:
            weights = weights * np.exp(1j*(theta_out-theta_in))
    return np.sum(np.conjugate(outgoing) * weights * incoming)


def dy_current(core, mass_ratio, q2, order=64, phase=None, transport_operator=True):
    """Direct shifted overlap using the represented core at both endpoints."""
    _pilot_domain(mass_ratio, q2)
    x, wx = _unit_rule(order)
    t, wt = _unit_rule(order)
    s = (t / (1-t))[:, None]
    ws = (wt / (1-t)**2)[:, None]
    cosine = np.cos(2*np.pi*np.arange(2*order)/(2*order))[None, :]
    result = 0j
    for z, wz in zip(x, wx):
        a = 1 - mass_ratio**2*z*(1-z)
        radial = core.bind(a)
        shift = (1-z)*np.sqrt(q2)
        shifted_s = np.maximum(0, s + shift**2 + 2*np.sqrt(s)*shift*cosine)
        result += wz*z*(1-z) * _bilinear(
            radial.h(shifted_s), radial.h(s), ws/(2*order),
            a, shifted_s, a, s, phase, transport_operator)
    return result / charge_integral(mass_ratio)


@dataclass(frozen=True)
class RepresentedCurrent:
    valence: complex
    pair: complex

    @property
    def total(self):
        return self.valence + self.pair


def longitudinal_current(core, mass_ratio, q2, order=64, phase=None,
                         transport_operator=True):
    """Same represented core for valence and explicit pair-contour terms.

    Pair virtuality is B=1+Q^2*z*(1-z), not an on-shell bound-state amplitude.
    No exact pair form factor is appended to a compressed valence result.
    """
    _pilot_domain(mass_ratio, q2)
    z, wz = _unit_rule(order)
    t, wt = _unit_rule(order)
    s, ws = t/(1-t), wt/(1-t)**2
    eta = q2 / mass_ratio**2
    beta = 1/(1 + eta/2 + np.sqrt(eta + eta**2/4))
    valence, pair = 0j, 0j
    for u, wu in zip(z, wz):
        y = beta*u
        a_in = 1-mass_ratio**2*y*(1-y)
        a_out = 1-mass_ratio**2*u*(1-u)
        valence += wu*(1+beta-2*y)*y * _bilinear(
            core.bind(a_out).h(s), core.bind(a_in).h(s), ws,
            a_out, s, a_in, s, phase, transport_operator)
        if q2 > 0:
            y = beta + (1-beta)*u
            a_in = 1-mass_ratio**2*y*(1-y)
            b_out = 1+q2*u*(1-u)
            pair += wu*(1+beta-2*y)*(1-y) * _bilinear(
                core.bind(b_out).h(s), core.bind(a_in).h(s), ws,
                b_out, s, a_in, s, phase, transport_operator)
    normalization = (1+beta)*charge_integral(mass_ratio)
    return RepresentedCurrent(valence/normalization, pair/normalization)

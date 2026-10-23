"""Necessary analytic test of a candidate nonlocal quark mass, not a solver.

M_E(z)=m_current+M0*exp(-z/Lambda^2) is a CONDITIONAL constant-background
quadratic sector of a nonlocal chiral action. No gap solution, confinement,
positive colored spectral measure, continuation prescription or physical
nucleon is inferred. In particular this is not installed as a retained-mode
Hamiltonian. The canonical Dirac matrices and all existing kernels survive.
"""
from dataclasses import dataclass
from math import e, sqrt

import numpy as np
from scipy.optimize import root
from scipy.special import lambertw

from .canonical_lf_fields import slash


CRITICAL_RATIO = 1 / sqrt(2 * e)


@dataclass(frozen=True)
class GaussianMass:
    m0: float
    scale: float
    current_mass: float = 0.

    def __post_init__(self):
        values = (self.m0, self.scale, self.current_mass)
        if not np.isfinite(values).all() or self.m0 <= 0 or self.scale <= 0 or self.current_mass < 0:
            raise ValueError('Positive finite M0/Lambda and nonnegative current mass required.')

    def mass(self, s):
        """Minkowski invariant s=p^2 in GeV^2; result in GeV."""
        with np.errstate(over='raise', invalid='raise'):
            return self.current_mass + self.m0 * np.exp(np.asarray(s) / self.scale**2)

    def denominator(self, s):
        return np.asarray(s) - self.mass(s)**2

    def derivative(self, s):
        mass = self.mass(s)
        return 1 - 2 * mass * (mass - self.current_mass) / self.scale**2

    def chiral_pole(self, branch):
        """Exact invariant pole; branches 0,-1 coalesce at the critical ratio.

        All integer Lambert branches belong to the declared global ansatz.
        Returning one branch is not a truncation/deletion of the others.
        """
        if self.current_mass != 0 or isinstance(branch, (bool, np.bool_)) or not isinstance(branch, (int, np.integer)):
            raise ValueError('Integer branch and zero current mass required.')
        argument = -2 * (self.m0 / self.scale)**2
        if branch in (0, -1) and self.m0 == self.scale*CRITICAL_RATIO:
            return complex(self.scale**2 / 2)
        return complex(-self.scale**2 * lambertw(argument, int(branch)) / 2)

    def numerical_pole(self, seed):
        """Independent real/imaginary root equation, no Lambert evaluation."""
        def residual(x):
            value = self.denominator((x[0]+1j*x[1])*self.scale**2) / self.scale**2
            return (value.real, value.imag)
        seed = complex(seed) / self.scale**2
        answer = root(residual, (seed.real, seed.imag), tol=1e-11)
        error = np.linalg.norm(residual(answer.x))
        if not np.isfinite(error) or error > 2e-11:
            raise ArithmeticError(f'Pole equation not resolved: {answer.message}; residual={error}')
        return complex(answer.x[0]+1j*answer.x[1])*self.scale**2

    def inverse_dirac(self, p):
        p = np.asarray(p, complex)
        if p.shape != (4,) or not np.isfinite(p).all():
            raise ValueError('Finite Cartesian four-momentum required.')
        s = p[0]**2 - np.dot(p[1:], p[1:])
        return slash(p) - self.mass(s)*np.eye(4)


def lf_momentum(plus, minus, transverse):
    """Cartesian momentum from symmetric LF coordinates; minus may be complex."""
    transverse = np.asarray(transverse, float)
    if not np.isfinite(plus) or plus <= 0 or transverse.shape != (2,) or not np.isfinite(transverse).all() or not np.isfinite(minus):
        raise ValueError('Finite LF coordinates and positive p+ required.')
    return np.array([(plus+minus)/sqrt(2), *transverse, (plus-minus)/sqrt(2)], complex)


def lf_pole(s, plus, transverse):
    # Coordinate validation is shared; this does not put a complex mode on shell physically.
    lf_momentum(plus, 0., transverse)
    return (s + np.dot(transverse, transverse)) / (2*plus)


def lf_residue(model, s, plus, transverse):
    """Full 4x4 residue of (slash p-M)^-1 in p-, no scalar/spin averaging."""
    p = lf_momentum(plus, lf_pole(s, plus, transverse), transverse)
    derivative = model.derivative(s)
    if abs(derivative) < 1e-12:
        raise ValueError('Simple-pole residue invalid at a coalescing/double pole.')
    return (slash(p) + model.mass(s)*np.eye(4)) / (2*plus*derivative)


def contour_count(function, derivative, bounds, order=96, subdivisions=4):
    """Argument principle for an entire denominator, counterclockwise rectangle.

    Returns the UNROUNDED complex count; callers must test convergence and
    boundary separation. Each edge is subdivided to resolve near-boundary
    exterior poles. This is not a complete-plane singularity inventory.
    """
    left, right, bottom, top = bounds
    if (not np.isfinite(bounds).all() or left >= right or bottom >= top or
            not isinstance(order, int) or order < 8 or
            not isinstance(subdivisions, int) or subdivisions < 1):
        raise ValueError('Ordered finite rectangle and quadrature order >=8 required.')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    corners = [left+1j*bottom, right+1j*bottom, right+1j*top, left+1j*top]
    result = 0j
    for start, stop in zip(corners, corners[1:]+corners[:1]):
        for index in range(subdivisions):
            a = start + (stop-start)*index/subdivisions
            b = start + (stop-start)*(index+1)/subdivisions
            z = (a+b)/2 + (b-a)*nodes/2
            result += (b-a)/2 * np.dot(weights, derivative(z)/function(z))
    return result / (2j*np.pi)

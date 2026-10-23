"""Protected nonsinglet pion-pole matching, not a nucleon Hamiltonian.

Cartesian (+---); momenta, F and masses in GeV. The exact pion results are
tree-level, quadratic in the pion/axial sources, with one EM insertion.
Polynomial elimination is a local derivative expansion, not a UV regulator.
Quark vertices below are longitudinal Ward-identity DIAGNOSTICS, not solutions
of QCD, the transverse identities, a gap equation or a bound-state equation.
"""
from dataclasses import dataclass
import numpy as np
from .canonical_lf_fields import GAMMA, slash
from .nonlocal_source import ETA

G5 = 1j * GAMMA[0] @ GAMMA[1] @ GAMMA[2] @ GAMMA[3]


def vector(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("Finite Cartesian four-momentum required.")
    return result


def square(value):
    p = vector(value)
    return float(p @ ETA @ p)


def polynomial_divided_difference(coefficients, x, y):
    """(f(x)-f(y))/(x-y), including x=y, without subtractive cancellation."""
    return sum(c * sum(x**j * y**(n-1-j) for j in range(n))
               for n, c in enumerate(coefficients) if n)


@dataclass(frozen=True)
class PionPole:
    mass: float
    decay_constant: float

    def __post_init__(self):
        if not np.isfinite([self.mass, self.decay_constant]).all() or self.mass < 0 or self.decay_constant <= 0:
            raise ValueError("Nonnegative mass and positive F, in GeV, required.")

    def propagator(self, invariant, order=None):
        """1/(m_pi^2-s-i0), off the pole, or its Taylor polynomial of degree N.

        The real function is evaluated away from poles only. No finite-width
        prescription is silently supplied. The expansion requires m_pi>0;
        its usefulness, unlike its evaluability, requires |s|<m_pi^2.
        """
        if not np.isfinite(invariant):
            raise ValueError("Finite invariant required.")
        if order is None:
            denominator = self.mass**2 - invariant
            if abs(denominator) <= 1e-13 * max(self.mass**2, abs(invariant), 1e-10):
                raise ValueError("On-pole evaluation requires a separate distribution prescription.")
            return 1 / denominator
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 0 or self.mass <= 0:
            raise ValueError("Local expansion requires a nonnegative integer order and positive mass.")
        return sum((invariant / self.mass**2)**n for n in range(order+1)) / self.mass**2

    def divided_difference(self, outgoing_s, incoming_s, order=None):
        if order is None:
            return self.propagator(outgoing_s) * self.propagator(incoming_s)
        # Validate the same domain as propagator, including order zero.
        self.propagator(outgoing_s, order)
        self.propagator(incoming_s, order)
        coefficients = [self.mass**(-2*(n+1)) for n in range(order+1)]
        return polynomial_divided_difference(coefficients, outgoing_s, incoming_s)

    def axial_response(self, momentum, order=None):
        """Pi_AA^{alpha beta}=F^2[g^{alpha beta}+k^alpha k^beta D(k)], GeV^2."""
        k = vector(momentum)
        return self.decay_constant**2 * (ETA + np.outer(k, k) * self.propagator(square(k), order))

    def explicit_response(self, momentum):
        """Independent elimination of the quadratic pion/source action block.

        K_pi=s-m^2, L=i F k, R=L^dagger, C=F^2 g. The Schur complement
        C-L K_pi^-1 R is exact at this stated tree/source order; no Fock/H2
        state elimination, pion loop or physical nuclear spectrum is used.
        """
        k = vector(momentum)
        self.propagator(square(k))  # Do not solve on an unqualified pole.
        left = 1j * self.decay_constant * k[:, None]
        inverse_action = np.array([[square(k) - self.mass**2]])
        return self.decay_constant**2 * ETA - left @ np.linalg.solve(inverse_action, left.conj().T)

    def axial_vector_source(self, outgoing, incoming, order=None, *, include_contacts=True):
        """Unit-charge one-vector-source derivative Pi^{mu;alpha beta}, GeV.

        Include both endpoint derivative contacts AND the pion-in-flight
        term. For a polynomial propagator, gauge the polynomial itself via
        its divided difference; multiplying two truncated propagators is
        not that operation. Multiply by the isospin generator/e separately.
        """
        k1, k0 = vector(outgoing), vector(incoming)
        d1, d0 = self.propagator(square(k1), order), self.propagator(square(k0), order)
        divided = self.divided_difference(square(k1), square(k0), order)
        result = np.einsum("m,a,b->mab", k1+k0, k1, k0) * divided
        if include_contacts:
            result += np.einsum("ma,b->mab", ETA, k0) * d0
            result += np.einsum("a,mb->mab", k1, ETA) * d1
        return self.decay_constant**2 * result

    def explicit_response_in_vector_source(self, outgoing, incoming, component, amplitude):
        """Two-Fourier-mode action block, for an independent derivative check.

        Off-diagonal linear source blocks are obtained by minimally gauging
        the pion kinetic operator and the two axial derivative vertices.
        Finite amplitudes here serve differences at zero, not a full finite
        field calculation (unretained mode shifts/quadratic terms omitted).
        """
        if component not in range(4) or not np.isfinite(amplitude):
            raise ValueError("A vector component and finite real source amplitude required.")
        momenta = [vector(incoming), vector(outgoing)]
        for k in momenta: self.propagator(square(k))
        kinetic = np.diag([square(k)-self.mass**2 for k in momenta])
        kinetic += amplitude * (momenta[0]+momenta[1])[component] * np.array([[0., 1.], [1., 0.]])
        coupling = np.zeros((8, 2), complex)
        for i in range(2):
            coupling[4*i:4*i+4, i] = 1j*self.decay_constant*momenta[i]
            coupling[4*i:4*i+4, 1-i] = amplitude*1j*self.decay_constant*ETA[component]
        response = self.decay_constant**2*np.kron(np.eye(2), ETA) - coupling @ np.linalg.solve(kinetic, coupling.conj().T)
        return response[4:, :4]

    def light_front_propagator(self, momentum):
        """Same covariant propagator with LF 1/(2 k+) and energy denominator.

        Signed nonzero k+ is allowed for exchanged off-shell momentum.
        k+=0 is excluded here, NOT assigned zero; the covariant function and
        two-sided limiting check remain available. No loop zero-mode proof.
        """
        p = vector(momentum)
        plus, minus = (p[0]+p[3])/np.sqrt(2), (p[0]-p[3])/np.sqrt(2)
        if plus == 0:
            raise ValueError("Zero-plus LF decomposition needs its explicit limit.")
        self.propagator(square(p))
        on_shell_minus = (self.mass**2+p[1]**2+p[2]**2)/(2*plus)
        return -1/(2*plus*(minus-on_shell_minus))


def pion_charge_generator():
    """EM generator in Cartesian (pi1,pi2,pi3); eigencharges -1,0,+1."""
    return np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], complex)


@dataclass(frozen=True)
class PolynomialQuarkDressing:
    """Explicit diagnostic A(s), b(s)=B(s)-m_q polynomials in s/scale^2.

    A coefficients are dimensionless, b coefficients in GeV. This is an
    algebraic test family, not a production self-energy or a UV asymptotic
    ansatz. Current quark mass is kept distinct from dynamical dressing.
    """
    a: tuple
    b: tuple
    scale: float

    def __post_init__(self):
        if not self.a or not self.b or self.scale <= 0 or not np.isfinite([*self.a, *self.b, self.scale]).all():
            raise ValueError("Finite diagnostic coefficients and positive GeV scale required.")

    def values(self, s):
        x = s/self.scale**2
        return np.polynomial.polynomial.polyval(x, self.a), np.polynomial.polynomial.polyval(x, self.b)

    def inverse(self, momentum, current_mass):
        if current_mass < 0 or not np.isfinite(current_mass):
            raise ValueError("Finite nonnegative current quark mass required.")
        a, b = self.values(square(momentum))
        return a*slash(vector(momentum)) - (current_mass+b)*np.eye(4)

    def axial_vertices(self, outgoing, incoming, current_mass, pion):
        """Longitudinal-identity witness (Gamma_A, 2 m_q Gamma_P).

        q.Gamma_A = S^-1(p') gamma5 + gamma5 S^-1(p) + 2m_q Gamma_P.
        Factors of i and nonsinglet flavor generator are stripped consistently;
        the free convention is Gamma_A=gamma^mu gamma5, Gamma_P=gamma5.
        A pion pole in the b part enforces this identity, but does NOT make
        these vertices physical or impose transverse/dynamical identities.
        """
        p1, p0 = vector(outgoing), vector(incoming)
        self.inverse(p0, current_mass)
        if current_mass == 0 and pion.mass != 0:
            raise ValueError("The tested exact chiral limit requires m_pi=0 too.")
        s1, s0 = square(p1), square(p0)
        a1, b1 = self.values(s1)
        a0, b0 = self.values(s0)
        da = polynomial_divided_difference(self.a, s1/self.scale**2, s0/self.scale**2)/self.scale**2
        regular = np.array([((a1+a0)/2*g + (p1+p0)[mu]*da*(slash(p1)+slash(p0))/2) @ G5 for mu, g in enumerate(GAMMA)])
        q = p1-p0
        pole = pion.propagator(square(q))
        axial = regular + np.einsum("m,ij->mij", q, G5)*(b1+b0)*pole
        weighted_pseudoscalar = (2*current_mass + (b1+b0)*pion.mass**2*pole)*G5
        return axial, weighted_pseudoscalar


def static_one_pion_exchange(transfer, pion, axial_coupling):
    """Leading NN OPE target in (isospin pair) x (spin pair) ordering, GeV^-2.

    V(k)=-g_A^2/(4F^2) tau1.tau2 (sigma1.k)(sigma2.k)/(k^2+m_pi^2).
    Matrix elements strip (2pi)^3 delta of total momentum; convolution uses
    d^3p/(2pi)^3 with continuum-normalized NR states. This is a low-energy
    matching target, NOT a bound-state potential selected for production.
    """
    k = np.asarray(transfer, float)
    if k.shape != (3,) or not np.isfinite(k).all() or not np.isfinite(axial_coupling):
        raise ValueError("Finite spatial transfer and explicit axial coupling required.")
    sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)
    isospin = sum(np.kron(t, t) for t in sigma)
    spin = np.einsum("a,aij->ij", k, sigma)
    return -axial_coupling**2/(4*pion.decay_constant**2)*pion.propagator(-k@k)*np.kron(isospin, np.kron(spin, spin))

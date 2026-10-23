"""Weak-external-source, local tree derivative expansion; not QCD matching.

Delta L = -C_ij (D_mu phi_i)^T F^{mu nu} D_nu phi_j
          +D_ij phi_i^T F_mu_nu F^{mu nu} phi_j.
Real O(4) fields are GeV, connections GeV, C,D GeV^-2. Both operators
vanish at zero external sources. This module does not generate new states,
claim an all-energy effective action, or select a production sigma model.
"""
from dataclasses import dataclass
import numpy as np
import torch

from .common_action_pilot import HBARC

METRIC = np.array([1., -1., -1., -1.])
ALPHA_EM = 1/137.035999084  # Low-energy convention, not a fitted running coupling.


def tensor(x):
    return torch.as_tensor(x, dtype=torch.complex128)


def connection(vector_source, axial_source):
    """A_mu acts on (sigma,pi1,pi2,pi3); no color-path replacement."""
    v, a = tensor(vector_source), tensor(axial_source)
    z = v[:, 0]*0
    return torch.stack([
        torch.stack([z, -a[:, 0], -a[:, 1], -a[:, 2]], dim=-1),
        torch.stack([a[:, 0], z, -v[:, 2], v[:, 1]], dim=-1),
        torch.stack([a[:, 1], v[:, 2], z, -v[:, 0]], dim=-1),
        torch.stack([a[:, 2], -v[:, 1], v[:, 0], z], dim=-1),
    ], dim=1)


def curvature(a, derivative_a):
    """F_mu_nu=partial_mu A_nu-partial_nu A_mu+[A_mu,A_nu]."""
    a, da = tensor(a), tensor(derivative_a)
    product = torch.einsum('mab,nbc->mnac', a, a)
    return da-da.transpose(0, 1)+product-product.transpose(0, 1)


def source_density(fields, derivatives, a, derivative_a, c, d):
    """One common polynomial action; differentiable in fields/sources/C/D.

    No complex conjugation: complex plane-wave components are the analytic
    continuation of real fields for vertex extraction, not complex O(4) fields.
    """
    f, df, a, c, d = map(tensor, (fields, derivatives, a, c, d))
    strength = curvature(a, derivative_a)
    covariant = df+torch.einsum('mab,ib->mia', a, f)
    eta = tensor(METRIC)
    radius = -torch.einsum('ij,mia,mnab,njb,m,n->', c, covariant, strength, covariant, eta, eta)
    two_source = torch.einsum('ij,ia,mnab,mnbc,jc,m,n->', d, f, strength, strength, f, eta, eta)
    return radius+two_source


@dataclass(frozen=True)
class SourceCurvature:
    c: np.ndarray
    d: np.ndarray

    def __post_init__(self):
        for name in ('c', 'd'):
            x = np.array(getattr(self, name), dtype=float, copy=True)
            if x.shape != (2, 2) or not np.isfinite(x).all() or not np.allclose(x, x.T, rtol=0, atol=1e-14):
                raise ValueError('Two finite symmetric inter-multiplet coefficient matrices required.')
            x.flags.writeable = False
            object.__setattr__(self, name, x)

    def density(self, fields, derivatives, a, derivative_a):
        return source_density(fields, derivatives, a, derivative_a, self.c.copy(), self.d.copy())

    def residual(self, retained):
        """Same-basis linear functional subtraction, NOT inferred QCD matching.

        Only valid with the same fields, sources, normalization and operator
        basis. Nonlinear coefficient functions must instead be subtracted as
        full functionals before taking derivatives.
        """
        return SourceCurvature(self.c-retained.c, self.d-retained.d)

    def projected(self, mode):
        e = np.asarray(mode)
        return float(e@self.c@e), float(e@self.d@e)


def mixed_derivative(function, order):
    """Actual mixed derivative at zero in <=4 real amplitude coordinates."""
    if not 1 <= order <= 4:
        raise ValueError('This bounded vertex extractor supports 1--4 amplitudes.')
    x = torch.zeros(order, dtype=torch.float64, requires_grad=True)
    value = function(x)
    result = []
    for part in (value.real, value.imag):
        for i in range(order):
            if not part.requires_grad:
                part = torch.zeros((), dtype=torch.float64)
                break
            part = torch.autograd.grad(part, x, create_graph=True, retain_graph=True)[0][i]
        result.append(float(part.detach()))
    return complex(*result)


def _pion_waves(x, p1, p0, e1, e0):
    incoming = tensor([0., 1., 1j, 0.])/np.sqrt(2)
    outgoing = incoming.conj()
    fi = x[0]*tensor(e0)[:, None]*incoming
    fo = x[1]*tensor(e1)[:, None]*outgoing
    df = -1j*tensor(METRIC*p0)[:, None, None]*fi+1j*tensor(METRIC*p1)[:, None, None]*fo
    return fi+fo, df


def _photon(epsilon):
    v = np.zeros((4, 3), dtype=complex)
    v[:, 2] = METRIC*np.asarray(epsilon)
    return connection(v, np.zeros((4, 3)))


def action_vertex(model, p1, p0, e1, e0):
    """Four-current correction from two pion and one external-source variation."""
    p1, p0 = np.asarray(p1), np.asarray(p0)
    q = p1-p0
    values = []
    for mu in range(4):
        # Source coordinate A_mu (lower index), so extracted J^mu is upper.
        epsilon = METRIC[mu]*np.eye(4)[mu]
        basis = _photon(epsilon)
        def expression(x):
            f, df = _pion_waves(x, p1, p0, e1, e0)
            a = x[2]*basis
            da = -1j*tensor(METRIC*q)[:, None, None, None]*a[None, :, :, :]
            return model.density(f, df, a, da)
        values.append(mixed_derivative(expression, 3))
    return np.array(values)


def action_contact(model, p1, p0, q1, q2, eps1, eps2, e1, e0):
    """Two-source contact from the FOURTH action derivative.

    Both photons incoming: p1=p0+q1+q2; an outgoing photon has negative q2.
    This is a contact, not by itself the general full Compton amplitude.
    """
    p1, p0, q1, q2 = map(np.asarray, (p1, p0, q1, q2))
    if not np.allclose(p1, p0+q1+q2, rtol=0, atol=1e-12):
        raise ValueError('Two-source momentum conservation required.')
    b1, b2 = _photon(eps1), _photon(eps2)
    def expression(x):
        f, df = _pion_waves(x, p1, p0, e1, e0)
        a1, a2 = x[2]*b1, x[3]*b2
        da = -1j*(tensor(METRIC*q1)[:, None, None, None]*a1[None, :, :, :]
                  +tensor(METRIC*q2)[:, None, None, None]*a2[None, :, :, :])
        return model.density(f, df, a1+a2, da)
    return mixed_derivative(expression, 4)


def dot(p, q):
    return np.dot(np.asarray(p)*METRIC, q)


def transverse_vertex(p1, p0, coefficient):
    p1, p0 = np.asarray(p1), np.asarray(p0)
    q = p1-p0
    return coefficient*(dot(q, q)*(p1+p0)-(dot(p1, p1)-dot(p0, p0))*q)


def radius_coefficient(radius_fm):
    return radius_fm**2/(6*HBARC**2)


def polarizability(c, d, mass_GeV):
    """alpha=-beta at this derivative/tree order, fm^3; charge restored.

    For real transverse photons: Delta L equivalent to
    -(c+2d)*e^2*pi+*pi-*F_em^2 modulo source Maxwell divergence/total derivative.
    Relativistic scalar normalization gives alpha=2 alpha_em (c+2d)/m.
    """
    if not np.isfinite(mass_GeV) or mass_GeV <= 0:
        raise ValueError('Positive external pion mass required.')
    return 2*ALPHA_EM*(c+2*d)*HBARC**3/mass_GeV


def match_projected(radius_fm, alpha_fm3, mass_GeV):
    """Two independent low-energy inputs fit two projected coefficients."""
    c = radius_coefficient(radius_fm)
    d = (alpha_fm3*mass_GeV/(2*ALPHA_EM*HBARC**3)-c)/2
    return c, d

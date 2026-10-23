"""Reusable canonical LF field algebra, independent of any finite model.

Extracted without formula changes from preserved project utilities at f509938c.
Symmetric p+/-=(p0+/-p3)/sqrt(2), metric (+---), A+=0, ubar u=2m,
physical antiquark labels and fixed Lepage--Brodsky phases. No oscillator
basis, fixed cell, H2 reduction, regulator, nucleon state or fit is imported.
"""
from __future__ import annotations
from math import sqrt
import numpy as np

def gamma_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bjørken--Drell matrices, with C43 light-front combinations."""
    z = np.zeros((2, 2), complex); ident = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]], complex); sz = np.array([[1, 0], [0, -1]], complex)
    g0 = np.block([[ident, z], [z, -ident]])
    return (g0, np.block([[z, sx], [-sx, z]]), np.block([[z, sy], [-sy, z]]), np.block([[z, sz], [-sz, z]]))

GAMMA = gamma_matrices()

GAMMA_PLUS = (GAMMA[0] + GAMMA[3]) / sqrt(2.0)

GAMMA_MINUS = (GAMMA[0] - GAMMA[3]) / sqrt(2.0)

LAMBDA_PLUS = GAMMA_MINUS @ GAMMA_PLUS / 2.0

LAMBDA_MINUS = GAMMA_PLUS @ GAMMA_MINUS / 2.0

def slash(p: np.ndarray) -> np.ndarray:
    return p[0] * GAMMA[0] - p[1] * GAMMA[1] - p[2] * GAMMA[2] - p[3] * GAMMA[3]

def polarization(kplus: float, kx: float, ky: float, helicity: int) -> np.ndarray:
    """C43 A^+=0 polarization (plus, minus, x, y); epsilon^- is constrained."""
    if kplus <= 0 or helicity not in (-1, 1):
        raise ValueError("nonzero k+ and helicity +/-1 required")
    eps_t = -np.array([helicity, 1j], dtype=complex) / sqrt(2.0)
    epsminus = (kx * eps_t[0] + ky * eps_t[1]) / kplus
    return np.array([0.0, epsminus, eps_t[0], eps_t[1]], dtype=complex)

def lf_dot(a: np.ndarray, b: np.ndarray) -> complex:
    return a[0] * b[1] + a[1] * b[0] - a[2] * b[2] - a[3] * b[3]

def polarization_cartesian(eps: np.ndarray) -> np.ndarray:
    return np.array([(eps[0] + eps[1]) / sqrt(2.0), eps[2], eps[3], (eps[0] - eps[1]) / sqrt(2.0)], complex)

def _helicity(value):
    if isinstance(value, (bool, np.bool_)) or value not in (-1,1):
        raise ValueError("helicity must be -1 or +1")
    return int(value)

def lf_spinor(pplus, px, py, mass, helicity):
    """Lepage--Brodsky u in Dirac representation, ubar u=2m.

    Uses p_B+=sqrt(2)*p_C43+. Its normalized good projection has
    momentum-independent spin axes; it is not the canonical instant-spin u.
    Mass zero is allowed. Arrays broadcast over physical momenta.
    """
    h=_helicity(helicity)
    pp,x,y=np.broadcast_arrays(np.asarray(pplus,float),np.asarray(px,float),np.asarray(py,float))
    mass=float(mass)
    if not np.all(np.isfinite([mass])) or mass<0 or np.any(pp<=0) or not all(
            np.all(np.isfinite(a)) for a in (pp,x,y)):
        raise ValueError("finite momenta, positive p+ and nonnegative mass required")
    p=sqrt(2)*pp
    r=x+1j*y; l=x-1j*y
    components=(p+mass,r,p-mass,r) if h==1 else (-l,p+mass,l,mass-p)
    return np.stack(components,axis=-1)/np.sqrt(2*p)[...,None]

def antiquark_spinor(pplus,px,py,mass,helicity):
    """v_h=C gamma0^T u_h^* = i gamma2 u_h^*, with physical antiparticle h."""
    u = lf_spinor(pplus,px,py,mass,helicity)
    return np.einsum("ij,...j->...i",1j*GAMMA[2],u.conj())

def _lf_momentum(p):
    pp,x,y = p
    return np.array([pp,(x*x+y*y)/(2*pp),x,y])

def three_gluon_numerator(left,right,parent,h_left,h_right,h_parent):
    """-i F contracted with eps_in, eps_left*, eps_right*, color f stripped.

    The minus i follows directly from the derivative in H_int; multiplying
    H_int by -i gives Appendix A's -g f F Dyson vertex. On-shell minus
    conservation is neither imposed nor needed, since epsilon+ vanishes.
    """
    momenta = (-_lf_momentum(parent),_lf_momentum(left),_lf_momentum(right))
    eps = (polarization(*parent,h_parent),polarization(*left,h_left).conj(),
           polarization(*right,h_right).conj())
    k0,k1,k2 = momenta
    e0,e1,e2 = eps
    contraction = (lf_dot(k0-k1,e2)*lf_dot(e0,e1)
                 + lf_dot(k1-k2,e0)*lf_dot(e1,e2)
                 + lf_dot(k2-k0,e1)*lf_dot(e2,e0))
    return -1j*contraction

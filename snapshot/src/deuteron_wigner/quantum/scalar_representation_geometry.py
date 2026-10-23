"""Local geometry/response audit of the fixed scalar control, not dynamics.

The chart is orthonormal in coefficient Frobenius norm, NOT in the state
metric. Complex tangent coordinates have independent real/imaginary parts.
All current derivatives retain the reference's fixed continuous I0.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.chebyshev import chebvander

from .scalar_composite_reference import _unit_rule, charge_integral
from .scalar_joint_representation import (
    A_MIN, A_MAX, PolynomialCore, _pilot_domain, state_samples,
)
from .scalar_constrained_representation import ConstrainedCore


def coefficient_matrix(core):
    return (core.coefficients if core.rank is None else core.left @ core.right).copy()


@dataclass(frozen=True)
class TangentChart:
    """T_k = left[:,k] right[:,k]^T; these coefficient matrices are orthonormal.

    The first m*r columns vary only the left factor and admit exact finite
    rank<=r affine paths. Remaining columns vary the right singular subspace.
    Base is real; complex directions are supported without identifying them
    with physical phases or a newly matched current.
    """

    left: np.ndarray
    right: np.ndarray
    fixed_right_count: int

    @property
    def dimension(self):
        return self.left.shape[1]

    def matrix(self, coordinates):
        c = np.asarray(coordinates)
        if c.shape != (self.dimension,) or not np.all(np.isfinite(c)):
            raise ValueError("Finite chart coordinates of the declared dimension required.")
        return (self.left*c) @ self.right.T

    def contract(self, coefficient_gradient):
        return np.einsum("ik,ij,jk->k", self.left, coefficient_gradient, self.right)


def tangent_chart(polynomial):
    if polynomial.rank is None or np.iscomplexobj(polynomial.left):
        raise ValueError("Use the real, explicitly ranked reference for this audit.")
    m, rank = polynomial.left.shape
    n = polynomial.right.shape[1]
    u, _, vh = np.linalg.svd(coefficient_matrix(polynomial), full_matrices=True)
    v = vh.T
    pairs = [(i,j) for i in range(m) for j in range(rank)]
    pairs += [(i,j) for i in range(rank) for j in range(rank,n)]
    left = np.column_stack([u[:,i] for i,j in pairs])
    right = np.column_stack([v[:,j] for i,j in pairs])
    left.setflags(write=False)
    right.setflags(write=False)
    return TangentChart(left,right,m*rank)


def horizontal_tangent(state, tangent):
    """(I-|psi><psi|) db / ||b||; removes radial/phase motion in state space.

    Does not assert those directions belong to the constrained coefficient
    chart. In particular its fixed nonzero boundaries forbid a free global
    rescaling/rephasing of the complete continuous core.
    """
    b = np.asarray(state).reshape(-1)
    d = np.asarray(tangent)
    norm = np.linalg.norm(b)
    if not np.isfinite(norm) or norm == 0 or d.shape[0] != b.size:
        raise ValueError("Nonzero state and compatible tangent required.")
    psi = b/norm
    return (d - np.multiply.outer(psi, psi.conj() @ d))/norm if d.ndim == 2 else (
        d-psi*np.vdot(psi,d))/norm


def state_jacobian(core, chart, mass, order=64):
    """Unnormalized b, raw db/dz and its horizontal/projective derivative.

    At the real reference d/d(Re z)=J, d/d(Im z)=iJ. The real metric is
    diag(A.T A,A.T A), A=horizontal(J); QGT includes imaginary cross blocks.
    """
    _pilot_domain(mass)
    if core.mode != "lift":
        raise ValueError("This audit differentiates the analytic lift only.")
    x, wx = _unit_rule(order)
    t, wt = _unit_rule(order)
    a = 1-mass**2*x*(1-x)
    degree_a, degree_t = core.core.degree
    av = chebvander(2*(a-A_MIN)/(A_MAX-A_MIN)-1,degree_a) @ chart.left
    tv = chebvander(2*t-1,degree_t) @ chart.right
    weight = np.sqrt(wx[:,None]*wt[None,:]*x[:,None]*(1-x[:,None])/charge_integral(mass))
    multiplier = (1-a[:,None])**2/a[:,None]*t[None,:]*(1-t[None,:])
    jac = (-weight[:,:,None]*multiplier[:,:,None]*av[:,None,:]*tv[None,:,:]).reshape(-1,chart.dimension)
    b = state_samples(core,mass,order).reshape(-1)
    return b,jac,horizontal_tangent(b,jac)


def _radial_derivative(a, s, degree):
    t = s/(1+s)
    weight = -(1-a)**2/a*t*(1-t)/(1+s)
    return weight[:,None]*chebvander(2*t-1,degree)


def _bilinear_gradient(core, a_out, s_out, a_in, s_in, weights):
    """dF=G_out:conj(dC)+G_in:dC; independently differentiates both factors."""
    so,si,w = [v.ravel() for v in np.broadcast_arrays(s_out,s_in,weights)]
    ho,hi = core.bind(a_out).h(so),core.bind(a_in).h(si)
    da,dt = core.core.degree
    va = lambda a: chebvander(2*(a-A_MIN)/(A_MAX-A_MIN)-1,da).reshape(-1)
    gout = np.outer(va(a_out),(w*hi) @ _radial_derivative(a_out,so,dt))
    gin = np.outer(va(a_in),(w*ho.conj()) @ _radial_derivative(a_in,si,dt))
    return gout,gin


def current_gradients(core, chart, mass, q2, order=64):
    """Rows are DY, longitudinal valence, longitudinal pair; never summed away.

    Returns derivatives w.r.t. real and imaginary coefficient coordinates,
    and each leg separately. There is no mass variation or implicit norm fit.
    """
    _pilot_domain(mass,q2)
    if core.mode != "lift":
        raise ValueError("This audit differentiates the analytic lift only.")
    z,wz = _unit_rule(order)
    t,wt = _unit_rule(order)
    s,ws = t/(1-t),wt/(1-t)**2
    size = tuple(d+1 for d in core.core.degree)
    gout = np.zeros((3,*size),complex)
    gin = np.zeros_like(gout)
    i0 = charge_integral(mass)
    cosine = np.cos(2*np.pi*np.arange(2*order)/(2*order))[None,:]
    eta = q2/mass**2
    beta = 1/(1+eta/2+np.sqrt(eta+eta**2/4))
    for u,wu in zip(z,wz):
        a = 1-mass**2*u*(1-u)
        shift = (1-u)*np.sqrt(q2)
        sout = np.maximum(0,s[:,None]+shift**2+2*np.sqrt(s[:,None])*shift*cosine)
        out,inc = _bilinear_gradient(core,a,sout,a,s[:,None],
                                    wu*u*(1-u)*ws[:,None]/(2*order*i0))
        gout[0] += out
        gin[0] += inc
        y = beta*u
        a_in = 1-mass**2*y*(1-y)
        out,inc = _bilinear_gradient(core,a,s,a_in,s,
                                    wu*(1+beta-2*y)*y*ws/((1+beta)*i0))
        gout[1] += out
        gin[1] += inc
        if q2 > 0:
            y = beta+(1-beta)*u
            a_in = 1-mass**2*y*(1-y)
            b_out = 1+q2*u*(1-u)
            out,inc = _bilinear_gradient(core,b_out,s,a_in,s,
                wu*(1+beta-2*y)*(1-y)*ws/((1+beta)*i0))
            gout[2] += out
            gin[2] += inc
    out = np.array([chart.contract(g) for g in gout])
    inc = np.array([chart.contract(g) for g in gin])
    return {"real":out+inc,"imaginary":1j*(inc-out),"outgoing":out,"incoming":inc}


def displaced_core(core, chart, coordinates, step):
    """Affine diagnostic path; rank is preserved for fixed-right directions.

    General tangent paths may change rank at second order; callers must not
    label them finite fixed-rank optimizer steps. No dynamics are inferred.
    """
    c = coefficient_matrix(core.core)+step*chart.matrix(coordinates)
    return ConstrainedCore(PolynomialCore(c),"lift")


def weak_response_direction(horizontal, response, relative_cutoff):
    """Unit coefficient direction maximizing one real response in weak space.

    Explicit diagnostic cutoff on Jacobian singular values, not an exact
    physical null space, regularization prescription or metric inverse.
    """
    if not 0 < relative_cutoff < 1:
        raise ValueError("Explicit relative singular-value cutoff in (0,1) required.")
    if np.iscomplexobj(horizontal):
        raise ValueError("Use the real reference Jacobian for this diagnostic.")
    _,s,vh = np.linalg.svd(horizontal,full_matrices=False)
    weak = s <= relative_cutoff*s[0]
    if not np.any(weak):
        raise ValueError("No weak directions at this diagnostic cutoff.")
    projection = vh[weak].T @ (vh[weak] @ np.asarray(response).real)
    norm = np.linalg.norm(projection)
    if norm == 0:
        raise ValueError("The chosen response is zero on the weak subspace.")
    return projection/norm,s,int(np.count_nonzero(weak))

"""Shared higher-derivative source action and conditional data response.

Adds -E_ij (D_mu phi_i)^T F^{mu rho} F_rho^nu D_nu phi_j, E GeV^-4,
to the existing C,D terms. Full O(4) sources, arbitrary common node count;
not a production dynamics model, all-order EFT basis or fitted coefficient.
"""
from dataclasses import dataclass
import numpy as np
import torch

from .transverse_source_matching import (
    tensor,curvature,source_density,_pion_waves,_photon,dot,METRIC,
    HBARC,ALPHA_EM,action_contact,
)


def stress_density(fields,derivatives,a,derivative_a,coefficient):
    """One common polynomial with all connection/curvature source partners.

    Analytically continued real fields: transpose, not Hermitian conjugate.
    Couplings remain differentiable; no independent alpha/beta output heads.
    """
    # Immutable coefficient matrices must not be shared as writable Torch
    # buffers; existing Torch coefficients keep their differentiation graph.
    if isinstance(coefficient,np.ndarray) and not coefficient.flags.writeable:
        coefficient=coefficient.copy()
    f,df,a,e=map(tensor,(fields,derivatives,a,coefficient))
    strength=curvature(a,derivative_a)
    covariant=df+torch.einsum('mab,ib->mia',a,f)
    eta=tensor(METRIC)
    return -torch.einsum('ij,mia,mrab,rnbc,njc,m,r,n->',e,covariant,strength,strength,covariant,eta,eta,eta)


@dataclass(frozen=True)
class JointSourceAction:
    c: np.ndarray
    d: np.ndarray
    e: np.ndarray

    def __post_init__(self):
        shape=None
        for name in ('c','d','e'):
            matrix=np.array(getattr(self,name),dtype=float,copy=True)
            if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or not len(matrix) or not np.isfinite(matrix).all() or not np.allclose(matrix,matrix.T,rtol=0,atol=1e-14):
                raise ValueError('Finite symmetric matrices on the common node space required.')
            if shape is not None and matrix.shape!=shape: raise ValueError('C,D,E must share the same node space.')
            shape=matrix.shape;matrix.flags.writeable=False;object.__setattr__(self,name,matrix)

    def density(self,fields,derivatives,a,derivative_a):
        return source_density(fields,derivatives,a,derivative_a,self.c.copy(),self.d.copy())+stress_density(fields,derivatives,a,derivative_a,self.e.copy())

    def projected(self,mode):
        mode=np.asarray(mode,float)
        if mode.shape!=(len(self.c),) or not np.isfinite(mode).all(): raise ValueError('Finite common-node mode required.')
        return tuple(float(mode@getattr(self,name)@mode) for name in ('c','d','e'))

    def residual(self,retained):
        """Linear SAME-basis/convention subtraction, not inferred QCD matching.

        Loop nonlocality and external wave/mode changes are not represented
        by subtracting bare matrices without first doing that transport.
        """
        if self.c.shape!=retained.c.shape: raise ValueError('Same source basis and node count required.')
        return JointSourceAction(self.c-retained.c,self.d-retained.d,self.e-retained.e)


def _mixed(function,order):
    x=torch.zeros(order,dtype=torch.float64,requires_grad=True)
    value=function(x);result=[]
    for part in (value.real,value.imag):
        for i in range(order):
            if not part.requires_grad:
                part=torch.zeros((),dtype=torch.float64);break
            part=torch.autograd.grad(part,x,create_graph=True,retain_graph=True)[0][i]
        result.append(float(part.detach()))
    return complex(*result)


def photon_contact(model,p1,p0,photons,polarizations,mode1,mode0):
    """Actual 2+N action derivative for 1<=N<=4 electromagnetic sources.

    Photons all incoming. N=3,4 retain the covariant-derivative seagulls;
    local non-Abelian covariance is separately tested at the action level.
    """
    p1,p0=np.asarray(p1,float),np.asarray(p0,float)
    photons=np.asarray(photons,float);polarizations=np.asarray(polarizations,float)
    n=len(photons)
    if not 1<=n<=4 or photons.shape!=(n,4) or polarizations.shape!=(n,4) or p1.shape!=(4,) or p0.shape!=(4,) or not np.isfinite(photons).all() or not np.isfinite(polarizations).all():
        raise ValueError('One to four finite photon momenta/polarizations required.')
    if not np.allclose(p1,p0+photons.sum(axis=0),rtol=0,atol=1e-12): raise ValueError('Source momentum conservation required.')
    bases=[_photon(eps) for eps in polarizations]
    def expression(x):
        f,df=_pion_waves(x,p1,p0,mode1,mode0)
        a=sum((x[2+i]*bases[i] for i in range(n)))
        da=sum((-1j*x[2+i]*tensor(METRIC*photons[i])[:,None,None,None]*bases[i][None,:,:,:] for i in range(n)))
        return model.density(f,df,a,da)
    return _mixed(expression,2+n)


def stress_contact(p1,p0,q1,q2,eps1,eps2,coefficient):
    """Independently reduced two-photon E vertex, no on-shell restriction.

    W(p,q,eps)=(p.q)*eps-(p.eps)*q; contact=2e[W(p1,q1).W(p0,q2)
    +W(p1,q2).W(p0,q1)]. Exactly transverse in each photon separately.
    """
    def w(p,q,eps): return dot(p,q)*np.asarray(eps)-dot(p,eps)*np.asarray(q)
    return 2*coefficient*(dot(w(p1,q1,eps1),w(p0,q2,eps2))+dot(w(p1,q2,eps2),w(p0,q1,eps1)))


def projected_observables(c,d,e,mass,*,loop=None):
    """(radius squared fm^2, alpha fm^3, beta fm^3), stated local order.

    loop, if provided, is an already normalized/matched observable triple
    in these same units. It is kept once; no bare coefficient transport is
    inferred. C/E do not set complete nonlocal momentum dependence.
    """
    if not np.isfinite([c,d,e,mass]).all() or mass<=0: raise ValueError('Finite coefficients and positive external mass required.')
    k=2*ALPHA_EM*HBARC**3/mass;b=c+2*d
    values=np.array([6*HBARC**2*c,k*(b+mass*mass*e),-k*b])
    if loop is not None:
        loop=np.asarray(loop,float)
        if loop.shape!=(3,) or not np.isfinite(loop).all(): raise ValueError('Finite matched loop observable triple required.')
        values+=loop
    return values


def match_projected_observables(target,mass,*,loop=None):
    """Parameterized inverse, NOT selection of physical targets or C,D,E.

    Requires three independent target values/conditions. A conditional
    alpha=-beta summary is not such a triplet. Only renormalized projections
    are inferred; the open matrices still have unresolved directions.
    """
    target=np.asarray(target,float)
    if target.shape!=(3,) or not np.isfinite(target).all() or not np.isfinite(mass) or mass<=0:
        raise ValueError('Three finite target observables and a positive mass required.')
    offset=np.zeros(3) if loop is None else np.asarray(loop,float)
    if offset.shape!=(3,) or not np.isfinite(offset).all(): raise ValueError('Finite loop triple required.')
    radius,alpha,beta=target-offset;k=2*ALPHA_EM*HBARC**3/mass
    c=radius/(6*HBARC**2);b=-beta/k;e=(alpha+beta)/(k*mass*mass)
    return np.array([c,(b-c)/2,e])


def match_conditional_observables(radius_squared,conditional_alpha,mass,*,sum_weight,e,loop=None):
    """Family compatible with radius and a supplied conditional response.

    e and sum_weight=kappa MUST be supplied: neither the unmeasured source
    projection nor an experimental response is silently fixed to zero.
    Radius is fm^2; conditional alpha and the loop dipoles are fm^3.
    This is a local linear-response matching family, not a physical fit or
    the nonlinear acceptance/correction model of the experiment.
    """
    if not np.isfinite([radius_squared,conditional_alpha,mass,sum_weight,e]).all() or mass<=0:
        raise ValueError('Finite conditions, response, E and positive mass required.')
    offset=np.zeros(3) if loop is None else np.asarray(loop,float)
    if offset.shape!=(3,) or not np.isfinite(offset).all(): raise ValueError('Finite loop triple required.')
    r,a,b=offset;k=2*ALPHA_EM*HBARC**3/mass
    c=(radius_squared-r)/(6*HBARC**2)
    loop_conditional=.5*(a-b)+sum_weight*(a+b)
    combination=(conditional_alpha-loop_conditional)/k-mass*mass*(.5+sum_weight)*e
    return np.array([c,(combination-c)/2,e])


def compass_dipole_response(mass,s,cosine):
    """Published Eq.(1) response to (alpha-beta,alpha+beta), in GeV^-3.

    Returns coefficients in d sigma/dOmega (GeV^-2); linear polarizability
    order. This is the PRIMARY equation, not detector-folded templates or
    a raw-data likelihood. It omits higher multipoles and baseline corrections.
    """
    if not np.isfinite([mass,s,cosine]).all() or mass<=0 or s<=mass*mass or abs(cosine)>1:
        raise ValueError('Physical mass, CM invariant and cosine required.')
    plus,minus=1+cosine,1-cosine
    prefactor=-ALPHA_EM*mass**3*(s-mass*mass)**2/(4*s*s*(s*plus+mass*mass*minus))
    return prefactor*np.array([minus*minus,s*s/mass**4*plus*plus])


def conditional_projection(response,covariance,*,nuisance=None):
    """Conditional fitted alpha=diff/2+kappa*sum, profiling nuisances.

    response columns are the ACTUAL folded diff/sum templates. Covariance
    and nuisance columns must be supplied, not invented from the published
    scalar error bar. Result exposes lost information; it is not a fit to
    COMPASS data. Normalization belongs among nuisance columns when floated.
    """
    r=np.asarray(response,float);v=np.asarray(covariance,float)
    if r.ndim!=2 or r.shape[1]!=2 or v.shape!=(len(r),len(r)) or not np.isfinite(r).all() or not np.isfinite(v).all() or not np.allclose(v,v.T,rtol=0,atol=1e-14):
        raise ValueError('Finite two-template response and symmetric covariance required.')
    try: l=np.linalg.cholesky(v)
    except np.linalg.LinAlgError as exc: raise ValueError('Positive definite data covariance required.') from exc
    white=np.linalg.solve(l,r);reference_norm=float(white[:,0]@white[:,0])
    if nuisance is not None:
        b=np.asarray(nuisance,float)
        if b.ndim!=2 or b.shape[0]!=len(r) or not np.isfinite(b).all(): raise ValueError('Finite nuisance columns on the same bins required.')
        wb=np.linalg.solve(l,b);u,singular,_=np.linalg.svd(wb,full_matrices=False)
        tol=max(wb.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0.)
        kept=u[:,singular>tol]
        white=white-kept@(kept.T@white)
    norm=white[:,0]@white[:,0]
    if norm<=np.finfo(float).eps*reference_norm or norm==0: raise ValueError('Conditional difference is unidentified after nuisance projection.')
    return dict(diff_weight=.5,sum_weight=float(white[:,0]@white[:,1]/(2*norm)),
                conditional_variance=float(1/(4*norm)),information_matrix=white.T@white)

"""Spinful architecture controls, not a selected nucleon or QCD model.

Metric +---; Cartesian gamma matrices; p_C^+=(p0+p3)/sqrt(2).
Yukawa current is on-shell charge-renormalized through g^2 with M=m=1.
The unsubtracted Fock probability diverges; no normalized continuum state
or all-component LIGHT-FRONT current is claimed from the two-body amplitudes.
"""

from functools import lru_cache

import numpy as np

from deuteron_wigner.bridge.modes.core import GAMMA, slash
from .m2_canonical_emission import lf_spinor


ETA = np.array([1.,-1.,-1.,-1.])
IDENTITY = np.eye(4,dtype=complex)


def dot(p,q):
    return np.dot(np.asarray(p)*ETA,q)


@lru_cache(maxsize=24)
def rule(n):
    if isinstance(n,bool) or not isinstance(n,int) or n<8:
        raise ValueError("Quadrature order must be an integer >=8.")
    x,w = np.polynomial.legendre.leggauss(n)
    return (x+1)/2,w/2


def spinors(p):
    pp = (p[0]+p[3])/np.sqrt(2)
    return np.column_stack([lf_spinor(pp,p[1],p[2],1.,h) for h in (1,-1)])


def bilinear(vertex,pout,pin):
    return np.array([spinors(pout).conj().T@GAMMA[0]@v@spinors(pin) for v in vertex])


def pauli_vertex(q):
    # i sigma^{mu nu} q_nu/(2m) = -[gamma^mu, slash(q)]/(4m), m=1.
    sq = slash(q)
    return np.array([-(g@sq-sq@g)/4 for g in GAMMA])


def structures(pout,pin):
    return np.array([bilinear(GAMMA,pout,pin),bilinear(pauli_vertex(pout-pin),pout,pin)])


def lf_components(current):
    return np.array([(current[0]+current[3])/np.sqrt(2),
                     (current[0]-current[3])/np.sqrt(2),current[1],current[2]])


def extract(current,pout,pin,components=(0,1,2,3)):
    """Independent current decomposition; reject components unable to identify F1/F2."""
    basis = structures(pout,pin)
    a = np.column_stack([lf_components(b)[list(components)].ravel() for b in basis])
    y = lf_components(current)[list(components)].ravel()
    values,_,rank,s = np.linalg.lstsq(a,y,rcond=None)
    if rank != 2:
        raise ValueError("These current components do not identify both form factors.")
    return values,float(np.linalg.norm(a@values-y)),float(s[0]/s[-1])


def momenta(q2,boost=(0.,0.,0.),angle=.4):
    if q2<0:
        raise ValueError("Spacelike Q^2>=0 required.")
    q = np.sqrt(q2)*np.array([np.cos(angle),np.sin(angle),0.])
    energy = np.sqrt(1+q2/4)
    pin,pout = np.r_[energy,-q/2],np.r_[energy,q/2]
    velocity = np.asarray(boost,float)
    b2 = np.dot(velocity,velocity)
    if b2>=1:
        raise ValueError("Sub-luminal boost required.")
    def transform(p):
        if b2==0:
            return p
        gamma = 1/np.sqrt(1-b2)
        bp = np.dot(velocity,p[1:])
        return np.r_[gamma*(p[0]+bp),p[1:]+((gamma-1)*bp/b2+gamma*p[0])*velocity]
    return transform(pout),transform(pin)


def _validate(mu,kappa):
    if not np.isfinite(mu) or mu<=0 or not np.isfinite(kappa) or kappa<0:
        raise ValueError("Positive scalar mass and finite nonnegative g^2/(16pi^2) required.")


def loop_vertex(pout,pin,mu,kappa=.01,order=48):
    """Full spinor numerator of the finite on-shell subtracted Yukawa triangle.

    r=alpha*pout+beta*pin; u=alpha+beta; alpha=u*v, beta=u*(1-v).
    Delta=u^2+(1-u)*mu^2+alpha*beta*Q^2. N=(/pout-/r+1)gamma(/pin-/r+1).
    The sole charge counterterm follows at Q^2=0, not from component fitting.
    Divergent gamma terms cancel before integration. This is an on-shell
    matrix representative; no off-shell WTI completion is inferred.
    """
    _validate(mu,kappa)
    if abs(dot(pin,pin)-1)>1e-9 or abs(dot(pout,pout)-1)>1e-9:
        raise ValueError("Equal on-shell fermions of mass one required.")
    q2 = -dot(pout-pin,pout-pin)
    if q2 < -1e-10:
        raise ValueError("Spacelike current required.")
    u,wu = rule(order)
    v,wv = rule(order)
    a = (u[:,None]*v[None,:]).ravel()
    b = (u[:,None]*(1-v[None,:])).ravel()
    us = a+b
    weight = (wu[:,None]*wv[None,:]*u[:,None]).ravel()
    d0 = us**2+(1-us)*mu**2
    delta = d0+a*b*q2
    shift = a[:,None]*pout+b[:,None]*pin
    sr = np.einsum("ni,ijk->njk",shift*ETA,np.asarray(GAMMA))
    left = slash(pout)+IDENTITY-sr
    right = slash(pin)+IDENTITY-sr
    subtract = (2-us)**2/d0+np.log(delta/d0)
    result = []
    for gamma in GAMMA:
        numerator = left@gamma@right
        correction = np.einsum("n,nij->ij",weight/delta,numerator)-np.sum(weight*subtract)*gamma
        result.append(gamma+kappa*correction)
    return np.asarray(result)


def parameter_form_factors(mu,q2,kappa=.01,order=96):
    """Independently reduced invariant integrals; not used by loop_vertex.

    Gordon reduction after symmetric alpha,beta integration yields
    F2=2 kappa int u^2(2-u)/Delta. F1 uses the finite charge subtraction.
    """
    _validate(mu,kappa)
    if q2<0:
        raise ValueError("Spacelike Q^2 required.")
    u,wu = rule(order)
    v,wv = rule(order)
    u = u[:,None]
    ab = u*u*v[None,:]*(1-v[None,:])
    d0 = u*u+(1-u)*mu**2
    delta = d0+ab*q2
    weights = wu[:,None]*wv[None,:]*u
    f1 = 1+kappa*np.sum(weights*((2-u)**2*(1/delta-1/d0)-ab*q2/delta-np.log(delta/d0)))
    f2 = 2*kappa*np.sum(weights*u*(2-u)/delta)
    return np.array([f1,f2])


def lf_amplitudes(x,kx,ky,mu):
    """Rows: constituent spin +/-. Columns: external LF spin +/-. Coupling g=1.

    Established local Yukawa numerator; amplitudes contain Lz=0,+1,-1.
    No mass split between the external pole and internal fermion is made.
    """
    if not 0<x<1 or mu<=0:
        raise ValueError("Interior fraction and positive scalar mass required.")
    kx,ky = np.broadcast_arrays(kx,ky)
    prefactor = -np.sqrt(1-x)/(kx*kx+ky*ky+(1-x)**2+x*mu**2)
    result = np.empty((*kx.shape,2,2),complex)
    result[...,0,0] = result[...,1,1] = (1+x)*prefactor
    result[...,1,0] = -(kx+1j*ky)*prefactor
    result[...,0,1] = (kx-1j*ky)*prefactor
    return result


def lightfront_form_factors(mu,q2,kappa=.01,order=64,orbital=True):
    """Direct two-body plus-current overlap, subtracted BEFORE radial integration.

    F1=1+overlap(q)-overlap(0), through g^2. F2 is the spin-flip overlap.
    For q2=0 use the analytic derivative of the shifted numerator, not 0/0.
    orbital=False removes both helicity/orbital off-diagonal amplitudes as
    a deliberate negative control. This is NOT a finite normalized Fock state.
    """
    _validate(mu,kappa)
    if q2<0:
        raise ValueError("Spacelike Q^2 required.")
    x,wx = rule(order)
    t,wt = rule(order)
    s = t/(1-t)
    ws = wt/(1-t)**2
    phi = 2*np.pi*np.arange(2*order)/(2*order)
    kx = np.sqrt(s)[:,None]*np.cos(phi)[None,:]
    ky = np.sqrt(s)[:,None]*np.sin(phi)[None,:]
    accum = np.zeros((2,2),complex)
    flip_zero = 0.
    for z,w in zip(x,wx):
        incoming = lf_amplitudes(z,kx,ky,mu)
        outgoing = lf_amplitudes(z,kx+(1-z)*np.sqrt(q2),ky,mu)
        if not orbital:
            incoming[...,0,1] = incoming[...,1,0] = 0
            outgoing[...,0,1] = outgoing[...,1,0] = 0
        product = np.einsum("...si,...sj->...ij",outgoing.conj(),incoming)
        norm = np.einsum("...si,...sj->...ij",incoming.conj(),incoming)
        # Only diagonal charge renormalization; off-diagonal norm is exactly zero.
        product[...,0,0] -= norm[...,0,0]
        product[...,1,1] -= norm[...,1,1]
        accum += w*np.einsum("i,ijab->ab",ws,product)/(2*order)
        if q2==0 and orbital:
            a = (1-z)**2+z*mu**2
            flip_zero += w*2*(1-z)**2*(1+z)*np.sum(ws/(s+a)**2)
    f1 = 1+kappa*accum[0,0]
    f2 = kappa*flip_zero if q2==0 else -2*kappa*accum[0,1]/np.sqrt(q2)
    return np.array([f1,f2])


def finite_cloud_probability(mu,cutoff,kappa=.01,order=96):
    """Two-body probability inside kT<=cutoff; not an infinite-space norm."""
    _validate(mu,kappa)
    if cutoff<=0:
        raise ValueError("Positive finite transverse cutoff required.")
    x,w = rule(order)
    a = (1-x)**2+x*mu**2
    s = cutoff**2
    integral = (1+x)**2*(1/a-1/(s+a))+np.log1p(s/a)+a/(s+a)-1
    return float(kappa*np.sum(w*(1-x)*integral))


def inverse_propagator(p,a=.13,c=.07):
    s = dot(p,p)
    av = 1+a*(s-1)
    bv = av+c*(s-1)**2
    return av*slash(p)-bv*IDENTITY


def ward_vertex(pout,pin,a=.13,c=.07,transverse=0.):
    """Ball--Chiu finite differences for the declared polynomial propagator.

    Entire polynomials make equal virtualities nonsingular. A transverse
    Pauli term leaves the same propagator, pole residue, charge and WTI.
    The arbitrary transverse coefficient is a witness, not a fitted repair.
    """
    so,si = dot(pout,pout),dot(pin,pin)
    av = 1+a*((so+si)/2-1)
    delta_b = a+c*(so+si-2)
    bracket = a*(slash(pout)+slash(pin))/2-delta_b*IDENTITY
    q = pout-pin
    t = pauli_vertex(q)
    return np.array([av*GAMMA[mu]+(pout+pin)[mu]*bracket+transverse*t[mu] for mu in range(4)])

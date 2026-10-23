"""One-free-quark-loop cuts and nonlocal bubble parts, not a full QCD model.

Cut normalization A_IJ=1/2 integral dPhi_2 sum M_I M_J*, with Nc=3,
two equal-mass flavors, Hermitian S/P/non-singlet axial/EM bilinears.
A is Im of the correlator and -Im of its induced collective inverse term.
No finite local subtraction coefficients, confinement, full EM triangle,
or continuation of homogeneous quantum mechanics is supplied here.
"""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

from .canonical_lf_fields import GAMMA, lf_spinor, antiquark_spinor
from .collective_quark_matching import bilinear_matrices
from .chiral_soft_matching import G5
from .nonzero_momentum_current import lf_coordinates, dot


@lru_cache(maxsize=16)
def quadrature(order):
    if isinstance(order, bool) or not isinstance(order, int) or order < 4:
        raise ValueError('Integer quadrature order >=4 required.')
    z,w=leggauss(order)
    return z,w


def kinematics(q, mass):
    q=np.asarray(q,float)
    if q.shape!=(4,) or not np.isfinite(q).all() or q[0]<=0 or dot(q,q)<=0:
        raise ValueError('Future timelike total four-momentum required.')
    if not np.isfinite(mass) or mass<=0:
        raise ValueError('Positive explicit free-quark mass required.')
    s=dot(q,q)
    return q,s,np.sqrt(max(0.,1-4*mass*mass/s))


def source_vertices():
    """20 open 24x24 vertices: S,P1..3,A1[0..3],A2,A3,V_EM[0..3].

    A^a= qbar gamma^mu gamma5 tau^a/2 q, P^a=qbar i gamma5 tau^a q.
    The axial identity is the free-quark SUBDIAGRAM identity with mass M;
    it is not total interacting PCAC with the small Lagrangian mass source.
    """
    b=list(bilinear_matrices())
    tau=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    b += [np.kron(np.kron(g@G5,t/2),np.eye(3)) for t in tau for g in GAMMA]
    b += [np.kron(np.kron(g,np.diag([2/3,-1/3])),np.eye(3)) for g in GAMMA]
    return np.array(b)


def pair_amplitudes(quark, antiquark, mass):
    """20 x (12 quark states) x (12 antiquark states), color/flavor resolved."""
    u=np.column_stack([lf_spinor(*lf_coordinates(quark),mass,h) for h in [-1,1]])
    v=np.column_stack([antiquark_spinor(*lf_coordinates(antiquark),mass,h) for h in [-1,1]])
    ubar=np.kron(u.conj().T@GAMMA[0],np.eye(6))
    return np.einsum('si,Aij,jt->Ast',ubar,source_vertices(),np.kron(v,np.eye(6)),optimize=True)


def phase_space_spectral(q, mass, *, geometry='rest', order=6):
    """Independent explicit spinor cuts in rest-angle or LF (x,kT) variables.

    dPhi_2=beta dcos(theta)dphi/(32pi^2)=dx dphi/(16pi^2).
    x=(1+beta*cos(theta))/2, kT_rel^2=s*x*(1-x)-M^2. Both particles
    carry positive plus; all colors, flavors and spin phases are retained.
    Instantaneous propagators have no two-particle on-shell cut; they are
    not thereby deleted from uncut loops, finite terms or current partners.
    """
    q,s,beta=kinematics(q,mass)
    if geometry not in ('rest','light_front'):
        raise ValueError('Unknown phase-space geometry.')
    z,w=quadrature(order)
    result=np.zeros((20,20),complex)
    if beta==0: return result
    qplus,qx,qy=lf_coordinates(q)
    for cosine,weight in zip(z,w/2):
        for phi in np.arange(2*order)*np.pi/order:
            if geometry=='rest':
                momentum=np.sqrt(s)*beta/2*np.array([
                    np.sqrt(1-cosine*cosine)*np.cos(phi),
                    np.sqrt(1-cosine*cosine)*np.sin(phi),cosine])
                rest=np.r_[np.sqrt(s)/2,momentum]
                # Rotation-free boost from total rest frame to q.
                p0=(q[0]*rest[0]+q[1:]@rest[1:])/np.sqrt(s)
                spatial=rest[1:]+q[1:]*(rest[0]/np.sqrt(s)+(q[1:]@rest[1:])/(np.sqrt(s)*(q[0]+np.sqrt(s))))
                p=np.r_[p0,spatial]
            else:
                x=(1+beta*cosine)/2
                kt=np.sqrt(max(0.,s*x*(1-x)-mass*mass))
                px,py=x*np.array([qx,qy])+kt*np.array([np.cos(phi),np.sin(phi)])
                pp=x*qplus;pm=(mass*mass+px*px+py*py)/(2*pp)
                p=np.array([(pp+pm)/np.sqrt(2),px,py,(pp-pm)/np.sqrt(2)])
            amplitudes=pair_amplitudes(p,q-p,mass).reshape(20,144)
            result += weight/(2*order)*amplitudes@amplitudes.conj().T
    return beta/(16*np.pi)*result


def analytic_spectral(q, mass):
    """Independent covariant spin-trace/angular reduction, Nc=3, Nf=2."""
    q,s,beta=kinematics(q,mass);nc=3
    a=np.zeros((20,20),complex)
    if beta==0: return a
    pseudoscalar=nc*s*beta/(4*np.pi)
    a[0,0]=nc*s*beta**3/(4*np.pi)
    transverse=np.outer(q,q)-s*np.diag([1,-1,-1,-1])
    axial=nc*beta**3/(24*np.pi)*transverse
    axial+=nc*mass*mass*beta/(4*np.pi*s)*np.outer(q,q)
    for flavor in range(3):
        p=1+flavor;sl=slice(4+4*flavor,8+4*flavor)
        a[p,p]=pseudoscalar;a[sl,sl]=axial
        a[p,sl]=1j*mass*pseudoscalar*q/s
        a[sl,p]=a[p,sl].conj()
    a[16:20,16:20]=nc*(5/9)*beta*(1+2*mass*mass/s)/(12*np.pi)*transverse
    return a


def collective_cut(model, s, channel):
    """-Im Delta K_ij(s+i0)=g_i g_j A_channel(s), GeV^2.

    Both multiplets share the SAME qqbar continuum, with interference.
    Dropping off-diagonal entries changes the physical coupling map.
    """
    if channel not in ('scalar','pion') or not np.isfinite(s) or s<=0:
        raise ValueError('Scalar/pion channel and positive invariant s required.')
    c=getattr(model,channel)
    index=0 if channel=='scalar' else 1
    a=analytic_spectral([np.sqrt(s),0,0,0],model.mass)[index,index].real
    return a*np.outer(c.couplings,c.couplings)


def _real_domain(mass, z):
    if not np.isfinite([mass,z]).all() or mass<=0 or z>=4*mass*mass:
        raise ValueError('Positive mass and real z below qqbar threshold required.')


def feynman_nonlocal(mass, z, channel, *, order=96):
    """Twice-subtracted scalar/pion inverse bubble, GeV^2, unit Yukawa.

    Delta K(z)-Delta K(0)-z Delta K'(0), NOT the full inverse or selected
    local counterterms. Independent S/P subtractions do not themselves
    specify a chirally/source-complete renormalized action. Nc=3,Nf=2.
    """
    _real_domain(mass,z)
    if channel not in ('scalar','pion'): raise ValueError('Scalar/pion channel required.')
    x,w=quadrature(order);x=(x+1)/2;w=w/2
    u=x*(1-x)*z/(mass*mass)
    log=-np.log1p(-u)
    if channel=='pion': integrand=z*log
    else:
        remainder=log-u
        small=abs(u)<1e-3
        # Stable L(u)-u = sum_{n>=2} u^n/n around the subtraction point.
        remainder[small]=sum(u[small]**n/n for n in range(2,9))
        integrand=z*log-4*mass*mass*remainder
    return float(-3/(4*np.pi**2)*(w@integrand))


def dispersive_nonlocal(mass,z,channel,*,order=96, beta_interval=(0.,1.)):
    """-z^2/pi int_threshold^infty A(s)/[s^2(s-z)] ds.

    Independent beta=sqrt(1-4M^2/s) integration. Optional interval is an
    additive spectral-quadrature split, NOT a chosen Wilsonian/Fock cutoff.
    No field modes or matching coefficients are assigned from this split.
    """
    _real_domain(mass,z)
    if channel not in ('scalar','pion'): raise ValueError('Scalar/pion channel required.')
    lo,hi=beta_interval
    if not np.isfinite([lo,hi]).all() or not 0<=lo<hi<=1:
        raise ValueError('Ordered beta interval inside [0,1] required.')
    x,w=quadrature(order);b=lo+(hi-lo)*(x+1)/2;w=w*(hi-lo)/2
    power=4 if channel=='scalar' else 2
    return float(-3*z*z/(2*np.pi**2)*np.sum(w*b**power/(4*mass*mass-z*(1-b*b))))


def feynman_upper_boundary(mass,s,channel,*,imaginary_part=1e-8):
    """Direct complex Feynman-parameter log, z=s+i*epsilon, GeV^2.

    Upper-side limit verifies the cut sign and normalization independently;
    finite epsilon is a numerical boundary approach, not a physical width.
    """
    if not np.isfinite([mass,s,imaginary_part]).all() or mass<=0 or s<=0 or imaginary_part<=0:
        raise ValueError('Positive mass, s and boundary epsilon required.')
    if channel not in ('scalar','pion'): raise ValueError('Scalar/pion channel required.')
    z=complex(s,imaginary_part)
    def integrand(x):
        u=x*(1-x)*z/(mass*mass);log=-np.log1p(-u)
        return z*log if channel=='pion' else (z-4*mass*mass)*log+4*mass*mass*u
    points=None
    if s>4*mass*mass:
        beta=np.sqrt(1-4*mass*mass/s)
        roots=[(1-beta)/2,(1+beta)/2]
        width=imaginary_part/(s*beta)
        # Resolve the regulated logarithm near each threshold root rather than
        # asking a large interval to discover an O(epsilon) layer adaptively.
        points=sorted({p+shift*width for p in roots for shift in [-100,-1,0,1,100]
                       if 0<p+shift*width<1})
    value,_=quad(integrand,0.,1.,points=points,epsabs=2e-11,epsrel=2e-11,limit=250,complex_func=True)
    return -3*value/(4*np.pi**2)


def tree_width_diagnostics(model):
    """Leading projected quark-pair Gamma/M; not complex poles or predictions.

    Near a bare nondegenerate mode, M Gamma=e^T A(M^2)e. If Gamma/M is
    not small, interpreting it as a reliable narrow-width calculation is
    unjustified; mass shifts, mixed modes and other channels are uncomputed.
    """
    rows=[]
    for name in ('scalar','pion'):
        channel=getattr(model,name)
        masses,modes=np.linalg.eigh(channel.hessian)
        for n,m2 in enumerate(masses):
            e=modes[:,n]
            index=0 if name=='scalar' else 1
            unit_cut=analytic_spectral([np.sqrt(m2),0,0,0],model.mass)[index,index].real
            ratio=float((e@channel.couplings)**2*unit_cut/m2)
            rows.append(dict(channel=name,mode=n,tree_mass_GeV=float(np.sqrt(m2)),
                yukawa=float(e@channel.couplings),projected_Gamma_over_M=ratio,
                channel_open=bool(m2>4*model.mass**2)))
    return rows

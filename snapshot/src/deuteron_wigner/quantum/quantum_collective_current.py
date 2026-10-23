"""One-quark-loop charged collective EM vertex, not a physical pion model.

Nc=3, two degenerate quarks, pointlike i*gamma5 Yukawa vertices; electric e
and mode couplings g_i*g_j stripped. The finite loop vertex matches the
twice-subtracted pion inverse from quantum_pair_threshold. Its independent
transverse shape cannot be inferred from that inverse's Ward identity.
No gluon/collective loops, finite physical C, resummation or full LF current.
"""
import numpy as np
from .canonical_lf_fields import GAMMA, slash, lf_spinor, antiquark_spinor
from .chiral_soft_matching import G5
from .quantum_pair_threshold import quadrature
from .nonzero_momentum_current import dot


def _domain(mass, outgoing, incoming):
    p1,p0=np.asarray(outgoing,float),np.asarray(incoming,float)
    if p1.shape!=(4,) or p0.shape!=(4,) or not np.isfinite([p1,p0]).all() or not np.isfinite(mass) or mass<=0:
        raise ValueError('Finite four-momenta and positive equal quark mass required.')
    if max(dot(p1,p1),dot(p0,p0))>=4*mass*mass or dot(p1-p0,p1-p0)>1e-12:
        raise ValueError('Subthreshold external virtualities and spacelike photon required.')
    return p1,p0


def _trace_numerator(a,b,c,mass):
    """Original gamma5-pair triangle after cyclic Dirac-trace rearrangement."""
    left=slash(a)+mass*np.eye(4);right=slash(b)+mass*np.eye(4)
    spectator=-slash(c)+mass*np.eye(4)
    return np.array([np.trace(left@g@right@spectator) for g in GAMMA])


def numerator_coefficients(a,b,c,mass,*,method='reduced'):
    """Angular-averaged numerator =4*(v_mu*l^2+w_mu), Cartesian (+---).

    The trace control extracts the quadratic tensor from +/- unit directions
    and contracts its Minkowski metric, without using the reduced formula.
    Evanescent d-dimensional terms affect only the removed local wave term;
    the zero-invariant subtraction below is applied to the common vertex.
    """
    if method=='reduced':
        v=.5*c-a-b
        w=dot(a,b)*c-dot(b,c)*a-dot(a,c)*b+mass*mass*(a+b-c)
        return v,w
    if method!='trace': raise ValueError('Reduced or direct trace numerator required.')
    n0=_trace_numerator(a,b,c,mass)
    average=np.zeros(4,complex)
    for axis,sign in zip(np.eye(4),[1,-1,-1,-1]):
        plus=_trace_numerator(a+axis,b+axis,c+axis,mass)
        minus=_trace_numerator(a-axis,b-axis,c-axis,mass)
        average+=sign*(plus+minus-2*n0)/8
    if max(abs(n0.imag).max(),abs(average.imag).max())>1e-10:
        raise ValueError('Unexpected imaginary trace in the real subthreshold control.')
    return average.real/4,n0.real/4


def triangle_unit(mass,outgoing,incoming,*,order=28,method='reduced'):
    """Single common equal-mass shape, GeV, with total unit meson charge.

    alpha=u*v,beta=u*(1-v),gamma=1-u. r=alpha*p1+beta*p0,
    a=p1-r,b=p0-r,c=-r, Delta=M^2-alpha*gamma*p1^2
    -beta*gamma*p0^2-alpha*beta*q^2. The subtracted vertex is
    Nc/pi^2 int_triangle [v log(Delta/M^2)+w/(2Delta)-(p1+p0-r)/2].
    This is finite zero-invariant wave subtraction, not selected physical
    matching coefficients. It vanishes at zero external invariants.
    """
    p1,p0=_domain(mass,outgoing,incoming);s1,s0,t=dot(p1,p1),dot(p0,p0),dot(p1-p0,p1-p0)
    nodes,weights=quadrature(order);nodes=(nodes+1)/2;weights=weights/2
    total=np.zeros(4)
    for u,wu in zip(nodes,weights):
        for x,wx in zip(nodes,weights):
            alpha,beta=u*x,u*(1-x);gamma=1-u
            r=alpha*p1+beta*p0
            v,w=numerator_coefficients(p1-r,p0-r,-r,mass,method=method)
            delta=mass*mass-alpha*gamma*s1-beta*gamma*s0-alpha*beta*t
            total+=wu*wx*u*(v*np.log(delta/(mass*mass))+w/(2*delta)-(p1+p0-r)/2)
    return 3/np.pi**2*total


def photon_attachments(mass,outgoing,incoming,*,charges=(2/3,-1/3),order=28,method='reduced'):
    """Photon on quark u and antiquark d; crossed route carries quark Q_d.

    Equal masses allow reversing both pion momenta to express the second
    loop in the same variables. Keeping Q_d*Gamma(-p1,-p0) supplies the
    antiparticle sign rather than replacing both flavors by unit charge.
    """
    if len(charges)!=2 or not np.isfinite(charges).all():
        raise ValueError('Two finite quark flavor charges required.')
    first=charges[0]*triangle_unit(mass,outgoing,incoming,order=order,method=method)
    second=charges[1]*triangle_unit(mass,-np.asarray(outgoing),-np.asarray(incoming),order=order,method=method)
    return dict(quark=first,antiquark=second,total=first+second)


def bubble_vertex_norm(mass,s,*,order=96):
    """-d DeltaK_P/ds=4Nc[B(s)+s B'(s)], unit Yukawa, dimensionless."""
    if not np.isfinite([mass,s]).all() or mass<=0 or s>=4*mass*mass:
        raise ValueError('Positive mass and subthreshold real virtuality required.')
    x,w=quadrature(order);x=(x+1)/2;w=w/2;b=x*(1-x)
    d=mass*mass-b*s
    return float(3/(4*np.pi**2)*np.sum(w*(-np.log(d/(mass*mass))+s*b/d)))


def elastic_momenta(mass_squared,q_squared):
    if not np.isfinite([mass_squared,q_squared]).all() or mass_squared<=0 or q_squared<0:
        raise ValueError('Positive external mass squared and nonnegative Q^2 required.')
    q=np.sqrt(q_squared);energy=np.sqrt(mass_squared+q_squared/4)
    return np.array([energy,q/2,0,0]),np.array([energy,-q/2,0,0])


def normalized_loop_shape(mass,s,q_squared,*,order=28,method='reduced'):
    """One-loop LSZ/charge-subtracted deltaF=vertex(t)-vertex(0), unit g^2.

    Evaluation at a prescribed tree pole is valid at this order, not a solved
    loop-shifted state. The free local current is normalized separately.
    Unknown renormalized local C is NOT set to zero by this subtraction.
    """
    p1,p0=elastic_momenta(s,q_squared)
    vertex=photon_attachments(mass,p1,p0,order=order,method=method)['total']
    return float(vertex[0]/(p1[0]+p0[0])-bubble_vertex_norm(mass,s))


def lf_spin_numerator(x,kx,ky,mass,*,plus=1.):
    """Independent canonical ubar i gamma5 v times sqrt(x(1-x)), 2x2.

    Spin order(-,+); pair state not normalized in the continuum. The local
    Yukawa cloud has a logarithmically divergent unsubtracted probability.
    """
    if not 0<x<1 or plus<=0: raise ValueError('Interior fraction and positive total plus required.')
    u=np.column_stack([lf_spinor(x*plus,kx,ky,mass,h) for h in [-1,1]])
    v=np.column_stack([antiquark_spinor((1-x)*plus,-kx,-ky,mass,h) for h in [-1,1]])
    return np.sqrt(x*(1-x))*u.conj().T@GAMMA[0]@(1j*G5)@v


def lf_overlap_shape(mass,s,q_squared,*,order=64,charges=(2/3,-1/3),orbital=True):
    """Independent Drell--Yan LF plus-current overlap; no other LF component.

    Psi=-sqrt(2)*N/(kT^2+M^2-x(1-x)s). Tr N(k')^dagger N(k)
    =2(M^2+k'.k), including both mass/helicity and orbital terms.
    Charge subtraction and +/- angular pairing occur BEFORE radial
    integration. This neither normalizes a finite Fock state nor asserts
    that missing LF minus/transverse or zero-mode terms vanish. orbital=False
    deliberately deletes the kT numerator as a negative information control.
    """
    _domain(mass,*elastic_momenta(s,q_squared))
    if len(charges)!=2 or not np.isfinite(charges).all():
        raise ValueError('Two finite flavor charges required.')
    x,wx=quadrature(order);x=(x+1)/2;wx=wx/2
    u,wu=quadrature(order);u=(u+1)/2;wu=wu/2
    radial=mass*mass*u/(1-u);wr=mass*mass*wu/(1-u)**2
    angle=np.arange(2*order)*np.pi/order
    kx=np.sqrt(radial)[:,None]*np.cos(angle)[None,:]
    total=0.
    for z,wz in zip(x,wx):
        d=mass*mass+radial[:,None]-z*(1-z)*s
        n=mass*mass+radial[:,None] if orbital else np.full_like(d,mass*mass)
        for charge,shift in [(charges[0],(1-z)*np.sqrt(q_squared)),(-charges[1],-z*np.sqrt(q_squared))]:
            b=shift*kx;c=shift*shift
            # Average paired kx and -kx; algebraically subtract the same norm.
            angular_numerator=d-2*n if orbital else -2*n
            difference=(-2*b*b*angular_numerator-n*c*(d+c))/(d*d*((d+c)**2-4*b*b))
            total+=wz*charge*np.sum(wr*np.mean(difference,axis=1))
    return float(3/(4*np.pi**2)*total)


def lf_parameter_shape(mass,s,q_squared,*,order=64):
    """Independently integrate the 2D LF transverse overlap analytically.

    Remaining x,v parameters are NOT the covariant triangle parameters.
    Both charge-weighted recoil shifts are retained, no normalized cloud.
    """
    _domain(mass,*elastic_momenta(s,q_squared))
    x,wx=quadrature(order);x=(x+1)/2;wx=wx/2
    v,wv=quadrature(order);v=(v+1)/2;wv=wv/2
    beta=v*(1-v);total=0.
    for z,wz in zip(x,wx):
        d=mass*mass-z*(1-z)*s
        for charge,shift_squared in [(2/3,(1-z)**2*q_squared),(1/3,z*z*q_squared)]:
            delta=beta*shift_squared
            # Pair the rational subtraction stably before integration.
            integrand=-np.log1p(delta/d)-delta*(d+mass*mass)/(d*(d+delta))
            total+=wz*charge*(wv@integrand)
    return float(3/(4*np.pi**2)*total)


def loop_slope(mass,s,*,order=96):
    """d deltaF/dt at t=0, GeV^-2, unit Yukawa; analytic LF-integral derivative."""
    _domain(mass,*elastic_momenta(s,0.))
    x,w=quadrature(order);x=(x+1)/2;w=w/2;d=mass*mass-x*(1-x)*s
    recoil=(2/3)*(1-x)**2+(1/3)*x*x
    return float(3/(24*np.pi**2)*np.sum(w*recoil*(2/d+mass*mass/(d*d))))

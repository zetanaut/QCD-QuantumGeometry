"""One Yukawa/one external EM insertion: constrained LF operator and tree.

All components are (+,-,x,y), symmetric LF convention. The source is a
classical spacelike Fourier mode with q+=0, NOT an on-shell photon. Full
spin/flavor/color and two collective nodes survive. No old loop is a bare
vertex, no production potential or physical nucleon is selected. D/E begin
at two photons; independent one-photon C is retained here.

EM convention: D_q=partial+i e Q A, so the external chiral vector source
used by the preserved O(4) density is v=-e Q A. Its action_vertex differentiates
with respect to positive v; the physical Hamiltonian insertion has the same
reported vertex after both the v=-e A and H=-L sign conversions.
"""
from itertools import product
from math import sqrt, pi

import numpy as np
from scipy.integrate import solve_ivp

from .canonical_lf_fields import GAMMA,GAMMA_PLUS,LAMBDA_PLUS,slash
from .canonical_lf_network import four_vector,momentum
from .collective_quark_matching import bilinear_matrices
from .finite_lf_coupling import spinors,collective_pair_leaf,reference_inputs,pair_routes
from .nonzero_momentum_current import fermion_propagators
from .transverse_source_matching import transverse_vertex

BETA=np.kron(GAMMA[0],np.eye(6))
ALPHA=tuple(np.kron(GAMMA[0]@GAMMA[j],np.eye(6)) for j in (1,2))
GOOD=np.kron(LAMBDA_PLUS,np.eye(6))
CHARGE=np.kron(np.kron(np.eye(4),np.diag([2/3,-1/3])),np.eye(3))
BILINEARS=bilinear_matrices()
PHI_CHARGE=np.zeros((4,4),complex)
PHI_CHARGE[1,2]=-1j;PHI_CHARGE[2,1]=1j


def cart(lf):
    p=np.asarray(lf)
    return np.array([(p[0]+p[1])/sqrt(2),p[2],p[3],(p[0]-p[1])/sqrt(2)])


def lf(cartesian):
    p=np.asarray(cartesian)
    return np.array([(p[0]+p[3])/sqrt(2),(p[0]-p[3])/sqrt(2),p[1],p[2]])


def good_spinors(p,mass,*,anti=False):
    return GOOD@spinors(p,mass,anti=anti)/sqrt(sqrt(2)*p[0])


def b0(p,mass):
    return mass*BETA+sum(x*a for x,a in zip(np.asarray(p)[1:],ALPHA))


def _source(epsilon):
    e=np.asarray(epsilon,complex)
    if e.shape!=(4,) or not np.isfinite(e).all():
        raise ValueError('Finite (+,-,x,y) classical source required.')
    return e


def _geometry(a,b,p):
    a,b,p=map(momentum,(a,b,p));q=a+b-p
    if abs(q[0])>1e-12 or np.linalg.norm(q[1:])<1e-12:
        raise ValueError('This control requires q+=0 and nonzero transverse transfer.')
    return a,b,p,q


def good_yukawa_pair(a,b,p,mass):
    """Independent good-field constraint, signed anti Fourier denominator."""
    a,b,p=map(momentum,(a,b,p))
    if not np.allclose(a+b,p,rtol=0,atol=1e-12):
        raise ValueError('Conserved Yukawa momentum required.')
    u,v=good_spinors(a,mass),good_spinors(b,mass,anti=True)
    out,inc=b0(a,mass),b0(-b,mass)
    values=[u.conj().T@(out@BETA@z/a[0]-BETA@z@inc/b[0])@v/2
            for z in BILINEARS]
    return np.array(values).transpose(1,2,0)/sqrt(2*p[0])


def fermion_source(outgoing,incoming,mass,epsilon,*,anti=False):
    """Canonical one-EM-source block, flavor charges and normal ordering.

    The direct full-spinor check is separate. No field normalization depends
    on mass, and signed anti momentum is NOT replaced by absolute value in H.
    """
    a,b=map(momentum,(outgoing,incoming));e=_source(epsilon)
    if abs(a[0]-b[0])>1e-12:
        raise ValueError('Zero-plus transfer only in this bounded source block.')
    ba=-sum(e[j+2]*ALPHA[j]@CHARGE for j in range(2))
    if not anti:
        u,v=good_spinors(a,mass),good_spinors(b,mass)
        kernel=(e[1]*CHARGE+(b0(a,mass)@ba/b[0]+ba@b0(b,mass)/a[0])/2
                +e[0]*b0(a,mass)@CHARGE@b0(b,mass)/(2*a[0]*b[0]))
        return u.conj().T@kernel@v
    # d(out)^dagger d(in) comes from reversing the two anti field endpoints.
    u,v=good_spinors(b,mass,anti=True),good_spinors(a,mass,anti=True)
    kernel=(e[1]*CHARGE-(b0(-b,mass)@ba/a[0]+ba@b0(-a,mass)/b[0])/2
            +e[0]*b0(-b,mass)@CHARGE@b0(-a,mass)/(2*a[0]*b[0]))
    return -(u.conj().T@kernel@v).T


def full_spinor_source(outgoing,incoming,mass,epsilon,*,anti=False):
    a,b=map(momentum,(outgoing,incoming));vertex=np.kron(slash(cart(_source(epsilon))),np.eye(6))@CHARGE
    if anti:
        u,v=spinors(b,mass,anti=True),spinors(a,mass,anti=True)
        return -(u.conj().T@BETA@vertex@v).T/(2*sqrt(a[0]*b[0]))
    return spinors(a,mass).conj().T@BETA@vertex@spinors(b,mass)/(2*sqrt(a[0]*b[0]))


def mixed_fermion_pair(a,b,p,mass,epsilon,volume):
    """Actual Schur-complement derivative H_(Y,A), shape q12,anti12,O4.

    H=e Q A^- + B[1/(p+ - e Q A+)]B/2;
    B=alpha.(pT-e Q AT)+beta*(m+g Gamma.phi).
    Includes gauging B AND the longitudinal inverse. g and e stripped;
    quantum collective mode factor 1/sqrt(2P+ V) retained, source A GeV.
    """
    a,b,p,_=_geometry(a,b,p);e=_source(epsilon)
    u,v=good_spinors(a,mass),good_spinors(b,mass,anti=True)
    ba=-sum(e[j+2]*ALPHA[j]@CHARGE for j in range(2))
    out,inc=b0(a,mass),b0(-b,mass)
    values=[]
    for z in BILINEARS:
        y=BETA@z
        kernel=(ba@y/a[0]-y@ba/b[0]
                +e[0]*(out@CHARGE@y/a[0]**2+y@CHARGE@inc/b[0]**2))/2
        values.append(u.conj().T@kernel@v/sqrt(2*p[0]*volume))
    return np.array(values).transpose(1,2,0)


def covariant_instant_pair(a,b,p,mass,epsilon,volume):
    """Independent gamma+ propagator pieces, signed crossed anti plus."""
    a,b,p,_=_geometry(a,b,p);u=spinors(a,mass);v=spinors(b,mass,anti=True)
    vertex=CHARGE@np.kron(slash(cart(_source(epsilon))),np.eye(6))
    gp=np.kron(GAMMA_PLUS,np.eye(6));values=[]
    for z in BILINEARS:
        kernel=vertex@gp@z/(2*a[0])-z@gp@vertex/(2*b[0])
        values.append(u.conj().T@BETA@kernel@v/sqrt(8*a[0]*b[0]*p[0]*volume))
    return np.array(values).transpose(1,2,0)


def parent_h(mass_squared,p):
    m=np.asarray(mass_squared,float);p=momentum(p)
    if (m.shape!=(4,2,2) or not np.isfinite(m).all()
            or not np.allclose(m,m.transpose(0,2,1),atol=1e-13,rtol=0)
            or np.min(np.linalg.eigvalsh(m))<=0
            or not np.allclose(m[1:],m[1],atol=1e-13,rtol=0)):
        raise ValueError('Positive symmetric two-node blocks and degenerate isovectors required.')
    result=np.zeros((8,8))
    for A in range(4):
        ix=[A,A+4];result[np.ix_(ix,ix)]=(m[A]+(p[1:]@p[1:])*np.eye(2))/(2*p[0])
    return result


def phi_source_coefficients(mass_squared,c,pout,pin,epsilon):
    """Canonical source J(omega)=J0+omega J1+omega^2 J2, q+=0.

    Includes the SAME independent transverse C, no D/E at one photon.
    The collective normalization W1=(I+q^2 C) Q A+/(2P+) generates an
    additional mixed Yukawa/source vertex; it must not be dropped.
    """
    a,b=map(momentum,(pout,pin));e=_source(epsilon);c=np.asarray(c,float)
    if (abs(a[0]-b[0])>1e-12 or c.shape!=(2,2) or not np.isfinite(c).all()
            or not np.allclose(c,c.T,rtol=0,atol=1e-13)):
        raise ValueError('Same positive plus and finite symmetric C required.')
    ho,hi=parent_h(mass_squared,a),parent_h(mass_squared,b)
    Q=np.kron(np.eye(2),PHI_CHARGE);C=np.kron(c,np.eye(4));ident=np.eye(8)
    q=a[1:]-b[1:];q2=-q@q;delta=a[1:]@a[1:]-b[1:]@b[1:]
    s=a[1:]+b[1:];pp=a[0]
    j0=(e[1]*(ident+q2*C)+e[0]*(ho+hi+q2*(ho@C+C@hi))/(2*pp)
        -(e[2:]@s)*(ident+q2*C)/(2*pp)-(e[2:]@q)*delta*C/(2*pp))@Q
    j1=C@Q*(e[0]*delta/(2*pp)+e[2:]@q)
    j2=-e[0]*C@Q
    w=e[0]*(ident+q2*C)@Q/(2*pp)
    return j0,j1,j2,w


def yukawa_matrix(a,b,p,mass,couplings,volume,*,good=False):
    fn=good_yukawa_pair if good else collective_pair_leaf
    leaf=fn(a,b,p,mass).reshape(144,4)/sqrt(volume)
    return np.concatenate([g*leaf for g in couplings],axis=1)


def tree_response(a,b,p,mass,mass_squared,c,couplings,initial,epsilon,omega,volume):
    """Full normalized phi + spacelike source -> q anti response, one g and e.

    Initial must be an energy eigenvector for this frequency-domain control.
    omega=q- is arbitrary: a time-dependent Yukawa test pulse can supply the
    remaining minus energy. All internal poles are excluded, not prescribed.
    Returns independent covariant and normalized LF time-ordered amplitudes.
    """
    a,b,p,q=_geometry(a,b,p);e=_source(epsilon);seed=np.asarray(initial,complex).reshape(8)
    h=parent_h(mass_squared,p);energy=float(np.vdot(seed,h@seed).real/np.vdot(seed,seed).real)
    if not np.isfinite(omega) or np.linalg.norm(h@seed-energy*seed)>1e-10:
        raise ValueError('Finite frequency and an initial energy eigenvector required.')
    qa=a-q;qb=b-q;out=p+q
    A,B=four_vector(a,mass),four_vector(b,mass)
    P=cart([p[0],energy,*p[1:]]);Q=cart([0,omega,*q[1:]])
    r,s=A-Q,Q-B
    fq,fa=fermion_propagators(r,mass),fermion_propagators(s,mass)
    bar=spinors(a,mass).conj().T@BETA;v=spinors(b,mass,anti=True)
    photon=CHARGE@np.kron(slash(cart(e)),np.eye(6))
    z=np.einsum('n,nA,Aij->ij',np.asarray(couplings),seed.reshape(2,4),BILINEARS)
    norm=sqrt(8*a[0]*b[0]*p[0]*volume)
    endpoint=bar@(photon@np.kron(fq['covariant'],np.eye(6))@z
                   +z@np.kron(fa['covariant'],np.eye(6))@photon)@v/norm
    # Independent covariant collective inverse and actual transverse vertex.
    rout=P+Q;vcur=np.zeros((8,8),complex);dphi=np.zeros((8,8),complex)
    eta=np.array([1,-1,-1,-1]);c=np.asarray(c)
    scalar_min=np.sum((rout+P)*eta*cart(e))
    scalar_c=np.sum(transverse_vertex(rout,P,1.)*eta*cart(e))
    for j,k in product(range(2),repeat=2):
        vcur[4*j:4*j+4,4*k:4*k+4]=((j==k)*scalar_min+c[j,k]*scalar_c)*PHI_CHARGE
    invmass=lf(rout)[0]*2*lf(rout)[1]-rout[1]**2-rout[2]**2
    for aa in range(4):
        ix=[aa,aa+4];block=invmass*np.eye(2)-mass_squared[aa]
        if min(abs(np.linalg.eigvalsh(block)))<1e-10:
            raise ValueError('Collective pole requires a separately declared prescription.')
        dphi[np.ix_(ix,ix)]=np.linalg.inv(block)
    yfinal=yukawa_matrix(a,b,out,mass,couplings,volume)
    collective=yfinal@(dphi@vcur@seed)
    # The classical EM source leaves the quantum collective external factor
    # unchanged at q+=0: yfinal already supplies its 1/sqrt(2P+ V).
    covariant=endpoint.reshape(144)+collective
    eq=float(lf(A)[1]);eb=float(lf(B)[1]);ef=eq+eb
    eqi=lf(four_vector(qa,mass))[1];ebi=lf(four_vector(qb,mass))[1]
    denq=ef-omega-eqi-eb;dena=ef-omega-eq-ebi
    yq=yukawa_matrix(qa,b,p,mass,couplings,volume,good=True)@seed
    ya=yukawa_matrix(a,qb,p,mass,couplings,volume,good=True)@seed
    jq=fermion_source(a,qa,mass,e);ja=fermion_source(b,qb,mass,e,anti=True)
    quark=(jq@yq.reshape(12,12)).reshape(144)/denq
    anti=(ya.reshape(12,12)@ja.T).reshape(144)/dena
    j0,j1,j2,w=phi_source_coefficients(mass_squared,c,out,p,e)
    source=j0+omega*j1+omega**2*j2
    before=yfinal@np.linalg.solve((energy+omega)*np.eye(8)-parent_h(mass_squared,out),source@seed)
    contact=np.einsum('ijA,n,nA->ij',mixed_fermion_pair(a,b,p,mass,e,volume),couplings,seed.reshape(2,4)).reshape(144)
    norm_contact=yfinal@w@seed
    return dict(covariant=covariant,light_front=quark+anti+before+contact+norm_contact,
                quark_after=quark,anti_after=anti,collective_before=before,
                fermion_contact=contact,collective_endpoint=norm_contact,
                covariant_endpoint=endpoint.reshape(144),covariant_collective=collective)


HARMONICS=((-2,1/16),(-1,-1/4),(0,3/8),(1,-1/4),(2,1/16))


def time_integral(delta,time):
    """Integral_0^time exp(-i delta s) ds, including exact resonances."""
    x=np.asarray(delta)*time
    return time*np.exp(-.5j*x)*np.sinc(x/(2*pi))


def partial_pulse(delta,time,duration):
    return sum(c*time_integral(delta-2*pi*k/duration,time) for k,c in HARMONICS)


class PulseTransition:
    """One Y/one EM response in a declared final momentum window.

    Every required shifted intermediate is included, even outside that
    window. This is NOT a global finite-Fock truncation or a claim about
    unrepresented final momenta. A real source uses BOTH +/-q branches;
    use Fourier amplitude 1/2 for each branch of a cosine source.
    """
    def __init__(self,cell,mass_squared,c,couplings,quark_mass,parent,transfer,
                 shell,duration,epsilon,*,pure_gauge=False,representation='good'):
        self.cell=cell;self.m=np.asarray(mass_squared);self.c=np.asarray(c)
        self.g=np.asarray(couplings);self.mass=quark_mass
        self.p=momentum(parent);self.q=np.asarray(transfer,float)
        self.T=float(duration);self.shell=shell
        if (self.q.shape!=(3,) or not np.isfinite(self.q).all() or abs(self.q[0])>1e-12
                or np.linalg.norm(self.q[1:])<1e-12 or self.T<=0 or not np.isfinite(self.T)
                or representation not in ('good','dirac')):
            raise ValueError('Nonzero qT, zero q+, positive duration and known representation required.')
        cell.mode(self.p+self.q,'boson')
        self.routes=pair_routes(cell,self.p+self.q,shell)
        if not self.routes: raise ValueError('Empty declared final momentum window.')
        self.h0=parent_h(self.m,self.p);self.ha=parent_h(self.m,self.p+self.q)
        self.w0,self.u0=np.linalg.eigh(self.h0);self.wa,self.ua=np.linalg.eigh(self.ha)
        eps=_source(epsilon)
        self.omega=np.array([-2*pi*k/self.T for k,_ in HARMONICS])
        # For pure gauge epsilon[0] specifies the dimensionless theta Fourier
        # amplitude; the other entries must vanish, avoiding hidden semantics.
        if pure_gauge and np.linalg.norm(eps[1:])>1e-14:
            raise ValueError('Pure gauge takes theta amplitude in epsilon[0] only.')
        self.fourier=np.array([(-1j*np.array([0,om,*self.q[1:]])*eps[0] if pure_gauge else eps)*coef
                               for om,(_,coef) in zip(self.omega,HARMONICS)])
        self.phi=[];self.phi0=[];self.norm=[]
        for om,e in zip(self.omega,self.fourier):
            j0,j1,j2,w=phi_source_coefficients(self.m,self.c,self.p+self.q,self.p,e)
            self.phi.append(j0+om*j1+om**2*j2);self.phi0.append(j0);self.norm.append(w)
        self.phi=np.array(self.phi);self.phi0=np.array(self.phi0);self.norm=np.array(self.norm)
        ys=[];yq=[];yb=[];energies=[];jq=[];jb=[];kf=[]
        ff=fermion_source if representation=='good' else full_spinor_source
        contact=mixed_fermion_pair if representation=='good' else covariant_instant_pair
        for a,b in self.routes:
            ai=a-self.q;bi=b-self.q
            # These are mode-valid even when they fall outside the final window.
            cell.mode(ai,'fermion');cell.mode(bi,'fermion')
            yy=lambda x,y,z:yukawa_matrix(x,y,z,quark_mass,self.g,cell.volume,good=representation=='good')
            ys.append(yy(a,b,self.p+self.q));yq.append(yy(ai,b,self.p));yb.append(yy(a,bi,self.p))
            em=lambda x:lf(four_vector(x,quark_mass))[1]
            energies.append([em(a)+em(b),em(ai)+em(b),em(a)+em(bi)])
            jq.append([ff(a,ai,quark_mass,e) for e in self.fourier])
            jb.append([ff(b,bi,quark_mass,e,anti=True) for e in self.fourier])
            kf.append([np.concatenate([gg*contact(a,b,self.p,quark_mass,e,cell.volume).reshape(144,4)
                                        for gg in self.g],axis=1) for e in self.fourier])
        self.y=np.array(ys);self.yq=np.array(yq);self.yb=np.array(yb)
        self.energy=np.array(energies);self.jq=np.array(jq);self.jb=np.array(jb);self.kf=np.array(kf)
        self.kn=np.einsum('rji,kil->rkjl',self.y,self.norm)
        self.qproduct=np.einsum('rkij,rjla->rkila',self.jq,self.yq.reshape(-1,12,12,8)).reshape(len(ys),5,144,8)
        self.bproduct=np.einsum('rija,rklj->rkila',self.yb.reshape(-1,12,12,8),self.jb).reshape(len(ys),5,144,8)

    def outside_intermediates(self):
        """Count route/attachment intermediates outside the final square grid."""
        def outside(p): return bool(np.any(abs(p[1:]/self.cell.spacing[1:])>self.shell+1e-10))
        return sum(outside(a-self.q)+outside(b-self.q) for a,b in self.routes)

    def response(self,initial,*,nodes=64,omit=None):
        """Spectral lower-order jets plus outer Gaussian time quadrature.

        Output is d^2 psi / (d lambda d source_amplitude) at zero sources.
        Generic source amplitude is GeV; pure-gauge theta is dimensionless.
        Its norm is not a probability and the jet is not normalized by hand.
        """
        allowed=(None,'quark','anti','collective','fermion_contact','collective_endpoint','source_derivatives','outside_intermediates')
        if omit not in allowed: raise ValueError('Unknown negative control.')
        v=np.asarray(initial,complex).reshape(8);coef=self.u0.conj().T@v
        j=self.phi0 if omit=='source_derivatives' else self.phi
        je=np.einsum('ai,kab,bj->kij',self.ua.conj(),j,self.u0)
        qprod=self.qproduct@self.u0;bprod=self.bproduct@self.u0
        if omit=='outside_intermediates':
            qprod=qprod.copy();bprod=bprod.copy()
            for r,(a,b) in enumerate(self.routes):
                if np.any(abs((a-self.q)[1:]/self.cell.spacing[1:])>self.shell+1e-10): qprod[r]*=0
                if np.any(abs((b-self.q)[1:]/self.cell.spacing[1:])>self.shell+1e-10): bprod[r]*=0
        x,weights=np.polynomial.legendre.leggauss(nodes)
        result=np.zeros((len(self.routes),144),complex)
        for t,weight in zip((x+1)*self.T/2,weights*self.T/2):
            f=np.sin(pi*t/self.T)**4;ph=np.exp(-1j*self.omega*t)
            zero=self.u0@(np.exp(-1j*self.w0*t)*coef)
            drive=np.zeros_like(result)
            for typ,prod in ((1,qprod),(2,bprod)):
                if omit==('quark' if typ==1 else 'anti'): continue
                en=self.energy[:,typ]
                factors=-1j*np.exp(-1j*en[:,None]*t)*partial_pulse(self.w0[None,:]-en[:,None],t,self.T)*coef
                drive+=np.einsum('k,rkji,ri->rj',ph,prod,factors)
            if omit!='collective':
                delta=self.w0[None,None,:]-self.wa[None,:,None]+self.omega[:,None,None]
                pa=-1j*np.exp(-1j*self.wa*t)*np.einsum('kij,kij,j->i',je,time_integral(delta,t),coef)
                drive+=f*np.einsum('rji,i->rj',self.y,self.ua@pa)
            k=self.kf*(omit!='fermion_contact')+self.kn*(omit!='collective_endpoint')
            drive+=f*np.einsum('k,rkji,i->rj',ph,k,zero)
            result+=-1j*weight*np.exp(-1j*self.energy[:,0,None]*(self.T-t))*drive
        return result

    def ode_response(self,initial,*,rtol=2e-10,atol=2e-12):
        """Independent initial-value solve for every lower-order and mixed jet.

        Two intermediate arrays label attachment routes, not additional
        orthogonal physical states or an independent quantum probability.
        """
        v=np.asarray(initial,complex).reshape(8);r=len(self.routes);size=r*144
        def unpack(y):
            return y[:8],y[8:8+size].reshape(r,12,12),y[8+size:8+2*size].reshape(r,12,12),y[8+2*size:].reshape(r,144)
        def rhs(t,y):
            pa,yq,yb,mixed=unpack(y);f=np.sin(pi*t/self.T)**4;ph=np.exp(-1j*self.omega*t)
            zero=self.u0@(np.exp(-1j*self.w0*t)*(self.u0.conj().T@v))
            ja=np.einsum('k,kij->ij',ph,self.phi)
            dq=-1j*(self.energy[:,1,None,None]*yq+f*(self.yq@zero).reshape(r,12,12))
            db=-1j*(self.energy[:,2,None,None]*yb+f*(self.yb@zero).reshape(r,12,12))
            drive=(np.einsum('k,rkij,rjl->ril',ph,self.jq,yq)
                   +np.einsum('rij,k,rklj->ril',yb,ph,self.jb)).reshape(r,144)
            drive+=f*(self.y@pa)+f*np.einsum('k,rkji,i->rj',ph,self.kf+self.kn,zero)
            return np.concatenate([-1j*(self.ha@pa+ja@zero),dq.ravel(),db.ravel(),
                                   (-1j*(self.energy[:,0,None]*mixed+drive)).ravel()])
        sol=solve_ivp(rhs,(0,self.T),np.zeros(8+3*size,complex),rtol=rtol,atol=atol,method='DOP853')
        if not sol.success: raise RuntimeError(sol.message)
        return unpack(sol.y[:,-1])[-1],sol.nfev


def pulse_parameter_features(cell,mass_squared,quark_mass,parent,transfer,shell,
                             duration,epsilon,initial,*,nodes=64):
    """Exact g_n*(1,C00,C01,C11) response tensors at FIXED free masses.

    All features are derived from the same Hamiltonian/source calculation,
    not learned observable heads. This does not differentiate masses or pick
    a fit. C01 changes both symmetric entries once, not twice independently.
    """
    basis=(np.array([[1.,0],[0,0]]),np.array([[0,1.],[1,0]]),np.array([[0,0],[0,1.]]))
    result=[]
    for g in np.eye(2):
        run=lambda c:PulseTransition(cell,mass_squared,c,g,quark_mass,parent,transfer,shell,
                                      duration,epsilon).response(initial,nodes=nodes)
        zero=run(np.zeros((2,2)))
        result.append([zero,*[run(c)-zero for c in basis]])
    return np.array(result)


def torch_pulse_response(features,couplings,c):
    """Native shared-coupling/source derivatives of the complete wave."""
    import torch
    data=torch.tensor(np.array(features,copy=True),dtype=torch.complex128)
    g=torch.as_tensor(couplings,dtype=torch.complex128)
    cc=torch.as_tensor(c,dtype=torch.complex128)
    if g.shape!=(2,) or cc.shape!=(3,): raise ValueError('Two g and three symmetric C entries required.')
    weights=torch.cat([torch.ones(1,dtype=torch.complex128),cc])
    return torch.einsum('n,b,nbrj->rj',g,weights,data)

"""Nonstationary positive-plus LF source map through a mixed two-source jet.

Same charged minimal/C/D/E action, all node entries and both source orders.
Not a zero-plus prescription, complete finite-source theory, quantum loop,
physical hadron or production dynamics choice. Geometry is extracted from
the actual density; algebraic maps retain Torch parameter derivatives.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.integrate import solve_ivp

from .transverse_source_matching import tensor as _tensor,dot,transverse_vertex,action_vertex
from .joint_source_matching import JointSourceAction
from .source_network_transport import contact_coefficients

ENERGY_DIRECTION=np.array([1.,0.,0.,-1.])/np.sqrt(2)


def tensor(value):
    if isinstance(value,np.ndarray) and not value.flags.writeable: value=value.copy()
    return _tensor(value)


def plus(p): return (np.asarray(p)[...,0]+np.asarray(p)[...,3])/np.sqrt(2)
def minus(p): return (np.asarray(p)[...,0]-np.asarray(p)[...,3])/np.sqrt(2)
def anticommutator(a,b): return a@b+b@a
def commutator(a,b): return a@b-b@a


def _hermitian(*matrices):
    for a in matrices:
        if a.ndim!=2 or a.shape[0]!=a.shape[1] or a.shape!=matrices[0].shape or not torch.isfinite(a).all() or not torch.allclose(a,a.conj().T,rtol=0,atol=2e-12):
            raise ValueError('Common finite Hermitian source-route matrices required.')


def canonical_mixed_map(h,frequency,ga,va,gb,vb,gab,vab,dab):
    """All a, b, a*b coefficients; D starts at a*b for the tested action.

    K=V-{G,p}/2-pDp. Source coefficient dot X=-i[frequency,X].
    H contains the time-dependent canonical-frame connection. W is the
    LEFT external map; its right partner is W dagger, not W itself.
    Sources a,b have GeV units in the actual C/D/E map.
    """
    h,p,ga,va,gb,vb,gab,vab,dab=map(tensor,(h,frequency,ga,va,gb,vb,gab,vab,dab))
    _hermitian(h,p,ga,va,gb,vb,gab,vab,dab)
    if not torch.allclose(h@p,p@h,rtol=0,atol=2e-12): raise ValueError('Source-free h must commute with the route frequency.')
    dga=-1j*commutator(p,ga);dgb=-1j*commutator(p,gb)
    connection=1j/8*(commutator(ga,dgb)+commutator(gb,dga))
    product=anticommutator(ga,gb)
    ha=va-anticommutator(ga,h)/2;hb=vb-anticommutator(gb,h)/2
    hab=(vab-anticommutator(gab,h)/2-(anticommutator(ga,vb)+anticommutator(gb,va))/2
         +3*anticommutator(product,h)/8+(ga@h@gb+gb@h@ga)/4+connection-h@dab@h)
    wa,wb=-ga/2,-gb/2;wab=-gab/2+3*product/8-dab@h
    return dict(H=(h,ha,hb,hab),W=(wa,wb,wab),direct=dab,connection=connection)


def inverse_jet(jet):
    k,ka,kb,kab=map(tensor,jet);r=torch.linalg.inv(k)
    return r,-r@ka@r,-r@kb@r,r@ka@r@kb@r+r@kb@r@ka@r-r@kab@r


def canonical_wave_jet(mapping,frequency,time,initial):
    """Differentiable exact monochromatic source-response amplitudes.

    The triangular matrix encodes derivatives, NOT a physical Hamiltonian
    on an enlarged Hilbert space. Each coefficient of the actual H is
    Hermitian. No jet vector is normalized as though it were a state.
    """
    h,ha,hb,hab=mapping['H'];p=tensor(frequency);u=tensor(initial);m=len(h)
    if u.shape!=(m,) or not torch.isfinite(u).all() or not torch.isclose(torch.vdot(u,u).real,tensor(1.).real):
        raise ValueError('Normalized canonical initial state required.')
    zero=h*0;rot=h-p
    block=torch.cat([torch.cat(row,dim=1) for row in [
        [rot,zero,zero,zero],[ha,rot,zero,zero],
        [hb,zero,rot,zero],[hab,hb,ha,rot]]],dim=0)
    result=(torch.matrix_exp(-1j*time*block)[:,:m]@u).reshape(4,m)
    phase=torch.matrix_exp(-1j*time*p)
    return torch.stack([phase@x for x in result])


def mapped_jet(mapping,frequency,*,omit=None):
    """Complete endpoint multiplication of both time-ordered H responses."""
    if omit not in (None,'connection','endpoints','direct','one_order','right_adjoint'):
        raise ValueError('Declared diagnostic omission required.')
    h,ha,hb,hab=mapping['H'];wa,wb,wab=mapping['W'];direct=mapping['direct']
    if omit=='connection': hab=hab-mapping['connection']
    if omit=='endpoints': wa,wb,wab=wa*0,wb*0,wab*0
    if omit=='direct': direct=direct*0
    r,ra,rb,rab=inverse_jet((h-tensor(frequency),ha,hb,hab))
    if omit=='one_order': rab=rab-r@hb@r@ha@r
    adjoint=(lambda x:x) if omit=='right_adjoint' else (lambda x:x.conj().T)
    a=ra+wa@r+r@adjoint(wa);b=rb+wb@r+r@adjoint(wb)
    ab=(rab+wa@rb+wb@ra+ra@adjoint(wb)+rb@adjoint(wa)+wab@r+r@adjoint(wab)
        +wa@r@adjoint(wb)+wb@r@adjoint(wa)+direct)
    return r,a,b,ab


@dataclass(frozen=True)
class SourceFrame:
    """Four momenta 0,a,b,a+b; exact for this mixed transition, not a full basis.

    Common omega shifts all minus energies while fixed source transfers and
    positive-plus kinetic factors stay explicit. Both reverse Fourier routes
    are represented. They are required by Hermiticity, not new fitted edges.
    """
    p0: np.ndarray
    qa: np.ndarray
    qb: np.ndarray
    epsa: np.ndarray
    epsb: np.ndarray
    routes: tuple = ((0,0),(1,0),(0,1),(1,1))

    def __post_init__(self):
        routes=tuple(tuple(x) for x in self.routes)
        if not routes or len(set(routes))!=len(routes) or any(len(x)!=2 or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) for i in x) for x in routes):
            raise ValueError('Distinct integer source-route coordinates required.')
        object.__setattr__(self,'routes',routes)
        for name in ('p0','qa','qb','epsa','epsb'):
            value=np.array(getattr(self,name),float,copy=True)
            if value.shape!=(4,) or not np.isfinite(value).all(): raise ValueError('Finite momentum and polarization four-vectors required.')
            value.flags.writeable=False;object.__setattr__(self,name,value)
        ps=self.momenta(0.)
        if min(plus(ps))<=0 or any(np.allclose(p,k,rtol=0,atol=1e-12) for i,p in enumerate(ps) for k in ps[:i]):
            raise ValueError('Distinct strictly positive-plus routes required; no zero-mode prescription inferred.')
        # The C/D/E action is polynomial of degree at most two in common
        # minus energy. C/minimal one-source terms are analytically linear;
        # no first-order LF velocity-squared term is silently discarded.
        center=self.geometry(0.);positive=self.geometry(.25);negative=self.geometry(-.25)
        coefficients=[]
        for k in range(3):
            linear=(positive[k]-negative[k])/.5
            quadratic=(positive[k]+negative[k]-2*center[k])/(2*.25**2)
            if k<2:
                if np.linalg.norm(quadratic)>2e-11: raise ValueError('Unexpected D1: derive its additional source partners first.')
                quadratic=np.zeros_like(quadratic)  # analytically absent, tested independently
            coefficients.append((center[k],linear,quadratic))
        object.__setattr__(self,'coefficients',tuple(coefficients))

    def momenta(self,omega):
        if not np.isfinite(omega): raise ValueError('Finite common minus-energy coordinate required.')
        return np.array([self.p0+i*self.qa+j*self.qb for i,j in self.routes])+omega*ENERGY_DIRECTION

    def weak_extension(self):
        """All momenta reached from current seeds through 0,a,b,a*b.

        Preserves the original seed ordering. Not closure through a^2/b^2,
        all photon orders, pair creation or a production Fock truncation.
        """
        extra=sorted({(i+di,j+dj) for i,j in self.routes for di in [-1,0,1] for dj in [-1,0,1]}-set(self.routes))
        return SourceFrame(self.p0,self.qa,self.qb,self.epsa,self.epsb,self.routes+tuple(extra))

    def geometry(self,omega,*,actual_one=False):
        """Geometric minimal/C and minimal/C/D/E matrices before node dressing."""
        ps=self.momenta(omega);count=len(ps);onea=np.zeros((count,count,2));oneb=np.zeros_like(onea);two=np.zeros((count,count,4))
        unit=JointSourceAction(np.ones((1,1)),np.zeros((1,1)),np.zeros((1,1)))
        for i,p1 in enumerate(ps):
            for j,p0 in enumerate(ps):
                if i==j: continue
                for target,q,eps in [(onea,self.qa,self.epsa),(oneb,self.qb,self.epsb)]:
                    if any(np.allclose(p1-p0,sign*q,rtol=0,atol=2e-12) for sign in [-1,1]):
                        vertex=action_vertex(unit,p1,p0,[1.],[1.]) if actual_one else transverse_vertex(p1,p0,1.)
                        contracted=dot(vertex,eps)
                        if abs(contracted.imag if np.iscomplexobj(contracted) else 0)>2e-12: raise ValueError('Unexpected complex EM density coefficient.')
                        target[i,j]=[dot(p1+p0,eps),float(np.real(contracted))]
                for sa in [-1,1]:
                    for sb in [-1,1]:
                        if np.allclose(p1-p0,sa*self.qa+sb*self.qb,rtol=0,atol=2e-12):
                            two[i,j]+=(np.r_[2*dot(self.epsa,self.epsb),contact_coefficients(p1,p0,sa*self.qa,sb*self.qb,self.epsa,self.epsb)])
        return onea,oneb,two

    def matrices(self,mass,c,d,e):
        mass,c,d,e=map(tensor,(mass,c,d,e));_hermitian(mass,c,d,e)
        if torch.linalg.eigvalsh(mass).min()<=0: raise ValueError('Positive reference mass-squared matrix required.')
        n=len(mass);eye=torch.eye(n,dtype=torch.complex128);ps=self.momenta(0.)
        norm=np.sqrt(2*plus(ps));offset=minus(ps)
        h=torch.block_diag(*[(mass+np.dot(p[1:3],p[1:3])*eye)/(2*plus(p)) for p in ps])
        frequency=torch.diag(tensor(np.repeat(offset,n)))
        def dress(geometry,node_matrices):
            out=mass.new_zeros((len(ps)*n,len(ps)*n))
            for field,node in enumerate(node_matrices):
                weights=geometry[:,:,field]/(norm[:,None]*norm[None,:])
                out=out+torch.kron(tensor(weights),node)
            return out
        polys=[]
        for k,coefficients in enumerate(self.coefficients):
            nodes=(eye,c) if k<2 else (eye,c,d,e)
            polys.append(tuple(-dress(geo,nodes) for geo in coefficients))
        # K1=A0+omega*A1; K2=A0+omega*A1+omega^2*A2.
        # Convert to V-{G,p}/2-pDp at p=omega+offset, without freezing q-.
        (a0,a1,_),(b0,b1,_),(ab0,ab1,ab2)=polys
        ga,gb=-a1,-b1
        va=a0+anticommutator(ga,frequency)/2;vb=b0+anticommutator(gb,frequency)/2
        dab=-ab2;gab=-ab1-anticommutator(frequency,dab)
        vab=ab0+anticommutator(gab,frequency)/2+frequency@dab@frequency
        return h,frequency,ga,va,gb,vb,gab,vab,dab

    def mapping(self,mass,c,d,e): return canonical_mixed_map(*self.matrices(mass,c,d,e))

    def inverse_jets(self,mass,c,d,e,omega,*,actual=False):
        mass,c,d,e=map(tensor,(mass,c,d,e));_hermitian(mass,c,d,e)
        n=len(mass);eye=torch.eye(n,dtype=torch.complex128);ps=self.momenta(omega);norm=np.sqrt(2*plus(ps))
        k=torch.block_diag(*[(mass-dot(p,p)*eye)/(2*plus(p)) for p in ps])
        geo=self.geometry(omega,actual_one=True) if actual else tuple(z+omega*l+omega**2*q for z,l,q in self.coefficients)
        jets=[]
        for i,geometry in enumerate(geo):
            node=(eye,c) if i<2 else (eye,c,d,e)
            jets.append(-sum(torch.kron(tensor(geometry[:,:,j]/(norm[:,None]*norm[None,:])),x) for j,x in enumerate(node)))
        return k,*jets

    def frequency(self,nodes,omega):
        return torch.diag(tensor(np.repeat(minus(self.momenta(omega)),nodes)))

    def unnormalize_response(self,value):
        """Both external 1/sqrt(2p+) factors, not a single response rescale."""
        n=value.shape[0]//len(self.routes);factor=tensor(np.repeat(1/np.sqrt(2*plus(self.momenta(0))),n))
        return factor[:,None]*value*factor[None,:]


def evolve_weak_wave(matrices,duration,initial,*,samples=21,rtol=2e-10,atol=2e-12):
    """Independent canonical and original LF equations through a,b,a*b.

    Monochromatic source coefficients X(t)=exp(-i delta t)X exp(i delta t)
    retain the fixed action frequencies. No artificial envelope with missing
    curvature derivatives. Initial canonical state is source independent;
    original initial fields include W. The returned jets are weak response
    connected one-charge wave amplitudes, NOT a complete sourced Fock
    state. Vacuum/pair sectors are not supplied by a one-body wave equation.
    """
    mapping=canonical_mixed_map(*matrices)
    h,delta,ga,va,gb,vb,gab,vab,dab=[tensor(x).detach().numpy() for x in matrices]
    _,ha,hb,hab=[x.detach().numpy() for x in mapping['H']]
    wa,wb,wab=[x.detach().numpy() for x in mapping['W']]
    if not np.allclose(delta,np.diag(np.diag(delta)),rtol=0,atol=1e-13): raise ValueError('Explicit diagonal route frequencies required for this evolution control.')
    u=np.asarray(initial,complex);m=len(h)
    if u.shape!=(m,) or not np.isfinite(u).all() or not np.isclose(np.vdot(u,u),1) or duration<=0 or not np.isfinite(duration) or samples<2:
        raise ValueError('Normalized finite initial state, positive duration and samples required.')
    freq=np.diag(delta).real;times=np.linspace(0,duration,samples)
    def moved(x,t):
        phase=np.exp(-1j*freq*t);return phase[:,None]*x*phase.conj()[None,:]
    def derivative(x): return -1j*(delta@x-x@delta)
    def canonical_rhs(t,flat):
        a0,a,b,ab=flat.reshape(4,m);aH,bH,abH=[moved(x,t) for x in [ha,hb,hab]]
        return (-1j*np.array([h@a0,h@a+aH@a0,h@b+bH@a0,h@ab+aH@b+bH@a+abH@a0])).ravel()
    def original_rhs(t,flat):
        f0,fa,fb,fab=flat.reshape(4,m)
        ag,av,bg,bv,abg,abv,abd=[moved(x,t) for x in [ga,va,gb,vb,gab,vab,dab]]
        agdot,bgdot,abgdot,abddot=[derivative(x) for x in [ag,bg,abg,abd]]
        f0dot=-1j*h@f0
        fadot=-1j*(h@fa+(av-.5j*agdot)@f0-1j*ag@f0dot)
        fbdot=-1j*(h@fb+(bv-.5j*bgdot)@f0-1j*bg@f0dot)
        fabdot=-1j*(h@fab+(av-.5j*agdot)@fb-1j*ag@fbdot
            +(bv-.5j*bgdot)@fa-1j*bg@fadot+(abv-.5j*abgdot)@f0-1j*abg@f0dot
            +abddot@f0dot-abd@h@h@f0)
        return np.array([f0dot,fadot,fbdot,fabdot]).ravel()
    starts=[np.array([u,0*u,0*u,0*u]),np.array([u,wa@u,wb@u,wab@u])]
    solutions=[]
    for rhs,start in zip([canonical_rhs,original_rhs],starts):
        sol=solve_ivp(rhs,(0,duration),start.ravel(),method='DOP853',t_eval=times,rtol=rtol,atol=atol)
        if not sol.success: raise ValueError(sol.message)
        solutions.append(sol.y.T.reshape(samples,4,m))
    canonical,original=solutions;mapped=[];canonical_norm=[];original_norm=[];wrong_norm=[]
    for t,psi,phi in zip(times,canonical,original):
        aW,bW,abW=[moved(x,t) for x in [wa,wb,wab]]
        p0,pa,pb,pab=psi
        mapped.append([p0,pa+aW@p0,pb+bW@p0,pab+aW@pb+bW@pa+abW@p0])
        canonical_norm.append(2*np.real(np.vdot(p0,pab)+np.vdot(pa,pb)))
        f0,fa,fb,fab=phi;ag,bg,abg,abd=[moved(x,t) for x in [ga,gb,gab,dab]]
        metric=(2*np.real(np.vdot(f0,fab)+np.vdot(fa,fb)+np.vdot(f0,ag@fb)+np.vdot(f0,bg@fa))+np.vdot(f0,abg@f0).real)
        dnorm=np.vdot(f0,(abd@h+h@abd)@f0).real
        original_norm.append(metric+dnorm);wrong_norm.append(metric)
    return dict(times=times,canonical=canonical,original=original,mapped_original=np.asarray(mapped),
        canonical_ab_norm=np.array(canonical_norm),original_ab_Noether_norm=np.array(original_norm),
        omitting_D_velocity_norm=np.array(wrong_norm))

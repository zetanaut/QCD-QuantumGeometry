"""Canonical source generator and a nonzero-mode quantum reference.

Charged quadratic slice of the existing C/D/E action, not production
binding dynamics or a quantum-matched pion/nucleon. Regular time and the
bounded stationary LF weak-source map are explicitly different controls.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy import sparse
from scipy.special import eval_hermite,gammaln

from .joint_source_matching import JointSourceAction
from .transverse_source_matching import connection,tensor,METRIC
from .retained_mode_quantum import oscillator_states
from .source_network_transport import symmetric_basis

T_CHARGE=np.array([[0.,-1.],[1.,0.]])
J_SPACE=-T_CHARGE
K_INTERNAL=np.kron(T_CHARGE,J_SPACE)


@dataclass(frozen=True)
class ChargedModes:
    mass_squared: np.ndarray
    source: JointSourceAction
    momentum: float

    def __post_init__(self):
        h=np.array(self.mass_squared,float,copy=True);k=self.momentum
        if h.shape!=self.source.c.shape or not np.isfinite(h).all() or not np.allclose(h,h.T,rtol=0,atol=1e-13) or np.linalg.eigvalsh(h).min()<=0 or not np.isfinite(k) or k<=0:
            raise ValueError('Common positive reference mass matrix and nonzero spatial momentum required.')
        h.flags.writeable=False;object.__setattr__(self,'mass_squared',h)

    @property
    def dimension(self): return 4*len(self.mass_squared)

    def matrices(self,a,adot,addot=0.):
        """Spatial integral: L=dot Q Z dot Q/2+dot Q B Q-Q V Q/2.

        A_x(t)=a(t) T, normalized cosine/sine pair at k>0. Q GeV^-1/2,
        P GeV^1/2, a GeV, adot GeV^2; Z dimensionless, B GeV, V GeV^2.
        Both real charged components and both spatial modes are retained.
        """
        if not np.isfinite([a,adot,addot]).all(): raise ValueError('Finite source and time derivatives required.')
        n=len(self.mass_squared);eye=np.eye(self.dimension)
        c,d,e=[np.kron(x,np.eye(4)) for x in (self.source.c,self.source.d,self.source.e)]
        charge=np.kron(np.eye(n),np.kron(T_CHARGE,np.eye(2)))
        dx=self.momentum*np.kron(np.eye(n),np.kron(np.eye(2),J_SPACE))+a*charge
        z=eye+2*adot**2*e;b=2*adot*c@charge@dx
        v=dx.T@z@dx+np.kron(self.mass_squared,np.eye(4))-4*adot**2*d
        zdot=4*adot*addot*e
        bdot=2*addot*c@charge@dx+2*adot**2*c@charge@charge
        if np.linalg.eigvalsh(z).min()<=0: raise ValueError('Source leaves the regular positive kinetic domain.')
        return z,b,v,zdot,bdot

    def hessian(self,a,adot,*,naive=False):
        """Weyl-ordered canonical H; naive is a deliberately wrong -L_int test."""
        z,b,v,_,_=self.matrices(a,adot)
        if naive:
            return np.block([[v,-b.T],[-b,2*np.eye(len(z))-z]])
        inverse=np.linalg.inv(z)
        return np.block([[v+b.T@inverse@b,-b.T@inverse],[-inverse@b,inverse]])

    def normal_basis(self):
        values,u=np.linalg.eigh(self.mass_squared)
        return np.repeat(np.sqrt(values+self.momentum**2),4),np.kron(u,np.eye(4))


def action_mode_lagrangian(system,coordinate,velocity,a,adot):
    """Independent normalized Fourier integral of the original field density.

    Spatial volume cancels in this quadratic slice after phi=Q*u_k with
    integral u_r*u_s=delta_rs. No finite-volume matching scale is inferred.
    Four quadrature points integrate its quadratic trigonometric form exactly.
    """
    q=tensor(coordinate).reshape(-1,2,2);v=tensor(velocity).reshape(-1,2,2)
    sources=np.zeros((4,3));sources[1,2]=a
    conn=connection(sources,np.zeros((4,3)))
    derivative=torch.zeros((4,4,4,4),dtype=torch.complex128)
    derivative[0,1,1,2]=-adot;derivative[0,1,2,1]=adot
    values=[]
    for angle in [0.,np.pi/2,np.pi,3*np.pi/2]:
        basis=tensor(np.sqrt(2)*np.array([np.cos(angle),np.sin(angle)]))
        dbasis=tensor(system.momentum*np.sqrt(2)*np.array([-np.sin(angle),np.cos(angle)]))
        charged=torch.einsum('ics,s->ic',q,basis);dt=torch.einsum('ics,s->ic',v,basis);dx=torch.einsum('ics,s->ic',q,dbasis)
        zeros=charged[:,0]*0
        fields=torch.stack([zeros,charged[:,0],charged[:,1],zeros],dim=1)
        time=torch.stack([zeros,dt[:,0],dt[:,1],zeros],dim=1)
        space=torch.stack([zeros,dx[:,0],dx[:,1],zeros],dim=1)
        df=torch.stack([time,space,fields*0,fields*0])
        cov=df+torch.einsum('mab,ib->mia',conn,fields)
        free=.5*torch.einsum('mia,mia,m->',cov,cov,tensor(METRIC))-.5*torch.einsum('ij,ia,ja->',tensor(system.mass_squared.copy()),fields,fields)
        values.append(free+system.source.density(fields,df,conn,derivative))
    return sum(values).real/4


def pulse(time,duration,amplitude):
    """Compact sin^4 pulse, a/adot/addot vanish at endpoints; units GeV."""
    if not np.isfinite([time,duration,amplitude]).all() or duration<=0: raise ValueError('Finite pulse and positive duration required.')
    if time<=0 or time>=duration: return 0.,0.,0.
    w=np.pi/duration;s,c=np.sin(w*time),np.cos(w*time)
    return amplitude*s**4,4*amplitude*w*s**3*c,4*amplitude*w*w*(3*s*s*c*c-s**4)


def phase_matrix(m):
    return np.block([[np.zeros((m,m)),np.eye(m)],[-np.eye(m),np.zeros((m,m))]])


def phase_evolution(system,duration,amplitude,*,method='canonical',rtol=2e-10,atol=2e-12):
    """Independent first/second-order fundamental matrices, not quantum fitting."""
    m=system.dimension;j=phase_matrix(m)
    if method not in ('canonical','lagrangian','naive','frozen_source_derivatives'): raise ValueError('Declared evolution control required.')
    def rhs(t,flat):
        a,adot,addot=pulse(t,duration,amplitude);matrix=flat.reshape(2*m,2*m)
        if method in ('canonical','naive'):
            generator=j@system.hessian(a,adot,naive=method=='naive')
        else:
            z,b,v,zdot,bdot=system.matrices(a,adot,addot)
            if method=='frozen_source_derivatives': zdot,bdot=zdot*0,bdot*0
            generator=np.block([[np.zeros((m,m)),np.eye(m)],[-np.linalg.solve(z,bdot+v),-np.linalg.solve(z,zdot+b-b.T)]])
        return (generator@matrix).ravel()
    result=solve_ivp(rhs,(0,duration),np.eye(2*m).ravel(),method='DOP853',rtol=rtol,atol=atol)
    if not result.success: raise ValueError(result.message)
    # Pulse endpoints are exactly source-free: P=velocity there, so the
    # second-order coordinate/velocity matrix has the same endpoint frame.
    return result.y[:,-1].reshape(2*m,2*m)


def initial_covariance(frequencies,excitation=None):
    w=np.asarray(frequencies,float);m=len(w)
    result=np.diag(np.r_[1/(2*w),w/2])
    if excitation is not None:
        u=np.asarray(excitation,complex)
        if u.shape!=(m,) or not np.isclose(np.vdot(u,u),1): raise ValueError('Normalized one-particle mode required.')
        density=np.outer(u.conj(),u)
        qq=density.real/np.sqrt(w[:,None]*w[None,:]);pp=density.real*np.sqrt(w[:,None]*w[None,:])
        qp=density.imag*np.sqrt(w[None,:]/w[:,None])
        result+=np.block([[qq,qp],[qp.T,pp]])
    return result


def charged_excitation(system):
    """Lowest mass mode, charge +1 and physical momentum +k in our basis."""
    node=np.zeros(len(system.mass_squared));node[0]=1.
    return np.kron(node,np.kron(np.array([1.,1j])/np.sqrt(2),np.array([1.,1j])/np.sqrt(2)))


def gaussian_state(system,duration,amplitude,*,rtol=2e-10,atol=2e-12):
    """Actual Schrödinger Gaussian and one-particle polynomial, including phase.

    psi0=N exp(-Q A Q/2), psi1=(b.Q)psi0; no classical covariance-only
    surrogate or removal of source-dependent vacuum energy.
    """
    w,u=system.normal_basis();m=len(w);transform=np.block([[u,u*0],[u*0,u]])
    b0=np.sqrt(2*w)*charged_excitation(system);logn0=.25*np.log(w/np.pi).sum()
    start=np.r_[np.diag(w).ravel().astype(complex),complex(logn0),b0]
    def rhs(t,state):
        a,adot,_=pulse(t,duration,amplitude);h=transform.T@system.hessian(a,adot)@transform
        k,l,inverse=h[:m,:m],h[m:,:m],h[m:,m:]
        width=state[:m*m].reshape(m,m);polynomial=state[m*m+1:]
        dwidth=1j*(k-width@inverse@width)-(l.T@width+width@l)
        dlog=-.5j*np.trace(inverse@width)-.5*np.trace(l)
        db=-1j*width@inverse@polynomial-l.T@polynomial
        return np.r_[dwidth.ravel(),dlog,db]
    result=solve_ivp(rhs,(0,duration),start,method='DOP853',rtol=rtol,atol=atol)
    if not result.success: raise ValueError(result.message)
    last=result.y[:,-1];width=last[:m*m].reshape(m,m);logn=last[m*m];b=last[m*m+1:]
    if np.linalg.eigvalsh(width.real).min()<=0: raise ValueError('Unresolved normalizable Gaussian width.')
    combined=np.diag(w)+width
    # Eigenvalues remain in the right half-plane; summing individual logs
    # keeps the Gaussian determinant phase, unlike a principal total sqrt.
    overlap=np.exp(logn0+logn+.5*m*np.log(2*np.pi)-.5*np.log(np.linalg.eigvals(combined)).sum())
    one_overlap=overlap*np.vdot(b0,np.linalg.solve(combined,b))
    return dict(width=width,log_normalization=logn,polynomial=b,vacuum_overlap=overlap,one_particle_overlap=one_overlap,frequencies=w)


def gaussian_wavefunction(result,coordinate,*,one_particle=False):
    q=np.asarray(coordinate,float);value=np.exp(result['log_normalization']-.5*q@result['width']@q)
    return value*(result['polynomial']@q if one_particle else 1.)


def gaussian_fock_coefficients(result,states):
    """Exact infinite-solution amplitudes on a complete oscillator prefix.

    Bargmann generating function v0 exp(z R z/2), and its linear
    polynomial for the charged excitation. No projection renormalization
    or phase alignment: missing norm is genuine out-of-shell probability.
    """
    w=result['frequencies'];root=np.sqrt(w)
    inverse=np.linalg.inv(np.diag(w)+result['width'])
    pair=2*root[:,None]*inverse*root[None,:]-np.eye(len(w))
    linear=np.sqrt(2*w)*(inverse@result['polynomial'])
    lookup={n:i for i,n in enumerate(states)};out=np.zeros((len(states),2),complex)
    out[0,0]=result['vacuum_overlap']
    for row,n in enumerate(states[1:],start=1):
        i=next(j for j,x in enumerate(n) if x)
        lower=list(n);lower[i]-=1
        for j in range(len(w)):
            if lower[j]:
                previous=lower.copy();previous[j]-=1
                out[row,0]+=pair[i,j]*np.sqrt(lower[j]/n[i])*out[lookup[tuple(previous)],0]
        for j,occupation in enumerate(n):
            if occupation:
                previous=list(n);previous[j]-=1
                out[row,1]+=linear[j]*np.sqrt(occupation)*out[lookup[tuple(previous)],0]
    return out


def quadratic_family(nodes):
    """Exact I/K charged-spatial tensor family, not a numerical sparsity cut."""
    m=4*nodes;out=[];spatial=[np.eye(4)/2,K_INTERNAL/2]
    for which in ('qq','pp'):
        for a in symmetric_basis(nodes):
            for s in (spatial if which=='qq' else spatial[:1]):
                v=np.zeros((2*m,2*m));block=slice(0,m) if which=='qq' else slice(m,2*m);v[block,block]=np.kron(a,s);out.append(v)
    for i in range(nodes):
        for j in range(nodes):
            a=np.zeros((nodes,nodes));a[i,j]=1.
            for s in spatial:
                v=np.zeros((2*m,2*m));v[m:,:m]=np.kron(a,s)/np.sqrt(2);v[:m,m:]=v[m:,:m].T;out.append(v)
    return np.array(out)


def oscillator_phase(states,frequencies):
    lookup={s:i for i,s in enumerate(states)};size=len(states);annihilators=[]
    for mode,w in enumerate(frequencies):
        rows=[];cols=[];values=[]
        for col,state in enumerate(states):
            if state[mode]:
                lowered=list(state);lowered[mode]-=1;row=lookup.get(tuple(lowered))
                if row is not None: rows.append(row);cols.append(col);values.append(np.sqrt(state[mode]))
        annihilators.append(sparse.csr_matrix((values,(rows,cols)),shape=(size,size),dtype=complex))
    q=[(a+a.T.conj())/np.sqrt(2*w) for a,w in zip(annihilators,frequencies)]
    p=[-1j*np.sqrt(w/2)*(a-a.T.conj()) for a,w in zip(annihilators,frequencies)]
    return tuple(q+p)


def quantize_quadratic(h,operators,size):
    """Project the full Weyl polynomial AFTER products using a shell buffer."""
    result=sparse.csr_matrix((size,size),dtype=complex)
    for i,x in enumerate(operators):
        if h[i,i]!=0: result+=.5*h[i,i]*(x@x)[:size,:size]
        for j in range(i):
            if h[i,j]!=0: result+=.5*h[i,j]*(x@operators[j]+operators[j]@x)[:size,:size]
    return result.tocsr()


class FockSourceControl:
    """Complete-shell quantization, physical quadratic H, not an HO potential.

    Source pulses can change excitation number. All charged/momentum modes
    and field phases are retained. Source-free normal modes set a basis.
    """
    def __init__(self,system,shell):
        if not isinstance(shell,(int,np.integer)) or shell<1:
            raise ValueError('A complete integer shell including one-particle modes is required.')
        self.system=system;self.frequencies,u=system.normal_basis();m=system.dimension
        self.transform=np.block([[u,u*0],[u*0,u]])
        self.states=oscillator_states(shell,m);size=len(self.states);self.size=size
        extended=oscillator_states(shell+2,m);operators=oscillator_phase(extended,self.frequencies)
        self.family=quadratic_family(len(system.mass_squared))
        self.operators=tuple(quantize_quadratic(h,operators,size) for h in self.family)
        self.phase_operators=tuple(x[:size,:size] for x in operators)
        self.extended_operators=operators
        self.initial=np.zeros((size,2),complex);self.initial[0,0]=1.
        excitation=charged_excitation(system)
        for i,state in enumerate(self.states):
            if sum(state)==1: self.initial[i,1]=excitation[state.index(1)]
        self.total_excitation=np.array([sum(s) for s in self.states])
        self.charge_phase=np.zeros((2*m,2*m));self.momentum_phase=np.zeros_like(self.charge_phase)
        charge=np.kron(np.eye(len(system.mass_squared)),np.kron(T_CHARGE,np.eye(2)))
        space=-system.momentum*np.kron(np.eye(len(system.mass_squared)),np.kron(np.eye(2),J_SPACE))
        for target,value in [(self.charge_phase,charge),(self.momentum_phase,space)]: target[m:,:m]=value;target[:m,m:]=value.T
        self.charge=quantize_quadratic(self.charge_phase,operators,size)
        self.momentum=quantize_quadratic(self.momentum_phase,operators,size)

    def coefficients(self,a,adot):
        h=self.transform.T@self.system.hessian(a,adot)@self.transform
        values=np.einsum('aij,ij->a',self.family,h)
        if np.linalg.norm(np.einsum('a,aij->ij',values,self.family)-h)>2e-12:
            raise ValueError('Generator leaves the exact declared tensor family.')
        return values

    def matrix(self,a,adot):
        return sum((v*op for v,op in zip(self.coefficients(a,adot),self.operators)),sparse.csr_matrix((self.size,self.size),dtype=complex))

    def evolve(self,duration,amplitude,*,rtol=2e-10,atol=2e-12):
        def rhs(t,flat):
            a,adot,_=pulse(t,duration,amplitude);state=flat.reshape(self.size,2)
            return (-1j*sum(v*(op@state) for v,op in zip(self.coefficients(a,adot),self.operators))).ravel()
        result=solve_ivp(rhs,(0,duration),self.initial.ravel(),method='DOP853',rtol=rtol,atol=atol)
        if not result.success: raise ValueError(result.message)
        return result.y[:,-1].reshape(self.size,2)

    def covariance(self,state):
        # Extend the vector before products; Q acting on the highest shell
        # must not erase its norm in an expectation of Q^2.
        extended=np.zeros(self.extended_operators[0].shape[0],complex);extended[:self.size]=state
        vectors=np.column_stack([op@extended for op in self.extended_operators])
        return (vectors.conj().T@vectors).real

    def wavefunction(self,state,coordinate):
        q=np.asarray(coordinate,float);w=self.frequencies;level=max(self.total_excitation)
        hermite=np.array([eval_hermite(n,np.sqrt(w)*q)*np.exp(-.5*(n*np.log(2)+gammaln(n+1))) for n in range(level+1)])
        weights=np.prod(hermite[np.array(self.states),np.arange(len(w))],axis=1)
        vacuum=np.prod((w/np.pi)**.25)*np.exp(-.5*np.sum(w*q*q))
        return vacuum*(weights@state)


def lf_weak_source(h,d,b=None,c=None):
    """K(omega)=h-omega-epsilon(d omega^2+b omega+c), canonical G=1.

    One stationary weak insertion, h,d,b,c Hermitian. Returns the Hermitian
    generator shift, left endpoint shift and direct contact. It is not an
    exact finite-source theory, zero-plus prescription or arbitrary q- map.
    """
    h,d=np.asarray(h,complex),np.asarray(d,complex);zero=np.zeros_like(h)
    b,c=(zero if x is None else np.asarray(x,complex) for x in (b,c))
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or not len(h) or any(x.shape!=h.shape or not np.isfinite(x).all() or not np.allclose(x,x.conj().T,rtol=0,atol=1e-13) for x in (h,d,b,c)):
        raise ValueError('Common finite Hermitian kernel matrices required.')
    return -(h@d@h+(h@b+b@h)/2+c),-(d@h+b/2),d


def lf_response_parts(h,d,omega,b=None,c=None):
    """Independent decomposition of the first weak-source Green coefficient."""
    h=np.asarray(h,complex);d=np.asarray(d,complex);zero=np.zeros_like(h)
    b,c=(zero if x is None else np.asarray(x,complex) for x in (b,c))
    shift,endpoint,contact=lf_weak_source(h,d,b,c);r=np.linalg.inv(h-omega*np.eye(len(h)))
    exact=r@(d*omega**2+b*omega+c)@r
    pieces=(-r@shift@r,endpoint@r+r@endpoint.conj().T,contact)
    return exact,pieces

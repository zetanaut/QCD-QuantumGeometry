"""Eight-component finite-cell quantum reference, NOT a nucleon/LF model.

All retained collective coordinates are quantized once. A complete total-
excitation basis preserves internal rotations; its oscillator frequencies
are numerical basis parameters, not a physical confinement interaction.
Exact finite-matrix elimination is a representation check, not H2 or a
Wilsonian QCD matching prescription. Spatial nonzero modes are not supplied.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, eigvalsh, solve
from scipy.optimize import brentq
from scipy.sparse.linalg import eigsh, LinearOperator, cg, gmres


@lru_cache(maxsize=24)
def oscillator_states(shell, modes=8):
    if not isinstance(shell,int) or shell<0 or modes<1:
        raise ValueError('Nonnegative integer shell and positive mode count required.')
    def partition(total, n):
        if n==1:
            yield (total,)
        else:
            for first in range(total+1):
                for rest in partition(total-first,n-1): yield (first,)+rest
    return tuple(s for total in range(shell+1) for s in partition(total,modes))


def coordinate_operators(states, volume, frequencies):
    lookup={s:i for i,s in enumerate(states)};size=len(states);result=[]
    for mode,omega in enumerate(frequencies):
        rows=[];cols=[];values=[]
        for j,s in enumerate(states):
            raised=list(s);raised[mode]+=1;i=lookup.get(tuple(raised))
            if i is not None:
                rows.extend([i,j]);cols.extend([j,i])
                value=np.sqrt((s[mode]+1)/(2*volume*omega));values.extend([value,value])
        result.append(sparse.csr_matrix((values,(rows,cols)),shape=(size,size)))
    return tuple(result)


@dataclass
class QuantumCell:
    states: tuple
    hamiltonian: sparse.csr_matrix
    fields: tuple
    volume: float
    frequencies: np.ndarray

    def linear_source(self, weights):
        """V sum w_iA phi_iA. For dimensionless g, dH/dB is GeV^-2.

        For c in GeV^2 this instead represents a mass-spurion derivative
        (dimensionless operator); the caller owns its source units/sign.
        """
        weights=np.asarray(weights,float).reshape(8)
        return sum((self.volume*w*x for w,x in zip(weights,self.fields)),
                   sparse.csr_matrix(self.hamiltonian.shape))


def polynomial_cell(quadratic,quartic,coupling,forcing,*,volume,shell,frequency=.35):
    """H=sum p_a^2/(2V)+V[sum(mu_i^2 r_i/2+lambda_i r_i^2/4)
        +kappa/2*(phi1-phi2)^2 - forcing.phi]. r_i=phi_i.phi_i.

    V is GeV^-3; phi GeV; frequencies GeV; H GeV. Complete polynomials
    are projected AFTER multiplication, using two extra intermediate shells.
    Dropping these paths would change the physical operator at the cutoff.
    """
    mu,lam=np.asarray(quadratic,float),np.asarray(quartic,float)
    force=np.asarray(forcing,float)
    freq=np.repeat(np.broadcast_to(np.asarray(frequency,float),(2,)),4)
    if (mu.shape!=(2,) or lam.shape!=(2,) or force.shape!=(2,4)
        or not all(np.isfinite(z).all() for z in [mu,lam,force,freq])
        or not np.isfinite(volume) or volume<=0 or np.any(freq<=0)
        or not np.isfinite(coupling) or np.any(lam<0)):
        raise ValueError('Finite coefficients, positive volume/frequencies and nonnegative quartics required.')
    states=oscillator_states(shell);extended=oscillator_states(shell+2);size=len(states)
    x=coordinate_operators(extended,volume,freq)
    r=[sum((z@z for z in x[4*i:4*i+4]),sparse.csr_matrix(x[0].shape)) for i in range(2)]
    number=np.asarray(states)
    h=sparse.diags((freq*(number+.5)).sum(1),format='csr')
    for i in range(2):
        h+=.5*volume*(mu[i]+coupling-freq[4*i]**2)*r[i][:size,:size]
        h+=.25*volume*lam[i]*(r[i]@r[i])[:size,:size]
    for a in range(4): h-=volume*coupling*(x[a]@x[a+4])[:size,:size]
    for weight,z in zip(force.ravel(),x): h-=volume*weight*z[:size,:size]
    return QuantumCell(states,h.tocsr(),tuple(z[:size,:size].tocsr() for z in x),volume,freq)


def chiral_cell(base,current_mass,*,volume,shell,frequency=.35,pseudoscalar=None):
    """Actual previous V, omitting only its field-independent constant.

    Old tree-calibrated parameters are reused as a diagnostic, not claimed
    re-matched after quantization or at this finite volume/cutoff.
    """
    force=np.zeros((2,4));force[:,0]=current_mass*base.scalar_source
    if pseudoscalar is not None:
        force[:,1:]=base.scalar_source[:,None]*np.asarray(pseudoscalar)[None,:]
    return polynomial_cell(-base.quartic*base.radial_squared,base.quartic,
        base.coupling,force,volume=volume,shell=shell,frequency=frequency)


def ground_state(hamiltonian):
    h=sparse.csr_matrix(hamiltonian);size=h.shape[0]
    if size<=100:
        values,vectors=eigh(h.toarray(),subset_by_index=[0,0]);e=values[0];psi=vectors[:,0]
    else:
        start=np.random.default_rng(181).normal(size=size)
        values,vectors=eigsh(h,k=1,which='SA',tol=2e-12,v0=start,maxiter=20000)
        e=values[0];psi=vectors[:,0]
    if psi[0]<0: psi=-psi
    residual=float(np.linalg.norm(h@psi-e*psi))
    if residual>1e-9: raise ValueError('Unresolved finite-cell ground-state equation.')
    return dict(energy=float(e),state=psi,residual=residual)


def connected_response(cell,ground,operators,euclidean_frequency=0.):
    """Connected quantum susceptibility, -d^2 E/dB_A dB_B / V at nu=0.

    chi_AB(nu)=2/V Re <v_A|(H-E0+i nu)^-1|v_B>,
    v_A=(O_A-<O_A>)|0>. Ground pole is removed, not fitted/subtracted data.
    The rank-one lift has eta=1 GeV, inert on the centered source vectors.
    All operators in this bounded implementation are real symmetric.
    """
    nu=float(euclidean_frequency)
    if not np.isfinite(nu) or nu<0: raise ValueError('Finite nonnegative Euclidean frequency required.')
    h=cell.hamiltonian;e=ground['energy'];psi=ground['state'];size=len(psi)
    vectors=[];means=[]
    for op in operators:
        value=op@psi;mean=float(psi@value);means.append(mean);vectors.append(value-mean*psi)
    dtype=complex if nu else float
    def matvec(v): return h@v-e*v+psi*(psi@v)+(1j*nu*v if nu else 0)
    operator=LinearOperator((size,size),matvec=matvec,dtype=dtype)
    solutions=[];errors=[]
    for rhs in vectors:
        if np.linalg.norm(rhs)<1e-15: solution=np.zeros(size,dtype=dtype);info=0
        elif nu: solution,info=gmres(operator,rhs.astype(complex),rtol=2e-11,atol=1e-12,restart=100,maxiter=500)
        else: solution,info=cg(operator,rhs,rtol=2e-11,atol=1e-12,maxiter=20000)
        error=float(np.linalg.norm(matvec(solution)-rhs));errors.append(error)
        if info or error>1e-8*max(1,np.linalg.norm(rhs)): raise ValueError('Quantum response solve not converged.')
        solutions.append(solution)
    chi=2/cell.volume*np.real(np.asarray(vectors)@np.asarray(solutions).T)
    return dict(susceptibility=chi,operator_means=np.array(means),linear_solve_errors=errors)


def rotation_generator(states,first,second):
    """Hermitian internal L_ab=-i(a_a^dagger a_b-a_b^dagger a_a)."""
    lookup={s:i for i,s in enumerate(states)};rows=[];cols=[];values=[]
    for j,s in enumerate(states):
        if s[second]:
            moved=list(s);moved[first]+=1;moved[second]-=1;i=lookup[tuple(moved)]
            value=-1j*np.sqrt((s[first]+1)*s[second])
            rows.extend([i,j]);cols.extend([j,i]);values.extend([value,value.conjugate()])
    return sparse.csr_matrix((values,(rows,cols)),shape=(len(states),len(states)))


def eliminate_exact(hamiltonian,retained,energy):
    """Exact finite-matrix projection, no coupling/order expansion or H2 input.

    W=P+Q(E-H_QQ)^-1 H_QP; H_eff=H_PP+H_PQ(E-H_QQ)^-1H_QP.
    N=W^dagger W=I-partial_E H_eff. External sources differentiate every
    block and the same energy-dependent map, not an independent current.
    """
    h=np.asarray(hamiltonian,float);p=np.asarray(retained,int)
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or not np.allclose(h,h.T):
        raise ValueError('Real symmetric finite Hamiltonian required.')
    if len(np.unique(p))!=len(p) or not 0<len(p)<len(h) or np.any(p<0) or np.any(p>=len(h)):
        raise ValueError('Distinct proper retained indices required.')
    q=np.setdiff1d(np.arange(len(h)),p);hqp=h[np.ix_(q,p)]
    response=solve(energy*np.eye(len(q))-h[np.ix_(q,q)],hqp,assume_a='sym')
    wave=np.zeros((len(h),len(p)));wave[p]=np.eye(len(p));wave[q]=response
    return dict(hamiltonian=h[np.ix_(p,p)]+hqp.T@response,
                embedding=wave,norm=wave.T@wave,retained=p,eliminated=q)


def eliminated_ground_energy(hamiltonian,retained):
    """Solve lowest nonlinear eigenvalue using an independent bracket.

    No full-space ground eigenvalue is used. Requires nonzero overlap with
    retained space and ground energy strictly below the Q-block spectrum.
    """
    h=np.asarray(hamiltonian);p=np.asarray(retained);q=np.setdiff1d(np.arange(len(h)),p)
    lo=float(np.min(np.diag(h)-np.sum(abs(h),axis=1)+abs(np.diag(h))))-.1
    hi=min(float(eigvalsh(h[np.ix_(p,p)],subset_by_index=[0,0])[0]),
           float(eigvalsh(h[np.ix_(q,q)],subset_by_index=[0,0])[0])-1e-8)
    def equation(e): return float(eigvalsh(eliminate_exact(h,p,e)['hamiltonian'],subset_by_index=[0,0])[0]-e)
    return float(brentq(equation,lo,hi,xtol=1e-13,rtol=1e-13))


def eliminated_response(cell,ground,operators,retained,euclidean_frequency=0.):
    """Exact response Schur complement, including direct Q-source response.

    The eta=1 GeV ground-pole lift equals the full connected resolvent on each centered
    source vector. For real sources and complex-symmetric Euclidean A, the
    two transported endpoints are transposes, not conjugate transposes.
    This dense control is intentionally restricted to small diagnostic cells.
    """
    h=cell.hamiltonian.toarray();psi=ground['state'];nu=euclidean_frequency
    p=np.asarray(retained,int);q=np.setdiff1d(np.arange(len(h)),p)
    a=h-ground['energy']*np.eye(len(h))+np.outer(psi,psi)+1j*nu*np.eye(len(h))
    v=np.stack([op@psi-float(psi@op@psi)*psi for op in operators],axis=1)
    aqq=a[np.ix_(q,q)];apq=a[np.ix_(p,q)];aqp=a[np.ix_(q,p)]
    solve_q=solve(aqq,np.column_stack([aqp,v[q]]),assume_a='sym')
    rqp,rqv=solve_q[:,:len(p)],solve_q[:,len(p):]
    schur=a[np.ix_(p,p)]-apq@rqp
    endpoint=v[p]-apq@rqv
    direct=v[q].T@rqv;mediated=endpoint.T@solve(schur,endpoint,assume_a='sym')
    return dict(susceptibility=2/cell.volume*(direct+mediated).real,
                direct=2/cell.volume*direct.real,mediated=2/cell.volume*mediated.real)

"""Source-preserving finite-momentum transport and mode accessibility.

These are off-shell matching kernels in the real retained collective
reference, not a production Hamiltonian, quantum hadron or data likelihood.
Source-free reachability is NOT permission to discard source-visible modes.
"""
from itertools import combinations
import numpy as np
import torch

from .joint_source_matching import JointSourceAction,action_contact,stress_contact
from .transverse_source_matching import dot
from .collective_quark_matching import bilinear_matrices


def symmetric_basis(n):
    """Orthonormal real symmetric matrices, Frobenius metric (not a prior)."""
    if not isinstance(n,int) or n<1: raise ValueError('Positive integer node count required.')
    basis=[]
    for i in range(n):
        for j in range(i,n):
            a=np.zeros((n,n));a[i,j]=a[j,i]=1. if i==j else 1/np.sqrt(2)
            basis.append(a)
    return np.array(basis)


def _carrier(h,g):
    h=np.asarray(h,float);g=np.asarray(g,float)
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or not len(h) or not np.isfinite(h).all() or not np.allclose(h,h.T,rtol=0,atol=1e-13):
        raise ValueError('Finite real symmetric inverse on the common node space required.')
    if np.linalg.eigvalsh(h).min()<=0: raise ValueError('Positive reference inverse required.')
    if g.ndim==1: g=g[:,None]
    if g.ndim!=2 or g.shape[0]!=len(h) or not np.isfinite(g).all(): raise ValueError('Finite common-node endpoints required.')
    return h,g


def accessibility(h,endpoints,*,source_matrices=(),rtol=1e-11):
    """Smallest numerically resolved invariant space of H and supplied sources.

    Matrix/endpoint normalization removes arbitrary units from the span
    iteration. rtol is a numerical rank threshold, not a physical zero.
    Omitting sources only answers the source-FREE question. Unknown source
    matrices cannot be replaced by zeros to claim an exact compression.
    This finite linear family does not cover omitted nonlinear vertices,
    Fock sectors or the full QCD/axial source algebra.
    """
    h,g=_carrier(h,endpoints)
    if not 0<rtol<1: raise ValueError('Explicit positive numerical rank threshold required.')
    operators=[]
    for m in (h,*source_matrices):
        m=np.asarray(m,float)
        if m.shape!=h.shape or not np.isfinite(m).all() or not np.allclose(m,m.T,rtol=0,atol=1e-13):
            raise ValueError('Finite symmetric source matrices on the common carrier required.')
        norm=np.linalg.norm(m,2)
        if norm: operators.append(m/norm)
    nonzero=np.linalg.norm(g,axis=0)>0
    seed=g[:,nonzero]/np.linalg.norm(g[:,nonzero],axis=0)
    def span(a):
        u,s,_=np.linalg.svd(a,full_matrices=False)
        return u[:,s>rtol*s[0]] if len(s) and s[0]>0 else np.empty((len(h),0))
    u=span(seed)
    for _ in range(len(h)):
        v=span(np.column_stack([u,*[a@u for a in operators]]))
        if v.shape[1]==u.shape[1]: u=v;break
        u=v
    return u


def insertion_design(h,endpoints,virtualities,*,diagonal=False):
    """Rows of G_a^T R(x) E R(y) G_b in a real symmetric E basis.

    Multiple endpoints and independent routes are kept. diagonal=True
    restricts virtualities, not endpoint labels. Singular values have the
    declared kernel units and are not data information without a covariance.
    """
    h,g=_carrier(h,endpoints);x=np.asarray(virtualities,float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or np.any(x<0): raise ValueError('Nonnegative spacelike squared virtualities required.')
    v=[np.linalg.solve(h+xx*np.eye(len(h)),g) for xx in x];basis=symmetric_basis(len(h))
    rows=[]
    for i,a in enumerate(v):
        for j,b in enumerate(v):
            if diagonal and i!=j: continue
            rows.extend(np.einsum('ia,kij,jb->abk',a,basis,b).reshape(-1,len(basis)))
    return np.array(rows)


def canonical_coordinates(h,endpoints,metric,*,source_matrices=()):
    """Transport a constant positive kinetic metric with ALL source partners.

    phi=Z^-1/2 chi: Hc=S^T H S, Gc=S^T G, Mc=S^T M S.
    Not a treatment of nonlocal/energy-dependent quantum pole normalization.
    """
    h,g=_carrier(h,endpoints);z=np.asarray(metric,float)
    if z.shape!=h.shape or not np.isfinite(z).all() or not np.allclose(z,z.T,rtol=0,atol=1e-13): raise ValueError('Common positive kinetic metric required.')
    values,u=np.linalg.eigh(z)
    if values.min()<=0: raise ValueError('Positive kinetic metric required.')
    s=(u/np.sqrt(values))@u.T;transformed=[]
    for m in source_matrices:
        m=np.asarray(m,float)
        if m.shape!=h.shape or not np.isfinite(m).all() or not np.allclose(m,m.T,rtol=0,atol=1e-13): raise ValueError('Common symmetric source partners required.')
        transformed.append(s.T@m@s)
    return s.T@h@s,s.T@g,tuple(transformed),s


def diagonal_cycle_null(eigenvalues,couplings,cycle,*,reference_mass_squared=1.):
    """Exact single-endpoint equal-virtuality null from a pole-graph cycle.

    Returns a dimensionless source SHAPE. For oriented edge i->j choose
    2*g_i*g_j*shape_ij=(h_j-h_i)/reference_mass_squared. The reference
    is in the same GeV^2 units as the poles, not a chosen matching scale;
    a physical E still needs its own GeV^-4 coefficient.
    Products then telescope into simple-pole differences. This is an
    algebraic response-graph cycle, NOT physical topological order.
    """
    h=np.asarray(eigenvalues,float);g=np.asarray(couplings,float);cycle=tuple(cycle)
    if h.ndim!=1 or g.shape!=h.shape or not np.isfinite(h).all() or not np.isfinite(g).all() or not np.isfinite(reference_mass_squared) or reference_mass_squared<=0 or len(cycle)<3 or len(set(cycle))!=len(cycle) or min(cycle)<0 or max(cycle)>=len(h) or np.any(g==0) or len(np.unique(h))!=len(h):
        raise ValueError('Distinct poles, nonzero couplings and a simple cycle required.')
    e=np.zeros((len(h),len(h)))
    for i,j in zip(cycle,cycle[1:]+cycle[:1]): e[i,j]=e[j,i]=(h[j]-h[i])/(2*g[i]*g[j]*reference_mass_squared)
    return e


def one_source(h_size,c,p1,p0,epsilon,*,metric=None):
    """Minimal vertex plus the same C action; C may carry Torch gradients."""
    from .transverse_source_matching import transverse_vertex
    z=torch.eye(h_size,dtype=torch.float64) if metric is None else torch.as_tensor(metric,dtype=torch.float64)
    return dot(p1+p0,epsilon)*z+dot(transverse_vertex(p1,p0,1.),epsilon)*c


def contact_coefficients(p1,p0,q1,q2,e1,e2):
    """C,D,E coefficients from actual fourth action derivatives, not fitting.

    Unit one-node extraction works because the density is bilinear in nodes.
    The coefficients can subsequently multiply trainable full matrices.
    """
    values=[]
    for i in range(3):
        matrices=[np.array([[float(i==j)]]) for j in range(3)]
        value=action_contact(JointSourceAction(*matrices),p1,p0,q1,q2,e1,e2,[1.],[1.])
        if abs(value.imag)>1e-12: raise ValueError('Real spacelike reference contact required.')
        values.append(value.real)
    return np.array(values)


def source_jet(h,c,d,e,p0,q1,q2,eps1,eps2,*,metric=None):
    """Four-route K, K_a,K_b,K_ab, arbitrary nodes, differentiable matrices.

    Both +/- source Fourier routes are included. No determinant/quantum
    contribution is inferred from this finite tree inverse. Source amplitudes
    have GeV units; the jet is through a*b only, not a finite-source action.
    """
    h,c,d,e=[torch.as_tensor(x,dtype=torch.float64) for x in (h,c,d,e)]
    n=h.shape[0]
    for a in (h,c,d,e):
        if a.shape!=(n,n) or not torch.isfinite(a).all() or not torch.allclose(a,a.T,rtol=0,atol=1e-13): raise ValueError('Common finite symmetric matrices required.')
    if torch.linalg.eigvalsh(h).min()<=0: raise ValueError('Positive reference Hessian required.')
    z=torch.eye(n,dtype=torch.float64) if metric is None else torch.as_tensor(metric,dtype=torch.float64)
    if z.shape!=(n,n) or not torch.isfinite(z).all() or not torch.allclose(z,z.T,rtol=0,atol=1e-13) or torch.linalg.eigvalsh(z).min()<=0: raise ValueError('Positive common kinetic metric required.')
    p0,q1,q2,eps1,eps2=[np.asarray(x,float) for x in (p0,q1,q2,eps1,eps2)]
    if any(x.shape!=(4,) or not np.isfinite(x).all() for x in (p0,q1,q2,eps1,eps2)): raise ValueError('Finite four-vectors required.')
    momenta=np.array([p0,p0+q1,p0+q2,p0+q1+q2])
    if any(dot(p,p)>=0 for p in momenta) or any(np.allclose(p,k,rtol=0,atol=1e-12) for i,p in enumerate(momenta) for k in momenta[:i]): raise ValueError('Four distinct spacelike routes required.')
    zero=h*0;blocks=[[],[],[],[]]
    for i,p in enumerate(momenta):
        row=[[],[],[],[]]
        for j,k in enumerate(momenta):
            row[0].append(h-dot(p,p)*z if i==j else zero)
            for target,transfer,eps in [(1,q1,eps1),(2,q2,eps2)]:
                match=any(np.allclose(p-k,s*transfer,rtol=0,atol=1e-13) for s in [-1,1]) if i!=j else False
                row[target].append(-one_source(n,c,p,k,eps,metric=z) if match else zero)
            contact=zero
            if i!=j:
                for sa in [-1,1]:
                    for sb in [-1,1]:
                        if np.allclose(p-k,sa*q1+sb*q2,rtol=0,atol=1e-13):
                            coefficients=contact_coefficients(p,k,sa*q1,sb*q2,eps1,eps2)
                            contact=contact-2*dot(eps1,eps2)*z-sum(a*b for a,b in zip(coefficients,(c,d,e)))
            row[3].append(contact)
        for target in range(4): blocks[target].append(torch.cat(row[target],dim=1))
    return momenta,tuple(torch.cat(a,dim=0) for a in blocks)


def response(jet,endpoints,sources):
    """Same normalized Gaussian source kernel, not a state expectation."""
    k,ka,kb,kab=jet;y=torch.as_tensor(endpoints,dtype=torch.float64);a,b=sources
    return y.T@torch.linalg.solve(k+a*ka+b*kb+a*b*kab,y)


def source_derivatives(jet,endpoints):
    k,ka,kb,kab=jet;y=torch.as_tensor(endpoints,dtype=torch.float64);r=torch.linalg.inv(k)
    return tuple(y.T@v@y for v in (r,-r@ka@r,-r@kb@r,r@ka@r@kb@r+r@kb@r@ka@r-r@kab@r))


def open_bilinear_tensor(channel_weights):
    """All 24^4 Dirac/flavor/color endpoints of the supplied 4x4 S/P block.

    Off-diagonal charged-channel phases are retained. Canonical QCD color-
    vector tensors remain separate; no S/P projection replaces them here.
    No identical-fermion antisymmetrization or external radiative legs implied.
    """
    w=np.asarray(channel_weights,complex)
    if w.shape!=(4,4) or not np.isfinite(w).all(): raise ValueError('Finite open S/P channel weights required.')
    b=bilinear_matrices()
    return np.einsum('AB,Aij,Bkl->ijkl',w,b,b)


def two_E_return(h,e,g,p0,photons,polarizations):
    """Four-photon kernel coefficient quadratic in E: six ordered pairings.

    E has no zero/one-photon vertex. Higher E contacts are linear in E;
    therefore this E^2 coefficient has exactly two two-photon insertions.
    Not the full four-photon kernel, quantum loop or external-quark amplitude.
    """
    h,g=_carrier(h,g);e=np.asarray(e,float);p0=np.asarray(p0,float)
    q=np.asarray(photons,float);eps=np.asarray(polarizations,float)
    if e.shape!=h.shape or not np.isfinite(e).all() or not np.allclose(e,e.T) or p0.shape!=(4,) or q.shape!=(4,4) or eps.shape!=(4,4) or not np.isfinite(p0).all() or not np.isfinite(q).all() or not np.isfinite(eps).all(): raise ValueError('Four finite photons and common symmetric E required.')
    def r(p):
        if dot(p,p)>=0: raise ValueError('Spacelike internal routes required.')
        return np.linalg.inv(h-dot(p,p)*np.eye(len(h)))
    p1=p0+q.sum(axis=0);left=g.T@r(p1);right=r(p0)@g
    terms=[]
    for pair in combinations(range(4),2):
        rest=tuple(i for i in range(4) if i not in pair);mid=p0+q[list(pair)].sum(axis=0)
        a,b=pair;c,d=rest
        first=stress_contact(mid,p0,q[a],q[b],eps[a],eps[b],1.)
        second=stress_contact(p1,mid,q[c],q[d],eps[c],eps[d],1.)
        terms.append(first*second*(left@e@r(mid)@e@right))
    return np.sum(terms,axis=0),np.array(terms)

"""Resolved collective/quark matching algebra, NOT QCD coefficient matching.

Gaussian response uses K=H-p^2 and spacelike routed momenta. Its inverse is
a real response convention (overall Minkowski propagator i stripped).
Source derivatives are interaction insertions, not full external-quark
radiative amplitudes. Nonlinear elimination below is a homogeneous tree
Legendre/stationary calculation, not a quantum gap or nucleon solution.
"""
import numpy as np
import torch
from scipy.optimize import root

from .chiral_soft_matching import G5
from .transverse_source_matching import action_contact, transverse_vertex, dot


def bilinear_matrices():
    """B=(qbar q,qbar i gamma5 tau^a q), Dirac x flavor x color, 24x24.

    Even Grassmann bilinears retain their open indices. No projection on
    baryons, identical-fermion antisymmetrization or Fierz channel deletion.
    """
    tau = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
    return np.array([np.eye(24), *[np.kron(np.kron(1j*G5,t),np.eye(3)) for t in tau]])


def yukawa_map(couplings):
    """Eight collective components (multiplet, O4) to four bilinears."""
    g=np.asarray(couplings,float)
    if g.shape!=(2,) or not np.isfinite(g).all():
        raise ValueError('Two explicit finite Yukawa coefficients required.')
    return np.kron(g[:,None],np.eye(4))


def exchange_weights(base, vacuum, q_squared):
    if not np.isfinite(q_squared) or q_squared<0:
        raise ValueError('Nonnegative finite spacelike Q^2 required.')
    hp,hs=base.matrices(vacuum);g=base.yukawa
    scalar=float(g@np.linalg.solve(hs+q_squared*np.eye(2),g))
    pion=float(g@np.linalg.solve(hp+q_squared*np.eye(2),g))
    return np.array([scalar,pion,pion,pion])


def open_quark_tensor(weights):
    """W_AB B^A_ab B^B_cd for diagonal invariant channels, not an S-matrix.

    The Euclidean eliminated action is -B^T W B/2. Canonical QCD color-vector
    exchange is separate and MUST NOT be replaced by this S/P channel block.
    """
    b=bilinear_matrices()
    return np.einsum('A,Aab,Acd->abcd',np.asarray(weights),b,b)


def one_source_matrix(model, p1, p0, polarization):
    scalar=dot(np.asarray(p1)+p0,polarization)
    correction=dot(transverse_vertex(p1,p0,1.),polarization)
    return scalar*np.eye(2)+correction*model.c


def two_source_matrix(model,p1,p0,q1,q2,eps1,eps2):
    basis=np.eye(2)
    correction=np.array([[action_contact(model,p1,p0,q1,q2,eps1,eps2,basis[i],basis[j]) for j in range(2)] for i in range(2)])
    if np.max(abs(correction.imag))>1e-12:
        raise ValueError('This real spacelike source control received a complex contact.')
    return 2*dot(eps1,eps2)*np.eye(2)+correction.real


def routed_jet(hessian, model, p0, q1, q2, eps1, eps2):
    """Four explicit momentum routes; source Taylor jet through mixed a*b.

    K0=diag(H-p_r^2), K_a=-Gamma_a, K_ab=-S_ab. Both +/- Fourier
    components of each real external source are retained. This is sufficient
    for the specified two-source route, not an all-source finite-grid theory.
    Source amplitudes carry GeV; eps1/2 specify dimensionless directions.
    """
    h=np.asarray(hessian,float);p0,q1,q2=map(np.asarray,(p0,q1,q2))
    if h.shape!=(2,2) or not np.isfinite(h).all() or not np.allclose(h,h.T):
        raise ValueError('Symmetric two-multiplet fluctuation Hessian required.')
    if any(p.shape!=(4,) or not np.isfinite(p).all() for p in [p0,q1,q2,np.asarray(eps1),np.asarray(eps2)]):
        raise ValueError('Finite four-momenta and polarization directions required.')
    momenta=np.array([p0,p0+q1,p0+q2,p0+q1+q2])
    if any(np.allclose(momenta[i],momenta[j],atol=1e-12,rtol=0) for i in range(4) for j in range(i)):
        raise ValueError('Four distinct momentum routes required by this bounded jet.')
    if any(dot(p,p)>=0 for p in momenta):
        raise ValueError('All declared internal routes must be spacelike in this control.')
    k0=np.zeros((8,8));ka=np.zeros_like(k0);kb=np.zeros_like(k0);kab=np.zeros_like(k0)
    for i,p in enumerate(momenta):
        k0[2*i:2*i+2,2*i:2*i+2]=h-dot(p,p)*np.eye(2)
        for j,k in enumerate(momenta):
            if i==j: continue
            row,col=slice(2*i,2*i+2),slice(2*j,2*j+2)
            for transfer,eps,target in [(q1,eps1,ka),(q2,eps2,kb)]:
                if any(np.allclose(p-k,s*transfer,atol=1e-13,rtol=0) for s in [-1,1]):
                    target[row,col]=-one_source_matrix(model,p,k,eps)
            for sa in [-1,1]:
                for sb in [-1,1]:
                    if np.allclose(p-k,sa*q1+sb*q2,atol=1e-13,rtol=0):
                        kab[row,col]-=two_source_matrix(model,p,k,sa*q1,sb*q2,eps1,eps2)
    return momenta,(k0,ka,kb,kab)


def response_derivatives(jet, endpoints):
    """W=Y^T K^-1 Y and first/mixed second source derivatives, fixed Y."""
    k,ka,kb,kab=map(np.asarray,jet);y=np.asarray(endpoints);r=np.linalg.inv(k)
    ra=-r@ka@r;rb=-r@kb@r
    rab=r@ka@r@kb@r+r@kb@r@ka@r-r@kab@r
    return tuple(y.T@z@y for z in [r,ra,rb,rab])


def matched_response(full_jet, full_y, retained_jet, retained_y):
    """Subtract full/retained induced FOUR-QUARK functionals, not Hessians.

    W_res=Y_full^T K_full^-1 Y_full -Y_ret^T K_ret^-1 Y_ret.
    Matrix shape agreement alone does not identify physical retained content.
    """
    return tuple(a-b for a,b in zip(response_derivatives(full_jet,full_y),response_derivatives(retained_jet,retained_y)))


def torch_response(jet, endpoints, sources):
    k,ka,kb,kab=[torch.as_tensor(v,dtype=torch.float64) for v in jet]
    y=torch.as_tensor(endpoints,dtype=torch.float64)
    a,b=sources
    return y.T@torch.linalg.solve(k+a*ka+b*kb+a*b*kab,y)


def potential_derivatives(base, vacuum):
    """Actual shifted quartic potential tensors H,T,U in all eight fields."""
    f=np.zeros((2,4));f[:,0]=vacuum
    h=np.zeros((8,8));t=np.zeros((8,)*3);u=np.zeros((8,)*4);eye=np.eye(4)
    for i in range(2):
        s=slice(4*i,4*i+4);lam=base.quartic[i]
        h[s,s]=lam*((f[i]@f[i]-base.radial_squared[i])*eye+2*np.outer(f[i],f[i]))+base.coupling*eye
        t[s,s,s]=2*lam*(np.einsum('ab,c->abc',eye,f[i])+np.einsum('ac,b->abc',eye,f[i])+np.einsum('bc,a->abc',eye,f[i]))
        u[s,s,s,s]=2*lam*(np.einsum('ab,cd->abcd',eye,eye)+np.einsum('ac,bd->abcd',eye,eye)+np.einsum('ad,bc->abcd',eye,eye))
    h[:4,4:]=h[4:,:4]=-base.coupling*eye
    return h,t,u


def induced_many_quark_tensors(h,t,u,endpoints):
    """Connected 4q/6q/8q tree coefficients in bilinear coordinates.

    W(B)=-1/2 W2 B^2 +1/6 W3 B^3 +1/24 W4 B^4+O(B^5).
    Use the actual mass convention m+g*phi: the Euclidean action has
    +B^T Y^T x, hence the stationary source is J=-Y*B. W3 has a minus
    sign. The separate linear term +(g.v)*B_scalar is not included here.
    W4 includes all three cubic-exchange pairings, not just the quartic.
    """
    r=np.linalg.inv(h);y=r@endpoints
    w2=endpoints.T@y
    w3=-np.einsum('abc,ai,bj,ck->ijk',t,y,y,y)
    w4=np.einsum('abcd,ai,bj,ck,dl->ijkl',u,y,y,y,y)
    pair=np.einsum('abc,bi,cj->aij',t,y,y)
    w4-=np.einsum('aij,ab,bkl->ijkl',pair,r,pair)
    w4-=np.einsum('aik,ab,bjl->ijkl',pair,r,pair)
    w4-=np.einsum('ail,ab,bjk->ijkl',pair,r,pair)
    return w2,w3,w4


def tree_series(tensors,bilinears):
    w2,w3,w4=tensors;b=np.asarray(bilinears)
    return np.array([-.5*b@w2@b,np.einsum('ijk,i,j,k',w3,b,b,b)/6,
                     np.einsum('ijkl,i,j,k,l',w4,b,b,b,b)/24])


def stationary_tree(h,t,u,source):
    """Solve the near-vacuum branch of full shifted homogeneous quartic V-J*x.

    source is GeV^3, displacement GeV, action density GeV^4. This is neither
    a QCD condensate nor a global-vacuum/finite-density calculation. A stable
    Hessian is required for the followed local branch.
    """
    j=np.asarray(source,float)
    def gradient(x):
        return h@x+.5*np.einsum('abc,b,c->a',t,x,x)+np.einsum('abcd,b,c,d->a',u,x,x,x)/6-j
    def hessian(x):
        return h+np.einsum('abc,c->ab',t,x)+.5*np.einsum('abcd,c,d->ab',u,x,x)
    fit=root(gradient,np.linalg.solve(h,j),jac=hessian,method='hybr',options={'xtol':1e-11})
    error=float(np.linalg.norm(gradient(fit.x)))
    curvature_min=float(np.linalg.eigvalsh(hessian(fit.x)).min())
    if error>1e-11 or curvature_min<=0:
        raise ValueError('No verified stable near-vacuum stationary branch.')
    x=fit.x
    value=.5*x@h@x+np.einsum('abc,a,b,c',t,x,x,x)/6+np.einsum('abcd,a,b,c,d',u,x,x,x,x)/24-j@x
    return dict(displacement=x,action_density=float(value),stationarity_error=error,min_curvature=curvature_min)


def nonlinear_bilinear_response(h,t,u,endpoints,bilinears):
    """Untruncated homogeneous tree value, gradient and Hessian in B.

    This retains the implicit collective branch, not a finite 4q/6q/8q
    polynomial. It is a reusable algebra check, not a fermion condensate
    substitution, loop integration or a many-body state expectation value.
    """
    y=np.asarray(endpoints);b=np.asarray(bilinears)
    result=stationary_tree(h,t,u,-y@b);x=result['displacement']
    curvature=h+np.einsum('abc,c->ab',t,x)+.5*np.einsum('abcd,c,d->ab',u,x,x)
    result['collective_curvature']=curvature
    result['bilinear_gradient']=y.T@x
    result['bilinear_hessian']=-y.T@np.linalg.solve(curvature,y)
    return result

"""Momentum-resolved matching and a shared invariant functional, not a model fit.

Bare g2 distinguishable-quark scattering is Minkowski (+---). The generating
functional uses finite Euclidean coordinate/Fourier quadrature and external
SU(3) transport. No QCD loop matching, causal continuation, finite LF
Hamiltonian, confinement or new nucleon/pion solution is supplied.
"""
from dataclasses import dataclass
import numpy as np

from .canonical_lf_fields import GAMMA, lf_spinor, polarization, polarization_cartesian
from .canonical_lf_network import dot, four_vector, momentum
from .chiral_soft_matching import G5
from .modular_gauge_network import generators


def dirac_basis():
    """HS-orthogonal 16 matrices, Tr(B_A^dagger B_B)=4 delta_AB."""
    matrices = [np.eye(4), G5, *GAMMA, *[g@G5 for g in GAMMA]]
    names = ["S", "P", *[f"V{j}" for j in range(4)], *[f"A{j}" for j in range(4)]]
    for mu in range(4):
        for nu in range(mu+1,4):
            matrices.append(.5j*(GAMMA[mu]@GAMMA[nu]-GAMMA[nu]@GAMMA[mu]))
            names.append(f"T{mu}{nu}")
    return tuple(names), np.array(matrices,complex)


def vector_fierz():
    """Exact open-index recoupling, not an additional fermion exchange diagram.

    T_abcd=sum_mu gamma^mu_ab gamma_mu_cd =
    sum_AB C_AB B_A_ad B_B_cb. Exchanging Grassmann fields in an operator
    also has a fermion sign; that is NOT part of this tensor permutation.
    """
    names, b = dirac_basis()
    tensor = sum(sign*np.einsum("ab,cd->abcd",g,g)
                 for sign,g in zip([1,-1,-1,-1],GAMMA))
    coefficients = np.einsum("Aad,Bcb,abcd->AB",b.conj(),b.conj(),tensor)/16
    rebuilt = np.einsum("AB,Aad,Bcb->abcd",coefficients,b,b)
    sp = np.einsum("AB,Aad,Bcb->abcd",coefficients[:2,:2],b[:2],b[:2])
    return dict(names=names,coefficients=coefficients,tensor=tensor,rebuilt=rebuilt,
                scalar_pseudoscalar=sp)


def color_fierz():
    """Open color tensor in the crossed pairing; no color channel discarded."""
    t = generators()
    tensor = np.einsum("aij,akl->ijkl",t,t)
    singlet = 4/9*np.einsum("il,kj->ijkl",np.eye(3),np.eye(3))
    octet = -1/3*np.einsum("ail,akj->ijkl",t,t)
    return tensor,singlet,octet


def flavor_fierz():
    """Flavor-blind direct current, recoupled without dropping the isovector."""
    tau=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
    direct=np.einsum("ij,kl->ijkl",np.eye(2),np.eye(2))
    scalar=.5*np.einsum("il,kj->ijkl",np.eye(2),np.eye(2))
    vector=.5*np.einsum("ail,akj->ijkl",tau,tau)
    return direct,scalar,vector


def two_quark_exchange(outgoing1,outgoing2,incoming1,incoming2,mass1,mass2,gs_squared):
    """Complete bare distinguishable-quark t channel, 36x36 spin/color matrix.

    Strip an overall S-matrix i. Canonical normalization ubar u=2m; no cell
    factors are inserted. g_s^2 J1.J2/t is independently reconstructed from
    two transverse LF gluons AND instantaneous exchange, at signed q+.
    """
    if not np.isfinite(gs_squared) or gs_squared <= 0:
        raise ValueError("Explicit positive finite g_s^2 required.")
    f1,f2,i1,i2 = map(momentum,(outgoing1,outgoing2,incoming1,incoming2))
    p1,p2,k1,k2 = four_vector(f1,mass1),four_vector(f2,mass2),four_vector(i1,mass1),four_vector(i2,mass2)
    if not np.allclose(p1+p2,k1+k2,rtol=0,atol=3e-12):
        raise ValueError("On-shell four-momentum conservation required.")
    q = p1-k1
    t,qp = dot(q,q),(q[0]+q[3])/np.sqrt(2)
    if t >= -1e-12 or abs(qp)<1e-12:
        raise ValueError("Spacelike nonzero transfer and nonzero signed q+ required.")
    def current(f,i,m):
        out = np.column_stack([lf_spinor(*f,m,h) for h in [-1,1]])
        inc = np.column_stack([lf_spinor(*i,m,h) for h in [-1,1]])
        return np.array([out.conj().T@GAMMA[0]@g@inc for g in GAMMA])
    j1,j2 = current(f1,i1,mass1),current(f2,i2,mass2)
    cov = sum(sign*np.einsum("ab,cd->acbd",a,b)
              for sign,a,b in zip([1,-1,-1,-1],j1,j2))/t
    prop = np.zeros_like(cov)
    mode = np.sign(qp)*np.array([qp,q[1],q[2]])
    for h in [-1,1]:
        eps = polarization_cartesian(polarization(*mode,h))
        left = eps[0]*j1[0]-sum(eps[n]*j1[n] for n in [1,2,3])
        right = eps[0].conjugate()*j2[0]-sum(eps[n].conjugate()*j2[n] for n in [1,2,3])
        prop -= np.einsum("ab,cd->acbd",left,right)/t
    instant = -np.einsum("ab,cd->acbd",(j1[0]+j1[3])/np.sqrt(2),(j2[0]+j2[3])/np.sqrt(2))/qp**2
    # Combine the common 1/(q+)^2 terms algebraically before evaluation.
    # This still uses the LF transverse/plus currents and signed transfer;
    # keep the raw polarization + instantaneous sum as a separate control.
    jm1,jm2 = (j1[0]+j1[3])/np.sqrt(2),(j2[0]+j2[3])/np.sqrt(2)
    transverse1=q[1]*j1[1]+q[2]*j1[2]
    transverse2=q[1]*j2[1]+q[2]*j2[2]
    qm=(q[0]-q[3])/np.sqrt(2)
    combined=(np.einsum("ab,cd->acbd",jm1,transverse2)+np.einsum("ab,cd->acbd",transverse1,jm2)
              -2*qm*np.einsum("ab,cd->acbd",jm1,jm2))/(qp*t)
    combined-=sum(np.einsum("ab,cd->acbd",j1[n],j2[n]) for n in [1,2])/t
    color = np.einsum("aij,akl->ikjl",generators(),generators())
    def assemble(spin):
        return gs_squared*np.einsum("abce,ijkl->aibjckel",spin,color).reshape(36,36)
    divergence = max(np.linalg.norm(q[0]*j[0]-sum(q[n]*j[n] for n in [1,2,3])) for j in [j1,j2])
    return dict(covariant=assemble(cov),propagating=assemble(prop),
                instantaneous=assemble(instant),light_front=assemble(prop+instant),
                light_front_combined=assemble(combined),
                virtuality=-t,plus_transfer=qp,current_divergence=float(divergence))


@dataclass(frozen=True)
class SpacelikeKernel:
    """One scalar momentum coefficient multiplying the full color-vector tensor.

    d(Q^2)=g_s^2/Q^2 + sum_n theta_n mu^-2 x^n/(1+x)^(n+2).
    The displayed Minkowski amplitude uses -d(Q^2) J1.J2 times color.
    This finite basis preserves the leading large-Q^2 term. Its scale,
    coefficients and domain are diagnostic inputs, NOT fitted QCD parameters.
    No timelike continuation of its rational basis is authorized here.
    """
    gs_squared: float
    scale: float
    coefficients: tuple
    domain: tuple

    def __post_init__(self):
        if len(self.domain)!=2 or not self.coefficients:
            raise ValueError("Two-endpoint spacelike domain and explicit coefficients required.")
        if not np.isfinite([self.gs_squared,self.scale,*self.coefficients,*self.domain]).all():
            raise ValueError("Finite coefficients and scales required.")
        if self.gs_squared<=0 or self.scale<=0 or not 0<self.domain[0]<self.domain[1]:
            raise ValueError("Positive g_s^2, GeV scale, and Q^2 domain required.")

    def _q(self,q2):
        q = np.asarray(q2,float)
        if not np.isfinite(q).all() or np.any(q<self.domain[0]) or np.any(q>self.domain[1]):
            raise ValueError("Q^2 is outside the declared spacelike matching domain.")
        return q

    def features(self,q2):
        x = self._q(q2)/self.scale**2
        return np.stack([x**n/(1+x)**(n+2)/self.scale**2 for n in range(len(self.coefficients))],axis=-1)

    def pieces(self,q2,retained_fraction):
        q = self._q(q2)
        r = np.asarray(retained_fraction,float)
        if not np.isfinite(r).all() or np.any(r<0) or np.any(r>1):
            raise ValueError("Explicit retained fraction in [0,1] required.")
        canonical = self.gs_squared/q
        soft = self.features(q) @ np.asarray(self.coefficients)
        retained = r*canonical
        return dict(canonical=canonical,retained=retained,
                    tree_remainder=canonical-retained,diagnostic_soft=soft,
                    residual=canonical-retained+soft,total=canonical+soft)

    def fourier(self,points,momenta,weights,retained_fraction):
        """Finite EUCLIDEAN Fourier quadrature, F_xy=sum w_q d(q^2) e^{iq(x-y)}.

        points in GeV^-1, q in GeV, weights include d^4q/(2pi)^4 in GeV^4.
        All q^2 must be in the declared domain; omitted momenta are not zeros
        of a physical kernel. Conjugate/symmetric modes must give real F_xy.
        """
        x,q,w = np.asarray(points,float),np.asarray(momenta,float),np.asarray(weights,float)
        if x.ndim!=2 or x.shape[1]!=4 or q.ndim!=2 or q.shape[1]!=4 or w.shape!=(len(q),):
            raise ValueError("Euclidean coordinates/momenta and one quadrature weight per mode required.")
        if not all(np.isfinite(v).all() for v in [x,q,w]) or np.any(w<=0):
            raise ValueError("Finite coordinates and positive quadrature weights required.")
        phase = np.exp(1j*np.einsum("qk,xyk->xyq",q,x[:,None,:]-x[None,:,:]))
        result = {}
        for name,value in self.pieces(np.sum(q*q,axis=1),retained_fraction).items():
            f = np.einsum("xyq,q->xy",phase,w*value)
            if np.max(abs(f.imag))>1e-11:
                raise ValueError("Fourier modes do not give a real reciprocal kernel.")
            result[name] = f.real
        return result


def color_map(block,transport):
    """Exact sum_ab Ad(W)_ab t_a S t_b with Dirac x flavor x color ordering."""
    s,w = np.asarray(block,complex).reshape(8,3,8,3),np.asarray(transport,complex)
    trace = np.einsum("ij,ajbi->ab",w.conj().T,s)
    return (.5*np.einsum("ij,ab->aibj",w,trace)-s/6).reshape(24,24)


EUCLIDEAN_GAMMA = np.array([np.kron(np.kron(g,np.eye(2)),np.eye(3))
                           for g in [GAMMA[0],*[1j*x for x in GAMMA[1:]]]])


def vector_map(block,transport):
    c = color_map(block,transport)
    return sum(g@c@g for g in EUCLIDEAN_GAMMA)


def vector_map_direct(block,transport):
    """Independent explicit 4x8 generator contraction, retaining open indices."""
    result = np.zeros((24,24),complex)
    for g4 in [GAMMA[0],*[1j*x for x in GAMMA[1:]]]:
        for t in generators():
            left = np.kron(np.kron(g4,np.eye(2)),transport@t@transport.conj().T)
            right = np.kron(np.kron(g4,np.eye(2)),t)
            result += left@block@right
    return result


class SharedInvariantFunctional:
    """Finite quadrature of a chirally/color covariant generating functional.

    v_xy=F_xy Tr[C_W(S_xy) S_yx] (GeV^8),
    Phi=1/2 sum w_x w_y [v_xy + lambda*v_xy^2/(2*mu^8)].
    S_xy is a coordinate propagator (GeV^3), w_x a d^4x weight (GeV^-4),
    F_xy GeV^2. This nonlinear extension is a test of derivative architecture,
    NOT a demonstrated microscopic 2PI truncation, positivity or causal action.
    """
    def __init__(self,fourier_kernel,weights,transports,scale,nonlinear):
        self.f = np.asarray(fourier_kernel,float)
        self.weights = np.asarray(weights,float)
        self.w = np.asarray(transports,complex)
        self.scale,self.nonlinear = float(scale),float(nonlinear)
        n = len(self.weights)
        if self.f.shape!=(n,n) or self.w.shape!=(n,n,3,3):
            raise ValueError("One reciprocal kernel and transporter per coordinate pair required.")
        if not all(np.isfinite(v).all() for v in [self.f,self.weights,self.w]) or not np.isfinite([scale,nonlinear]).all():
            raise ValueError("Finite functional data required.")
        if scale<=0 or np.any(self.weights<=0) or not np.allclose(self.f,self.f.T,atol=1e-13,rtol=0):
            raise ValueError("Positive scale/weights and symmetric real kernel required.")
        for i in range(n):
            for j in range(n):
                u = self.w[i,j]
                determinant = np.dot(u[0],np.cross(u[1],u[2]))
                if not np.allclose(u.conj().T@u,np.eye(3),atol=2e-12,rtol=0) or abs(determinant-1)>2e-12:
                    raise ValueError("SU(3) transport required.")
                if not np.allclose(u,self.w[j,i].conj().T,atol=2e-12,rtol=0):
                    raise ValueError("Reverse paths must be Hermitian inverses.")

    def _blocks(self,value):
        s = np.asarray(value,complex)
        n = len(self.weights)
        if s.shape!=(n,n,24,24) or not np.isfinite(s).all():
            raise ValueError("Finite coordinate-pair propagator blocks in Dirac x flavor x color order required.")
        return s

    def local_data(self,propagator):
        s = self._blocks(propagator)
        n = len(self.weights)
        mapped = np.array([[vector_map(s[i,j],self.w[i,j]) for j in range(n)] for i in range(n)])
        invariant = self.f*np.einsum("xyab,yxba->xy",mapped,s)
        return mapped,invariant

    def value(self,propagator):
        _,v = self.local_data(propagator)
        u = v+self.nonlinear*v*v/(2*self.scale**8)
        return .5*np.einsum("x,y,xy->",self.weights,self.weights,u)

    def self_energy(self,propagator):
        mapped,v = self.local_data(propagator)
        return ((1+self.nonlinear*v/self.scale**8)*self.f)[...,None,None]*mapped

    def kernel_action(self,propagator,variation,*,include_curvature=True):
        """K[dS]=-dSigma[dS], including the nonlinear learned curvature.

        include_curvature=False is an explicit NEGATIVE CONTROL, not an
        alternative production prescription. Chiral tests alone can miss it.
        """
        s,ds = self._blocks(propagator),self._blocks(variation)
        c,v = self.local_data(s)
        dc,_ = self.local_data(ds)
        result = -((1+self.nonlinear*v/self.scale**8)*self.f)[...,None,None]*dc
        if include_curvature:
            dv = self.f*(np.einsum("xyab,yxba->xy",dc,s)+np.einsum("xyab,yxba->xy",c,ds))
            result -= (self.nonlinear/self.scale**8*self.f*dv)[...,None,None]*c
        return result

    def pairing(self,left,right):
        """Quadrature trace sum_xy w_x w_y Tr[left_xy right_yx]."""
        a,b = self._blocks(left),self._blocks(right)
        return np.einsum("x,y,xyab,yxba->",self.weights,self.weights,a,b)

    def path_source(self,propagator,transport_tangent):
        """One external color-path source derivative of Phi and Sigma.

        The tangent must include reverse occurrences of every affected edge.
        This is NOT a complete electromagnetic bound-state current.
        """
        s = self._blocks(propagator)
        dw = np.asarray(transport_tangent,complex)
        if dw.shape!=self.w.shape or not np.isfinite(dw).all():
            raise ValueError("One finite path tangent per pair required.")
        c,v = self.local_data(s)
        dc = np.zeros_like(c)
        for i in range(len(self.weights)):
            for j in range(len(self.weights)):
                block,w,d = s[i,j].reshape(8,3,8,3),self.w[i,j],dw[i,j]
                trace = np.einsum("ij,ajbi->ab",w.conj().T,block)
                dtrace = np.einsum("ij,ajbi->ab",d.conj().T,block)
                cm = .5*(np.einsum("ij,ab->aibj",d,trace)+np.einsum("ij,ab->aibj",w,dtrace))
                dc[i,j] = sum(g@cm.reshape(24,24)@g for g in EUCLIDEAN_GAMMA)
        dv = self.f*np.einsum("xyab,yxba->xy",dc,s)
        slope = 1+self.nonlinear*v/self.scale**8
        source = .5*np.einsum("x,y,xy,xy->",self.weights,self.weights,slope,dv)
        sigma_source = (slope*self.f)[...,None,None]*dc
        sigma_source += (self.nonlinear/self.scale**8*dv*self.f)[...,None,None]*c
        return source,sigma_source

    def torch_value(self,propagator,*,transports=None,fourier_kernel=None,nonlinear=None):
        """Independent differentiable implementation for actual source/Hessian checks.

        Complex analytic polynomial; test both real and imaginary directional
        derivatives. Physical source/propagator restrictions are not inferred
        from arbitrary finite control matrices.
        """
        import torch
        s = propagator
        w = torch.as_tensor(self.w,dtype=torch.complex128) if transports is None else transports
        f = torch.as_tensor(self.f,dtype=torch.float64) if fourier_kernel is None else fourier_kernel
        lam = self.nonlinear if nonlinear is None else nonlinear
        gamma = torch.as_tensor(EUCLIDEAN_GAMMA,dtype=torch.complex128)
        result = s.sum()*0
        for i in range(len(self.weights)):
            for j in range(len(self.weights)):
                block = s[i,j].reshape(8,3,8,3)
                trace = torch.einsum("ij,ajbi->ab",w[i,j].conj().T,block)
                color = (.5*torch.einsum("ij,ab->aibj",w[i,j],trace)-block/6).reshape(24,24)
                c = sum(g@color@g for g in gamma)
                v = f[i,j]*torch.trace(c@s[j,i])
                result = result+.5*self.weights[i]*self.weights[j]*(v+lam*v*v/(2*self.scale**8))
        return result


class FunctionalRemainder:
    """Subtract retained physics at the FUNCTIONAL level, then differentiate.

    For nonlinear dynamics Phi[F_full]-Phi[F_ret] != Phi[F_full-F_ret].
    Identical field/source and quadrature coordinates are required. This
    operation does not provide unknown physical matching amplitudes.
    """
    def __init__(self,full,retained):
        if (full.f.shape!=retained.f.shape or not np.array_equal(full.weights,retained.weights)
                or not np.array_equal(full.w,retained.w)):
            raise ValueError("Full and retained functionals need the same field/source coordinates.")
        self.full,self.retained = full,retained

    def value(self,s):
        return self.full.value(s)-self.retained.value(s)

    def self_energy(self,s):
        return self.full.self_energy(s)-self.retained.self_energy(s)

    def kernel_action(self,s,ds):
        return self.full.kernel_action(s,ds)-self.retained.kernel_action(s,ds)

    def path_source(self,s,dw):
        a,b = self.full.path_source(s,dw),self.retained.path_source(s,dw)
        return a[0]-b[0],a[1]-b[1]

"""Independent smooth minus gauging of the global-Jacobi oscillator projector.

This is the continuous positive-plus family at FIXED mode-envelope membership,
not the gauging of the finite longitudinal projector or retained isospin.
It supplies no physical finite condition and is not automatically assembled.

For physical leg a, Q^2=sum_i p_iT^2/x_i-P_T^2 implies
 sym(dF/dp_a+ F_inverse) = S_a=(1-w_a w_a^T)/(2P+),
 (dF/dp_a+) x = -w_a/P+, where w_a=F[:,a].
The antisymmetric frame connection commutes with a complete HO shell.
The normalized-basis derivative is L=(S Q-w_a P_T/P+) . grad + tr(S).
The minus commutator is [Pi,L], independently of a transverse Ward identity.
Its quadratic part reaches exactly N+2, and the CM term is (P_T/P+)[Pi,T].
"""
from operator import index
from dataclasses import dataclass
import numpy as np

from .m2_coupled_longitudinal_jacobi import oscillator_shell
from .m2_gaussian_field_regulator import _array
from .m2_gaussian_cluster_projector_kernel import (
    GaussianHamiltonianOrbital, _cutoff, _commutator)
from .m2_galerkin_projector_current import _rectangular_displacement

SMOOTH_PRESCRIPTION = "continuous-global-Jacobi-fixed-mode-membership"


def _parameters(dimension, cutoff, weights, b_GeV, total_pplus_GeV):
    N = _cutoff(cutoff)
    if isinstance(dimension, (bool, np.bool_)) or index(dimension) < 1:
        raise ValueError("positive exact Jacobi dimension required")
    d = index(dimension)
    w = _array(weights, (d,), "physical Jacobi recoil row")
    b, pp = float(b_GeV), float(total_pplus_GeV)
    if not np.all(np.isfinite([b, pp])) or min(b, pp) <= 0:
        raise ValueError("positive finite width and total P+ required")
    return d, N, w, b, pp


def smooth_minus_projector_commutator(dimension, cutoff, weights, *, b_GeV,
        total_pplus_GeV, total_transverse_GeV, prescription):
    """Charge-stripped [Pi,L_a] on N+2; GeV^-1, Hermitian at zero transfer.

    This is -dPi/dp_a+ at fixed physical transverse momenta, represented
    in the unperturbed normalized HO basis. Differentiating only the
    recoil displacement would miss the squeeze and normalization terms.
    """
    if prescription != SMOOTH_PRESCRIPTION:
        raise ValueError("explicit continuous-family, fixed-envelope prescription required")
    d, N, w, b, pp = _parameters(dimension, cutoff, weights, b_GeV, total_pplus_GeV)
    pt = _array(total_transverse_GeV, (2,), "intermediate total transverse momentum")
    inside, shell = oscillator_shell(d, N), oscillator_shell(d, N+2)
    lookup = {q:i for i,q in enumerate(shell)}
    n = len(inside)
    result = np.zeros((len(shell),)*2, complex)
    S = (np.eye(d)-np.outer(w,w))/(2*pp)
    # L_squeeze = sum_rs S_rs (a_+r a_-s - a_+r^dag a_-s^dag).
    for col, state in enumerate(shell):
        for r in range(d):
            for s in range(d):
                for step in (-1, 1):
                    out = list(state)
                    out[r] += step
                    out[d+s] += step
                    if min(out) < 0 or sum(out) > N+2:
                        continue
                    row = lookup[tuple(out)]
                    factor = np.sqrt((state[r]+(step==1))*(state[d+s]+(step==1)))
                    result[row,col] += (int(row<n)-int(col<n))*(-step)*S[r,s]*factor
    transverse = _commutator(d, N, w, b)
    nb = transverse.shape[1]
    result[:nb,:nb] += np.einsum("j,jab->ab", pt/pp, transverse)
    return result


def _path(quadrature_order):
    if isinstance(quadrature_order, (bool, np.bool_)) or index(quadrature_order) < 2:
        raise ValueError("explicit integer path order at least two required")
    nodes, weights = np.polynomial.legendre.leggauss(index(quadrature_order))
    return zip((nodes+1)/2, weights/2)


def straight_path_smooth_minus_projector_vertex(dimension, cutoff, outer_cutoff,
        weights, transfer_GeV, *, b_GeV, total_pplus_GeV,
        mean_total_transverse_GeV, quadrature_order, prescription):
    """Off-diagonal smooth K-minus, with continuum translations on both sides.

    The intermediate CM momentum is Pbar_T+(t-1/2)Delta_T. Consequently
    K-minus(Pbar+P+ beta)=K-minus(Pbar)+beta.K_T, while its intrinsic
    value is derived from the independent plus derivative, not set to zero.
    Reported outer cutoff does not truncate any intermediate displacement.
    """
    d,N,w,b,pp = _parameters(dimension,cutoff,weights,b_GeV,total_pplus_GeV)
    outer = _cutoff(outer_cutoff)
    if outer < N:
        raise ValueError("reported outer shell must contain the projector")
    delta = _array(transfer_GeV,(2,),"transverse transfer")
    mean = _array(mean_total_transverse_GeV,(2,),"mean total transverse momentum")
    n,no = len(oscillator_shell(d,N)),len(oscillator_shell(d,outer))
    result = np.zeros((no,no),complex)
    for t,weight in _path(quadrature_order):
        comm = smooth_minus_projector_commutator(d,N,w,b_GeV=b,total_pplus_GeV=pp,
            total_transverse_GeV=mean+(t-.5)*delta,prescription=prescription)
        left = _rectangular_displacement(d,outer,N+2,w[:,None]*delta*(1-t),b)
        right = _rectangular_displacement(d,N+2,outer,w[:,None]*delta*t,b)
        result += weight*(left@comm@right)
    result[:n,:n] = 0
    result[n:,n:] = 0
    return result


def exact_smooth_minus_projector_boundary(hamiltonian, *, out_cutoff, in_cutoff,
        out_weights, in_weights, out_charges, in_charges, transfer_GeV,
        total_pplus_GeV, mean_total_transverse_GeV, quadrature_order, prescription):
    """Exact Pi_f K-minus_f H Pi_i + Pi_f H K-minus_i Pi_i, over 2P+.

    Same original Gaussian H and endpoint-product convention as the exact
    transverse boundary. N+2 contains the plus-derivative commutator, not
    the Hamiltonian's infinite Gaussian tail. No envelope/isospin minus
    current, physical calibration or dynamical covariance is implied.
    """
    if not isinstance(hamiltonian,GaussianHamiltonianOrbital):
        raise ValueError("explicit original Gaussian Hamiltonian orbital required")
    if prescription != SMOOTH_PRESCRIPTION:
        raise ValueError("explicit continuous-family, fixed-envelope prescription required")
    co,ci = _cutoff(out_cutoff),_cutoff(in_cutoff)
    delta = _array(transfer_GeV,(2,),"transverse transfer")
    mean = _array(mean_total_transverse_GeV,(2,),"mean total transverse momentum")
    df,di = hamiltonian.out_map.shape[0],hamiltonian.in_map.shape[0]
    groups = []
    for rows,charges,d in ((out_weights,out_charges,df),(in_weights,in_charges,di)):
        e = np.asarray(charges,float)
        if e.ndim != 1 or not len(e) or not np.all(np.isfinite(e)):
            raise ValueError("explicit finite physical charges required")
        w = _array(rows,(len(e),d),"physical Jacobi recoil rows")
        _parameters(d,0,w[0],hamiltonian.b_GeV,total_pplus_GeV)
        groups.append((w,e))
    pp,b = float(total_pplus_GeV),float(hamiltonian.b_GeV)
    h = hamiltonian.integrate(co,ci)
    result = np.zeros_like(h,dtype=complex)
    path = tuple(_path(quadrature_order))
    for side,(rows,charges) in enumerate(groups):
        d,N = (df,co) if side==0 else (di,ci)
        for w,e in zip(rows,charges):
            if e==0:
                continue
            for t,weight in path:
                comm = smooth_minus_projector_commutator(d,N,w,b_GeV=b,total_pplus_GeV=pp,
                    total_transverse_GeV=mean+(t-.5)*delta,prescription=prescription)
                left = _rectangular_displacement(d,N,N+2,w[:,None]*delta*(1-t),b)
                right = _rectangular_displacement(d,N+2,N,w[:,None]*delta*t,b)
                if side==0:
                    shifted = hamiltonian.integrate(co+2,ci,out_shift=w[:,None]*delta*t)
                    result += e*weight*(left@comm@(shifted-right@h))
                else:
                    shifted = hamiltonian.integrate(co,ci+2,in_shift=-w[:,None]*delta*(1-t))
                    result += e*weight*((shifted-h@left)@comm@right)
    return result/(2*pp)


@dataclass(frozen=True,kw_only=True)
class SmoothMinusProjectorContext:
    """Explicit smooth-minus representative; no isospin/envelope completion.

    Existing transverse APIs retain their old default. Supplying this
    context selects ONE smooth-minus component, not an entire four-current.
    """
    mean_total_transverse_GeV: tuple
    prescription: str

    def __post_init__(self):
        if self.prescription != SMOOTH_PRESCRIPTION:
            raise ValueError("explicit continuous-family, fixed-envelope prescription required")
        mean = _array(self.mean_total_transverse_GeV,(2,),"mean transverse momentum")
        object.__setattr__(self,"mean_total_transverse_GeV",tuple(mean))


def projector_component_count(minus_context):
    if minus_context is None:
        return 2
    if not isinstance(minus_context,SmoothMinusProjectorContext):
        raise ValueError("explicit smooth-minus context required, not an inferred component")
    return 1


def exact_projector_components(hamiltonian, *, minus_context=None, **kwargs):
    """Same physical H/Fock assembly, with the independently derived tangent."""
    projector_component_count(minus_context)
    if minus_context is None:
        from .m2_gaussian_cluster_projector_kernel import exact_transverse_projector_boundary
        return exact_transverse_projector_boundary(hamiltonian,**kwargs)
    return exact_smooth_minus_projector_boundary(hamiltonian,**kwargs,
        mean_total_transverse_GeV=minus_context.mean_total_transverse_GeV,
        prescription=minus_context.prescription)[None]


def free_projector_components(dimension,cutoff,weights,transfer_GeV,*,b_GeV,
        total_pplus_GeV,quadrature_order,minus_context=None):
    """Finite-support free-H product; no intermediate Gaussian-tail cutoff."""
    projector_component_count(minus_context)
    if minus_context is None:
        from .m2_galerkin_projector_current import free_galerkin_boundary
        return free_galerkin_boundary(dimension,cutoff,weights,transfer_GeV,
            b_GeV=b_GeV,total_pplus_GeV=total_pplus_GeV,quadrature_order=quadrature_order)
    from .m2_coupled_pilot import free_orbital
    d,N,w,b,pp = _parameters(dimension,cutoff,weights,b_GeV,total_pplus_GeV)
    k = straight_path_smooth_minus_projector_vertex(d,N,N+2,w,transfer_GeV,
        b_GeV=b,total_pplus_GeV=pp,quadrature_order=quadrature_order,
        mean_total_transverse_GeV=minus_context.mean_total_transverse_GeV,
        prescription=minus_context.prescription)
    h = free_orbital(d,N+2,b).toarray()
    n = len(oscillator_shell(d,N))
    return ((k[:n,n:]@h[n:,:n]+h[:n,n:]@k[n:,:n])/(2*pp))[None]


def retained_free_projector_components(outgoing,incoming,transfer,*,total_pplus_GeV,
        charge,quadrature_order,minus_context=None):
    """Actual retained symmetry bases, preserving separate gluon helicities."""
    from scipy import sparse
    from .m2_coupled_longitudinal_jacobi import longitudinal_basis,jacobi_frame
    from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS
    from .m2_canonical_good_current import parton_charge
    nm = projector_component_count(minus_context)
    for name in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV",
                 "gluon_mass_GeV","coupling"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("current endpoints disagree on "+name)
    sf = np.array(SPIN_FLAVOR_BASIS)
    result = [[] for _ in range(nm)]
    p = incoming
    for sector,uf,ui in zip(("qqq","qqqg","qqqg"),outgoing.bases,incoming.bases):
        d = 2 if sector=="qqq" else 3
        nc,n = d-1,len(oscillator_shell(d,p.cutoff))
        blocks = [[] for _ in range(nm)]
        for modes in longitudinal_basis(sector,p.total_k2):
            frame = jacobi_frame(modes)
            current = [sparse.csr_matrix((n*nc*64,)*2,dtype=complex) for _ in range(nm)]
            for a in range(3):
                orbital = free_projector_components(d,p.cutoff,frame.forward[:-1,a],transfer,
                    b_GeV=p.width_GeV,total_pplus_GeV=total_pplus_GeV,
                    quadrature_order=quadrature_order,minus_context=minus_context)
                charges = sparse.diags(np.tile(parton_charge(sf[:,a],"q",charge),nc))
                for j in range(nm):current[j]+=sparse.kron(orbital[j],charges,format="csr")
            for j in range(nm):blocks[j].append(current[j])
        for j in range(nm):
            result[j].append((uf.getH()@sparse.block_diag(blocks[j],format="csr")@ui).toarray())
    return np.array([sparse.block_diag(parts).toarray() for parts in result])

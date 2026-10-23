"""Regular gauging of the noncommuting transverse oscillator projector.

An explicit straight-path representative, NOT a physical current selection.
All displacements are continuum matrix elements, not exponentials of a
truncated position matrix. The off-diagonal projector tangent preserves
idempotency to first order. Transverse path freedom remains unconstrained.

The gauging/product-rule principle is described in Kvinikhidze et al.,
arXiv:0904.4128. Their commuting momentum cutoff formula is NOT used here.
For T_j = d D(Delta)/d Delta_j at zero and Pi the oscillator projector,
 C_j = integral_0^1 D((1-t)Delta) [Pi,T_j] D(t Delta) dt,
 K_j = Pi C_j Q + Q C_j Pi.
Thus Delta.K=[Pi,D], K=Pi K+K Pi, and K(-Delta)^dagger=K(Delta).
The finite-rank commutator has support only through shell N+1, exactly.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from .m2_canonical_good_current import parton_charge
from .m2_coupled_longitudinal_jacobi import (
    oscillator_shell,jacobi_frame,longitudinal_basis,
)
from .m2_coupled_pilot import _displacement,free_orbital
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS


def _rectangular_displacement(d,bra_cutoff,ket_cutoff,shift,b):
    bra=np.array(oscillator_shell(d,bra_cutoff))
    ket=np.array(oscillator_shell(d,ket_cutoff))
    a=(shift[:,0]-1j*shift[:,1])/(2*b);alpha=np.r_[a,a.conj()]
    tables=np.array([[[_displacement(o,i,z) for i in range(ket_cutoff+1)]
                      for o in range(bra_cutoff+1)] for z in alpha])
    result=np.ones((len(bra),len(ket)),complex)
    for k in range(2*d):
        result*=tables[k,bra[:,k,None],ket[None,:,k]]
    return result


def straight_path_projector_vertex(dimension,cutoff,outer_cutoff,weights,transfer,*,
                                   b_GeV,quadrature_order):
    """Two off-diagonal Pi-to-Q/Q-to-Pi vertices, units GeV^-1.

    D translates Jacobi momenta by weights[a]*Delta. Charges are external.
    Outer cutoff selects reported matrix elements, NOT an intermediate
    translation truncation. An explicit quadrature order controls only the
    smooth path integral; callers must check numerical convergence.
    This supplies no plus/minus, dynamical spin or physical finite term.
    """
    shell=oscillator_shell(dimension,cutoff)
    outer=oscillator_shell(dimension,outer_cutoff)
    if outer_cutoff<cutoff:raise ValueError("outer cutoff must contain Pi")
    if not isinstance(quadrature_order,(int,np.integer)) or quadrature_order<2:
        raise ValueError("explicit integer path quadrature order >=2 required")
    d=len(shell[0])//2;n=len(shell);no=len(outer)
    w=np.asarray(weights,float);delta=np.asarray(transfer,float);b=float(b_GeV)
    if (w.shape!=(d,) or delta.shape!=(2,)
            or not np.all(np.isfinite(np.r_[w,delta,b])) or b<=0):
        raise ValueError("finite recoil/transfer and positive width required")
    boundary=oscillator_shell(d,cutoff+1);nb=len(boundary)
    lookup={q:i for i,q in enumerate(boundary)}
    commutators=np.zeros((2,nb,nb),complex)
    alpha_derivative=np.array([np.r_[w,w],np.r_[-1j*w,1j*w]])/(2*b)
    for col,q in enumerate(boundary):
        for k in range(2*d):
            for step in (-1,1):
                out=list(q);out[k]+=step
                if out[k]<0 or sum(out)>cutoff+1:continue
                row=lookup[tuple(out)]
                pi_difference=int(row<n)-int(col<n)
                value=alpha_derivative[:,k] if step==1 else -alpha_derivative[:,k].conj()
                commutators[:,row,col]+=pi_difference*np.sqrt(q[k]+(step==1))*value
    nodes,quadrature=leggauss(quadrature_order)
    result=np.zeros((2,no,no),complex)
    # Only the finite-rank [Pi,T] is multiplied between continuum D blocks.
    for t,weight in zip((nodes+1)/2,quadrature/2):
        left=_rectangular_displacement(d,outer_cutoff,cutoff+1,w[:,None]*delta*(1-t),b)
        right=_rectangular_displacement(d,cutoff+1,outer_cutoff,w[:,None]*delta*t,b)
        for j in (0,1):
            result[j]+=weight*(left@commutators[j]@right)
    # Tangent to Pi^2=Pi; diagonal pieces of C are transverse and discarded
    # only as part of this explicit path convention, not numerical pruning.
    result[:,:n,:n]=0
    result[:,n:,n:]=0
    return result


def free_galerkin_boundary(dimension,cutoff,weights,transfer,*,b_GeV,
                           total_pplus_GeV,quadrature_order):
    """Charge-stripped J_T regulator partner for Pi H0 Pi, not a full current.

    delta.J_boundary = (-Pi H0 Q D Pi + Pi D Q H0 Pi)/(2P+).
    H0=sum Q_a^2 has exact support through N+2. Add this partner to the
    canonical projected current to close the finite free-H Ward identity.
    Mass constants commute with Pi and contribute zero. Interaction and
    field-contraction regulator partners must be derived separately.
    """
    pp=float(total_pplus_GeV)
    if not np.isfinite(pp) or pp<=0:raise ValueError("positive total P+ required")
    n=len(oscillator_shell(dimension,cutoff))
    vertex=straight_path_projector_vertex(dimension,cutoff,cutoff+2,weights,transfer,
        b_GeV=b_GeV,quadrature_order=quadrature_order)
    h=free_orbital(dimension,cutoff+2,b_GeV).toarray()
    return np.array([(k[:n,n:]@h[n:,:n]+h[:n,n:]@k[n:,:n])/(2*pp) for k in vertex])


def retained_free_boundary_between(outgoing,incoming,transfer,*,total_pplus_GeV,
                                   charge,quadrature_order):
    """Explicit free-projector partner on the actual retained symmetry bases.

    This is a separately callable exploratory operator, not silently added
    to any current, Hamiltonian, matched snapshot or production observable.
    """
    for name in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV",
                 "gluon_mass_GeV","coupling"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("current endpoints disagree on "+name)
    sf=np.array(SPIN_FLAVOR_BASIS);p=incoming;result=[[],[]]
    for sector,uf,ui in zip(("qqq","qqqg","qqqg"),outgoing.bases,incoming.bases):
        d=2 if sector=="qqq" else 3;nc=d-1
        n=len(oscillator_shell(d,p.cutoff));blocks=[[],[]]
        for modes in longitudinal_basis(sector,p.total_k2):
            frame=jacobi_frame(modes)
            current=[sparse.csr_matrix((n*nc*64,)*2,dtype=complex) for _ in (0,1)]
            for active in range(3):
                orbital=free_galerkin_boundary(d,p.cutoff,frame.forward[:-1,active],transfer,
                    b_GeV=p.width_GeV,total_pplus_GeV=total_pplus_GeV,
                    quadrature_order=quadrature_order)
                charges=sparse.diags(np.tile(parton_charge(sf[:,active],"q",charge),nc))
                for j in (0,1):
                    current[j]+=sparse.kron(orbital[j],charges,format="csr")
            for j in (0,1):blocks[j].append(current[j])
        for j in (0,1):
            result[j].append((uf.getH()@sparse.block_diag(blocks[j],format="csr")@ui).toarray())
    return np.array([sparse.block_diag(blocks).toarray() for blocks in result])

def retained_projector_vertex_between(outgoing, incoming, *, inner_cutoff,
                                      transfer, charge, quadrature_order):
    """Project the straight-path inner-cutoff tangent on supplied OUTER bases.

    The outer bases may have different Jz but share every other parameter.
    Every reported continuum translation matrix element is exact up to the
    explicit path quadrature. This does not remove the outer-H truncation.
    """
    for name in ("total_k2", "cutoff", "target", "width_GeV", "quark_mass_GeV",
                 "gluon_mass_GeV", "coupling"):
        if getattr(outgoing, name) != getattr(incoming, name):
            raise ValueError("projector endpoints disagree on "+name)
    if (isinstance(inner_cutoff, (bool, np.bool_))
            or not isinstance(inner_cutoff, (int, np.integer))
            or not 0 <= inner_cutoff < incoming.cutoff):
        raise ValueError("explicit inner cutoff strictly below outer cutoff required")
    sf = np.array(SPIN_FLAVOR_BASIS)
    p = incoming
    result = [[], []]
    for sector, uf, ui in zip(("qqq", "qqqg", "qqqg"), outgoing.bases, incoming.bases):
        d = 2 if sector == "qqq" else 3
        nc = d-1
        n = len(oscillator_shell(d, p.cutoff))
        blocks = [[], []]
        for modes in longitudinal_basis(sector, p.total_k2):
            frame = jacobi_frame(modes)
            current = [sparse.csr_matrix((n*nc*64,)*2, dtype=complex) for _ in (0, 1)]
            for active in range(3):
                orbital = straight_path_projector_vertex(
                    d, inner_cutoff, p.cutoff, frame.forward[:-1, active], transfer,
                    b_GeV=p.width_GeV, quadrature_order=quadrature_order)
                charges = sparse.diags(np.tile(parton_charge(sf[:, active], "q", charge), nc))
                for j in (0, 1):
                    current[j] += sparse.kron(orbital[j], charges, format="csr")
            for j in (0, 1):
                blocks[j].append(current[j])
        for j in (0, 1):
            result[j].append((uf.getH()@sparse.block_diag(blocks[j], format="csr")@ui).toarray())
    return np.array([sparse.block_diag(blocks).toarray() for blocks in result])


def interacting_galerkin_boundary_between(outgoing, incoming, *, parent_outgoing,
                                           parent_incoming, transfer, charge,
                                           total_pplus_GeV, quadrature_order):
    """Inner-projector partner for the ACTUAL supplied interacting outer H.

    J_boundary = P_f^dag [K H_i + H_f K] P_i / (2P+).
    K has only inner-to-complement blocks, so no diagonal H or free current
    is counted twice. Contains all represented cubic and quartic sources.
    The parent oscillator cutoff is explicit: contact vertices have tails
    beyond every finite shell, so this is not an infinite-cutoff result.
    Smaller pilots supply coordinates; the inner H is p-dagger H_parent p,
    not an independently changed smaller-pilot matrix.
    No term is added automatically to a canonical current or H2 snapshot.
    """
    from .m2_matched_h2_reduction import nested_pilot_isometry

    pp = float(total_pplus_GeV)
    if not np.isfinite(pp) or pp <= 0:
        raise ValueError("positive total P+ required")
    if outgoing.cutoff != incoming.cutoff:
        raise ValueError("common inner cutoff required")
    pf = nested_pilot_isometry(outgoing, parent_outgoing)
    pi = nested_pilot_isometry(incoming, parent_incoming)
    vertex = retained_projector_vertex_between(parent_outgoing, parent_incoming,
        inner_cutoff=incoming.cutoff, transfer=transfer, charge=charge,
        quadrature_order=quadrature_order)
    return np.array([pf.conj().T@(k@parent_incoming.matrix
                    + parent_outgoing.matrix@k)@pi/(2*pp) for k in vertex])

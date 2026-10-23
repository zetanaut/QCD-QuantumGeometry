"""Normal-ordered A_perp^4 contact within qqqgg, per g_s².

SB H_int contains +g²/4 f^(a b c) f^(a d e) A_b^i A_c^j A_d^i A_e^j.
All 24 external-leg assignments precede the normalized two-boson 1/2.
Quadratic Wick descendants are separate; no zero-transfer removal applies.
"""
from functools import lru_cache
from itertools import permutations,product
from math import pi,sqrt
import numpy as np
from scipy import sparse

from deuteron_wigner.pilot.color import structure_constants
from .m2_excluded_fock_kinematics import color_basis
from .m2_excluded_current_scattering import five_body_contact
from .m2_coupled_pair_action import _width

ORDERS=tuple(permutations(range(4)))


def distinguished_quartic_tensor(constants,polarizations):
    """Unreduced color tensor after all four external-field assignments.

    External legs0/1 are outgoing and2/3 incoming; conjugate outgoing
    polarizations BEFORE calling. No source1/4, Fock1/2 or mode norms here.
    """
    f=np.asarray(constants,float);e=np.asarray(polarizations,complex)
    if f.ndim!=3 or len(set(f.shape))!=1 or e.shape!=(4,2) or not np.all(np.isfinite(e)):
        raise ValueError("finite cubic color tensor and four transverse vectors required")
    base=np.einsum("xab,xcd->abcd",f,f)
    result=np.zeros_like(base,dtype=complex)
    for order in ORDERS:
        spin=np.dot(e[order[0]],e[order[2]])*np.dot(e[order[1]],e[order[3]])
        result+=spin*base.transpose(np.argsort(order))
    return result


@lru_cache(maxsize=24)
def quartic_ordered_color(order):
    if tuple(order) not in ORDERS:
        raise ValueError("external-leg permutation required")
    f=structure_constants()
    tensor=np.einsum("xab,xcd->abcd",f,f).transpose(np.argsort(order))
    u=color_basis("qqqgg")
    raw=np.einsum("abcd,ijkcdn->ijkabn",tensor,u.reshape(3,3,3,8,8,6)).reshape(u.shape)
    reduced=u.conj().T@raw
    if np.linalg.norm(raw-u@reduced)>2e-12:
        raise ArithmeticError("quartic contact leaves complete six-singlet carrier")
    reduced.setflags(write=False)
    return reduced


@lru_cache(maxsize=16,typed=True)
def quartic_spin_color(h1_out,h2_out,h1_in,h2_in):
    helicities=(h1_out,h2_out,h1_in,h2_in)
    if any(h not in (-1,1) or isinstance(h,(bool,np.bool_)) for h in helicities):
        raise ValueError("four signed physical gluon helicities required")
    e=[-np.array([h,1j])/sqrt(2) for h in helicities]
    e[:2]=[a.conj() for a in e[:2]]
    return sum(np.dot(e[p[0]],e[p[2]])*np.dot(e[p[1]],e[p[3]])*quartic_ordered_color(p)
               for p in ORDERS)


def four_gluon_QQ_per_gs2(space,*,b_GeV):
    """Canonical A^4 Galerkin operator on both gluon slots, complete spin/color.

    K2/[16 pi sqrt(k1' k2' k1 k2)] multiplies the distinguished tensor
    and the normalized global-Jacobi transverse pair contact.
    The coefficient includes source1/4 and the two normalized Bose states.
    """
    if space.sector!="qqqgg":
        raise ValueError("explicit qqqgg sector required")
    _width(b_GeV)
    lookup={m:i for i,m in enumerate(space.longitudinal)}
    no,nc,ns=len(space.orbitals),space.ncolor,space.nspin
    rows,cols,values=[],[],[]
    for li,incoming in enumerate(space.longitudinal):
        for k1 in range(2,incoming[3]+incoming[4],2):
            outgoing=incoming[:3]+(k1,incoming[3]+incoming[4]-k1)
            orbital=five_body_contact("qqqgg",incoming,outgoing,(3,4),b_GeV,space.cutoff)
            factor=space.total_k2/(16*pi*sqrt(np.prod(incoming[3:]+outgoing[3:])))
            for ho1,ho2,hi1,hi2 in product((-1,1),repeat=4):
                color=quartic_spin_color(ho1,ho2,hi1,hi2)
                if not np.any(color):
                    continue
                selected=np.flatnonzero((2*space.spins[:,3]-1==hi1)&(2*space.spins[:,4]-1==hi2))
                changed=space.spins[selected].copy()
                changed[:,3],changed[:,4]=(ho1+1)//2,(ho2+1)//2
                outspin=np.ravel_multi_index(changed.T,space.spin_dims)
                block=factor*np.kron(orbital,color)
                a,b=np.nonzero(block)
                ir=(lookup[outgoing]*no*nc+a[:,None])*ns+outspin
                ic=(li*no*nc+b[:,None])*ns+selected
                pr,pc=space.lookup[ir],space.lookup[ic]
                keep=(pr>=0)&(pc>=0);outside=(pr<0)&(pc>=0)
                v=np.broadcast_to(block[a,b][:,None],keep.shape)
                if np.linalg.norm(v[outside])>2e-12:
                    raise ArithmeticError("A^4 contact violates exact Iz/Jz")
                rows.append(pr[keep]);cols.append(pc[keep]);values.append(v[keep])
    return sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                             shape=(space.dimension,space.dimension)).tocsr()


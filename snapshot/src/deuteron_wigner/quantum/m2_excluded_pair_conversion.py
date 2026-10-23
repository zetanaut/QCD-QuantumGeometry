"""Canonical gg <-> quark-antiquark conversion within five-body Q.

Both signed constrained-fermion orderings (IF), or an explicitly requested
IG cross-check kernel. A common ordered-slot conversion map retains exact
Jacobi measure and normalized 2 A4 / sqrt(2!) Fock factors. The IG kernel is
a validation alternative and must NOT be added to the pair-current square.
"""
from functools import lru_cache
from itertools import product
from math import pi,sqrt
import numpy as np
from scipy import sparse

from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS
from deuteron_wigner.microscopic.h0.color import representation_generators
from .m2_canonical_emission import lf_spinor
from .m2_excluded_fock_vertices import antiquark_spinor
from .m2_excluded_fock_kinematics import frame,color_basis,color_source
from .m2_excluded_fermion_scattering import _pair_rotation
from .m2_coupled_pair_action import _contact_shell,_width
from .m2_instantaneous_sea_exit import sea_pair_spin_factor
from .m2_excluded_pair_current import adjoint_pair_current_factor


def _route(incoming,outgoing):
    fi,fo=frame("qqqgg",tuple(incoming)),frame("qqqq-qbar",tuple(outgoing))
    if incoming[:3]!=outgoing[:3] or sum(incoming)!=sum(outgoing):
        raise ValueError("spectator quarks and pair total must be conserved")
    return fi,fo


@lru_cache(maxsize=2,typed=True)
def conversion_ordered_color(reverse=False):
    if not isinstance(reverse,bool):
        raise ValueError("explicit color ordering required")
    u=color_basis("qqqgg").reshape(27,8,8,6)
    t=np.asarray(representation_generators("3"))
    raw=np.einsum("bij,ajk,rabn->rikn",t,t,u) if reverse else np.einsum("aij,bjk,rabn->rikn",t,t,u)
    raw=raw.reshape(243,6)
    target=color_basis("qqqq-qbar")
    out=target.conj().T@raw
    if np.linalg.norm(raw-target@out)>2e-12:
        raise ArithmeticError("pair conversion leaves complete sea singlets")
    out.setflags(write=False)
    return out


@lru_cache(maxsize=16,typed=True)
def conversion_spin(hq,ha,h1,h2):
    if any(h not in (-1,1) or isinstance(h,(bool,np.bool_)) for h in (hq,ha,h1,h2)):
        raise ValueError("four signed physical helicities required")
    u=lf_spinor(1.,0.,0.,.35,hq)/sqrt(2)
    v=antiquark_spinor(1.,0.,0.,.35,ha)/sqrt(2)
    eps=[-np.array([h,1j])/sqrt(2) for h in (h1,h2)]
    left,right=[sum(GAMMA[i+1]*e[i] for i in range(2)) for e in eps]
    return complex(u.conj()@GAMMA[0]@left@GAMMA_PLUS@right@v)


def instantaneous_conversion_kernel(incoming,outgoing,helicities):
    """Distinguished-gluon IF kernel, before transverse integration and Fock factors."""
    _route(incoming,outgoing)
    hq,ha,h1,h2=helicities
    k1,k2=incoming[3:];anti=outgoing[4]
    direct=conversion_spin(hq,ha,h1,h2)*conversion_ordered_color()/(k2-anti)
    reverse=conversion_spin(hq,ha,h2,h1)*conversion_ordered_color(True)/(k1-anti)
    return sum(incoming)/(2*pi*sqrt(k1*k2))*(direct+reverse)


def current_conversion_kernel(incoming,outgoing,helicities):
    """Independent direct IG cross kernel; used to check B_sea D B_gg†."""
    _route(incoming,outgoing)
    hq,ha,h1,h2=helicities
    color=color_source("g_to_pair")@color_source("g_to_gg").conj().T
    factor=sqrt(2)*adjoint_pair_current_factor(*incoming[3:],h1,h2).conjugate()
    return 2*sum(incoming)/(pi*sum(incoming[3:])**2)*sea_pair_spin_factor(hq,ha)*factor*color


def conversion_contact(incoming,outgoing,b_GeV,cutoff):
    fi,fo=_route(incoming,outgoing);b=_width(b_GeV)
    ai=np.prod(fi.fractions[3:])/sum(fi.fractions[3:])
    ao=np.prod(fo.fractions[3:])/sum(fo.fractions[3:])
    ri=_pair_rotation("qqqgg",tuple(incoming),(3,4),cutoff)
    ro=_pair_rotation("qqqq-qbar",tuple(outgoing),(3,4),cutoff)
    return b*b/pi*sqrt(ai*ao)*(ro.T@_contact_shell(4,cutoff,cutoff)@ri)


class FiveBodyPairConversion:
    """F=2 A4 M/sqrt(2), with its exact adjoint; no raw dense QQ matrix."""
    def __init__(self,spaces,*,b_GeV,channel):
        if set(spaces)!={"qqqgg","qqqq-qbar"} or channel not in ("IF","IG"):
            raise ValueError("both five-body spaces and explicit IF/IG component required")
        self.spaces=dict(spaces);self.channel=channel
        inside,outside=spaces["qqqgg"],spaces["qqqq-qbar"]
        for name in ("total_k2","cutoff","target","jz_twice"):
            if getattr(inside,name)!=getattr(outside,name):
                raise ValueError("pair conversion bases disagree on "+name)
        kernel=instantaneous_conversion_kernel if channel=="IF" else current_conversion_kernel
        lookup={m:i for i,m in enumerate(outside.longitudinal)}
        ni,no=len(inside.orbitals),len(outside.orbitals)
        rows,cols,values=[],[],[]
        for li,incoming in enumerate(inside.longitudinal):
            for kq in range(1,sum(incoming[3:]),2):
                outgoing=incoming[:3]+(kq,sum(incoming[3:])-kq)
                orbital=conversion_contact(incoming,outgoing,b_GeV,inside.cutoff)
                for hq,ha,h1,h2 in product((-1,1),repeat=4):
                    color=kernel(incoming,outgoing,(hq,ha,h1,h2))
                    if not np.any(color):continue
                    selected=np.flatnonzero((2*inside.spins[:,3]-1==h1)&(2*inside.spins[:,4]-1==h2))
                    for flavor in range(2):
                        converted=np.column_stack((inside.spins[selected,:3],
                            np.full(len(selected),2*flavor+(hq+1)//2),
                            np.full(len(selected),2*flavor+(ha+1)//2)))
                        outspin=np.ravel_multi_index(converted.T,outside.spin_dims)
                        block=np.kron(orbital,color)
                        a,b=np.nonzero(block)
                        ir=(lookup[outgoing]*no*outside.ncolor+a[:,None])*outside.nspin+outspin
                        ic=(li*ni*inside.ncolor+b[:,None])*inside.nspin+selected
                        pr,pc=outside.lookup[ir],inside.lookup[ic]
                        keep=(pr>=0)&(pc>=0);bad=(pr<0)&(pc>=0)
                        v=np.broadcast_to(block[a,b][:,None],keep.shape)
                        if np.linalg.norm(v[bad])>2e-12:
                            raise ArithmeticError("pair conversion violates Iz/Jz")
                        rows.append(pr[keep]);cols.append(pc[keep]);values.append(v[keep])
        self.distinguished=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                                             shape=(outside.dimension,inside.dimension)).tocsr()

    def apply(self,columns):
        if set(columns)!=set(self.spaces):
            raise ValueError("both sector column sets required")
        count=None
        for name,space in self.spaces.items():
            a=np.asarray(columns[name],complex)
            if a.ndim!=2 or a.shape[0]!=space.dimension or not np.all(np.isfinite(a)):
                raise ValueError("finite columns on exact raw Q coordinates required")
            if count is not None and count!=a.shape[1]:raise ValueError("common column count required")
            count=a.shape[1]
        sea=self.spaces["qqqq-qbar"]
        forward=sqrt(2)*sea.statistics_project(self.distinguished@columns["qqqgg"])
        reverse=sqrt(2)*(self.distinguished.getH()@sea.statistics_project(columns["qqqq-qbar"]))
        return {"qqqgg":reverse,"qqqq-qbar":forward}


"""Canonical q-antiquark annihilation/recreation term within qqqq-qbar.

The two pair-current orders in normal-ordered j_q^+ j_q^+ are retained.
Positive total pair transfer is never Q0, including identical incoming and
outgoing longitudinal tuples. All four possible annihilating quarks and
coherent u/d recreation are summed. This is a QQ component, not full QCD.
"""
from functools import lru_cache
from math import pi

import numpy as np
from scipy import sparse

from deuteron_wigner.microscopic.h0.color import representation_generators
from .m2_coupled_pair_action import _pair, _width
from .m2_excluded_fock_kinematics import color_basis
from .m2_excluded_fermion_scattering import fermion_contact_block


@lru_cache(maxsize=4, typed=True)
def annihilation_color(quark):
    _pair((quark,4),5)
    if quark not in range(4):
        raise ValueError("one of the four quarks required")
    u=color_basis("qqqq-qbar")
    tensor=u.reshape(3,3,3,3,3,3)
    rest=tuple(i for i in range(4) if i!=quark)
    order=rest+(quark,4,5)
    adapted=tensor.transpose(order).reshape(27,3,3,3)
    generators=np.asarray(representation_generators("3"))
    raw=np.einsum("aij,akl,rkln->rijn",generators,generators.conj(),adapted)
    raw=raw.reshape((3,)*6).transpose(np.argsort(order)).reshape(u.shape)
    result=u.conj().T@raw
    if np.linalg.norm(raw-u@result)>2e-12:
        raise ArithmeticError("annihilation leaves the complete three-singlet carrier")
    result.setflags(write=False)
    return result


def annihilation_per_gs2(space, *, b_GeV):
    """Both current orders: 2 K2/[pi (k_q2+k_anti2)^2] times two pair spins.

    S_out S_in* = 1 for opposite physical helicities in each same-flavor
    quark-antiquark pair. Unordered-pair counting is one per quark/antiquark,
    with no extra four-quark creation factor for a number-preserving action.
    """
    if space.sector!="qqqq-qbar":
        raise ValueError("explicit sea sector required")
    _width(b_GeV)
    lookup={m:i for i,m in enumerate(space.longitudinal)}
    no,nc,ns=len(space.orbitals),space.ncolor,space.nspin
    rows,cols,values=[],[],[]
    for quark in range(4):
        color=annihilation_color(quark)
        selected=np.flatnonzero((space.spins[:,quark]//2==space.spins[:,4]//2)&
                                (space.spins[:,quark]%2!=space.spins[:,4]%2))
        spin_maps=[]
        for flavor in range(2):
            for helicity in range(2):
                changed=space.spins[selected].copy()
                changed[:,quark]=2*flavor+helicity
                changed[:,4]=2*flavor+1-helicity
                spin_maps.append(np.ravel_multi_index(changed.T,space.spin_dims))
        for li,incoming in enumerate(space.longitudinal):
            total=incoming[quark]+incoming[4]  # always positive, not a difference
            for ki in range(1,total,2):
                outgoing=list(incoming);outgoing[quark],outgoing[4]=ki,total-ki
                outgoing=tuple(outgoing)
                contact=fermion_contact_block(space.sector,incoming,outgoing,(quark,4),
                                              b_GeV,space.cutoff)
                block=np.kron(contact,color)*2*space.total_k2/(pi*total**2)
                a,b=np.nonzero(block)
                raw_cols=(li*no*nc+b[:,None])*ns+selected
                packed_cols=space.lookup[raw_cols]
                for out_spins in spin_maps:
                    raw_rows=(lookup[outgoing]*no*nc+a[:,None])*ns+out_spins
                    packed_rows=space.lookup[raw_rows]
                    keep=(packed_rows>=0)&(packed_cols>=0)
                    outside=(packed_rows<0)&(packed_cols>=0)
                    elements=np.broadcast_to(block[a,b][:,None],keep.shape)
                    if np.linalg.norm(elements[outside])>2e-12:
                        raise ArithmeticError("annihilation violates exact Iz/Jz")
                    rows.append(packed_rows[keep]);cols.append(packed_cols[keep]);values.append(elements[keep])
    return sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                             shape=(space.dimension,space.dimension)).tocsr()


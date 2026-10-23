"""Canonical direct/crossed qg instantaneous-fermion contacts inside qqqgg.

All three quarks and both gluons, complete six-singlet color carrier and
signed odd constrained denominators. This is a separate diagonal QQ term,
not a propagating cubic return, A^4, or the gg <-> sea IF conversion.
"""
from functools import lru_cache
from math import pi,sqrt
import numpy as np
from scipy import sparse

from deuteron_wigner.microscopic.h0.color import representation_generators
from .m2_canonical_instantaneous_fermion import transverse_spin_factor
from .m2_excluded_fock_kinematics import color_basis,_act
from .m2_excluded_current_scattering import five_body_contact,current_pair_color
from .m2_coupled_pair_action import _width


@lru_cache(maxsize=12,typed=True)
def ordered_five_qg_color(quark,gluon,crossed):
    if (quark not in (0,1,2) or gluon not in (3,4)
            or any(isinstance(k,(bool,np.bool_)) for k in (quark,gluon))
            or not isinstance(crossed,bool)):
        raise ValueError("explicit quark/gluon slots and color ordering required")
    u=color_basis("qqqgg");tensor=u.reshape(3,3,3,8,8,6)
    out=np.zeros_like(tensor)
    for a,ta in enumerate(representation_generators("3")):
        for b,tb in enumerate(representation_generators("3")):
            matrix=tb@ta if crossed else ta@tb
            changed=_act(matrix,np.take(tensor,b,axis=gluon),quark)
            destination=[slice(None)]*6;destination[gluon]=a
            out[tuple(destination)]+=changed
    raw=out.reshape(u.shape);reduced=u.conj().T@raw
    if np.linalg.norm(raw-u@reduced)>2e-12:
        raise ArithmeticError("ordered qg contact leaves complete singlet carrier")
    reduced.setflags(write=False)
    return reduced


def instantaneous_fermion_QQ_per_gs2(space,*,b_GeV):
    """Both outer-gluon contractions, with r_direct=p+kin and r_cross=p-kout.

    The constrained denominator can be negative but is odd and never zero.
    Equal external longitudinal modes are retained. Spin/flavor is diagonal
    for this constant transverse contact; both physical gluon helicities
    remain separately resolved in the packed basis.
    """
    if space.sector!="qqqgg":
        raise ValueError("explicit qqqgg space required")
    _width(b_GeV)
    lookup={m:i for i,m in enumerate(space.longitudinal)}
    no,nc,ns=len(space.orbitals),space.ncolor,space.nspin
    spins=np.arange(ns);rows,cols,values=[],[],[]
    for li,incoming in enumerate(space.longitudinal):
        for q in range(3):
            for g in (3,4):
                hq=2*(space.spins[:,q]%2)-1;hg=2*space.spins[:,g]-1
                hd=np.array([transverse_spin_factor(h,k) for h,k in zip(hq,hg)])
                hc=np.array([transverse_spin_factor(h,k,crossed=True) for h,k in zip(hq,hg)])
                for kg in range(2,incoming[q]+incoming[g],2):
                    outgoing=list(incoming);outgoing[g]=kg;outgoing[q]=incoming[q]+incoming[g]-kg
                    outgoing=tuple(outgoing)
                    direct=incoming[q]+incoming[g];crossed=incoming[q]-kg
                    if direct%2!=1 or crossed%2!=1:
                        raise ArithmeticError("constrained fermion denominator must be odd")
                    orbital=five_body_contact("qqqgg",incoming,outgoing,(q,g),b_GeV,space.cutoff)
                    prefactor=space.total_k2/(2*pi*sqrt(incoming[g]*kg))
                    for reverse,spin,denom in ((False,hd,direct),(True,hc,crossed)):
                        block=np.kron(orbital,ordered_five_qg_color(q,g,reverse))*prefactor/denom
                        a,b=np.nonzero(block)
                        ir=(lookup[outgoing]*no*nc+a[:,None])*ns+spins
                        ic=(li*no*nc+b[:,None])*ns+spins
                        pr,pc=space.lookup[ir],space.lookup[ic]
                        keep=(pr>=0)&(pc>=0);outside=(pr<0)&(pc>=0)
                        v=block[a,b][:,None]*spin
                        if np.linalg.norm(v[outside])>2e-12:
                            raise ArithmeticError("instantaneous contact violates Iz/Jz")
                        rows.append(pr[keep]);cols.append(pc[keep]);values.append(v[keep])
    return sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                             shape=(space.dimension,space.dimension)).tocsr()


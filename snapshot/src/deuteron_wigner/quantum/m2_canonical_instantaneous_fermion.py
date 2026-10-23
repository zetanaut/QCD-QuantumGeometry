"""Direct and crossed normal-ordered qg instantaneous-fermion contact.

This is the retained bdagger adag a b descendant of SB Eq.24, NOT a
propagating Vdagger(E-H0)^-1 V interaction. Self-induced inertia from the
bosonic commutator, pair creation and excluded sectors are separate/open.
"""
from functools import lru_cache
from math import pi,sqrt

import numpy as np
from scipy import sparse

from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS
from deuteron_wigner.microscopic.h0.color import ColorSingletBasis,representation_generators
from .m2_canonical_emission import lf_spinor
from .m2_coupled_longitudinal_jacobi import longitudinal_basis,oscillator_shell
from .m2_coupled_pair_action import _nonnegative,_width,pair_contact_block
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS


@lru_cache(maxsize=6)
def ordered_qg_color(quark,crossed):
    """C7 matrices of T^a T^b (direct) or T^b T^a (crossed);
    a is outgoing gluon color, b incoming. No color-charge substitution.
    """
    if quark not in (0,1,2) or not isinstance(crossed,bool):
        raise ValueError("quark slot and Boolean ordering required")
    basis=ColorSingletBasis.construct("qqqg").tensors
    t=basis.reshape(3,3,3,8,2)
    out=np.zeros_like(t)
    generators=representation_generators("3")
    for a,ta in enumerate(generators):
        for b,tb in enumerate(generators):
            matrix=tb@ta if crossed else ta@tb
            out[:,:,:,a,:]+=np.moveaxis(np.tensordot(matrix,t[:,:,:,b,:],axes=(1,quark)),0,quark)
    return basis.conj().T@out.reshape(basis.shape)


@lru_cache(maxsize=8)
def transverse_spin_factor(hq,hg,*,crossed=False):
    """ubar gamma.A_left gamma+ gamma.A_right u / sqrt(2p' 2p).
    Independent of mass and physical momenta in the LF spin basis.
    """
    if hq not in (-1,1) or hg not in (-1,1):
        raise ValueError("quark and gluon helicities must be +/-1")
    u=lf_spinor(1.,0.,0.,.35,hq)/sqrt(2)
    eps=-np.array([hg,1j])/sqrt(2)
    left,right=(eps,eps.conj()) if crossed else (eps.conj(),eps)
    a=sum(GAMMA[i+1]*left[i] for i in range(2))
    b=sum(GAMMA[i+1]*right[i] for i in range(2))
    return complex(u.conj()@GAMMA[0]@a@GAMMA_PLUS@b@u)


def instantaneous_fermion_mass_per_gs2(total_k2,b_GeV,cutoff,h_g,*,_contact_kernel=None):
    """Full (longitudinal, orbital, C7 color, spin/flavor) retained matrix.

    Per ordered field term the coefficient before the transverse contact is
    K2/[2*pi*sqrt(k_g,2' k_g,2)*r_2] times the normalized spin and color
    factors. r_2=k_q,2+k_g,2 for direct; k_q,2-k_g,2' for crossed.
    r_2 is odd and cannot vanish; its sign must NOT be clipped.
    """
    basis=longitudinal_basis("qqqg",total_k2)
    c=_nonnegative(cutoff,"cutoff")
    b=_width(b_GeV)
    if h_g not in (-1,1):
        raise ValueError("fixed gluon helicity must be +/-1")
    no=len(oscillator_shell(3,c))
    lookup={s:i for i,s in enumerate(basis)}
    rows,cols,values=[],[],[]
    for column,state in enumerate(basis):
        for quark in range(3):
            direct_color=ordered_qg_color(quark,False)
            crossed_color=ordered_qg_color(quark,True)
            hd=np.array([transverse_spin_factor(2*(s[quark]%2)-1,h_g) for s in SPIN_FLAVOR_BASIS])
            hc=np.array([transverse_spin_factor(2*(s[quark]%2)-1,h_g,crossed=True) for s in SPIN_FLAVOR_BASIS])
            for kg in range(2,state[quark]+state[3],2):
                out=list(state);out[3]=kg;out[quark]=state[quark]+state[3]-kg;out=tuple(out)
                direct=state[quark]+state[3]
                crossed=state[quark]-kg
                if direct%2!=1 or crossed%2!=1:
                    raise ArithmeticError("APBC/PBC routing lost odd fermion denominator")
                contact=(pair_contact_block if _contact_kernel is None else _contact_kernel)(state,out,(quark,3),b,c)
                cs=(np.kron(direct_color,np.diag(hd))/direct+
                    np.kron(crossed_color,np.diag(hc))/crossed)
                block=(sparse.kron(sparse.csr_matrix(contact),sparse.csr_matrix(cs),
                                  format="coo")*sum(state)/(2*pi*sqrt(state[3]*kg))).tocoo()
                r,ci=block.row,block.col
                rows.extend(lookup[out]*no*128+r)
                cols.extend(column*no*128+ci)
                values.extend(block.data)
    n=len(basis)*no*128
    return sparse.csr_matrix((values,(rows,cols)),shape=(n,n))

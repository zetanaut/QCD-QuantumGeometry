"""Bare normal-ordered qg cross current in qqqg, not full C117.

SB Eq.25 is J_g^a=f_abc (partial_- A_b,mu) A_c^mu.
The transverse Minkowski contraction AND antisymmetric color-slot exchange
give +f_abc A_b^i partial_- A_c^i in Euclidean transverse components.
Its canonical number-preserving matrix is +(k'+k)/(2 sqrt(k' k)) F^a.
C192/C406's historical Euclidean-minus descendant is NOT used as authority
for a physical current here. Its diagonal -F fails the C7 singlet charge test.
"""
from __future__ import annotations

from math import pi, sqrt

import numpy as np
from scipy import sparse

from .m2_c117_pair_current import color_pair_matrix
from .m2_coupled_longitudinal_jacobi import longitudinal_basis, oscillator_shell
from .m2_coupled_pair_action import _nonnegative, _width, pair_contact_block


def canonical_gluon_current_factor(k_out, k_in):
    """Positive normal-ordered mode coefficient; factors 1/(2L) excluded.

    Doubled or undoubled labels give the same dimensionless ratio; use one
    convention consistently. Equal modes give the adjoint charge +F.
    """
    a,b=_width(k_out),_width(k_in)
    return (a+b)/(2*sqrt(a*b))


def qg_cross_mass_per_gs2(total_k2,b_GeV,in_excitation,out_excitation=None,*,_contact_kernel=None):
    """Both mixed orders, every quark, full C7 two-color matrix.

    Axes (longitudinal, orbital, color); identity on spin/flavor and gluon
    helicity. Self-contractions, pair-creation branches, instantaneous fermion,
    pure-gluon contact and sector counterterms are NOT represented.
    """
    basis=longitudinal_basis("qqqg",total_k2)
    ci=_nonnegative(in_excitation,"in_excitation")
    co=ci if out_excitation is None else _nonnegative(out_excitation,"out_excitation")
    b=_width(b_GeV)
    ni,no=(len(oscillator_shell(3,n)) for n in (ci,co))
    lookup={s:i for i,s in enumerate(basis)}
    rows,columns,values=[],[],[]
    for col,state in enumerate(basis):
        for quark in range(3):
            color=color_pair_matrix("qqqg",quark,3)
            for kg in range(2,state[quark]+state[3],2):
                transfer=(kg-state[3])//2
                if transfer==0:
                    continue
                out=list(state)
                out[3],out[quark]=kg,state[quark]+state[3]-kg
                out=tuple(out)
                contact=(pair_contact_block if _contact_kernel is None else _contact_kernel)(state,out,(quark,3),b,ci,co)
                factor=sum(state)/(2*pi*transfer**2)*canonical_gluon_current_factor(kg,state[3])
                block=np.kron(contact,color)*factor
                r,c=np.nonzero(block)
                rows.extend(lookup[out]*no*2+r)
                columns.extend(col*ni*2+c)
                values.extend(block[r,c])
    return sparse.csr_matrix((values,(rows,columns)),shape=(len(basis)*no*2,len(basis)*ni*2))

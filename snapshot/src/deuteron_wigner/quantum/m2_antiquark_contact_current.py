"""Canonical antiquark one-gluon electromagnetic contact, per g_s.

Use the physical antiquark charge e_qbar=-e_q AND color Tbar=-T.T.
Normal ordering the single bilinear supplies one CAR minus, not two:
after factoring both physical charge and Tbar, the signed crossed spin
kernel needs an additional minus. Its denominators remain signed.
This fixes a tree kernel; no loop, regulator or physical-current completion.
"""
from math import sqrt
import numpy as np

from .m2_constrained_fermion_current import (
    ANTI_AXES,ALPHA,bad_field_kernel,_momentum,_field_vertex,
)


def antiquark_one_gluon_current_spin(outgoing,incoming,*,mass_GeV,polarization,
                                     right_plus,left_plus):
    """Physical outgoing/incoming antiquarks, spin order (-1,+1).

    Before CAR reordering the signed fermion endpoints are (-p_in,-p_out).
    At Delta+=0 creation of a gluon uses r_R=-p_in+, r_L=-p_out+ and
    epsilon*. Multiply e_qbar, Tbar, the field factor and Fock map separately.
    The extra minus here prevents double-counting the common CAR sign when
    both physical charge and physical color representation are factored out.
    """
    pf=_momentum(outgoing,positive=True);pi=_momentum(incoming,positive=True)
    bf=bad_field_kernel(-pi,mass_GeV);bi=bad_field_kernel(-pf,mass_GeV)
    right=_field_vertex(right_plus,polarization)
    left=_field_vertex(left_plus,polarization)
    operators=(np.zeros((4,4),complex),bf.conj().T@right+left@bi,
               *((a@right+left@a)/sqrt(2) for a in ALPHA))
    return -np.array([ANTI_AXES.conj().T@op@ANTI_AXES for op in operators]).transpose(0,2,1)

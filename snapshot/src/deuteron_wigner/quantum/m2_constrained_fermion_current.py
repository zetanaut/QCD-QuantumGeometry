"""Canonical local fermion current after eliminating psi_minus, before projection.

Symmetric LF coordinates; components are (+,-,x,y). The good spin axes follow
the project's Lepage--Brodsky phases, in doubled-helicity order (-1,+1).
Free kernels strip 2*sqrt(p_out+ p_in+). Multiply the physical parton charge,
including the antiquark CAR sign, separately. Interaction spin kernels also
strip g_s, color, each gluon's discrete field factor and transverse projection.

Source: Srivastava--Brodsky hep-ph/0011372v2 Eqs.3,21,24.
This supplies ordered canonical current kernels, not a matched finite-basis
physical current. The quadratic product's Wick descendants, loop/current
counterterms, zero modes and projection/gauging corrections remain separate.
"""
from math import pi,sqrt
from operator import index
import numpy as np

from .canonical_lf_fields import GAMMA,GAMMA_PLUS,GAMMA_MINUS


ALPHA=tuple(GAMMA[0]@GAMMA[i] for i in (1,2))
GOOD_AXES=np.array([[0,1],[1,0],[0,1],[-1,0]],complex)/sqrt(2)
ANTI_AXES=1j*GAMMA[2]@GOOD_AXES.conj()
GOOD_AXES.setflags(write=False);ANTI_AXES.setflags(write=False)


def _momentum(value,*,positive=False):
    p=np.asarray(value,float)
    if (p.shape!=(3,) or not np.all(np.isfinite(p)) or p[0]==0
            or (positive and p[0]<0)):
        raise ValueError("finite (p+,px,py) with declared nonzero/positive p+ required")
    return p


def _mass(value):
    m=float(value)
    if not np.isfinite(m) or m<0:
        raise ValueError("finite nonnegative quark mass required")
    return m


def bad_field_kernel(momentum,mass_GeV):
    """B0=(alpha_perp.p_perp+beta*m)/(sqrt(2)*p+), including signed modes.

    Negative momentum labels are Fourier creation modes, not negative-energy
    physical particles. Their physical antiparticle spin axes are ANTI_AXES.
    """
    p=_momentum(momentum);m=_mass(mass_GeV)
    return (ALPHA[0]*p[1]+ALPHA[1]*p[2]+GAMMA[0]*m)/(sqrt(2)*p[0])


def _signed_free(outgoing,incoming,mass,axes):
    bf=bad_field_kernel(outgoing,mass);bi=bad_field_kernel(incoming,mass)
    operators=(np.eye(4),bf.conj().T@bi,
               *( (a@bi+bf.conj().T@a)/sqrt(2) for a in ALPHA))
    return np.array([axes.conj().T@op@axes for op in operators])


def free_fermion_current(outgoing,incoming,*,mass_GeV,species):
    """Number-preserving free spin kernel; physical charge is NOT included.

    For qbar this is [vbar_in gamma^mu v_out]^T in physical endpoint order;
    the minus from normal ordering belongs to the separately supplied charge.
    Thus both species have +I as their charge-stripped plus component.
    """
    pf=_momentum(outgoing,positive=True);pi_=_momentum(incoming,positive=True)
    if species=="q":
        return _signed_free(pf,pi_,mass_GeV,GOOD_AXES)
    if species=="qbar":
        return _signed_free(-pi_,-pf,mass_GeV,ANTI_AXES).transpose(0,2,1)
    raise ValueError("explicit quark or antiquark species required")


def _field_vertex(signed_plus,polarization):
    p=float(signed_plus);eps=np.asarray(polarization,complex)
    if p==0 or not np.isfinite(p) or eps.shape!=(2,) or not np.all(np.isfinite(eps)):
        raise ValueError("finite transverse polarization and signed nonzero denominator required")
    return sum(a*e for a,e in zip(ALPHA,eps))/(sqrt(2)*p)


def one_gluon_current_spin(outgoing,incoming,*,mass_GeV,polarization,
                           right_plus,left_plus):
    """Quark J[1] with one specified A_perp Fourier leg, per g_s, GeV^-1.

    right_plus is the signed momentum of A psi at the right constraint;
    left_plus belongs to the conjugate constraint. For gluon emission at
    Delta+=0 they are p_out+ and p_in+, respectively. Supply epsilon* for
    creation, epsilon for annihilation. No abs(p+), mass conversion, field
    factor, color generator, charge, Fock factor or HO integral is inserted.
    """
    pf=_momentum(outgoing,positive=True);pi_=_momentum(incoming,positive=True)
    bf=bad_field_kernel(pf,mass_GeV);bi=bad_field_kernel(pi_,mass_GeV)
    right=_field_vertex(right_plus,polarization)
    left=_field_vertex(left_plus,polarization)  # same Fourier leg, not its conjugate
    operators=(np.zeros((4,4),complex),bf.conj().T@right+left@bi,
               *((a@right+left@a)/sqrt(2) for a in ALPHA))
    return np.array([GOOD_AXES.conj().T@op@GOOD_AXES for op in operators])


def two_gluon_current_spin(*,left_polarization,right_polarization,left_plus,right_plus):
    """Ordered quark J[2] spin kernel, GeV^-2; only the minus component survives.

    The ordered color product T_left T_right, both gluon leg factors and
    g_s^2 are external. Do not assume the A_left A_right product has already
    been normal ordered or omit its contraction when converting conventions.
    """
    left=_field_vertex(left_plus,left_polarization)
    right=_field_vertex(right_plus,right_polarization)
    result=np.zeros((4,2,2),complex)
    result[1]=GOOD_AXES.conj().T@left@right@GOOD_AXES
    return result


def pair_creation_current_spin(quark,antiquark,*,mass_GeV,polarization,
                               right_plus,left_plus):
    """Ordered b_q^dagger d_qbar^dagger A contact per g_s, GeV^-1.

    The incoming gluon annihilation polarization is NOT conjugated.
    At Delta+=0 the signed denominators are +p_q+ and -p_qbar+.
    Rows/columns label physical quark/antiquark helicities (-1,+1).
    Multiply the quark flavor charge (not a second antiquark charge),
    T^a_ij, the gluon field factor and the declared Fock creation map.
    No CAR permutation of that creation map is inserted here.
    """
    pq=_momentum(quark,positive=True);pb=_momentum(antiquark,positive=True)
    bf=bad_field_kernel(pq,mass_GeV);bi=bad_field_kernel(-pb,mass_GeV)
    right=_field_vertex(right_plus,polarization)
    left=_field_vertex(left_plus,polarization)
    operators=(np.zeros((4,4),complex),bf.conj().T@right+left@bi,
               *((a@right+left@a)/sqrt(2) for a in ALPHA))
    return np.array([GOOD_AXES.conj().T@op@ANTI_AXES for op in operators])


def discrete_gluon_field_factor(gluon_mode_twice):
    """1/sqrt(2 ell_- k+) = 1/sqrt(2 pi k2), no 2P+ mass conversion."""
    if isinstance(gluon_mode_twice,(bool,np.bool_)):
        raise ValueError("positive even doubled gluon mode required")
    k=index(gluon_mode_twice)
    if k<=0 or k%2:
        raise ValueError("positive even doubled gluon mode required")
    return 1/sqrt(2*pi*k)

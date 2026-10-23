"""Canonical instantaneous qqq -> qqqgg contacts, per g_s².

Both the instantaneous-fermion and mixed quark/adjoint-current descendants
of SB Eq.24 are supplied separately and coherently. Sea-pair sources,
QQ interactions and higher six-body exits remain separate/open.
No propagating cubic denominator is substituted for the constrained line.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import pi,sqrt

import numpy as np

from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS
from deuteron_wigner.microscopic.h0.color import ColorSingletBasis,representation_generators
from .m2_canonical_emission import lf_spinor
from .m2_coupled_longitudinal_jacobi import jacobi_frame,longitudinal_basis,oscillator_shell,oscillator_rotation
from .m2_excluded_fock_action import IntermediateSpace
from .m2_excluded_fock_kinematics import color_basis,frame
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS


def _route(incoming,outgoing,quark,sector="qqqgg"):
    incoming,outgoing=tuple(incoming),tuple(outgoing)
    jacobi_frame(incoming)
    f=frame(sector,outgoing)
    if (len(incoming)!=3 or quark not in (0,1,2) or isinstance(quark,(bool,np.bool_))
            or sum(incoming)!=sum(outgoing)):
        raise ValueError("valid qqq -> qqqgg emitter route required")
    collapsed=list(outgoing[:3]);collapsed[quark]+=outgoing[3]+outgoing[4]
    if incoming!=tuple(collapsed):
        raise ValueError("outgoing fractions do not collapse to the incoming quark")
    return f


@lru_cache(maxsize=24,typed=True)
def pair_spin_factor(h_out,h_in,h_left,h_right):
    """ubar gamma.eps_left* gamma+ gamma.eps_right* u / sqrt(2p' 2p)."""
    if any(h not in (-1,1) or isinstance(h,(bool,np.bool_)) for h in (h_out,h_in,h_left,h_right)):
        raise ValueError("explicit signed helicities required")
    u=lf_spinor(1.,0.,0.,.35,h_out)/sqrt(2)
    v=lf_spinor(1.,0.,0.,.35,h_in)/sqrt(2)
    def contracted(h):
        eps=(-np.array([h,1j])/sqrt(2)).conj()
        return GAMMA[1]*eps[0]+GAMMA[2]*eps[1]
    return complex(u.conj()@GAMMA[0]@contracted(h_left)@GAMMA_PLUS@contracted(h_right)@v)


@lru_cache(maxsize=6,typed=True)
def pair_ordered_color(quark,reverse=False):
    """Six complete qqqgg singlets <- one qqq singlet; retain matrix order."""
    if quark not in (0,1,2) or isinstance(quark,(bool,np.bool_)) or not isinstance(reverse,bool):
        raise ValueError("quark slot and explicit color ordering required")
    source=ColorSingletBasis.construct("qqq").tensors.reshape(3,3,3,1)
    target=color_basis("qqqgg")
    t=representation_generators("3")
    raw=np.zeros((3,3,3,8,8,1),complex)
    for a,ta in enumerate(t):
        for b,tb in enumerate(t):
            matrix=tb@ta if reverse else ta@tb
            raw[:,:,:,a,b,:]=np.moveaxis(np.tensordot(matrix,source,axes=(1,quark)),0,quark)
    raw=raw.reshape(1728,1)
    reduced=target.conj().T@raw
    if np.linalg.norm(raw-target@reduced)>2e-12:
        raise ArithmeticError("ordered instantaneous source leaves the complete singlet carrier")
    return reduced


def pair_mass_kernel(incoming,outgoing,quark,helicities):
    """Distinguished two-gluon amplitude per g_s² before Jacobi integration.

    Both outgoing-gluon assignments are summed here; the ordered-slot Fock
    image divides by sqrt(2!) later. Per assignment the coefficient is
    K2/(2*pi*sqrt(k_g1,2 k_g2,2)*r_2). The right-created gluon has been
    removed from the incoming quark, so r_2=p_q,2-k_right,2>0 (odd).
    """
    _route(incoming,outgoing,quark)
    h_out,h_in,h1,h2=tuple(helicities)
    p,k1,k2=incoming[quark],outgoing[3],outgoing[4]
    left=pair_spin_factor(h_out,h_in,h1,h2)*pair_ordered_color(quark)/(p-k2)
    right=pair_spin_factor(h_out,h_in,h2,h1)*pair_ordered_color(quark,True)/(p-k1)
    return sum(incoming)/(2*pi*sqrt(k1*k2))*(left+right)


def adjoint_pair_mass_kernel(incoming,outgoing,quark,helicities):
    """Mixed j_q^+ j_g^+ pair creation, with positive pair transfer.

    j_g=f A_perp d_- A_perp gives -i f^(c a b)(k1-k2)
    eps1* dot eps2*. Combining the two current orderings gives +g²/q+².
    -i f^(c a b)T^c=-(T^a T^b-T^b T^a). No Q0 mode is present since
    q_2=k1+k2>0. This is not the retained qg scattering coefficient.
    """
    _route(incoming,outgoing,quark)
    h_out,h_in,h1,h2=tuple(helicities)
    # Common validation also rejects unphysical Boolean helicities.
    pair_spin_factor(h_out,h_in,h1,h2)
    k1,k2=outgoing[3:]
    eps_dot=(h1*h2-1)/2
    spin=int(h_out==h_in)
    color=pair_ordered_color(quark)-pair_ordered_color(quark,True)
    return -sum(incoming)/(pi*sqrt(k1*k2))*(k1-k2)/(k1+k2)**2*eps_dot*spin*color


def pair_emission_frame(incoming,outgoing,quark,*,sector="qqqgg"):
    """Global Q_out -> inherited qqq Q's and two local splitting coordinates."""
    f=_route(incoming,outgoing,quark,sector);parent=jacobi_frame(incoming)
    collapse=np.column_stack((np.eye(3),np.eye(3)[:,quark],np.eye(3)[:,quark]))
    x,y,z=f.fractions[[quark,3,4]]
    first=np.zeros(5);first[quark]=sqrt(y/(x*(x+y)));first[3]=-sqrt(x/(y*(x+y)))
    second=np.zeros(5)
    second[[quark,3]]=sqrt(z/((x+y)*(x+y+z)))
    second[4]=-sqrt((x+y)/(z*(x+y+z)))
    rotation=np.vstack((parent.forward[:-1]@collapse,first,second))@f.inverse[:,:-1]
    if np.linalg.norm(rotation@rotation.T-np.eye(4))>2e-12:
        raise ArithmeticError("double-emission Jacobi map is not orthogonal")
    return rotation


def pair_contact_orbital(incoming,outgoing,quark,in_cutoff,out_cutoff,b_GeV,*,sector="qqqgg"):
    """Two independent constant-kernel HO integrals, not one pair contact.

    The normalized measure factor is sqrt(x_q*x_g1*x_g2/X_parent).
    Each new m=0 canonical circular oscillator contributes b/sqrt(pi);
    this is exact for every radial n, not an assumed radial suppression.
    """
    f=_route(incoming,outgoing,quark,sector)
    b=float(b_GeV)
    if not np.isfinite(b) or b<=0:
        raise ValueError("positive finite width required")
    inc=oscillator_shell(2,in_cutoff);out=oscillator_shell(4,out_cutoff)
    lookup={s:i for i,s in enumerate(inc)}
    adapted=np.zeros((len(out),len(inc)))
    for row,s in enumerate(out):
        inherited=s[:2]+s[4:6]
        if inherited in lookup and s[2]==s[6] and s[3]==s[7]:
            adapted[row,lookup[inherited]]=b*b/pi
    x,y,z=f.fractions[[quark,3,4]]
    return sqrt(x*y*z/(x+y+z))*oscillator_rotation(
        pair_emission_frame(incoming,outgoing,quark,sector=sector),out_cutoff).T@adapted


@dataclass
class InstantaneousPairSource:
    retained_pilot: object
    space: IntermediateSpace
    retained_columns: np.ndarray
    matrix_per_gs2: np.ndarray
    source_name: str


def build_instantaneous_pair_source(pilot,*,out_cutoff):
    """Explicit quartic IF image of the retained qqq columns, with Bose factor."""
    return _build_pair_source(pilot,out_cutoff,pair_mass_kernel,"instantaneous fermion")


def build_adjoint_pair_source(pilot,*,out_cutoff):
    """Explicit mixed quark/adjoint-current pair image, with Bose factor."""
    return _build_pair_source(pilot,out_cutoff,adjoint_pair_mass_kernel,"mixed quark/adjoint current")


def build_combined_pair_source(pilot,*,out_cutoff):
    """Coherent sum of both qqq -> qqqgg normal-ordered canonical contacts."""
    return _build_pair_source(pilot,out_cutoff,
        lambda *args: pair_mass_kernel(*args)+adjoint_pair_mass_kernel(*args),
        "coherent instantaneous fermion + mixed quark/adjoint current")


def _build_pair_source(pilot,out_cutoff,kernel,source_name,_retained_vectors=None,_orbital_kernel=None):
    space=IntermediateSpace("qqqgg",pilot.total_k2,out_cutoff,pilot.target,pilot.jz_twice)
    if _retained_vectors is None and pilot.cutoff>out_cutoff:
        raise ValueError("excluded shell must include the retained shell")
    incoming=longitudinal_basis("qqq",pilot.total_k2)
    ins={s:i for i,s in enumerate(incoming)}
    no=len(oscillator_shell(2,pilot.cutoff))
    nq=pilot.bases[0].shape[1]
    from .m2_source_vectors import SourceImage, source_tensor, image_symmetry_tolerance
    source=(pilot.bases[0].toarray().reshape(len(incoming),no,1,64,nq)
            if _retained_vectors is None else source_tensor(pilot,"qqq",_retained_vectors))
    count=source.shape[-1]
    sf=np.array(SPIN_FLAVOR_BASIS)
    image=np.zeros((space.dimension,count),complex)
    for li,out in enumerate(space.longitudinal):
        for emitter in range(3):
            inc=list(out[:3]);inc[emitter]+=out[3]+out[4];inc=tuple(inc)
            orbital=(pair_contact_orbital if _orbital_kernel is None else _orbital_kernel)(
                inc,out,emitter,pilot.cutoff,out_cutoff,pilot.width_GeV)
            for h,h1,h2 in product((-1,1),repeat=3):
                color=kernel(inc,out,emitter,(h,h,h1,h2))
                if not np.any(color):
                    continue
                selected=np.flatnonzero(2*(sf[:,emitter]%2)-1==h)
                value=np.take(source[ins[inc]],selected,axis=2)
                amplitude=np.einsum("oi,ab,ibsk->oask",orbital,color,value,optimize=True)/sqrt(2)
                spins=(selected*2+(h1+1)//2)*2+(h2+1)//2
                space.add(image,li,spins,amplitude)
    if np.linalg.norm(space.statistics_project(image)-image)>image_symmetry_tolerance(
            image,vector_action=_retained_vectors is not None):
        raise ArithmeticError("instantaneous pair image violates Bose symmetry")
    if _retained_vectors is not None:
        return SourceImage(pilot,space,2,image,{source_name:image})
    return InstantaneousPairSource(pilot,space,np.arange(nq),image,source_name)


def apply_pair_source(pilot,vectors,*,out_cutoff,component):
    """Exact vector action, retaining separate IF/IG or their coherent sum."""
    from .m2_source_vectors import validate_source_vectors
    kernels={"IF":pair_mass_kernel,"IG":adjoint_pair_mass_kernel,
             "combined":lambda *args: pair_mass_kernel(*args)+adjoint_pair_mass_kernel(*args)}
    if component not in kernels:
        raise ValueError("explicit IF, IG or combined pair source required")
    return _build_pair_source(pilot,out_cutoff,kernels[component],component,
                              _retained_vectors=validate_source_vectors(pilot,vectors))

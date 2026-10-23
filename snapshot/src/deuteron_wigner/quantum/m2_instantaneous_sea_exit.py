"""Normal-ordered canonical j_q^+ j_q^+ qqq -> qqqq-qbar exit, per gs².

The valence-number and pair-creation currents are both explicitly present.
The two current orders add; the four-quark Fock creation factor is 2 A4.
This is not the cubic g -> q qbar exit from an incoming qqqg sector.
"""
from functools import lru_cache
from math import pi,sqrt
import numpy as np

from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS
from deuteron_wigner.microscopic.h0.color import ColorSingletBasis,representation_generators
from .m2_canonical_emission import lf_spinor
from .m2_coupled_longitudinal_jacobi import longitudinal_basis,oscillator_shell
from .m2_excluded_fock_action import IntermediateSpace
from .m2_excluded_fock_kinematics import color_basis
from .m2_excluded_fock_vertices import antiquark_spinor
from .m2_instantaneous_pair_exit import _route,pair_contact_orbital,InstantaneousPairSource


@lru_cache(maxsize=4,typed=True)
def sea_pair_spin_factor(quark_helicity,antiquark_helicity):
    """ubar gamma+ v / sqrt(2p_q+ 2p_anti+), canonical physical antiparticle h."""
    for h in (quark_helicity,antiquark_helicity):
        if h not in (-1,1) or isinstance(h,(bool,np.bool_)):
            raise ValueError("explicit signed pair helicities required")
    if quark_helicity==antiquark_helicity:
        return 0j
    u=lf_spinor(1.,0.,0.,.35,quark_helicity)/sqrt(2)
    v=antiquark_spinor(1.,0.,0.,.35,antiquark_helicity)/sqrt(2)
    return complex(u.conj()@GAMMA[0]@GAMMA_PLUS@v)


@lru_cache(maxsize=3,typed=True)
def sea_pair_color(emitter):
    """Complete three-singlet qqqq-qbar color vector, with appended pair slots."""
    if emitter not in (0,1,2) or isinstance(emitter,(bool,np.bool_)):
        raise ValueError("valence quark emitter required")
    source=ColorSingletBasis.construct("qqq").tensors.reshape(3,3,3,1)
    raw=np.zeros((3,3,3,3,3,1),complex)
    for t in representation_generators("3"):
        valence=np.moveaxis(np.tensordot(t,source,axes=(1,emitter)),0,emitter)
        raw+=np.einsum("ijkn,ab->ijkabn",valence,t)
    raw=raw.reshape(243,1)
    target=color_basis("qqqq-qbar")
    reduced=target.conj().T@raw
    if np.linalg.norm(raw-target@reduced)>2e-12:
        raise ArithmeticError("canonical sea pair leaves the full singlet carrier")
    return reduced


def sea_pair_mass_kernel(incoming,outgoing,emitter,helicities):
    """K2*2/[pi (k_q+k_anti)²] times valence/pair spin currents and color.

    The positive pair transfer is even; Q0 is not invoked. Both external
    fermion current normalizations are included. Pair h must sum to zero.
    """
    _route(incoming,outgoing,emitter,"qqqq-qbar")
    h_out,h_in,hq,ha=tuple(helicities)
    if any(h not in (-1,1) or isinstance(h,(bool,np.bool_)) for h in (h_out,h_in)):
        raise ValueError("explicit valence helicities required")
    transfer=outgoing[3]+outgoing[4]
    return (2*sum(incoming)/(pi*transfer**2)*int(h_out==h_in)
            *sea_pair_spin_factor(hq,ha)*sea_pair_color(emitter))


def build_instantaneous_sea_source(pilot,*,out_cutoff,_retained_vectors=None,_orbital_kernel=None):
    """Coherent u ubar+d dbar source, inherited A3 followed by normalized 2 A4."""
    space=IntermediateSpace("qqqq-qbar",pilot.total_k2,out_cutoff,pilot.target,pilot.jz_twice)
    if _retained_vectors is None and pilot.cutoff>out_cutoff:
        raise ValueError("excluded shell must include the retained shell")
    incoming=longitudinal_basis("qqq",pilot.total_k2)
    ins={s:i for i,s in enumerate(incoming)}
    no=len(oscillator_shell(2,pilot.cutoff));nq=pilot.bases[0].shape[1]
    from .m2_source_vectors import SourceImage, source_tensor
    source=(pilot.bases[0].toarray().reshape(len(incoming),no,1,64,nq)
            if _retained_vectors is None else source_tensor(pilot,"qqq",_retained_vectors))
    image=np.zeros((space.dimension,source.shape[-1]),complex)
    for li,out in enumerate(space.longitudinal):
        for emitter in range(3):
            inc=list(out[:3]);inc[emitter]+=out[3]+out[4];inc=tuple(inc)
            orbital=(pair_contact_orbital if _orbital_kernel is None else _orbital_kernel)(inc,out,emitter,pilot.cutoff,out_cutoff,pilot.width_GeV,
                                        sector="qqqq-qbar")
            for hq,ha in ((1,-1),(-1,1)):
                # Valence good current is the identity on its helicity.
                color=sea_pair_mass_kernel(inc,out,emitter,(1,1,hq,ha))
                amplitude=np.einsum("oi,ab,ibsk->oask",orbital,color,source[ins[inc]],optimize=True)
                for flavor in (0,1):
                    q,anti=2*flavor+(hq+1)//2,2*flavor+(ha+1)//2
                    spins=(np.arange(64)*4+q)*4+anti
                    space.add(image,li,spins,amplitude)
    image=2*space.statistics_project(image)
    if _retained_vectors is not None:
        return SourceImage(pilot,space,2,image,{"sea_pair":image})
    return InstantaneousPairSource(pilot,space,np.arange(nq),image,
                                    "instantaneous quark-current sea pair")


def apply_instantaneous_sea_source(pilot,vectors,*,out_cutoff):
    """Coherent sea-flavor vector action with exact four-quark antisymmetry."""
    from .m2_source_vectors import validate_source_vectors
    return build_instantaneous_sea_source(pilot,out_cutoff=out_cutoff,
                                         _retained_vectors=validate_source_vectors(pilot,vectors))

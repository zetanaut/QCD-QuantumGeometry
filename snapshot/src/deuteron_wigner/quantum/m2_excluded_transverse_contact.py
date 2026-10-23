"""Canonical qqqg -> Q5 transverse-current contacts, per g_s.

Includes quark emission into qqqgg and gluon conversion into a sea pair.
The QCD three-gluon vertex has no tree-level local electromagnetic contact.
These are J_T/(2P+) raw endpoint maps, not yet a complete H2-matched current.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import pi,sqrt

import numpy as np

from deuteron_wigner.bridge.modes.core import polarization
from .m2_canonical_good_current import parton_charge
from .m2_constrained_fermion_current import (
    one_gluon_current_spin,pair_creation_current_spin,discrete_gluon_field_factor,
)
from .m2_coupled_longitudinal_jacobi import oscillator_shell,oscillator_rotation
from .m2_coupled_pilot import _displacement
from .m2_excluded_fock_kinematics import (
    frame,split_frame,split_routes,longitudinal_states,color_source,
)


@lru_cache(maxsize=1024)
def split_contact_orbital(route,in_cutoff,out_cutoff,b_GeV,transfer):
    """Local relative coincidence times exact recoiling-parent overlap, GeV."""
    if route.channel not in ("q_to_qg","g_to_pair"):
        raise ValueError("quark emission or sea-pair current channel required")
    delta=np.asarray(transfer,float);b=float(b_GeV)
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)) or not np.isfinite(b) or b<=0:
        raise ValueError("finite transfer and positive width required")
    inc,out=oscillator_shell(3,in_cutoff),oscillator_shell(4,out_cutoff)
    fi,fo=frame("qqqg",route.incoming),frame(route.sector,route.outgoing)
    x,y=fo.fractions[list(route.daughters)]
    weight=sqrt(x*y/(x+y))*b/sqrt(pi)
    shift=fi.forward[:-1,route.parent,None]*delta[None,:]
    plus=(shift[:,0]-1j*shift[:,1])/(2*b)
    alpha=np.r_[plus,plus.conj()]
    adapted=np.zeros((len(out),len(inc)),complex)
    for row,o in enumerate(out):
        if o[3]!=o[7]:
            continue
        parent=o[:3]+o[4:7]
        for col,i in enumerate(inc):
            adapted[row,col]=weight*np.prod([_displacement(a,c,z) for a,c,z in zip(parent,i,alpha)])
    value=oscillator_rotation(split_frame(route),out_cutoff).T@adapted
    value.setflags(write=False)
    return value


def _add_current(space,output,li,spins,amplitudes):
    """Project onto the explicitly requested final Jz, not a scalar-H add.

    A Cartesian current need not preserve Jz. Components outside this target
    block are not declared zero: they belong to other requested endpoints.
    Charge/Iz conservation is nevertheless checked before gathering.
    """
    spin=np.asarray(spins)
    labels=space.spins[spin]
    signs=np.array([1,1,1,0,0] if space.sector=="qqqgg" else [1,1,1,1,-1])
    iz=np.sum(np.where(labels<2,1,-1)*signs,axis=1)
    bad=iz!=(1 if space.target=="PROTON" else -1)
    if np.any(amplitudes[:,:,bad]!=0):
        raise ArithmeticError("electromagnetic current changed exact flavor charge")
    raw=(((li*len(space.orbitals)+np.arange(len(space.orbitals))[:,None,None])
          *space.ncolor+np.arange(space.ncolor)[None,:,None])*space.nspin+spin[None,None,:])
    rows=space.lookup[raw]
    allowed=rows>=0
    output[rows[allowed]]+=amplitudes[allowed]


@dataclass(frozen=True)
class ExcludedTransverseContact:
    incoming_pilot: object
    outgoing_space: object
    retained_columns: np.ndarray
    transfer: tuple
    total_pplus_GeV: float
    charge: str
    channel: str
    current_per_gs: np.ndarray  # (x/y, raw final rows, explicit qqqg columns)


def build_excluded_transverse_contact(pilot,space,transfer,*,total_pplus_GeV,charge):
    """Coherent current endpoint, with the Hamiltonian's same Fock factors.

    The supplied final Q5 space determines target Jz and all raw coordinates.
    Its Jz may differ from the incoming pilot. Absorption is the reversed-
    transfer endpoint's adjoint, not this matrix at unchanged transfer.
    """
    if space.sector not in ("qqqgg","qqqq-qbar"):
        raise ValueError("explicit five-particle current endpoint required")
    for name in ("total_k2","target"):
        if getattr(space,name)!=getattr(pilot,name):
            raise ValueError("current endpoints disagree on "+name)
    delta=tuple(float(v) for v in transfer);pplus=float(total_pplus_GeV)
    if len(delta)!=2 or not np.all(np.isfinite(delta)) or not np.isfinite(pplus) or pplus<=0:
        raise ValueError("finite transverse transfer and explicit positive P+ required")
    parton_charge(np.arange(4),"q",charge)
    incoming=longitudinal_states("qqqg",pilot.total_k2)
    ni=len(oscillator_shell(3,pilot.cutoff))
    ranks=[u.shape[1] for u in pilot.bases]
    columns=np.arange(ranks[0],sum(ranks));count=len(columns)
    source=np.zeros((len(incoming),ni,2,64,2,count),complex)
    offset=0
    for hg,u in enumerate(pilot.bases[1:]):
        n=u.shape[1]
        source[:,:,:,:,hg,offset:offset+n]=u.toarray().reshape(len(incoming),ni,2,64,n)
        offset+=n
    ins,outs=({s:i for i,s in enumerate(states)} for states in (incoming,space.longitudinal))
    sf=np.array(list(product(range(4),repeat=3)))
    channel="q_to_qg" if space.sector=="qqqgg" else "g_to_pair"
    image=np.zeros((2,space.dimension,count),complex)
    for route in split_routes(pilot.total_k2,channel):
        color=color_source(channel,route.parent)
        orbital=split_contact_orbital(route,pilot.cutoff,space.cutoff,pilot.width_GeV,delta)
        x,y=frame(space.sector,route.outgoing).fractions[list(route.daughters)]
        pq,pb,parent=x*pplus,y*pplus,(x+y)*pplus
        for hg in (-1,1):
            if channel=="q_to_qg":
                eps=polarization(pb,0.,0.,hg).conj()[2:]
                spin=one_gluon_current_spin((pq,0.,0.),(parent,0.,0.),
                    mass_GeV=0.,polarization=eps,right_plus=pq,left_plus=parent)[2:]
                spin*=discrete_gluon_field_factor(route.outgoing[4])
                for h0,h1 in product(range(2),repeat=2):
                    selected=np.flatnonzero(sf[:,route.parent]%2==h0)
                    changed=sf[selected].copy()
                    changed[:,route.parent]=2*(changed[:,route.parent]//2)+h1
                    out_sf=np.ravel_multi_index(changed.T,(4,4,4))
                    weight=parton_charge(sf[selected,route.parent],"q",charge)
                    for old_hg in range(2):
                        value=np.take(source[ins[route.incoming],:,:,:,old_hg],selected,axis=2)
                        amp=np.einsum("oi,ab,ibsk->oask",orbital,color,value,optimize=True)
                        amp*=weight[None,None,:,None]
                        spins=(out_sf*2+old_hg)*2+(hg+1)//2
                        for j in (0,1):
                            if spin[j,h1,h0]!=0:
                                _add_current(space,image[j],outs[route.outgoing],spins,spin[j,h1,h0]*amp)
            else:
                eps=polarization(parent,0.,0.,hg)[2:]
                spin=pair_creation_current_spin((pq,0.,0.),(pb,0.,0.),mass_GeV=0.,
                    polarization=eps,right_plus=pq,left_plus=-pb)[2:]
                spin*=discrete_gluon_field_factor(route.incoming[3])
                value=source[ins[route.incoming],:,:,:,(hg+1)//2]
                amp=np.einsum("oi,ab,ibsk->oask",orbital,color,value,optimize=True)
                for h1,h2,flavor in product(range(2),repeat=3):
                    q,anti=2*flavor+h1,2*flavor+h2
                    weight=parton_charge(np.array([q]),"q",charge)[0]
                    spins=(np.arange(64)*4+q)*4+anti
                    for j in (0,1):
                        if spin[j,h1,h2]!=0:
                            _add_current(space,image[j],outs[route.outgoing],spins,
                                         weight*spin[j,h1,h2]*amp)
    fock=sqrt(2) if channel=="q_to_qg" else 2.
    image=np.array([fock*space.statistics_project(value) for value in image])
    return ExcludedTransverseContact(pilot,space,columns,delta,pplus,charge,channel,image)

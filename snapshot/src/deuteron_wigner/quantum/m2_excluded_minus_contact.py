"""Canonical one-gluon J-minus actions P↔Q5 and Q5↔Q6, per gs.

All spin/momentum, charge, color and statistics factors are explicit.
These supply no two-gluon, Wick, loop or regulator current and are not
automatically included in a physical H2 current.
"""
from dataclasses import dataclass
from itertools import product
from math import sqrt
import numpy as np

from .m2_canonical_good_current import parton_charge
from .m2_constrained_fermion_current import discrete_gluon_field_factor
from .m2_coupled_longitudinal_jacobi import oscillator_shell
from .m2_excluded_fock_kinematics import (
    frame as frame5,split_routes,split_frame,longitudinal_states,color_source,
)
from .m2_six_body_kinematics import frame as frame6
from .m2_six_body_vertices import split_frame as split_frame6,color_source as color_source6
from .m2_six_body_cubic import five_statistics_project,SixBodyCubicAction
from .m2_six_body_transverse_contact import SixBodyTransverseContact,_block,_add
from .m2_excluded_transverse_contact import _add_current
from .m2_one_gluon_minus_contact import one_gluon_minus_orbital


@dataclass(frozen=True)
class ExcludedMinusContact:
    incoming_pilot: object
    outgoing_space: object
    retained_columns: np.ndarray
    transfer: tuple
    total_pplus_GeV: float
    charge: str
    channel: str
    current_per_gs: np.ndarray


def build_excluded_minus_contact(pilot,space,transfer,*,total_pplus_GeV,charge,
                                 mean_total_transverse=(0.,0.)):
    """P's qqqg block to explicit Q5 raw coordinates, including spin flips."""
    if space.sector not in ("qqqgg","qqqq-qbar"):
        raise ValueError("explicit five-particle endpoint required")
    if any(getattr(space,k)!=getattr(pilot,k) for k in ("total_k2","target")):
        raise ValueError("current endpoints disagree on momentum/target")
    delta=tuple(float(x) for x in transfer);pp=float(total_pplus_GeV)
    if len(delta)!=2 or not np.all(np.isfinite(np.r_[delta,pp])) or pp<=0:
        raise ValueError("finite transfer and positive total P+ required")
    parton_charge(np.arange(4),"q",charge)
    incoming=longitudinal_states("qqqg",pilot.total_k2);ni=len(oscillator_shell(3,pilot.cutoff))
    ranks=[u.shape[1] for u in pilot.bases];columns=np.arange(ranks[0],sum(ranks));count=len(columns)
    source=np.zeros((len(incoming),ni,2,64,2,count),complex);offset=0
    for hg,u in enumerate(pilot.bases[1:]):
        n=u.shape[1];source[:,:,:,:,hg,offset:offset+n]=u.toarray().reshape(len(incoming),ni,2,64,n);offset+=n
    ins,outs=({s:i for i,s in enumerate(states)} for states in (incoming,space.longitudinal))
    sf=np.array(list(product(range(4),repeat=3)))
    channel="q_to_qg" if space.sector=="qqqgg" else "g_to_pair"
    image=np.zeros((space.dimension,count),complex)
    for route in split_routes(pilot.total_k2,channel):
        color=color_source(channel,route.parent);fi=frame5("qqqg",route.incoming)
        x,y=frame5(space.sector,route.outgoing).fractions[list(route.daughters)]
        for hg in (-1,1):
            orbital=one_gluon_minus_orbital(fi,route.parent,x,y,split_frame(route),
                pilot.cutoff,space.cutoff,delta,channel=channel,b_GeV=pilot.width_GeV,
                total_pplus_GeV=pp,mass_GeV=pilot.quark_mass_GeV,
                gluon_helicity=hg,mean_total_transverse=mean_total_transverse)
            mode=route.outgoing[4] if channel=="q_to_qg" else route.incoming[3]
            orbital*=discrete_gluon_field_factor(mode)
            if channel=="q_to_qg":
                for hi,hf in product(range(2),repeat=2):
                    selected=np.flatnonzero(sf[:,route.parent]%2==hi)
                    changed=sf[selected].copy();changed[:,route.parent]=2*(changed[:,route.parent]//2)+hf
                    out_sf=np.ravel_multi_index(changed.T,(4,4,4))
                    charge_weights=parton_charge(sf[selected,route.parent],"q",charge)
                    for old_hg in range(2):
                        value=np.take(source[ins[route.incoming],:,:,:,old_hg],selected,axis=2)
                        amp=np.einsum("oi,ab,ibsk->oask",orbital[hf,hi],color,value,optimize=True)
                        amp*=charge_weights[None,None,:,None]
                        spins=(out_sf*2+old_hg)*2+(hg+1)//2
                        _add_current(space,image,outs[route.outgoing],spins,amp)
            else:
                value=source[ins[route.incoming],:,:,:,(hg+1)//2]
                for hf,ha,flavor in product(range(2),repeat=3):
                    q,anti=2*flavor+hf,2*flavor+ha
                    amp=np.einsum("oi,ab,ibsk->oask",orbital[hf,ha],color,value,optimize=True)
                    amp*=float(parton_charge(q,"q",charge))
                    spins=(np.arange(64)*4+q)*4+anti
                    _add_current(space,image,outs[route.outgoing],spins,amp)
    fock=sqrt(2) if channel=="q_to_qg" else 2.
    return ExcludedMinusContact(pilot,space,columns,delta,pp,charge,channel,
                                fock*space.statistics_project(image))


class SixBodyMinusContact:
    """Factorized Q5→Q6 linear minus-current; no raw-state amplitude pruning."""
    physical=False
    complete_H2_matched_current=False

    def __init__(self,incoming,outgoing,transfer,*,b_GeV,total_pplus_GeV,mass_GeV,
                 charge,mean_total_transverse=(0.,0.)):
        # Reuse only the checked endpoint/route inventory, never a transverse
        # spin kernel or its momentum-independent projection.
        routing=SixBodyTransverseContact(incoming,outgoing,transfer,b_GeV=b_GeV,
                                         total_pplus_GeV=total_pplus_GeV,charge=charge)
        self.incoming=routing.incoming;self.outgoing=routing.outgoing
        self.routes=routing.routes;self.transfer=routing.transfer
        self.b=routing.b;self.P=routing.P;self.charge=charge
        self.mass=float(mass_GeV);self.mean=tuple(float(x) for x in mean_total_transverse)
        if len(self.mean)!=2 or not np.all(np.isfinite(np.r_[self.mass,self.mean])) or self.mass<0:
            raise ValueError("nonnegative finite mass and finite mean transverse momentum required")

    def _terms(self):
        for s,t,route,fock in self.routes:
            source,target=self.incoming[s],self.outgoing[t]
            li=source.longitudinal.index(route.incoming);lo=target.longitudinal.index(route.outgoing)
            fi=frame5(s,route.incoming);x,y=frame6(t,route.outgoing).fractions[list(route.daughters)]
            color=color_source6(route);spins=source.spins
            for hg in (-1,1):
                orbital=one_gluon_minus_orbital(fi,route.parent,x,y,split_frame6(route),
                    source.cutoff,target.cutoff,self.transfer,channel=route.channel,b_GeV=self.b,
                    total_pplus_GeV=self.P,mass_GeV=self.mass,
                    gluon_helicity=hg,mean_total_transverse=self.mean)
                mode=route.incoming[4] if route.channel=="g_to_pair" else route.outgoing[5]
                orbital*=fock*discrete_gluon_field_factor(mode)
                if route.channel!="g_to_pair":
                    species="qbar" if route.channel=="anti_to_antig" else "q"
                    for hi,hf in product(range(2),repeat=2):
                        selected=np.flatnonzero(spins[:,route.parent]%2==hi)
                        changed=spins[selected].copy()
                        changed[:,route.parent]=2*(changed[:,route.parent]//2)+hf
                        out=np.column_stack((changed,np.full(len(selected),(hg+1)//2)))
                        out_spins=np.ravel_multi_index(out.T,target.spin_dims)
                        weight=parton_charge(spins[selected,route.parent],species,self.charge)
                        yield s,t,li,lo,orbital[hf,hi],color,selected,out_spins,weight
                else:
                    selected=np.flatnonzero(2*spins[:,4]-1==hg);changed=spins[selected]
                    for hf,ha,flavor in product(range(2),repeat=3):
                        q,anti=2*flavor+hf,2*flavor+ha
                        out=np.column_stack((changed[:,:3],np.full(len(selected),q),
                            np.full(len(selected),anti),changed[:,3]))
                        out_spins=np.ravel_multi_index(out.T,target.spin_dims)
                        weight=np.full(len(selected),float(parton_charge(q,"q",self.charge)))
                        yield s,t,li,lo,orbital[hf,ha],color,selected,out_spins,weight

    def apply(self,columns):
        n=SixBodyCubicAction._validate(self.incoming,columns)
        source={s:five_statistics_project(space,columns[s]) for s,space in self.incoming.items()}
        result={s:np.zeros((space.dimension,n),complex) for s,space in self.outgoing.items()}
        for s,t,li,lo,orb,color,si,sf,weight in self._terms():
            block=_block(self.incoming[s],li,si,source[s])
            out=np.einsum("oi,ab,ibsk->oask",orb,color,block,optimize=True)
            out*=weight[None,None,:,None]
            _add(self.outgoing[t],lo,sf,out,result[t])
        return {s:space.statistics_project(result[s]) for s,space in self.outgoing.items()}

    def adjoint(self,columns):
        """Exact same-transfer adjoint; physical absorption uses reversed Delta."""
        n=SixBodyCubicAction._validate(self.outgoing,columns)
        source={s:space.statistics_project(columns[s]) for s,space in self.outgoing.items()}
        result={s:np.zeros((space.dimension,n),complex) for s,space in self.incoming.items()}
        for s,t,li,lo,orb,color,si,sf,weight in self._terms():
            block=_block(self.outgoing[t],lo,sf,source[t])
            block*=weight[None,None,:,None].conj()
            back=np.einsum("oi,ab,oask->ibsk",orb.conj(),color.conj(),block,optimize=True)
            _add(self.incoming[s],li,si,back,result[s])
        return {s:five_statistics_project(space,result[s]) for s,space in self.incoming.items()}

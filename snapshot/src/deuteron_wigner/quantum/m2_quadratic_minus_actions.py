"""All normal-ordered quadratic minus-current blocks needed through H2 g^4.

P↔Q5, P↔Q6 and QQ actions from the canonical two-transverse-gluon fermion
bilinear. Every endpoint/charge/Fock factor is explicit. Wick, regulator,
propagating-loop and physical finite terms are not silently supplied.
"""
from dataclasses import dataclass
from itertools import product
from math import sqrt,pi
import numpy as np

from .m2_canonical_good_current import parton_charge
from .m2_canonical_instantaneous_fermion import transverse_spin_factor
from .m2_excluded_instantaneous_fermion import ordered_five_qg_color
from .m2_coupled_longitudinal_jacobi import oscillator_shell,longitudinal_basis
from .m2_excluded_fock_kinematics import longitudinal_states
from .m2_six_body_quartic import direct_routes
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS
from .m2_excluded_transverse_contact import _add_current
from .m2_six_body_transverse_contact import _add,_block
from .m2_six_body_cubic import five_statistics_project,SixBodyCubicAction
from .m2_quadratic_minus_fock import (
    pair_emission_minus_kernel,pair_emission_minus_orbital,direct_six_minus_kernel,
    direct_six_minus_orbital,conversion_minus_kernel,five_displaced_pair_contact,
)


@dataclass(frozen=True)
class QuadraticMinusEndpoint:
    pilot: object
    space: object
    retained_columns: np.ndarray
    current_per_gs2: np.ndarray


def _endpoint(pilot,space,transfer,P,charge):
    if any(getattr(pilot,k)!=getattr(space,k) for k in ("total_k2","target")):
        raise ValueError("current endpoints disagree on momentum/target")
    d=tuple(float(x) for x in transfer);p=float(P)
    if len(d)!=2 or not np.all(np.isfinite(np.r_[d,p])) or p<=0:
        raise ValueError("finite transfer and positive P+ required")
    parton_charge(np.arange(4),"q",charge)
    return d,p


def build_pair_minus_endpoint(pilot,space,transfer,*,total_pplus_GeV,charge):
    """Retained qqq→qqqgg two-gluon current, divided by sqrt(2!) once."""
    if space.sector!="qqqgg":raise ValueError("explicit qqqgg endpoint required")
    delta,P=_endpoint(pilot,space,transfer,total_pplus_GeV,charge)
    incoming=longitudinal_basis("qqq",pilot.total_k2);ins={s:i for i,s in enumerate(incoming)}
    ni=len(oscillator_shell(2,pilot.cutoff));nq=pilot.bases[0].shape[1]
    source=pilot.bases[0].toarray().reshape(len(incoming),ni,1,64,nq)
    sf=np.array(SPIN_FLAVOR_BASIS);image=np.zeros((space.dimension,nq),complex)
    for li,out in enumerate(space.longitudinal):
        for q in range(3):
            inc=list(out[:3]);inc[q]+=out[3]+out[4];inc=tuple(inc)
            orbital=pair_emission_minus_orbital(inc,out,q,pilot.cutoff,space.cutoff,pilot.width_GeV,delta)
            for hf,hi,h1,h2 in product((-1,1),repeat=4):
                color=pair_emission_minus_kernel(inc,out,q,(hf,hi,h1,h2),total_pplus_GeV=P)
                selected=np.flatnonzero(2*(sf[:,q]%2)-1==hi)
                changed=sf[selected].copy();changed[:,q]=2*(changed[:,q]//2)+(hf+1)//2
                outsf=np.ravel_multi_index(changed.T,(4,4,4))
                value=np.take(source[ins[inc]],selected,axis=2)
                amplitude=np.einsum("oi,ab,ibsk->oask",orbital,color,value,optimize=True)/sqrt(2)
                amplitude*=parton_charge(sf[selected,q],"q",charge)[None,None,:,None]
                spins=(outsf*2+(h1+1)//2)*2+(h2+1)//2
                _add_current(space,image,li,spins,amplitude)
    return QuadraticMinusEndpoint(pilot,space,np.arange(nq),space.statistics_project(image))


def build_six_minus_endpoint(pilot,space,transfer,*,component,total_pplus_GeV,charge):
    """Retained qqqg→Q6 quadratic current; source Fock factors already included."""
    expected={"valence_IF":"qqqggg","gluon_IF_pair":"qqqq-qbar-g"}
    if expected.get(component)!=space.sector:raise ValueError("explicit matching IF component/sector required")
    delta,P=_endpoint(pilot,space,transfer,total_pplus_GeV,charge)
    incoming=longitudinal_states("qqqg",pilot.total_k2);ni=len(oscillator_shell(3,pilot.cutoff))
    ranks=[u.shape[1] for u in pilot.bases];nq=ranks[0];count=sum(ranks[1:])
    source=np.zeros((len(incoming),ni,2,64,2,count),complex);offset=0
    for hg,u in enumerate(pilot.bases[1:]):
        n=u.shape[1];source[:,:,:,:,hg,offset:offset+n]=u.toarray().reshape(len(incoming),ni,2,64,n);offset+=n
    ins={s:i for i,s in enumerate(incoming)};outs={s:i for i,s in enumerate(space.longitudinal)}
    sf=np.array(SPIN_FLAVOR_BASIS);image=np.zeros((space.dimension,count),complex)
    for route in direct_routes(pilot.total_k2,component):
        orbital=direct_six_minus_orbital(route,pilot.cutoff,space.cutoff,pilot.width_GeV,delta)
        for hs in product((-1,1),repeat=4):
            color=direct_six_minus_kernel(route,hs,total_pplus_GeV=P)
            if component=="valence_IF":
                hf,hi,h1,h2=hs;q=route.emitter
                selected=np.flatnonzero(2*(sf[:,q]%2)-1==hi)
                changed=sf[selected].copy();changed[:,q]=2*(changed[:,q]//2)+(hf+1)//2
                for hg in range(2):
                    value=np.take(source[ins[route.incoming],:,:,:,hg],selected,axis=2)
                    amp=np.einsum("oi,ab,ibsk->oask",orbital,color,value,optimize=True)
                    amp*=parton_charge(sf[selected,q],"q",charge)[None,None,:,None]
                    spins=np.column_stack((changed,np.full(len(selected),hg),
                                           np.full(len(selected),(h1+1)//2),np.full(len(selected),(h2+1)//2)))
                    _add(space,outs[route.outgoing],np.ravel_multi_index(spins.T,space.spin_dims),amp,image)
            else:
                hq,ha,hout,hin=hs
                value=source[ins[route.incoming],:,:,:,(hin+1)//2]
                amp=np.einsum("oi,ab,ibsk->oask",orbital,color,value,optimize=True)
                for flavor in (0,1):
                    q,anti=2*flavor+(hq+1)//2,2*flavor+(ha+1)//2
                    spins=np.column_stack((sf,np.full(64,q),np.full(64,anti),np.full(64,(hout+1)//2)))
                    _add(space,outs[route.outgoing],np.ravel_multi_index(spins.T,space.spin_dims),
                         float(parton_charge(q,"q",charge))*amp,image)
    return QuadraticMinusEndpoint(pilot,space,np.arange(nq,nq+count),space.statistics_project(image))


class FiveBodyQuadraticMinus:
    """Factorized qg scattering and gg↔sea current on all raw five-body tails."""
    physical=False
    complete_physical_current=False

    def __init__(self,outgoing,incoming,transfer,*,b_GeV,total_pplus_GeV,charge):
        if set(outgoing)!=set(incoming) or set(incoming)!={"qqqgg","qqqq-qbar"}:
            raise ValueError("both five-body sectors at both endpoints required")
        for group in (incoming,outgoing):
            example=group["qqqgg"]
            if any(any(getattr(s,k)!=getattr(example,k) for k in ("total_k2","cutoff","target","jz_twice"))
                   or s.sector!=name for name,s in group.items()):
                raise ValueError("inconsistent current endpoint")
        if any(getattr(incoming["qqqgg"],k)!=getattr(outgoing["qqqgg"],k) for k in ("total_k2","target","cutoff")):
            raise ValueError("current endpoints disagree on momentum/target/cutoff")
        self.incoming=dict(incoming);self.outgoing=dict(outgoing)
        self.delta=tuple(float(x) for x in transfer);self.P=float(total_pplus_GeV);self.b=float(b_GeV);self.charge=charge
        if len(self.delta)!=2 or not np.all(np.isfinite(np.r_[self.delta,self.P,self.b])) or min(self.P,self.b)<=0:
            raise ValueError("finite transfer and positive scales required")
        parton_charge(np.arange(4),"q",charge)

    def _conversion(self,source,target,delta):
        lookup={m:i for i,m in enumerate(target.longitudinal)}
        for li,inc in enumerate(source.longitudinal):
            for kq in range(1,sum(inc[3:]),2):
                out=inc[:3]+(kq,sum(inc[3:])-kq)
                orbital=five_displaced_pair_contact("qqqgg","qqqq-qbar",inc,out,(3,4),
                    source.cutoff,target.cutoff,self.b,delta)
                for hq,ha,h1,h2 in product((-1,1),repeat=4):
                    color=sqrt(2)*conversion_minus_kernel(inc,out,(hq,ha,h1,h2),total_pplus_GeV=self.P)
                    selected=np.flatnonzero((2*source.spins[:,3]-1==h1)&(2*source.spins[:,4]-1==h2))
                    for flavor in (0,1):
                        q,anti=2*flavor+(hq+1)//2,2*flavor+(ha+1)//2
                        changed=np.column_stack((source.spins[selected,:3],np.full(len(selected),q),np.full(len(selected),anti)))
                        outspins=np.ravel_multi_index(changed.T,target.spin_dims)
                        weight=np.full(len(selected),float(parton_charge(q,"q",self.charge)))
                        yield li,lookup[out],orbital,color,selected,outspins,weight

    def _terms(self):
        source=self.incoming["qqqgg"];target=self.outgoing["qqqgg"]
        lookup={m:i for i,m in enumerate(target.longitudinal)};K=source.total_k2
        for li,inc in enumerate(source.longitudinal):
            for q,g in product(range(3),(3,4)):
                for kg in range(2,inc[q]+inc[g],2):
                    out=list(inc);out[g]=kg;out[q]=inc[q]+inc[g]-kg;out=tuple(out)
                    orbital=five_displaced_pair_contact("qqqgg","qqqgg",inc,out,(q,g),
                        source.cutoff,target.cutoff,self.b,self.delta)
                    prefactor=K**2/(4*pi*self.P**2*sqrt(inc[g]*kg))
                    for hq,hg in product((-1,1),repeat=2):
                        selected=np.flatnonzero((2*(source.spins[:,q]%2)-1==hq)&(2*source.spins[:,g]-1==hg))
                        color=sum(transverse_spin_factor(hq,hg,crossed=c)*ordered_five_qg_color(q,g,c)/r**2
                                  for c,r in ((False,inc[q]+inc[g]),(True,inc[q]-kg)))
                        weight=parton_charge(source.spins[selected,q],"q",self.charge)
                        yield "qqqgg","qqqgg",li,lookup[out],orbital,prefactor*color,selected,selected,weight
        for values in self._conversion(self.incoming["qqqgg"],self.outgoing["qqqq-qbar"],self.delta):
            yield "qqqgg","qqqq-qbar",*values
        # Physical reverse uses the separately constructed -Delta endpoint.
        for li,lo,orb,col,si,sf,w in self._conversion(self.outgoing["qqqgg"],self.incoming["qqqq-qbar"],tuple(-x for x in self.delta)):
            yield "qqqq-qbar","qqqgg",lo,li,orb.conj().T,col.conj().T,sf,si,w.conj()

    def apply(self,columns):
        n=SixBodyCubicAction._validate(self.incoming,columns)
        source={s:five_statistics_project(sp,columns[s]) for s,sp in self.incoming.items()}
        result={s:np.zeros((sp.dimension,n),complex) for s,sp in self.outgoing.items()}
        for s,t,li,lo,orb,col,si,sf,w in self._terms():
            block=_block(self.incoming[s],li,si,source[s])
            out=np.einsum("oi,ab,ibsk->oask",orb,col,block,optimize=True)
            out*=w[None,None,:,None]
            _add(self.outgoing[t],lo,sf,out,result[t])
        return {s:five_statistics_project(sp,result[s]) for s,sp in self.outgoing.items()}

"""Factorized canonical Q5 -> Q6 transverse contact, per g_s.

Explicit quark/antiquark charges, complete color spaces and normalized
statistics. No giant Q6-by-Q5 matrix, source-rank compression or amplitude
threshold. This is an endpoint action, not by itself the matched H2 current.
"""
from functools import lru_cache
from itertools import product
from math import pi,sqrt
import numpy as np

from deuteron_wigner.bridge.modes.core import polarization
from .m2_canonical_good_current import parton_charge
from .m2_constrained_fermion_current import (
    one_gluon_current_spin,pair_creation_current_spin,discrete_gluon_field_factor,
)
from .m2_antiquark_contact_current import antiquark_one_gluon_current_spin
from .m2_coupled_longitudinal_jacobi import oscillator_shell,oscillator_rotation
from .m2_coupled_pilot import _displacement
from .m2_excluded_fock_kinematics import frame as five_frame
from .m2_six_body_kinematics import frame
from .m2_six_body_vertices import split_routes,split_frame,color_source
from .m2_six_body_cubic import five_statistics_project,SixBodyCubicAction


@lru_cache(maxsize=2048)
def six_contact_orbital(route,in_cutoff,out_cutoff,b_GeV,transfer):
    """Recoiling four-Jacobi parent times exact local relative coincidence."""
    if route.channel not in ("q_to_qg","anti_to_antig","g_to_pair"):
        raise ValueError("charged fermion current channel required")
    delta=np.asarray(transfer,float);b=float(b_GeV)
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)) or not np.isfinite(b) or b<=0:
        raise ValueError("finite transfer and positive width required")
    inc,out=oscillator_shell(4,in_cutoff),oscillator_shell(5,out_cutoff)
    fi,fo=five_frame(route.source_sector,route.incoming),frame(route.sector,route.outgoing)
    x,y=fo.fractions[list(route.daughters)]
    weight=sqrt(x*y/(x+y))*b/sqrt(pi)
    shift=fi.forward[:-1,route.parent,None]*delta
    plus=(shift[:,0]-1j*shift[:,1])/(2*b);alpha=np.r_[plus,plus.conj()]
    adapted=np.zeros((len(out),len(inc)),complex)
    for row,o in enumerate(out):
        if o[4]!=o[9]:continue
        parent=o[:4]+o[5:9]
        for column,i in enumerate(inc):
            adapted[row,column]=weight*np.prod([_displacement(a,c,z) for a,c,z in zip(parent,i,alpha)])
    value=oscillator_rotation(split_frame(route),out_cutoff).T@adapted
    value.setflags(write=False)
    return value


def _rows(space,li,spins):
    raw=(((li*len(space.orbitals)+np.arange(len(space.orbitals))[:,None,None])
           *space.ncolor+np.arange(space.ncolor)[None,:,None])*space.nspin+spins[None,None,:])
    return space.lookup[raw]


def _block(space,li,spins,values):
    rows=_rows(space,li,spins);valid=rows>=0
    block=np.zeros((*rows.shape,values.shape[1]),complex)
    block[valid]=values[rows[valid]]
    return block


def _add(space,li,spins,block,values):
    rows=_rows(space,li,spins);valid=rows>=0
    # Cartesian J has several Jz blocks. Only this explicitly requested
    # endpoint is gathered; no scalar-Hamiltonian Jz equality is imposed.
    values[rows[valid]]+=block[valid]


class SixBodyTransverseContact:
    physical=False
    complete_H2_matched_current=False

    def __init__(self,incoming,outgoing,transfer,*,b_GeV,total_pplus_GeV,charge):
        if set(incoming)!={"qqqgg","qqqq-qbar"} or set(outgoing)!={"qqqggg","qqqq-qbar-g"}:
            raise ValueError("both explicit Q5 and Q6 sectors required")
        self.incoming=dict(incoming);self.outgoing=dict(outgoing)
        first=next(iter(incoming.values()));last=next(iter(outgoing.values()))
        for group,example in ((incoming,first),(outgoing,last)):
            for name,s in group.items():
                if s.sector!=name or any(getattr(s,k)!=getattr(example,k)
                    for k in ("total_k2","cutoff","target","jz_twice")):
                    raise ValueError("spaces disagree inside a current endpoint")
        if any(getattr(first,k)!=getattr(last,k) for k in ("total_k2","target")):
            raise ValueError("current endpoints disagree on total momentum/target")
        self.transfer=tuple(float(x) for x in transfer)
        self.b=float(b_GeV);self.P=float(total_pplus_GeV);self.charge=charge
        if (len(self.transfer)!=2 or not np.all(np.isfinite([*self.transfer,self.b,self.P]))
                or min(self.b,self.P)<=0):
            raise ValueError("finite transfer and positive width/total P+ required")
        parton_charge(np.arange(4),"q",charge)
        self.routes=[]
        for s,t,channel,fock in (
            ("qqqgg","qqqggg","q_to_qg",sqrt(3)),
            ("qqqgg","qqqq-qbar-g","g_to_pair",2*sqrt(2)),
            ("qqqq-qbar","qqqq-qbar-g","q_to_qg",1.),
            ("qqqq-qbar","qqqq-qbar-g","anti_to_antig",1.),
        ):
            for route in split_routes(first.total_k2,s,channel):
                self.routes.append((s,t,route,fock))

    def _terms(self):
        for s,t,route,fock in self.routes:
            source,target=self.incoming[s],self.outgoing[t]
            li=source.longitudinal.index(route.incoming);lo=target.longitudinal.index(route.outgoing)
            orbital=six_contact_orbital(route,source.cutoff,target.cutoff,self.b,self.transfer)
            color=color_source(route)
            x,y=frame(route.sector,route.outgoing).fractions[list(route.daughters)]
            spins=source.spins
            if route.channel!="g_to_pair":
                species="qbar" if route.channel=="anti_to_antig" else "q"
                kernel=antiquark_one_gluon_current_spin if species=="qbar" else one_gluon_current_spin
                right,left=(-self.P*(x+y),-self.P*x) if species=="qbar" else (self.P*x,self.P*(x+y))
                for hg in (-1,1):
                    eps=polarization(y*self.P,0.,0.,hg).conj()[2:]
                    spin=kernel((x*self.P,0.,0.),((x+y)*self.P,0.,0.),mass_GeV=0.,
                        polarization=eps,right_plus=right,left_plus=left)[2:]
                    spin*=fock*discrete_gluon_field_factor(route.outgoing[5])
                    for hi,hf in product(range(2),repeat=2):
                        selected=np.flatnonzero(spins[:,route.parent]%2==hi)
                        changed=spins[selected].copy()
                        changed[:,route.parent]=2*(changed[:,route.parent]//2)+hf
                        out=np.column_stack((changed,np.full(len(selected),(hg+1)//2)))
                        out_spins=np.ravel_multi_index(out.T,target.spin_dims)
                        weight=parton_charge(spins[selected,route.parent],species,self.charge)
                        yield s,t,li,lo,orbital,color,selected,out_spins,weight,spin[:,hf,hi]
            else:
                for hg in (-1,1):
                    eps=polarization((x+y)*self.P,0.,0.,hg)[2:]
                    spin=pair_creation_current_spin((x*self.P,0.,0.),(y*self.P,0.,0.),
                        mass_GeV=0.,polarization=eps,right_plus=x*self.P,left_plus=-y*self.P)[2:]
                    spin*=fock*discrete_gluon_field_factor(route.incoming[4])
                    selected=np.flatnonzero(2*spins[:,4]-1==hg)
                    changed=spins[selected]
                    for hf,ha,flavor in product(range(2),repeat=3):
                        q,anti=2*flavor+hf,2*flavor+ha
                        out=np.column_stack((changed[:,:3],np.full(len(selected),q),
                            np.full(len(selected),anti),changed[:,3]))
                        out_spins=np.ravel_multi_index(out.T,target.spin_dims)
                        weight=np.full(len(selected),float(parton_charge(q,"q",self.charge)))
                        yield s,t,li,lo,orbital,color,selected,out_spins,weight,spin[:,hf,ha]

    def apply(self,columns,*,component):
        if component not in (0,1) or isinstance(component,(bool,np.bool_)):
            raise ValueError("explicit Cartesian component 0 or 1 required")
        n=SixBodyCubicAction._validate(self.incoming,columns)
        source={s:five_statistics_project(space,columns[s]) for s,space in self.incoming.items()}
        result={s:np.zeros((space.dimension,n),complex) for s,space in self.outgoing.items()}
        for s,t,li,lo,orb,color,si,sf,weight,spin in self._terms():
            if spin[component]==0:continue
            block=_block(self.incoming[s],li,si,source[s])
            out=np.einsum("oi,ab,ibsk->oask",orb,color,block,optimize=True)
            out*=spin[component]*weight[None,None,:,None]
            _add(self.outgoing[t],lo,sf,out,result[t])
        return {s:space.statistics_project(result[s]) for s,space in self.outgoing.items()}

    def adjoint(self,columns,*,component):
        """Adjoint of this Delta endpoint; physical absorption uses -Delta."""
        if component not in (0,1) or isinstance(component,(bool,np.bool_)):
            raise ValueError("explicit Cartesian component 0 or 1 required")
        n=SixBodyCubicAction._validate(self.outgoing,columns)
        source={s:space.statistics_project(columns[s]) for s,space in self.outgoing.items()}
        result={s:np.zeros((space.dimension,n),complex) for s,space in self.incoming.items()}
        for s,t,li,lo,orb,color,si,sf,weight,spin in self._terms():
            if spin[component]==0:continue
            block=_block(self.outgoing[t],lo,sf,source[t])
            block*=np.conj(spin[component]*weight[None,None,:,None])
            back=np.einsum("oi,ab,oask->ibsk",orb.conj(),color.conj(),block,optimize=True)
            _add(self.incoming[s],li,si,back,result[s])
        return {s:five_statistics_project(space,result[s]) for s,space in self.incoming.items()}

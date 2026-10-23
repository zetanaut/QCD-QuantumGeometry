"""Exact Q-sector cubic and free oscillator-projector actions.

Continuum Gaussian H tails are integrated before projection. Charged
spectators are kept even for neutral-gluon Hamiltonian monomials.
Transverse components remain the default. The independently derived smooth
minus family requires an explicit context and leaves the Hamiltonian fixed.
"""
from functools import lru_cache
from itertools import product
from math import sqrt
import numpy as np
from .m2_gaussian_cluster_projector_kernel import cubic_hamiltonian_orbital
from .m2_gaussian_cluster_interactions import _frame
from .m2_gaussian_cluster_excluded_current import _definition
from .m2_gaussian_cluster_h2 import GaussianClusterPilot,GaussianClusterH2,require_same_cluster_definition
from .m2_excluded_fock_action import IntermediateSpace
from .m2_excluded_fock_kinematics import SPECIES as FIVE,split_routes as routes5,color_source as color5
from .m2_six_body_kinematics import SPECIES as SIX
from .m2_six_body_vertices import split_routes as routes6,color_source as color6
from .m2_six_body_cubic import SixBodyCubicAction,five_statistics_project
from .m2_six_body_transverse_contact import _block,_add
from .m2_canonical_good_current import parton_charge
from .m2_source_vectors import source_tensor
from .m2_smooth_longitudinal_projector import (
    exact_projector_components,free_projector_components,projector_component_count)

SPECIES={"qqqg":("q","q","q","g"),**FIVE,**SIX}


def _mapped_spins(spins,route,hs,out_dims):
    h1,h2,h0=hs;channel=route.channel
    helicity=2*spins[:,route.parent]-1 if channel.startswith("g_to") else 2*(spins[:,route.parent]%2)-1
    selected=np.flatnonzero(helicity==h0);changed=spins[selected].copy()
    if channel=="g_to_pair":
        for flavor in range(2):
            cols=[changed[:,:3],np.full((len(selected),1),2*flavor+(h1+1)//2),
                  np.full((len(selected),1),2*flavor+(h2+1)//2)]
            if len(route.incoming)==5:cols.append(changed[:,3:4])
            mapped=np.ravel_multi_index(np.column_stack(cols).T,out_dims)
            yield selected,mapped,flavor
    else:
        changed[:,route.parent]=((h1+1)//2 if channel=="g_to_gg" else
                                 2*(changed[:,route.parent]//2)+(h1+1)//2)
        mapped=np.ravel_multi_index(np.column_stack((changed,np.full(len(selected),(h2+1)//2))).T,out_dims)
        yield selected,mapped,None


def _charge_patterns(route,source_sector):
    """One physical flavor weight per inherited fermion, plus the new pair.

    Antiquark charge signs belong to the inherited physical weight, once.
    For pair creation the new pattern is (+1,-1), weighted by its flavor's
    quark charge; the neutral parent is not assigned that charge.
    """
    si,so=SPECIES[source_sector],SPECIES[route.sector]
    for q,kind in enumerate(si):
        if kind=="g":continue
        ei,ef=np.zeros(len(si)),np.zeros(len(so));ei[q]=ef[q]=1.
        yield q,tuple(ei),tuple(ef)
    if route.channel=="g_to_pair":
        ei,ef=np.zeros(len(si)),np.zeros(len(so))
        ef[route.daughters[0]]=1.;ef[route.daughters[1]]=-1.
        yield None,tuple(ei),tuple(ef)


@lru_cache(maxsize=4096)
def _cubic_kernel(route,source_sector,hs,ci,co,b,mass,reg,pp,delta,order,ei,ef,minus_context=None):
    si,so=SPECIES[source_sector],SPECIES[route.sector]
    h=cubic_hamiltonian_orbital(route,hs,b_GeV=b,mass_GeV=mass,regulator=reg,
        total_pplus_GeV=pp,total_transverse_GeV=(0.,0.))
    fi,fo=_frame(route.incoming,si),_frame(route.outgoing,so)
    v=exact_projector_components(h,minus_context=minus_context,out_cutoff=co,in_cutoff=ci,
        out_weights=fo.forward[:-1].T,in_weights=fi.forward[:-1].T,
        out_charges=ef,in_charges=ei,transfer_GeV=delta,total_pplus_GeV=pp,quadrature_order=order)
    v.setflags(write=False)
    return v


def _weight(spins,selected,source_sector,q,flavor,charge):
    if q is None:
        return np.full(len(selected),float(parton_charge(2*flavor,"q",charge)))
    return parton_charge(spins[selected,q],SPECIES[source_sector][q],charge)


def cluster_excluded_cubic_orbital_projector(pilot,space,transfer,*,charge,prescription,quadrature_order,
                                           minus_context=None):
    """Oscillator only: two transverse or one explicitly selected smooth minus.

    P's internal isospin projector is separate in either component family.
    """
    if not isinstance(pilot,GaussianClusterPilot) or not isinstance(space,IntermediateSpace):
        raise ValueError("actual common-cluster pilot and explicit Q5 space required")
    if any(getattr(space,k)!=getattr(pilot,k) for k in ("total_k2","target")):
        raise ValueError("projector endpoints disagree on momentum/target")
    delta,_,order,b,mass,pp=_definition(pilot.field_regulator,transfer,(0.,0.),prescription,
        quadrature_order,pilot.width_GeV,pilot.quark_mass_GeV,pilot.total_pplus_GeV)
    parton_charge(np.arange(4),"q",charge)
    tensor=source_tensor(pilot,"qqqg",np.eye(len(pilot.matrix)))
    source=tensor.reshape(*tensor.shape[:3],128,tensor.shape[-1])
    spins=np.array(list(product(range(4),range(4),range(4),range(2))))
    from .m2_excluded_fock_kinematics import longitudinal_states
    ins={s:i for i,s in enumerate(longitudinal_states("qqqg",pilot.total_k2))}
    outs={s:i for i,s in enumerate(space.longitudinal)}
    nm=projector_component_count(minus_context)
    result=np.zeros((nm,space.dimension,len(pilot.matrix)),complex)
    channels=("q_to_qg","g_to_gg") if space.sector=="qqqgg" else ("g_to_pair",)
    for channel in channels:
        image=np.zeros_like(result)
        fock={"q_to_qg":sqrt(2),"g_to_gg":1/sqrt(2),"g_to_pair":2.}[channel]
        for route in routes5(pilot.total_k2,channel):
            color=color5(channel,route.parent)
            for hs in product((-1,1),repeat=3):
                for selected,mapped,flavor in _mapped_spins(spins,route,hs,space.spin_dims):
                    value=np.take(source[ins[route.incoming]],selected,axis=2)
                    for q,ei,ef in _charge_patterns(route,"qqqg"):
                        orbital=_cubic_kernel(route,"qqqg",hs,pilot.cutoff,space.cutoff,b,mass,
                            pilot.field_regulator,pp,delta,order,ei,ef,minus_context)
                        weight=_weight(spins,selected,"qqqg",q,flavor,charge)
                        amp=np.einsum("moi,ab,ibsn,s->moasn",orbital,color,value,weight,optimize=True)
                        for j in range(nm):_add(space,outs[route.outgoing],mapped,amp[j],image[j])
        result+=fock*np.array([space.statistics_project(a) for a in image])
    return result


def cluster_free_Q_projector_h2_coefficients(outgoing,incoming,transfer,*,charge,prescription,quadrature_order,
                                            minus_context=None):
    """Free Q5/Q6 boundary on every H2 wave coefficient through g4."""
    from .m2_gaussian_cluster_h2_current import _endpoints
    delta,_,order=_endpoints(outgoing,incoming,transfer,(0.,0.),prescription,quadrature_order)
    f,i=outgoing.owner,incoming.owner;p=i.pilot
    nm=projector_component_count(minus_context)
    result={n:np.zeros((nm,len(f.pilot.matrix),len(p.matrix)),complex) for n in (2,3,4)}
    args=dict(b_GeV=p.width_GeV,total_pplus_GeV=p.total_pplus_GeV,charge=charge,
              quadrature_order=order,minus_context=minus_context)
    af,ai=outgoing.five_wave_coefficients,incoming.five_wave_coefficients;ni=len(p.matrix)
    for n,s in i.spaces5.items():
        joined=np.column_stack([a[n] for a in ai])
        for j in range(nm):
            v=_shell_free_projector(f.spaces5[n],s,joined,delta,component=j,**args)
            for left in range(3):
                for right in range(3):
                    power=left+right+2
                    if power<=4:result[power][j]+=af[left][n].conj().T@v[:,right*ni:(right+1)*ni]
    for n,s in i.spaces6.items():
        for j in range(nm):
            v=_shell_free_projector(f.spaces6[n],s,incoming.six_wave_coefficient[n],delta,
                                            component=j,**args)
            result[4][j]+=outgoing.six_wave_coefficient[n].conj().T@v
    return result



def _apply_excluded_raw(pilot,space,tensor):
    """The same physical cubic H on an explicit raw qqqg tensor.

    Unlike the retained basis, this intermediate tensor may contain I=3/2.
    The final physical Q space and Fock statistics are unchanged. This
    helper supplies H for the internal-projector product, not a new source.
    """
    from .m2_gaussian_field_vertices import cubic_field_orbital
    from .m2_excluded_fock_kinematics import longitudinal_states
    source=tensor.reshape(*tensor.shape[:3],128,tensor.shape[-1])
    spins=np.array(list(product(range(4),range(4),range(4),range(2))))
    ins={s:i for i,s in enumerate(longitudinal_states("qqqg",pilot.total_k2))}
    outs={s:i for i,s in enumerate(space.longitudinal)}
    result=np.zeros((space.dimension,tensor.shape[-1]),complex)
    channels=("q_to_qg","g_to_gg") if space.sector=="qqqgg" else ("g_to_pair",)
    for channel in channels:
        image=np.zeros_like(result)
        fock={"q_to_qg":sqrt(2),"g_to_gg":1/sqrt(2),"g_to_pair":2.}[channel]
        for route in routes5(pilot.total_k2,channel):
            color=color5(channel,route.parent)
            for hs in product((-1,1),repeat=3):
                orbital=cubic_field_orbital(route,pilot.cutoff,space.cutoff,pilot.width_GeV,
                    pilot.quark_mass_GeV,hs,regulator=pilot.field_regulator,
                    total_pplus_GeV=pilot.total_pplus_GeV,total_transverse_GeV=(0.,0.))
                for selected,mapped,_ in _mapped_spins(spins,route,hs,space.spin_dims):
                    value=np.take(source[ins[route.incoming]],selected,axis=2)
                    amp=np.einsum("oi,ab,ibsn->oasn",orbital,color,value,optimize=True)
                    _add(space,outs[route.outgoing],mapped,amp,image)
        result+=fock*space.statistics_project(image)
    return result


def cluster_excluded_cubic_projector(pilot,space,transfer,*,charge,prescription,quadrature_order):
    """Full P -> Q5 boundary: exact orbital tails AND retained isospin."""
    from .m2_isospin_projector_current import isospin_outside_tangent
    orbital=cluster_excluded_cubic_orbital_projector(pilot,space,transfer,
        charge=charge,prescription=prescription,quadrature_order=quadrature_order)
    tensor=source_tensor(pilot,"qqqg",np.eye(len(pilot.matrix)))
    k=isospin_outside_tangent(pilot,tensor,transfer,charge=charge,quadrature_order=quadrature_order)
    joined=np.concatenate(tuple(k),axis=-1)
    value=_apply_excluded_raw(pilot,space,joined)/(2*pilot.total_pplus_GeV)
    return orbital+np.array(np.split(value,2,axis=-1))


class ClusterSixCubicProjector:
    """Exact full Q5 -> Q6 boundary, including neutral g -> gg spectators."""
    physical=False
    complete_H2_matched_current=False
    minus_projector_supplied=False

    def __init__(self,incoming,outgoing,transfer,*,regulator,b_GeV,mass_GeV,
                 total_pplus_GeV,charge,prescription,quadrature_order,minus_context=None):
        self.ncomponents=projector_component_count(minus_context)
        self.minus_context=minus_context
        self.minus_projector_supplied=minus_context is not None
        self.transfer,_,self.order,self.b,self.mass,self.P=_definition(
            regulator,transfer,(0.,0.),prescription,quadrature_order,b_GeV,mass_GeV,total_pplus_GeV)
        if set(incoming)!={"qqqgg","qqqq-qbar"} or set(outgoing)!={"qqqggg","qqqq-qbar-g"}:
            raise ValueError("both complete Q5/Q6 families required")
        first=next(iter(incoming.values()))
        for group in (incoming,outgoing):
            example=next(iter(group.values()))
            for n,s in group.items():
                if (s.sector!=n or any(getattr(s,k)!=getattr(example,k) for k in ("total_k2","target","cutoff","jz_twice"))
                        or any(getattr(s,k)!=getattr(first,k) for k in ("total_k2","target"))):
                    raise ValueError("one consistent physical endpoint per Q family required")
        self.incoming,self.outgoing=dict(incoming),dict(outgoing)
        self.regulator,self.charge,self.prescription=regulator,charge,prescription
        parton_charge(np.arange(4),"q",charge)
        self.routes=[]
        for source,target,channel,fock in (
            ("qqqgg","qqqggg","q_to_qg",sqrt(3)),
            ("qqqgg","qqqggg","g_to_gg",sqrt(3)),
            ("qqqgg","qqqq-qbar-g","g_to_pair",2*sqrt(2)),
            ("qqqq-qbar","qqqq-qbar-g","q_to_qg",1.),
            ("qqqq-qbar","qqqq-qbar-g","anti_to_antig",1.)):
            for route in routes6(first.total_k2,source,channel):
                self.routes.append((source,target,route,fock))

    @classmethod
    def from_models(cls,outgoing,incoming,transfer,*,charge,prescription,quadrature_order,minus_context=None):
        if not all(isinstance(x,GaussianClusterH2) for x in (outgoing,incoming)):
            raise ValueError("both actual common-cluster H2 models required")
        f,i=outgoing.pilot,incoming.pilot
        require_same_cluster_definition(f,i)
        for key in ("total_k2","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
            if getattr(f,key)!=getattr(i,key):raise ValueError("endpoints disagree on "+key)
        return cls(incoming.spaces5,outgoing.spaces6,transfer,regulator=i.field_regulator,
            b_GeV=i.width_GeV,mass_GeV=i.quark_mass_GeV,total_pplus_GeV=i.total_pplus_GeV,
            charge=charge,prescription=prescription,quadrature_order=quadrature_order,minus_context=minus_context)

    def _terms(self,component):
        if isinstance(component,(bool,np.bool_)) or component not in range(self.ncomponents):
            raise ValueError("component outside transverse x/y or explicitly selected smooth-minus family")
        for s,t,route,fock in self.routes:
            inc,out=self.incoming[s],self.outgoing[t];spins=inc.spins
            li,lo=inc.longitudinal.index(route.incoming),out.longitudinal.index(route.outgoing)
            color=fock*color6(route)
            for hs in product((-1,1),repeat=3):
                for selected,mapped,flavor in _mapped_spins(spins,route,hs,out.spin_dims):
                    for q,ei,ef in _charge_patterns(route,s):
                        orbital=_cubic_kernel(route,s,hs,inc.cutoff,out.cutoff,self.b,self.mass,
                            self.regulator,self.P,self.transfer,self.order,ei,ef,self.minus_context)[component]
                        weight=_weight(spins,selected,s,q,flavor,self.charge)
                        yield s,t,li,lo,selected,mapped,orbital,color,weight

    def apply(self,columns,*,component):
        n=SixBodyCubicAction._validate(self.incoming,columns)
        source={s:five_statistics_project(sp,columns[s]) for s,sp in self.incoming.items()}
        result={s:np.zeros((sp.dimension,n),complex) for s,sp in self.outgoing.items()}
        for s,t,li,lo,si,sf,orb,color,weight in self._terms(component):
            block=_block(self.incoming[s],li,si,source[s])
            amp=np.einsum("oi,ab,ibsn,s->oasn",orb,color,block,weight,optimize=True)
            _add(self.outgoing[t],lo,sf,amp,result[t])
        return {s:sp.statistics_project(result[s]) for s,sp in self.outgoing.items()}

    def adjoint(self,columns,*,component):
        n=SixBodyCubicAction._validate(self.outgoing,columns)
        source={s:sp.statistics_project(columns[s]) for s,sp in self.outgoing.items()}
        result={s:np.zeros((sp.dimension,n),complex) for s,sp in self.incoming.items()}
        for s,t,li,lo,si,sf,orb,color,weight in self._terms(component):
            block=_block(self.outgoing[t],lo,sf,source[t])
            amp=np.einsum("oi,ab,oasn,s->ibsn",orb.conj(),color.conj(),block,weight.conj(),optimize=True)
            _add(self.incoming[s],li,si,amp,result[s])
        return {s:five_statistics_project(sp,result[s]) for s,sp in self.incoming.items()}


@lru_cache(maxsize=1024)
def _free_kernel(d,cutoff,w,delta,b,pp,order,minus_context=None):
    a=free_projector_components(d,cutoff,w,delta,b_GeV=b,total_pplus_GeV=pp,
                                quadrature_order=order,minus_context=minus_context)
    a.setflags(write=False);return a


def _shell_free_projector(outgoing,incoming,values,transfer,*,component,
                          b_GeV,total_pplus_GeV,charge,quadrature_order,minus_context):
    """Exact free Q boundary using N+2 only; no parent-tail approximation."""
    from .m2_shell_projector_current import _columns,_component
    j=_component(component);a=_columns(incoming,values)
    if j>=projector_component_count(minus_context):
        raise ValueError("component outside explicitly selected projector family")
    for key in ("sector","total_k2","cutoff","target","orbitals","longitudinal","ncolor"):
        if getattr(incoming,key)!=getattr(outgoing,key):raise ValueError("free projector endpoints disagree on "+key)
    if not np.array_equal(incoming.spins,outgoing.spins):raise ValueError("same ordered spins required")
    delta=tuple(np.asarray(transfer,float));species=SPECIES[incoming.sector];d=len(species)-1
    if incoming.sector not in (*FIVE,*SIX):raise ValueError("actual Q5 or Q6 space required")
    result=np.zeros((outgoing.dimension,a.shape[1]),complex)
    # Gather every raw spin/color slot, not just the source-wave span.
    spins=np.arange(incoming.nspin)
    for li,modes in enumerate(incoming.longitudinal):
        fr=_frame(modes,species);block=_block(incoming,li,spins,a)
        image=np.zeros((len(outgoing.orbitals),outgoing.ncolor,len(spins),a.shape[1]),complex)
        for q,kind in enumerate(species):
            if kind=="g":continue
            orbital=_free_kernel(d,incoming.cutoff,tuple(fr.forward[:-1,q]),delta,
                                 b_GeV,total_pplus_GeV,quadrature_order,minus_context)[j]
            weight=parton_charge(incoming.spins[:,q],kind,charge)
            image+=np.einsum("oi,ibsn,s->obsn",orbital,block,weight,optimize=True)
        _add(outgoing,li,spins,image,result)
    return result


def shell_free_transverse_projector(outgoing,incoming,values,transfer,*,component,
                                   b_GeV,total_pplus_GeV,charge,quadrature_order):
    """Original exact free Q transverse action, no minus component inferred."""
    return _shell_free_projector(outgoing,incoming,values,transfer,component=component,
        b_GeV=b_GeV,total_pplus_GeV=total_pplus_GeV,charge=charge,
        quadrature_order=quadrature_order,minus_context=None)


def shell_free_smooth_minus_projector(outgoing,incoming,values,transfer,*,
        b_GeV,total_pplus_GeV,charge,quadrature_order,minus_context):
    """One independently selected smooth oscillator minus action on raw Q."""
    if minus_context is None:raise ValueError("explicit smooth-minus context required")
    return _shell_free_projector(outgoing,incoming,values,transfer,component=0,
        b_GeV=b_GeV,total_pplus_GeV=total_pplus_GeV,charge=charge,
        quadrature_order=quadrature_order,minus_context=minus_context)


def cluster_cubic_Q_projector_h2_coefficients(outgoing,incoming,transfer,*,charge,prescription,quadrature_order,
                                             minus_context=None):
    """Cubic Q words through g4: full transverse OR smooth orbital minus.

    The independently selected minus family does NOT include retained-isospin
    minus. Retained PP, quartic terms and norm maps are separate.
    """
    from .m2_gaussian_cluster_h2_current import _endpoints
    delta,_,order=_endpoints(outgoing,incoming,transfer,(0.,0.),prescription,quadrature_order)
    reverse=tuple(-x for x in delta);f,i=outgoing.owner,incoming.owner
    nm=projector_component_count(minus_context)
    result={n:np.zeros((nm,len(f.pilot.matrix),len(i.pilot.matrix)),complex) for n in (2,3,4)}
    args=dict(charge=charge,prescription=prescription,quadrature_order=order)
    source_action=cluster_excluded_cubic_projector
    if minus_context is not None:
        source_action=cluster_excluded_cubic_orbital_projector
        args["minus_context"]=minus_context
    af,ai=outgoing.five_wave_coefficients,incoming.five_wave_coefficients
    for name in i.spaces5:
        jf=source_action(i.pilot,f.spaces5[name],delta,**args)
        jb=source_action(f.pilot,i.spaces5[name],reverse,**args)
        for k in range(3):
            result[k+2]+=np.einsum("ab,mbc->mac",af[k][name].conj().T,jf,optimize=True)
            result[k+2]+=np.einsum("mba,bc->mac",jb.conj(),ai[k][name],optimize=True)
    forward=ClusterSixCubicProjector.from_models(f,i,delta,**args)
    backward=ClusterSixCubicProjector.from_models(i,f,reverse,**args)
    for j in range(nm):
        jf=forward.apply(ai[0],component=j);jb=backward.apply(af[0],component=j)
        for n in i.spaces6:
            result[4][j]+=outgoing.six_wave_coefficient[n].conj().T@jf[n]
            result[4][j]+=jb[n].conj().T@incoming.six_wave_coefficient[n]
    return result

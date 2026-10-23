"""Exact quartic P-to-Q5/Q6 projector products, per g_s².

Gauge the physical oscillator and retained isospin projectors around the
SAME Hamiltonian. All inherited charged spectators and created flavors
are explicit. A neutral-gluon monomial is not a zero projector current.
The explicit smooth-minus entry point gauges the oscillator only; the raw
isospin-minus tangent remains separate. Q5Q5 actions are in the pair module.
"""
from functools import lru_cache
from itertools import product
import numpy as np
from .m2_gaussian_cluster_h2 import GaussianClusterPilot
from .m2_gaussian_cluster_excluded_current import _definition
from .m2_gaussian_cluster_projector_kernel import quartic_source_hamiltonian_orbital
from .m2_gaussian_cluster_quartic_source_current import (
    PairSourceRoute,quartic_source_coefficients)
from .m2_gaussian_cluster_interactions import _frame
from .m2_excluded_fock_action import IntermediateSpace
from .m2_excluded_fock_kinematics import SPECIES as FIVE
from .m2_six_body_space import SixBodySpace
from .m2_six_body_kinematics import SPECIES as SIX
from .m2_six_body_quartic import direct_routes,COMPONENTS
from .m2_six_body_transverse_contact import _add
from .m2_coupled_longitudinal_jacobi import longitudinal_basis
from .m2_source_vectors import source_tensor
from .m2_canonical_good_current import parton_charge
from .m2_isospin_projector_current import isospin_outside_tangent
from .m2_smooth_longitudinal_projector import exact_projector_components,projector_component_count

SPECIES={"qqq":("q","q","q"),"qqqg":("q","q","q","g"),**FIVE,**SIX}


def _source_routes(pilot,space,component):
    if not isinstance(pilot,GaussianClusterPilot) or not isinstance(space,(IntermediateSpace,SixBodySpace)):
        raise ValueError("actual common-cluster pilot and physical Q5/Q6 space required")
    if any(getattr(space,k)!=getattr(pilot,k) for k in ("total_k2","target")):
        raise ValueError("source endpoints disagree on momentum/target")
    if pilot.field_regulator!=pilot.wick_operators.regulator:
        raise ValueError("pilot field and Wick definitions disagree")
    pair=isinstance(space,IntermediateSpace)
    if component not in (COMPONENTS[:3] if pair else COMPONENTS):
        raise ValueError("explicit quartic component for this Q family required")
    sea=component in ("valence_sea","gluon_IF_pair","gluon_IG_pair")
    expected=("qqqq-qbar" if sea else "qqqgg") if pair else ("qqqq-qbar-g" if sea else "qqqggg")
    if space.sector!=expected:
        raise ValueError("matching quartic component and physical endpoint required")
    if not pair:return "qqqg",direct_routes(pilot.total_k2,component)
    routes=[]
    for out in space.longitudinal:
        for q in range(3):
            inc=list(out[:3]);inc[q]+=out[3]+out[4]
            routes.append(PairSourceRoute(component,tuple(inc),out,q))
    return "qqq",tuple(routes)


def _mapped(spins,route,hs,out_dims):
    """Injective raw spin maps; flavor is never inferred from target charge."""
    pair=isinstance(route,PairSourceRoute)
    if route.component.startswith("valence"):
        ho,hi,h1,h2=hs;q=route.emitter
        selected=np.flatnonzero(2*(spins[:,q]%2)-1==hi)
        changed=spins[selected].copy()
        changed[:,q]=2*(changed[:,q]//2)+(ho+1)//2
        for flavor in (range(2) if route.component=="valence_sea" else (None,)):
            if flavor is not None:
                labels=[changed[:,:3],np.full(len(selected),2*flavor+(h1+1)//2),
                        np.full(len(selected),2*flavor+(h2+1)//2)]
                if not pair:labels.append(changed[:,3])
            else:
                labels=[changed,np.full(len(selected),(h1+1)//2),np.full(len(selected),(h2+1)//2)]
            yield selected,np.ravel_multi_index(np.column_stack(labels).T,out_dims),flavor
    else:
        h1,h2,h3,hi=hs
        selected=np.flatnonzero(2*spins[:,3]-1==hi)
        for flavor in (range(2) if route.component.endswith("_pair") else (None,)):
            labels=[spins[selected,:3]]
            if flavor is not None:
                labels.extend(np.full(len(selected),2*flavor+(h+1)//2) for h in (h1,h2))
            else:labels.extend(np.full(len(selected),(h+1)//2) for h in (h1,h2))
            labels.append(np.full(len(selected),(h3+1)//2))
            yield selected,np.ravel_multi_index(np.column_stack(labels).T,out_dims),flavor


def _patterns(route,sector):
    for q in range(3):
        ei,ef=np.zeros(len(SPECIES[sector])),np.zeros(len(SPECIES[route.sector]))
        ei[q]=ef[q]=1.
        yield q,tuple(ei),tuple(ef)
    if route.component=="valence_sea" or route.component.endswith("_pair"):
        ei,ef=np.zeros(len(SPECIES[sector])),np.zeros(len(SPECIES[route.sector]))
        ef[3],ef[4]=1.,-1.
        yield None,tuple(ei),tuple(ef)


@lru_cache(maxsize=512)
def _hamiltonian(route,ci,co,b,reg):
    h=quartic_source_hamiltonian_orbital(route,b_GeV=b,regulator=reg,total_transverse_GeV=(0.,0.))
    value=h.integrate(co,ci)
    value.setflags(write=False)
    return value


@lru_cache(maxsize=2048)
def _boundary(route,sector,ci,co,b,reg,pp,delta,order,ei,ef,minus_context=None):
    h=quartic_source_hamiltonian_orbital(route,b_GeV=b,regulator=reg,total_transverse_GeV=(0.,0.))
    fi,fo=_frame(route.incoming,SPECIES[sector]),_frame(route.outgoing,SPECIES[route.sector])
    value=exact_projector_components(h,minus_context=minus_context,out_cutoff=co,in_cutoff=ci,
        out_weights=fo.forward[:-1].T,in_weights=fi.forward[:-1].T,
        out_charges=ef,in_charges=ei,transfer_GeV=delta,total_pplus_GeV=pp,quadrature_order=order)
    value.setflags(write=False)
    return value


def _apply_source_raw(pilot,space,tensor,*,component,transfer=None,charge=None,quadrature_order=None,
                      minus_context=None):
    """Physical H on raw qqq/qqqg, or its exact orbital projector product.

    Raw I=3/2 intermediates are permitted here for the isospin product.
    The stored physical H and all Fock coefficients remain unchanged.
    """
    sector,routes=_source_routes(pilot,space,component)
    dims=(4,4,4) if sector=="qqq" else (4,4,4,2)
    spins=np.array(list(product(*(range(d) for d in dims))))
    source=tensor.reshape(*tensor.shape[:3],len(spins),tensor.shape[-1])
    incoming={s:j for j,s in enumerate(longitudinal_basis(sector,pilot.total_k2))}
    outgoing={s:j for j,s in enumerate(space.longitudinal)}
    if transfer is None and minus_context is not None:
        raise ValueError("a minus context cannot silently change a Hamiltonian action")
    nm=1 if transfer is None else projector_component_count(minus_context)
    result=np.zeros((nm,space.dimension,tensor.shape[-1]),complex)
    for route in routes:
        for hs in product((-1,1),repeat=4):
            color,_=quartic_source_coefficients(route,hs,total_pplus_GeV=pilot.total_pplus_GeV,
                                               emitter_charge=1.,pair_charge=1.)
            if not np.any(color):continue
            for selected,mapped,flavor in _mapped(spins,route,hs,space.spin_dims):
                value=np.take(source[incoming[route.incoming]],selected,axis=2)
                if transfer is None:
                    orb=_hamiltonian(route,pilot.cutoff,space.cutoff,pilot.width_GeV,pilot.field_regulator)
                    amp=np.einsum("oi,ab,ibsn->oasn",orb,color,value,optimize=True)
                    _add(space,outgoing[route.outgoing],mapped,amp,result[0])
                else:
                    for q,ei,ef in _patterns(route,sector):
                        orb=_boundary(route,sector,pilot.cutoff,space.cutoff,pilot.width_GeV,
                            pilot.field_regulator,pilot.total_pplus_GeV,transfer,quadrature_order,ei,ef,minus_context)
                        weight=(parton_charge(spins[selected,q],"q",charge) if q is not None else
                                np.full(len(selected),parton_charge(2*flavor,"q",charge)))
                        amp=np.einsum("moi,ab,ibsn,s->moasn",orb,color,value,weight,optimize=True)
                        for j in range(nm):_add(space,outgoing[route.outgoing],mapped,amp[j],result[j])
    result=np.array([space.statistics_project(a) for a in result])
    return result[0] if transfer is None else result


def cluster_quartic_source_projector(pilot,space,transfer,*,component,charge,prescription,quadrature_order):
    """Full P→Q source boundary, all retained columns, transverse x/y.

    Reversed-transfer adjoints define absorption. No canonical current,
    propagator, Wick Hamiltonian or normalization is added by this action.
    """
    sector,_=_source_routes(pilot,space,component)
    delta,_,order,_,_,pp=_definition(pilot.field_regulator,transfer,(0.,0.),prescription,
        quadrature_order,pilot.width_GeV,pilot.quark_mass_GeV,pilot.total_pplus_GeV)
    parton_charge(np.arange(4),"q",charge)
    tensor=source_tensor(pilot,sector,np.eye(len(pilot.matrix)))
    result=_apply_source_raw(pilot,space,tensor,component=component,transfer=delta,
                             charge=charge,quadrature_order=order)
    tangent=isospin_outside_tangent(pilot,tensor,delta,charge=charge,quadrature_order=order,sector=sector)
    raw=_apply_source_raw(pilot,space,np.concatenate(tuple(tangent),axis=-1),component=component)/(2*pp)
    return result+np.array(np.split(raw,2,axis=-1))


def cluster_quartic_source_smooth_minus_projector(pilot,space,transfer,*,component,charge,
        prescription,quadrature_order,minus_context):
    """One raw P-to-Q smooth oscillator minus action, no isospin minus.

    All ten physical sources retain their original color/Fock factors.
    The leading axis has length one; reversed transfer defines absorption.
    """
    if minus_context is None:raise ValueError("explicit smooth-minus context required")
    projector_component_count(minus_context)
    sector,_=_source_routes(pilot,space,component)
    delta,_,order,_,_,_=_definition(pilot.field_regulator,transfer,(0.,0.),prescription,
        quadrature_order,pilot.width_GeV,pilot.quark_mass_GeV,pilot.total_pplus_GeV)
    parton_charge(np.arange(4),"q",charge)
    tensor=source_tensor(pilot,sector,np.eye(len(pilot.matrix)))
    return _apply_source_raw(pilot,space,tensor,component=component,transfer=delta,
        charge=charge,quadrature_order=order,minus_context=minus_context)


def cluster_quartic_source_projector_h2_coefficients(outgoing,incoming,transfer,*,charge,prescription,quadrature_order,
                                                   minus_context=None):
    """Source return words at g3/g4: full transverse OR smooth oscillator minus.

    The minus selection has no retained-isospin minus partner. Q5Q5,
    canonical/Wick currents and both norm maps remain separate.
    """
    from .m2_gaussian_cluster_h2_current import _endpoints
    delta,_,order=_endpoints(outgoing,incoming,transfer,(0.,0.),prescription,quadrature_order)
    f,i=outgoing.owner,incoming.owner;reverse=tuple(-x for x in delta)
    args=dict(charge=charge,prescription=prescription,quadrature_order=order)
    nm=projector_component_count(minus_context)
    source_action=cluster_quartic_source_projector
    if minus_context is not None:
        source_action=cluster_quartic_source_smooth_minus_projector
        args["minus_context"]=minus_context
    result={n:np.zeros((nm,len(f.pilot.matrix),len(i.pilot.matrix)),complex) for n in (3,4)}
    for fs,ins,components,waves in (
        (f.spaces5,i.spaces5,COMPONENTS[:3],[(3,outgoing.five_wave_coefficients[0],incoming.five_wave_coefficients[0]),
                                          (4,outgoing.five_wave_coefficients[1],incoming.five_wave_coefficients[1])]),
        (f.spaces6,i.spaces6,COMPONENTS,[(4,outgoing.six_wave_coefficient,incoming.six_wave_coefficient)])):
        for c in components:
            sea=c in ("valence_sea","gluon_IF_pair","gluon_IG_pair")
            name=("qqqq-qbar" if sea else "qqqgg") if fs is f.spaces5 else ("qqqq-qbar-g" if sea else "qqqggg")
            jf=source_action(i.pilot,fs[name],delta,component=c,**args)
            jb=source_action(f.pilot,ins[name],reverse,component=c,**args)
            for n,af,ai in waves:
                result[n]+=np.einsum("ab,mbc->mac",af[name].conj().T,jf,optimize=True)
                result[n]+=np.einsum("mba,bc->mac",jb.conj(),ai[name],optimize=True)
    return result

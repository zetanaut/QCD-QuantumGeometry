"""Exact continuum-projector boundaries for all physical 1-to-2/1-to-3 routes."""
from dataclasses import replace
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_projector_kernel import (
    cubic_hamiltonian_orbital,quartic_source_hamiltonian_orbital,exact_transverse_projector_boundary)
from deuteron_wigner.quantum.m2_gaussian_cluster_interactions import _frame,cluster_retained_emission,cluster_six_cubic
from deuteron_wigner.quantum.m2_gaussian_field_vertices import cubic_field_orbital,pair_field_orbital,direct_field_orbital
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import EmissionRoute
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import SplitRoute,SPECIES as FIVE
from deuteron_wigner.quantum.m2_six_body_kinematics import SPECIES as SIX
from test_m2_gaussian_cluster_current import ROUTES as CUBIC,route_current,regulator
from test_m2_gaussian_cluster_quartic_source_current import (
    ROUTES as QUARTIC,PairSourceRoute,frames,hs_nonzero,current,coefficients,quartic_source_charges)
from test_m2_gaussian_cluster_projector_kernel import B,PP,D,density


def checked(h,wi,wf,ei,ef,physical):
    kwargs=dict(out_cutoff=1,in_cutoff=1,out_weights=wf,in_weights=wi,
        out_charges=ef,in_charges=ei,transfer_GeV=D,total_pplus_GeV=PP,quadrature_order=12)
    k=exact_transverse_projector_boundary(h,**kwargs)
    projected=h.integrate(1,1)
    expected=projected@density(wi,ei,1,1)-density(wf,ef,1,1)@projected
    np.testing.assert_allclose(2*PP*np.einsum("j,jab->ab",D,physical+k),expected,atol=8e-15)
    full=sum(e*h.integrate(1,1,in_shift=-w[:,None]*D) for e,w in zip(ei,wi))
    full-=sum(e*h.integrate(1,1,out_shift=w[:,None]*D) for e,w in zip(ef,wf))
    np.testing.assert_allclose(2*PP*np.einsum("j,jab->ab",D,physical),full,atol=7e-15)
    reverse=replace(h,out_map=h.in_map,in_map=h.out_map,constant=np.conj(h.constant),
        linear=None if h.linear is None else h.linear.conj())
    back=exact_transverse_projector_boundary(reverse,**{**kwargs,
        "out_weights":wi,"in_weights":wf,"out_charges":ei,"in_charges":ef,"transfer_GeV":-D})
    np.testing.assert_allclose(k,back.conj().transpose(0,2,1),atol=8e-15)
    return k


@pytest.mark.parametrize("route",CUBIC)
def test_every_cubic_route_exact_projector_closes_full_physical_current_and_matches_H(route):
    if isinstance(route,EmissionRoute):
        si,so=("q",)*3,("q",)*3+("g",)
        channel,parent,children="q_to_qg",route.emitter,(route.emitter,3)
    else:
        si=FIVE[route.source_sector] if hasattr(route,"source_sector") else ("q",)*3+("g",)
        so=(SIX if hasattr(route,"source_sector") else FIVE)[route.sector]
        channel,parent,children=route.channel,route.parent,route.daughters
    fi,fo=_frame(route.incoming,si),_frame(route.outgoing,so)
    ei=np.array([0. if s=="g" else -.2 if s=="qbar" else .2+.1*j for j,s in enumerate(si)])
    if channel=="q_to_qg":ei[parent]=.7
    if channel=="anti_to_antig":ei[parent]=-.7
    ef=np.array([ei[j] if j<len(si) and si[j]==s else 0. for j,s in enumerate(so)])
    if channel=="g_to_pair":ef[list(children)]=(.7,-.7)
    elif channel!="g_to_gg":ef[children[0]]=ei[parent]
    hs=(1,-1,-1) if channel=="g_to_pair" else (1,-1,1)
    h=cubic_hamiltonian_orbital(route,hs,b_GeV=B,mass_GeV=.35,regulator=regulator(),
        total_pplus_GeV=PP,total_transverse_GeV=(0.,0.))
    if isinstance(route,EmissionRoute):
        reference=cluster_retained_emission(route,1,B,.35,hs[0],hs[2],hs[1],
            regulator=regulator(),total_pplus_GeV=PP)
    elif isinstance(route,SplitRoute):
        reference=cubic_field_orbital(route,1,1,B,.35,hs,regulator=regulator(),
            total_pplus_GeV=PP,total_transverse_GeV=(0.,0.))
    else:
        reference=cluster_six_cubic(route,1,1,B,.35,hs,regulator=regulator(),total_pplus_GeV=PP)
    np.testing.assert_allclose(h.integrate(1,1),reference,atol=5e-15)
    physical=route_current(route,hs,delta=D,mean=(0.,0.),ci=1,co=1,order=12).total[2:]
    k=checked(h,fi.forward[:-1].T,fo.forward[:-1].T,ei,ef,physical)
    if channel=="g_to_gg":
        np.testing.assert_array_equal(physical,0.)
        # A neutral interaction can still need a charged-SPECTATOR
        # oscillator boundary; this is not a photon on the gluon fields.
        assert np.linalg.norm(k)>1e-7


@pytest.mark.parametrize("route",QUARTIC)
def test_every_quartic_source_exact_projector_closes_full_physical_current_and_matches_H(route):
    fi,fo,_=frames(route)
    ei=np.array([.3,-.2,.1]+([] if isinstance(route,PairSourceRoute) else [0.]))
    legs=quartic_source_charges(route,emitter_charge=.7,pair_charge=-.2)
    ei[route.emitter]=legs[-1]
    ef=np.zeros(len(route.outgoing));ef[:3]=ei[:3]
    ef[list(route.daughters)]=legs[:3]
    h=quartic_source_hamiltonian_orbital(route,b_GeV=B,regulator=regulator(),
                                        total_transverse_GeV=(0.,0.))
    args=dict(regulator=regulator(),total_pplus_GeV=PP,total_transverse_GeV=(0.,0.))
    if isinstance(route,PairSourceRoute):
        reference=pair_field_orbital(route.incoming,route.outgoing,route.emitter,1,1,B,sector=route.sector,**args)
    else:reference=direct_field_orbital(route,1,1,B,**args)
    np.testing.assert_allclose(h.integrate(1,1),reference,atol=3e-16)
    hs=hs_nonzero(route);color=coefficients(route,hs)[0]
    row,col=np.unravel_index(np.abs(color).argmax(),color.shape)
    value=current(route,hs,ci=1,co=1,delta=D,mean=(0.,0.),order=12)
    physical=value.total[2:,:,:,row,col]/color[row,col]
    checked(h,fi.forward[:-1].T,fo.forward[:-1].T,ei,ef,physical)

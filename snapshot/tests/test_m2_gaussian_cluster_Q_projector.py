"""Exact full-Q projector actions against independent projected Hamiltonians."""
from functools import lru_cache
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_Q_projector import (
    cluster_excluded_cubic_projector,ClusterSixCubicProjector,shell_free_transverse_projector)
from deuteron_wigner.quantum.m2_gaussian_cluster_excluded_current import (
    build_cluster_excluded_cubic_current,ClusterSixCubicCurrent)
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_shell_current import shell_density_between
from deuteron_wigner.quantum.m2_shell_transverse_current import shell_transverse_current_between
from deuteron_wigner.quantum.m2_shell_free_propagation import ShellFreePropagation
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_six_body_space import SixBodySpace
from deuteron_wigner.quantum.m2_six_body_cubic import five_statistics_project
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION
from test_m2_gaussian_cluster_h2 import model

D=np.array([.12,.16]);PP=1.3;B=.4
ARGS=dict(charge="electromagnetic",prescription=PRESCRIPTION,quadrature_order=8)
FAMILIES=("qqqgg","qqqq-qbar","qqqggg","qqqq-qbar-g")


def raw(spaces,seed=507):
    rng=np.random.default_rng(seed)
    return {n:(rng.normal(size=(s.dimension,2))+1j*rng.normal(size=(s.dimension,2)))/np.sqrt(s.dimension)
            for n,s in spaces.items()}


def rho(f,i,v,delta=D):
    return shell_density_between(f,i,v,delta,b_GeV=B,charge="electromagnetic")


@pytest.mark.parametrize("sector",FAMILIES)
@pytest.mark.parametrize("target,jz,cutoff",[("PROTON",1,0),("NEUTRON",-1,1),("PROTON",1,1)])
def test_free_exact_full_raw_Ward_and_reverse_no_interaction_parent(sector,target,jz,cutoff):
    cls=IntermediateSpace if sector in FAMILIES[:2] else SixBodySpace
    i=cls(sector,9,cutoff,target,1);f=cls(sector,9,cutoff,target,jz)
    x,y=raw({sector:i})[sector],raw({sector:f},508)[sector]
    hi,hf=[ShellFreePropagation(s,b_GeV=B,quark_mass_GeV=.35,gluon_mass_GeV=.12) for s in (i,f)]
    expected=hf.apply(rho(f,i,x))-rho(f,i,hi.apply(x))
    full=np.zeros_like(expected)
    for j in range(2):
        args=dict(component=j,b_GeV=B,total_pplus_GeV=PP,charge="electromagnetic",quadrature_order=8)
        k=shell_free_transverse_projector(f,i,x,D,**args)
        back=shell_free_transverse_projector(i,f,y,-D,**args)
        np.testing.assert_allclose(np.vdot(y,k),np.vdot(back,x),atol=4e-15)
        jt=shell_transverse_current_between(f,i,x,D,component=j,b_GeV=B,total_pplus_GeV=PP,charge="electromagnetic")
        full+=2*PP*D[j]*(jt+k)
    np.testing.assert_allclose(full,expected,atol=6e-15)


@pytest.mark.parametrize("sector",FAMILIES[:2])
@pytest.mark.parametrize("target,jz,cutoff",[("PROTON",1,0),("NEUTRON",-1,0),("PROTON",1,1)])
def test_P_to_Q5_exact_cubic_boundary_matches_full_actual_source_Hamiltonian(sector,target,jz,cutoff):
    f,i=model(cutoff,0,target,jz),model(cutoff,0,target,1)
    sf,si=f.spaces5[sector],i.spaces5[sector]
    k=cluster_excluded_cubic_projector(i.pilot,sf,D,**ARGS)
    j=build_cluster_excluded_cubic_current(i.pilot,sf,D,mean_total_transverse_GeV=(0.,0.),**ARGS)
    physical=np.zeros_like(k);physical[:,:,j.retained_columns]=j.total[2:]
    hi,hf=i.qq.sources_by_power[1][sector],f.qq.sources_by_power[1][sector]
    p=good_current_density_between(f.pilot,i.pilot,D,charge="electromagnetic")
    expected=hf@p-rho(sf,si,hi)
    np.testing.assert_allclose(2*PP*np.einsum("j,jab->ab",D,physical+k),expected,atol=2e-14)
    # Input retained qqq columns are absent from the cubic Q5 source.
    nqqq=i.pilot.bases[0].shape[1]
    np.testing.assert_array_equal(k[:,:,:nqqq],0.)
    assert np.linalg.norm(k)>1e-7


def component_H(m,values,source,channel):
    u=m.six.U
    v={n:five_statistics_project(s,values[n]) for n,s in m.spaces5.items()}
    label=source+":"+channel;_,target,_=u.components[label]
    out={n:np.zeros((s.dimension,next(iter(values.values())).shape[1]),complex) for n,s in m.spaces6.items()}
    out[target]=u.component_action(label,v[source])
    return {n:s.statistics_project(out[n]) for n,s in m.spaces6.items()}


CHANNELS=[("qqqgg","q_to_qg"),("qqqgg","g_to_gg"),("qqqgg","g_to_pair"),
          ("qqqq-qbar","q_to_qg"),("qqqq-qbar","anti_to_antig")]


@pytest.mark.parametrize("source,channel",CHANNELS)
def test_every_Q5_Q6_cubic_projector_with_spectators_matches_actual_U65(source,channel):
    i=model(1,1,"PROTON",1)
    f=model(1,1,"PROTON",-1) if channel=="g_to_gg" else i
    x,y=raw(i.spaces5),raw(f.spaces6,508)
    k=ClusterSixCubicProjector.from_models(f,i,D,**ARGS)
    j=ClusterSixCubicCurrent.from_models(f,i,D,mean_total_transverse_GeV=(0.,0.),**ARGS)
    for a in (k,j):a.routes=[r for r in a.routes if r[0]==source and r[2].channel==channel]
    hx=component_H(i,x,source,channel)
    shifted={n:rho(f.spaces5[n],s,x[n]) for n,s in i.spaces5.items()}
    expected=component_H(f,shifted,source,channel)
    for n in expected:expected[n]-=rho(f.spaces6[n],i.spaces6[n],hx[n])
    full={n:np.zeros_like(v) for n,v in expected.items()};norm=0.
    for a in range(2):
        ka=k.apply(x,component=a);kb=k.adjoint(y,component=a)
        np.testing.assert_allclose(sum(np.vdot(y[n],v) for n,v in ka.items()),
            sum(np.vdot(v,x[n]) for n,v in kb.items()),atol=5e-16)
        physical=j.apply(x,component=a+2,part="total")
        for n in full:
            full[n]+=2*PP*D[a]*(ka[n]+physical[n])
            norm+=np.linalg.norm(ka[n])
    for n in full:np.testing.assert_allclose(full[n],expected[n],atol=5e-15)
    assert norm>1e-8  # Includes g->gg with charged spectators, not a gluon photon.


def test_projector_action_rejects_missing_physical_families_and_minus_slot():
    m=model()
    with pytest.raises(ValueError):
        cluster_excluded_cubic_projector(object(),m.spaces5["qqqgg"],D,**ARGS)
    a=ClusterSixCubicProjector.from_models(m,m,D,**ARGS)
    with pytest.raises(ValueError,match="transverse"):a.apply(raw(m.spaces5),component=2)
    assert not a.complete_H2_matched_current and not a.minus_projector_supplied

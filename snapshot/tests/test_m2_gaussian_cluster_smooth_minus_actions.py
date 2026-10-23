"""Physical P/Q minus action: boost, adjoint, flavor and statistics controls."""
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_smooth_longitudinal_projector import (
    SmoothMinusProjectorContext,SMOOTH_PRESCRIPTION)
from deuteron_wigner.quantum.m2_gaussian_cluster_retained_projector import (
    exact_retained_transverse_projector_coefficients,exact_retained_smooth_minus_projector_coefficients)
from deuteron_wigner.quantum.m2_gaussian_cluster_Q_projector import (
    cluster_excluded_cubic_orbital_projector,ClusterSixCubicProjector,
    shell_free_smooth_minus_projector,shell_free_transverse_projector)
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_Q_projector import (
    cluster_quartic_source_smooth_minus_projector,_apply_source_raw)
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_Q_pair_projector import ClusterFiveQuarticProjector
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_pair_actions import CHARGED_COMPONENTS,NEUTRAL_COMPONENTS
from deuteron_wigner.quantum.m2_source_vectors import source_tensor
from deuteron_wigner.quantum.m2_six_body_cubic import five_statistics_project
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_six_body_space import SixBodySpace
from test_m2_gaussian_cluster_Q_projector import ARGS,D,PP,B,FAMILIES,CHANNELS,raw
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_quartic_source_actions import FAMILIES as SOURCES,space

BETA=np.array([.13,-.07])
ZERO=SmoothMinusProjectorContext(mean_total_transverse_GeV=(0.,0.),prescription=SMOOTH_PRESCRIPTION)
BOOST=SmoothMinusProjectorContext(mean_total_transverse_GeV=tuple(PP*BETA),prescription=SMOOTH_PRESCRIPTION)


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_full_retained_minus_all_powers_boost_reversal_and_flavor(target,jz):
    f,i=model(target=target,jz=jz).pilot,model(target=target).pilot
    values=exact_retained_smooth_minus_projector_coefficients(f,i,D,minus_context=ZERO,**ARGS)
    boost=exact_retained_smooth_minus_projector_coefficients(f,i,D,minus_context=BOOST,**ARGS)
    transverse=exact_retained_transverse_projector_coefficients(f,i,D,**ARGS)
    reverse=exact_retained_smooth_minus_projector_coefficients(i,f,-D,minus_context=ZERO,**ARGS)
    up,down=[exact_retained_smooth_minus_projector_coefficients(f,i,D,minus_context=ZERO,
        **dict(ARGS,charge=q)) for q in ("up_number","down_number")]
    for n,v in values.items():
        assert v.shape==(1,len(f.matrix),len(i.matrix))
        np.testing.assert_allclose(boost[n]-v,np.einsum("j,jab->ab",BETA,transverse[n])[None],atol=2e-14)
        np.testing.assert_allclose(v,reverse[n].conj().transpose(0,2,1),atol=2e-14)
        np.testing.assert_allclose(v,(2*up[n]-down[n])/3,atol=2e-14)
        if jz==-1 and n in (0,2):
            # Ground-shell free/quartic scalar kernels cannot transfer Jz.
            np.testing.assert_allclose(v,0.,atol=2e-14)
        else:assert np.linalg.norm(v)>1e-7


@pytest.mark.parametrize("sector",FAMILIES)
@pytest.mark.parametrize("cutoff",[0,1])
def test_all_free_raw_Q_minus_components_boost_and_reversal(sector,cutoff):
    cls=IntermediateSpace if sector in FAMILIES[:2] else SixBodySpace
    i=cls(sector,9,cutoff,"NEUTRON",1)
    f=cls(sector,9,cutoff,"NEUTRON",-1 if cutoff else 1)
    x,y=raw({sector:i})[sector],raw({sector:f},509)[sector]
    args=dict(b_GeV=B,total_pplus_GeV=PP,charge="electromagnetic",quadrature_order=8)
    a=shell_free_smooth_minus_projector(f,i,x,D,minus_context=ZERO,**args)
    b=shell_free_smooth_minus_projector(f,i,x,D,minus_context=BOOST,**args)
    back=shell_free_smooth_minus_projector(i,f,y,-D,minus_context=ZERO,**args)
    transverse=sum(BETA[j]*shell_free_transverse_projector(f,i,x,D,component=j,**args) for j in range(2))
    np.testing.assert_allclose(b-a,transverse,atol=4e-15)
    np.testing.assert_allclose(np.vdot(y,a),np.vdot(back,x),atol=4e-15)
    if sector=="qqqggg" and cutoff==0:
        # K9 fixes all three quark modes to one. Their equal radial factors
        # multiply the neutron's exactly zero total charge.
        np.testing.assert_allclose(a,0.,atol=4e-15)
    else:assert np.linalg.norm(a)>1e-7


@pytest.mark.parametrize("sector",FAMILIES[:2])
def test_raw_P_to_Q5_smooth_minus_keeps_boost_and_flavor_without_isospin_substitution(sector):
    p=model(target="NEUTRON").pilot
    s=IntermediateSpace(sector,9,1,"NEUTRON",-1)
    a=cluster_excluded_cubic_orbital_projector(p,s,D,minus_context=ZERO,**ARGS)
    b=cluster_excluded_cubic_orbital_projector(p,s,D,minus_context=BOOST,**ARGS)
    transverse=cluster_excluded_cubic_orbital_projector(p,s,D,**ARGS)
    np.testing.assert_allclose(b-a,np.einsum("j,jab->ab",BETA,transverse)[None],atol=6e-15)
    up,down=[cluster_excluded_cubic_orbital_projector(p,s,D,minus_context=ZERO,
        **dict(ARGS,charge=q)) for q in ("up_number","down_number")]
    np.testing.assert_allclose(a,(2*up-down)/3,atol=6e-15)
    np.testing.assert_allclose(s.statistics_project(a[0]),a[0],atol=3e-15)
    assert np.linalg.norm(a)>1e-7


@pytest.mark.parametrize("source,channel",CHANNELS)
def test_all_Q5_Q6_smooth_minus_cubic_routes_boost_adjoint_and_statistics(source,channel):
    # The pure-gluon linear source needs an odd orbital control; N0 is a
    # selection zero and is not used as proof that its spectator current vanishes.
    cutoff=1 if channel=="g_to_gg" else 0
    i=model(q5=cutoff,q6=cutoff)
    f=model(q5=cutoff,q6=cutoff,jz=-1) if channel=="g_to_gg" else i
    x,y=raw(i.spaces5),raw(f.spaces6,511)
    actions=[ClusterSixCubicProjector.from_models(f,i,D,minus_context=c,**ARGS) for c in (ZERO,BOOST,None)]
    for a in actions:a.routes=[r for r in a.routes if r[0]==source and r[2].channel==channel]
    a,b,t=actions
    v=a.apply(x,component=0);boost=b.apply(x,component=0)
    back=a.adjoint(y,component=0)
    ts=[t.apply(x,component=j) for j in range(2)]
    for n,s in f.spaces6.items():
        np.testing.assert_allclose(boost[n]-v[n],sum(BETA[j]*ts[j][n] for j in range(2)),atol=5e-15)
        np.testing.assert_allclose(s.statistics_project(v[n]),v[n],atol=3e-15)
    np.testing.assert_allclose(sum(np.vdot(y[n],v[n]) for n in v),
                               sum(np.vdot(back[n],x[n]) for n in x),atol=5e-15)
    assert sum(np.linalg.norm(z) for z in v.values())>1e-8
    assert a.minus_projector_supplied and not a.complete_H2_matched_current


@pytest.mark.parametrize("n,c",SOURCES)
def test_every_quartic_source_smooth_minus_keeps_original_physical_flavor_and_Fock_maps(n,c):
    p=model().pilot;s=space(n,c)
    a=cluster_quartic_source_smooth_minus_projector(p,s,D,component=c,minus_context=ZERO,**ARGS)
    b=cluster_quartic_source_smooth_minus_projector(p,s,D,component=c,minus_context=BOOST,**ARGS)
    tensor=source_tensor(p,"qqq" if n==5 else "qqqg",np.eye(len(p.matrix)))
    transverse=_apply_source_raw(p,s,tensor,component=c,transfer=tuple(D),
                                charge="electromagnetic",quadrature_order=8)
    np.testing.assert_allclose(b-a,np.einsum("j,jab->ab",BETA,transverse)[None],atol=5e-15)
    up,down=[cluster_quartic_source_smooth_minus_projector(p,s,D,component=c,minus_context=ZERO,
        **dict(ARGS,charge=q)) for q in ("up_number","down_number")]
    np.testing.assert_allclose(a,(2*up-down)/3,atol=5e-15)
    np.testing.assert_allclose(s.statistics_project(a[0]),a[0],atol=3e-15)
    norm=np.linalg.norm(a)
    print("quartic",n,c,"minus norm",norm,flush=True)
    if (n,c) in ((6,"valence_IG"),(6,"gluon_IG_ggg")):
        np.testing.assert_array_equal(a,0.)  # All gluon modes equal two at K9.
    else:assert norm>1e-8


@pytest.mark.parametrize("component",CHARGED_COMPONENTS+NEUTRAL_COMPONENTS)
def test_every_Q5_pair_smooth_minus_family_boost_adjoint_and_intrinsic_nonzero(component):
    m=model();x,y=raw(m.spaces5),raw(m.spaces5,701)
    actions=[ClusterFiveQuarticProjector.from_models(m,m,D,minus_context=c,**ARGS) for c in (ZERO,BOOST,None)]
    a,b,t=actions
    args=dict(component=0,components=[component])
    v=a.apply(x,**args);boost=b.apply(x,**args);back=a.adjoint(y,**args)
    ts=[t.apply(x,component=j,components=[component]) for j in range(2)]
    for n,s in m.spaces5.items():
        np.testing.assert_allclose(boost[n]-v[n],sum(BETA[j]*ts[j][n] for j in range(2)),atol=5e-15)
        np.testing.assert_allclose(five_statistics_project(s,v[n]),v[n],atol=3e-15)
    np.testing.assert_allclose(sum(np.vdot(y[n],v[n]) for n in v),
                               sum(np.vdot(back[n],x[n]) for n in x),atol=5e-15)
    norm=sum(np.linalg.norm(z) for z in v.values())
    print("pair",component,"minus norm",norm,flush=True)
    assert norm>1e-8
    if component in NEUTRAL_COMPONENTS:
        for j in range(2):
            for z in ts[j].values():np.testing.assert_allclose(z,0.,atol=3e-15)
    assert a.minus_projector_supplied and not a.complete_H2_matched_current


def test_no_implicit_minus_context_or_four_current_promotion():
    p=model().pilot
    with pytest.raises(ValueError):SmoothMinusProjectorContext(mean_total_transverse_GeV=(0,0),prescription="unknown")
    with pytest.raises(ValueError):exact_retained_smooth_minus_projector_coefficients(p,p,D,minus_context=None,**ARGS)
    with pytest.raises(ValueError):exact_retained_smooth_minus_projector_coefficients(p,p,D,minus_context=True,**ARGS)
    a=ClusterSixCubicProjector.from_models(model(),model(),D,minus_context=ZERO,**ARGS)
    with pytest.raises(ValueError):a.apply(raw(model().spaces5),component=1)


def test_equal_share_free_neutron_zero_does_not_remove_the_proton_minus_current():
    s=SixBodySpace("qqqggg",9,0,"PROTON",1)
    x=raw({s.sector:s})[s.sector]
    value=shell_free_smooth_minus_projector(s,s,x,D,b_GeV=B,total_pplus_GeV=PP,
        charge="electromagnetic",quadrature_order=8,minus_context=ZERO)
    assert np.linalg.norm(value)>1e-4

"""Full common-regulated H2: independent Neumann words, norms and raw actions."""
from functools import lru_cache
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import (
    build_cluster_pilot, GaussianClusterPilot, GaussianClusterH2, cluster_plus_density_between)
from deuteron_wigner.quantum.m2_gaussian_cluster_wick import GaussianClusterWick
from deuteron_wigner.quantum.m2_five_body_factorized import apply_terms
from deuteron_wigner.quantum.m2_six_body_cubic import five_statistics_project
from deuteron_wigner.quantum.m2_canonical_quark_self import quark_mass_squared_direction
from deuteron_wigner.quantum.m2_canonical_gluon_self import gluon_mass_squared_direction
from deuteron_wigner.quantum.m2_spin_resolved_current import transverse_parity_map
from test_m2_gaussian_cluster_regulator import regulator
from test_m2_interacting_q_loop_regulator import model as bare_model


@lru_cache(maxsize=3)
def model(q5=0,q6=0,target="PROTON",jz=1,width=.8,q=.011,g=-.007,coupling=.3,subtract=True):
    reg=regulator(width)
    wick=GaussianClusterWick(regulator=reg,quark_flavors=2,normal_order_subtraction=subtract,
        finite_quark_mass2_per_gs2_GeV2=q,finite_gluon_mass2_per_gs2_GeV2=g)
    p=build_cluster_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=coupling,
        regulator=reg,total_pplus_GeV=1.3,wick_operators=wick)
    return GaussianClusterH2(p,excluded_cutoff=q5,six_cutoff=q6)


def dot(a,b):
    return sum(a[n].conj().T@b[n] for n in a)


def independent_neumann(model,e):
    """Solve the coefficient recurrence, not the H2 split into named returns."""
    props={**model.qq.propagation,**model.six.propagation6}
    s1={**model.qq.sources_by_power[1],**{n:np.zeros_like(v) for n,v in model.six.W.items()}}
    s2={**model.qq.sources_by_power[2],**model.six.W}
    resolve=lambda x:{n:props[n].resolve(e,v) for n,v in x.items()}
    def u(x):
        return {**model.six.U.adjoint({n:x[n] for n in model.spaces6}),
                **model.six.U.apply({n:x[n] for n in model.spaces5})}
    def w(x):
        return {**model.qq.action.apply({n:x[n] for n in model.spaces5}),
                **{n:np.zeros_like(x[n]) for n in model.spaces6}}
    a1=resolve(s1)
    ua1=u(a1)
    a2=resolve({n:s2[n]+ua1[n] for n in s2})
    ua2,wa1=u(a2),w(a1)
    a3=resolve({n:ua2[n]+wa1[n] for n in ua2})
    g=model.pilot.coupling
    h=model.pilot.matrix+g*g*dot(s1,a1)+g**3*(dot(s1,a2)+dot(s2,a1))
    h+=g**4*(dot(s1,a3)+dot(s2,a2))
    norm=np.eye(len(h))+g*g*dot(a1,a1)+g**3*(dot(a1,a2)+dot(a2,a1))
    norm+=g**4*(dot(a2,a2)+dot(a1,a3)+dot(a3,a1))
    return h,norm


@pytest.fixture(scope="module",params=[("PROTON",0,0),("NEUTRON",0,0),("PROTON",1,1),("NEUTRON",1,0)])
def m(request):
    target,q5,q6=request.param
    return model(q5,q6,target)


def test_shared_regulator_all_H_blocks_and_complete_order4_Neumann_reference(m):
    assert isinstance(m.pilot,GaussianClusterPilot)
    assert m.reference.pilot is m.pilot and m.qq.reference is m.reference and m.six.reference is m.reference
    assert m.field_regulator==m.wick_operators.regulator
    for n,s in m.spaces5.items():
        assert m.qq.action.spaces[n] is s and m.six.U.incoming[n] is s
    snap=m.snapshot(1.4)
    h,n=independent_neumann(m,1.4)
    np.testing.assert_allclose(snap.effective_hamiltonian,h,atol=3e-14)
    np.testing.assert_allclose(snap.metric,n,atol=6e-15)
    np.testing.assert_allclose(snap.whitener@n@snap.whitener,np.eye(len(n)),atol=7e-15)
    assert set(m.six.direct_component_norms)=={
        "valence_IF","valence_IG","valence_sea","gluon_A4","gluon_IG_ggg","gluon_IF_pair","gluon_IG_pair"}
    assert not m.complete_regulated_H2_matching and not m.complete_gauged_contractions
    assert not m.common_regulator_renormalization_complete and not m.physical
    with pytest.raises(ValueError,match="four-current"):
        snap.current((.12,.16),charge="electromagnetic")


def test_independent_energy_derivative_of_full_common_regulated_H(m):
    e,eps=1.4,1e-4
    h=lambda x:m.snapshot(x).effective_hamiltonian
    derivative=(-h(e+2*eps)+8*h(e+eps)-8*h(e-eps)+h(e-2*eps))/(12*eps)
    np.testing.assert_allclose(derivative,np.eye(37)-m.snapshot(e).metric,atol=3e-11)


@pytest.mark.parametrize("charge",["baryon","electromagnetic","up_number","down_number"])
def test_two_energy_zero_transfer_density_matches_H_divided_difference(m,charge):
    q={"baryon":1,"electromagnetic":int(m.pilot.target=="PROTON"),
       "up_number":2 if m.pilot.target=="PROTON" else 1,
       "down_number":1 if m.pilot.target=="PROTON" else 2}[charge]
    f,i=m.snapshot(1.39),m.snapshot(1.47)
    raw=cluster_plus_density_between(f,i,(0.,0.),charge=charge,normalized=False)
    expected=q*(np.eye(37)-(f.effective_hamiltonian-i.effective_hamiltonian)/(1.39-1.47))
    np.testing.assert_allclose(raw,expected,atol=5e-14)
    np.testing.assert_allclose(i.plus_density((0.,0.),charge=charge,normalized=True),q*np.eye(37),atol=8e-15)


def test_all_raw_W55_components_and_U65_preserve_adjointness_and_statistics(m):
    rng=np.random.default_rng(623)
    def columns(spaces):
        return {n:(rng.normal(size=(s.dimension,2))+1j*rng.normal(size=(s.dimension,2)))/np.sqrt(s.dimension)
                for n,s in spaces.items()}
    a,b=columns(m.spaces5),columns(m.spaces5)
    aa,bb=m.qq.action.apply(a),m.qq.action.apply(b)
    np.testing.assert_allclose(dot(a,bb),dot(aa,b),atol=2e-14)
    parts=m.qq.action.component_returns(a)
    np.testing.assert_allclose(sum(parts.values()),dot(a,aa),atol=2e-14)
    projected={n:five_statistics_project(s,a[n]) for n,s in m.spaces5.items()}
    acted=m.qq.action.apply(projected)
    for n,s in m.spaces5.items():
        np.testing.assert_allclose(acted[n],five_statistics_project(s,acted[n]),atol=4e-14)
    z=columns(m.spaces6)
    forward,back=m.six.U.apply(a),m.six.U.adjoint(z)
    np.testing.assert_allclose(dot(z,forward),dot(back,a),atol=3e-15)
    for n,s in m.spaces6.items():
        np.testing.assert_allclose(forward[n],s.statistics_project(forward[n]),atol=3e-15)


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
def test_bare_limit_of_actual_PP_QQ_U65_and_full_H2(target):
    new=model(0,0,target,width=1e7,q=0.,g=0.)
    old=bare_model(0,1,target)
    np.testing.assert_allclose(new.pilot.matrix,old.pilot.matrix,atol=3e-14)
    for e in (1.39,1.47):
        a,b=new.snapshot(e),old.snapshot(e)
        np.testing.assert_allclose(a.effective_hamiltonian,b.effective_hamiltonian,atol=3e-14)
        np.testing.assert_allclose(a.metric,b.metric,atol=5e-15)
    for n in new.spaces5:
        for power in (1,2):
            np.testing.assert_allclose(new.qq.sources_by_power[power][n],old.qq.sources_by_power[power][n],atol=3e-14)
    for n in new.six.W:
        np.testing.assert_allclose(new.six.W[n],old.six.W[n],atol=3e-14)


def test_finite_kinetic_conditions_enter_PP_and_Q5_once_not_vertex_mass_or_Q6_at_order4():
    base=model(0,0,q=0.,g=0.)
    shifted=model(0,0,q=.021,g=-.013)
    e=1.4
    a,b=base.snapshot(e),shifted.snapshot(e)
    op=shifted.wick_operators
    pp=.021*quark_mass_squared_direction(base.pilot)-.013*gluon_mass_squared_direction(base.pilot)
    wave=base.qq.leading_waves(e,1)
    sw={n:op.shell_mass_diagonal(s)[:,None]*wave[n] for n,s in base.spaces5.items()}
    rs={n:base.qq.propagation[n].resolve(e,v) for n,v in sw.items()}
    dh=.3**2*pp+.3**4*dot(wave,sw)
    dn=.3**4*(dot(wave,rs)+dot(rs,wave))
    np.testing.assert_allclose(b.effective_hamiltonian-a.effective_hamiltonian,dh,atol=3e-14)
    np.testing.assert_allclose(b.metric-a.metric,dn,atol=5e-15)
    np.testing.assert_array_equal(base.pilot.emission_per_gs,shifted.pilot.emission_per_gs)
    for n in base.six.W:
        np.testing.assert_array_equal(base.six.W[n],shifted.six.W[n])


def test_opposite_helicity_parity_and_two_energy_oblique_density_reversal():
    plus,minus=model(1,1),model(1,1,jz=-1)
    f,i=minus.snapshot(1.39),plus.snapshot(1.47)
    a=cluster_plus_density_between(f,i,(.12,.16),charge="electromagnetic",normalized=True)
    b=cluster_plus_density_between(i,f,(-.12,-.16),charge="electromagnetic",normalized=True)
    np.testing.assert_allclose(a,b.conj().T,atol=6e-16)
    p=transverse_parity_map(minus.pilot,plus.pilot)
    np.testing.assert_allclose(minus.snapshot(1.47).effective_hamiltonian,
        p@i.effective_hamiltonian@p.conj().T,atol=4e-14)
    assert np.linalg.norm(a)>1e-6


def test_unsubtracted_common_Wick_operator_returns_through_PP_Q5_and_both_norms():
    base=model(0,0,q=0.,g=0.)
    ordered=model(0,0,q=0.,g=0.,subtract=False)
    e=1.4
    a,b=base.snapshot(e),ordered.snapshot(e)
    op=ordered.wick_operators
    wave=base.qq.leading_waves(e,1)
    sw={n:op.shell_mass_diagonal(s)[:,None]*wave[n] for n,s in base.spaces5.items()}
    rs={n:base.qq.propagation[n].resolve(e,v) for n,v in sw.items()}
    dh=.3**2*op.retained_mass_coefficient(base.pilot)+.3**4*dot(wave,sw)
    dn=.3**4*(dot(wave,rs)+dot(rs,wave))
    assert np.linalg.norm(dh)>1e-3 and np.linalg.norm(dn)>1e-7
    np.testing.assert_allclose(b.effective_hamiltonian-a.effective_hamiltonian,dh,atol=3e-14)
    np.testing.assert_allclose(b.metric-a.metric,dn,atol=5e-15)


def test_retained_shell_embedding_binds_each_regulated_PP_component_without_refitting():
    from deuteron_wigner.quantum.m2_matched_h2_reduction import nested_pilot_isometry
    small=model().pilot
    large=build_cluster_pilot(total_k2=9,cutoff=1,target=small.target,jz_twice=small.jz_twice,
        b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,
        regulator=small.field_regulator,total_pplus_GeV=1.3,wick_operators=small.wick_operators)
    p=nested_pilot_isometry(small,large)
    for key in ("h0","cross_pairs_per_gs2","emission_per_gs",
                "instantaneous_fermion_per_gs2","wick_mass_per_gs2","matrix"):
        np.testing.assert_allclose(p.conj().T@getattr(large,key)@p,getattr(small,key),atol=4e-14)


def test_independent_Q5_radial_and_Q6_resolution_with_non_diagonal_free_propagation():
    m=model(2,1,q=0.,g=0.)
    s=m.snapshot(1.4)
    h,n=independent_neumann(m,1.4)
    np.testing.assert_allclose(s.effective_hamiltonian,h,atol=3e-14)
    np.testing.assert_allclose(s.metric,n,atol=7e-15)
    lower=model(1,1,q=0.,g=0.).snapshot(1.4)
    assert np.linalg.norm(s.effective_hamiltonian-lower.effective_hamiltonian)>1e-7
    assert np.linalg.norm(s.metric-lower.metric)>1e-8
    # A real change is resolution dependence, not an established physical limit.


def test_field_envelope_does_not_remove_free_basis_spectators_or_invent_Wick_terms():
    from dataclasses import replace
    reg=replace(regulator(),boson_modes=1,fermion_modes=2)
    w=GaussianClusterWick(regulator=reg,quark_flavors=2,normal_order_subtraction=False,
        finite_quark_mass2_per_gs2_GeV2=.011,finite_gluon_mass2_per_gs2_GeV2=-.007)
    assert w.mass_components("q",5,total_k2=9)==(0.,0.,.011*9/5)
    assert w.mass_components("g",4,total_k2=9)==(0.,0.,-.007*9/4)
    p=build_cluster_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,
        b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,
        regulator=reg,total_pplus_GeV=1.3,wick_operators=w)
    m=GaussianClusterH2(p,excluded_cutoff=0,six_cutoff=0)
    s=m.snapshot(1.4)
    assert s.matrix.shape==(37,37)
    np.testing.assert_allclose(p.h0,model().pilot.h0,atol=0.)
    np.testing.assert_allclose(s.effective_hamiltonian,s.effective_hamiltonian.conj().T,atol=3e-14)


def test_cannot_mix_field_flavors_finite_conditions_or_unequal_resolutions_at_current_endpoints():
    from dataclasses import replace
    p=model().pilot
    args=dict(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,
        b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,
        regulator=p.field_regulator,total_pplus_GeV=1.3,wick_operators=p.wick_operators)
    for bad in (dict(regulator=None),dict(total_pplus_GeV=0.),
        dict(wick_operators=replace(p.wick_operators,quark_flavors=1)),
        dict(wick_operators=replace(p.wick_operators,regulator=regulator(.7)))):
        with pytest.raises(ValueError):
            build_cluster_pilot(**{**args,**bad})
    i=model().snapshot(1.4)
    for f in (model(width=.7).snapshot(1.4),model(q=.012).snapshot(1.4),
              model(1,0).snapshot(1.4)):
        with pytest.raises(ValueError):
            cluster_plus_density_between(f,i,(.12,.16),charge="electromagnetic",normalized=True)

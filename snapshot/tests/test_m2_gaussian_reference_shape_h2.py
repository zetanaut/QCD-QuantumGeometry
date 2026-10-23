"""Whole-vertex reference binding, correlated halves and common H2 words."""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_laguerre
from deuteron_wigner.quantum.m2_gaussian_reference_shape import GaussianReferenceShape,radial_half_moment
from deuteron_wigner.quantum.m2_gaussian_cluster_interactions import (
    cluster_pair_contact,cluster_pair_integral_reference,cluster_pair_current_orbital)
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import (
    build_cluster_pilot,GaussianClusterH2,cluster_plus_density_between,require_same_cluster_definition)
from deuteron_wigner.quantum.m2_gaussian_cluster_wick import GaussianClusterWick,cluster_wick_summands
from deuteron_wigner.quantum.m2_gaussian_cluster_wick_current import cluster_contraction_paths,ordered_cluster_wick_terms
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_Q_pair_projector import ClusterFiveQuarticProjector
from test_m2_gaussian_cluster_interactions import CASES
from test_m2_gaussian_cluster_regulator import regulator
from test_m2_gaussian_cluster_h2 import independent_neumann,model,dot
from test_m2_gaussian_cluster_quartic_pair_actions import hamiltonian_terms
from test_m2_gaussian_cluster_Q_projector import D,PP,ARGS,raw


def shaped(degree=1,width=.8,reference=1.2):
    return replace(regulator(width),reference_shape=GaussianReferenceShape(
        base_width_GeV=width,reference_width_GeV=reference,degree=degree))


@lru_cache(maxsize=6)
def shape_model(degree=1,q5=0,q6=0,reference=1.2,subtract=True):
    reg=shaped(degree,reference=reference)
    wick=GaussianClusterWick(regulator=reg,quark_flavors=2,normal_order_subtraction=subtract,
        finite_quark_mass2_per_gs2_GeV2=.011,finite_gluon_mass2_per_gs2_GeV2=-.007)
    p=build_cluster_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,
        regulator=reg,total_pplus_GeV=1.3,wick_operators=wick)
    return GaussianClusterH2(p,excluded_cutoff=q5,six_cutoff=q6)


@pytest.mark.parametrize("n",[0,1,2])
@pytest.mark.parametrize("order",[0,1,2,3])
@pytest.mark.parametrize("t",[.2,1.,2.3])
def test_radial_derivatives_against_independent_Laguerre_integral(n,order,t):
    scale=.37
    expected=quad(lambda r:(-1)**n*eval_laguerre(n,2*r)*r**order*np.exp(-(1+t)*r),
                  0,np.inf,epsabs=2e-12,epsrel=2e-12)[0]*scale**order
    np.testing.assert_allclose(radial_half_moment(n,t,order,scale),expected,atol=8e-14,rtol=2e-12)


@pytest.mark.parametrize("case",CASES)
@pytest.mark.parametrize("degree,width,ci,co",[(1,.8,0,2),(2,.2,2,1),(2,.8,2,2)])
def test_correlated_contacts_match_direct_whole_vertex_integrals(case,degree,width,ci,co):
    inc,out,pair,si,so=case;reg=shaped(degree,width)
    args=dict(regulator=reg,species=si,out_species=so)
    got=cluster_pair_contact(inc,out,pair,.4,ci,co,**args)
    expected=cluster_pair_integral_reference(inc,out,pair,.4,ci,co,
        **args,total_transverse_GeV=(.23,-.17))
    np.testing.assert_allclose(got,expected,atol=8e-15,rtol=5e-12)
    reverse=cluster_pair_contact(out,inc,pair,.4,co,ci,regulator=reg,species=so,out_species=si)
    np.testing.assert_allclose(got,reverse.conj().T,atol=8e-15)


@pytest.mark.parametrize("degree",[0,1,3])
def test_bound_Wick_trace_current_and_analytic_derivatives_use_shape_exactly_once(degree):
    reg=shaped(degree);base=replace(reg,reference_shape=None);shape=reg.reference_shape
    for species,p in (("q",3),("qbar",3),("g",2)):
        bound=cluster_wick_summands(species,p,regulator=reg,quark_flavors=2)
        explicit=cluster_wick_summands(species,p,regulator=base,quark_flavors=2,reference_shape=shape)
        for key in bound:np.testing.assert_array_equal(bound[key][1],explicit[key][1])
        args=dict(external_mode=p,loop_mode=5,transfer_GeV=D)
        assert cluster_contraction_paths(regulator=reg,**args)==cluster_contraction_paths(
            regulator=base,reference_shape=shape,**args)
        kw=dict(quark_flavors=2,total_k2=9,total_pplus_GeV=PP,transfer_GeV=D,
            external_charge=0. if species=="g" else .7,loop_flavor_charges=(2/3,-1/3),prescription=ARGS["prescription"])
        assert ordered_cluster_wick_terms(species,p,regulator=reg,**kw)==ordered_cluster_wick_terms(
            species,p,regulator=base,reference_shape=shape,**kw)
    p,k=3.2,4.7;out=np.array([.23,-.17]);inc=np.array([-.07,.04])
    args=dict(external_mode=p,loop_mode=k,outgoing_transverse_GeV=out,incoming_transverse_GeV=inc)
    derivative=reg.contracted_kernel_derivatives(**args);eps=1e-4
    for key,result in derivative.items():
        name=key+"_GeV" if "transverse" in key else key
        v=args[name]
        for j in range(2 if np.ndim(v) else 1):
            shift=np.eye(2)[j]*eps if np.ndim(v) else eps
            fun=lambda q:reg.contracted_kernel_trace(**{**args,name:v+q*shift})
            measured=(-fun(2)+8*fun(1)-8*fun(-1)+fun(-2))/(12*eps)
            np.testing.assert_allclose(measured,result[j] if np.ndim(v) else result,atol=2e-12)
    c=reg.contraction_trace_GeV2(external_mode=p,loop_mode=k)
    np.testing.assert_allclose(c,reg.contracted_kernel_trace(**{**args,"incoming_transverse_GeV":out}),atol=2e-16)


@pytest.mark.parametrize("degree,q5",[(1,0),(2,0),(1,1)])
def test_correlated_auxiliary_pair_equals_independent_four_physical_legs_on_H2_waves(degree,q5):
    m=shape_model(degree,q5);x=m.snapshot(1.4).five_wave_coefficients[0]
    a=ClusterFiveQuarticProjector.from_models(m,m,D,**ARGS)
    for origin,target,component in (("qqqq-qbar","qqqq-qbar","sea_annihilation"),
            ("qqqgg","qqqq-qbar","IG_conversion"),("qqqq-qbar","qqqgg","IG_conversion"),
            ("qqqgg","qqqgg","gg_annihilation")):
        probe={n:v if n==origin else np.zeros_like(v) for n,v in x.items()}
        got=m.qq.action.pair_current.apply(probe)
        expected=hamiltonian_terms(a,probe,[component])
        np.testing.assert_allclose(got[target],expected[target],atol=2e-14)
        assert np.linalg.norm(expected[target])>1e-8
    u,v=raw(m.spaces5,601),raw(m.spaces5,603)
    pu,pv=m.qq.action.pair_current.apply(u),m.qq.action.pair_current.apply(v)
    np.testing.assert_allclose(dot(u,pv),dot(pu,v),atol=2e-14)


@pytest.mark.parametrize("degree,reference",[(0,1.2),(2,.8)])
def test_zero_shape_reproduces_original_H2(degree,reference):
    new=shape_model(degree,reference=reference);old=model()
    np.testing.assert_array_equal(new.pilot.matrix,old.pilot.matrix)
    for e in (1.39,1.47):
        a,b=new.snapshot(e),old.snapshot(e)
        np.testing.assert_array_equal(a.effective_hamiltonian,b.effective_hamiltonian)
        np.testing.assert_array_equal(a.metric,b.metric)


@pytest.mark.parametrize("degree,subtract",[(1,True),(2,True),(1,False)])
def test_shaped_full_H2_recurrence_energy_derivative_and_both_norms(degree,subtract):
    m=shape_model(degree,subtract=subtract);e=1.4;s=m.snapshot(e)
    h,n=independent_neumann(m,e)
    np.testing.assert_allclose(s.effective_hamiltonian,h,atol=3e-14)
    np.testing.assert_allclose(s.metric,n,atol=6e-15)
    np.testing.assert_allclose(s.whitener@n@s.whitener,np.eye(len(n)),atol=8e-15)
    eps=1e-4;H=lambda z:m.snapshot(z).effective_hamiltonian
    derivative=(-H(e+2*eps)+8*H(e+eps)-8*H(e-eps)+H(e-2*eps))/(12*eps)
    np.testing.assert_allclose(derivative,np.eye(len(h))-n,atol=4e-11)
    f,i=m.snapshot(1.39),m.snapshot(1.47)
    density=cluster_plus_density_between(f,i,(0.,0.),charge="electromagnetic",normalized=False)
    expected=np.eye(len(h))-(f.effective_hamiltonian-i.effective_hamiltonian)/(f.energy_GeV2-i.energy_GeV2)
    np.testing.assert_allclose(density,expected,atol=7e-14)
    assert np.linalg.norm(s.effective_hamiltonian-model(subtract=subtract).snapshot(e).effective_hamiltonian)>1e-6


def test_H_current_shape_mismatch_is_not_silently_overridden():
    reg=shaped()
    with pytest.raises(ValueError,match="identical"):
        reg.shape_for(replace(reg.reference_shape,degree=2))
    with pytest.raises(ValueError,match="base Gaussian width"):
        replace(reg,width_GeV=.4)
    with pytest.raises(ValueError,match="identical"):
        require_same_cluster_definition(shape_model().pilot,model().pilot)
    from deuteron_wigner.quantum.m2_excluded_fock_kinematics import split_routes
    with pytest.raises(ValueError,match="correlated"):
        cluster_pair_current_orbital(split_routes(9,"g_to_gg")[0],0,.4,regulator=reg)


def test_finite_H2_H_and_norm_matrix_errors_decrease_toward_named_reference():
    reference=model(width=1.2).snapshot(1.4)
    members=[model().snapshot(1.4),shape_model(1).snapshot(1.4),shape_model(2).snapshot(1.4)]
    result={}
    for key in ("effective_hamiltonian","metric"):
        target=getattr(reference,key)
        errors=[np.linalg.norm(getattr(s,key)-target)/np.linalg.norm(target) for s in members]
        # Measured finite-matrix control, not a universal convergence bound
        # or a physical condition. The fourth current coefficient is tested
        # separately so a free-H-dominated total cannot certify its error.
        assert 0<errors[2]<.1*errors[1]<.01*errors[0]
        result[key]=[float(x) for x in errors]
    print("finite H2 reference matrix errors",result,flush=True)


def test_saved_full_Q_shell_control_does_not_hide_order_four_resolution_error():
    root=Path(__file__).resolve().parents[1]/"docs/next_level"
    prefix="m2_gaussian_cluster_four_current_"
    base=json.loads((root/(prefix+"baseline.json")).read_text())
    larger=json.loads((root/(prefix+"Q5_Q6_shell1.json")).read_text())
    for key in ("total_k2","retained_cutoff","jz_twice","energy_out_GeV2","energy_in_GeV2", "transfer_GeV"):
        assert base[key]==larger[key]
    with np.load(root/base["full_matrix_file"]) as a,np.load(root/larger["full_matrix_file"]) as b:
        from scripts.benchmark_m2_gaussian_cluster_four_current import compare_matrices
        diff=compare_matrices(a,b)
        assert diff["current_power_0"]["bitwise_identical"] and diff["current_power_1"]["bitwise_identical"]
        np.testing.assert_allclose(diff["current_power_4"]["relative_difference"],.9053418556828291,rtol=2e-12)
        assert diff["normalized_current"]["relative_difference"]<.0002
        assert set(a)==set(b) and len(a)==11
    assert larger["Ward_Frobenius"]<3e-14 and not larger["continuum_convergence_established"]


def test_bound_cubic_current_and_Hamiltonian_factory_share_explicit_shape():
    from deuteron_wigner.quantum.m2_gaussian_cluster_current import cluster_cubic_current_orbitals
    from deuteron_wigner.quantum.m2_gaussian_cluster_projector_kernel import cubic_hamiltonian_orbital
    from test_m2_gaussian_cluster_current import ROUTES
    reg=shaped();base=replace(reg,reference_shape=None);route=ROUTES[0]
    args=dict(total_pplus_GeV=PP,transfer_GeV=D,mean_total_transverse_GeV=(.03,.07),
        flavor_charge=.7,prescription=ARGS["prescription"],quadrature_order=8)
    bound=cluster_cubic_current_orbitals(route,0,1,.4,.35,(1,-1,1),regulator=reg,**args)
    explicit=cluster_cubic_current_orbitals(route,0,1,.4,.35,(1,-1,1),regulator=base,
        reference_shape=reg.reference_shape,**args)
    np.testing.assert_array_equal(bound.total,explicit.total)
    assert bound.reference_shape is reg.reference_shape
    h=cubic_hamiltonian_orbital(route,(1,-1,1),b_GeV=.4,mass_GeV=.35,regulator=reg,
        total_pplus_GeV=PP,total_transverse_GeV=(0.,0.))
    assert h.reference_shape is reg.reference_shape
    unshaped=replace(h,reference_shape=None)
    assert np.linalg.norm(h.integrate(1,0)-unshaped.integrate(1,0))>1e-8

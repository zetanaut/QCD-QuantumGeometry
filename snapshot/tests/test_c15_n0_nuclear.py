import numpy as np, pytest
from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.nuclear.n0.core import *
from deuteron_wigner.nuclear.n0.injections import INJECTIONS

def test_plans_members_are_correlated_and_exclusive():
    p=plans(); assert len(p)==4 and len({x.plan_id for x in p})==4
    m=correlated_member(p[0]); assert m.proton_h7_member!=m.neutron_h7_member
    assert compile_plan((p[0].plan_id,),2)["scope"]=="NN_ONLY_N0_VALIDATION"
    with pytest.raises(ArchitectureError): compile_plan((p[0].plan_id,p[1].plan_id),2)
    with pytest.raises(ArchitectureError): compile_plan((p[0].plan_id,),3)
    with pytest.raises(ArchitectureError): compile_plan((p[0].plan_id,),2,"production")

def test_recoil_exact_closures_and_reversal():
    r=recoil(.43,(.12,-.07),(.18,.09)); c=recoil_closure(r); assert max(c.values())<1e-14
    rr=recoil(.43,(.12,-.07),(-.18,-.09)); assert np.allclose(r.kappa_in,rr.kappa_out) and np.allclose(r.kappa_out,rr.kappa_in)

def test_spin1_state_normalization_interference_and_limits():
    r=state_report(); assert r["shape"]==(3,6) and r["normalization_residual"]<1e-14 and r["sd_interference"]!=0
    assert state_report(build_state(d_probability=0))["d_probability"]==0

def test_spin1_projector_complete_and_ll_adapter():
    r=projector_report(); assert r["gram_rank"]==9 and r["gram_condition"]<1.0000000001
    assert r["reconstruction_residual"]<1e-13 and r["ll_adapter_residual"]==0

def test_spectral_full_helicity_positive_and_reconstructs():
    r=spectral_report(); assert r["shape"]==(6,6) and r["forward_min_eigenvalue"]>-1e-13
    assert r["hermiticity_reversal_residual"]==r["full_bond_residual"]==0 and r["quadrature_residual"]<1e-6

def test_deuteron_parents_all_species_and_wilson_orders():
    for s in SPECIES:
        for o in (0,1,2):
            p=deuteron_parent(s,wilson_order=o); assert p.values.shape==(6,6) and p.wilson_order==o
            assert reductions(p)["maximum_residual"]==0
            if o==0: assert np.max(abs(p.values-p.values.conj().T))<1e-14
    g=deuteron_parent("g"); assert g.ordered_links==("++","+-","-+","--") and g.color_channels==("f","d")

def test_link_odd_zero_at_order_zero_and_future_past_identity():
    p=deuteron_parent("u",wilson_order=0); assert np.max(abs(p.values-p.values.conj().T))<1e-14
    assert deuteron_parent("ubar",wilson_order=2).color_channels==("antifundamental",)

def test_b1_tensor_closure_without_independent_normalization():
    r=b1_report(); assert r["b1"]!=0 and r["pure_s_tensor"]==0
    assert r["ll_adapter_residual"]==r["direct_projection_residual"]==r["reduction_residual"]==0
    assert r["reduced_bond_signal_loss"]>.5

def test_current_charge_angular_and_moment_closure():
    r=current_report(); assert r["G_C_0"]==1 and r["charge_residual"]==0
    assert r["angular_condition_residual"]<1e-12 and r["gtmd_moment_residual"]<1e-12

def test_offshell_unitary_covariance_requires_induced_term():
    r=offshell_report(); assert r["full_invariance_residual"]<1e-12 and r["one_body_noninvariance"]>0
    assert r["induced_two_body_norm"]>0 and r["visible_remainder"]>0

def test_tagged_inclusive_and_cp_reductions():
    t=tagged_report(); c=cp_report(); assert t["inclusive_residual"]==t["spectator_recoil_residual"]==0
    assert t["reduced_bond_tensor_loss"]>.4 and not t["acceptance_in_amplitude"]
    assert c["choi_min_eigenvalue"]>=0 and c["trace_residual"]==0 and c["trace_after_interference"]

def test_ttn_full_bond_and_visible_tensor_loss():
    r=ttn_report(); assert r["direct_full_bond_residual"]==0 and r["low_bond_tensor_loss"]>.5
    assert r["low_bond_norm_error"]<.002

def test_sensitivity_axes_separate_common_member():
    r=sensitivity_report(); assert len(r["axes"])==5 and all(not x["combined"] for x in r["axes"])

def test_provenance_count_once_and_no_downstream_edges():
    r=provenance_report(); assert r["count_once_residual"]==0 and not r["unresolved_cycles"] and r["production_edges"]==0

def test_readiness_nonclaims_and_unavailable_sectors():
    r=readiness_report(); assert not r["production_reachable"] and "PRODUCTION_READY" in r["not_issued"]
    assert set(UNAVAILABLE)==set(r["unavailable_sectors"])

def test_c15_injections(): assert len(INJECTIONS)>=200 and len({x[0] for x in INJECTIONS})==len(INJECTIONS)

import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.microscopic.h7.core import *
from deuteron_wigner.microscopic.h7.diagnostics import *
from deuteron_wigner.microscopic.h7.injections import INJECTIONS


def test_h7_color_multiplicity_and_s3_s2_content():
    r=color_permutation_report(); assert [x["multiplicity"] for x in r["rows"]]==[22,28,28]
    assert r["rows"][0]["permutation_content"]=={"S3_SYMMETRIC":4,"S3_ANTISYMMETRIC":4,"S3_MIXED_COPIES":7,"S3_MIXED_STATES":14}
    assert all(x["permutation_content"]=={"S2_SYMMETRIC":14,"S2_ANTISYMMETRIC":14} for x in r["rows"][1:])
    assert derive_new_color_multiplicity("QQQGGG")[0]==22
    assert derive_new_color_multiplicity("QQQUUBARGG")[0]==28
    assert max(x["generator_residual"] for x in r["rows"])==0


def test_gluon_product_statistics_fail_closed():
    for x in (("S","S"),("A","A"),("M","M")): GluonPermutationState(*x)
    with pytest.raises(ArchitectureError): GluonPermutationState("A","S")


def test_ten_sector_tower_increases_nontrivially():
    t=basis_tower(); assert [b.dimension for b in t]==[140,227,314]
    assert all(len(b.specs)==10 for b in t) and t[0].dimensions!=t[1].dimensions


def test_hamiltonian_exact_krylov_matrix_free_and_full_bond():
    r=hamiltonian_report(); assert r["maximum_hermiticity_residual"]==r["maximum_matrix_free_residual"]==0
    assert r["maximum_krylov_residual"]<1e-10 and r["maximum_full_bond_residual"]==0
    assert all(x["generated_adjoint_count"]==len(SUPPORTED_LINKS) for x in r["rows"])


def test_renormalization_refits_and_retains_null_holdouts():
    rows=renormalization_trajectory(); assert all(abs(x["mass_residual"])<1e-12 for x in rows)
    assert all(x["null_directions"]==1 and set(x["unfitted_holdouts"])=={"antiquark_order2","gluon_order2"} for x in rows)
    assert len({x["parameters"]["counterterm"] for x in rows})==3


def test_all_species_order_two_supported_order_three_fails():
    s=support_table(); assert all(s[x][2]=="EXPLICIT_FOCK_SUPPORTED" for x in s)
    assert all(s[x][3]=="UNAVAILABLE_AT_THIS_WILSON_ORDER" for x in s)
    for species in s: assert require_support(species,2)=="EXPLICIT_FOCK_SUPPORTED"
    with pytest.raises(ArchitectureError): require_support("gluon",3)


def test_dyson_magnus_all_representations_and_topologies():
    r=dyson_magnus_report(); assert r["maximum_dyson_magnus_residual"]<1e-13
    assert r["fundamental_antifundamental_conjugation_residual"]==0 and r["adjoint_algebra_residual"]<1e-13
    assert {x["topology"] for x in r["rows"] if x["representation"]=="two_link"}=={"left_left","right_right","left_right","right_left"}
    assert r["commutator_required"] and r["defect_scaling_order"]==3


def test_spectral_cut_count_once_no_epsilon_or_squared_delta():
    r=spectral_cut_report(); assert not r["physical_epsilon_used"] and not r["squared_delta_used"]
    assert r["two_cell_count_once_residual"]==r["future_past_residual"]==0
    assert all(x["finite_volume_residual"]<5e-6 and x["double_cut_real"]>0 for x in r["rows"])


def test_soft_overlap_each_geometry_and_signed_ablations():
    r=soft_overlap_report(); assert len(r["rows"])==4
    for x in r["rows"]:
        assert x["rapidity_derivative_residual"]==x["dyson_magnus_residual"]==0
        assert x["missing_s1w1"]==-x["duplicate_s1w1"] and x["missing_s2"]==-x["duplicate_s2"]


def test_finite_gauge_closure_and_nonclaim():
    r=gauge_closure_report(); assert r["residual"]==0 and not r["full_slavnov_taylor_closure"]
    assert all(x!=0 for x in r["ablation_residuals"].values())


def test_order_resolved_matrix_parents_and_distinct_projectors():
    r=matrix_parent_report(); assert [x["shape"] for x in r["parents"]]==[(4,4),(4,4),(3,3,2,2)]
    assert r["sivers_boer_mulders_distinct"] and r["ordered_links_independent"]
    assert r["f_d_reconstruction_residual"]==r["orthogonal_color_residual"]==0


def test_explicit_induced_transformed_operator_remainder_visible():
    rows=explicit_induced_comparison(); assert {x["species"] for x in rows}=={"antiquark","gluon"}
    assert all(x["remainder_norm"]>0 and x["operator_transform_residual"]==0 for x in rows)


def test_ttn_observable_loss_and_separate_convergence_axes():
    r=convergence_report(); assert r["full_bond_exact"]
    assert r["reduced_bond_antiquark_wilson_loss"]>.4 and r["reduced_bond_gluon_wilson_loss"]>.4
    assert len({x["axis"] for x in r["axes"]})==len(r["axes"]) and all(not x["combined"] for x in r["axes"])


def test_plan_compiler_and_downstream_gates():
    p=plans()[0]; assert compile_plan((p.plan_id,),2)["scope"]=="H7_VALIDATION_ONLY"
    with pytest.raises(ArchitectureError): compile_plan(tuple(x.plan_id for x in plans()),2)
    with pytest.raises(ArchitectureError): compile_plan((p.plan_id,),3)
    with pytest.raises(ArchitectureError): compile_plan((p.plan_id,),2,"nuclear")
    r=prediction_plan_report(); assert not r["production_reachable"] and "PRODUCTION_READY" in r["not_issued"]


def test_h7_injection_inventory():
    assert len(INJECTIONS)>=168 and len({x[0] for x in INJECTIONS})==len(INJECTIONS)

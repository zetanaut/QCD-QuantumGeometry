import numpy as np,pytest
from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.microscopic.h6.core import *
from deuteron_wigner.microscopic.h6.diagnostics import *
from deuteron_wigner.microscopic.h6.injections import INJECTIONS
def test_color_nullspace_multiplicities():
 r=color_report();assert [x["multiplicity"] for x in r["rows"]]==[6,8,8] and max(x["generator_residual"] for x in r["rows"])==0
def test_statistics_and_incomplete_boson_rejected():
 TwoGluonExchangeSymmetry(1,1);TwoGluonExchangeSymmetry(-1,-1)
 with pytest.raises(ArchitectureError):TwoGluonExchangeSymmetry(-1,1)
 assert color_report()["antisymmetry_residual"]==color_report()["bosonic_exchange_residual"]==0
def test_plans_exclusive_and_basis_growth():
 assert len({p.plan_id for p in plans()})==2
 assert [b.dimensions for b in basis_tower()]==[(4,6,9,9,12,16,16),(7,10,15,15,20,24,24),(10,14,21,21,28,32,32)]
def test_hamiltonian_hermitian_krylov_fullbond():
 r=hamiltonian_report();assert r["maximum_hermiticity"]==r["maximum_full_bond"]==0 and r["maximum_krylov"]<1e-10
 assert all(x["low_bond_wilson_loss"]>.4 for x in r["rows"])
def test_renormalization_refit_and_null_direction():
 r=renormalization_trajectory();assert len(r)==3 and all(abs(x["mass_residual"])<1e-12 for x in r)
 assert all(x["null_directions"]==1 for x in r) and len({x["parameters"]["counterterm"] for x in r})==3
def test_support_table_and_fail_closed_orders():
 s=support_table();assert all(s[x][1]=="EXPLICIT_FOCK_SUPPORTED" for x in s)
 assert s["quark"][2]=="EXPLICIT_FOCK_SUPPORTED" and s["antiquark"][2]==s["gluon"][2]=="UNAVAILABLE_AT_THIS_FOCK_ORDER"
 with pytest.raises(ArchitectureError):require_support("gluon",2)
def test_dyson_magnus_commuting_noncommuting_and_commutator_ablation():
 r=dyson_report();assert r["maximum_dyson_magnus"]<1e-14 and r["commutator_required"]
 assert all(x["commuting_omega2"]==0 and x["commuting_residual"]<1e-14 for x in r["rows"])
def test_order2_scaling_path_and_unitarity():
 r=dyson_report();assert r["scaling_order"]==r["order2_unitarity_scaling"]==3
 assert r["path_composition_residual"]==r["path_reversal_residual"]==0
 assert r["rows"][0]["dyson_exact"]>r["rows"][-1]["dyson_exact"]
def test_second_order_spectral_cut_bookkeeping():
 r=cut_report();assert r["below_threshold_zero"] and r["future_past_residual"]==r["ledger_residual"]==0
 assert r["finite_volume_residual"]<5e-6 and not r["squared_delta_used"]
def test_second_order_soft_signed_ablations():
 r=soft_report();assert r["central"]["rapidity_residual"]==r["dyson_magnus_subtracted_residual"]==0
 assert r["missing_s1w1"]["rapidity_residual"]==-r["duplicate_s1w1"]["rapidity_residual"]
 assert r["missing_s2"]["rapidity_residual"]==-r["duplicate_s2"]["rapidity_residual"]
def test_gauge_closure_requires_owned_terms():
 r=gauge_report();assert abs(r["residual"])<1e-15 and all(v!=0 for v in r["omission_residuals"].values())
def test_explicit_induced_comparisons_not_fitted():
 r=explicit_induced_comparison();assert {x["species"] for x in r}=={"ubar","dbar","gluon"}
 assert all(x["explicit"]!=x["induced_remainder"] for x in r)
def test_convergence_and_capability_gates():
 r=convergence_report();c=capability_report();assert r["exact_full_bond"]==0 and r["low_bond_wilson_loss"]>.4
 assert not c["production_reachable"] and "SECOND_ORDER_GLUON_READY" in c["not_issued"]
def test_injection_inventory():assert len(INJECTIONS)==148 and len({x[0] for x in INJECTIONS})==148

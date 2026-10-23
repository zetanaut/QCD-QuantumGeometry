import numpy as np,pytest
from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.nuclear.n1.core import *
from deuteron_wigner.nuclear.n1.injections import INJECTIONS
def test_plans_compile_exclusively():
 p=plans();assert len(p)==4 and compile_plan((p[0].plan_id,))["scope"]=="C16_N1_VALIDATION_ONLY"
 with pytest.raises(ArchitectureError):compile_plan((p[0].plan_id,p[1].plan_id))
 with pytest.raises(ArchitectureError):compile_plan((p[0].plan_id,),subtraction=False)
 with pytest.raises(ArchitectureError):compile_plan((p[-1].plan_id,),coherent_overlap=False)
 with pytest.raises(ArchitectureError):compile_plan((p[0].plan_id,),downstream="production")
def test_three_body_coordinates_and_recoil():
 c=sample_coordinates();assert abs(sum(c.fractions)-1)<1e-14
 for active in range(3):
  r=diagonal_recoil(c,active);assert r["intrinsic_residual"]<1e-14 and r["physical_transfer_residual"]==r["spectator_residual"]==0 and r["jacobian"]==1
 t=transition_recoil(c);assert t["momentum_residual"]==t["reverse_residual"]==0
 with pytest.raises(ArchitectureError):ThreeBodyCoordinates((.4,.4,.3),c.kappa)
def test_nnpi_basis_quantum_numbers():
 b=basis_manifest();assert tuple(b["charge_channels"])==CHARGE_CHANNELS and b["total_charge"]==1 and b["total_isospin"]==0 and b["parity"]==1 and b["J"]==1
 assert b["orthonormality_residual"]==b["exchange_residual"]==0
def test_tower_hamiltonian_and_flow():
 assert [b.dimension for b in basis_tower()]==[30,52,78]
 r=hamiltonian_report();assert r["max_hermiticity"]==r["max_matrix_free"]==0 and r["max_krylov"]<1e-12 and r["normalization_residual"]<1e-14
 assert all(x["null_directions"]==1 for x in r["rows"])
def test_state_ledgers():
 r=state_report();assert r["Z_NNPI"]>0 and abs(r["Z_NN"]+r["Z_NNPI"]-1)<1e-14
 assert max(r[k] for k in ("normalization_residual","baryon_residual","charge_residual","plus_momentum_residual","isospin_residual","parity_residual","jz_residual"))==0
def test_pion_active_and_transition_operators():
 r=operator_report();assert r["pion_active_residual"]==r["transition_hermiticity_residual"]==r["transition_recoil_residual"]==0
 assert {x["species"] for x in r["pion_species"]}==set(SPECIES)
def test_pion_subtraction_signed_and_stable():
 r=subtraction_report();assert r["two_cell_residual"]==0 and r["matched_variation"]<r["truncation_tolerance"]
 assert r["missing_residual"]==-r["duplicate_residual"] and r["missing_residual"]>0
def test_current_continuity_signed_ablations():
 r=current_report();assert abs(r["continuity_residual"])<1e-14 and all(x!=0 for x in r["ablation_residuals"].values())
 assert r["angular_condition_residual"]<1e-12 and r["gtmd_moment_residual"]<1e-12
def test_coherent_helicity_pilot():
 r=coherent_report();assert coherent_report(0)["scalar"]==0 and r["scalar"]!=r["tensor"]
 assert r["order_reversal_residual"]==0 and r["copied_ratio_failure"]>0 and r["early_trace_error"]>0
def test_parton_nuclear_overlap_count_once():
 c=overlap_report();m=overlap_report(0);d=overlap_report(2);assert c["residual"]==0 and m["residual"]==-d["residual"]
 assert c["partonic_path_immutable"] and c["identities_distinct"]
def test_cp_after_coherence():
 r=cp_report();assert r["choi_min_eigenvalue"]>=0 and r["trace_residual"]==0 and r["kraus_partial_trace_residual"]<1e-12 and r["early_trace_interference_error"]>0
def test_parent_all_species_orders_reductions():
 r=parent_report();assert len(r["rows"])==15 and r["common_parent_residual"]==r["b1_adapter_residual"]==0
 assert all(x["shape"]==(6,6) and x["reduction_residual"]==0 for x in r["rows"])
def test_ttn_full_and_reduced_bond():
 r=ttn_report();assert r["full_bond_state_residual"]==r["full_bond_observable_residual"]==0 and r["low_bond_norm_error"]<.002
 assert min(r[k] for k in ("low_bond_pion_loss","low_bond_transition_loss","low_bond_tensor_loss","low_bond_current_loss","low_bond_coherent_loss"))>.35
def test_provenance_rollback_and_readiness():
 p=provenance_report();r=readiness_report();assert p["count_once_residual"]==0 and not p["unresolved_cycles"] and p["production_edges"]==0
 assert p["rollback_to"]=="C15_N0_EXACT" and not r["production_reachable"] and "PRODUCTION_READY" in r["not_issued"]
def test_injections():assert len(INJECTIONS)>=280 and len({x[0] for x in INJECTIONS})==len(INJECTIONS)

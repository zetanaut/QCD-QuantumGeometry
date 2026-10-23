import pytest
from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.nuclear.n2.core import *
from deuteron_wigner.nuclear.n2.injections import INJECTIONS
def test_plans_exclusive():
 p=plans();assert len(p)==3 and compile_plan((p[0].plan_id,))["scope"]=="C17_N2_VALIDATION_ONLY"
 with pytest.raises(ArchitectureError):compile_plan((p[0].plan_id,p[1].plan_id))
 with pytest.raises(ArchitectureError):compile_plan((p[0].plan_id,),"production")
def test_channels_charge_complete():
 c=channels();assert tuple(x.charge_channel for x in c)==CHARGE_CHANNELS and all(x.charge==1 and x.isospin==0 and x.parity==1 for x in c)
def test_spectral_threshold_pv_cut():
 assert spectral_density(THRESHOLD-.1)==0 and self_energy(THRESHOLD-.1)["cut"]==0
 r=self_energy(THRESHOLD+.2);assert r["cut"]<0 and not r["epsilon_in_identity"]
def test_finite_volume_convergence():
 r=finite_volume_report();assert r["convergent"] and r["rows"][-1]["normalization_residual"]<r["rows"][0]["normalization_residual"] and not r["physical_epsilon_used"]
def test_pole_residue_and_calibration_holdouts():
 r=pole_report();c=calibration_report();assert r["below_threshold_cut"]==0 and r["residue"]>0 and r["pole_residual"]==0
 assert c["null_directions"]==1 and len(c["holdouts"])>=7 and not c["hidden_tmd_fit"]
def test_current_certificate_complete_declared_scope():
 r=current_certificate();assert r["complete"] and not r["unexplained_gaps"] and set(r["attachments"])==set(r["hamiltonian_terms"])
def test_block_continuity_and_current_closure():
 r=continuity_report();assert abs(r["residual"])<1e-14 and r["max_block_residual"]==0 and all(x!=0 for x in r["ablation_residuals"].values())
 assert r["angular_condition_residual"]<1e-12 and r["gtmd_current_residual"]<1e-12
def test_separator_stability():
 r=separator_report();assert r["count_once_residual"]==0 and r["matched_variation"]<r["tolerance"]
def test_feshbach_transforms_all_operators():
 r=feshbach_report();assert r["visible_remainder_norm"]>0 and not r["additive"]
 assert max(v for k,v in r.items() if k.endswith("_residual"))==0
def test_pion_active_unmatched_and_not_fitted():
 r=pion_active_report();assert not r["normalization_fitted"] and r["direct_sequential_residual"]==0 and "PION_PARTON_PARENT_UNMATCHED" in r["status"]
def test_coherent_and_cp_after_amplitudes():
 c=coherent_report();p=cp_report();assert c["zero_amplitude_residual"]==c["overlap_residual"]==0 and c["copied_ratio_failure"]>0
 assert p["choi_min_eigenvalue"]>=0 and p["kraus_partial_trace_residual"]<1e-12 and p["premature_trace_error"]>0
def test_ttn_and_convergence_axes():
 r=tensor_network_report();c=convergence_report();assert r["dimensions"]==[36,68,104] and r["full_bond_residual"]==0 and r["low_bond_tensor_loss"]>.5
 assert all(not x["combined"] for x in c["axes"])
def test_provenance_and_readiness():
 p=provenance_report();r=readiness_report();assert p["count_once_residual"]==0 and p["production_edges"]==0 and not p["unresolved_cycles"]
 assert not r["production_reachable"] and "PRODUCTION_READY" in r["not_issued"]
def test_injections():assert len(INJECTIONS)>=340 and len({x[0] for x in INJECTIONS})==len(INJECTIONS)

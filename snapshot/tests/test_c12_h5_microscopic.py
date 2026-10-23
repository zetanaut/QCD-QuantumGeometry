import numpy as np
import pytest
from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.formal.gauge_path import ColorRepresentation,StapleOrientation
from deuteron_wigner.pilot.wilson_line.identity import derived_eikonal_pole
from deuteron_wigner.pilot.active_gluon.color import reject_default_mixture
from deuteron_wigner.microscopic.h4.core import MicroscopicOverlapKernel,plans as h4_plans
from deuteron_wigner.microscopic.h5.core import *
from deuteron_wigner.microscopic.h5.diagnostics import *
from deuteron_wigner.microscopic.h5.injections import INJECTIONS

def parent(species="u",target="PROTON"):
 return MicroscopicOverlapKernel().matrix(h4_plans()[0],target,species)

def test_spectral_threshold_sign_and_finite_volume_convergence():
 r=spectral_report();assert r["below_threshold"]==0 and r["future"]==-r["past"]
 assert r["maximum_final_residual"]<6e-6
 assert all(a["residual"]>b["residual"] for a,b in zip(r["rows"],r["rows"][1:]))
 assert all(not x["epsilon_physical"] for x in r["rows"])

def test_pole_sign_is_derived_and_manual_sign_fails():
 f=wilson_segment(StapleOrientation.FUTURE);p=wilson_segment(StapleOrientation.PAST)
 assert derived_eikonal_pole(f).eta==-derived_eikonal_pole(p).eta==1
 with pytest.raises(ArchitectureError):derived_eikonal_pole(f,manual_sign=-1)

def test_cut_ledger_counts_equivalent_once_and_distinct_separately():
 r=cut_ledger_report();assert r["residual"]<1e-15 and r["equivalent_counted_once"] and r["distinct_retained"]
 bad=MicroscopicCutLedger().canonical()
 with pytest.raises(ArchitectureError):bad.base.add(IntermediateStateCut("dup",CutKind.LF_ENERGY,bad.support_id,"x",True,.2))

def test_quark_matrix_first_and_distinct_projectors():
 x=H4WilsonKernel().apply(parent("u"));q=QuarkLinkOddProjectorRegistry().project(x)
 assert x.odd.shape==(4,4) and np.linalg.norm(x.odd)>0
 assert q["SIVERS"]!=q["BOER_MULDERS"] and not q["aliased"] and q["shared_parent_id"]==x.stable_id

def test_zero_coupling_cut_oam_and_link_average_limits():
 p=parent("d");r=default_spectral_rule();K=H4WilsonKernel()
 assert np.linalg.norm(K.apply(p,r,coupling=0).odd)<1e-15
 x=H5WilsonInsertion(wilson_segment(StapleOrientation.FUTURE),r,.3,.4,False);assert x.strength()==0
 assert np.linalg.norm(K.apply(p,r,oam_strength=0).odd)<1e-15
 a=K.apply(p,r);assert np.linalg.norm(.5*(a.odd-a.odd))==0

def test_antiquark_is_direct_positive_x_antifundamental_and_not_copy():
 K=H4WilsonKernel();q=K.apply(parent("u"));a=K.apply(parent("ubar"));v=AntiquarkLinkOddProjectorRegistry().project(a)
 assert a.representation=="ANTI_FUNDAMENTAL" and a.species=="ubar" and a.support.state.sector=="QQQUUBAR"
 assert a.remainder>0 and not np.allclose(q.odd,a.odd) and v["shared_parent_id"]==a.stable_id
 with pytest.raises(ArchitectureError):AntiquarkLinkOddProjectorRegistry().project(q)

def test_flavor_target_and_plan_dependence_comes_from_h4_state():
 rows=link_odd_report();assert rows["projectors_distinct"] and rows["all_matrix_first"]
 norms={(r["plan_id"],r["target"],r["species"]):r["odd_norm"] for r in rows["rows"]}
 assert norms[(h4_plans()[0].plan_id,"PROTON","u")]!=norms[(h4_plans()[0].plan_id,"PROTON","d")]
 assert norms[(h4_plans()[0].plan_id,"PROTON","dbar")]!=norms[(h4_plans()[1].plan_id,"PROTON","dbar")]

def test_all_four_gluon_words_fd_color_and_polarizations():
 r=gluon_report();assert len(r["ordered_pairs"])==4 and r["row_count"]==24
 assert r["channels"]==["D_TYPE","F_TYPE"] and r["polarizations"]==["HELICITY","LINEAR","TRACE"]
 assert abs(r["f_norm"]-24)<1e-14 and abs(r["d_norm"]-40/3)<1e-14 and abs(r["fd_inner"])<1e-15
 assert r["process_mixture"] is None and r["both_outer_multiplicities"]
 with pytest.raises(ArchitectureError):reject_default_mixture()

def test_ordered_pair_swap_changes_identity():
 pairs=ordered_gluon_pairs();assert pairs[1].ordered_pair_id!=pairs[1].swapped().ordered_pair_id
 assert pairs[0].antiunitary_pair().orientation_word==("PAST","PAST")

def test_soft_overlap_one_missing_duplicate_signed():
 r=soft_report();assert r["one_subtraction_residual"]==0
 assert r["missing_signed"]==[-x for x in r["duplicate_signed"]]
 assert r["exclusive_route"]=="BOUNDARY_ONLY_RESCATTERING"

def test_fock_order_honest_and_higher_order_closed():
 r=fock_report();assert not r["all_orders_ready"] and r["recommended_next"]=="C13/H6"
 assert {x["support"] for x in r["rows"]}=={"EXPLICIT_FOCK_SUPPORTED","INDUCED_OPERATOR_SUPPORTED_WITH_REMAINDER"}
 with pytest.raises(ArchitectureError):FockOrderSupportManifest("u",2,intermediate("u","m"))

def test_exact_full_bond_and_visible_reduced_bond_loss():
 r=convergence_report();assert r["exact_full_bond_residual"]<1e-14 and len(r["axes"])==16
 assert r["bond_rows"][0]["relative_loss"]>.5 and r["bond_rows"][1]["relative_loss"]>.2
 assert all(not x["combined"] for x in r["axes"]) and not r["energy_only_acceptance"]

def test_scoped_replacement_and_readiness_fail_closed():
 r=replacement_report();c=capability_report()
 assert r["scope"]=="C12_H5_VALIDATION_ONLY" and not r["production"]
 assert not c["production_reachable"] and "PHYSICAL_TMD" in c["not_issued"]

def test_injections_stable_and_complete():
 assert len(INJECTIONS)==124 and len({x[0] for x in INJECTIONS})==124

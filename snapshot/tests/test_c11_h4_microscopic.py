import numpy as np
import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.microscopic.h4.core import *
from deuteron_wigner.microscopic.h4.diagnostics import *
from deuteron_wigner.microscopic.h4.injections import INJECTIONS


def test_h4_plans_are_exact_distinct_h3_descendants():
    ps=plans()
    assert [p.h3_plan_id for p in ps]==["C10:H3:PLAN:7c520ce5e2e04d2c7719","C10:H3:PLAN:b2efe73052d1c9d1004b"]
    assert len({p.plan_id for p in ps})==2 and all(p.wilson_order==0 for p in ps)
    a=MicroscopicOverlapKernel().matrix(ps[0],"PROTON","dbar").values
    b=MicroscopicOverlapKernel().matrix(ps[1],"PROTON","dbar").values
    assert not np.allclose(a,b) # owned chiral dynamics propagates through H3 state


def test_typed_fibers_reject_member_mismatch():
    a,b=microscopic_frame(plans()[0])
    a.require_compatible(b)
    bad=MicroscopicMomentumFiber(b.base,b.target,b.helicity,"OTHER",b.resolution_id)
    with pytest.raises(ArchitectureError): a.require_compatible(bad)


def test_complete_joint_helicity_parent_coverage():
    bundle=common_parent_bundle()
    assert {(m.target,m.species) for m in bundle.matrices}==set((t,s) for t in TARGETS for s in SPECIES)
    assert all(m.values.shape==(4,4) and m.member_id==bundle.plan.state_bundle_id for m in bundle.matrices)
    assert not np.allclose(bundle.matrices[0].values,bundle.matrices[1].values)


def test_quark_antiquark_and_gluon_gram_closure():
    report=projector_report()
    assert report["generic_rank"]==16 and report["degenerate_rank"]==8
    assert report["degenerate_status"]=="EXPLICIT_REDUCED_BASIS"
    assert report["maximum_residual"]<4e-16
    assert {r["species"] for r in report["rows"]}==set(SPECIES)


def test_projectors_refuse_singular_gram(monkeypatch):
    p=QuarkGTMDProjectorBasis()
    p.gram[0]=p.gram[1]
    with pytest.raises(ArchitectureError): p.coefficients(np.eye(4))


def test_hermiticity_and_wilson_zero_link_parity():
    report=symmetry_report()
    assert report["maximum_residual"]<2e-15
    assert all(r["link_odd"]==0 for r in report["rows"])
    assert all(v==0 for v in t_odd_coefficients().values())


def test_forward_reduction_routes_share_parent():
    m=common_parent_bundle().matrices[0]
    r=MicroscopicReductionMap().routes(m)
    assert r["DIRECT_FORWARD"]==r["GTMD_TMD_PDF"]==r["GTMD_GPD_PDF"]
    assert r["parent_id"]==m.stable_id and not r["named_normalization"]


def test_local_current_axial_emt_holdouts():
    r=current_emt_report()
    assert r["maximum_residual"]==0 and sum(x["holdout"] for x in r["rows"])>=4
    assert r["tensor_status"]=="LOCAL_TENSOR_OPERATOR_UNAVAILABLE"
    assert not r["pcac_double_counted"]


def test_wigner_oam_independent_routes_and_bond_sensitivity():
    r=wigner_oam_report()
    assert r["maximum_route_residual"]==0 and r["maximum_finite_difference_residual"]<3e-9
    assert all(x["low_bond"]!=x["full_bond"] for x in r["rows"])
    assert not r["canonical_kinetic_identity_claimed"]


def test_forward_psd_and_offforward_cauchy_not_psd():
    r=positivity_report()
    assert r["minimum_forward_eigenvalue"]>-2e-15 and r["maximum_bound_residual"]<2e-15
    assert not r["wigner_pointwise_positivity_required"] and not r["clipping"]


def test_convergence_axes_are_separate_and_complete():
    r=convergence_report()
    assert len(r["rows"])==12 and all(not x["combined"] for x in r["rows"])
    assert len({x["axis"] for x in r["rows"]})==12


def test_scoped_replacement_and_downstream_gates():
    r=replacement_manifest(); c=capability_snapshot()
    assert r["scope"]["root"]=="C11_H4_VALIDATION_ONLY" and not r["scope"]["production"]
    assert not c["production_reachable"] and "PHYSICAL_GTMD" in c["not_issued"]


def test_wavefunction_exact_direct_full_bond_and_low_bond():
    h,psi=h3_reference(); exact=MicroscopicWaveFunctionEvaluator(h,psi)
    a=exact.evaluate(.3,.12,-.08,delta=(.1,.04))
    assert abs(a.value-exact.direct_basis_sum(.3,.12,-.08,delta=(.1,.04)))<1e-14
    full=MicroscopicWaveFunctionEvaluator(h,psi,"FULL_BOND_TTN",len(psi)).evaluate(.3,.12,-.08,delta=(.1,.04))
    low=MicroscopicWaveFunctionEvaluator(h,psi,"FINITE_BOND_TTN",8).evaluate(.3,.12,-.08,delta=(.1,.04))
    assert abs(a.value-full.value)<1e-14 and abs(a.value-low.value)>1e-7
    with pytest.raises(ArchitectureError): exact.evaluate(0,.1,.1)


def test_negative_injection_inventory():
    assert len(INJECTIONS)==104 and len({x[0] for x in INJECTIONS})==104

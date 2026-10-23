"""C3 reduction bridge, provenance isolation, and remaining injections."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from deuteron_wigner.formal.accepted_reductions import accepted_reduction_registry
from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.formal.operator_identity import IdentityState
from deuteron_wigner.formal.provenance_graph import ProvenanceEdge, ProvenanceGraph, Relation
from deuteron_wigner.formal.reduction import NativeReduction, ReductionRegistry
from deuteron_wigner.pilot.bridge import PilotReductionBridge
from deuteron_wigner.pilot.configuration import IntrinsicConfiguration
from deuteron_wigner.pilot.overlap import AnalyticOverlapEvaluator
from deuteron_wigner.pilot.provenance import pilot_provenance_graph, require_isolated
from deuteron_wigner.pilot.recoil import RecoilResult, SymmetricXiZeroRecoil
from deuteron_wigner.pilot.states import GaussianScalarState, ThreeQuarkColorState
from test_c3_benchmarks import frame, kernel, point_config
from test_c3_fibers_recoil import frame as recoil_frame, two_body

ROOT = Path(__file__).resolve().parents[1]


def accepted_graph():
    return ProvenanceGraph.from_dict(json.loads((ROOT / "docs/next_level/c2_provenance_graph.json").read_text()))


def point_result():
    from deuteron_wigner.pilot.states import PointState
    config=point_config(); frm=frame(config.sector)
    recoil=SymmetricXiZeroRecoil().apply(config,frm)
    return AnalyticOverlapEvaluator().evaluate(PointState(),config,recoil,kernel(frm,config.sector))


def test_pilot_graph_is_disjoint_and_production_builder_has_no_pilot_import():
    require_isolated(accepted_graph(), pilot_provenance_graph())
    source=(ROOT/"scripts/build_wp12_resolved_nuclear_parent.py").read_text()
    assert "deuteron_wigner.pilot" not in source


def test_injected_cross_graph_reachability_fails():
    accepted=accepted_graph(); pilot=pilot_provenance_graph()
    poisoned=ProvenanceGraph(tuple(accepted.nodes.values())+tuple(pilot.nodes.values()), accepted.edges+pilot.edges+(
        ProvenanceEdge(next(iter(accepted.nodes)), "c3:result:A", Relation.CONSUMED_BY, "forbidden promotion"),
    ))
    with pytest.raises(ArchitectureError, match="C3.ISOLATE.PROVENANCE"):
        require_isolated(accepted, poisoned)


def test_validation_bridge_does_not_mutate_accepted_registry():
    accepted=accepted_reduction_registry()
    before=tuple(item.identity.stable_id for item in accepted.entries())
    template=accepted.entries()[0]
    pilot_reduction=NativeReduction(replace(template.identity, stable_id="C3:VALIDATION:FORWARD"), lambda value:value, "analytic pilot identity")
    validation=ReductionRegistry((pilot_reduction,))
    bridge=PilotReductionBridge(validation)
    assert bridge.reduce(point_result(),pilot_reduction)==1
    with pytest.raises(ArchitectureError, match="C3.ISOLATE.REGISTRY"):
        bridge.insert_into_production(accepted,pilot_reduction)
    assert tuple(item.identity.stable_id for item in accepted.entries())==before


def test_wrong_recoil_sign_and_omitted_half_are_detected():
    authority=SymmetricXiZeroRecoil()
    good=authority.apply(two_body(),recoil_frame())
    wrong_sign=replace(good,incoming=good.outgoing,outgoing=good.incoming)
    with pytest.raises(ArchitectureError, match="C3.RECOIL.PHYSICAL"):
        authority.verify_physical_assignment(wrong_sign)
    doubled=authority.apply(two_body(),recoil_frame((.6,-.4)))
    omitted_half=replace(doubled,frame=recoil_frame((.3,-.2)))
    with pytest.raises(ArchitectureError, match="C3.RECOIL.PHYSICAL"):
        authority.verify_physical_assignment(omitted_half)


def test_incomplete_operator_spectator_mismatch_nonsinglet_and_width_promotion():
    config=point_config(); frm=frame(config.sector)
    incomplete=replace(kernel(frm,config.sector).operator_identity,uv_regulator=IdentityState.UNSPECIFIED)
    with pytest.raises(ArchitectureError, match="C1.OPID"):
        replace(kernel(frm,config.sector),operator_identity=incomplete)
    with pytest.raises(ArchitectureError, match="C3.COLOR.FLAVOR"):
        ThreeQuarkColorState(("u","d","s"))
    with pytest.raises(ArchitectureError, match="C3.ISOLATE.WIDTH"):
        GaussianScalarState(.4).promote_width_to_production()

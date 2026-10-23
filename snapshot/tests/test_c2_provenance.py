"""C2 provenance DAG, composition, trace, and exclusion injections."""

from __future__ import annotations

import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.formal.provenance_graph import (
    CompositionPlan, NodeKind, ProvenanceEdge, ProvenanceGraph,
    ProvenanceNode, Relation, SelectionRole,
)


def node(name, role=SelectionRole.ADDITIVE, group=None, central=True):
    return ProvenanceNode(name, NodeKind.COMPONENT, name, role, group, central)


def accepted_fixture():
    nodes = (
        node("parent", SelectionRole.BASELINE),
        node("cp_response", SelectionRole.REPLACEMENT),
        node("legacy_response"),
        node("nnpi"),
        node("phase_direct", SelectionRole.ALTERNATIVE, "phase"),
        node("phase_screened", SelectionRole.ALTERNATIVE, "phase"),
        node("wave_av18", SelectionRole.ALTERNATIVE, "wave"),
        node("wave_cdbonn", SelectionRole.ALTERNATIVE, "wave"),
        node("hessian_plus", SelectionRole.ENSEMBLE_MEMBER, "hessian"),
        node("hessian_minus", SelectionRole.ENSEMBLE_MEMBER, "hessian"),
        node("gluon_f", SelectionRole.ALTERNATIVE, "gluon_color"),
        node("gluon_d", SelectionRole.ALTERNATIVE, "gluon_color"),
        node("future", SelectionRole.ALTERNATIVE, "link"),
        node("past", SelectionRole.ALTERNATIVE, "link"),
        node("spectator_rescale", SelectionRole.BENCHMARK_ONLY, central=False),
        node("hidden_color", SelectionRole.ALTERNATIVE, "non_nucleonic", central=False),
        ProvenanceNode("final", NodeKind.FINAL_ARTIFACT, "accepted final", SelectionRole.BASELINE),
    )
    edges = (
        ProvenanceEdge("cp_response", "legacy_response", Relation.REPLACES, "ordered CP response replaces legacy coefficient response"),
        ProvenanceEdge("phase_direct", "phase_screened", Relation.ALTERNATIVE_TO, "same absorptive phase mechanism"),
        ProvenanceEdge("final", "parent", Relation.DERIVES_FROM, "accepted numerical ancestry"),
        ProvenanceEdge("final", "cp_response", Relation.DERIVES_FROM, "accepted nuclear response"),
        ProvenanceEdge("final", "nnpi", Relation.DERIVES_FROM, "NNpi included once"),
    )
    return ProvenanceGraph(nodes, edges)


def test_accepted_plan_dry_run_and_trace_are_deterministic():
    graph = accepted_fixture()
    plan = CompositionPlan("accepted", ("parent", "cp_response", "nnpi", "phase_screened", "wave_av18", "future"))
    dry = plan.dry_run(graph)
    assert dry["ordered_selection"] == list(plan.ordered_selection)
    assert graph.trace("final") == ("cp_response", "final", "nnpi", "parent")
    assert ProvenanceGraph.from_dict(graph.to_dict()).to_dict() == graph.to_dict()
    assert CompositionPlan.from_dict(dry) == plan


@pytest.mark.parametrize(
    "selection,code",
    [
        (("cp_response", "legacy_response"), "C2.EXCLUSION.CONFLICT"),
        (("phase_direct", "phase_screened"), "C2.EXCLUSION.ALTERNATIVE"),
        (("nnpi", "nnpi"), "C2.EXCLUSION.DUPLICATE"),
        (("wave_av18", "wave_cdbonn"), "C2.EXCLUSION.ALTERNATIVE"),
        (("hessian_plus", "hessian_minus"), "C2.EXCLUSION.ALTERNATIVE"),
        (("future", "past"), "C2.EXCLUSION.ALTERNATIVE"),
        (("gluon_f", "gluon_d"), "C2.EXCLUSION.ALTERNATIVE"),
    ],
)
def test_exclusion_groups_fail_with_stable_codes(selection, code):
    with pytest.raises(ArchitectureError, match=code):
        CompositionPlan("bad", selection).validate(accepted_fixture())


@pytest.mark.parametrize("selection", [("spectator_rescale",), ("hidden_color",)])
def test_benchmark_or_unsupported_sensitivity_cannot_be_central(selection):
    with pytest.raises(ArchitectureError, match="C2.EXCLUSION.BENCHMARK"):
        CompositionPlan("bad", selection).validate(accepted_fixture())


def test_directed_cycle_and_unknown_ancestry_rejected():
    nodes = (node("a"), node("b"))
    with pytest.raises(ArchitectureError, match="C2.PROVGRAPH.CYCLE"):
        ProvenanceGraph(nodes, (
            ProvenanceEdge("a", "b", Relation.DERIVES_FROM, "fixture"),
            ProvenanceEdge("b", "a", Relation.DERIVES_FROM, "fixture"),
        ))
    with pytest.raises(ArchitectureError, match="C2.COMPOSE"):
        CompositionPlan("bad", ("missing",)).validate(accepted_fixture())


def test_orphan_final_output_and_duplicate_csb_are_rejected():
    orphan = ProvenanceGraph((
        ProvenanceNode("final-only", NodeKind.FINAL_ARTIFACT, "orphan", SelectionRole.BASELINE),
    ))
    with pytest.raises(ArchitectureError, match="C2.TRACE.ORPHAN"):
        orphan.require_final_ancestry()
    graph = ProvenanceGraph((node("exact_isospin"), node("csb_once")))
    with pytest.raises(ArchitectureError, match="C2.EXCLUSION.DUPLICATE"):
        CompositionPlan("double-csb", ("csb_once", "csb_once")).validate(graph)

"""C4 finite Feshbach equivalence and explicit/induced exclusion."""

import json
from pathlib import Path

import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.formal.provenance_graph import (
    CompositionPlan, ProvenanceEdge, ProvenanceGraph, Relation,
)
from deuteron_wigner.pilot.c4_provenance import (
    c4_provenance_graph, explicit_plan, induced_plan, require_c4_isolated,
)
from deuteron_wigner.pilot.feshbach import (
    FiniteFeshbachModel, require_exclusive_representation,
)


def accepted_provenance_graph():
    value = json.loads(
        Path("docs/next_level/c2_provenance_graph.json").read_text()
    )
    return ProvenanceGraph.from_dict(value)


def test_exact_feshbach_energy_and_induced_operator_equivalence():
    result = FiniteFeshbachModel().solve()
    assert result.energy_residual < 1e-15
    assert result.operator_residual < 1e-15
    assert result.pop_failure > 0.2
    assert result.norm_kernel > 1
    assert result.gap > 2


def test_feshbach_singular_resolvent_fails_closed():
    model = FiniteFeshbachModel()
    with pytest.raises(ArchitectureError, match="C4.FESHBACH.SINGULAR"):
        model.omega(model.h_qq)


def test_explicit_and_induced_representations_are_exclusive():
    graph = c4_provenance_graph()
    explicit_plan().validate(graph)
    induced_plan().validate(graph)
    with pytest.raises(ArchitectureError, match="C2.EXCLUSION"):
        CompositionPlan(
            "bad", ("c4:state:sea-explicit", "c4:induced:sea"),
            central=False,
        ).validate(graph)
    with pytest.raises(ArchitectureError, match="C4.INDUCED_OPERATOR.DOUBLE_COUNT"):
        require_exclusive_representation(explicit_sector=True, induced_operator=True)


def test_c4_graph_is_deterministic_and_disjoint_from_production():
    accepted = accepted_provenance_graph()
    first = c4_provenance_graph()
    second = c4_provenance_graph()
    assert first.to_dict() == second.to_dict()
    require_c4_isolated(accepted, first)
    edge = ProvenanceEdge(
        next(iter(first.nodes)), next(iter(accepted.nodes)),
        Relation.DERIVES_FROM, "injected production connection",
    )
    combined = ProvenanceGraph(
        (*first.nodes.values(), *accepted.nodes.values()),
        (*first.edges, *accepted.edges, edge),
    )
    with pytest.raises(ArchitectureError, match="C4.ISOLATE.PROVENANCE"):
        require_c4_isolated(accepted, combined)

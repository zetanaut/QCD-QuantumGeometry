"""Metadata-only queries across native reductions and provenance."""

from __future__ import annotations

from dataclasses import dataclass

from .provenance_graph import ProvenanceGraph
from .reduction import ReductionRegistry


@dataclass(frozen=True)
class BoundaryTraceIndex:
    graph: ProvenanceGraph
    reductions: ReductionRegistry

    def trace_named_output(self, output_id: str) -> dict[str, object]:
        return {"output_id": output_id, "ancestry": list(self.graph.trace(output_id))}

    def trace_artifact_row(self, artifact_id: str, row_key: str) -> dict[str, object]:
        return {"artifact_id": artifact_id, "row_key": row_key, "ancestry": list(self.graph.trace(artifact_id))}

    def explain_composition(self, plan) -> dict[str, object]:
        return plan.dry_run(self.graph)

    def explain_exclusion(self, node_a: str, node_b: str) -> dict[str, object]:
        relations = [
            edge for edge in self.graph.edges
            if {edge.source, edge.target} == {node_a, node_b}
            and edge.relation.value in ("EXCLUDES", "ALTERNATIVE_TO", "REPLACES")
        ]
        return {
            "nodes": [node_a, node_b],
            "relations": [
                {"relation": edge.relation.value, "physical_reason": edge.physical_reason}
                for edge in relations
            ],
        }

    def list_reductions(self, operator_name: str) -> tuple[str, ...]:
        return tuple(
            entry.identity.stable_id for entry in self.reductions.entries()
            if entry.identity.source_operator.name == operator_name
        )

    def list_consumers(self, parent_id: str) -> tuple[str, ...]:
        return self.graph.consumers(parent_id)

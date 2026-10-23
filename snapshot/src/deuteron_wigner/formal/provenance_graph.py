"""Deterministic typed provenance graph and fail-closed composition plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from .diagnostics import ArchitectureError


class NodeKind(str, Enum):
    ARTIFACT = "ARTIFACT"
    OPERATOR = "OPERATOR"
    PROJECTION = "PROJECTION"
    COMPONENT = "COMPONENT"
    NUCLEAR_MECHANISM = "NUCLEAR_MECHANISM"
    PHASE_OAM = "PHASE_OAM"
    ENSEMBLE = "ENSEMBLE"
    ENSEMBLE_MEMBER = "ENSEMBLE_MEMBER"
    SCHEME = "SCHEME"
    BENCHMARK = "BENCHMARK"
    EVIDENCE = "EVIDENCE"
    FINAL_ARTIFACT = "FINAL_ARTIFACT"


class Relation(str, Enum):
    DERIVES_FROM = "DERIVES_FROM"
    PROJECTS_TO = "PROJECTS_TO"
    MATCHES_TO = "MATCHES_TO"
    ADDS_TO = "ADDS_TO"
    REPLACES = "REPLACES"
    EXCLUDES = "EXCLUDES"
    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    MEMBER_OF = "MEMBER_OF"
    BENCHMARKS = "BENCHMARKS"
    NORMALIZES = "NORMALIZES"
    VALIDATES = "VALIDATES"
    CONSUMED_BY = "CONSUMED_BY"


class SelectionRole(str, Enum):
    BASELINE = "BASELINE"
    ADDITIVE = "ADDITIVE"
    REPLACEMENT = "REPLACEMENT"
    ALTERNATIVE = "ALTERNATIVE"
    ENSEMBLE_MEMBER = "ENSEMBLE_MEMBER"
    BENCHMARK_ONLY = "BENCHMARK_ONLY"


@dataclass(frozen=True)
class ProvenanceNode:
    stable_id: str
    kind: NodeKind
    physical_identity: str
    selection_role: SelectionRole
    alternative_group: str | None = None
    central_allowed: bool = True
    version: int = 1


@dataclass(frozen=True)
class ProvenanceEdge:
    source: str
    target: str
    relation: Relation
    physical_reason: str
    version: int = 1


class ProvenanceGraph:
    def __init__(self, nodes=(), edges=()) -> None:
        node_items = tuple(nodes)
        self.nodes = {item.stable_id: item for item in node_items}
        self.edges = tuple(sorted(edges, key=lambda x: (x.source, x.target, x.relation.value)))
        if len(self.nodes) != len(node_items):
            raise ArchitectureError("C2.PROVGRAPH", "duplicate provenance node", expected="unique stable ID", received="duplicate")
        self.validate()

    def validate(self) -> None:
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ArchitectureError("C2.PROVGRAPH", "edge references unknown node", expected="known endpoints", received=(edge.source, edge.target))
        adjacency = {node: [] for node in self.nodes}
        for edge in self.edges:
            if edge.relation in (Relation.DERIVES_FROM, Relation.PROJECTS_TO, Relation.MATCHES_TO, Relation.CONSUMED_BY):
                adjacency[edge.source].append(edge.target)
        visiting, visited = set(), set()
        def visit(node):
            if node in visiting:
                raise ArchitectureError("C2.PROVGRAPH.CYCLE", "directed ancestry cycle", expected="DAG", received=node)
            if node in visited:
                return
            visiting.add(node)
            for target in adjacency[node]:
                visit(target)
            visiting.remove(node)
            visited.add(node)
        for node in sorted(self.nodes):
            visit(node)

    def trace(self, stable_id: str) -> tuple[str, ...]:
        if stable_id not in self.nodes:
            raise KeyError(stable_id)
        seen, queue = set(), [stable_id]
        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            queue.extend(edge.target for edge in self.edges if edge.source == node and edge.relation in (Relation.DERIVES_FROM, Relation.PROJECTS_TO, Relation.MATCHES_TO, Relation.VALIDATES))
        return tuple(sorted(seen))

    def consumers(self, stable_id: str) -> tuple[str, ...]:
        return tuple(sorted(edge.source for edge in self.edges if edge.target == stable_id and edge.relation == Relation.CONSUMED_BY))

    def require_final_ancestry(self, authoritative_kinds=(NodeKind.ARTIFACT, NodeKind.COMPONENT, NodeKind.OPERATOR)) -> None:
        for node in self.nodes.values():
            if node.kind != NodeKind.FINAL_ARTIFACT:
                continue
            ancestry = set(self.trace(node.stable_id))
            if not any(self.nodes[item].kind in authoritative_kinds for item in ancestry - {node.stable_id}):
                raise ArchitectureError("C2.TRACE.ORPHAN", "final output has no authoritative ancestry", expected="path to parent/component/operator", received=node.stable_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": [{**asdict(self.nodes[key]), "kind": self.nodes[key].kind.value, "selection_role": self.nodes[key].selection_role.value} for key in sorted(self.nodes)],
            "edges": [{**asdict(edge), "relation": edge.relation.value} for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ProvenanceGraph":
        nodes = [
            ProvenanceNode(
                stable_id=item["stable_id"], kind=NodeKind(item["kind"]),
                physical_identity=item["physical_identity"],
                selection_role=SelectionRole(item["selection_role"]),
                alternative_group=item.get("alternative_group"),
                central_allowed=bool(item.get("central_allowed", True)),
                version=int(item.get("version", 1)),
            )
            for item in value["nodes"]
        ]
        edges = [
            ProvenanceEdge(
                source=item["source"], target=item["target"],
                relation=Relation(item["relation"]),
                physical_reason=item["physical_reason"],
                version=int(item.get("version", 1)),
            )
            for item in value["edges"]
        ]
        return cls(nodes, edges)


@dataclass(frozen=True)
class CompositionPlan:
    stable_id: str
    ordered_selection: tuple[str, ...]
    central: bool = True
    version: int = 1

    def validate(self, graph: ProvenanceGraph) -> None:
        if len(set(self.ordered_selection)) != len(self.ordered_selection):
            raise ArchitectureError("C2.EXCLUSION.DUPLICATE", "physical mechanism selected twice", expected="unique selection", received=self.ordered_selection)
        selected = set(self.ordered_selection)
        for node_id in self.ordered_selection:
            if node_id not in graph.nodes:
                raise ArchitectureError("C2.COMPOSE", "unknown plan node", expected="known node", received=node_id)
            node = graph.nodes[node_id]
            if self.central and (node.selection_role == SelectionRole.BENCHMARK_ONLY or not node.central_allowed):
                raise ArchitectureError("C2.EXCLUSION.BENCHMARK", "benchmark/temporary component cannot be central", expected="central-allowed component", received=node_id)
        groups: dict[str, list[str]] = {}
        for node_id in selected:
            group = graph.nodes[node_id].alternative_group
            if group:
                groups.setdefault(group, []).append(node_id)
        for group, choices in groups.items():
            if len(choices) > 1:
                raise ArchitectureError("C2.EXCLUSION.ALTERNATIVE", "mutually exclusive alternatives selected", expected=f"at most one from {group}", received=tuple(sorted(choices)))
        for edge in graph.edges:
            if edge.source in selected and edge.target in selected and edge.relation in (Relation.EXCLUDES, Relation.ALTERNATIVE_TO, Relation.REPLACES):
                raise ArchitectureError("C2.EXCLUSION.CONFLICT", edge.physical_reason, expected=f"select one of {edge.source}/{edge.target}", received=(edge.source, edge.target))

    def dry_run(self, graph: ProvenanceGraph) -> dict[str, object]:
        self.validate(graph)
        return {"plan_id": self.stable_id, "ordered_selection": list(self.ordered_selection), "ancestry": {node: list(graph.trace(node)) for node in self.ordered_selection}}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "CompositionPlan":
        return cls(
            stable_id=str(value["plan_id"]),
            ordered_selection=tuple(value["ordered_selection"]),
            central=bool(value.get("central", True)),
            version=int(value.get("version", 1)),
        )

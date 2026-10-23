"""Typed, dependency-light identities wrapping the accepted numerical model."""

from .coordinates import CoordinateKind, CoordinateSpec, coordinate_spec
from .diagnostics import ArchitectureError
from .gauge_path import (
    ColorClass, ColorRepresentation, GluonLinkId, StapleOrientation,
    WilsonPathId, standard_staple,
)
from .maps import AdapterRegistry, MapClass, TypedAdapter, TypedMap
from .operator_identity import (
    DecoratedOperatorId, IdentityState, OperationKind, assess_completeness,
)
from .provenance_graph import (
    CompositionPlan, NodeKind, ProvenanceEdge, ProvenanceGraph,
    ProvenanceNode, Relation, SelectionRole,
)
from .reduction import (
    Availability, NativeReduction, ReductionId, ReductionKind,
    ReductionRegistry,
)
from .sector_space import ResolutionLayer, SectorId
from .transverse_rank import CoefficientRole, RankSpec, rank_spec
from .accepted_reductions import accepted_reduction_registry
from .trace import BoundaryTraceIndex

__all__ = [
    "AdapterRegistry", "ArchitectureError", "CoefficientRole", "ColorClass",
    "ColorRepresentation", "CoordinateKind", "CoordinateSpec",
    "DecoratedOperatorId", "GluonLinkId", "IdentityState", "MapClass",
    "OperationKind", "RankSpec", "ResolutionLayer", "SectorId",
    "StapleOrientation", "TypedAdapter", "TypedMap", "WilsonPathId",
    "assess_completeness", "coordinate_spec", "rank_spec", "standard_staple",
    "Availability", "CompositionPlan", "NativeReduction", "NodeKind",
    "ProvenanceEdge", "ProvenanceGraph", "ProvenanceNode", "ReductionId",
    "ReductionKind", "ReductionRegistry", "Relation", "SelectionRole",
    "accepted_reduction_registry",
    "BoundaryTraceIndex",
]

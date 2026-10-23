"""Native typed reductions for the accepted forward canonical boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable

from .coordinates import CoordinateKind, CoordinateSpec
from .diagnostics import ArchitectureError
from .maps import MapClass, TypedMap
from .operator_identity import DecoratedOperatorId, IdentityState, OperationKind
from .transverse_rank import CoefficientRole, RankSpec


class ReductionKind(str, Enum):
    TARGET_SPIN = "TARGET_SPIN"
    PARTON_POLARIZATION = "PARTON_POLARIZATION"
    NAMED_TMD = "NAMED_TMD"
    FORWARD_LIMIT = "FORWARD_LIMIT"
    K_TO_B_TMD = "K_TO_B_TMD"
    B_TMD_TO_K = "B_TMD_TO_K"
    COLLINEAR_INTEGRAL = "COLLINEAR_INTEGRAL"
    WEIGHTED_MOMENT = "WEIGHTED_MOMENT"
    TENSOR_HELICITY_DIFFERENCE = "TENSOR_HELICITY_DIFFERENCE"
    TENSOR_CONVENTION = "TENSOR_CONVENTION"
    FLAVOR_ADAPTER = "FLAVOR_ADAPTER"
    CONSTITUENT_PROJECTION = "CONSTITUENT_PROJECTION"
    EVIDENCE_VIEW = "EVIDENCE_VIEW"


class Availability(str, Enum):
    AVAILABLE_FORWARD = "AVAILABLE_FORWARD"
    UNAVAILABLE_NONZERO_TRANSFER = "UNAVAILABLE_NONZERO_TRANSFER"


@dataclass(frozen=True)
class ReductionId:
    stable_id: str
    kind: ReductionKind
    source_operator: DecoratedOperatorId
    source_parent_identity: str
    target_identity: str
    source_coordinate: CoordinateSpec
    target_coordinate: CoordinateSpec
    source_rank: RankSpec
    target_rank: RankSpec
    target_channel: str
    parton_polarization: str
    collinear_status: str
    moment_weight: str
    scheme_adapter: str | IdentityState
    convention_adapter: str | IdentityState
    availability: Availability
    evidence_status: str
    version: int = 1

    def __post_init__(self) -> None:
        if not self.stable_id or not self.source_parent_identity or not self.target_identity:
            raise ArchitectureError("C2.REDTYPE", "reduction identity is incomplete", expected="stable source/target IDs", received=(self.stable_id, self.target_identity))
        self.source_operator.require_complete(OperationKind.SUBTRACTED_TMD)
        if self.kind == ReductionKind.K_TO_B_TMD:
            self.source_coordinate.require_kind(CoordinateKind.K_T)
            self.target_coordinate.require_kind(CoordinateKind.B_TMD)
        if self.kind == ReductionKind.B_TMD_TO_K:
            self.source_coordinate.require_kind(CoordinateKind.B_TMD)
            self.target_coordinate.require_kind(CoordinateKind.K_T)
        if self.kind == ReductionKind.COLLINEAR_INTEGRAL and self.source_rank.angular_weight:
            raise ArchitectureError("C2.TRANSFORM", "positive-rank coefficient has no nonzero unweighted collinear PDF", expected="rank zero or weighted moment", received=self.source_rank.angular_weight, suggested_adapter="WEIGHTED_MOMENT")
        if self.kind == ReductionKind.WEIGHTED_MOMENT and (not self.moment_weight or self.moment_weight == "none"):
            raise ArchitectureError("C2.TRANSFORM", "weighted moment lacks exact weight", expected="explicit kT/M weight", received=self.moment_weight)
        if self.kind not in (ReductionKind.K_TO_B_TMD, ReductionKind.B_TMD_TO_K) and self.source_coordinate.kind != self.target_coordinate.kind:
            raise ArchitectureError("C2.REDTYPE", "non-Fourier reduction changed coordinate role", expected=self.source_coordinate.kind, received=self.target_coordinate.kind)
        if self.source_rank.coefficient_role != self.target_rank.coefficient_role and self.convention_adapter == IdentityState.UNSPECIFIED:
            raise ArchitectureError("C2.TRANSFORM", "coefficient/modulation conversion lacks adapter", expected=self.source_rank.coefficient_role, received=self.target_rank.coefficient_role)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["source_operator"] = self.source_operator.to_dict()
        value["source_coordinate"] = self.source_coordinate.to_dict()
        value["target_coordinate"] = self.target_coordinate.to_dict()
        value["source_rank"] = self.source_rank.to_dict()
        value["target_rank"] = self.target_rank.to_dict()
        value["availability"] = self.availability.value
        for field in ("scheme_adapter", "convention_adapter"):
            if isinstance(getattr(self, field), IdentityState):
                value[field] = getattr(self, field).value
        return value


@dataclass(frozen=True)
class NativeReduction:
    identity: ReductionId
    callable: Callable
    implementation_symbol: str

    @property
    def typed_map(self) -> TypedMap:
        return TypedMap(
            self.identity.stable_id, MapClass.RED,
            self.identity.source_operator.codomain_type,
            self.identity.target_identity, self.callable,
            f"native accepted reduction: {self.implementation_symbol}",
        )

    def __call__(self, value):
        return self.callable(value)


class ReductionRegistry:
    def __init__(self, entries=()) -> None:
        self._entries: dict[str, NativeReduction] = {}
        for entry in entries:
            self.register(entry)

    def register(self, reduction: NativeReduction) -> None:
        key = reduction.identity.stable_id
        if key in self._entries:
            raise ArchitectureError("C2.REDREG", "duplicate reduction stable ID", expected="unique ID", received=key)
        self._entries[key] = reduction

    def get(self, stable_id: str) -> NativeReduction:
        return self._entries[stable_id]

    def entries(self) -> tuple[NativeReduction, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def validate(self) -> None:
        for entry in self.entries():
            if entry.typed_map.map_class != MapClass.RED:
                raise ArchitectureError("C2.REDTYPE", "reduction has wrong map class", expected=MapClass.RED, received=entry.typed_map.map_class)

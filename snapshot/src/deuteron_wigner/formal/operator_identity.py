"""Fully decorated, operation-aware operator identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from .diagnostics import ArchitectureError
from .gauge_path import ColorRepresentation, GluonLinkId, WilsonPathId
from .transverse_rank import RankSpec


class IdentityState(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNSPECIFIED = "UNSPECIFIED"


class OperationKind(str, Enum):
    LOCAL_CURRENT = "LOCAL_CURRENT"
    BARE_GTMD = "BARE_GTMD"
    SUBTRACTED_TMD = "SUBTRACTED_TMD"
    PROCESS = "PROCESS"


@dataclass(frozen=True)
class DecoratedOperatorId:
    name: str
    parton_species: str
    flavor: str | IdentityState
    projection: str
    domain_type: str
    codomain_type: str
    initial_momentum_fiber: str
    final_momentum_fiber: str
    coordinate_kinds: tuple[str, ...]
    rank_spec: RankSpec
    wilson_identity: WilsonPathId | GluonLinkId | IdentityState
    color_representation: ColorRepresentation
    uv_regulator: str | IdentityState
    rapidity_regulator: str | IdentityState
    soft_subtraction: str | IdentityState
    mu_gev: float | IdentityState
    zeta_gev2: float | IdentityState
    renormalization_factorization_scheme: str | IdentityState
    normalization_convention: str
    evidence_or_status_class: str
    version: int = 1

    def __post_init__(self) -> None:
        if not all((self.name, self.parton_species, self.projection, self.domain_type, self.codomain_type, self.normalization_convention, self.evidence_or_status_class)):
            raise ArchitectureError("C1.OPID", "operator identity has empty required text", expected="nonempty identity fields", received=self)

    def completeness(self, operation: OperationKind) -> tuple[str, ...]:
        required = {
            OperationKind.LOCAL_CURRENT: ("flavor", "renormalization_factorization_scheme"),
            OperationKind.BARE_GTMD: ("flavor", "wilson_identity", "uv_regulator"),
            OperationKind.SUBTRACTED_TMD: ("flavor", "wilson_identity", "uv_regulator", "rapidity_regulator", "soft_subtraction", "mu_gev", "zeta_gev2", "renormalization_factorization_scheme"),
            OperationKind.PROCESS: ("flavor", "wilson_identity", "mu_gev", "zeta_gev2", "renormalization_factorization_scheme"),
        }[operation]
        return tuple(name for name in required if getattr(self, name) == IdentityState.UNSPECIFIED)

    def require_complete(self, operation: OperationKind) -> None:
        missing = self.completeness(operation)
        if missing:
            raise ArchitectureError("C1.OPID", f"operator incomplete for {operation.value}", expected=f"specified {missing}", received=IdentityState.UNSPECIFIED.value)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in ("flavor", "uv_regulator", "rapidity_regulator", "soft_subtraction", "mu_gev", "zeta_gev2", "renormalization_factorization_scheme"):
            if isinstance(getattr(self, key), Enum):
                value[key] = getattr(self, key).value
        value["color_representation"] = self.color_representation.value
        value["rank_spec"] = self.rank_spec.to_dict()
        if isinstance(self.wilson_identity, IdentityState):
            value["wilson_identity"] = self.wilson_identity.value
        else:
            value["wilson_identity"] = self.wilson_identity.to_dict()
        return value


def assess_completeness(operator: DecoratedOperatorId) -> dict[str, object]:
    return {
        "name": operator.name,
        "species": operator.parton_species,
        "flavor": operator.flavor.value if isinstance(operator.flavor, Enum) else operator.flavor,
        "operations": {
            operation.value: {
                "complete": not operator.completeness(operation),
                "unspecified": list(operator.completeness(operation)),
            }
            for operation in OperationKind
        },
    }

"""Versioned identity for microscopic, nuclear, and phenomenological sectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from .diagnostics import ArchitectureError


class ResolutionLayer(str, Enum):
    MICROSCOPIC_FOCK = "MICROSCOPIC_FOCK"
    HADRONIC_NUCLEAR = "HADRONIC_NUCLEAR"
    PHENOMENOLOGICAL_COMPONENT = "PHENOMENOLOGICAL_COMPONENT"


@dataclass(frozen=True)
class SectorId:
    resolution_layer: ResolutionLayer
    quark_occupations_by_flavor: tuple[tuple[str, int], ...]
    antiquark_occupations_by_flavor: tuple[tuple[str, int], ...]
    gluon_count: int | None
    exact_charge_thirds: int | None
    jz_or_helicity_block: str
    parity_class: str
    color_status: str
    basis_id: str
    source_role: str
    version: int = 1

    def __post_init__(self) -> None:
        if not self.basis_id or not self.source_role:
            raise ArchitectureError("C1.SECTOR", "sector identity is incomplete", expected="basis_id and source_role", received=(self.basis_id, self.source_role))
        if len(dict(self.quark_occupations_by_flavor)) != len(self.quark_occupations_by_flavor) or len(dict(self.antiquark_occupations_by_flavor)) != len(self.antiquark_occupations_by_flavor):
            raise ArchitectureError("C1.SECTOR", "duplicate flavor occupation", expected="one occupation per flavor", received=(self.quark_occupations_by_flavor, self.antiquark_occupations_by_flavor))
        object.__setattr__(self, "quark_occupations_by_flavor", tuple(sorted(self.quark_occupations_by_flavor)))
        object.__setattr__(self, "antiquark_occupations_by_flavor", tuple(sorted(self.antiquark_occupations_by_flavor)))
        occupations = dict(self.quark_occupations_by_flavor)
        anti = dict(self.antiquark_occupations_by_flavor)
        if any(value < 0 for value in (*occupations.values(), *anti.values())):
            raise ArchitectureError("C1.SECTOR", "negative occupation", expected="nonnegative integers", received=(occupations, anti))
        if self.resolution_layer == ResolutionLayer.MICROSCOPIC_FOCK and self.exact_charge_thirds is not None:
            charges = {"u": 2, "d": -1, "s": -1, "c": 2, "b": -1, "t": 2}
            derived = sum(charges.get(f, 0) * n for f, n in occupations.items())
            derived -= sum(charges.get(f, 0) * n for f, n in anti.items())
            if derived != self.exact_charge_thirds:
                raise ArchitectureError("C1.SECTOR", "occupation-derived charge mismatch", expected=self.exact_charge_thirds, received=derived)

    def require_same_sector(self, other: "SectorId") -> None:
        if self != other:
            raise ArchitectureError("C1.SECTOR", "equal array shape does not establish sector identity", expected=self, received=other)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["resolution_layer"] = self.resolution_layer.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SectorId":
        data = dict(value)
        data["resolution_layer"] = ResolutionLayer(str(data["resolution_layer"]))
        data["quark_occupations_by_flavor"] = tuple(tuple(x) for x in data["quark_occupations_by_flavor"])
        data["antiquark_occupations_by_flavor"] = tuple(tuple(x) for x in data["antiquark_occupations_by_flavor"])
        return cls(**data)

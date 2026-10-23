"""Physical identities for non-interchangeable transverse coordinates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from .diagnostics import ArchitectureError


class CoordinateKind(str, Enum):
    DELTA_T = "DELTA_T"
    B_DELTA = "B_DELTA"
    K_T = "K_T"
    B_TMD = "B_TMD"
    DELTA_NT = "DELTA_NT"
    P_T_NUCLEAR = "P_T_NUCLEAR"
    R_T_NUCLEAR = "R_T_NUCLEAR"
    Q_T_MEASURED = "Q_T_MEASURED"


@dataclass(frozen=True)
class CoordinateSpec:
    kind: CoordinateKind
    dimension: int
    units: str
    frame: str
    role: str
    conjugate_kind: CoordinateKind | None
    fourier_sign: int | None
    measure_convention: str
    version: int = 1

    def __post_init__(self) -> None:
        if self.dimension != 2 or not self.units or not self.frame:
            raise ArchitectureError(
                "C1.COORD", "invalid coordinate dimension/units/frame",
                expected="2D coordinate with explicit units and frame",
                received=(self.dimension, self.units, self.frame),
            )
        if self.fourier_sign not in (-1, 1, None):
            raise ArchitectureError(
                "C1.COORD", "invalid Fourier sign", expected="-1, +1 or N/A",
                received=self.fourier_sign,
            )

    def require_kind(self, expected: CoordinateKind, adapter: str | None = None) -> None:
        if self.kind != expected:
            raise ArchitectureError(
                "C1.COORD", "coordinate role mismatch", expected=expected.value,
                received=self.kind.value, suggested_adapter=adapter,
            )

    def require_conjugate(self, other: "CoordinateSpec") -> None:
        if self.conjugate_kind != other.kind or other.conjugate_kind != self.kind:
            raise ArchitectureError(
                "C1.COORD", "coordinates are not a declared conjugate pair",
                expected=self.conjugate_kind, received=other.kind,
            )

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["conjugate_kind"] = (
            self.conjugate_kind.value if self.conjugate_kind else None
        )
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "CoordinateSpec":
        data = dict(value)
        data["kind"] = CoordinateKind(str(data["kind"]))
        conjugate = data.get("conjugate_kind")
        data["conjugate_kind"] = CoordinateKind(str(conjugate)) if conjugate else None
        return cls(**data)


_SPECS = {
    CoordinateKind.DELTA_T: CoordinateSpec(CoordinateKind.DELTA_T, 2, "GeV", "partonic symmetric LF", "GTMD momentum transfer", CoordinateKind.B_DELTA, -1, "d2Delta/(2pi)^2"),
    CoordinateKind.B_DELTA: CoordinateSpec(CoordinateKind.B_DELTA, 2, "GeV^-1", "partonic symmetric LF", "Wigner imaging", CoordinateKind.DELTA_T, -1, "d2Delta/(2pi)^2"),
    CoordinateKind.K_T: CoordinateSpec(CoordinateKind.K_T, 2, "GeV", "partonic TMD", "intrinsic parton momentum", CoordinateKind.B_TMD, 1, "d2k"),
    CoordinateKind.B_TMD: CoordinateSpec(CoordinateKind.B_TMD, 2, "GeV^-1", "partonic TMD", "TMD impact coordinate", CoordinateKind.K_T, 1, "d2k"),
    CoordinateKind.DELTA_NT: CoordinateSpec(CoordinateKind.DELTA_NT, 2, "GeV", "nuclear LF", "nucleon/nuclear transfer", CoordinateKind.R_T_NUCLEAR, -1, "d2DeltaN/(2pi)^2"),
    CoordinateKind.R_T_NUCLEAR: CoordinateSpec(CoordinateKind.R_T_NUCLEAR, 2, "GeV^-1", "nuclear LF", "nuclear impact coordinate", CoordinateKind.DELTA_NT, -1, "d2DeltaN/(2pi)^2"),
    CoordinateKind.P_T_NUCLEAR: CoordinateSpec(CoordinateKind.P_T_NUCLEAR, 2, "GeV", "nuclear LF", "internal nuclear momentum", None, None, "d2pT"),
    CoordinateKind.Q_T_MEASURED: CoordinateSpec(CoordinateKind.Q_T_MEASURED, 2, "GeV", "measurement", "process observable", None, None, "process-defined"),
}


def coordinate_spec(kind: CoordinateKind) -> CoordinateSpec:
    return _SPECS[kind]

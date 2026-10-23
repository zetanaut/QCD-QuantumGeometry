"""Bare Wilson-path and ordered gluon-link identities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from .diagnostics import ArchitectureError


class StapleOrientation(str, Enum):
    FUTURE = "FUTURE"
    PAST = "PAST"
    STRAIGHT = "STRAIGHT"
    UNSPECIFIED = "UNSPECIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ColorRepresentation(str, Enum):
    FUNDAMENTAL = "FUNDAMENTAL"
    ADJOINT = "ADJOINT"
    SINGLET = "SINGLET"
    UNSPECIFIED = "UNSPECIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ColorClass(str, Enum):
    F_TYPE = "F_TYPE"
    D_TYPE = "D_TYPE"
    DIAGONAL_ADJOINT = "DIAGONAL_ADJOINT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNSPECIFIED = "UNSPECIFIED"


@dataclass(frozen=True)
class WilsonPathId:
    endpoints: tuple[str, str]
    ordered_segments: tuple[str, ...]
    staple_orientation: StapleOrientation
    transverse_closure: str
    rapidity_direction: str
    color_representation: ColorRepresentation
    boundary_class: str
    version: int = 1

    def __post_init__(self) -> None:
        if len(self.endpoints) != 2 or not self.ordered_segments:
            raise ArchitectureError("C1.PATH", "path endpoints/segments missing", expected="two endpoints and ordered segments", received=(self.endpoints, self.ordered_segments))

    def inverted(self) -> "WilsonPathId":
        orientation = {StapleOrientation.FUTURE: StapleOrientation.PAST, StapleOrientation.PAST: StapleOrientation.FUTURE}.get(self.staple_orientation, self.staple_orientation)
        return WilsonPathId((self.endpoints[1], self.endpoints[0]), tuple(reversed(self.ordered_segments)), orientation, self.transverse_closure, self.rapidity_direction, self.color_representation, self.boundary_class, self.version)

    def require_production(self) -> None:
        if self.staple_orientation == StapleOrientation.UNSPECIFIED or self.color_representation == ColorRepresentation.UNSPECIFIED:
            raise ArchitectureError("C1.PATH", "unspecified path metadata in production", expected="fully specified path", received=self)

    def to_dict(self) -> dict[str, object]:
        return _path_dict(self)


@dataclass(frozen=True)
class GluonLinkId:
    first_path: WilsonPathId
    second_path: WilsonPathId
    color_class: ColorClass
    version: int = 1

    def __post_init__(self) -> None:
        if self.first_path.color_representation != ColorRepresentation.ADJOINT or self.second_path.color_representation != ColorRepresentation.ADJOINT:
            raise ArchitectureError("C1.PATH", "gluon links require adjoint Wilson paths", expected=ColorRepresentation.ADJOINT, received=(self.first_path.color_representation, self.second_path.color_representation))

    def require_color_class(self, expected: ColorClass) -> None:
        if self.color_class == ColorClass.UNSPECIFIED or self.color_class != expected:
            raise ArchitectureError("C1.PATH", "gluon color class mismatch", expected=expected.value, received=self.color_class.value)

    def to_dict(self) -> dict[str, object]:
        return {"first_path": _path_dict(self.first_path), "second_path": _path_dict(self.second_path), "color_class": self.color_class.value, "version": self.version}


def _path_dict(path: WilsonPathId) -> dict[str, object]:
    value = asdict(path)
    value["staple_orientation"] = path.staple_orientation.value
    value["color_representation"] = path.color_representation.value
    return value


def standard_staple(orientation: StapleOrientation, representation: ColorRepresentation) -> WilsonPathId:
    if orientation not in (StapleOrientation.FUTURE, StapleOrientation.PAST):
        raise ArchitectureError("C1.PATH", "standard staple requires future/past orientation", expected="FUTURE or PAST", received=orientation)
    infinity = "+infinity" if orientation == StapleOrientation.FUTURE else "-infinity"
    return WilsonPathId(("0", "xi"), (f"0->{infinity}", f"{infinity}->xi"), orientation, "transverse_at_infinity", orientation.value.lower(), representation, "staple")

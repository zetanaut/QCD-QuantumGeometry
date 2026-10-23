"""Distinct typed map classes and explicit adapter registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, TypeVar

from .diagnostics import ArchitectureError

T = TypeVar("T")
U = TypeVar("U")


class MapClass(str, Enum):
    AMP = "AMP"
    DENS = "DENS"
    MATCH = "MATCH"
    RED = "RED"
    PROC = "PROC"


@dataclass(frozen=True)
class TypedMap(Generic[T, U]):
    name: str
    map_class: MapClass
    domain: str
    codomain: str
    callable: Callable[[T], U] = field(compare=False, hash=False, repr=False)
    provenance: str
    version: int = 1

    def __post_init__(self) -> None:
        if not self.provenance:
            raise ArchitectureError("C1.MAP", "map lacks provenance", expected="nonempty provenance", received=self.provenance)

    def __call__(self, value: T) -> U:
        return self.callable(value)


@dataclass(frozen=True)
class TypedAdapter(TypedMap[T, U]):
    convention_change: str = "none"
    losslessness: str = ""
    remainder: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.losslessness or not self.remainder:
            raise ArchitectureError("C1.ADAPT", "adapter lacks loss/remainder declaration", expected="explicit losslessness and remainder", received=(self.losslessness, self.remainder))


class AdapterRegistry:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], TypedAdapter] = {}

    def register(self, adapter: TypedAdapter) -> None:
        key = (adapter.domain, adapter.codomain)
        if key in self._items:
            raise ArchitectureError("C1.ADAPT", "duplicate adapter", expected="unique endpoint pair", received=key)
        self._items[key] = adapter

    def get(self, domain: str, codomain: str) -> TypedAdapter | None:
        return self._items.get((domain, codomain))

    def compose(self, g: TypedMap, f: TypedMap) -> TypedMap:
        bridge = self.get(f.codomain, g.domain)
        if f.codomain != g.domain and bridge is None:
            raise ArchitectureError("C1.MAP", "map endpoints cannot compose", expected=g.domain, received=f.codomain, suggested_adapter=f"{f.codomain}->{g.domain}")
        def composed(value):
            middle = f(value)
            if bridge is not None:
                middle = bridge(middle)
            return g(middle)
        return TypedMap(f"{g.name}∘{f.name}", g.map_class, f.domain, g.codomain, composed, f"{f.provenance}; {g.provenance}")

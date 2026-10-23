"""Structured fail-closed architecture diagnostics."""

from __future__ import annotations


class ArchitectureError(ValueError):
    def __init__(
        self, requirement_id: str, message: str, *, expected: object,
        received: object, suggested_adapter: str | None = None,
    ) -> None:
        self.requirement_id = requirement_id
        self.expected = expected
        self.received = received
        self.suggested_adapter = suggested_adapter
        suffix = (
            f"; suggested_adapter={suggested_adapter}"
            if suggested_adapter else "; suggested_adapter=none"
        )
        super().__init__(
            f"[{requirement_id}] {message}; expected={expected}; "
            f"received={received}{suffix}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "message": str(self),
            "expected": str(self.expected),
            "received": str(self.received),
            "suggested_adapter": self.suggested_adapter,
        }

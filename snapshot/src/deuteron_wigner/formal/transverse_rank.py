"""Explicit transverse-rank and extracted-mass conventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from .diagnostics import ArchitectureError


class CoefficientRole(str, Enum):
    SCALAR_COEFFICIENT = "scalar_coefficient"
    PHYSICAL_MODULATION = "physical_modulation"


@dataclass(frozen=True)
class RankSpec:
    angular_weight: int
    tensor_basis: str
    k_power: int
    b_power: int
    reference_mass_gev: float | None
    mass_units: str
    bessel_order: int
    fourier_phase: complex
    coefficient_role: CoefficientRole
    convention_adapter: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        if self.angular_weight < 0 or self.k_power < 0 or self.b_power < 0:
            raise ArchitectureError("C1.RANK", "rank powers must be nonnegative", expected=">=0", received=(self.angular_weight, self.k_power, self.b_power))
        if self.bessel_order != abs(self.angular_weight) and not self.convention_adapter:
            raise ArchitectureError("C1.RANK", "Bessel order/rank mismatch", expected=abs(self.angular_weight), received=self.bessel_order)
        if (self.k_power != self.angular_weight or self.b_power != self.angular_weight) and not self.convention_adapter:
            raise ArchitectureError("C1.RANK", "extracted transverse power/rank mismatch", expected=(self.angular_weight, self.angular_weight), received=(self.k_power, self.b_power))
        if (self.k_power or self.b_power) and self.reference_mass_gev is None:
            raise ArchitectureError("C1.RANK", "rank requires a reference mass", expected="positive GeV mass", received="UNSPECIFIED")
        if self.reference_mass_gev is not None and (self.reference_mass_gev <= 0 or self.mass_units != "GeV"):
            raise ArchitectureError("C1.RANK", "invalid reference mass or units", expected="positive mass in GeV", received=(self.reference_mass_gev, self.mass_units))

    def require_transform(self, *, bessel_order: int, phase: complex) -> None:
        if bessel_order != self.bessel_order or phase != self.fourier_phase:
            raise ArchitectureError("C1.RANK", "rank-aware transform mismatch", expected=(self.bessel_order, self.fourier_phase), received=(bessel_order, phase))

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["coefficient_role"] = self.coefficient_role.value
        value["fourier_phase"] = [self.fourier_phase.real, self.fourier_phase.imag]
        return value


def rank_spec(rank: int, reference_mass_gev: float | None = None) -> RankSpec:
    mass = reference_mass_gev if rank else None
    return RankSpec(rank, "symmetric_traceless_SO2", rank, rank, mass, "GeV", rank, complex(0, 1) ** rank, CoefficientRole.SCALAR_COEFFICIENT)

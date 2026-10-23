"""Read-only identities and wrappers for accepted legacy boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from ..conventions import GTMD_IMAGING_CONVENTION, TMD_EVOLUTION_CONVENTION
from ..fourier import bessel_b_to_k
from ..gtmd import Species
from ..registry import TMDEntry
from ..tmd_scheme import TMDScheme, TMDScalePoint
from .coordinates import CoordinateKind, CoordinateSpec
from .diagnostics import ArchitectureError
from .gauge_path import (
    ColorClass, ColorRepresentation, GluonLinkId, StapleOrientation,
    standard_staple,
)
from .operator_identity import DecoratedOperatorId, IdentityState
from .transverse_rank import RankSpec, rank_spec


@dataclass(frozen=True)
class LegacyRadialGrid:
    values: object
    coordinate: CoordinateSpec


def typed_bessel_b_to_k(grid: LegacyRadialGrid, values, k, rank: RankSpec):
    grid.coordinate.require_kind(CoordinateKind.B_TMD)
    rank.require_transform(bessel_order=rank.bessel_order, phase=rank.fourier_phase)
    return bessel_b_to_k(grid.values, values, k, rank=rank.bessel_order)


def registry_operator_identity(
    entry: TMDEntry, *, flavor: str | IdentityState, scale: TMDScalePoint,
    scheme: TMDScheme, orientation: StapleOrientation,
    gluon_color_class: ColorClass = ColorClass.NOT_APPLICABLE,
    second_orientation: StapleOrientation | None = None,
    reference_mass_gev: float | None = None,
) -> DecoratedOperatorId:
    representation = (
        ColorRepresentation.ADJOINT
        if entry.species == Species.GLUON else ColorRepresentation.FUNDAMENTAL
    )
    first = standard_staple(orientation, representation)
    if entry.species == Species.GLUON:
        second = standard_staple(
            second_orientation if second_orientation is not None else orientation,
            representation,
        )
        wilson = GluonLinkId(first, second, gluon_color_class)
        if gluon_color_class == ColorClass.UNSPECIFIED:
            pass
    else:
        if gluon_color_class != ColorClass.NOT_APPLICABLE:
            raise ArchitectureError("C1.PATH", "quark operator cannot carry gluon color class", expected=ColorClass.NOT_APPLICABLE, received=gluon_color_class)
        wilson = first
    rank = rank_spec(entry.transverse_rank, reference_mass_gev)
    return DecoratedOperatorId(
        name=entry.name, parton_species=entry.species.value, flavor=flavor,
        projection=entry.parent_projection, domain_type="spin1_target_state",
        codomain_type="leading_twist_correlator",
        initial_momentum_fiber="zero_skewness_incoming",
        final_momentum_fiber="zero_skewness_outgoing",
        coordinate_kinds=(CoordinateKind.K_T.value, CoordinateKind.B_TMD.value),
        rank_spec=rank, wilson_identity=wilson,
        color_representation=representation, uv_regulator=scheme.uv_scheme,
        rapidity_regulator=scheme.rapidity_regulator.value,
        soft_subtraction=scheme.soft_subtraction.value, mu_gev=scale.mu_gev,
        zeta_gev2=scale.zeta_gev2,
        renormalization_factorization_scheme=scheme.uv_scheme,
        normalization_convention="accepted spin-1 definite-rank registry",
        evidence_or_status_class=entry.matching_status.value,
    )


def accepted_fourier_metadata() -> dict[str, dict[str, object]]:
    return {
        "GTMD_IMAGING": {
            "sign": GTMD_IMAGING_CONVENTION.forward_sign,
            "normalization": GTMD_IMAGING_CONVENTION.forward_normalization,
        },
        "TMD_EVOLUTION": {
            "sign": TMD_EVOLUTION_CONVENTION.forward_sign,
            "normalization": TMD_EVOLUTION_CONVENTION.forward_normalization,
        },
    }

"""Deterministic native reduction registry for accepted named TMD outputs."""

from __future__ import annotations

from ..gtmd import Species
from ..registry import TMDEntry, leading_twist_gluon_registry, leading_twist_quark_registry
from ..tmd_scheme import DELTA_COLLINS_ZETA_SCHEME, TMDScalePoint
from .coordinates import CoordinateKind, coordinate_spec
from .gauge_path import ColorClass, StapleOrientation
from .legacy_adapters import registry_operator_identity
from .operator_identity import IdentityState
from .reduction import Availability, NativeReduction, ReductionId, ReductionKind, ReductionRegistry
from .transverse_rank import rank_spec

TARGET_MASS_GEV = 1.87561294257


def _accepted_gluon_entries() -> tuple[TMDEntry, ...]:
    registry = leading_twist_gluon_registry()
    entries = []
    for entry in registry.select():
        if entry.name in ("f1TT", "h1TTperp"):
            continue
        entries.append(entry)
    f1tt = registry.get(Species.GLUON, "f1TT")
    entries.append(TMDEntry(
        name="f1TT_minus_h1TTperp", species=Species.GLUON,
        parent_projection="Phi_ij:TT:trace-linear-identifiable-combination",
        target_channel=f1tt.target_channel,
        parton_polarization="trace_linear_combination",
        transverse_rank=2, gauge_link_required=True,
        collinear_limit=f1tt.collinear_limit,
        matching_status=f1tt.matching_status,
        positivity_block=f1tt.positivity_block, t_odd=False,
        notes="Only the identifiable two-dimensional TT combination.",
    ))
    return tuple(sorted(entries, key=lambda item: item.name))


def accepted_reduction_registry() -> ReductionRegistry:
    """Return 216 link/color-resolved accepted forward named reductions."""

    reductions = []
    scale = TMDScalePoint.canonical(5.0)
    for species, flavors in (
        (Species.QUARK, ("u", "d")),
        (Species.ANTIQUARK, ("ubar", "dbar")),
    ):
        for entry in leading_twist_quark_registry(species).select():
            for flavor in flavors:
                for orientation in (StapleOrientation.FUTURE, StapleOrientation.PAST):
                    operator = registry_operator_identity(
                        entry, flavor=flavor, scale=scale,
                        scheme=DELTA_COLLINS_ZETA_SCHEME,
                        orientation=orientation,
                        reference_mass_gev=TARGET_MASS_GEV if entry.transverse_rank else None,
                    )
                    link_label = "[+,+]" if orientation == StapleOrientation.FUTURE else "[-,-]"
                    reductions.append(_named_reduction(entry, operator, flavor, link_label, "NOT_APPLICABLE"))
    gluon_links = (
        (StapleOrientation.FUTURE, StapleOrientation.FUTURE, ColorClass.F_TYPE, "[+,+]"),
        (StapleOrientation.PAST, StapleOrientation.PAST, ColorClass.F_TYPE, "[-,-]"),
        (StapleOrientation.FUTURE, StapleOrientation.PAST, ColorClass.D_TYPE, "[+,-]"),
        (StapleOrientation.PAST, StapleOrientation.FUTURE, ColorClass.D_TYPE, "[-,+]"),
    )
    for entry in _accepted_gluon_entries():
        for orientation, second_orientation, color, link_label in gluon_links:
            operator = registry_operator_identity(
                entry, flavor=IdentityState.NOT_APPLICABLE, scale=scale,
                scheme=DELTA_COLLINS_ZETA_SCHEME,
                orientation=orientation, gluon_color_class=color,
                second_orientation=second_orientation,
                reference_mass_gev=TARGET_MASS_GEV if entry.transverse_rank else None,
            )
            reductions.append(_named_reduction(entry, operator, "NOT_APPLICABLE", link_label, color.value))
    registry = ReductionRegistry(reductions)
    registry.validate()
    return registry


def _named_reduction(entry, operator, flavor, orientation, color) -> NativeReduction:
    rank = rank_spec(entry.transverse_rank, TARGET_MASS_GEV if entry.transverse_rank else None)
    stable_id = f"RED:{entry.species.value}:{flavor}:{entry.name}:{orientation}:{color}"
    identity = ReductionId(
        stable_id=stable_id, kind=ReductionKind.NAMED_TMD,
        source_operator=operator,
        source_parent_identity=(
            "accepted_resolved_gluon_correlator"
            if entry.species == Species.GLUON
            else "accepted_resolved_quark_correlator"
        ),
        target_identity=f"named_tmd:{stable_id}",
        source_coordinate=coordinate_spec(CoordinateKind.K_T),
        target_coordinate=coordinate_spec(CoordinateKind.K_T),
        source_rank=rank, target_rank=rank,
        target_channel=entry.target_channel.value,
        parton_polarization=entry.parton_polarization,
        collinear_status=entry.collinear_limit.value,
        moment_weight=(
            "none" if entry.transverse_rank == 0
            else f"(k_T/{TARGET_MASS_GEV}GeV)^{entry.transverse_rank}"
        ),
        scheme_adapter=IdentityState.NOT_APPLICABLE,
        convention_adapter="accepted definite-rank named projector",
        availability=Availability.AVAILABLE_FORWARD,
        evidence_status=entry.matching_status.value,
    )
    return NativeReduction(identity, lambda value: value, entry.parent_projection)

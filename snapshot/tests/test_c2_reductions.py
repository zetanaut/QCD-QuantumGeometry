"""C2 native-reduction registry, transform semantics, and injections."""

from __future__ import annotations

from dataclasses import replace

import pytest

from deuteron_wigner.formal.accepted_reductions import accepted_reduction_registry
from deuteron_wigner.formal.coordinates import CoordinateKind, coordinate_spec
from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.formal.maps import MapClass
from deuteron_wigner.formal.operator_identity import IdentityState
from deuteron_wigner.formal.reduction import ReductionKind, ReductionRegistry
from deuteron_wigner.formal.transverse_rank import CoefficientRole, RankSpec, rank_spec


def test_native_registry_is_complete_deterministic_and_red_typed():
    registry = accepted_reduction_registry()
    entries = registry.entries()
    assert len(entries) == 216
    assert [item.identity.stable_id for item in entries] == sorted(item.identity.stable_id for item in entries)
    assert all(item.typed_map.map_class == MapClass.RED for item in entries)
    assert sum(":q:" in item.identity.stable_id for item in entries) == 72
    assert sum(":qbar:" in item.identity.stable_id for item in entries) == 72
    assert sum(":g:" in item.identity.stable_id for item in entries) == 72
    assert all(item.identity.availability.value == "AVAILABLE_FORWARD" for item in entries)


def test_duplicate_reduction_identity_rejected():
    entry = accepted_reduction_registry().entries()[0]
    with pytest.raises(ArchitectureError, match="C2.REDREG"):
        ReductionRegistry((entry, entry))


def test_transform_rejects_wrong_coordinates():
    base = accepted_reduction_registry().entries()[0].identity
    with pytest.raises(ArchitectureError, match="C1.COORD"):
        replace(base, stable_id="bad-k-bdelta", kind=ReductionKind.K_TO_B_TMD, source_coordinate=coordinate_spec(CoordinateKind.K_T), target_coordinate=coordinate_spec(CoordinateKind.B_DELTA))
    with pytest.raises(ArchitectureError, match="C2.REDTYPE"):
        replace(base, stable_id="bad-nuclear-parton", source_coordinate=coordinate_spec(CoordinateKind.P_T_NUCLEAR), target_coordinate=coordinate_spec(CoordinateKind.K_T))


def test_positive_rank_collinear_and_weight_semantics_fail_closed():
    base = next(item.identity for item in accepted_reduction_registry().entries() if item.identity.source_rank.angular_weight == 1)
    with pytest.raises(ArchitectureError, match="C2.TRANSFORM"):
        replace(base, stable_id="bad-collinear", kind=ReductionKind.COLLINEAR_INTEGRAL)
    with pytest.raises(ArchitectureError, match="C2.TRANSFORM"):
        replace(base, stable_id="bad-moment", kind=ReductionKind.WEIGHTED_MOMENT, moment_weight="none")


def test_coefficient_modulation_requires_explicit_adapter():
    base = accepted_reduction_registry().entries()[0].identity
    modulation = RankSpec(0, "symmetric_traceless_SO2", 0, 0, None, "GeV", 0, 1, CoefficientRole.PHYSICAL_MODULATION)
    with pytest.raises(ArchitectureError, match="C2.TRANSFORM"):
        replace(base, stable_id="bad-role", target_rank=modulation, convention_adapter=IdentityState.UNSPECIFIED)


def test_rank_power_bessel_phase_and_mass_injections_remain_rejected():
    with pytest.raises(ArchitectureError, match="C1.RANK"):
        RankSpec(1, "STF", 2, 1, 1.8756, "GeV", 0, 1j, CoefficientRole.SCALAR_COEFFICIENT)
    with pytest.raises(ArchitectureError, match="C1.RANK"):
        rank_spec(2)
    with pytest.raises(ArchitectureError, match="C1.RANK"):
        rank_spec(2, 1.8756).require_transform(bessel_order=2, phase=1)

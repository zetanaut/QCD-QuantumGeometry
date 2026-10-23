"""C1 positive, negative-injection, and legacy-equivalence tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from deuteron_wigner.formal.coordinates import CoordinateKind, CoordinateSpec, coordinate_spec
from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.formal.gauge_path import ColorClass, ColorRepresentation, GluonLinkId, StapleOrientation, standard_staple
from deuteron_wigner.formal.legacy_adapters import LegacyRadialGrid, registry_operator_identity, typed_bessel_b_to_k
from deuteron_wigner.formal.maps import AdapterRegistry, MapClass, TypedAdapter, TypedMap
from deuteron_wigner.formal.operator_identity import IdentityState, OperationKind
from deuteron_wigner.formal.sector_space import ResolutionLayer, SectorId
from deuteron_wigner.formal.transverse_rank import CoefficientRole, RankSpec, rank_spec
from deuteron_wigner.fourier import bessel_b_to_k
from deuteron_wigner.gtmd import Species
from deuteron_wigner.registry import leading_twist_quark_registry
from deuteron_wigner.tmd_scheme import DELTA_COLLINS_ZETA_SCHEME, TMDScalePoint


def test_coordinate_serialization_conjugacy_and_stable_hash():
    value = coordinate_spec(CoordinateKind.B_TMD)
    restored = CoordinateSpec.from_dict(json.loads(json.dumps(value.to_dict())))
    assert value == restored and hash(value) == hash(restored)
    value.require_conjugate(coordinate_spec(CoordinateKind.K_T))


def test_c1_inject_wrong_coordinate_and_partonic_nuclear_alias():
    with pytest.raises(ArchitectureError, match=r"C1.COORD.*B_TMD.*B_DELTA"):
        coordinate_spec(CoordinateKind.B_DELTA).require_kind(CoordinateKind.B_TMD)
    with pytest.raises(ArchitectureError, match=r"C1.COORD.*P_T_NUCLEAR.*K_T"):
        coordinate_spec(CoordinateKind.K_T).require_kind(CoordinateKind.P_T_NUCLEAR)


@pytest.mark.parametrize("rank", [1, 2])
def test_c1_inject_j0_for_positive_rank(rank):
    spec = rank_spec(rank, 1.8756)
    with pytest.raises(ArchitectureError, match="C1.RANK"):
        spec.require_transform(bessel_order=0, phase=1)


def test_c1_inject_missing_mass_units_phase():
    with pytest.raises(ArchitectureError, match="reference mass"):
        rank_spec(1)
    with pytest.raises(ArchitectureError, match="mass or units"):
        RankSpec(1, "STF", 1, 1, 1.0, "MeV", 1, 1j, CoefficientRole.SCALAR_COEFFICIENT)
    with pytest.raises(ArchitectureError, match="transform mismatch"):
        rank_spec(2, 1.8756).require_transform(bessel_order=2, phase=1)


def test_c1_future_past_representation_color_and_order_are_distinct():
    future_q = standard_staple(StapleOrientation.FUTURE, ColorRepresentation.FUNDAMENTAL)
    past_q = standard_staple(StapleOrientation.PAST, ColorRepresentation.FUNDAMENTAL)
    assert future_q != past_q and future_q.inverted().staple_orientation == StapleOrientation.PAST
    future_g = standard_staple(StapleOrientation.FUTURE, ColorRepresentation.ADJOINT)
    with pytest.raises(ArchitectureError, match="adjoint"):
        GluonLinkId(future_q, future_g, ColorClass.F_TYPE)
    f_link = GluonLinkId(future_g, future_g.inverted(), ColorClass.F_TYPE)
    d_link = GluonLinkId(future_g, future_g.inverted(), ColorClass.D_TYPE)
    assert f_link != d_link
    with pytest.raises(ArchitectureError, match="color class mismatch"):
        f_link.require_color_class(ColorClass.D_TYPE)
    assert GluonLinkId(future_g, future_g.inverted(), ColorClass.F_TYPE) != GluonLinkId(future_g.inverted(), future_g, ColorClass.F_TYPE)


def test_c1_unspecified_path_rejected():
    path = standard_staple(StapleOrientation.FUTURE, ColorRepresentation.FUNDAMENTAL)
    bad = type(path)(path.endpoints, path.ordered_segments, StapleOrientation.UNSPECIFIED, path.transverse_closure, path.rapidity_direction, path.color_representation, path.boundary_class)
    with pytest.raises(ArchitectureError, match="unspecified path"):
        bad.require_production()


def _sector(role: str) -> SectorId:
    return SectorId(ResolutionLayer.HADRONIC_NUCLEAR, (), (), None, None, "all", "positive", "singlet", "accepted", role)


def test_c1_equal_shape_cannot_conflate_sectors():
    with pytest.raises(ArchitectureError, match="C1.SECTOR"):
        _sector("proton-in-deuteron").require_same_sector(_sector("neutron-in-deuteron"))


def test_c1_map_classes_endpoints_and_adapter_contracts():
    amp = TypedMap("amp", MapClass.AMP, "state", "amplitude", lambda x: x + 1, "fixture")
    dens = TypedMap("dens", MapClass.DENS, "density", "observable", lambda x: x * 2, "fixture")
    registry = AdapterRegistry()
    with pytest.raises(ArchitectureError, match="C1.MAP"):
        registry.compose(dens, amp)
    with pytest.raises(ArchitectureError, match="C1.ADAPT"):
        TypedAdapter("bad", MapClass.MATCH, "amplitude", "density", lambda x: x, "fixture")
    bridge = TypedAdapter("outer-product", MapClass.DENS, "amplitude", "density", lambda x: x, "fixture", losslessness="lossless fixture", remainder="none")
    registry.register(bridge)
    assert registry.compose(dens, amp)(2) == 6
    assert amp.map_class != dens.map_class


def test_c1_wrong_reduction_domain_and_scheme_without_match_rejected():
    red = TypedMap("red", MapClass.RED, "GTMD", "TMD", lambda x: x, "fixture")
    wrong = TypedMap("wrong", MapClass.AMP, "state", "amplitude", lambda x: x, "fixture")
    with pytest.raises(ArchitectureError, match="C1.MAP"):
        AdapterRegistry().compose(red, wrong)
    evolved = TypedMap("evolve", MapClass.MATCH, "subtracted-TMD-MSbar", "subtracted-TMD-MSbar-Q", lambda x: x, "fixture")
    bare = TypedMap("bare", MapClass.RED, "GTMD", "bare-TMD", lambda x: x, "fixture")
    with pytest.raises(ArchitectureError, match="C1.MAP"):
        AdapterRegistry().compose(evolved, bare)


def test_c1_registry_identity_and_unspecified_completeness():
    entry = leading_twist_quark_registry(Species.QUARK).get(Species.QUARK, "f1")
    operator = registry_operator_identity(entry, flavor="u", scale=TMDScalePoint.canonical(5.0), scheme=DELTA_COLLINS_ZETA_SCHEME, orientation=StapleOrientation.FUTURE)
    operator.require_complete(OperationKind.SUBTRACTED_TMD)
    incomplete = registry_operator_identity(entry, flavor=IdentityState.UNSPECIFIED, scale=TMDScalePoint.canonical(5.0), scheme=DELTA_COLLINS_ZETA_SCHEME, orientation=StapleOrientation.FUTURE)
    with pytest.raises(ArchitectureError, match="C1.OPID"):
        incomplete.require_complete(OperationKind.SUBTRACTED_TMD)


def test_c1_typed_radial_wrapper_is_numerically_identical():
    b = np.linspace(0.0, 8.0, 801)
    values = np.exp(-0.5 * b**2)
    k = np.linspace(0.0, 2.0, 20)
    grid = LegacyRadialGrid(b, coordinate_spec(CoordinateKind.B_TMD))
    actual = typed_bessel_b_to_k(grid, values, k, rank_spec(0))
    assert np.array_equal(actual, bessel_b_to_k(b, values, k, rank=0))
    with pytest.raises(ArchitectureError, match="C1.COORD"):
        typed_bessel_b_to_k(LegacyRadialGrid(b, coordinate_spec(CoordinateKind.B_DELTA)), values, k, rank_spec(0))

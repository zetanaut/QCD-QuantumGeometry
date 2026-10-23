"""C4 regulated common-parent TMD/GPD/PDF/current route closure."""

from dataclasses import replace

import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.gtmd import Species
from deuteron_wigner.pilot.c4_benchmarks import parents_from_state
from deuteron_wigner.pilot.routes import (
    CommonReductionRoutes, GluonPolarizationProjector, MatchingStatus,
    MellinConvention, project_gluon_polarization,
)
import numpy as np
from deuteron_wigner.pilot.sectors import gluon_state, sea_state


def representative_parents():
    sea = sea_state(0.25)
    q = parents_from_state(sea, Species.QUARK, flavor="d")[-1]
    qbar = parents_from_state(sea, Species.ANTIQUARK, flavor="d")[0]
    gluon = parents_from_state(gluon_state(0.3), Species.GLUON)[0]
    return q, qbar, gluon


@pytest.mark.parametrize("delta", ((0.0, 0.0), (0.15, 0.0), (0.25, -0.1)))
def test_direct_sequential_route_closure_at_multiple_transfer(delta):
    routes = CommonReductionRoutes()
    for parent in representative_parents():
        residuals = routes.close(parent, delta)
        assert residuals.maximum() <= routes.combined_tolerance
        assert routes.direct_double_integral(parent, delta).value == pytest.approx(
            routes.moment(
                parent,
                MellinConvention.GLUON_EMT_XG
                if parent.species == Species.GLUON
                else MellinConvention.QUARK_VECTOR_NET,
                delta,
            ).value,
            abs=2e-15,
        )


def test_tmd_gpd_pdf_common_forward_parent_and_metadata():
    routes = CommonReductionRoutes()
    for parent in representative_parents():
        p1 = routes.pdf_from_tmd(parent, 0.31)
        p2 = routes.pdf_from_gpd(parent, 0.31)
        assert p1.value == p2.value
        assert p1.operator_id == parent.operator_id
        assert p1.path_id == parent.path_id
        assert p1.matching_status == MatchingStatus.REGULATED_ANALYTIC
        assert routes.tmd(parent, 0.31, 0.1, -0.2).transfer == (0.0, 0.0)
        assert routes.gpd(parent, 0.31).value == p1.value


def test_deterministic_quadrature_refines_to_analytic_gpd():
    routes = CommonReductionRoutes()
    parent = representative_parents()[1]
    coarse = routes.numerical_gpd(parent, 0.3, (0.2, -0.1), points=81)
    fine = routes.numerical_gpd(parent, 0.3, (0.2, -0.1), points=161)
    assert fine.residuals.quadrature <= coarse.residuals.quadrature + 1e-15
    assert fine.residuals.quadrature < 2e-13


def test_quark_and_gluon_mellin_conventions_are_distinct():
    routes = CommonReductionRoutes()
    quark, _, gluon = representative_parents()
    assert routes.moment(quark, MellinConvention.QUARK_VECTOR_NET).mellin_convention == MellinConvention.QUARK_VECTOR_NET
    assert routes.moment(gluon, MellinConvention.GLUON_EMT_XG).mellin_convention == MellinConvention.GLUON_EMT_XG
    with pytest.raises(ArchitectureError, match="C4.CURRENT_ROUTE.GLUON_NUMBER"):
        routes.moment(gluon, MellinConvention.QUARK_VECTOR_NET)
    with pytest.raises(ArchitectureError, match="C4.CURRENT_ROUTE.MELLIN"):
        routes.moment(quark, MellinConvention.GLUON_EMT_XG)


def test_matching_and_route_injections_fail_closed():
    routes = CommonReductionRoutes()
    parent = representative_parents()[1]
    with pytest.raises(ArchitectureError, match="C4.TMD_ROUTE.TRANSFER"):
        routes.tmd(parent, 0.3, 0.0, 0.0, delta=(0.1, 0.0))
    with pytest.raises(ArchitectureError, match="C4.MATCHING_STATUS"):
        routes.gpd(replace(parent, matching_status=MatchingStatus.UV_MATCHING_REQUIRED), 0.3)
    with pytest.raises(ArchitectureError, match="C4.MATCHING_STATUS.PHYSICAL"):
        routes.reject_physical_promotion(parent.matching_status)
    with pytest.raises(ArchitectureError, match="C4.ISOLATE.PROMOTION"):
        parent.promote_to_production()


def test_typed_gluon_polarization_projectors_reconstruct_matrix():
    matrix = np.asarray(((2.0, 0.3 + 0.4j), (0.3 - 0.4j, 1.0)), complex)
    unpolarized = project_gluon_polarization(
        matrix, GluonPolarizationProjector.TRACE_UNPOLARIZED
    )
    helicity = project_gluon_polarization(
        matrix, GluonPolarizationProjector.ANTISYMMETRIC_HELICITY
    )
    linear = project_gluon_polarization(
        matrix, GluonPolarizationProjector.SYMMETRIC_TRACELESS_LINEAR
    )
    assert unpolarized == pytest.approx(3)
    assert helicity == pytest.approx(-0.8)
    assert np.trace(linear) == pytest.approx(0)
    reconstructed = (
        0.5 * np.eye(2) * unpolarized
        - 0.5j * np.asarray(((0, 1), (-1, 0))) * helicity
        + linear
    )
    assert np.allclose(reconstructed, matrix)


def test_routes_record_outstanding_matching_morphisms():
    routes = CommonReductionRoutes()
    parent = representative_parents()[1]
    assert routes.tmd(parent, .3, 0, 0).required_matching == (
        MatchingStatus.UV_MATCHING_REQUIRED,
        MatchingStatus.RAPIDITY_SOFT_MATCHING_REQUIRED,
    )
    assert MatchingStatus.LINK_SHORTENING_REQUIRED in (
        routes.gpd(parent, .3).required_matching
    )

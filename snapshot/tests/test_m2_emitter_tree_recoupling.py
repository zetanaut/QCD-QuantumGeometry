from __future__ import annotations

import json
from math import sqrt

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_emitter_tree_recoupling import (
    M2K_EMITTER_TREE_STATUS,
    build_canonical_emitter_orbit_audit,
    build_emitter_tree_recoupling_audit,
    emitter_tree_closure_record,
    jacobi_tree_matrix,
    mass_weighted_tree_matrix,
    transverse_ground_overlap,
)


def test_reference_and_cyclic_tree_matrices_are_exact_equal_share_maps() -> None:
    np.testing.assert_allclose(jacobi_tree_matrix((0, 1, 2)), np.eye(2), atol=2.0e-15)
    np.testing.assert_allclose(
        jacobi_tree_matrix((1, 2, 0)),
        ((-0.5, -1.0 / sqrt(2.0)), (3.0 / (2.0 * sqrt(2.0)), -0.5)),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        mass_weighted_tree_matrix((1, 2, 0)),
        ((-0.5, -sqrt(3.0) / 2.0), (sqrt(3.0) / 2.0, -0.5)),
        atol=2.0e-15,
    )


def test_all_six_tree_changes_preserve_jacobian_and_free_mass_metric() -> None:
    audit = build_emitter_tree_recoupling_audit()

    assert len(audit.trees) == 6
    assert audit.jacobian_residual < 1.0e-14
    assert audit.metric_residual < 1.0e-14
    assert audit.orthogonality_residual < 1.0e-14
    for matrix in audit.mass_weighted_matrices:
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(2), atol=2.0e-15)


def test_m2i_equal_local_width_ground_is_not_cyclic_s3_closed() -> None:
    audit = build_emitter_tree_recoupling_audit()
    overlaps = audit.local_equal_scale_overlaps

    assert overlaps[0] == pytest.approx(1.0, abs=2.0e-15)
    assert overlaps[1] == pytest.approx(1.0, abs=2.0e-15)
    assert min(overlaps) == pytest.approx(32.0 / 33.0, abs=2.0e-15)
    assert audit.local_ground_closure_defect == pytest.approx(1.0 / 33.0)
    assert audit.local_ground_leakage_norm == pytest.approx(sqrt(65.0) / 33.0)


def test_mass_weighted_width_choice_closes_the_single_ground_mode() -> None:
    audit = build_emitter_tree_recoupling_audit()

    assert audit.symmetry_adapted_b_rho_GeV == pytest.approx(
        0.4 * sqrt(2.0 / 3.0), abs=2.0e-15
    )
    np.testing.assert_allclose(audit.symmetry_adapted_overlaps, 1.0, atol=2.0e-15)


def test_raw_color_span_is_two_but_covariant_projected_emitter_rank_is_one() -> None:
    audit = build_canonical_emitter_orbit_audit()

    assert audit.raw_same_orbit_color_rank == 2
    assert audit.covariant_projected_rank == 1
    assert audit.orbit_vertex_residual < 1.0e-14
    assert audit.branch_collinearity_residual < 1.0e-14
    assert audit.orthogonal_source_residual < 1.0e-14
    np.testing.assert_allclose(
        audit.reduced_orbit_vertex[:, 0],
        audit.active_emitter_coordinates,
        atol=2.0e-15,
    )


def test_emitter_tree_record_changes_the_next_step_without_physical_promotion() -> None:
    first = emitter_tree_closure_record()
    second = emitter_tree_closure_record()

    assert first == second
    assert first["status"] == M2K_EMITTER_TREE_STATUS
    assert first["physical"] is False
    assert first["cyclic_ground_overlap_exact"] == "32/33"
    assert first["independent_spectator_c53_couplings_required"] is False
    assert first["covariant_projected_emitter_rank"] == 1
    assert first["m2j_full_noncolor_s3_closure_status"] == (
        "NOT_CLOSED_BY_EQUAL_LOCAL_WIDTH_GROUND_TRUNCATION"
    )
    assert "global-Jacobi" in first["next_required_object"]
    assert first["production_observable_claimed"] is False
    json.dumps(first, sort_keys=True)


def test_invalid_tree_and_width_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="permutation"):
        jacobi_tree_matrix((0, 0, 1))
    with pytest.raises(ValueError, match="positive"):
        transverse_ground_overlap(
            (0, 1, 2), b_rho_GeV=0.0, b_lambda_GeV=0.4
        )

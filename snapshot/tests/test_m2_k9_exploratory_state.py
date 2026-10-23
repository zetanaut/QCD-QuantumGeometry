from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_k9_exploratory_state import (
    K9_EXPLORATORY_BASELINE,
    ExploratoryK9ParameterPoint,
    build_parameter_explicit_k9_hamiltonian,
    derivative_report,
    exploratory_k9_state_record,
    q0_codec_report,
    sensitivity_report,
    solve_low_k9_eigenspace,
    stability_report,
)


@lru_cache(maxsize=1)
def _baseline_study():
    return solve_low_k9_eigenspace(
        build_parameter_explicit_k9_hamiltonian(K9_EXPLORATORY_BASELINE)
    )


def test_named_k9_point_constructs_an_exact_sparse_nonphysical_hamiltonian() -> None:
    assembly = _baseline_study().assembly

    assert assembly.point.point_id == "M2_K9_EXPLORATORY_BASELINE_V1"
    assert assembly.point.physical is False
    assert assembly.point.record()["physical_fit"] is False
    assert assembly.bundle.h0_is_sparse is True
    assert assembly.bundle.dimension == 1350
    assert assembly.bundle.physical is False
    assert assembly.bundle.matrix().shape == (1350, 1350)


def test_k9_lowest_space_is_an_isolated_sixfold_q_subspace_not_a_selected_state() -> None:
    study = _baseline_study()
    record = exploratory_k9_state_record(
        study,
        include_stability=False,
        include_derivatives=False,
        include_sensitivity=False,
    )

    assert study.multiplicity == 6
    assert study.degenerate is True
    assert study.energy_GeV2 == pytest.approx(0.194586374083865, abs=2.0e-13)
    assert study.gap_after_cluster_GeV2 == pytest.approx(0.412129729715685, abs=2.0e-12)
    assert max(study.residual_norms) < 1.0e-11
    assert study.sparse_matrix_free_max_abs_residual < 1.0e-12
    assert np.linalg.norm(study.projector @ study.projector - study.projector) < 1.0e-11
    assert study.symmetry_labels["q_weight"] == pytest.approx(1.0, abs=1.0e-12)
    assert study.symmetry_labels["qg_weight"] < 1.0e-24
    assert study.symmetry_labels["q_to_qg_block_max_abs"] == 0.0
    assert study.symmetry_labels["physical_state_selected"] is False
    assert record["deuteron_claim"] is False
    assert record["current_response"].startswith("NOT_EVALUATED")


def test_k9_invariant_projector_is_stable_and_derivatives_are_branch_independent() -> None:
    study = _baseline_study()
    stability = stability_report(study)
    derivatives = derivative_report(study)

    assert stability["individual_vector_tracking"] == "FORBIDDEN_DEGENERATE_SUBSPACE_TRACKED"
    assert stability["max_energy_delta_GeV2"] < 2.0e-12
    assert stability["max_residual_norm"] < 1.0e-11
    assert stability["max_principal_angle_rad"] < 1.0e-6
    assert stability["max_projector_frobenius_distance"] < 1.0e-6
    assert derivatives["individual_eigenvector_derivative"].startswith("NOT_REPORTED")
    assert derivatives["max_HF_FD_abs"] < 1.0e-8
    assert all(row["branch_independent"] for row in derivatives["rows"])


def test_declared_sensitivity_set_is_not_a_physical_fit() -> None:
    report = sensitivity_report(_baseline_study())
    rows = {row["point"]["point_id"]: row for row in report["rows"]}

    assert report["physical_parameter_inference"] is False
    assert len(rows) == 3
    assert rows["M2_K9_EXPLORATORY_MU_Q_SQ_PLUS_0P05_GEV2_V1"]["energy_shift_GeV2"] == pytest.approx(0.05, abs=2.0e-12)
    assert rows["M2_K9_EXPLORATORY_DELTA_MU_G_SQ_MINUS_0P05_GEV2_V1"]["energy_shift_GeV2"] == pytest.approx(0.0, abs=2.0e-12)
    assert rows["M2_K9_EXPLORATORY_C117_COEFFICIENT_PLUS_0P03_V1"]["energy_shift_GeV2"] == pytest.approx(-0.002320125392628, abs=2.0e-12)
    assert all(row["multiplicity"] == 6 for row in rows.values())


def test_q0_codec_round_trip_is_exact_while_q1_q2_remain_fixture_only() -> None:
    report = q0_codec_report(_baseline_study())

    assert report["Q0_encoding"] == "COMPACT_INDEX_DIRECT_ORDER_V1"
    assert report["Q0_compact_dimension"] == 1350
    assert report["Q0_padded_dimension"] == 2048
    assert report["Q0_qubits"] == 11
    assert all(row["compact_round_trip_max_abs"] == 0.0 for row in report["rows"])
    assert all(row["padded_leakage"] == 0.0 for row in report["rows"])
    assert report["Q1_route"].startswith("FIXTURE_ONLY")
    assert report["Q2_route"].startswith("FIXTURE_ONLY")


def test_k9_point_rejects_a_physical_promotion() -> None:
    with pytest.raises(ValueError, match="exploratory and nonphysical"):
        ExploratoryK9ParameterPoint(
            point_id="NOT_ALLOWED",
            mu_q_sq_GeV2=0.2,
            delta_mu_g_sq_GeV2=0.1,
            c117_residual_normalization=0.5,
            c117_mixing_coefficient=0.8,
            c117_coefficient=0.07,
            purpose="must fail closed",
            physical=True,
        )

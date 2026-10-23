from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_c53_mixed_singlet_state import (
    C53_MIXED_K9_EXPLORATORY_BASELINE,
    C53_PARAMETER_ID,
    ExploratoryC53K9Point,
    build_c53_mixed_k9_hamiltonian,
    c117_response_report,
    c53_coupling_scan,
    c53_mixed_state_record,
    helicity_derivative_report,
    solve_c53_mixed_k9_helicity_pair,
)


@lru_cache(maxsize=1)
def _baseline_pair():
    return solve_c53_mixed_k9_helicity_pair(
        build_c53_mixed_k9_hamiltonian(C53_MIXED_K9_EXPLORATORY_BASELINE)
    )


def test_c53_is_mapped_reduced_and_independently_replayed_before_mixing() -> None:
    assembly = _baseline_pair().assembly
    validation = assembly.validation

    assert assembly.dimension == 450
    assert assembly.q_dimension == 2
    assert validation["basis_permutation_applied"] is True
    assert validation["source_basis"] == "C47_SHELL_MAJOR"
    assert validation["target_basis"] == "C401_C410_TARGET_ORDER"
    assert validation["c53_source_nnz"] == 624
    assert validation["c53_reduced_nnz"] == 208
    assert validation["c53_sparse_vs_direct_reduced_max_abs"] == 0.0
    assert validation["c53_direct_reduced_vs_full_product_max_abs"] < 1.0e-14
    assert validation["base_reconstruction_max_abs"] == 0.0
    assert validation["c53_jz_commutator_max_abs"] == 0.0
    assert validation["total_jz_commutator_max_abs"] == 0.0
    assert validation["total_hermiticity_max_abs"] == 0.0
    assert _baseline_pair().sparse_matrix_free_max_abs < 1.0e-14


def test_baseline_is_a_degenerate_jz_resolved_pair_with_nonzero_qg_support() -> None:
    pair = _baseline_pair()

    assert tuple(state.helicity_twice for state in pair.states) == (-1, 1)
    np.testing.assert_allclose(
        [state.energy_GeV2 for state in pair.states],
        [0.05007934225260527, 0.05007934225260526],
        rtol=0.0,
        atol=2.0e-13,
    )
    assert abs(pair.helicity_splitting_GeV2) < 1.0e-13
    assert pair.gap_after_pair_GeV2 > 0.55
    assert np.linalg.norm(pair.projector @ pair.projector - pair.projector) < 1.0e-13
    for state in pair.states:
        assert state.residual_norm_GeV2 < 1.0e-13
        assert state.q_weight + state.qg_weight == pytest.approx(1.0, abs=2.0e-15)
        assert state.qg_weight == pytest.approx(0.0819939207777390, abs=2.0e-13)
        assert state.sector_gap_GeV2 > 0.55


def test_zero_coupling_limit_and_continuous_qg_generation_are_explicit() -> None:
    scan = c53_coupling_scan()
    rows = scan["rows"]

    assert tuple(row["c53_coupling"] for row in rows) == (0.0, 0.01, 0.03, 0.05)
    assert rows[0]["energies_GeV2"] == pytest.approx(
        (0.1945863740838668, 0.1945863740838668), abs=2.0e-13
    )
    assert rows[0]["qg_weights"] == pytest.approx((0.0, 0.0), abs=1.0e-15)
    assert all(abs(row["helicity_splitting_GeV2"]) < 1.0e-12 for row in rows)
    assert all(
        rows[index]["qg_weights"][0] < rows[index + 1]["qg_weights"][0]
        for index in range(len(rows) - 1)
    )
    assert all(
        rows[index]["energies_GeV2"][0] > rows[index + 1]["energies_GeV2"][0]
        for index in range(len(rows) - 1)
    )
    assert scan["physical_coupling_selected"] is False


def test_lawful_parameter_derivatives_agree_on_both_helicity_branches() -> None:
    report = helicity_derivative_report(_baseline_pair())
    rows = {row["parameter_id"]: row for row in report["rows"]}

    assert set(rows) == {
        "c396:D_mu_q_sq",
        "c396:D_delta_mu_g_sq",
        "c117:I2_density_projector",
        C53_PARAMETER_ID,
    }
    assert report["max_HF_FD_abs"] < 1.0e-8
    assert report["helicity_branches_tracked_by"].startswith("EXACT_JZ")
    assert all(
        row["helicity_spread"] < 1.0e-12 and len(row["branches"]) == 2
        for row in rows.values()
    )
    assert rows[C53_PARAMETER_ID]["helicity_average"] < -5.20
    assert report["physical_parameter_inference"] is False


def test_mapped_c117_is_jz_conserving_and_qg_support_changes_its_response() -> None:
    pair = _baseline_pair()
    report = c117_response_report(pair)

    assert pair.assembly.validation["c117_baseline_coefficient"] == 0.07
    assert pair.assembly.validation["c117_helicity_hamiltonian_status"].startswith(
        "MAPPED_JZ_CONSERVING"
    )
    assert report["jz_commutator_max_abs"] == 0.0
    assert report["zero_c53_q_sector_first_order_slopes"] == pytest.approx(
        (-0.07733751308761731, -0.07733751308761731), abs=2.0e-13
    )
    assert report["projected_first_order_slopes"] == pytest.approx(
        (-0.12786328537578978, -0.12786328537578970), abs=2.0e-13
    )
    assert all(abs(value) > 0.05 for value in report["qg_induced_first_order_slope_shift"])
    assert min(report["outside_lowest_pair_action_norms"]) > 0.22
    assert report["nonzero_response_requires_qg_support"] is False
    assert report["q_sector_response_already_present"] is True
    assert report["qg_support_changes_c117_response"] is True
    assert report["finite_exploratory_coefficient_authorized"] is True
    assert report["physical_coefficient_authorized"] is False


def test_c53_coupling_sign_changes_phase_not_energy_or_fock_weights() -> None:
    positive = _baseline_pair()
    negative = solve_c53_mixed_k9_helicity_pair(
        build_c53_mixed_k9_hamiltonian(
            replace(
                C53_MIXED_K9_EXPLORATORY_BASELINE,
                c53_coupling=-C53_MIXED_K9_EXPLORATORY_BASELINE.c53_coupling,
            )
        )
    )
    np.testing.assert_allclose(
        [state.energy_GeV2 for state in negative.states],
        [state.energy_GeV2 for state in positive.states],
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        [state.qg_weight for state in negative.states],
        [state.qg_weight for state in positive.states],
        rtol=0.0,
        atol=2.0e-13,
    )


def test_physical_promotion_is_rejected() -> None:
    with pytest.raises(ValueError, match="exploratory and nonphysical"):
        ExploratoryC53K9Point(
            point_id="NOT_ALLOWED",
            base_point=C53_MIXED_K9_EXPLORATORY_BASELINE.base_point,
            c53_coupling=0.05,
            purpose="must fail",
            physical=True,
        )


def test_state_record_stops_before_spin_flavor_deuteron_and_current_claims() -> None:
    record = c53_mixed_state_record(_baseline_pair(), include_derivatives=False)

    assert record["status"] == "M2_C53_MIXED_COLOR_SINGLET_K9_HELICITY_PAIR_READY"
    assert record["color_singlet_completed"] is True
    assert record["spectator_spin_flavor_charge_constructed"] is False
    assert record["deuteron_H_D_constructed"] is False
    assert record["current_constructed"] is False
    assert record["physical"] is False

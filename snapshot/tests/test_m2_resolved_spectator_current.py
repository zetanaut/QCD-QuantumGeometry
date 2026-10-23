from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from fractions import Fraction
import json
from math import exp, sqrt

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_active_quark_current import (
    build_m2_active_quark_impulse_operator,
)
from deuteron_wigner.quantum.m2_c53_mixed_singlet_state import (
    C53_MIXED_K9_EXPLORATORY_BASELINE,
    build_c53_mixed_k9_hamiltonian,
    solve_c53_mixed_k9_helicity_pair,
)
from deuteron_wigner.quantum.m2_nn_deuteron_state import (
    NN_AV18_NUCLEAR_R1,
    build_av18_wave_member,
    build_nn_light_front_wave_grid,
    compose_nn_deuteron_state,
)
from deuteron_wigner.quantum.m2_nn_transfer_current import build_nn_transfer_fibers
from deuteron_wigner.quantum.m2_nucleon_spectator_composition import (
    AXIAL_SPECTATOR,
    SCALAR_SPECTATOR,
    build_nucleon_spectator_doublet,
)
from deuteron_wigner.quantum.m2_resolved_spectator_current import (
    M2_RESOLVED_CURRENT_STATUS,
    M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1,
    M2_RESOLVED_SPECTATOR_STATUS,
    ResolvedSpectatorResolution,
    build_resolved_spectator_current_factors,
    build_resolved_spectator_hamiltonian,
    evaluate_nn_resolved_spectator_impulse_current,
    nn_resolved_spectator_current_pair_record,
    nucleon_resolved_impulse_components,
    resolved_sector_longitudinal_records,
    resolved_spectator_hamiltonian_record,
)
from deuteron_wigner.quantum.m2_resolved_spectator_response import (
    M2_RESOLVED_RESPONSE_STATUS,
    evaluate_nn_resolved_spectator_response,
    nn_resolved_spectator_response_record,
)


@lru_cache(maxsize=1)
def _pair():
    return solve_c53_mixed_k9_helicity_pair(
        build_c53_mixed_k9_hamiltonian(C53_MIXED_K9_EXPLORATORY_BASELINE)
    )


@lru_cache(maxsize=1)
def _state():
    grid = build_nn_light_front_wave_grid(
        resolution=NN_AV18_NUCLEAR_R1,
        wave_member=build_av18_wave_member(),
        nucleon_mass_GeV=0.9389,
    )
    return compose_nn_deuteron_state(
        _pair(), wave_grid=grid, channel=SCALAR_SPECTATOR
    )


@lru_cache(maxsize=1)
def _currents():
    state = _state()
    hamiltonian = build_resolved_spectator_hamiltonian(state.pair)
    result = []
    for delta_x in (0.2, -0.2, 0.0):
        transfer = build_nn_transfer_fibers(
            state,
            delta_x_GeV=delta_x,
            delta_y_GeV=0.0,
            p_plus_GeV=2.0,
            reference_deuteron_mass_GeV=1.87561294257,
        )
        operator = build_m2_active_quark_impulse_operator(
            "K9", delta_x_GeV=delta_x, delta_y_GeV=0.0
        )
        result.append(
            evaluate_nn_resolved_spectator_impulse_current(
                transfer, operator, hamiltonian
            )
        )
    return tuple(result)


@lru_cache(maxsize=1)
def _response():
    return evaluate_nn_resolved_spectator_response(_state())


def test_resolved_longitudinal_modes_close_exactly_at_equal_valence_share() -> None:
    resolution = M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1

    assert resolution.active_cluster_K2 == 9
    assert resolution.spectator_quark_K2 == (9, 9)
    assert resolution.nucleon_K2 == 27
    assert resolution.nucleon_K == Fraction(27, 2)
    assert resolution.active_cluster_fraction == Fraction(1, 3)
    assert resolution.spectator_quark_fractions == (Fraction(1, 3),) * 2
    assert resolution.spectator_cluster_fraction == Fraction(2, 3)
    assert resolution.spectator_internal_fractions == (Fraction(1, 2),) * 2
    records = resolved_sector_longitudinal_records(resolution)
    assert len(records) == 5
    assert records[0]["K2_modes"] == (9, 9, 9)
    assert all(record["fraction_sum_exact"] == "1" for record in records)
    assert all(record["K2_sum_residual"] == 0 for record in records)


def test_resolved_member_uses_common_c45_ground_mode_and_rejects_unsafe_aliases() -> None:
    resolution = M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1

    assert resolution.rho_mode == resolution.lambda_mode == (0, 0)
    assert resolution.b_rho_GeV == resolution.b_lambda_GeV == 0.4
    with pytest.raises(ValueError, match="share the K9 C45 scale"):
        replace(resolution, b_rho_GeV=0.45)
    with pytest.raises(ValueError, match="exchange-equal"):
        replace(resolution, spectator_quark_K2=(7, 11))
    with pytest.raises(ValueError, match="Jacobi ground modes"):
        replace(resolution, rho_mode=(0, 1))


def test_projected_free_mass_hamiltonian_has_exact_nested_formula_and_eigenvectors() -> None:
    embedding = build_resolved_spectator_hamiltonian(_pair())
    record = resolved_spectator_hamiltonian_record(embedding)

    assert record["status"] == M2_RESOLVED_SPECTATOR_STATUS
    assert embedding.spectator_quark_mass_sq_GeV2 == 0.2
    assert embedding.rho_k2_expectation_GeV2 == pytest.approx(0.16)
    assert embedding.lambda_k2_expectation_GeV2 == pytest.approx(0.16)
    assert embedding.spectator_cluster_mass_sq_GeV2 == pytest.approx(0.96)
    assert embedding.active_mass_scale == 3.0
    assert embedding.additive_identity_shift_GeV2 == pytest.approx(1.60)
    assert record["normalized_transverse_fraction_weights_consumed_once"] is True
    assert record["projected_free_mass_formula"] == (
        "M_N^2=M_A^2/y_A+M_S^2/y_S+q_lambda^2; "
        "M_S^2=sum_i(m_q^2/z_i)+q_rho^2"
    )
    assert record["hermiticity_residual"] == 0.0
    assert embedding.embedded_pair_residual_GeV2 < 1.2e-14
    assert record["physical_nucleon_mass_claimed"] is False
    assert record["complete_c7_h2_basis_constructed"] is False


def test_resolved_hamiltonian_is_not_the_old_identity_extension() -> None:
    embedding = build_resolved_spectator_hamiltonian(_pair())
    active = _pair().assembly.matrix()

    assert np.max(np.abs((embedding.matrix - active).data)) > 2.0
    expected = 3.0 * np.asarray(
        [_pair().state(-1).energy_GeV2, _pair().state(1).energy_GeV2]
    ) + 1.60
    np.testing.assert_allclose(
        embedding.embedded_pair_energies_GeV2, expected, atol=2.0e-14
    )


def test_resolved_recoil_factors_follow_two_nested_jacobi_maps() -> None:
    resolution = M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1
    positive = build_resolved_spectator_current_factors(
        resolution, delta_x_GeV=0.2, delta_y_GeV=0.0
    )
    negative = build_resolved_spectator_current_factors(
        resolution, delta_x_GeV=-0.2, delta_y_GeV=0.0
    )
    forward = build_resolved_spectator_current_factors(
        resolution, delta_x_GeV=0.0, delta_y_GeV=0.0
    )

    np.testing.assert_allclose(positive.active_lambda_shift_GeV, (0.2 * sqrt(2), 0.0))
    np.testing.assert_allclose(positive.spectator_lambda_shift_GeV, (-0.2 / sqrt(2), 0.0))
    np.testing.assert_allclose(positive.spectator_rho_shift_GeV, (0.2, 0.0))
    assert positive.active_cluster_overlap == pytest.approx(exp(-0.125), abs=2.0e-15)
    assert positive.spectator_overlap == pytest.approx(exp(-0.09375), abs=2.0e-15)
    assert negative.active_cluster_overlap == pytest.approx(
        positive.active_cluster_overlap.conjugate(), abs=2.0e-15
    )
    assert negative.spectator_overlap == pytest.approx(
        positive.spectator_overlap.conjugate(), abs=2.0e-15
    )
    assert forward.active_cluster_overlap == 1.0
    assert forward.spectator_overlap == 1.0


@pytest.mark.parametrize("channel", (SCALAR_SPECTATOR, AXIAL_SPECTATOR))
def test_all_three_resolved_quark_currents_close_proton_and_neutron_charge(
    channel: str,
) -> None:
    factors = build_resolved_spectator_current_factors(
        M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1,
        delta_x_GeV=0.0,
        delta_y_GeV=0.0,
    )
    pair_current = np.eye(2, dtype=np.complex128)
    proton = build_nucleon_spectator_doublet(
        _pair(), target="PROTON", channel=channel
    )
    neutron = build_nucleon_spectator_doublet(
        _pair(), target="NEUTRON", channel=channel
    )

    p_active, p_spectator, p_total = nucleon_resolved_impulse_components(
        proton, pair_current, factors
    )
    n_active, n_spectator, n_total = nucleon_resolved_impulse_components(
        neutron, pair_current, factors
    )
    np.testing.assert_allclose(p_total, np.eye(2), atol=4.0e-16)
    np.testing.assert_allclose(n_total, 0.0, atol=4.0e-16)
    np.testing.assert_allclose(p_active + n_active, np.eye(2) / 3.0, atol=4.0e-16)
    np.testing.assert_allclose(
        p_spectator + n_spectator, 2.0 * np.eye(2) / 3.0, atol=4.0e-16
    )


def test_resolved_spectator_replaces_pointlike_identity_at_nonzero_transfer() -> None:
    positive, _, _ = _currents()

    assert abs(positive.factors.active_cluster_overlap - 1.0) > 0.1
    assert abs(positive.factors.spectator_overlap - 1.0) > 0.08
    assert np.max(np.abs(positive.active_target_matrix)) > 0.1
    assert np.max(np.abs(positive.spectator_target_matrix)) > 0.3
    # The active and spectator mechanisms are accumulated independently over
    # 5376 nodes; their recombination carries the corresponding summation floor.
    assert positive.mechanism_reconstruction_residual() < 2.0e-13


def test_matched_current_closes_charge_reversal_parity_and_sd_decomposition() -> None:
    record = nn_resolved_spectator_current_pair_record(*_currents())

    assert record["status"] == M2_RESOLVED_CURRENT_STATUS
    assert record["forward_deuteron_charge_residual"] < 2.0e-14
    assert record["target_reversal_adjoint_residual"] < 2.0e-15
    assert record["component_reversal_residual"] < 2.0e-15
    assert record["parity_residual"] < 2.0e-15
    assert record["zero_plus_relation_residual"] < 2.0e-15
    assert record["sd_component_reconstruction_residual"] < 2.0e-15
    assert record["active_spectator_reconstruction_residual"] < 2.0e-13
    assert record["pointlike_spectator_identity_used"] is False
    assert record["hamiltonian_consistent_interaction_current_constructed"] is False


def test_m2i_response_retains_m2h_rank_two_and_m2g_null_control() -> None:
    response = _response()
    record = nn_resolved_spectator_response_record(response)

    assert record["status"] == M2_RESOLVED_RESPONSE_STATUS
    assert response.diagnostic_pair_response_rank == 2
    assert response.diagnostic_current_response_rank == 2
    assert record["m2h_parent_diagnostic_rank"] == 2
    assert record["m2g_identity_control_max_abs"] == 0.0
    assert response.active_response_factorization_residual == 0.0
    assert record["physical_response_rank"] is None


def test_resolved_current_response_is_the_active_overlap_times_m2h() -> None:
    response = _response()
    parent = response.parent_active_response
    factor = response.baseline_current.factors.active_cluster_overlap

    np.testing.assert_allclose(
        response.finite_difference_current_response,
        factor * parent.finite_difference_current_response,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        response.current_response_singular_values,
        abs(factor) * parent.current_response_singular_values,
        rtol=2.0e-13,
        atol=2.0e-15,
    )


def test_resolved_spectator_changes_but_does_not_repair_angular_condition() -> None:
    response = _response()
    record = nn_resolved_spectator_response_record(response)
    angular = complex(record["baseline_angular_condition"])

    assert abs(angular) > 1.5e-4
    assert abs(angular) < 1.7e-4
    assert np.max(np.abs(response.angular_condition_response)) > 8.0e-4
    assert record["production_current_selected"] is False
    assert record["interaction_and_number_changing_currents_complete"] is False


def test_resolved_record_is_deterministic_json_safe_and_preserves_nonclaims() -> None:
    response = _response()
    first = nn_resolved_spectator_response_record(response)
    second = nn_resolved_spectator_response_record(response)

    assert first["root"] == second["root"]
    assert first["hamiltonian"]["full_active_spectator_antisymmetrization_constructed"] is False
    assert first["current"]["complete_diagonal_one_body_current_in_represented_factorization"] is True
    assert first["current"]["production_current"] is False
    json.dumps(first, sort_keys=True, default=str)
    with pytest.raises(ValueError, match="remain exploratory"):
        replace(response, physical=True)


def test_invalid_resolved_current_inputs_fail_closed() -> None:
    positive, reversed_current, forward = _currents()

    with pytest.raises(ValueError, match="opposite transfer"):
        nn_resolved_spectator_current_pair_record(positive, positive, forward)
    with pytest.raises(ValueError, match="zero transfer"):
        nn_resolved_spectator_current_pair_record(
            positive, reversed_current, positive
        )
    with pytest.raises(TypeError, match="ResolvedSpectatorResolution"):
        build_resolved_spectator_current_factors(
            "K9", delta_x_GeV=0.2, delta_y_GeV=0.0
        )


def test_reference_values_guard_the_m2i_regulator_and_current_convention() -> None:
    response = _response()
    record = nn_resolved_spectator_response_record(response)

    np.testing.assert_allclose(
        response.baseline_current.target_matrix.real,
        (
            (0.5572702514, 0.0015925720, -0.0388785646),
            (-0.0015925720, 0.5211615700, 0.0015925720),
            (-0.0388785646, -0.0015925720, 0.5572702514),
        ),
        rtol=0.0,
        atol=8.0e-9,
    )
    assert record["hamiltonian"]["embedded_pair_energies_GeV2"][0] == pytest.approx(
        1.750238026757816, abs=2.0e-13
    )
    assert complex(record["baseline_angular_condition"]).real == pytest.approx(
        0.00015814202954123058, abs=2.0e-13
    )

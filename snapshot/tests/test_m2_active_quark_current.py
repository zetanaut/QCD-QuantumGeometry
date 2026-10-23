from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
from math import sqrt

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_active_quark_current import (
    M2_ACTIVE_CURRENT_STATUS,
    NN_ACTIVE_IMPULSE_STATUS,
    active_operator_pair_record,
    build_m2_active_quark_impulse_operator,
    displaced_ho_overlap,
    nn_active_impulse_pair_record,
    nucleon_active_impulse_matrix,
    pair_active_current_matrix,
)
from deuteron_wigner.quantum.m2_c53_mixed_singlet_state import (
    C53_MIXED_K9_EXPLORATORY_BASELINE,
    build_c53_mixed_k9_hamiltonian,
    solve_c53_mixed_k9_helicity_pair,
)
from deuteron_wigner.quantum.m2_color_singlet_bridge import m2_color_modules
from deuteron_wigner.quantum.m2_internal_current_response import (
    DIAGNOSTIC_RANK_RELATIVE_TOLERANCE,
    M2_INTERNAL_CURRENT_RESPONSE_STATUS,
    evaluate_nn_internal_current_response,
    nn_internal_current_response_record,
)
from deuteron_wigner.quantum.m2_nn_deuteron_state import (
    NN_AV18_NUCLEAR_R1,
    build_av18_wave_member,
    build_nn_light_front_wave_grid,
    compose_nn_deuteron_state,
)
from deuteron_wigner.quantum.m2_nucleon_spectator_composition import (
    AXIAL_SPECTATOR,
    SCALAR_SPECTATOR,
    build_nucleon_spectator_doublet,
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
def _response():
    return evaluate_nn_internal_current_response(_state())


def test_displaced_ho_overlap_has_exact_forward_limit_and_converged_quadrature() -> None:
    modes = ((0, 0), (0, -1), (0, 1), (1, 0))
    forward = displaced_ho_overlap(modes, 0.4, 0.0, 0.0, 28)
    low = displaced_ho_overlap(modes, 0.4, 0.2, -0.07, 24)
    high = displaced_ho_overlap(modes, 0.4, 0.2, -0.07, 36)

    np.testing.assert_array_equal(forward, np.eye(len(modes)))
    assert np.max(np.abs(low - high)) < 2.0e-14


def test_active_operator_obeys_charge_reversal_and_source_recoil() -> None:
    response = _response()
    record = response.operator_record

    assert record["status"] == M2_ACTIVE_CURRENT_STATUS
    assert record["source_qualified_operator_form"] is True
    assert record["complete_nucleon_current"] is False
    assert record["forward_charge_residual"] == 0.0
    assert record["reversal_adjoint_residual"] < 3.0e-15
    np.testing.assert_allclose(
        [row["relative_shift_factor"] for row in record["partition_records"]],
        [sqrt(2.0 / 7.0), sqrt(4.0 / 5.0), sqrt(2.0), sqrt(8.0)],
        atol=2.0e-15,
    )


def test_active_operator_is_diagonal_in_fock_partition_and_parton_helicity() -> None:
    operator = _response().operator.matrix.tocoo()
    labels = m2_color_modules("K9").multiplicity_labels

    for row, column in zip(operator.row, operator.col):
        out = labels[int(row)]
        inn = labels[int(column)]
        assert out[0] == inn[0]
        if out[0] == "q":
            assert out == inn
        else:
            assert out[1] == inn[1]
            assert out[4:6] == inn[4:6]


def test_active_operator_retains_finite_shell_leakage_without_pruning() -> None:
    record = _response().operator_record

    assert record["finite_shell_projection_is_unitary"] is False
    assert record["threshold_pruned"] is False
    assert record["minimum_retained_partition_singular_value"] > 0.05
    assert record["maximum_partition_contraction_excess"] < 2.0e-14


def test_pair_current_is_nonidentity_and_reverses_by_adjunction() -> None:
    response = _response()
    positive = pair_active_current_matrix(_pair(), response.operator)
    negative = pair_active_current_matrix(_pair(), response.reversed_operator)

    assert np.max(np.abs(positive - np.eye(2))) > 3.0e-3
    np.testing.assert_allclose(negative, positive.conj().T, atol=8.0e-16)
    assert abs(positive[0, 1]) > 7.0e-4


@pytest.mark.parametrize("channel", (SCALAR_SPECTATOR, AXIAL_SPECTATOR))
def test_pointlike_spectator_lift_closes_target_charge_only_at_forward_limit(
    channel: str,
) -> None:
    pair = _pair()
    forward = np.eye(2, dtype=np.complex128)
    proton = build_nucleon_spectator_doublet(
        pair, target="PROTON", channel=channel
    )
    neutron = build_nucleon_spectator_doublet(
        pair, target="NEUTRON", channel=channel
    )

    np.testing.assert_allclose(
        nucleon_active_impulse_matrix(proton, forward), np.eye(2), atol=3.0e-16
    )
    np.testing.assert_allclose(
        nucleon_active_impulse_matrix(neutron, forward), 0.0, atol=3.0e-16
    )


def test_nn_active_impulse_current_closes_charge_reversal_and_sd_reconstruction() -> None:
    record = _response().current_record

    assert record["status"] == NN_ACTIVE_IMPULSE_STATUS
    assert record["internal_operator_source_qualified"] is True
    assert record["pointlike_spectator_is_physical_matching"] is False
    assert record["forward_deuteron_charge_residual"] < 1.2e-14
    assert record["target_reversal_adjoint_residual"] < 2.0e-16
    assert record["component_reversal_residual"] < 2.0e-16
    assert record["parity_residual"] < 2.0e-16
    assert record["zero_plus_relation_residual"] < 2.0e-16
    assert record["sd_component_reconstruction_residual"] == 0.0


def test_m2h_is_the_first_nonnull_microscopic_current_response() -> None:
    response = _response()
    record = nn_internal_current_response_record(response)

    assert record["status"] == M2_INTERNAL_CURRENT_RESPONSE_STATUS
    assert np.max(np.abs(response.pair_current_response)) > 0.1
    assert np.max(np.abs(response.finite_difference_current_response)) > 0.02
    assert np.max(np.abs(response.identity_control_response)) == 0.0
    assert record["pointlike_identity_control_preserved"] is True


def test_pair_and_spin_one_response_each_resolve_two_directions() -> None:
    response = _response()
    record = nn_internal_current_response_record(response)

    assert response.diagnostic_pair_response_rank == 2
    assert response.diagnostic_current_response_rank == 2
    assert record["diagnostic_rank_relative_tolerance"] == (
        DIAGNOSTIC_RANK_RELATIVE_TOLERANCE
    )
    assert response.pair_response_singular_values[1] > 2.0e-4
    # Subtracting two O(1) pair currents and dividing by 2h gives an
    # O(eps/h) entrywise floor. After column scaling, bound its real/imag
    # embedding by the Frobenius norm (and hence each singular-value error).
    # Do not demand identical sub-roundoff singular values across BLAS builds.
    scales = np.asarray([response.parameter_scales[p] for p in response.parameter_ids])
    fd_roundoff_floor = (
        np.sqrt(2 * response.pair_current_response.shape[0])
        * np.linalg.norm(scales)
        * np.finfo(float).eps / response.finite_difference_step
    )
    assert fd_roundoff_floor < response.pair_response_singular_values[1] * 1e-6
    assert response.pair_response_singular_values[2] < fd_roundoff_floor
    assert response.current_response_singular_values[1] > 2.0e-5
    assert response.current_response_singular_values[2] < 2.0e-10


def test_state_tangent_and_direct_pair_current_derivatives_agree() -> None:
    response = _response()

    assert response.tangent_pair_consistency_residual < 4.0e-10
    assert np.max(np.abs(response.state_tangent_matrix)) > 0.1


def test_internal_current_changes_but_does_not_repair_the_angular_condition() -> None:
    response = _response()
    record = nn_internal_current_response_record(response)

    active = complex(**record["baseline_angular_condition"])
    identity = complex(**record["identity_control_angular_condition"])
    assert abs(active) > 1.7e-4
    assert abs(identity) == pytest.approx(1.4975815808637716e-4, abs=2.0e-14)
    assert abs(active - identity) > 2.0e-5
    assert np.max(np.abs(response.angular_condition_response)) > 1.0e-3


def test_response_record_is_deterministic_json_safe_and_keeps_physical_rank_open() -> None:
    response = _response()
    first = nn_internal_current_response_record(response)
    second = nn_internal_current_response_record(response)

    assert first["root"] == second["root"]
    assert first["operator_source_qualified_within_scope"] is True
    assert first["complete_microscopic_to_nucleon_matching_constructed"] is False
    assert first["physical_response_rank"] is None
    json.dumps(first, sort_keys=True)
    with pytest.raises(ValueError, match="exploratory response"):
        replace(response, physical=True)


def test_invalid_operator_and_response_inputs_fail_closed() -> None:
    response = _response()

    with pytest.raises(ValueError, match="at least 16"):
        displaced_ho_overlap(((0, 0),), 0.4, 0.2, 0.0, 8)
    with pytest.raises(ValueError, match="opposite transfer"):
        active_operator_pair_record(
            response.operator, response.operator, response.forward_operator
        )
    with pytest.raises(ValueError, match="positive x-directed"):
        evaluate_nn_internal_current_response(_state(), delta_x_GeV=-0.2)


def test_reference_values_guard_the_current_convention_and_parameter_order() -> None:
    record = nn_internal_current_response_record(_response())

    np.testing.assert_allclose(
        record["baseline_pair_current_real"],
        ((0.9965320376106648, -0.0007069602677318234),
         (0.0007069602677318231, 0.9965320376106650)),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        record["standardized_pair_response_singular_values"][:2],
        (0.0017859819689507131, 0.00024033124363649188),
        rtol=2.0e-7,
        atol=2.0e-12,
    )

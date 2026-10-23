from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_c53_mixed_singlet_state import (
    C53_MIXED_K9_EXPLORATORY_BASELINE,
    build_c53_mixed_k9_hamiltonian,
    solve_c53_mixed_k9_helicity_pair,
    vary_c53_mixed_k9_point,
)
from deuteron_wigner.quantum.m2_nn_current_response import (
    DEFAULT_PARAMETER_SCALES,
    EXPECTED_PARAMETER_IDS,
    HELICITY_CURRENT_IDS,
    NN_CURRENT_RESPONSE_STATUS,
    evaluate_nn_microscopic_current_response,
    nn_microscopic_current_response_record,
)
from deuteron_wigner.quantum.m2_nn_deuteron_state import (
    NN_AV18_NUCLEAR_R1,
    build_av18_wave_member,
    build_nn_light_front_wave_grid,
    compose_nn_deuteron_state,
)
from deuteron_wigner.quantum.m2_nn_transfer_current import NNOneBodyCurrentMember
from deuteron_wigner.quantum.m2_nucleon_spectator_composition import SCALAR_SPECTATOR


@lru_cache(maxsize=1)
def _state():
    pair = solve_c53_mixed_k9_helicity_pair(
        build_c53_mixed_k9_hamiltonian(C53_MIXED_K9_EXPLORATORY_BASELINE)
    )
    grid = build_nn_light_front_wave_grid(
        resolution=NN_AV18_NUCLEAR_R1,
        wave_member=build_av18_wave_member(),
        nucleon_mass_GeV=0.9389,
    )
    return compose_nn_deuteron_state(
        pair,
        wave_grid=grid,
        channel=SCALAR_SPECTATOR,
    )


@lru_cache(maxsize=1)
def _response():
    return evaluate_nn_microscopic_current_response(_state())


def test_m2g_owns_the_four_microscopic_directions_and_dimensionless_scales() -> None:
    response = _response()

    assert response.parameter_ids == EXPECTED_PARAMETER_IDS
    assert response.parameter_scales == DEFAULT_PARAMETER_SCALES
    assert response.finite_difference_step == pytest.approx(1.0e-6)
    assert tuple(response.parameter_units.values()) == (
        "GeV^2",
        "GeV^2",
        "dimensionless",
        "dimensionless",
    )


def test_all_endpoint_states_are_recomputed_and_have_nonzero_tangents() -> None:
    response = _response()

    assert np.min(response.state_tangent_norms) > 0.18
    np.testing.assert_allclose(
        response.state_tangent_norms[0],
        response.state_tangent_norms[1],
        rtol=0.0,
        atol=4.0e-8,
    )
    assert response.diagnostic_internal_state_tangent_rank == 4
    assert np.min(response.internal_state_tangent_singular_values) > 0.004


def test_energy_and_qg_weight_responses_show_real_microscopic_state_motion() -> None:
    response = _response()
    expected_energy = np.asarray((1.06927742, 0.31533206, -0.12786329, -5.30633333))
    expected_qg = np.asarray((-0.06897626, -0.37415080, 0.06384872, 2.44270299))

    np.testing.assert_allclose(
        response.energy_response_by_helicity[0], expected_energy, atol=8.0e-8
    )
    np.testing.assert_allclose(
        response.qg_weight_response_by_helicity[0], expected_qg, atol=8.0e-8
    )
    np.testing.assert_allclose(
        response.energy_response_by_helicity[0],
        response.energy_response_by_helicity[1],
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        response.qg_weight_response_by_helicity[0],
        response.qg_weight_response_by_helicity[1],
        atol=1.0e-8,
    )


def test_independently_recomputed_helicity_current_jacobian_is_exactly_null() -> None:
    response = _response()
    record = nn_microscopic_current_response_record(response)

    assert HELICITY_CURRENT_IDS == ("I11", "I10", "I1m1", "I00")
    assert np.max(np.abs(response.finite_difference_current_response)) == 0.0
    assert record["finite_difference_endpoint_current_evaluations"] == 8
    assert record["internal_state_variations_recomputed"] is True


def test_internal_identity_route_and_central_differences_agree_exactly() -> None:
    response = _response()

    assert np.max(np.abs(response.analytic_identity_current_response)) == 0.0
    assert response.central_difference_consistency_residual == 0.0
    assert np.max(response.internal_gram_finite_difference_residuals) < 1.0e-8
    assert np.max(response.internal_gram_derivative_residuals) < 1.0e-6


def test_angular_condition_is_nonzero_but_its_microscopic_response_is_null() -> None:
    response = _response()
    record = nn_microscopic_current_response_record(response)

    assert record["baseline_angular_condition"]["real"] == pytest.approx(
        0.00014975815808637716,
        abs=2.0e-14,
    )
    assert record["baseline_angular_condition"]["imag"] == pytest.approx(0.0, abs=2.0e-17)
    assert record["angular_condition_response_max_abs"] == 0.0


def test_current_singular_spectrum_is_null_without_a_physical_rank_claim() -> None:
    response = _response()
    record = nn_microscopic_current_response_record(response)

    np.testing.assert_array_equal(response.current_response_singular_values, 0.0)
    assert response.diagnostic_current_response_rank == 0
    assert record["rank_is_physical"] is False
    assert record["physical_response_rank"] is None
    assert record["source_qualified_microscopic_current_constructed"] is False


def test_full_matrix_and_each_coherent_sd_component_are_independently_null() -> None:
    response = _response()

    assert np.max(np.abs(response.full_target_matrix_response)) == 0.0
    assert set(response.component_response_max_abs) == {"SS", "SD", "DS", "DD"}
    assert all(
        np.max(values) == 0.0
        for values in response.component_response_max_abs.values()
    )


def test_response_record_is_json_safe_deterministic_and_fail_closed() -> None:
    response = _response()
    first = nn_microscopic_current_response_record(response)
    second = nn_microscopic_current_response_record(response)

    assert first["status"] == NN_CURRENT_RESPONSE_STATUS
    assert first["root"] == second["root"]
    assert first["null_response_interpretation"].startswith("FACTORIZED_POINTLIKE")
    json.dumps(first, sort_keys=True)
    with pytest.raises(ValueError, match="exploratory response diagnostic"):
        replace(response, physical=True)


def test_invalid_transfer_step_current_and_parameter_route_are_rejected() -> None:
    state = _state()
    alternate = NNOneBodyCurrentMember(
        member_id="UNSUPPORTED_DIAGNOSTIC_MEMBER",
        proton_f1=0.9,
        proton_f2=0.0,
        neutron_f1=0.1,
        neutron_f2=0.0,
        q2_dependence="CONSTANT_TEST_ONLY",
    )

    with pytest.raises(ValueError, match="positive x-directed"):
        evaluate_nn_microscopic_current_response(state, delta_x_GeV=-0.2)
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_nn_microscopic_current_response(state, finite_difference_step=0.0)
    with pytest.raises(ValueError, match="fixed pointlike"):
        evaluate_nn_microscopic_current_response(state, current_member=alternate)
    with pytest.raises(KeyError):
        vary_c53_mixed_k9_point(
            state.pair.assembly.point,
            "unsupported:parameter",
            1.0e-6,
        )

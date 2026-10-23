from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

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
from deuteron_wigner.quantum.m2_nn_transfer_current import (
    NN_POINTLIKE_CHARGE_CURRENT_V1,
    NN_RECOIL_AUTHORITY_ID,
    OMITTED_CURRENT_TERMS,
    REFERENCE_MASS_ROLE,
    build_nn_transfer_fibers,
    evaluate_nn_one_body_diagnostic_current,
    nn_one_body_current_pair_record,
)
from deuteron_wigner.quantum.m2_nucleon_spectator_composition import (
    AXIAL_SPECTATOR,
    SCALAR_SPECTATOR,
)
from deuteron_wigner.light_front import spectator_preserving_recoil_coordinates


@lru_cache(maxsize=1)
def _pair():
    return solve_c53_mixed_k9_helicity_pair(
        build_c53_mixed_k9_hamiltonian(C53_MIXED_K9_EXPLORATORY_BASELINE)
    )


@lru_cache(maxsize=1)
def _grid():
    return build_nn_light_front_wave_grid(
        resolution=NN_AV18_NUCLEAR_R1,
        wave_member=build_av18_wave_member(),
        nucleon_mass_GeV=0.9389,
    )


@lru_cache(maxsize=2)
def _state(channel: str = SCALAR_SPECTATOR):
    return compose_nn_deuteron_state(_pair(), wave_grid=_grid(), channel=channel)


def _transfer(delta_x: float, channel: str = SCALAR_SPECTATOR):
    return build_nn_transfer_fibers(
        _state(channel),
        delta_x_GeV=delta_x,
        delta_y_GeV=0.0,
        p_plus_GeV=2.0,
        reference_deuteron_mass_GeV=1.87561294257,
    )


@lru_cache(maxsize=1)
def _current_pair():
    positive = evaluate_nn_one_body_diagnostic_current(
        _transfer(0.2), current_member=NN_POINTLIKE_CHARGE_CURRENT_V1
    )
    reversed_transfer = evaluate_nn_one_body_diagnostic_current(
        _transfer(-0.2), current_member=NN_POINTLIKE_CHARGE_CURRENT_V1
    )
    return positive, reversed_transfer


def test_shared_volume_iv_recoil_authority_closes_intrinsic_and_physical_maps() -> None:
    recoil = spectator_preserving_recoil_coordinates(
        y=0.37,
        p_x=0.12,
        p_y=-0.08,
        delta_x=0.20,
        delta_y=-0.06,
    )

    np.testing.assert_allclose(
        np.asarray(recoil.active_in_t) + np.asarray(recoil.spectator_in_t), 0.0
    )
    np.testing.assert_allclose(
        np.asarray(recoil.active_out_t) + np.asarray(recoil.spectator_out_t), 0.0
    )
    np.testing.assert_allclose(recoil.active_physical_transfer, recoil.delta_t)
    np.testing.assert_allclose(recoil.spectator_physical_transfer, 0.0)
    assert recoil.authority_id == NN_RECOIL_AUTHORITY_ID
    assert recoil.jacobian == 1.0


def test_transfer_fibers_are_distinct_symmetric_and_keep_both_resolutions() -> None:
    transfer = _transfer(0.2)

    assert transfer.frame.incoming.stable_id != transfer.frame.outgoing.stable_id
    assert transfer.frame.incoming.p_transverse.x == pytest.approx(-0.1)
    assert transfer.frame.outgoing.p_transverse.x == pytest.approx(0.1)
    assert transfer.frame.delta_plus == 0.0
    assert transfer.frame.xi == 0.0
    assert transfer.frame.incoming.hilbert_space_id.endswith(";K9)::NN_ONLY")
    assert (
        transfer.frame.incoming.regulator_or_pilot_id
        == NN_AV18_NUCLEAR_R1.resolution_id
    )
    assert transfer.reference_mass_role == REFERENCE_MASS_ROLE
    assert transfer.q2_GeV2 == pytest.approx(0.04)


def test_reversing_transfer_exchanges_initial_and_final_fibers() -> None:
    positive = _transfer(0.2)
    negative = _transfer(-0.2)

    assert positive.frame.incoming.p_transverse == negative.frame.outgoing.p_transverse
    assert positive.frame.outgoing.p_transverse == negative.frame.incoming.p_transverse
    assert positive.eta == pytest.approx(negative.eta)


def test_current_member_is_explicit_pointlike_charge_not_a_physical_default() -> None:
    member = NN_POINTLIKE_CHARGE_CURRENT_V1

    assert member.proton_f1 == 1.0
    assert member.neutron_f1 == 0.0
    assert member.isosinglet_f1 == 1.0
    assert member.isosinglet_f2 == 0.0
    assert member.q2_dependence == "CONSTANT_POINTLIKE_DIAGNOSTIC"
    assert member.physical is False


def test_finite_grid_one_body_target_matrix_is_reproducible() -> None:
    positive, _ = _current_pair()

    expected = np.asarray(
        [
            [0.61907812, 0.00185957, -0.04319585],
            [-0.00185957, 0.57897165, 0.00185957],
            [-0.04319585, -0.00185957, 0.61907812],
        ]
    )
    np.testing.assert_allclose(positive.target_matrix.real, expected, atol=5.0e-9)
    assert np.max(np.abs(positive.target_matrix.imag)) < 7.0e-18


def test_charge_and_full_reversal_hermiticity_close() -> None:
    positive, negative = _current_pair()
    record = nn_one_body_current_pair_record(positive, negative)

    # Forward normalization sums 5376 complex-grid contributions. Requiring
    # fewer than four ulps was a platform-specific rounding assertion, not
    # charge conservation. Keep a 64-epsilon, O(1)-charge numerical budget.
    assert record["charge_q2_zero_residual"] < 64 * np.finfo(float).eps
    assert record["reversal_hermiticity_residual"] < 2.0e-15
    assert record["active_physical_transfer_residual"] == 0.0
    assert record["spectator_physical_transfer_residual"] == 0.0


def test_parity_and_zero_plus_relations_close_in_the_x_directed_frame() -> None:
    positive, negative = _current_pair()
    record = nn_one_body_current_pair_record(positive, negative)

    assert record["parity_residual"] < 3.0e-15
    assert record["zero_plus_relation_residual"] < 2.0e-15


def test_coherent_sd_current_components_reconstruct_and_reverse_with_sd_swap() -> None:
    positive, negative = _current_pair()
    record = nn_one_body_current_pair_record(positive, negative)

    assert set(positive.component_matrices) == {"SS", "SD", "DS", "DD"}
    assert record["sd_component_reconstruction_residual"] < 2.0e-15
    assert record["component_reversal_residual"] < 2.0e-15
    assert np.max(np.abs(positive.component_matrices["SD"])) > 1.0e-4
    assert np.max(np.abs(positive.component_matrices["DS"])) > 1.0e-4


def test_nonzero_angular_condition_residual_is_reported_not_repaired() -> None:
    positive, negative = _current_pair()
    record = nn_one_body_current_pair_record(positive, negative)

    assert record["angular_condition_residual"].real == pytest.approx(
        0.00014975815808637716, abs=2.0e-14
    )
    assert record["relative_angular_condition_violation"] == pytest.approx(
        0.00012028327047759704, abs=2.0e-14
    )
    assert record["hamiltonian_consistent_current_constructed"] is False
    assert record["production_current_selected"] is False


def test_missing_current_terms_and_invalid_authority_remain_fail_closed() -> None:
    positive, negative = _current_pair()
    record = nn_one_body_current_pair_record(positive, negative)

    assert set(record["omitted_current_terms"]) == set(OMITTED_CURRENT_TERMS)
    assert record["nuclear_hamiltonian_constructed"] is False
    assert record["deuteron_mass_eigenvalue_constructed"] is False
    with pytest.raises(ValueError, match="recoil authority"):
        replace(_transfer(0.2), recoil_authority_id="LOCAL_SHIFT_REIMPLEMENTATION")
    with pytest.raises(ValueError, match="opposite transverse transfer"):
        nn_one_body_current_pair_record(positive, positive)
    with pytest.raises(ValueError, match="positive x-directed"):
        nn_one_body_current_pair_record(negative, positive)
    with pytest.raises(ValueError, match="positive"):
        build_nn_transfer_fibers(
            _state(AXIAL_SPECTATOR),
            delta_x_GeV=0.2,
            delta_y_GeV=0.0,
            p_plus_GeV=0.0,
            reference_deuteron_mass_GeV=1.87561294257,
        )

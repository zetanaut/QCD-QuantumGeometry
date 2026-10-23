from __future__ import annotations

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_total_sign_h2_hamiltonian import (
    M2M_HAMILTONIAN_STATUS,
    M2TotalSignH2Hamiltonian,
    build_total_sign_h2_hamiltonian,
    solve_total_sign_h2_helicity_pair,
    total_sign_h2_hamiltonian_record,
    zero_c53_point,
)


@pytest.fixture(scope="module")
def proton() -> M2TotalSignH2Hamiltonian:
    return build_total_sign_h2_hamiltonian("PROTON")


def test_global_ground_mass_embedding_uses_the_s3_scalar_width(
    proton: M2TotalSignH2Hamiltonian,
) -> None:
    value = proton.mass_embedding

    assert value.active_fraction == pytest.approx(1.0 / 3.0)
    assert value.b_rho_GeV == pytest.approx(np.sqrt(2.0 / 3.0) * 0.4)
    assert value.spectator_cluster_mass_sq_GeV2 == pytest.approx(0.9066666666666667)
    assert value.active_mass_scale == pytest.approx(3.0)
    assert value.additive_identity_shift_GeV2 == pytest.approx(1.52)


def test_complete_fixed_weight_carrier_enlarges_the_c53_source_rays(
    proton: M2TotalSignH2Hamiltonian,
) -> None:
    validation = proton.validation

    assert proton.dimension == 266
    assert proton.q_dimension == 2
    assert proton.qg_dimension == 264
    assert validation["jz_block_dimensions"] == {"JZ-1/2": 133, "JZ+1/2": 133}
    assert validation["all_external_qg_modes"] == 224
    assert validation["retained_c53_external_modes"] == 80
    assert validation["fixed_mode_carrier_rank_counts"] == {
        "rank_1": 56,
        "rank_2": 104,
    }
    assert validation["one_source_ray_per_mode_dimension"] == 162
    assert validation["complete_carrier_extra_dimensions"] == 104


def test_represented_fock_diagonal_operator_is_scalar_and_support_closed(
    proton: M2TotalSignH2Hamiltonian,
) -> None:
    validation = proton.validation

    assert validation["q_spin_scalar_residual"] == 0.0
    assert validation["q_qg_cross_residual"] == 0.0
    assert validation["qg_spin_scalar_residual"] == 0.0
    assert validation["retained_support_leakage_residual"] == 0.0
    assert validation["c117_qg_spin_scalar_residual"] == 0.0
    assert validation["c117_retained_support_leakage_residual"] == 0.0


def test_c53_source_is_complete_in_the_total_sign_carrier(
    proton: M2TotalSignH2Hamiltonian,
) -> None:
    validation = proton.validation

    assert validation["c53_source_projection_residual"] < 3.0e-14
    assert validation["c53_source_norm_residual"] < 3.0e-14
    assert validation["c53_emission_norm_residual"] < 3.0e-14
    assert validation["c53_opposite_helicity_residual"] == 0.0
    assert validation["c53_source_norm_squared_by_helicity"] == pytest.approx(
        (90.54871727212179, 90.54871727212179), abs=3.0e-13
    )


def test_total_matrix_is_hermitian_and_exactly_jz_block_diagonal(
    proton: M2TotalSignH2Hamiltonian,
) -> None:
    validation = proton.validation

    assert validation["fock_diagonal_hermiticity_residual"] < 3.0e-14
    assert validation["hamiltonian_hermiticity_residual"] < 3.0e-14
    assert validation["hamiltonian_jz_commutator_residual"] == 0.0
    np.testing.assert_allclose(
        proton.matrix.toarray(), proton.matrix.getH().toarray(), rtol=0.0, atol=3.0e-14
    )


def test_lowest_helicity_pair_is_degenerate_with_nonzero_qg_support(
    proton: M2TotalSignH2Hamiltonian,
) -> None:
    pair = solve_total_sign_h2_helicity_pair(proton)

    assert pair.helicity_splitting_GeV2 == pytest.approx(0.0, abs=2.0e-14)
    for state in pair.states:
        assert state.energy_GeV2 == pytest.approx(1.789591467572781, abs=2.0e-13)
        assert state.residual_norm_GeV2 < 1.0e-13
        assert state.q_weight == pytest.approx(0.943333437129513, abs=2.0e-13)
        assert state.qg_weight == pytest.approx(0.056666562870487, abs=2.0e-13)
        assert state.sector_gap_GeV2 > 1.5


def test_proton_and_neutron_are_isospin_spectral_partners(
    proton: M2TotalSignH2Hamiltonian,
) -> None:
    neutron = build_total_sign_h2_hamiltonian("NEUTRON")
    proton_pair = solve_total_sign_h2_helicity_pair(proton)
    neutron_pair = solve_total_sign_h2_helicity_pair(neutron)

    assert tuple(state.energy_GeV2 for state in neutron_pair.states) == pytest.approx(
        tuple(state.energy_GeV2 for state in proton_pair.states), abs=2.0e-14
    )
    assert tuple(state.qg_weight for state in neutron_pair.states) == pytest.approx(
        tuple(state.qg_weight for state in proton_pair.states), abs=2.0e-14
    )


def test_zero_c53_limit_returns_pure_qqq_lowest_states() -> None:
    assembly = build_total_sign_h2_hamiltonian("PROTON", zero_c53_point())
    pair = solve_total_sign_h2_helicity_pair(assembly)

    assert assembly.c53_emission.nnz == 0
    for state in pair.states:
        assert state.q_weight == pytest.approx(1.0, abs=2.0e-14)
        assert state.qg_weight == pytest.approx(0.0, abs=2.0e-14)
        assert state.energy_GeV2 == pytest.approx(2.1037591222516006, abs=2.0e-13)


def test_record_preserves_the_exact_scope_and_next_boundary() -> None:
    record = total_sign_h2_hamiltonian_record()

    assert record["status"] == M2M_HAMILTONIAN_STATUS
    assert record["basis_dimension_per_target"] == 266
    assert record["one_source_ray_per_mode_rejected_as_incomplete_carrier"] is True
    assert record["complete_fixed_weight_total_sign_carrier_used"] is True
    assert record["represented_c117_included"] is True
    assert record["complete_h2_hamiltonian_constructed"] is False
    assert record["complete_h2_missing_terms"]
    assert record["hamiltonian_gauged_current_constructed"] is False
    assert record["pennylane_implementation_ready"] is False
    assert record["production_observable_claimed"] is False
    assert record["proton_neutron_energy_residual_GeV2"] < 2.0e-14
    assert len(record["root"]) == 64


def test_invalid_target_fails_closed() -> None:
    with pytest.raises(ValueError, match="target"):
        build_total_sign_h2_hamiltonian("DEUTERON")

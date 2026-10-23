from __future__ import annotations

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_spin_flavor_s3_completion import (
    HELICITIES_TWICE,
    M2L_C53_SOURCE_STATUS,
    M2L_SPIN_FLAVOR_STATUS,
    TARGETS,
    build_c53_spin_flavor_orbit_audit,
    build_global_ground_nucleon_spin_flavor_state,
    build_spin_flavor_s3_algebra,
    build_total_sign_spin_flavor_carrier,
    spin_flavor_s3_completion_record,
)


def test_spin_flavor_s3_representation_and_total_sign_projector_close() -> None:
    algebra = build_spin_flavor_s3_algebra()

    assert len(algebra.group) == 6
    assert all(matrix.shape == (64, 64) for matrix in algebra.representations)
    assert algebra.symmetric_projector.shape == (64, 64)
    assert algebra.qqqg_total_antisymmetrizer.shape == (128, 128)
    assert algebra.qqqg_sign_rank == 20
    assert algebra.representation_residual == 0.0
    assert algebra.projector_residual < 1.0e-14


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("helicity_twice", HELICITIES_TWICE)
def test_global_ground_nucleon_is_unique_normalized_symmetric_s_and_i_half(
    target: str, helicity_twice: int
) -> None:
    state = build_global_ground_nucleon_spin_flavor_state(
        target, helicity_twice
    )

    assert state.vector.shape == (64,)
    assert state.normalization_residual < 3.0e-15
    assert state.symmetric_residual < 3.0e-15
    assert state.spin_half_residual < 3.0e-15
    assert state.isospin_half_residual < 3.0e-15
    assert state.qqq_total_sign_residual < 3.0e-15


def test_global_ground_derives_equal_scalar_axial_weights_without_mixed_pairs() -> None:
    for target in TARGETS:
        for helicity_twice in HELICITIES_TWICE:
            state = build_global_ground_nucleon_spin_flavor_state(
                target, helicity_twice
            )
            assert state.scalar_spectator_weight == pytest.approx(0.5, abs=3.0e-15)
            assert state.axial_spectator_weight == pytest.approx(0.5, abs=3.0e-15)
            assert state.mixed_spectator_weight < 3.0e-30


def test_proton_neutron_and_helicity_states_are_mutually_orthogonal() -> None:
    states = [
        build_global_ground_nucleon_spin_flavor_state(target, helicity).vector
        for target in TARGETS
        for helicity in HELICITIES_TWICE
    ]
    gram = np.column_stack(states).conj().T @ np.column_stack(states)
    np.testing.assert_allclose(gram, np.eye(4), rtol=0.0, atol=3.0e-15)


def test_c53_orbit_keeps_only_spin_traceless_source_and_is_total_antisymmetric() -> None:
    audit = build_c53_spin_flavor_orbit_audit()

    assert audit.nonzero_colorless_matrix_entries == 104
    assert audit.nonzero_spin_channels == 80
    assert audit.source_reachable_channels == 80
    assert audit.color_insertion_norm == pytest.approx(np.sqrt(4.0 / 3.0), abs=3.0e-15)
    assert audit.color_neutrality_residual < 3.0e-15
    assert audit.scalar_emission_cancellation_residual < 3.0e-14
    assert audit.traceless_spin_reconstruction_residual < 3.0e-14
    assert audit.total_sign_projection_residual < 3.0e-14


def test_c53_orbit_preserves_isospin_and_exact_total_jz() -> None:
    audit = build_c53_spin_flavor_orbit_audit()
    norm_scale = max(audit.source_norm_squared_by_state.values())
    accumulated_roundoff = 16.0 * np.finfo(float).eps * norm_scale

    assert audit.isospin_half_residual < 3.0e-14
    assert audit.total_jz_residual == 0.0
    assert audit.proton_neutron_source_norm_residual < accumulated_roundoff
    assert audit.helicity_source_norm_residual < accumulated_roundoff
    assert set(audit.source_norm_squared_by_state) == {
        "PROTON:JZ-1/2",
        "PROTON:JZ+1/2",
        "NEUTRON:JZ-1/2",
        "NEUTRON:JZ+1/2",
    }
    assert min(audit.source_norm_squared_by_state.values()) > 90.0


def test_bare_color_coordinates_are_maximally_mixed_not_independent_vertices() -> None:
    audit = build_c53_spin_flavor_orbit_audit()

    assert audit.reduced_color_isotropy_residual < 3.0e-14
    assert audit.active_color_weight_range == pytest.approx((0.5, 0.5), abs=3.0e-14)
    assert audit.orthogonal_color_weight_range == pytest.approx(
        (0.5, 0.5), abs=3.0e-14
    )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("quark_spin_twice", (-3, -1, 1, 3))
def test_fixed_weight_total_sign_carrier_is_complete_and_orthonormal(
    target: str, quark_spin_twice: int
) -> None:
    carrier = build_total_sign_spin_flavor_carrier(target, quark_spin_twice)

    assert carrier.basis.shape == (
        128,
        1 if abs(quark_spin_twice) == 3 else 2,
    )
    assert carrier.rank == carrier.basis.shape[1]
    assert carrier.projector_residual < 3.0e-14
    assert carrier.orthonormality_residual < 3.0e-14
    assert carrier.total_sign_residual < 3.0e-14
    assert carrier.isospin_half_residual < 3.0e-14
    assert carrier.spin_projection_residual < 3.0e-14


def test_record_keeps_the_global_ground_claim_boundary_explicit() -> None:
    record = spin_flavor_s3_completion_record()

    assert record["status"] == M2L_SPIN_FLAVOR_STATUS
    assert record["c53_source_status"] == M2L_C53_SOURCE_STATUS
    assert record["orbital_member"] == "S3_SCALAR_GLOBAL_JACOBI_GROUND"
    assert record["complete_noncolor_quark_antisymmetrizer_applied"] is True
    assert set(record["qqqg_fixed_target_spin_carrier_ranks"].values()) == {1, 2}
    assert record["orthogonal_qqqg_singlet_status"].startswith(
        "POPULATED_AS_REQUIRED_ENTANGLED"
    )
    assert record["independent_orthogonal_c53_coupling"] is False
    assert record["complete_h2_hamiltonian_constructed"] is False
    assert record["hamiltonian_gauged_current_constructed"] is False
    assert record["spin_flavor_symmetry_breaking_constructed"] is False
    assert record["physical_nucleon_or_deuteron_state_constructed"] is False
    assert record["production_observable_claimed"] is False

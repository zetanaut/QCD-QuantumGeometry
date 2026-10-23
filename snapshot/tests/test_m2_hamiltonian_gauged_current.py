from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from deuteron_wigner.quantum.m2_hamiltonian_gauged_current import (
    M2N_CURRENT_STATUS,
    M2N_DENSITY_STATUS,
    M2N_PROBE_ID,
    M2BaryonDensityOperator,
    M2HamiltonianGaugedBaryonCurrent,
    build_hamiltonian_gauged_baryon_current,
    build_m2_baryon_density_operator,
    hamiltonian_gauged_current_record,
)
from deuteron_wigner.quantum.m2_total_sign_h2_hamiltonian import (
    build_total_sign_h2_hamiltonian,
    zero_c53_point,
)


@pytest.fixture(scope="module")
def operators() -> tuple[
    M2BaryonDensityOperator,
    M2BaryonDensityOperator,
    M2BaryonDensityOperator,
    M2HamiltonianGaugedBaryonCurrent,
    M2HamiltonianGaugedBaryonCurrent,
]:
    assembly = build_total_sign_h2_hamiltonian("PROTON")
    positive = build_m2_baryon_density_operator(
        assembly, delta_x_GeV=0.2, delta_y_GeV=0.0
    )
    reversed_transfer = build_m2_baryon_density_operator(
        assembly, delta_x_GeV=-0.2, delta_y_GeV=0.0
    )
    forward = build_m2_baryon_density_operator(
        assembly, delta_x_GeV=0.0, delta_y_GeV=0.0
    )
    return (
        positive,
        reversed_transfer,
        forward,
        build_hamiltonian_gauged_baryon_current(positive),
        build_hamiltonian_gauged_baryon_current(reversed_transfer),
    )


def _max_abs(matrix: sparse.spmatrix) -> float:
    values = matrix.tocsr().data
    return float(np.max(np.abs(values))) if values.size else 0.0


def test_density_uses_one_third_per_quark_and_the_m2m_regulator(
    operators: tuple,
) -> None:
    positive = operators[0]

    assert positive.probe_id == M2N_PROBE_ID
    assert positive.active_lambda_overlap == pytest.approx(0.8824969025845955)
    assert positive.spectator_total_overlap == pytest.approx(0.8824969025845955)
    assert positive.validation["quark_probe_charge"] == "1/3"
    assert positive.validation["gluon_probe_charge"] == "0"
    assert positive.validation["q_to_qg_density_residual"] == 0.0
    assert positive.validation["finite_shell_projection_explicit"] is True


def test_forward_baryon_density_is_exact_identity(operators: tuple) -> None:
    forward = operators[2]
    identity = sparse.identity(266, format="csr", dtype=np.complex128)

    assert forward.validation["forward"] is True
    assert forward.validation["forward_identity_residual"] < 3.0e-14
    assert _max_abs(forward.matrix - identity) < 3.0e-14


def test_density_transfer_reversal_is_adjoint(operators: tuple) -> None:
    positive, reversed_transfer = operators[:2]

    assert _max_abs(reversed_transfer.matrix - positive.matrix.getH()) < 3.0e-14


def test_finite_hamiltonian_continuity_closes(operators: tuple) -> None:
    current = operators[3]
    density = current.density
    divergence = density.delta_x_GeV * current.current_x
    divergence += density.delta_y_GeV * current.current_y

    assert current.validation["continuity_residual_GeV2"] < 3.0e-14
    assert current.validation[
        "hamiltonian_commutator_reconstruction_residual_GeV2"
    ] < 3.0e-14
    assert _max_abs(divergence - current.continuity_numerator) < 3.0e-14


def test_c53_gauging_produces_only_the_number_changing_block(
    operators: tuple,
) -> None:
    current = operators[3]

    assert current.validation["c53_current_fock_diagonal_residual_GeV"] == 0.0
    assert current.validation["fock_current_number_changing_residual_GeV"] == 0.0
    assert current.validation["c53_number_changing_current_max_abs_GeV"] == pytest.approx(
        0.31026505994186515, abs=3.0e-13
    )


def test_current_decomposes_into_fock_and_c53_ancestry(operators: tuple) -> None:
    current = operators[3]

    assert current.validation["current_decomposition_residual_GeV"] == 0.0
    assert _max_abs(
        current.current_x
        - current.fock_diagonal_current_x
        - current.c53_number_changing_current_x
    ) == 0.0


def test_current_transfer_reversal_is_adjoint(operators: tuple) -> None:
    positive, reversed_transfer = operators[3:]

    assert _max_abs(reversed_transfer.current_x - positive.current_x.getH()) < 1.0e-13
    assert _max_abs(reversed_transfer.current_y - positive.current_y.getH()) < 1.0e-13


def test_record_keeps_the_abelian_benchmark_boundary_explicit(
    operators: tuple,
) -> None:
    record = hamiltonian_gauged_current_record(
        operators[3], operators[4], operators[2]
    )

    assert record["status"] == M2N_CURRENT_STATUS
    assert record["density_status"] == M2N_DENSITY_STATUS
    assert record["probe_id"] == M2N_PROBE_ID
    assert record["number_changing_current_constructed"] is True
    assert record["number_changing_current_ancestry"] == "M2M_C53_BLOCK"
    assert record["finite_hamiltonian_continuity_closed"] is True
    assert record["divergence_free_transverse_current_fixed"] is False
    assert record["physical_electromagnetic_current_constructed"] is False
    assert record["complete_hamiltonian_gauged_current_constructed"] is False
    assert record["unavailable_current_terms"]
    assert record["pennylane_observable_contract_ready"] is True
    assert record["production_observable_claimed"] is False
    assert len(record["root"]) == 64


def test_forward_current_requires_a_limit_not_division_by_zero(operators: tuple) -> None:
    with pytest.raises(ValueError, match="nonzero transverse transfer"):
        build_hamiltonian_gauged_baryon_current(operators[2])


@pytest.mark.parametrize("coupling", [0.0, 1.0e-14, -1.0e-14])
def test_current_accepts_zero_and_small_c53_couplings(
    operators: tuple, coupling: float
) -> None:
    reference = operators[3]
    point = zero_c53_point(reference.density.assembly.point)
    if coupling != 0.0:
        point = replace(point, c53_coupling=coupling)
    assembly = build_total_sign_h2_hamiltonian("PROTON", point)
    density = build_m2_baryon_density_operator(
        assembly, delta_x_GeV=0.2, delta_y_GeV=0.0
    )
    current = build_hamiltonian_gauged_baryon_current(density)

    # The regulator and density do not depend on the C53 coefficient.
    assert _max_abs(density.matrix - reference.density.matrix) == 0.0
    for component in ("x", "y"):
        block = getattr(current, f"c53_number_changing_current_{component}")
        reference_block = getattr(reference, f"c53_number_changing_current_{component}")
        if coupling == 0.0:
            assert _max_abs(block) == 0.0
        else:
            # Compare after rescaling: an absolute tolerance could accept a
            # wrongly zeroed tiny-coupling current.
            np.testing.assert_allclose(
                (block / coupling).toarray(),
                (reference_block / reference.density.assembly.point.c53_coupling).toarray(),
                rtol=2.0e-13, atol=2.0e-13,
            )
        assert _max_abs(
            getattr(current, f"fock_diagonal_current_{component}")
            - getattr(reference, f"fock_diagonal_current_{component}")
        ) == 0.0
    assert current.validation["continuity_residual_GeV2"] < 1.0e-13
    assert current.validation["hamiltonian_commutator_reconstruction_residual_GeV2"] < 1.0e-13
    assert current.validation["current_decomposition_residual_GeV"] < 1.0e-13
    assert current.validation["fock_current_number_changing_residual_GeV"] == 0.0
    assert current.validation["c53_current_fock_diagonal_residual_GeV"] == 0.0


def test_oblique_transfer_exercises_both_current_components(operators: tuple) -> None:
    assembly = operators[0].assembly
    dx, dy = 0.12, 0.16
    density = build_m2_baryon_density_operator(
        assembly, delta_x_GeV=dx, delta_y_GeV=dy
    )
    current = build_hamiltonian_gauged_baryon_current(density)
    reverse = build_hamiltonian_gauged_baryon_current(
        build_m2_baryon_density_operator(
            assembly, delta_x_GeV=-dx, delta_y_GeV=-dy
        )
    )
    rng = np.random.default_rng(205)
    probe = rng.normal(size=assembly.dimension) + 1j * rng.normal(size=assembly.dimension)
    probe /= np.linalg.norm(probe)
    independent_action = assembly.matrix @ (density.matrix @ probe) - density.matrix @ (
        assembly.matrix @ probe
    )
    assert np.linalg.norm(independent_action) > 1.0e-3
    for delta, matrix, reversed_matrix in (
        (dx, current.current_x, reverse.current_x),
        (dy, current.current_y, reverse.current_y),
    ):
        assert np.linalg.norm(matrix @ probe) > 1.0e-3
        np.testing.assert_allclose(
            matrix @ probe, delta / (dx**2 + dy**2) * independent_action,
            rtol=2.0e-12, atol=2.0e-13,
        )
        assert _max_abs(reversed_matrix - matrix.getH()) < 1.0e-13
    assert current.validation["continuity_residual_GeV2"] < 1.0e-13


def test_eigenstate_current_zero_does_not_replace_transition_check(operators: tuple) -> None:
    current = operators[3]
    energies, states = np.linalg.eigh(current.density.assembly.matrix.toarray())
    rho = states.conj().T @ (current.density.matrix @ states)
    expected = (energies[:, None] - energies[None, :]) * rho / 0.2
    actual = states.conj().T @ (current.current_x @ states)

    assert np.max(np.abs(actual)) > 1.0e-3
    np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=3.0e-13)
    assert np.max(np.abs(np.diag(actual))) < 3.0e-13

from __future__ import annotations

from functools import lru_cache
import json

import numpy as np
import pytest
from scipy import sparse

from deuteron_wigner.quantum.m2_active_quark_current import (
    build_m2_active_quark_impulse_operator,
)
from deuteron_wigner.quantum.m2_c53_mixed_singlet_state import (
    C53_MIXED_K9_EXPLORATORY_BASELINE,
    build_c53_mixed_k9_hamiltonian,
    solve_c53_mixed_k9_helicity_pair,
)
from deuteron_wigner.quantum.m2_permutation_complete_h2 import (
    M2J_ACTIVE_CURRENT_STATUS,
    M2J_BASIS_STATUS,
    M2J_HAMILTONIAN_STATUS,
    M2J_TARGET_RESOLUTION_ID,
    UNAVAILABLE_M2J_CURRENT_TERMS,
    UNAVAILABLE_M2J_HAMILTONIAN_TERMS,
    build_m2_to_permutation_complete_h2_embedding,
    build_m2j_permutation_complete_active_current,
    build_m2j_permutation_complete_hamiltonian,
    build_permutation_complete_h2_basis,
    build_quark_permutation_algebra,
    lift_active_emitter_operator,
    lift_color_scalar_fock_diagonal_operator,
    m2j_active_current_pair_record,
    m2j_embedding_record,
    permutation_algebra_record,
)
from deuteron_wigner.quantum.m2_resolved_spectator_current import (
    build_resolved_spectator_hamiltonian,
)


def _max_abs(matrix: sparse.spmatrix) -> float:
    data = matrix.tocsr().data
    return float(np.max(np.abs(data))) if data.size else 0.0


@lru_cache(maxsize=1)
def _objects():
    assembly = build_c53_mixed_k9_hamiltonian(
        C53_MIXED_K9_EXPLORATORY_BASELINE
    )
    pair = solve_c53_mixed_k9_helicity_pair(assembly)
    source = build_resolved_spectator_hamiltonian(pair)
    basis = build_permutation_complete_h2_basis(source.resolution)
    embedding = build_m2_to_permutation_complete_h2_embedding(basis)
    hamiltonian = build_m2j_permutation_complete_hamiltonian(
        source, embedding=embedding
    )
    return source, basis, embedding, hamiltonian


@lru_cache(maxsize=1)
def _currents():
    embedding = _objects()[2]
    result = []
    for delta_x in (0.2, -0.2, 0.0):
        source = build_m2_active_quark_impulse_operator(
            "K9",
            delta_x_GeV=delta_x,
            delta_y_GeV=0.0,
            quadrature_nodes=18,
        )
        result.append(
            build_m2j_permutation_complete_active_current(source, embedding)
        )
    return tuple(result)


def test_s3_sign_projectors_retain_one_qqq_and_two_qqqg_multiplicities() -> None:
    algebra = build_quark_permutation_algebra()

    assert algebra.qqq_multiplicity == 1
    assert algebra.qqqg_multiplicity == 2
    assert algebra.emitter_span_rank == 2
    assert algebra.representation_residual < 1.0e-14
    assert algebra.projector_residual < 1.0e-14
    assert algebra.exchange_eigen_residual < 1.0e-14
    assert algebra.emitter_sum_residual < 1.0e-14


def test_active_emitter_and_retained_orthogonal_singlet_are_exactly_resolved() -> None:
    algebra = build_quark_permutation_algebra()
    active = algebra.active_emitter_coordinates
    orthogonal = algebra.orthogonal_emitter_coordinates

    assert np.linalg.norm(active) == pytest.approx(1.0)
    assert np.linalg.norm(orthogonal) == pytest.approx(1.0)
    assert abs(np.vdot(active, orthogonal)) < 1.0e-14
    assert min(np.abs(active)) > 0.49
    assert sorted(np.abs(active)) == pytest.approx([0.5, np.sqrt(3.0) / 2.0])


def test_permutation_algebra_record_is_deterministic_and_nonphysical() -> None:
    first = permutation_algebra_record()
    second = permutation_algebra_record()

    assert first == second
    assert first["status"] == M2J_BASIS_STATUS
    assert first["physical"] is False
    assert first["qqq_sign_multiplicity"] == 1
    assert first["qqqg_sign_multiplicity"] == 2
    assert first["orthogonal_singlet_status"] == (
        "RETAINED_UNSOURCED_NOT_PHYSICAL_ZERO"
    )
    json.dumps(first, sort_keys=True)


def test_target_basis_has_exact_qqq_plus_qqqg_dimensions_and_multiplicities() -> None:
    _, basis, _, _ = _objects()

    assert basis.resolution_id == M2J_TARGET_RESOLUTION_ID
    assert basis.source_dimension == 450
    assert basis.dimension == 898
    assert basis.qqq_dimension == 2
    assert basis.qqqg_source_dimension == 448
    assert basis.qqqg_dimension == 896
    assert all(len(rows) == 1 for rows in basis.source_to_target_rows[:2])
    assert all(len(rows) == 2 for rows in basis.source_to_target_rows[2:])
    assert all(
        {basis.states[row].color_multiplicity for row in rows} == {0, 1}
        for rows in basis.source_to_target_rows[2:]
    )


def test_target_longitudinal_modes_close_exactly_in_three_quark_slot_order() -> None:
    _, basis, _, _ = _objects()

    assert all(sum(state.longitudinal_K2) == 27 for state in basis.states)
    assert {
        state.longitudinal_K2 for state in basis.states if state.sector == "qqq"
    } == {(9, 9, 9)}
    qqqg_modes = {
        state.longitudinal_K2
        for state in basis.states
        if state.sector == "qqqg"
    }
    assert len(qqqg_modes) == 4
    assert all(
        modes[0] % 2 == modes[1] % 2 == modes[2] % 2 == 1
        and modes[3] % 2 == 0
        for modes in qqqg_modes
    )
    assert all(
        state.transverse_modes[-2:] == ((0, 0), (0, 0))
        for state in basis.states
    )


def test_source_embedding_is_exact_isometry_and_retains_orthogonal_channel() -> None:
    _, basis, embedding, _ = _objects()
    identity = sparse.identity(basis.source_dimension, format="csr")
    source_projector = embedding.source_projector
    orthogonal = embedding.orthogonal_qqqg_projector

    assert embedding.matrix.shape == (898, 450)
    assert embedding.matrix.nnz == 898
    assert _max_abs(embedding.matrix.getH() @ embedding.matrix - identity) < 1.0e-14
    assert source_projector.diagonal().sum().real == pytest.approx(450.0)
    assert orthogonal.diagonal().sum().real == pytest.approx(448.0)
    assert _max_abs(source_projector @ orthogonal) < 1.0e-14
    assert _max_abs(orthogonal @ source_projector) < 1.0e-14
    assert _max_abs(source_projector + orthogonal - sparse.eye(898)) < 1.0e-14


def test_color_scalar_lift_copies_qg_operator_to_both_multiplicities() -> None:
    _, basis, embedding, _ = _objects()
    diagonal = np.arange(1, basis.source_dimension + 1, dtype=float)
    source = sparse.diags(diagonal, format="csr")
    target = lift_color_scalar_fock_diagonal_operator(source, embedding)
    qg = target[2:, 2:]

    assert target.shape == (898, 898)
    assert _max_abs(qg[0::2, 0::2] - source[2:, 2:]) == 0.0
    assert _max_abs(qg[1::2, 1::2] - source[2:, 2:]) == 0.0
    assert _max_abs(qg[0::2, 1::2]) == 0.0
    assert _max_abs(qg[1::2, 0::2]) == 0.0
    assert _max_abs(embedding.matrix.getH() @ target @ embedding.matrix - source) < 2.0e-13


def test_color_scalar_lift_rejects_number_changing_source_operator() -> None:
    _, basis, embedding, _ = _objects()
    source = sparse.csr_matrix(
        ([1.0], ([0], [2])), shape=(basis.source_dimension,) * 2
    )

    with pytest.raises(ValueError, match="Fock-diagonal"):
        lift_color_scalar_fock_diagonal_operator(source, embedding)


def test_active_emitter_lift_has_no_action_on_retained_orthogonal_singlet() -> None:
    _, basis, embedding, _ = _objects()
    source = sparse.identity(basis.source_dimension, format="csr")
    target = lift_active_emitter_operator(source, embedding)
    orthogonal = embedding.orthogonal_qqqg_projector

    assert _max_abs(target - embedding.source_projector) < 1.0e-14
    assert _max_abs(target @ orthogonal) < 1.0e-14
    assert _max_abs(orthogonal @ target) < 1.0e-14


def test_m2j_hamiltonian_exactly_reconstructs_and_pulls_back_m2i() -> None:
    source, _, embedding, hamiltonian = _objects()
    pullback = embedding.matrix.getH() @ hamiltonian.matrix @ embedding.matrix

    assert hamiltonian.dimension == 898
    assert hamiltonian.matrix.nnz > source.matrix.nnz
    assert _max_abs(hamiltonian.matrix - hamiltonian.matrix.getH()) < 1.0e-13
    assert _max_abs(pullback - source.matrix) < 4.0e-12
    assert max(hamiltonian.validation.values()) < 4.0e-12
    assert set(hamiltonian.universal_term_ids) == {
        "M2_K9_H0",
        "M2I_NESTED_JACOBI_IDENTITY_SHIFT",
        "c396:D_mu_q_sq",
        "c396:D_delta_mu_g_sq",
    }
    assert set(hamiltonian.represented_term_ids) == {
        "c117:I2_density_projector",
        "c53:g_s_canonical_q_qg",
    }


def test_m2j_interaction_is_nonzero_but_does_not_invent_orthogonal_action() -> None:
    _, _, embedding, hamiltonian = _objects()
    represented = hamiltonian.represented_interaction
    orthogonal = embedding.orthogonal_qqqg_projector

    assert represented[2:, :2].nnz > 0
    assert _max_abs(represented[2:, :2]) > 0.0
    assert _max_abs(represented @ orthogonal) < 1.0e-13
    assert _max_abs(orthogonal @ represented) < 1.0e-13
    assert _max_abs(hamiltonian.universal_operator @ orthogonal) > 1.0


def test_m2j_embedding_record_preserves_all_scientific_nonclaims() -> None:
    _, _, embedding, hamiltonian = _objects()
    record = m2j_embedding_record(embedding, hamiltonian)

    assert record["status"] == M2J_BASIS_STATUS
    assert record["hamiltonian_status"] == M2J_HAMILTONIAN_STATUS
    assert record["source_dimension"] == 450
    assert record["target_dimension"] == 898
    assert record["orthogonal_qqqg_projector_rank"] == 448
    assert record["synthetic_c9_h2_hamiltonian_consumed"] is False
    assert record["physical_state_constructed"] is False
    assert record["complete_h2_hamiltonian_constructed"] is False
    assert tuple(record["unavailable_hamiltonian_terms"]) == (
        UNAVAILABLE_M2J_HAMILTONIAN_TERMS
    )
    json.dumps(record, sort_keys=True)


def test_color_scalar_current_acts_on_complete_target_and_recovers_source() -> None:
    positive, _, forward = _currents()
    embedding = positive.embedding
    source = positive.source_operator.matrix
    pullback = embedding.matrix.getH() @ positive.matrix @ embedding.matrix

    assert positive.matrix.shape == (898, 898)
    assert _max_abs(pullback - source) < 1.0e-13
    assert _max_abs(forward.matrix - sparse.eye(898)) < 1.0e-13
    qg = positive.matrix[2:, 2:]
    assert _max_abs(qg[0::2, 0::2] - source[2:, 2:]) < 1.0e-14
    assert _max_abs(qg[1::2, 1::2] - source[2:, 2:]) < 1.0e-14


def test_m2j_current_pair_closes_charge_reversal_and_scope_ledger() -> None:
    positive, reversed_transfer, forward = _currents()
    record = m2j_active_current_pair_record(
        positive, reversed_transfer, forward
    )

    assert record["status"] == M2J_ACTIVE_CURRENT_STATUS
    assert record["target_dimension"] == 898
    assert record["acts_on_both_qqqg_color_multiplicities"] is True
    assert record["source_pullback_max_abs"] < 1.0e-13
    assert record["forward_charge_residual"] < 1.0e-13
    assert record["reversal_adjoint_residual"] < 1.0e-12
    assert record["number_changing_current_constructed"] is False
    assert record["complete_three_quark_current_constructed"] is False
    assert tuple(record["unavailable_current_terms"]) == UNAVAILABLE_M2J_CURRENT_TERMS
    json.dumps(record, sort_keys=True)


def test_wrong_m2j_inputs_fail_closed() -> None:
    _, basis, embedding, _ = _objects()

    with pytest.raises(TypeError):
        build_permutation_complete_h2_basis("K9")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_m2_to_permutation_complete_h2_embedding(basis.states)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        lift_active_emitter_operator(sparse.eye(3), embedding)
    with pytest.raises(TypeError):
        build_m2j_permutation_complete_hamiltonian(basis)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_m2j_permutation_complete_active_current(
            embedding, embedding  # type: ignore[arg-type]
        )

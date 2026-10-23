from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from deuteron_wigner.quantum.m2_hamiltonian_gauged_current import (
    build_hamiltonian_gauged_baryon_current,
    build_m2_baryon_density_operator,
)
from deuteron_wigner.quantum.m2_pennylane_contract import (
    M2O_PENNYLANE_CONTRACT_STATUS,
    M2CompactQubitBasis,
    M2PennyLaneContract,
    build_full_qubit_basis,
    build_jz_block_qubit_basis,
    build_m2_pennylane_contract,
    compact_to_padded_state,
    decode_bitstring,
    encode_compact_index,
    m2_pennylane_contract_record,
    padded_to_compact_state,
    padding_projector,
    physical_projector,
)
from deuteron_wigner.quantum.m2_total_sign_h2_hamiltonian import (
    build_total_sign_h2_hamiltonian,
)


@pytest.fixture(scope="module")
def contract() -> M2PennyLaneContract:
    assembly = build_total_sign_h2_hamiltonian("PROTON")
    positive_density = build_m2_baryon_density_operator(
        assembly, delta_x_GeV=0.2, delta_y_GeV=0.0
    )
    reversed_density = build_m2_baryon_density_operator(
        assembly, delta_x_GeV=-0.2, delta_y_GeV=0.0
    )
    return build_m2_pennylane_contract(
        assembly,
        build_hamiltonian_gauged_baryon_current(positive_density),
        build_hamiltonian_gauged_baryon_current(reversed_density),
    )


def _max_abs(matrix: sparse.spmatrix) -> float:
    values = matrix.tocsr().data
    return float(np.max(np.abs(values))) if values.size else 0.0


def test_full_and_jz_block_resources_are_minimal_power_of_two(contract: M2PennyLaneContract) -> None:
    full = contract.full_basis

    assert (full.compact_dimension, full.padded_dimension, full.qubits) == (266, 512, 9)
    for helicity in (-1, 1):
        block = contract.jz_block_bases[helicity]
        assert (block.compact_dimension, block.padded_dimension, block.qubits) == (
            133,
            256,
            8,
        )


def test_big_endian_index_codec_round_trips_and_rejects_padding(
    contract: M2PennyLaneContract,
) -> None:
    basis = contract.full_basis

    for index in (0, 1, 132, 265):
        bitstring = encode_compact_index(basis, index)
        assert len(bitstring) == 9
        assert decode_bitstring(basis, bitstring) == index
    with pytest.raises(ValueError, match="padded leakage"):
        decode_bitstring(basis, format(266, "09b"))


def test_state_padding_round_trip_and_leakage_guard(contract: M2PennyLaneContract) -> None:
    basis = contract.full_basis
    state = contract.helicity_pair.state(-1).state
    padded = compact_to_padded_state(basis, state)

    np.testing.assert_array_equal(padded_to_compact_state(basis, padded), state)
    assert np.vdot(padded[266:], padded[266:]).real == 0.0
    padded[300] = 1.0e-4
    with pytest.raises(ValueError, match="padded leakage"):
        padded_to_compact_state(basis, padded)


def test_padding_penalty_preserves_physical_action_and_excludes_false_ground(
    contract: M2PennyLaneContract,
) -> None:
    physical = physical_projector(contract.full_basis)
    padding = padding_projector(contract.full_basis)

    assert _max_abs(physical @ padding) == 0.0
    assert contract.padding_penalty_GeV2 > contract.validation[
        "hamiltonian_gershgorin_upper_GeV2"
    ]
    assert contract.validation["physical_hamiltonian_preservation_residual_GeV2"] == 0.0
    assert contract.validation["physical_ground_below_padding_penalty"] is True
    assert _max_abs(
        physical @ contract.penalized_hamiltonian
        - contract.penalized_hamiltonian @ physical
    ) == 0.0


def test_encoded_ground_states_retain_the_compact_eigenpairs(
    contract: M2PennyLaneContract,
) -> None:
    assert contract.validation["padded_state_leakage"] == 0.0
    assert contract.validation["state_round_trip_residual"] == 0.0
    assert contract.validation["encoded_ground_state_residual_GeV2"] < 1.0e-13
    assert contract.validation["compact_padded_hamiltonian_action_residual_GeV2"] == 0.0


def test_reversal_pairs_are_exported_as_hermitian_quadratures(
    contract: M2PennyLaneContract,
) -> None:
    assert set(contract.hermitian_observables) == {
        "density_cosine",
        "density_sine",
        "current_x_cosine",
        "current_x_sine",
        "current_y_cosine",
        "current_y_sine",
    }
    assert contract.validation["all_observables_hermitian"] is True
    assert contract.validation[
        "quadrature_hermiticity_and_reconstruction_residual"
    ] < 1.0e-13
    assert all(
        _max_abs(matrix - matrix.getH()) < 1.0e-13
        for matrix in contract.hermitian_observables.values()
    )


def test_sparse_observable_action_is_identical_before_and_after_padding(
    contract: M2PennyLaneContract,
) -> None:
    assert contract.validation["compact_padded_observable_action_residual"] < 1.0e-13
    assert contract.validation["generic_pauli_decomposition_required"] is False
    assert contract.validation["sparse_hamiltonian_route"] == "qml.SparseHamiltonian"
    assert contract.validation["exact_oracle_state_route"] == "qml.StatePrep"


def test_record_declares_ready_without_selecting_algorithm_or_activation(
    contract: M2PennyLaneContract,
) -> None:
    record = m2_pennylane_contract_record(contract)

    assert record["status"] == M2O_PENNYLANE_CONTRACT_STATUS
    assert record["pennylane_implementation_ready"] is True
    assert record["pennylane_execution_performed"] is False
    assert record["variational_algorithm_selected"] is False
    assert record["physical_hamiltonian_complete"] is False
    assert record["physical_current_complete"] is False
    assert record["physical_activation"] is False
    assert len(record["hermitian_observables"]) == 6
    assert len(record["root"]) == 64


def test_basis_helpers_reject_invalid_dimensions_and_helicity(
    contract: M2PennyLaneContract,
) -> None:
    assembly = contract.assembly
    assert isinstance(build_full_qubit_basis(assembly), M2CompactQubitBasis)
    with pytest.raises(ValueError, match="helicity_twice"):
        build_jz_block_qubit_basis(assembly, 3)
    with pytest.raises(IndexError, match="outside"):
        encode_compact_index(contract.full_basis, 266)

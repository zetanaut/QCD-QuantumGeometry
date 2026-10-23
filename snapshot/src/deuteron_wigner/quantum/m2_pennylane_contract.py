"""Sparse-to-qubit implementation contract for the M2m/M2n system.

The complete per-target M2m carrier has 266 states and therefore embeds in
nine qubits.  Each exact Jz block has 133 states and embeds in eight qubits.
Compact index order is preserved verbatim and encoded as a big-endian
computational bitstring.

Zero padding is sufficient for exact StatePrep expectations, but is unsafe
for variational ground-state searches because the physical exploratory
Hamiltonian has positive eigenvalues.  This contract therefore supplies a
separate padding-penalized Hamiltonian whose penalty lies above a certified
Gershgorin upper bound.  It leaves every physical matrix element unchanged.

Finite-transfer density and current operators are reversal-paired rather
than Hermitian at one directed transfer.  Their cosine and sine quadratures
are Hermitian sparse observables, and reconstruct the directed operator
exactly.  No generic Pauli expansion, ansatz, optimizer, hardware claim, or
physical activation is selected here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import log2
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

from .m2_hamiltonian_gauged_current import M2HamiltonianGaugedBaryonCurrent
from .m2_spin_flavor_s3_completion import HELICITIES_TWICE
from .m2_total_sign_h2_hamiltonian import (
    M2TotalSignH2Hamiltonian,
    TotalSignH2HelicityPair,
    solve_total_sign_h2_helicity_pair,
)


M2O_PENNYLANE_CONTRACT_STATUS = "M2O_PENNYLANE_IMPLEMENTATION_CONTRACT_READY"
M2O_CLAIM_TIER = "EXACT_SPARSE_QUBIT_ENCODING_CONTRACT_NONPHYSICAL"
M2O_ENCODING_ID = "M2O_DIRECT_COMPACT_BIG_ENDIAN_V1"

_TOLERANCE = 1.0e-10


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _max_abs(matrix: sparse.spmatrix | np.ndarray) -> float:
    if sparse.issparse(matrix):
        values = matrix.tocsr().data
        return float(np.max(np.abs(values))) if values.size else 0.0
    values = np.asarray(matrix)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _sparse_root(matrix: sparse.csr_matrix) -> str:
    value = matrix.tocsr()
    digest = sha256()
    for array in (value.indptr, value.indices, value.data):
        contiguous = np.ascontiguousarray(array)
        if np.issubdtype(contiguous.dtype, np.inexact):
            contiguous = np.round(contiguous, 14)
            contiguous[contiguous == 0] = 0
        digest.update(contiguous.dtype.str.encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(_plain(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, complex):
        return (float(value.real), float(value.imag))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    return value


def _json_root(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class M2CompactQubitBasis:
    encoding_id: str
    compact_dimension: int
    padded_dimension: int
    qubits: int
    q_dimension: int
    qg_dimension: int
    compact_indices: tuple[int, ...]
    basis_order: str = "qqq helicities followed by qqqg total-sign carriers"
    endianness: str = "big-endian; wire 0 is the leftmost bit"


def build_full_qubit_basis(
    assembly: M2TotalSignH2Hamiltonian,
) -> M2CompactQubitBasis:
    if not isinstance(assembly, M2TotalSignH2Hamiltonian):
        raise TypeError("assembly must be an M2TotalSignH2Hamiltonian")
    padded = _next_power_of_two(assembly.dimension)
    return M2CompactQubitBasis(
        encoding_id=M2O_ENCODING_ID,
        compact_dimension=assembly.dimension,
        padded_dimension=padded,
        qubits=int(log2(padded)),
        q_dimension=assembly.q_dimension,
        qg_dimension=assembly.qg_dimension,
        compact_indices=tuple(range(assembly.dimension)),
    )


def build_jz_block_qubit_basis(
    assembly: M2TotalSignH2Hamiltonian, helicity_twice: int
) -> M2CompactQubitBasis:
    if helicity_twice not in HELICITIES_TWICE:
        raise ValueError("helicity_twice must be -1 or 1")
    indices = tuple(
        state.index
        for state in assembly.basis_states
        if state.helicity_twice == helicity_twice
    )
    padded = _next_power_of_two(len(indices))
    return M2CompactQubitBasis(
        encoding_id=f"{M2O_ENCODING_ID}:JZ{helicity_twice:+d}/2",
        compact_dimension=len(indices),
        padded_dimension=padded,
        qubits=int(log2(padded)),
        q_dimension=1,
        qg_dimension=len(indices) - 1,
        compact_indices=indices,
        basis_order=f"Jz={helicity_twice:+d}/2 subsequence of the full compact basis",
    )


def encode_compact_index(basis: M2CompactQubitBasis, compact_index: int) -> str:
    if isinstance(compact_index, bool) or not isinstance(compact_index, (int, np.integer)):
        raise TypeError("compact_index must be an integer")
    index = int(compact_index)
    if not 0 <= index < basis.compact_dimension:
        raise IndexError("compact_index is outside the physical compact basis")
    return format(index, f"0{basis.qubits}b")


def decode_bitstring(basis: M2CompactQubitBasis, bitstring: str) -> int:
    if (
        not isinstance(bitstring, str)
        or len(bitstring) != basis.qubits
        or set(bitstring) - {"0", "1"}
    ):
        raise ValueError(f"bitstring must contain exactly {basis.qubits} bits")
    index = int(bitstring, 2)
    if index >= basis.compact_dimension:
        raise ValueError("bitstring denotes a padded leakage state")
    return index


def compact_to_padded_state(
    basis: M2CompactQubitBasis, compact_state: Sequence[Any]
) -> np.ndarray:
    values = np.asarray(compact_state, dtype=np.complex128)
    if values.shape != (basis.compact_dimension,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"compact_state must be finite with shape ({basis.compact_dimension},)"
        )
    padded = np.zeros(basis.padded_dimension, dtype=np.complex128)
    padded[: basis.compact_dimension] = values
    return padded


def padded_to_compact_state(
    basis: M2CompactQubitBasis,
    padded_state: Sequence[Any],
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    values = np.asarray(padded_state, dtype=np.complex128)
    if values.shape != (basis.padded_dimension,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"padded_state must be finite with shape ({basis.padded_dimension},)"
        )
    leakage = float(
        np.vdot(
            values[basis.compact_dimension :],
            values[basis.compact_dimension :],
        ).real
    )
    if leakage > tolerance:
        raise ValueError(f"state has padded leakage {leakage:.3e}")
    return values[: basis.compact_dimension].copy()


def pad_sparse_operator(
    basis: M2CompactQubitBasis, compact_operator: sparse.spmatrix
) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(compact_operator, dtype=np.complex128)
    if matrix.shape != (basis.compact_dimension,) * 2:
        raise ValueError("compact operator has the wrong shape")
    tail = basis.padded_dimension - basis.compact_dimension
    result = sparse.block_diag(
        (matrix, sparse.csr_matrix((tail, tail), dtype=np.complex128)),
        format="csr",
    )
    result.eliminate_zeros()
    result.sort_indices()
    return result


def physical_projector(basis: M2CompactQubitBasis) -> sparse.csr_matrix:
    return sparse.diags(
        np.r_[
            np.ones(basis.compact_dimension),
            np.zeros(basis.padded_dimension - basis.compact_dimension),
        ],
        format="csr",
        dtype=np.complex128,
    )


def padding_projector(basis: M2CompactQubitBasis) -> sparse.csr_matrix:
    return sparse.identity(
        basis.padded_dimension, format="csr", dtype=np.complex128
    ) - physical_projector(basis)


def _gershgorin_bounds(matrix: sparse.csr_matrix) -> tuple[float, float]:
    diagonal = np.real(matrix.diagonal())
    row_sum = np.asarray(np.abs(matrix).sum(axis=1)).reshape(-1)
    radii = row_sum - np.abs(matrix.diagonal())
    return float(np.min(diagonal - radii)), float(np.max(diagonal + radii))


def _hermitian_quadratures(
    positive: sparse.csr_matrix, reversed_transfer: sparse.csr_matrix
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, float]:
    cosine = (0.5 * (positive + reversed_transfer)).tocsr()
    sine = ((positive - reversed_transfer) / (2.0j)).tocsr()
    reconstruction = (cosine + 1.0j * sine).tocsr()
    residual = max(
        _max_abs(cosine - cosine.getH()),
        _max_abs(sine - sine.getH()),
        _max_abs(reconstruction - positive),
    )
    return cosine, sine, residual


@dataclass(frozen=True)
class M2PennyLaneContract:
    assembly: M2TotalSignH2Hamiltonian
    helicity_pair: TotalSignH2HelicityPair
    positive_current: M2HamiltonianGaugedBaryonCurrent
    reversed_current: M2HamiltonianGaugedBaryonCurrent
    full_basis: M2CompactQubitBasis
    jz_block_bases: Mapping[int, M2CompactQubitBasis]
    padded_hamiltonian: sparse.csr_matrix
    penalized_hamiltonian: sparse.csr_matrix
    padding_penalty_GeV2: float
    hermitian_observables: Mapping[str, sparse.csr_matrix]
    padded_ground_states: Mapping[int, np.ndarray]
    validation: Mapping[str, Any]
    claim_tier: str = M2O_CLAIM_TIER
    physical: bool = False


def build_m2_pennylane_contract(
    assembly: M2TotalSignH2Hamiltonian,
    positive_current: M2HamiltonianGaugedBaryonCurrent,
    reversed_current: M2HamiltonianGaugedBaryonCurrent,
) -> M2PennyLaneContract:
    """Freeze every mathematical input required by a PennyLane adapter."""

    if not isinstance(assembly, M2TotalSignH2Hamiltonian):
        raise TypeError("assembly must be an M2TotalSignH2Hamiltonian")
    if not all(
        isinstance(item, M2HamiltonianGaugedBaryonCurrent)
        for item in (positive_current, reversed_current)
    ):
        raise TypeError("current inputs have the wrong type")
    if (
        positive_current.density.assembly is not assembly
        or reversed_current.density.assembly is not assembly
    ):
        raise ValueError("Hamiltonian and currents must share one assembly")
    if (
        reversed_current.density.delta_x_GeV
        != -positive_current.density.delta_x_GeV
        or reversed_current.density.delta_y_GeV
        != -positive_current.density.delta_y_GeV
    ):
        raise ValueError("current inputs must be an exact transfer-reversal pair")

    full_basis = build_full_qubit_basis(assembly)
    block_bases = {
        helicity: build_jz_block_qubit_basis(assembly, helicity)
        for helicity in HELICITIES_TWICE
    }
    padded_hamiltonian = pad_sparse_operator(full_basis, assembly.matrix)
    lower_bound, upper_bound = _gershgorin_bounds(assembly.matrix)
    span = max(1.0, upper_bound - lower_bound)
    penalty = max(1.0, upper_bound + span)
    penalized_hamiltonian = (
        padded_hamiltonian + penalty * padding_projector(full_basis)
    ).tocsr()

    compact_pairs = {
        "density": (
            positive_current.density.matrix,
            reversed_current.density.matrix,
        ),
        "current_x": (
            positive_current.current_x,
            reversed_current.current_x,
        ),
        "current_y": (
            positive_current.current_y,
            reversed_current.current_y,
        ),
    }
    compact_observables: dict[str, sparse.csr_matrix] = {}
    quadrature_residual = 0.0
    for name, (positive, reversed_value) in compact_pairs.items():
        cosine, sine, residual = _hermitian_quadratures(
            positive, reversed_value
        )
        compact_observables[f"{name}_cosine"] = cosine
        compact_observables[f"{name}_sine"] = sine
        quadrature_residual = max(quadrature_residual, residual)
    padded_observables = {
        name: pad_sparse_operator(full_basis, matrix)
        for name, matrix in compact_observables.items()
    }

    pair = solve_total_sign_h2_helicity_pair(assembly)
    padded_states = {
        state.helicity_twice: compact_to_padded_state(full_basis, state.state)
        for state in pair.states
    }
    projector = physical_projector(full_basis)
    rng = np.random.default_rng(20527)
    probe = rng.normal(size=assembly.dimension) + 1.0j * rng.normal(
        size=assembly.dimension
    )
    probe = np.asarray(probe / np.linalg.norm(probe), dtype=np.complex128)
    padded_probe = compact_to_padded_state(full_basis, probe)
    action_parity = _max_abs(
        (padded_hamiltonian @ padded_probe)[: assembly.dimension]
        - assembly.matrix @ probe
    )
    operator_action_parity = max(
        _max_abs(
            (padded_observables[name] @ padded_probe)[: assembly.dimension]
            - compact_observables[name] @ probe
        )
        for name in compact_observables
    )
    state_round_trip = max(
        _max_abs(
            padded_to_compact_state(full_basis, padded_states[helicity])
            - pair.state(helicity).state
        )
        for helicity in HELICITIES_TWICE
    )
    padded_leakage = max(
        float(
            np.vdot(
                state[full_basis.compact_dimension :],
                state[full_basis.compact_dimension :],
            ).real
        )
        for state in padded_states.values()
    )
    encoded_residual = max(
        float(
            np.linalg.norm(
                penalized_hamiltonian @ padded_states[helicity]
                - pair.state(helicity).energy_GeV2 * padded_states[helicity]
            )
        )
        for helicity in HELICITIES_TWICE
    )
    validation = {
        "compact_dimension": full_basis.compact_dimension,
        "padded_dimension": full_basis.padded_dimension,
        "qubits": full_basis.qubits,
        "jz_block_compact_dimensions": {
            helicity: basis.compact_dimension for helicity, basis in block_bases.items()
        },
        "jz_block_padded_dimensions": {
            helicity: basis.padded_dimension for helicity, basis in block_bases.items()
        },
        "jz_block_qubits": {
            helicity: basis.qubits for helicity, basis in block_bases.items()
        },
        "hamiltonian_gershgorin_lower_GeV2": lower_bound,
        "hamiltonian_gershgorin_upper_GeV2": upper_bound,
        "padding_penalty_GeV2": penalty,
        "padding_penalty_above_upper_bound_GeV2": penalty - upper_bound,
        "hamiltonian_hermiticity_residual_GeV2": _max_abs(
            padded_hamiltonian - padded_hamiltonian.getH()
        ),
        "penalized_hamiltonian_hermiticity_residual_GeV2": _max_abs(
            penalized_hamiltonian - penalized_hamiltonian.getH()
        ),
        "physical_projector_commutator_residual_GeV2": _max_abs(
            projector @ penalized_hamiltonian - penalized_hamiltonian @ projector
        ),
        "physical_hamiltonian_preservation_residual_GeV2": _max_abs(
            (projector @ penalized_hamiltonian @ projector)[: assembly.dimension, : assembly.dimension]
            - assembly.matrix
        ),
        "quadrature_hermiticity_and_reconstruction_residual": quadrature_residual,
        "compact_padded_hamiltonian_action_residual_GeV2": action_parity,
        "compact_padded_observable_action_residual": operator_action_parity,
        "state_round_trip_residual": state_round_trip,
        "padded_state_leakage": padded_leakage,
        "encoded_ground_state_residual_GeV2": encoded_residual,
        "observable_count": len(padded_observables),
        "all_observables_hermitian": all(
            _max_abs(matrix - matrix.getH()) < _TOLERANCE
            for matrix in padded_observables.values()
        ),
        "physical_ground_below_padding_penalty": max(
            state.energy_GeV2 for state in pair.states
        ) < penalty,
        "generic_pauli_decomposition_required": False,
        "sparse_hamiltonian_route": "qml.SparseHamiltonian",
        "exact_oracle_state_route": "qml.StatePrep",
        "variational_ansatz_selected": False,
        "optimizer_selected": False,
        "hardware_selected": False,
        "shots": None,
        "physical_activation": False,
    }
    residuals = tuple(
        float(value)
        for key, value in validation.items()
        if "residual" in key
    )
    if (
        full_basis.compact_dimension != 266
        or full_basis.padded_dimension != 512
        or full_basis.qubits != 9
        or any(basis.compact_dimension != 133 for basis in block_bases.values())
        or any(basis.padded_dimension != 256 for basis in block_bases.values())
        or any(basis.qubits != 8 for basis in block_bases.values())
        or max(residuals) > _TOLERANCE
        or not validation["all_observables_hermitian"]
        or not validation["physical_ground_below_padding_penalty"]
    ):
        raise RuntimeError("M2 PennyLane implementation contract did not close")
    return M2PennyLaneContract(
        assembly=assembly,
        helicity_pair=pair,
        positive_current=positive_current,
        reversed_current=reversed_current,
        full_basis=full_basis,
        jz_block_bases=block_bases,
        padded_hamiltonian=padded_hamiltonian,
        penalized_hamiltonian=penalized_hamiltonian,
        padding_penalty_GeV2=penalty,
        hermitian_observables=padded_observables,
        padded_ground_states=padded_states,
        validation=validation,
    )


def m2_pennylane_contract_record(contract: M2PennyLaneContract) -> Mapping[str, Any]:
    if not isinstance(contract, M2PennyLaneContract):
        raise TypeError("contract must be an M2PennyLaneContract")
    basis = contract.full_basis
    payload = {
        "status": M2O_PENNYLANE_CONTRACT_STATUS,
        "claim_tier": M2O_CLAIM_TIER,
        "physical": False,
        "encoding_id": basis.encoding_id,
        "target": contract.assembly.target,
        "compact_dimension": basis.compact_dimension,
        "padded_dimension": basis.padded_dimension,
        "qubits": basis.qubits,
        "basis_order": basis.basis_order,
        "endianness": basis.endianness,
        "first_compact_bitstring": encode_compact_index(basis, 0),
        "last_compact_bitstring": encode_compact_index(
            basis, basis.compact_dimension - 1
        ),
        "first_padding_bitstring": format(
            basis.compact_dimension, f"0{basis.qubits}b"
        ),
        "jz_block_encodings": {
            f"JZ{helicity:+d}/2": {
                "compact_dimension": block.compact_dimension,
                "padded_dimension": block.padded_dimension,
                "qubits": block.qubits,
            }
            for helicity, block in contract.jz_block_bases.items()
        },
        "hamiltonian_root": _sparse_root(contract.padded_hamiltonian),
        "penalized_hamiltonian_root": _sparse_root(
            contract.penalized_hamiltonian
        ),
        "padding_penalty_GeV2": contract.padding_penalty_GeV2,
        "hermitian_observables": {
            name: {
                "root": _sparse_root(matrix),
                "nnz": matrix.nnz,
            }
            for name, matrix in contract.hermitian_observables.items()
        },
        "validation": dict(contract.validation),
        "implementation_api": {
            "device_first": "lightning.qubit",
            "wire_order": tuple(range(basis.qubits)),
            "hamiltonian": "qml.SparseHamiltonian(penalized_hamiltonian,wires)",
            "oracle_state": "qml.StatePrep(padded_state,wires)",
            "expectation": "qml.expval(sparse_hermitian_observable)",
        },
        "pennylane_implementation_ready": True,
        "pennylane_execution_performed": False,
        "variational_algorithm_selected": False,
        "physical_hamiltonian_complete": False,
        "physical_current_complete": False,
        "physical_activation": False,
        "next_action": (
            "implement the nine-qubit sparse exact-oracle route first, compare "
            "against compact SciPy expectations, then choose an eight-qubit "
            "Jz-block ansatz only as a separate optimization decision"
        ),
    }
    return {**_plain(payload), "root": _json_root(payload)}


__all__ = [
    "M2O_PENNYLANE_CONTRACT_STATUS",
    "M2O_CLAIM_TIER",
    "M2O_ENCODING_ID",
    "M2CompactQubitBasis",
    "M2PennyLaneContract",
    "build_full_qubit_basis",
    "build_jz_block_qubit_basis",
    "encode_compact_index",
    "decode_bitstring",
    "compact_to_padded_state",
    "padded_to_compact_state",
    "pad_sparse_operator",
    "physical_projector",
    "padding_projector",
    "build_m2_pennylane_contract",
    "m2_pennylane_contract_record",
]

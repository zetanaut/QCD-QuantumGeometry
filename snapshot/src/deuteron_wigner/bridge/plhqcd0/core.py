"""Q0/PLHQCD0: compact-index operators and an exact PennyLane oracle.

This module consumes only the immutable public surfaces of C131, C142, C144,
C149, and C150.  It keeps the source-owned ``q followed by qg`` compact basis
and embeds that basis into the smallest padded computational basis required by
PennyLane.  The default numerical boundary is C144's explicit
``FIXTURE-FREE`` diagnostic record; it is not a physical parameter point.

The production path is sparse throughout.  A generic Pauli decomposition is a
deliberate Q0 boundary and is never used to compile the Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

from deuteron_wigner.bridge import hqcd4 as _c131
from deuteron_wigner.bridge import hqcdfield as _c142
from deuteron_wigner.bridge import hqcdopapi as _c144
from deuteron_wigner.bridge import hqcdmproj as _c149
from deuteron_wigner.bridge import hqcdzqmass as _c150


BASELINE = "8b866b3d69276b976c913ab23842aa5d9b171018"
SCHEMA = "Q0-PLHQCD0-V1"
RESOLUTIONS = tuple(_c142.RESOLUTIONS)
FIXTURE_FREE = "FIXTURE-FREE"
FIXTURE_IDS = tuple(row["fixture_id"] for row in _c144.diagnostic_fixture_manifest()["fixtures"])
COMPACT_TO_C131 = dict(zip(RESOLUTIONS, _c131.RESOLUTIONS))
_PUBLIC_C131_TO_COMPACT = {value: key for key, value in COMPACT_TO_C131.items()}
_PAULI_BOUNDARY = "GENERIC_FULL_PAULI_DECOMPOSITION_FORBIDDEN_IN_Q0"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _as_complex_vector(vector: Sequence[Any], dimension: int, *, name: str = "vector") -> np.ndarray:
    values = np.asarray(vector, dtype=np.complex128)
    if values.ndim != 1 or values.shape[0] != dimension:
        raise ValueError(f"{name} must have shape ({dimension},), got {values.shape}")
    return values


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class CompactBasis:
    """The source-ordered compact basis and its computational embedding."""

    resolution: str
    c131_resolution: str
    q_dimension: int
    qg_dimension: int
    compact_dimension: int
    padded_dimension: int
    qubits: int
    basis_order: str = "q followed by qg"


@dataclass(frozen=True)
class CertifiedPaddedHamiltonian:
    """Sparse compact and padded Hamiltonians plus their source certificate."""

    basis: CompactBasis
    fixture_id: str
    fixture_record: Mapping[str, Any]
    compact: sparse.csr_matrix
    padded: sparse.csr_matrix
    certificate: Mapping[str, Any]


def load_public_authorities() -> Mapping[str, Any]:
    """Load and cross-check the five allowed immutable public authorities."""

    authorities = {
        "C131": _c131.load_verified_projected_bare_hqcd_authority(),
        "C142": _c142.load_verified_hqcd_field_authority(),
        "C144": _c144.load_verified_hqcd_operator_authority(),
        "C149": _c149.load_verified_hqcd_mass_projector_authority(),
        "C150": _c150.load_verified_hqcd_zq_mass_authority(),
    }
    if not all(bool(report["positive_gate"]) for report in authorities.values()):
        raise ValueError("one or more required public authorities failed its positive gate")
    if authorities["C144"]["C131_package_root"] != authorities["C131"]["package_root"]:
        raise ValueError("C144 does not bind the loaded C131 package root")
    if authorities["C144"]["C142_package_root"] != authorities["C142"]["package_root"]:
        raise ValueError("C144 does not bind the loaded C142 package root")
    if authorities["C150"]["C149_package_root"] != authorities["C149"]["package_root"]:
        raise ValueError("C150 does not bind the loaded C149 package root")
    if authorities["C150"]["physical_Z_q"] or authorities["C150"]["physical_mass"]:
        raise ValueError("Q0 received a physical C150 quantity")
    if authorities["C150"]["counterterms_solved"] or authorities["C150"]["null_representatives"]:
        raise ValueError("Q0 received a forbidden C150 selection")
    if authorities["C149"]["physical_Z_q"] or authorities["C149"]["physical_mass"]:
        raise ValueError("Q0 received a physical C149 quantity")
    return _freeze({
        "schema": "Q0-PUBLIC-AUTHORITY-IMPORT-V1",
        "baseline": BASELINE,
        "authorities": authorities,
        "C131_package_root": authorities["C131"]["package_root"],
        "C142_package_root": authorities["C142"]["package_root"],
        "C144_package_root": authorities["C144"]["package_root"],
        "C149_package_root": authorities["C149"]["package_root"],
        "C150_package_root": authorities["C150"]["package_root"],
        "physical_parameter_selected": False,
        "C150_scheme_selected": False,
        "counterterms_solved": 0,
        "null_representatives": 0,
        "root": "PUBLIC_ROOTS_BOUND_BY_C131_C142_C144_C149_C150",
    })


def verify_public_authorities() -> Mapping[str, Any]:
    """Return a compact, machine-readable authority-import report."""

    report = load_public_authorities()
    return _freeze({
        "schema": "Q0-PUBLIC-AUTHORITY-VERIFY-V1",
        "positive_gate": True,
        "baseline": BASELINE,
        "loaded": tuple(report["authorities"]),
        "physical_parameter_selected": False,
        "C150_scheme_selected": False,
        "forbidden_layers_constructed": 0,
        "root": report["root"],
    })


def basis_metadata(resolution: str) -> CompactBasis:
    """Describe one C142 compact resolution and its padded encoding."""

    if resolution not in RESOLUTIONS:
        raise ValueError(f"unsupported Q0 resolution: {resolution!r}")
    q_dimension = int(_c142.Q_DIMS[resolution])
    qg_dimension = int(_c142.QG_DIMS[resolution])
    compact_dimension = int(_c142.DIRECT_DIMS[resolution])
    if compact_dimension != q_dimension + qg_dimension:
        raise ValueError("C142 direct dimension is not q plus qg")
    if int(_c144.DIMS[resolution]) != compact_dimension:
        raise ValueError("C144 and C142 compact dimensions disagree")
    padded_dimension = _next_power_of_two(compact_dimension)
    return CompactBasis(
        resolution=resolution,
        c131_resolution=COMPACT_TO_C131[resolution],
        q_dimension=q_dimension,
        qg_dimension=qg_dimension,
        compact_dimension=compact_dimension,
        padded_dimension=padded_dimension,
        qubits=padded_dimension.bit_length() - 1,
    )


def encode_index(resolution: str, compact_index: int) -> str:
    """Encode a compact source index as a big-endian computational bitstring."""

    basis = basis_metadata(resolution)
    if isinstance(compact_index, bool) or not isinstance(compact_index, (int, np.integer)):
        raise TypeError("compact index must be an integer")
    index = int(compact_index)
    if not 0 <= index < basis.compact_dimension:
        raise IndexError("compact index is outside the physical basis")
    return format(index, f"0{basis.qubits}b")


def decode_bitstring(resolution: str, bitstring: str) -> int:
    """Decode a Q0 big-endian bitstring and reject padded leakage states."""

    basis = basis_metadata(resolution)
    if not isinstance(bitstring, str) or len(bitstring) != basis.qubits or set(bitstring) - {"0", "1"}:
        raise ValueError(f"bitstring must contain exactly {basis.qubits} binary characters")
    index = int(bitstring, 2)
    if index >= basis.compact_dimension:
        raise ValueError("bitstring denotes a padded leakage state")
    return index


def basis_state(resolution: str, compact_index: int) -> Mapping[str, Any]:
    """Return sector and bitstring metadata for one compact basis index."""

    basis = basis_metadata(resolution)
    index = decode_bitstring(resolution, encode_index(resolution, compact_index))
    if index < basis.q_dimension:
        sector, local_index = "q", index
    else:
        sector, local_index = "qg", index - basis.q_dimension
    return _freeze({
        "resolution": resolution,
        "compact_index": index,
        "sector": sector,
        "local_index": local_index,
        "bitstring": encode_index(resolution, index),
        "basis_order": basis.basis_order,
    })


def compact_to_padded_state(resolution: str, compact_state: Sequence[Any]) -> np.ndarray:
    basis = basis_metadata(resolution)
    compact = _as_complex_vector(compact_state, basis.compact_dimension, name="compact_state")
    padded = np.zeros(basis.padded_dimension, dtype=np.complex128)
    padded[: basis.compact_dimension] = compact
    return padded


def padded_to_compact_state(resolution: str, padded_state: Sequence[Any], *, tolerance: float = 1e-12) -> np.ndarray:
    basis = basis_metadata(resolution)
    padded = _as_complex_vector(padded_state, basis.padded_dimension, name="padded_state")
    leakage = float(np.vdot(padded[basis.compact_dimension :], padded[basis.compact_dimension :]).real)
    if leakage > tolerance:
        raise ValueError(f"state has padded leakage {leakage:.3e} > {tolerance:.3e}")
    return np.array(padded[: basis.compact_dimension], dtype=np.complex128, copy=True)


def physical_subspace_projector(resolution: str) -> sparse.csr_matrix:
    """Return the exact projector onto the compact source basis."""

    basis = basis_metadata(resolution)
    return sparse.diags(
        np.r_[np.ones(basis.compact_dimension), np.zeros(basis.padded_dimension - basis.compact_dimension)],
        offsets=0,
        shape=(basis.padded_dimension, basis.padded_dimension),
        format="csr",
        dtype=np.complex128,
    )


def sector_projector(resolution: str, sector: str) -> sparse.csr_matrix:
    basis = basis_metadata(resolution)
    if sector not in ("q", "qg"):
        raise ValueError("sector must be 'q' or 'qg'")
    first = 0 if sector == "q" else basis.q_dimension
    last = basis.q_dimension if sector == "q" else basis.compact_dimension
    diagonal = np.zeros(basis.padded_dimension, dtype=np.complex128)
    diagonal[first:last] = 1.0
    return sparse.diags(diagonal, format="csr")


def sector_leakage_diagnostics(resolution: str, state: Sequence[Any]) -> Mapping[str, float]:
    """Measure q/qg weights and padded leakage without renormalizing the state."""

    basis = basis_metadata(resolution)
    values = np.asarray(state, dtype=np.complex128)
    if values.ndim != 1 or values.shape[0] not in (basis.compact_dimension, basis.padded_dimension):
        raise ValueError("state must be compact or padded for this resolution")
    padded = compact_to_padded_state(resolution, values) if values.size == basis.compact_dimension else values
    q_weight = float(np.vdot(padded[: basis.q_dimension], padded[: basis.q_dimension]).real)
    qg_weight = float(np.vdot(padded[basis.q_dimension : basis.compact_dimension], padded[basis.q_dimension : basis.compact_dimension]).real)
    leakage = float(np.vdot(padded[basis.compact_dimension :], padded[basis.compact_dimension :]).real)
    return _freeze({
        "norm_squared": float(np.vdot(padded, padded).real),
        "q_weight": q_weight,
        "qg_weight": qg_weight,
        "physical_weight": q_weight + qg_weight,
        "padded_leakage": leakage,
        "projector_residual": abs((q_weight + qg_weight) - float(np.vdot(padded[: basis.compact_dimension], padded[: basis.compact_dimension]).real)),
    })


def _fixture_record(fixture_id: str) -> Mapping[str, Any]:
    if fixture_id not in FIXTURE_IDS:
        raise ValueError(f"unknown C144 diagnostic fixture: {fixture_id!r}")
    record = _c144.load_diagnostic_fixture(fixture_id)
    if record["no_physical_claim"] is not True or record["no_default"] is not True:
        raise ValueError("Q0 requires an explicit no-physical-claim diagnostic record")
    return record


def _coo_from_entries(shape: tuple[int, int], entries: Sequence[Sequence[Any]]) -> sparse.csr_matrix:
    rows, cols, values = zip(*((int(row), int(col), complex(value)) for row, col, value in entries)) if entries else ((), (), ())
    return sparse.coo_matrix((np.asarray(values, dtype=np.complex128), (rows, cols)), shape=shape, dtype=np.complex128).tocsr()


def certified_padded_hamiltonian(resolution: str, *, fixture_id: str = FIXTURE_FREE) -> CertifiedPaddedHamiltonian:
    """Build the C144 compact operator and its certified zero-padded form."""

    basis = basis_metadata(resolution)
    record = _fixture_record(fixture_id)
    operator = _c144.parameterized_sparse_operator(resolution, parameter_record=record)
    entries = tuple(operator["entries"])
    compact = _coo_from_entries((basis.compact_dimension, basis.compact_dimension), entries)
    padded = _coo_from_entries((basis.padded_dimension, basis.padded_dimension), entries)
    hermitian_residual = float(np.max(np.abs((compact - compact.getH()).data))) if (compact - compact.getH()).nnz else 0.0
    if hermitian_residual != 0.0:
        raise ValueError(f"C144 operator is not exactly Hermitian: residual {hermitian_residual}")
    if any(row >= basis.compact_dimension or col >= basis.compact_dimension for row, col, _ in entries):
        raise ValueError("C144 entry escaped the compact physical basis")
    source_terms = tuple(row["term_id"] for row in _c131.retained_term_manifest()["terms"])
    certificate = _freeze({
        "schema": "Q0-CERTIFIED-PADDED-HAMILTONIAN-V1",
        "resolution": resolution,
        "c131_resolution": basis.c131_resolution,
        "fixture_id": fixture_id,
        "fixture_root": record["root"],
        "operator_root": operator["root"],
        "basis_order": basis.basis_order,
        "compact_shape": (basis.compact_dimension, basis.compact_dimension),
        "padded_shape": (basis.padded_dimension, basis.padded_dimension),
        "compact_nnz": int(compact.nnz),
        "padded_nnz": int(padded.nnz),
        "source_terms": source_terms,
        "hermiticity_residual": hermitian_residual,
        "physical_support": True,
        "physical_parameter_selected": False,
        "C150_scheme_selected": False,
        "dense_materialized": False,
        "generic_pauli_decomposition": False,
    })
    return CertifiedPaddedHamiltonian(basis, fixture_id, record, compact, padded, certificate)


def sparse_action(resolution: str, vector: Sequence[Any], *, fixture_id: str = FIXTURE_FREE) -> np.ndarray:
    hamiltonian = certified_padded_hamiltonian(resolution, fixture_id=fixture_id)
    values = _as_complex_vector(vector, hamiltonian.basis.compact_dimension)
    return np.asarray(hamiltonian.compact @ values, dtype=np.complex128)


def matrix_free_action(resolution: str, vector: Sequence[Any], *, fixture_id: str = FIXTURE_FREE) -> np.ndarray:
    hamiltonian = certified_padded_hamiltonian(resolution, fixture_id=fixture_id)
    values = _as_complex_vector(vector, hamiltonian.basis.compact_dimension)
    return np.asarray(_c144.apply_parameterized_operator(resolution, values, parameter_record=hamiltonian.fixture_record), dtype=np.complex128)


def encoded_action(resolution: str, padded_vector: Sequence[Any], *, fixture_id: str = FIXTURE_FREE) -> np.ndarray:
    hamiltonian = certified_padded_hamiltonian(resolution, fixture_id=fixture_id)
    values = _as_complex_vector(padded_vector, hamiltonian.basis.padded_dimension, name="padded_vector")
    return np.asarray(hamiltonian.padded @ values, dtype=np.complex128)


def expectation(resolution: str, state: Sequence[Any], *, fixture_id: str = FIXTURE_FREE) -> complex:
    hamiltonian = certified_padded_hamiltonian(resolution, fixture_id=fixture_id)
    values = _as_complex_vector(state, hamiltonian.basis.compact_dimension)
    return complex(np.vdot(values, hamiltonian.compact @ values))


def _qnode_expectation_for_matrix(resolution: str, state: Sequence[Any], matrix: sparse.spmatrix) -> complex:
    import pennylane as qml

    basis = basis_metadata(resolution)
    values = _as_complex_vector(state, basis.compact_dimension)
    padded = compact_to_padded_state(resolution, values)
    device = qml.device("lightning.qubit", wires=basis.qubits, shots=None, c_dtype=np.complex128)
    observable = qml.SparseHamiltonian(matrix, wires=list(range(basis.qubits)))

    @qml.qnode(device)
    def circuit():
        qml.StatePrep(padded, wires=list(range(basis.qubits)))
        return qml.expval(observable)

    return complex(circuit())


def qnode_expectation(resolution: str, state: Sequence[Any], *, fixture_id: str = FIXTURE_FREE) -> complex:
    """Evaluate the certified sparse operator with lightning.qubit exactly."""

    hamiltonian = certified_padded_hamiltonian(resolution, fixture_id=fixture_id)
    return _qnode_expectation_for_matrix(resolution, state, hamiltonian.padded)


def derivative_sparse_operator(resolution: str, direction_id: str, *, fixture_id: str = FIXTURE_FREE) -> sparse.csr_matrix:
    basis = basis_metadata(resolution)
    record = _fixture_record(fixture_id)
    derivative = _c144.operator_derivative(resolution, direction_id, parameter_record=record)
    return _coo_from_entries((basis.padded_dimension, basis.padded_dimension), derivative["entries"])


def derivative_matrix_free_action(resolution: str, direction_id: str, vector: Sequence[Any], *, fixture_id: str = FIXTURE_FREE) -> np.ndarray:
    basis = basis_metadata(resolution)
    values = _as_complex_vector(vector, basis.compact_dimension)
    record = _fixture_record(fixture_id)
    derivative = _c144.operator_derivative(resolution, direction_id, parameter_record=record)
    output = np.zeros(basis.compact_dimension, dtype=np.complex128)
    for row, col, value in derivative["entries"]:
        output[int(row)] += complex(value) * values[int(col)]
    return output


def derivative_qnode_expectation(resolution: str, direction_id: str, state: Sequence[Any], *, fixture_id: str = FIXTURE_FREE) -> complex:
    return _qnode_expectation_for_matrix(resolution, state, derivative_sparse_operator(resolution, direction_id, fixture_id=fixture_id))


def derivative_parity(resolution: str, direction_id: str, vector: Sequence[Any], *, fixture_id: str = FIXTURE_FREE) -> Mapping[str, Any]:
    basis = basis_metadata(resolution)
    values = _as_complex_vector(vector, basis.compact_dimension)
    sparse_result = np.asarray(derivative_sparse_operator(resolution, direction_id, fixture_id=fixture_id) @ compact_to_padded_state(resolution, values), dtype=np.complex128)[: basis.compact_dimension]
    matrix_free_result = derivative_matrix_free_action(resolution, direction_id, values, fixture_id=fixture_id)
    residual = float(np.max(np.abs(sparse_result - matrix_free_result))) if values.size else 0.0
    return _freeze({"resolution": resolution, "direction": direction_id, "max_abs_residual": residual, "exact": residual == 0.0})


def resource_report(resolution: str, *, fixture_id: str = FIXTURE_FREE) -> Mapping[str, Any]:
    hamiltonian = certified_padded_hamiltonian(resolution, fixture_id=fixture_id)
    basis = hamiltonian.basis
    return _freeze({
        "schema": "Q0-RESOURCE-BOUNDARY-V1",
        "resolution": resolution,
        "compact_dimension": basis.compact_dimension,
        "padded_dimension": basis.padded_dimension,
        "qubits": basis.qubits,
        "compact_nnz": int(hamiltonian.compact.nnz),
        "padded_nnz": int(hamiltonian.padded.nnz),
        "dense_matrix_entries_if_materialized": basis.padded_dimension * basis.padded_dimension,
        "sparse_production_path": True,
        "generic_full_pauli_decomposition": _PAULI_BOUNDARY,
        "vqe_or_ansatz": "NOT_STARTED_IN_Q0",
        "physical_parameter": "NOT_SELECTED",
        "C150_Z_q": "NOT_CONSUMED",
    })


def pauli_decomposition(*_args: Any, **_kwargs: Any) -> None:
    """Explicitly close the forbidden generic Pauli production boundary."""

    raise NotImplementedError(_PAULI_BOUNDARY)


__all__ = [
    "BASELINE", "SCHEMA", "RESOLUTIONS", "FIXTURE_FREE", "FIXTURE_IDS", "COMPACT_TO_C131",
    "CompactBasis", "CertifiedPaddedHamiltonian", "load_public_authorities",
    "verify_public_authorities", "basis_metadata", "encode_index", "decode_bitstring",
    "basis_state", "compact_to_padded_state", "padded_to_compact_state",
    "physical_subspace_projector", "sector_projector", "sector_leakage_diagnostics",
    "certified_padded_hamiltonian", "sparse_action", "matrix_free_action", "encoded_action",
    "expectation", "qnode_expectation", "derivative_sparse_operator",
    "derivative_matrix_free_action", "derivative_qnode_expectation", "derivative_parity",
    "resource_report", "pauli_decomposition",
]

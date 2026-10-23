"""Permutation-complete C7/H2 embedding of the resolved M2 K9 member.

This module performs the first color/orbit enlargement of the M2i
active-cluster space into a three-quark plus three-quark-one-gluon target.  It
constructs the exact S3 sign projector on an abstract orbit of the three quark
slots together with the complete C7 color-singlet spaces.  The resulting
color-orbit multiplicities are one in qqq and two in qqqg, as required by
Volume IX.  M2k subsequently proves that the equal-local-width M2i Jacobi
ground state is not closed under cyclic tree changes, so this projector is
not by itself a complete noncolor quark antisymmetrizer.

The M2 source occupies one active-emitter direction in each qqqg doublet.  Its
orthogonal color singlet is retained as an explicit target state and is never
declared zero.  Terms known to be color scalar and Fock diagonal are extended
to both singlets.  The represented C117 and C53 interactions are extended only
through the exact source isometry.  This makes a source-supported interacting
Hamiltonian, not a complete H2 Hamiltonian.

The same distinction is used for the current.  The source-derived diagonal
active-quark current is color scalar and therefore acts on both qqqg
multiplicities.  Spectator-emitter, number-changing, induced, constrained,
zero-mode, and counterterm currents remain unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import permutations
import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

from deuteron_wigner.bridge.c401_c396_mass_directions import canonical_partitions
from deuteron_wigner.microscopic.h0.color import (
    ColorSingletBasis,
    emitted_gluon_color_amplitudes,
)

from .m2_active_quark_current import M2ActiveQuarkImpulseOperator
from .m2_c53_mixed_singlet_state import (
    C53_PARAMETER_ID,
    C53MixedK9Hamiltonian,
    K9,
)
from .m2_color_singlet_bridge import (
    color_completion_algebra_record,
    m2_color_modules,
)
from .m2_resolved_spectator_current import (
    M2ResolvedSpectatorHamiltonian,
    ResolvedSpectatorResolution,
)


M2J_BASIS_STATUS = "M2J_PERMUTATION_COMPLETE_C7_H2_BASIS_READY"
M2J_HAMILTONIAN_STATUS = (
    "M2J_SOURCE_SUPPORTED_INTERACTING_HAMILTONIAN_PARTIAL_H2_READY"
)
M2J_ACTIVE_CURRENT_STATUS = (
    "M2J_COLOR_SCALAR_ACTIVE_QUARK_CURRENT_ON_COMPLETE_MULTIPLICITY_READY"
)
M2J_CLAIM_TIER = "EXACT_FINITE_K_COLOR_ORBIT_SIGN_EMBEDDING_PARTIAL_DYNAMICS"
M2J_TARGET_RESOLUTION_ID = (
    "M2J_K27_QQQ_QQQG_S3_SIGN_C7_COMPLETE_COLOR_MULTIPLICITY_V1"
)

UNAVAILABLE_M2J_HAMILTONIAN_TERMS = (
    "NONCOLOR_S3_COVARIANT_EMITTER_TREE_COMPLETION",
    "EXCITED_JACOBI_MODES_OR_SYMMETRY_ADAPTED_GLOBAL_GROUND",
    "FULL_SPIN_FLAVOR_S3_RECOUPLING",
    "ACTIVE_SPECTATOR_INTERACTION",
    "ORTHOGONAL_QQQG_SINGLET_C117_ACTION",
    "INSTANTANEOUS_PARTNER_INTERACTIONS",
    "INDUCED_EFFECTIVE_INTERACTIONS",
    "CONSTRAINED_FIELD_INTERACTIONS",
    "ZERO_MODE_INTERACTIONS",
    "SECTOR_DEPENDENT_COUNTERTERMS_AND_MATCHING",
)

UNAVAILABLE_M2J_CURRENT_TERMS = (
    "SPECTATOR_QUARK_ONE_BODY_CURRENT_ON_PERMUTED_EMITTER_TREES",
    "Q_TO_QG_AND_QG_TO_Q_NUMBER_CHANGING_CURRENT",
    "HAMILTONIAN_GAUGED_INTERACTION_CURRENT",
    "INDUCED_EFFECTIVE_CURRENT",
    "CONSTRAINED_FIELD_CURRENT",
    "PAIR_CURRENT",
    "ZERO_MODE_CURRENT",
    "CURRENT_COUNTERTERMS_AND_MATCHING",
)

_GROUP = tuple(permutations(range(3)))
_TOLERANCE = 3.0e-12


def _max_abs(matrix: sparse.spmatrix) -> float:
    values = matrix.tocsr().data
    return float(np.max(np.abs(values))) if values.size else 0.0


def _sparse_root(matrix: sparse.spmatrix) -> str:
    value = matrix.tocsr()
    digest = sha256()
    for array in (value.indptr, value.indices, value.data):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _plain(value: Any) -> Any:
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


def _permutation_sign(value: tuple[int, int, int]) -> int:
    inversions = sum(
        value[left] > value[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )
    return -1 if inversions % 2 else 1


def _compose(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(left[right[index]] for index in range(3))


def _regular_representation(value: tuple[int, int, int]) -> np.ndarray:
    """Right action on slot-assignment orbit labels."""

    index = {item: position for position, item in enumerate(_GROUP)}
    result = np.zeros((len(_GROUP), len(_GROUP)), dtype=np.complex128)
    for column, prior in enumerate(_GROUP):
        result[index[_compose(prior, value)], column] = 1.0
    return result


def _color_permutation_representations(
    sector: str,
) -> tuple[np.ndarray, ...]:
    """Represent quark-slot permutations in a complete C7 singlet basis."""

    basis = ColorSingletBasis.construct(sector)
    dimensions = (3, 3, 3) if sector == "qqq" else (3, 3, 3, 8)
    if sector not in {"qqq", "qqqg"}:
        raise ValueError("only qqq and qqqg permutation representations exist")
    matrices = []
    for value in _GROUP:
        columns = []
        for column in range(basis.multiplicity):
            tensor = basis.tensors[:, column].reshape(dimensions)
            axes = value if sector == "qqq" else value + (3,)
            columns.append(np.transpose(tensor, axes=axes).reshape(-1))
        transformed = np.column_stack(columns)
        matrices.append(basis.tensors.conj().T @ transformed)
    return tuple(matrices)


def _antisymmetrizer(
    color_representations: Sequence[np.ndarray],
) -> np.ndarray:
    if len(color_representations) != len(_GROUP):
        raise ValueError("one color representation is required per S3 element")
    dimension = len(_GROUP) * color_representations[0].shape[0]
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for value, color in zip(_GROUP, color_representations):
        result += _permutation_sign(value) * np.kron(
            _regular_representation(value), color
        )
    return result / float(len(_GROUP))


def _canonical_projector_basis(
    projector: np.ndarray, *, tolerance: float = 1.0e-10
) -> np.ndarray:
    """Deterministic pivot-positive basis for a projector image."""

    rank = int(np.linalg.matrix_rank(projector, tol=tolerance))
    vectors: list[np.ndarray] = []
    for column in range(projector.shape[1]):
        vector = projector[:, column].copy()
        for prior in vectors:
            vector -= prior * np.vdot(prior, vector)
        norm = float(np.linalg.norm(vector))
        if norm <= tolerance:
            continue
        vector /= norm
        pivot = int(np.flatnonzero(np.abs(vector) > tolerance)[0])
        vector *= np.exp(-1j * np.angle(vector[pivot]))
        vectors.append(vector)
        if len(vectors) == rank:
            break
    if len(vectors) != rank:
        raise RuntimeError("failed to construct the complete projector image")
    return np.column_stack(vectors)


@dataclass(frozen=True)
class QuarkPermutationAlgebra:
    group: tuple[tuple[int, int, int], ...]
    qqq_color_representations: tuple[np.ndarray, ...]
    qqqg_color_representations: tuple[np.ndarray, ...]
    qqq_antisymmetrizer: np.ndarray
    qqqg_antisymmetrizer: np.ndarray
    qqq_basis: np.ndarray
    qqqg_basis: np.ndarray
    active_emitter_coordinates: np.ndarray
    orthogonal_emitter_coordinates: np.ndarray
    representation_residual: float
    projector_residual: float
    exchange_eigen_residual: float
    emitter_sum_residual: float
    emitter_span_rank: int

    @property
    def qqq_multiplicity(self) -> int:
        return self.qqq_basis.shape[1]

    @property
    def qqqg_multiplicity(self) -> int:
        return self.qqqg_basis.shape[1]


@lru_cache(maxsize=1)
def build_quark_permutation_algebra() -> QuarkPermutationAlgebra:
    """Construct the exact S3 sign sectors and active-emitter coordinate."""

    qqq_color = _color_permutation_representations("qqq")
    qqqg_color = _color_permutation_representations("qqqg")
    qqq_a = _antisymmetrizer(qqq_color)
    qqqg_a = _antisymmetrizer(qqqg_color)
    qqq_basis = _canonical_projector_basis(qqq_a)
    qqqg_basis = _canonical_projector_basis(qqqg_a)

    identity_index = _GROUP.index((0, 1, 2))
    emitter_color = emitted_gluon_color_amplitudes(0)
    emitter_color = emitter_color / np.linalg.norm(emitter_color)
    seed = np.zeros(6 * emitter_color.size, dtype=np.complex128)
    start = identity_index * emitter_color.size
    seed[start : start + emitter_color.size] = emitter_color
    active_vector = qqqg_a @ seed
    active_vector /= np.linalg.norm(active_vector)
    active_coordinates = qqqg_basis.conj().T @ active_vector
    active_coordinates /= np.linalg.norm(active_coordinates)
    orthogonal_coordinates = np.asarray(
        [-np.conjugate(active_coordinates[1]), np.conjugate(active_coordinates[0])],
        dtype=np.complex128,
    )

    representation_residual = 0.0
    for left_index, left in enumerate(_GROUP):
        for right_index, right in enumerate(_GROUP):
            product = _compose(right, left)
            product_index = _GROUP.index(product)
            for representations in (qqq_color, qqqg_color):
                representation_residual = max(
                    representation_residual,
                    float(
                        np.max(
                            np.abs(
                                representations[left_index]
                                @ representations[right_index]
                                - representations[product_index]
                            )
                        )
                    ),
                )
            representation_residual = max(
                representation_residual,
                float(
                    np.max(
                        np.abs(
                            _regular_representation(left)
                            @ _regular_representation(right)
                            - _regular_representation(product)
                        )
                    )
                ),
            )

    projector_residual = max(
        float(np.max(np.abs(qqq_a - qqq_a.conj().T))),
        float(np.max(np.abs(qqqg_a - qqqg_a.conj().T))),
        float(np.max(np.abs(qqq_a @ qqq_a - qqq_a))),
        float(np.max(np.abs(qqqg_a @ qqqg_a - qqqg_a))),
        float(np.max(np.abs(qqq_basis.conj().T @ qqq_basis - np.eye(1)))),
        float(np.max(np.abs(qqqg_basis.conj().T @ qqqg_basis - np.eye(2)))),
    )

    transposition = (1, 0, 2)
    exchange_eigen_residual = 0.0
    for color, basis in (
        (qqq_color[_GROUP.index(transposition)], qqq_basis),
        (qqqg_color[_GROUP.index(transposition)], qqqg_basis),
    ):
        action = np.kron(_regular_representation(transposition), color)
        exchange_eigen_residual = max(
            exchange_eigen_residual,
            float(np.max(np.abs(action @ basis + basis))),
        )

    emitter_matrix = np.column_stack(
        [emitted_gluon_color_amplitudes(index) for index in range(3)]
    )
    emitter_sum_residual = float(
        np.max(np.abs(np.sum(emitter_matrix, axis=1)))
    )
    emitter_span_rank = int(np.linalg.matrix_rank(emitter_matrix, tol=1.0e-10))

    worst = max(
        representation_residual,
        projector_residual,
        exchange_eigen_residual,
        emitter_sum_residual,
    )
    if qqq_basis.shape[1] != 1 or qqqg_basis.shape[1] != 2:
        raise RuntimeError("S3 sign multiplicities do not match the C7/H2 contract")
    if emitter_span_rank != 2 or worst > _TOLERANCE:
        raise RuntimeError("quark permutation algebra did not close")
    if min(np.abs(active_coordinates)) <= 1.0e-10:
        raise RuntimeError("active emitter does not resolve both qqqg singlets")

    return QuarkPermutationAlgebra(
        group=_GROUP,
        qqq_color_representations=qqq_color,
        qqqg_color_representations=qqqg_color,
        qqq_antisymmetrizer=qqq_a,
        qqqg_antisymmetrizer=qqqg_a,
        qqq_basis=qqq_basis,
        qqqg_basis=qqqg_basis,
        active_emitter_coordinates=active_coordinates,
        orthogonal_emitter_coordinates=orthogonal_coordinates,
        representation_residual=representation_residual,
        projector_residual=projector_residual,
        exchange_eigen_residual=exchange_eigen_residual,
        emitter_sum_residual=emitter_sum_residual,
        emitter_span_rank=emitter_span_rank,
    )


def permutation_algebra_record() -> Mapping[str, Any]:
    algebra = build_quark_permutation_algebra()
    emitter_amplitudes = tuple(
        tuple(complex(item) for item in emitted_gluon_color_amplitudes(index))
        for index in range(3)
    )
    payload = {
        "status": M2J_BASIS_STATUS,
        "claim_tier": M2J_CLAIM_TIER,
        "physical": False,
        "permutation_group": "S3",
        "fermion_projector": "A3=(1/6) sum_pi sign(pi) R_orbit(pi) tensor D_color(pi)",
        "qqq_sign_multiplicity": algebra.qqq_multiplicity,
        "qqqg_sign_multiplicity": algebra.qqqg_multiplicity,
        "C7_qqq_color_singlet_multiplicity": ColorSingletBasis.construct("qqq").multiplicity,
        "C7_qqqg_color_singlet_multiplicity": ColorSingletBasis.construct("qqqg").multiplicity,
        "emitter_color_amplitudes": emitter_amplitudes,
        "emitter_amplitude_sum_residual": algebra.emitter_sum_residual,
        "emitter_span_rank": algebra.emitter_span_rank,
        "active_emitter_coordinates": tuple(algebra.active_emitter_coordinates),
        "orthogonal_emitter_coordinates": tuple(algebra.orthogonal_emitter_coordinates),
        "active_orthogonal_inner_product": complex(
            np.vdot(
                algebra.active_emitter_coordinates,
                algebra.orthogonal_emitter_coordinates,
            )
        ),
        "representation_residual": algebra.representation_residual,
        "projector_residual": algebra.projector_residual,
        "exchange_eigen_residual": algebra.exchange_eigen_residual,
        "orthogonal_singlet_status": "RETAINED_UNSOURCED_NOT_PHYSICAL_ZERO",
    }
    return {**_plain(payload), "root": _json_root(payload)}


@dataclass(frozen=True)
class PermutationCompleteH2BasisState:
    index: int
    sector: str
    source_index: int
    source_label: tuple[object, ...]
    color_multiplicity: int
    longitudinal_K2: tuple[int, ...]
    transverse_modes: tuple[tuple[int, int], ...]
    twice_active_Jz: int


@dataclass(frozen=True)
class PermutationCompleteH2Basis:
    resolution_id: str
    source_resolution: str
    spectator_resolution: ResolvedSpectatorResolution
    states: tuple[PermutationCompleteH2BasisState, ...]
    source_labels: tuple[tuple[object, ...], ...]
    source_to_target_rows: tuple[tuple[int, ...], ...]
    claim_tier: str = M2J_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if self.resolution_id != M2J_TARGET_RESOLUTION_ID:
            raise ValueError("unexpected M2j target resolution")
        if self.source_resolution != K9:
            raise ValueError("the first M2j member must preserve K9")
        if len(self.source_labels) != 450 or len(self.source_to_target_rows) != 450:
            raise ValueError("M2j requires the complete 450-dimensional M2 source")
        if len(self.states) != 898:
            raise ValueError("M2j target must contain 2 qqq and 896 qqqg states")
        if self.claim_tier != M2J_CLAIM_TIER or self.physical:
            raise ValueError("M2j basis must remain a nonphysical finite-basis member")

    @property
    def dimension(self) -> int:
        return len(self.states)

    @property
    def source_dimension(self) -> int:
        return len(self.source_labels)

    @property
    def qqq_dimension(self) -> int:
        return sum(state.sector == "qqq" for state in self.states)

    @property
    def qqqg_dimension(self) -> int:
        return sum(state.sector == "qqqg" for state in self.states)

    @property
    def qqqg_source_dimension(self) -> int:
        return sum(label[0] == "qg" for label in self.source_labels)


def build_permutation_complete_h2_basis(
    spectator_resolution: ResolvedSpectatorResolution,
) -> PermutationCompleteH2Basis:
    """Lift each M2 module into all required S3/C7 singlet multiplicities."""

    if not isinstance(spectator_resolution, ResolvedSpectatorResolution):
        raise TypeError("spectator_resolution must be ResolvedSpectatorResolution")
    if spectator_resolution.active_resolution != K9:
        raise ValueError("the first permutation-complete member preserves K9")
    algebra = build_quark_permutation_algebra()
    labels = m2_color_modules(K9).multiplicity_labels
    partitions = {
        partition.partition_id: partition for partition in canonical_partitions(K9)
    }
    states: list[PermutationCompleteH2BasisState] = []
    rows_by_source: list[tuple[int, ...]] = []

    for source_index, label in enumerate(labels):
        row_start = len(states)
        if label[0] == "q":
            multiplicities = range(algebra.qqq_multiplicity)
            longitudinal = (
                spectator_resolution.active_cluster_K2,
                *spectator_resolution.spectator_quark_K2,
            )
            transverse = (
                spectator_resolution.rho_mode,
                spectator_resolution.lambda_mode,
            )
            twice_jz = int(label[1])
            sector = "qqq"
        elif label[0] == "qg":
            multiplicities = range(algebra.qqqg_multiplicity)
            partition = partitions[label[1]]
            longitudinal = (
                int(2 * partition.kq),
                *spectator_resolution.spectator_quark_K2,
                int(2 * partition.kg),
            )
            transverse = (
                (int(label[2]), int(label[3])),
                spectator_resolution.rho_mode,
                spectator_resolution.lambda_mode,
            )
            twice_jz = int(label[4]) + 2 * int(label[5]) + 2 * int(label[3])
            sector = "qqqg"
        else:
            raise RuntimeError("unexpected M2 source sector")

        if sum(longitudinal) != spectator_resolution.nucleon_K2:
            raise RuntimeError("M2j longitudinal modes do not close the target K2")
        for multiplicity in multiplicities:
            states.append(
                PermutationCompleteH2BasisState(
                    index=len(states),
                    sector=sector,
                    source_index=source_index,
                    source_label=tuple(label),
                    color_multiplicity=int(multiplicity),
                    longitudinal_K2=tuple(int(item) for item in longitudinal),
                    transverse_modes=tuple(transverse),
                    twice_active_Jz=twice_jz,
                )
            )
        rows_by_source.append(tuple(range(row_start, len(states))))

    result = PermutationCompleteH2Basis(
        resolution_id=M2J_TARGET_RESOLUTION_ID,
        source_resolution=K9,
        spectator_resolution=spectator_resolution,
        states=tuple(states),
        source_labels=tuple(labels),
        source_to_target_rows=tuple(rows_by_source),
    )
    if result.qqq_dimension != 2 or result.qqqg_dimension != 896:
        raise RuntimeError("M2j sector dimensions do not match the H2 contract")
    return result


@dataclass(frozen=True)
class M2ToPermutationCompleteH2Embedding:
    basis: PermutationCompleteH2Basis
    matrix: sparse.csr_matrix
    orthogonal_qqqg_projector: sparse.csr_matrix
    qqq_phase: complex
    active_emitter_coordinates: np.ndarray
    orthogonal_emitter_coordinates: np.ndarray
    claim_tier: str = M2J_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if self.matrix.shape != (self.basis.dimension, self.basis.source_dimension):
            raise ValueError("M2j embedding has the wrong shape")
        if self.orthogonal_qqqg_projector.shape != (self.basis.dimension,) * 2:
            raise ValueError("orthogonal projector has the wrong target shape")
        identity = sparse.identity(
            self.basis.source_dimension, format="csr", dtype=np.complex128
        )
        if _max_abs(self.matrix.getH() @ self.matrix - identity) > _TOLERANCE:
            raise ValueError("M2j source embedding is not isometric")
        if self.claim_tier != M2J_CLAIM_TIER or self.physical:
            raise ValueError("M2j embedding must remain nonphysical")

    @property
    def source_projector(self) -> sparse.csr_matrix:
        return (self.matrix @ self.matrix.getH()).tocsr()

    @property
    def embedding_root(self) -> str:
        return _sparse_root(self.matrix)


def build_m2_to_permutation_complete_h2_embedding(
    basis: PermutationCompleteH2Basis,
) -> M2ToPermutationCompleteH2Embedding:
    """Embed M2 into the active-emitter subspace of the complete target."""

    if not isinstance(basis, PermutationCompleteH2Basis):
        raise TypeError("basis must be PermutationCompleteH2Basis")
    algebra = build_quark_permutation_algebra()
    bridge = color_completion_algebra_record()
    qqq_phase = complex(bridge["C7_qqq_overlap"])
    if abs(abs(qqq_phase) - 1.0) > _TOLERANCE:
        raise RuntimeError("C7 qqq bridge phase is not unit modulus")
    active = qqq_phase * algebra.active_emitter_coordinates
    orthogonal = qqq_phase * algebra.orthogonal_emitter_coordinates

    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for source_index, (label, target_rows) in enumerate(
        zip(basis.source_labels, basis.source_to_target_rows)
    ):
        if label[0] == "q":
            if len(target_rows) != 1:
                raise RuntimeError("qqq source must have one sign multiplicity")
            rows.append(target_rows[0])
            columns.append(source_index)
            values.append(qqq_phase)
        else:
            if len(target_rows) != 2:
                raise RuntimeError("qqqg source must retain both sign multiplicities")
            for target_row, coefficient in zip(target_rows, active):
                rows.append(target_row)
                columns.append(source_index)
                values.append(complex(coefficient))

    matrix = sparse.csr_matrix(
        (np.asarray(values), (rows, columns)),
        shape=(basis.dimension, basis.source_dimension),
        dtype=np.complex128,
    )
    q_projector = sparse.csr_matrix((basis.qqq_dimension, basis.qqq_dimension))
    v = orthogonal.reshape(2, 1)
    qg_projector = sparse.kron(
        sparse.identity(basis.qqqg_source_dimension, format="csr"),
        sparse.csr_matrix(v @ v.conj().T),
        format="csr",
    )
    orthogonal_projector = sparse.block_diag(
        (q_projector, qg_projector), format="csr"
    )
    return M2ToPermutationCompleteH2Embedding(
        basis=basis,
        matrix=matrix,
        orthogonal_qqqg_projector=orthogonal_projector,
        qqq_phase=qqq_phase,
        active_emitter_coordinates=active,
        orthogonal_emitter_coordinates=orthogonal,
    )


def lift_color_scalar_fock_diagonal_operator(
    source_operator: sparse.spmatrix,
    embedding: M2ToPermutationCompleteH2Embedding,
    *,
    tolerance: float = _TOLERANCE,
) -> sparse.csr_matrix:
    """Extend a q/qg Fock-diagonal scalar to both qqqg singlets."""

    if not isinstance(embedding, M2ToPermutationCompleteH2Embedding):
        raise TypeError("embedding must be an M2ToPermutationCompleteH2Embedding")
    source = sparse.csr_matrix(source_operator, dtype=np.complex128)
    expected = embedding.basis.source_dimension
    if source.shape != (expected, expected):
        raise ValueError(f"source operator must have shape ({expected},{expected})")
    q = embedding.basis.qqq_dimension
    if _max_abs(source[:q, q:]) > tolerance or _max_abs(source[q:, :q]) > tolerance:
        raise ValueError("color-scalar lift requires a Fock-diagonal source operator")
    q_block = source[:q, :q].tocsr()
    qg_block = sparse.kron(
        source[q:, q:],
        sparse.identity(2, format="csr", dtype=np.complex128),
        format="csr",
    )
    result = sparse.block_diag((q_block, qg_block), format="csr")
    result.eliminate_zeros()
    result.sort_indices()
    return result


def lift_active_emitter_operator(
    source_operator: sparse.spmatrix,
    embedding: M2ToPermutationCompleteH2Embedding,
) -> sparse.csr_matrix:
    """Lift only the represented active-emitter action through the isometry."""

    if not isinstance(embedding, M2ToPermutationCompleteH2Embedding):
        raise TypeError("embedding must be an M2ToPermutationCompleteH2Embedding")
    source = sparse.csr_matrix(source_operator, dtype=np.complex128)
    expected = embedding.basis.source_dimension
    if source.shape != (expected, expected):
        raise ValueError(f"source operator must have shape ({expected},{expected})")
    result = (embedding.matrix @ source @ embedding.matrix.getH()).tocsr()
    result.eliminate_zeros()
    result.sort_indices()
    return result


@dataclass(frozen=True)
class M2JPermutationCompleteHamiltonian:
    source: M2ResolvedSpectatorHamiltonian
    embedding: M2ToPermutationCompleteH2Embedding
    source_universal_operator: sparse.csr_matrix
    source_represented_interaction: sparse.csr_matrix
    universal_operator: sparse.csr_matrix
    represented_interaction: sparse.csr_matrix
    matrix: sparse.csr_matrix
    universal_term_ids: tuple[str, ...]
    represented_term_ids: tuple[str, ...]
    validation: Mapping[str, float]
    claim_tier: str = M2J_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        shape = (self.embedding.basis.dimension,) * 2
        for name in ("universal_operator", "represented_interaction", "matrix"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has the wrong M2j target shape")
        if _max_abs(self.matrix - self.matrix.getH()) > _TOLERANCE:
            raise ValueError("M2j Hamiltonian must be Hermitian")
        if self.claim_tier != M2J_CLAIM_TIER or self.physical:
            raise ValueError("M2j Hamiltonian must remain exploratory")

    @property
    def dimension(self) -> int:
        return self.matrix.shape[0]

    @property
    def hamiltonian_root(self) -> str:
        return _sparse_root(self.matrix)


def build_m2j_permutation_complete_hamiltonian(
    source: M2ResolvedSpectatorHamiltonian,
    *,
    embedding: M2ToPermutationCompleteH2Embedding | None = None,
) -> M2JPermutationCompleteHamiltonian:
    """Build the maximal source-supported M2j Hamiltonian without completion guesses."""

    if not isinstance(source, M2ResolvedSpectatorHamiltonian):
        raise TypeError("source must be an M2ResolvedSpectatorHamiltonian")
    if embedding is None:
        basis = build_permutation_complete_h2_basis(source.resolution)
        embedding = build_m2_to_permutation_complete_h2_embedding(basis)
    if not isinstance(embedding, M2ToPermutationCompleteH2Embedding):
        raise TypeError("embedding has the wrong type")
    if embedding.basis.spectator_resolution != source.resolution:
        raise ValueError("source and target spectator resolutions do not match")

    assembly: C53MixedK9Hamiltonian = source.pair.assembly
    scale = source.active_mass_scale
    identity = sparse.identity(
        assembly.dimension, format="csr", dtype=np.complex128
    )
    source_universal = (
        scale * assembly.h0.reduced_operator
        + source.additive_identity_shift_GeV2 * identity
    ).tocsr()
    source_represented = sparse.csr_matrix(
        (assembly.dimension, assembly.dimension), dtype=np.complex128
    )
    universal_ids = ["M2_K9_H0", "M2I_NESTED_JACOBI_IDENTITY_SHIFT"]
    represented_ids: list[str] = []
    for term in assembly.base_terms:
        contribution = scale * term.coefficient * term.matrix
        if term.parameter_id.startswith("c396:"):
            source_universal = (source_universal + contribution).tocsr()
            universal_ids.append(term.parameter_id)
        else:
            source_represented = (source_represented + contribution).tocsr()
            represented_ids.append(term.parameter_id)
    source_represented = (
        source_represented
        + scale * float(assembly.point.c53_coupling) * assembly.c53.reduced_operator
    ).tocsr()
    represented_ids.append(C53_PARAMETER_ID)

    source_reconstruction = (source_universal + source_represented).tocsr()
    universal = lift_color_scalar_fock_diagonal_operator(
        source_universal, embedding
    )
    represented = lift_active_emitter_operator(source_represented, embedding)
    target = (universal + represented).tocsr()
    target.eliminate_zeros()
    target.sort_indices()
    pullback = (embedding.matrix.getH() @ target @ embedding.matrix).tocsr()
    orthogonal = embedding.orthogonal_qqqg_projector
    validation = {
        "source_reconstruction_max_abs": _max_abs(
            source_reconstruction - source.matrix
        ),
        "target_pullback_max_abs": _max_abs(pullback - source.matrix),
        "target_hermiticity_max_abs": _max_abs(target - target.getH()),
        "represented_orthogonal_action_max_abs": max(
            _max_abs(represented @ orthogonal),
            _max_abs(orthogonal @ represented),
        ),
        "universal_source_pullback_max_abs": _max_abs(
            embedding.matrix.getH() @ universal @ embedding.matrix
            - source_universal
        ),
    }
    if max(validation.values()) > _TOLERANCE:
        raise RuntimeError("M2j Hamiltonian validation failed")
    return M2JPermutationCompleteHamiltonian(
        source=source,
        embedding=embedding,
        source_universal_operator=source_universal,
        source_represented_interaction=source_represented,
        universal_operator=universal,
        represented_interaction=represented,
        matrix=target,
        universal_term_ids=tuple(universal_ids),
        represented_term_ids=tuple(represented_ids),
        validation=validation,
    )


@dataclass(frozen=True)
class M2JPermutationCompleteActiveCurrent:
    source_operator: M2ActiveQuarkImpulseOperator
    embedding: M2ToPermutationCompleteH2Embedding
    matrix: sparse.csr_matrix
    claim_tier: str = M2J_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source_operator, M2ActiveQuarkImpulseOperator):
            raise TypeError("source_operator must be M2ActiveQuarkImpulseOperator")
        if self.source_operator.resolution != self.embedding.basis.source_resolution:
            raise ValueError("current and M2j basis resolutions do not match")
        if self.matrix.shape != (self.embedding.basis.dimension,) * 2:
            raise ValueError("M2j current has the wrong target shape")
        if self.claim_tier != M2J_CLAIM_TIER or self.physical:
            raise ValueError("M2j current must remain exploratory")

    @property
    def operator_root(self) -> str:
        return _sparse_root(self.matrix)


def build_m2j_permutation_complete_active_current(
    source_operator: M2ActiveQuarkImpulseOperator,
    embedding: M2ToPermutationCompleteH2Embedding,
) -> M2JPermutationCompleteActiveCurrent:
    """Lift the diagonal active-quark current as a color scalar."""

    if not isinstance(source_operator, M2ActiveQuarkImpulseOperator):
        raise TypeError("source_operator must be M2ActiveQuarkImpulseOperator")
    matrix = lift_color_scalar_fock_diagonal_operator(
        source_operator.matrix, embedding
    )
    pullback = embedding.matrix.getH() @ matrix @ embedding.matrix
    if _max_abs(pullback - source_operator.matrix) > _TOLERANCE:
        raise RuntimeError("M2j current does not recover its M2 source")
    return M2JPermutationCompleteActiveCurrent(
        source_operator=source_operator,
        embedding=embedding,
        matrix=matrix,
    )


def m2j_embedding_record(
    embedding: M2ToPermutationCompleteH2Embedding,
    hamiltonian: M2JPermutationCompleteHamiltonian | None = None,
) -> Mapping[str, Any]:
    if not isinstance(embedding, M2ToPermutationCompleteH2Embedding):
        raise TypeError("embedding must be an M2ToPermutationCompleteH2Embedding")
    basis = embedding.basis
    source_projector = embedding.source_projector
    orthogonal = embedding.orthogonal_qqqg_projector
    payload: dict[str, Any] = {
        "status": M2J_BASIS_STATUS,
        "hamiltonian_status": (
            M2J_HAMILTONIAN_STATUS if hamiltonian is not None else "NOT_SUPPLIED"
        ),
        "claim_tier": M2J_CLAIM_TIER,
        "physical": False,
        "resolution_id": basis.resolution_id,
        "source_resolution": basis.source_resolution,
        "source_dimension": basis.source_dimension,
        "target_dimension": basis.dimension,
        "qqq_dimension": basis.qqq_dimension,
        "qqqg_dimension": basis.qqqg_dimension,
        "qqq_color_singlet_multiplicity": 1,
        "qqqg_color_singlet_multiplicity": 2,
        "embedding_shape": embedding.matrix.shape,
        "embedding_nnz": int(embedding.matrix.nnz),
        "embedding_isometry_residual": _max_abs(
            embedding.matrix.getH() @ embedding.matrix
            - sparse.identity(basis.source_dimension, format="csr")
        ),
        "source_projector_rank": int(np.rint(source_projector.diagonal().sum().real)),
        "orthogonal_qqqg_projector_rank": int(
            np.rint(orthogonal.diagonal().sum().real)
        ),
        "projector_overlap_max_abs": _max_abs(source_projector @ orthogonal),
        "qqq_phase": embedding.qqq_phase,
        "active_emitter_coordinates": tuple(embedding.active_emitter_coordinates),
        "orthogonal_emitter_coordinates": tuple(
            embedding.orthogonal_emitter_coordinates
        ),
        "orthogonal_singlet_status": "RETAINED_UNSOURCED_NOT_PHYSICAL_ZERO",
        "exact_color_orbit_sign_projector_applied_before_operator_assembly": True,
        "complete_noncolor_quark_antisymmetrizer_applied": False,
        "m2k_emitter_tree_audit_required_for_full_s3_interpretation": True,
        "exact_total_K2": basis.spectator_resolution.nucleon_K2,
        "longitudinal_order": {
            "qqq": "(active_quark,spectator_quark_1,spectator_quark_2)",
            "qqqg": "(active_quark,spectator_quark_1,spectator_quark_2,gluon)",
        },
        "synthetic_c9_h2_hamiltonian_consumed": False,
        "physical_state_constructed": False,
        "production_observable_claimed": False,
        "unavailable_hamiltonian_terms": UNAVAILABLE_M2J_HAMILTONIAN_TERMS,
        "unavailable_current_terms": UNAVAILABLE_M2J_CURRENT_TERMS,
        "embedding_root": embedding.embedding_root,
    }
    if hamiltonian is not None:
        if hamiltonian.embedding is not embedding:
            raise ValueError("Hamiltonian and record embedding must be identical")
        payload.update(
            {
                "hamiltonian_dimension": hamiltonian.dimension,
                "universal_term_ids": hamiltonian.universal_term_ids,
                "represented_active_emitter_term_ids": hamiltonian.represented_term_ids,
                "hamiltonian_validation": hamiltonian.validation,
                "hamiltonian_root": hamiltonian.hamiltonian_root,
                "complete_h2_hamiltonian_constructed": False,
            }
        )
    return {**_plain(payload), "root": _json_root(payload)}


def m2j_active_current_pair_record(
    positive: M2JPermutationCompleteActiveCurrent,
    reversed_transfer: M2JPermutationCompleteActiveCurrent,
    forward: M2JPermutationCompleteActiveCurrent,
) -> Mapping[str, Any]:
    currents = (positive, reversed_transfer, forward)
    if not all(
        isinstance(item, M2JPermutationCompleteActiveCurrent) for item in currents
    ):
        raise TypeError("all inputs must be M2j active currents")
    embedding = positive.embedding
    if any(item.embedding is not embedding for item in currents[1:]):
        raise ValueError("M2j current comparison requires one exact embedding")
    plus_source = positive.source_operator
    minus_source = reversed_transfer.source_operator
    forward_source = forward.source_operator
    if (
        minus_source.delta_x_GeV != -plus_source.delta_x_GeV
        or minus_source.delta_y_GeV != -plus_source.delta_y_GeV
    ):
        raise ValueError("reversed current must carry exactly opposite transfer")
    if forward_source.delta_x_GeV != 0.0 or forward_source.delta_y_GeV != 0.0:
        raise ValueError("forward current must carry zero transfer")
    identity = sparse.identity(
        embedding.basis.dimension, format="csr", dtype=np.complex128
    )
    payload = {
        "status": M2J_ACTIVE_CURRENT_STATUS,
        "claim_tier": M2J_CLAIM_TIER,
        "physical": False,
        "resolution_id": embedding.basis.resolution_id,
        "source_dimension": embedding.basis.source_dimension,
        "target_dimension": embedding.basis.dimension,
        "current_scope": "ACTIVE_QUARK_FOCK_DIAGONAL_COLOR_SCALAR",
        "acts_on_both_qqqg_color_multiplicities": True,
        "source_pullback_max_abs": max(
            _max_abs(
                item.embedding.matrix.getH()
                @ item.matrix
                @ item.embedding.matrix
                - item.source_operator.matrix
            )
            for item in currents
        ),
        "forward_charge_residual": _max_abs(forward.matrix - identity),
        "reversal_adjoint_residual": _max_abs(
            reversed_transfer.matrix - positive.matrix.getH()
        ),
        "positive_operator_root": positive.operator_root,
        "reversed_operator_root": reversed_transfer.operator_root,
        "forward_operator_root": forward.operator_root,
        "number_changing_current_constructed": False,
        "hamiltonian_gauged_current_constructed": False,
        "complete_three_quark_current_constructed": False,
        "production_observable_claimed": False,
        "unavailable_current_terms": UNAVAILABLE_M2J_CURRENT_TERMS,
    }
    return {**_plain(payload), "root": _json_root(payload)}


__all__ = [
    "M2J_BASIS_STATUS",
    "M2J_HAMILTONIAN_STATUS",
    "M2J_ACTIVE_CURRENT_STATUS",
    "M2J_CLAIM_TIER",
    "M2J_TARGET_RESOLUTION_ID",
    "UNAVAILABLE_M2J_HAMILTONIAN_TERMS",
    "UNAVAILABLE_M2J_CURRENT_TERMS",
    "QuarkPermutationAlgebra",
    "build_quark_permutation_algebra",
    "permutation_algebra_record",
    "PermutationCompleteH2BasisState",
    "PermutationCompleteH2Basis",
    "build_permutation_complete_h2_basis",
    "M2ToPermutationCompleteH2Embedding",
    "build_m2_to_permutation_complete_h2_embedding",
    "lift_color_scalar_fock_diagonal_operator",
    "lift_active_emitter_operator",
    "M2JPermutationCompleteHamiltonian",
    "build_m2j_permutation_complete_hamiltonian",
    "M2JPermutationCompleteActiveCurrent",
    "build_m2j_permutation_complete_active_current",
    "m2j_embedding_record",
    "m2j_active_current_pair_record",
]

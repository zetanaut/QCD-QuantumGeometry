"""Exact color completion of the modern M2 open-triplet operator space.

The C47/M2 basis is a direct sum of fundamental SU(3) modules.  It is not a
color-singlet target space.  This module strips the repeated open-color copy,
checks that an M2 operator is SU(3)-equivariant, and completes each remaining
module with an explicit anti-fundamental spectator.  The completed color
singlet agrees with the C7 ``qqq`` singlet and selects the active-emitter
direction inside C7's two-dimensional ``qqqg`` singlet space.

This is an exact representation bridge.  It does not select spectator spin,
flavor, charge, longitudinal/orbital composition, a deuteron state, or a
current, and therefore does not by itself construct ``H_D``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from deuteron_wigner.bridge.basis1 import core as c47
from deuteron_wigner.bridge.c401_c396_mass_directions import resolution_record
from deuteron_wigner.microscopic.h0.color import (
    ColorSingletBasis,
    emitted_gluon_color_amplitudes,
    representation_generators,
)
from deuteron_wigner.microscopic.h0.k_local import (
    c401_target_basis_labels,
    c47_source_basis_labels,
)


COLOR_BRIDGE_CLAIM_TIER = "EXACT_REPRESENTATION_BRIDGE_NONPHYSICAL"
COLORS = (0, 1, 2)
DEFAULT_EQUIVARIANCE_TOLERANCE = 5.0e-12


class M2ColorBridgeError(RuntimeError):
    """Raised when an open-color M2 object cannot descend to a singlet."""


@dataclass(frozen=True)
class M2ColorModules:
    """The multiplicity-space factor in ``H_M2 = M_K tensor 3``."""

    resolution: str
    source_labels: tuple[tuple[object, ...], ...]
    multiplicity_labels: tuple[tuple[object, ...], ...]
    source_indices_by_module: tuple[tuple[int, int, int], ...]

    @property
    def source_dimension(self) -> int:
        return len(self.source_labels)

    @property
    def multiplicity(self) -> int:
        return len(self.multiplicity_labels)

    @property
    def module_major_indices(self) -> tuple[int, ...]:
        return tuple(index for row in self.source_indices_by_module for index in row)

    @property
    def color_zero_indices(self) -> tuple[int, ...]:
        return tuple(row[0] for row in self.source_indices_by_module)


@dataclass(frozen=True)
class M2SingletOperator:
    """One M2 operator represented on its color-singlet completion."""

    resolution: str
    operator_id: str
    reduced_operator: sparse.csr_matrix
    source_dimension: int
    singlet_dimension: int
    source_nnz: int
    reduced_nnz: int
    cross_color_residual: float
    equal_color_block_residual: float
    reconstruction_residual: float
    singlet_projection_residual: float
    claim_tier: str = COLOR_BRIDGE_CLAIM_TIER
    physical: bool = False

    def apply(self, vector: Any) -> np.ndarray:
        values = np.asarray(vector, dtype=np.complex128)
        if values.shape != (self.singlet_dimension,):
            raise ValueError(
                f"singlet vector must have shape ({self.singlet_dimension},)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("singlet vector contains nonfinite entries")
        return np.asarray(self.reduced_operator @ values, dtype=np.complex128)


def m2_color_modules(resolution: str) -> M2ColorModules:
    """Factor the live C401/C410 basis labels into multiplicity and color."""

    labels = tuple(c401_target_basis_labels(resolution))
    expected = int(resolution_record(resolution)["direct_sum_dimension"])
    if len(labels) != expected:
        raise M2ColorBridgeError("live M2 labels do not close the declared dimension")

    order: list[tuple[object, ...]] = []
    grouped: dict[tuple[object, ...], dict[int, int]] = {}
    for index, label in enumerate(labels):
        if label[0] not in {"q", "qg"}:
            raise M2ColorBridgeError("unexpected M2 Fock-sector label")
        module = tuple(label[:-1])
        color = int(label[-1])
        if color not in COLORS:
            raise M2ColorBridgeError("M2 color label is outside the fundamental triplet")
        if module not in grouped:
            order.append(module)
            grouped[module] = {}
        if color in grouped[module]:
            raise M2ColorBridgeError("duplicate color member in an M2 module")
        grouped[module][color] = index

    indices: list[tuple[int, int, int]] = []
    for module in order:
        colors = grouped[module]
        if tuple(sorted(colors)) != COLORS:
            raise M2ColorBridgeError("an M2 module is not a complete fundamental triplet")
        indices.append(tuple(colors[color] for color in COLORS))
    if 3 * len(indices) != expected:
        raise M2ColorBridgeError("fundamental-module count does not close the M2 space")
    return M2ColorModules(resolution, labels, tuple(order), tuple(indices))


def c47_to_m2_target_embedding(resolution: str) -> sparse.csr_matrix:
    """Return the exact C47-shell-order to C401/C410-target permutation."""

    source_labels = tuple(c47_source_basis_labels(resolution))
    target_labels = tuple(c401_target_basis_labels(resolution))
    if len(source_labels) != len(target_labels) or set(source_labels) != set(target_labels):
        raise M2ColorBridgeError("C47 and M2 target basis labels are not bijective")
    source_index = {label: index for index, label in enumerate(source_labels)}
    rows = np.arange(len(target_labels), dtype=np.int64)
    columns = np.asarray([source_index[label] for label in target_labels], dtype=np.int64)
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.complex128), (rows, columns)),
        shape=(len(target_labels), len(source_labels)),
    )


def map_c47_operator_to_m2_target(
    resolution: str, operator: sparse.spmatrix
) -> sparse.csr_matrix:
    """Permute a square C47-order operator into the live M2 target order.

    C53 is assembled in C47's shell-major basis.  Equal dimensions do not
    authorize combining it directly with C401/C410 target-order matrices.
    This function consumes the exact label bijection before any color descent.
    """

    embedding = c47_to_m2_target_embedding(resolution)
    matrix = sparse.csr_matrix(operator, dtype=np.complex128)
    expected_shape = (embedding.shape[1], embedding.shape[1])
    if matrix.shape != expected_shape:
        raise ValueError(f"C47 operator must have shape {expected_shape}")
    if matrix.data.size and not np.all(np.isfinite(matrix.data)):
        raise ValueError("C47 operator contains nonfinite entries")
    mapped = (embedding @ matrix @ embedding.getH()).tocsr()
    mapped.eliminate_zeros()
    mapped.sort_indices()
    return mapped


def antitriplet_diquark_isometry() -> np.ndarray:
    """Return ``D: 3bar -> antisymmetric(3 tensor 3)``.

    The tensor convention is ``D[(b,c),abar] = epsilon[abar,b,c]/sqrt(2)``.
    It is fixed by the C7 generator convention ``T_3bar = -T_3^T``.
    """

    epsilon = np.zeros((3, 3, 3), dtype=np.complex128)
    for a in COLORS:
        for b in COLORS:
            for c in COLORS:
                epsilon[a, b, c] = ((a - b) * (b - c) * (a - c)) / 2.0
    return np.asarray(np.einsum("abc->bca", epsilon).reshape(9, 3) / sqrt(2.0))


def _color_singlet_vector() -> np.ndarray:
    return np.asarray(np.eye(3, dtype=np.complex128).reshape(-1) / sqrt(3.0))


def color_singlet_embedding(resolution: str) -> sparse.csr_matrix:
    """Embed M2 multiplicity space into ``H_M2 tensor 3bar`` singlets.

    The product basis is source-M2-major and spectator-color-minor.  Each
    multiplicity column is ``sum_c |module,c>|cbar>/sqrt(3)``.
    """

    modules = m2_color_modules(resolution)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for module_index, source_indices in enumerate(modules.source_indices_by_module):
        for color, source_index in zip(COLORS, source_indices):
            rows.append(3 * source_index + color)
            columns.append(module_index)
            values.append(1.0 / sqrt(3.0))
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(3 * modules.source_dimension, modules.multiplicity),
        dtype=np.complex128,
    )


def _max_abs(matrix: sparse.spmatrix) -> float:
    values = matrix.tocsr().data
    return float(np.max(np.abs(values))) if values.size else 0.0


def complete_m2_operator_to_singlet(
    resolution: str,
    operator: sparse.spmatrix,
    *,
    operator_id: str,
    tolerance: float = DEFAULT_EQUIVARIANCE_TOLERANCE,
) -> M2SingletOperator:
    """Descend an equivariant M2 operator to the completed singlet space."""

    if not isinstance(operator_id, str) or not operator_id:
        raise ValueError("operator_id must be nonempty")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    modules = m2_color_modules(resolution)
    matrix = sparse.csr_matrix(operator, dtype=np.complex128)
    expected_shape = (modules.source_dimension, modules.source_dimension)
    if matrix.shape != expected_shape:
        raise ValueError(f"operator must have shape {expected_shape}")
    if matrix.data.size and not np.all(np.isfinite(matrix.data)):
        raise ValueError("operator contains nonfinite entries")

    order = np.asarray(modules.module_major_indices, dtype=int)
    module_major = matrix[order, :][:, order].tocsr()
    reduced = module_major[0::3, 0::3].tocsr()
    cross_color_residual = max(
        _max_abs(module_major[color::3, other::3])
        for color in COLORS
        for other in COLORS
        if color != other
    )
    equal_color_block_residual = max(
        _max_abs(module_major[color::3, color::3] - reduced)
        for color in (1, 2)
    )
    reconstructed = sparse.kron(reduced, sparse.eye(3, format="csr"), format="csr")
    reconstruction_residual = _max_abs(module_major - reconstructed)
    worst = max(
        cross_color_residual,
        equal_color_block_residual,
        reconstruction_residual,
    )
    if worst > tolerance:
        raise M2ColorBridgeError(
            f"{operator_id} is not fundamental-color equivariant at {resolution}: "
            f"residual {worst:.3e} exceeds {tolerance:.1e}"
        )

    embedding = color_singlet_embedding(resolution)
    lifted = sparse.kron(matrix, sparse.eye(3, format="csr"), format="csr")
    projected = (embedding.getH() @ lifted @ embedding).tocsr()
    singlet_projection_residual = _max_abs(projected - reduced)
    if singlet_projection_residual > tolerance:
        raise M2ColorBridgeError(
            f"{operator_id} singlet projection disagrees with its reduced operator"
        )

    return M2SingletOperator(
        resolution=resolution,
        operator_id=operator_id,
        reduced_operator=reduced,
        source_dimension=modules.source_dimension,
        singlet_dimension=modules.multiplicity,
        source_nnz=int(matrix.nnz),
        reduced_nnz=int(reduced.nnz),
        cross_color_residual=cross_color_residual,
        equal_color_block_residual=equal_color_block_residual,
        reconstruction_residual=reconstruction_residual,
        singlet_projection_residual=singlet_projection_residual,
    )


def color_completion_algebra_record() -> Mapping[str, Any]:
    """Relate the spectator completion to the exact C7 singlet bases."""

    fundamental = representation_generators("3")
    antifundamental = representation_generators("3bar")
    identity = np.eye(3, dtype=np.complex128)
    singlet = _color_singlet_vector()
    singlet_generator_residual = max(
        float(
            np.linalg.norm(
                (np.kron(generator, identity) + np.kron(identity, conjugate))
                @ singlet
            )
        )
        for generator, conjugate in zip(fundamental, antifundamental)
    )

    diquark = antitriplet_diquark_isometry()
    diquark_isometry_residual = float(
        np.linalg.norm(diquark.conj().T @ diquark - identity)
    )
    diquark_intertwining_residual = max(
        float(
            np.linalg.norm(
                (np.kron(generator, identity) + np.kron(identity, generator))
                @ diquark
                - diquark @ conjugate
            )
        )
        for generator, conjugate in zip(fundamental, antifundamental)
    )

    shaped_diquark = diquark.reshape(3, 3, 3)
    qqq_tensor = np.einsum("bca->abc", shaped_diquark) / sqrt(3.0)
    qqq_vector = qqq_tensor.reshape(-1)
    c7_qqq = ColorSingletBasis.construct("qqq").tensors[:, 0]
    qqq_overlap = complex(np.vdot(c7_qqq, qqq_vector))
    qqq_projection_residual = float(
        np.linalg.norm(c7_qqq * qqq_overlap - qqq_vector)
    )

    qg_triplet = c47.triplet_isometry().reshape(3, 8, 3)
    qqqg_tensor = np.einsum("iaA,bcA->ibca", qg_triplet, shaped_diquark) / sqrt(3.0)
    qqqg_vector = qqqg_tensor.reshape(-1)
    c7_qqqg = ColorSingletBasis.construct("qqqg").tensors
    qqqg_coordinates = c7_qqqg.conj().T @ qqqg_vector
    qqqg_projection_residual = float(
        np.linalg.norm(c7_qqqg @ qqqg_coordinates - qqqg_vector)
    )
    active_emitter = emitted_gluon_color_amplitudes(0)
    active_emitter /= np.linalg.norm(active_emitter)
    active_emitter_residual = float(
        np.linalg.norm(qqqg_coordinates - qqq_overlap * active_emitter)
    )

    return {
        "claim_tier": COLOR_BRIDGE_CLAIM_TIER,
        "physical": False,
        "color_completion": "3 tensor 3bar -> 1",
        "spectator_realization": "3bar -> antisymmetric spectator-quark pair",
        "singlet_generator_residual": singlet_generator_residual,
        "diquark_isometry_residual": diquark_isometry_residual,
        "diquark_intertwining_residual": diquark_intertwining_residual,
        "C7_qqq_overlap": qqq_overlap,
        "C7_qqq_projection_residual": qqq_projection_residual,
        "C7_qqqg_coordinates": tuple(complex(value) for value in qqqg_coordinates),
        "C7_qqqg_projection_residual": qqqg_projection_residual,
        "C7_active_emitter_index": 0,
        "C7_active_emitter_coordinate_residual": active_emitter_residual,
        "C7_qqqg_orthogonal_singlet_status": "UNMAPPED_NOT_ZERO",
        "spin_flavor_orbital_longitudinal_map": "NOT_YET_CONSTRUCTED",
        "deuteron_H_D_constructed": False,
        "finite_K_current_constructed": False,
    }


def color_completion_record(resolution: str) -> Mapping[str, Any]:
    """Describe the exact bridge dimensions and the remaining physics gap."""

    modules = m2_color_modules(resolution)
    embedding = color_singlet_embedding(resolution)
    identity = sparse.eye(modules.multiplicity, format="csr", dtype=np.complex128)
    isometry_residual = _max_abs(embedding.getH() @ embedding - identity)
    payload = {
        "resolution": resolution,
        "claim_tier": COLOR_BRIDGE_CLAIM_TIER,
        "physical": False,
        "source_space": f"H_M2,{resolution}",
        "source_dimension": modules.source_dimension,
        "fundamental_module_multiplicity": modules.multiplicity,
        "factorization": f"H_M2,{resolution} = C^{modules.multiplicity} tensor 3",
        "enlarged_product_dimension": 3 * modules.source_dimension,
        "completed_singlet_dimension": modules.multiplicity,
        "embedding_shape": embedding.shape,
        "embedding_nnz": int(embedding.nnz),
        "embedding_isometry_residual": isometry_residual,
        "C7_color_bridge": dict(color_completion_algebra_record()),
        "status": "M2_COLOR_SINGLET_NUCLEON_COMPLETION_READY",
        "not_yet_H_D": (
            "spectator spin/flavor/charge, nucleon state, nuclear S/D composition, "
            "separate nuclear resolution, transfer fibers, and current remain open"
        ),
    }
    return payload


__all__ = [
    "COLOR_BRIDGE_CLAIM_TIER",
    "COLORS",
    "DEFAULT_EQUIVARIANCE_TOLERANCE",
    "M2ColorBridgeError",
    "M2ColorModules",
    "M2SingletOperator",
    "m2_color_modules",
    "c47_to_m2_target_embedding",
    "map_c47_operator_to_m2_target",
    "antitriplet_diquark_isometry",
    "color_singlet_embedding",
    "complete_m2_operator_to_singlet",
    "color_completion_algebra_record",
    "color_completion_record",
]

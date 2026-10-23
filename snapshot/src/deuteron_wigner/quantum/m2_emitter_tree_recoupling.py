"""M2k emitter-tree Jacobi and canonical-vertex closure audit.

The M2j color/orbit construction retains the two C7 ``qqqg`` singlet
multiplicities, but it does not by itself prove that the M2i local Jacobi
ground state is closed under changing the emitting quark.  This module makes
that missing noncolor calculation explicit.

For the equal-share ``qqq`` source, an ordered tree is
``(emitter, spectator_1, spectator_2)``.  With the M2i normalized nested
coordinates

``q_rho = kappa_s1 - kappa_s2`` and
``q_lambda = 3 kappa_emitter / sqrt(2)``,

all six tree changes have unit Jacobian and preserve the exact free-mass
metric ``diag(3/2, 1)``.  They are ordinary orthogonal transformations only
after ``q_rho`` is rescaled by ``1/sqrt(y_S)``.  Consequently the M2i choice
``b_rho=b_lambda`` is spectator-exchange closed but not cyclic-S3 closed: the
two-dimensional transverse ground-state overlap with a cyclic tree is
exactly ``32/33``.  A one-ground-mode S3 closure would instead require
``b_rho=sqrt(y_S) b_lambda``; otherwise excited Jacobi modes are mandatory.

The color calculation resolves a separate potential ambiguity.  The three
canonical one-body emitters are conjugate terms of one S3-invariant operator.
After the target sign projection their covariant tree images have rank one,
and the normalized orbit sum is exactly the active-emitter direction already
used by M2j.  The rank-two span of the three raw color insertions at one fixed
orbit point must not be interpreted as two additional independently coupled
vertices.  The second singlet remains a valid basis channel, but sourcing it
requires noncolor structure beyond the retained permutation-covariant seed or
another independently derived interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import sqrt
from typing import Any, Mapping

import numpy as np

from deuteron_wigner.microscopic.h0.color import emitted_gluon_color_amplitudes

from .m2_permutation_complete_h2 import build_quark_permutation_algebra
from .m2_resolved_spectator_current import (
    M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1,
)


M2K_EMITTER_TREE_STATUS = "M2K_EMITTER_TREE_RECOUPLING_AND_C53_ORBIT_CLOSURE_READY"
M2K_CLAIM_TIER = "EXACT_EQUAL_SHARE_JACOBI_AND_S3_COLOR_ORBIT_AUDIT"
M2K_RESOLUTION_ID = "M2K_K27_EQUAL_SHARE_EMITTER_TREE_AUDIT_V1"

_TOLERANCE = 3.0e-12
_TREES = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


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
    index = {item: position for position, item in enumerate(_TREES)}
    result = np.zeros((6, 6), dtype=np.complex128)
    for column, prior in enumerate(_TREES):
        result[index[_compose(prior, value)], column] = 1.0
    return result


def _kappa_from_reference_jacobi() -> np.ndarray:
    """Map ``(q_rho,q_lambda)`` to ``(kappa_0,kappa_1,kappa_2)``."""

    return np.asarray(
        (
            (0.0, sqrt(2.0) / 3.0),
            (0.5, -sqrt(2.0) / 6.0),
            (-0.5, -sqrt(2.0) / 6.0),
        ),
        dtype=float,
    )


def jacobi_tree_matrix(tree: tuple[int, int, int]) -> np.ndarray:
    """Return the exact tree coordinates as a map from the reference tree."""

    if tree not in _TREES:
        raise ValueError("tree must be a permutation of (0,1,2)")
    emitter, spectator_1, spectator_2 = tree
    extract = np.zeros((2, 3), dtype=float)
    extract[0, spectator_1] = 1.0
    extract[0, spectator_2] = -1.0
    extract[1, emitter] = 3.0 / sqrt(2.0)
    return extract @ _kappa_from_reference_jacobi()


def mass_weighted_tree_matrix(tree: tuple[int, int, int]) -> np.ndarray:
    """Return the tree map in ``(q_rho/sqrt(y_S),q_lambda)`` coordinates."""

    y_spectator = 2.0 / 3.0
    scale = np.diag((1.0 / sqrt(y_spectator), 1.0))
    return scale @ jacobi_tree_matrix(tree) @ np.linalg.inv(scale)


def transverse_ground_overlap(
    tree: tuple[int, int, int], *, b_rho_GeV: float, b_lambda_GeV: float
) -> float:
    """Overlap of two normalized 2D-transverse product ground states."""

    if b_rho_GeV <= 0.0 or b_lambda_GeV <= 0.0:
        raise ValueError("oscillator scales must be positive")
    transform = jacobi_tree_matrix(tree)
    precision = np.diag((b_rho_GeV**-2, b_lambda_GeV**-2))
    transformed = transform.T @ precision @ transform
    # One Cartesian component contains two Gaussian variables.  The physical
    # transverse wave function has two identical Cartesian components, so the
    # one-component overlap is squared.
    overlap_1d = (
        2.0
        * np.sqrt(np.linalg.det(precision))
        / np.sqrt(np.linalg.det(precision + transformed))
    )
    return float(overlap_1d**2)


@dataclass(frozen=True)
class EmitterTreeRecouplingAudit:
    trees: tuple[tuple[int, int, int], ...]
    jacobi_matrices: tuple[np.ndarray, ...]
    mass_weighted_matrices: tuple[np.ndarray, ...]
    free_mass_metric: np.ndarray
    local_equal_scale_overlaps: tuple[float, ...]
    symmetry_adapted_overlaps: tuple[float, ...]
    symmetry_adapted_b_rho_GeV: float
    jacobian_residual: float
    metric_residual: float
    orthogonality_residual: float
    local_ground_closure_defect: float
    local_ground_leakage_norm: float


def build_emitter_tree_recoupling_audit() -> EmitterTreeRecouplingAudit:
    resolution = M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1
    y_spectator = float(resolution.spectator_cluster_fraction)
    metric = np.diag((1.0 / y_spectator, 1.0))
    jacobi = tuple(jacobi_tree_matrix(tree) for tree in _TREES)
    weighted = tuple(mass_weighted_tree_matrix(tree) for tree in _TREES)
    b_lambda = float(resolution.b_lambda_GeV)
    b_rho = float(resolution.b_rho_GeV)
    adapted_b_rho = sqrt(y_spectator) * b_lambda
    local_overlaps = tuple(
        transverse_ground_overlap(
            tree, b_rho_GeV=b_rho, b_lambda_GeV=b_lambda
        )
        for tree in _TREES
    )
    adapted_overlaps = tuple(
        transverse_ground_overlap(
            tree, b_rho_GeV=adapted_b_rho, b_lambda_GeV=b_lambda
        )
        for tree in _TREES
    )
    jacobian_residual = max(abs(abs(np.linalg.det(item)) - 1.0) for item in jacobi)
    metric_residual = max(
        float(np.max(np.abs(item.T @ metric @ item - metric))) for item in jacobi
    )
    orthogonality_residual = max(
        float(np.max(np.abs(item.T @ item - np.eye(2)))) for item in weighted
    )
    minimum_overlap = min(local_overlaps)
    result = EmitterTreeRecouplingAudit(
        trees=_TREES,
        jacobi_matrices=jacobi,
        mass_weighted_matrices=weighted,
        free_mass_metric=metric,
        local_equal_scale_overlaps=local_overlaps,
        symmetry_adapted_overlaps=adapted_overlaps,
        symmetry_adapted_b_rho_GeV=adapted_b_rho,
        jacobian_residual=float(jacobian_residual),
        metric_residual=metric_residual,
        orthogonality_residual=orthogonality_residual,
        local_ground_closure_defect=float(1.0 - minimum_overlap),
        local_ground_leakage_norm=float(sqrt(max(0.0, 1.0 - minimum_overlap**2))),
    )
    if max(
        result.jacobian_residual,
        result.metric_residual,
        result.orthogonality_residual,
        max(abs(value - 1.0) for value in result.symmetry_adapted_overlaps),
    ) > _TOLERANCE:
        raise RuntimeError("emitter-tree Jacobi recoupling did not close")
    return result


@dataclass(frozen=True)
class CanonicalEmitterOrbitAudit:
    reduced_orbit_vertex: np.ndarray
    active_emitter_coordinates: np.ndarray
    orthogonal_emitter_coordinates: np.ndarray
    covariant_branch_coordinates: tuple[np.ndarray, ...]
    raw_same_orbit_color_rank: int
    covariant_projected_rank: int
    orbit_vertex_residual: float
    branch_collinearity_residual: float
    orthogonal_source_residual: float


def build_canonical_emitter_orbit_audit() -> CanonicalEmitterOrbitAudit:
    algebra = build_quark_permutation_algebra()
    emitter = emitted_gluon_color_amplitudes(0)
    emitter = emitter / np.linalg.norm(emitter)
    local = np.zeros((12, 6), dtype=np.complex128)
    local[:2, 0] = emitter

    orbit_vertex = np.zeros_like(local)
    branch_coordinates = []
    for index, group_element in enumerate(algebra.group):
        regular = _regular_representation(group_element)
        target_action = np.kron(
            regular, algebra.qqqg_color_representations[index]
        )
        source_action = np.kron(
            regular, algebra.qqq_color_representations[index]
        )
        orbit_vertex += target_action @ local @ source_action.conj().T

        branch = target_action @ local[:, 0]
        coordinates = algebra.qqqg_basis.conj().T @ branch
        coordinates *= _permutation_sign(group_element)
        coordinates /= np.linalg.norm(coordinates)
        branch_coordinates.append(coordinates)

    reduced = algebra.qqqg_basis.conj().T @ orbit_vertex @ algebra.qqq_basis
    active = algebra.active_emitter_coordinates
    orthogonal = algebra.orthogonal_emitter_coordinates
    branches = np.column_stack(branch_coordinates)
    raw_color = np.column_stack(
        [emitted_gluon_color_amplitudes(index) for index in range(3)]
    )
    result = CanonicalEmitterOrbitAudit(
        reduced_orbit_vertex=reduced,
        active_emitter_coordinates=active,
        orthogonal_emitter_coordinates=orthogonal,
        covariant_branch_coordinates=tuple(branch_coordinates),
        raw_same_orbit_color_rank=int(np.linalg.matrix_rank(raw_color, tol=1.0e-10)),
        covariant_projected_rank=int(np.linalg.matrix_rank(branches, tol=1.0e-10)),
        orbit_vertex_residual=float(np.max(np.abs(reduced[:, 0] - active))),
        branch_collinearity_residual=float(
            np.max(np.abs(branches - active[:, None]))
        ),
        orthogonal_source_residual=float(abs(np.vdot(orthogonal, reduced[:, 0]))),
    )
    if (
        result.raw_same_orbit_color_rank != 2
        or result.covariant_projected_rank != 1
        or max(
            result.orbit_vertex_residual,
            result.branch_collinearity_residual,
            result.orthogonal_source_residual,
        )
        > _TOLERANCE
    ):
        raise RuntimeError("canonical emitter orbit did not close")
    return result


def emitter_tree_closure_record() -> Mapping[str, Any]:
    recoupling = build_emitter_tree_recoupling_audit()
    orbit = build_canonical_emitter_orbit_audit()
    payload = {
        "status": M2K_EMITTER_TREE_STATUS,
        "claim_tier": M2K_CLAIM_TIER,
        "physical": False,
        "resolution_id": M2K_RESOLUTION_ID,
        "tree_order": "(emitter,spectator_1,spectator_2)",
        "jacobi_coordinate_order": "(q_rho,q_lambda)",
        "jacobi_definitions": (
            "q_rho=kappa_s1-kappa_s2",
            "q_lambda=3*kappa_emitter/sqrt(2)",
        ),
        "free_mass_metric": recoupling.free_mass_metric,
        "tree_matrices": tuple(
            {"tree": tree, "jacobi": jacobi, "mass_weighted": weighted}
            for tree, jacobi, weighted in zip(
                recoupling.trees,
                recoupling.jacobi_matrices,
                recoupling.mass_weighted_matrices,
            )
        ),
        "jacobian_residual": recoupling.jacobian_residual,
        "free_mass_metric_residual": recoupling.metric_residual,
        "mass_weighted_orthogonality_residual": recoupling.orthogonality_residual,
        "m2i_local_equal_scale_overlaps": recoupling.local_equal_scale_overlaps,
        "cyclic_ground_overlap_exact": "32/33",
        "m2i_local_ground_closure_defect": recoupling.local_ground_closure_defect,
        "m2i_local_ground_leakage_norm": recoupling.local_ground_leakage_norm,
        "symmetry_adapted_b_rho_GeV": recoupling.symmetry_adapted_b_rho_GeV,
        "symmetry_adapted_b_lambda_GeV": (
            M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1.b_lambda_GeV
        ),
        "symmetry_adapted_overlaps": recoupling.symmetry_adapted_overlaps,
        "raw_same_orbit_emitter_color_rank": orbit.raw_same_orbit_color_rank,
        "covariant_projected_emitter_rank": orbit.covariant_projected_rank,
        "normalized_orbit_vertex_residual": orbit.orbit_vertex_residual,
        "covariant_branch_collinearity_residual": orbit.branch_collinearity_residual,
        "orthogonal_singlet_canonical_source_residual": orbit.orthogonal_source_residual,
        "independent_spectator_c53_couplings_required": False,
        "canonical_one_body_vertex_interpretation": (
            "one S3 orbit sum generated from one representative emitter"
        ),
        "m2j_active_emitter_normalization_status": "EXACT_ORBIT_SUM_NORMALIZATION",
        "m2j_full_noncolor_s3_closure_status": (
            "NOT_CLOSED_BY_EQUAL_LOCAL_WIDTH_GROUND_TRUNCATION"
        ),
        "orthogonal_qqqg_singlet_status": (
            "RETAINED_BASIS_CHANNEL_NOT_SOURCED_BY_THIS_COVARIANT_ONE_BODY_SEED"
        ),
        "next_required_object": (
            "symmetry-adapted global-Jacobi ground member or an enlarged Jacobi "
            "basis carrying the exact tree recoupling, followed by full spin-flavor "
            "S3 coupling and a Hamiltonian-gauged current"
        ),
        "production_observable_claimed": False,
    }
    return {**_plain(payload), "root": _json_root(payload)}


__all__ = [
    "M2K_EMITTER_TREE_STATUS",
    "M2K_CLAIM_TIER",
    "M2K_RESOLUTION_ID",
    "EmitterTreeRecouplingAudit",
    "CanonicalEmitterOrbitAudit",
    "jacobi_tree_matrix",
    "mass_weighted_tree_matrix",
    "transverse_ground_overlap",
    "build_emitter_tree_recoupling_audit",
    "build_canonical_emitter_orbit_audit",
    "emitter_tree_closure_record",
]

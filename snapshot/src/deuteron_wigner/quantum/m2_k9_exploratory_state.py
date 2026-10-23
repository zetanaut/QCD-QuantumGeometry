"""Parameter-explicit, nonphysical K9 eigenspace diagnostics for M2.

This module is deliberately narrower than a state-to-current calculation.  It
combines the mapped exploratory H0, the two C401/C396 mass directions, and the
C411 exploratory action at named diagnostic parameter points.  It identifies
the lowest invariant eigenspace before it permits any expectation value to be
reported.  A degenerate eigenspace is kept as a subspace/projector; no member
is promoted to a deuteron state.

The Q0 compact/padded codec can check that this K9 subspace has the same public
coordinate order and no padded leakage.  Q1 and Q2 intentionally remain
fixture-only diagnostic packages, so this module does not inject the M2
Hamiltonian into their frozen fixture APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import acos, isfinite
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from deuteron_wigner.bridge import plhqcd0 as q0
from deuteron_wigner.bridge.c401_c396_mass_directions import (
    D_DELTA_MU_G_SQ,
    D_MU_Q_SQ,
    resolution_record,
)
from deuteron_wigner.bridge.c411_c117_i2_finite_c43_adapter import (
    ExploratoryC1171Parameters,
)
from deuteron_wigner.microscopic.h0 import (
    KLocalH0Supply,
    build_exploratory_k_local_h0,
)

from .operator_bundle import (
    CLAIM_TIER,
    ExploratoryHamiltonian,
    build_mapped_exploratory_hamiltonian,
)


_C117_PARAMETER_ID = "c117:I2_density_projector"
_K9 = "K9"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _root(value: Any) -> str:
    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): plain(member) for key, member in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(member) for member in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, complex):
            return [float(item.real), float(item.imag)]
        return item

    payload = json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class ExploratoryK9ParameterPoint:
    """One named Lane-A numerical point, never a physical fit or selection."""

    point_id: str
    mu_q_sq_GeV2: float
    delta_mu_g_sq_GeV2: float
    c117_residual_normalization: float
    c117_mixing_coefficient: float
    c117_coefficient: float
    purpose: str
    claim_tier: str = CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.point_id, str) or not self.point_id:
            raise ValueError("point_id must be nonempty")
        for name in (
            "mu_q_sq_GeV2",
            "delta_mu_g_sq_GeV2",
            "c117_residual_normalization",
            "c117_mixing_coefficient",
            "c117_coefficient",
        ):
            _finite(getattr(self, name), name)
        if not isinstance(self.purpose, str) or not self.purpose:
            raise ValueError("purpose must be nonempty")
        if self.claim_tier != CLAIM_TIER or self.physical:
            raise ValueError("M2 parameter points must remain exploratory and nonphysical")

    @property
    def c396_coefficients(self) -> Mapping[str, float]:
        return {
            D_MU_Q_SQ: float(self.mu_q_sq_GeV2),
            D_DELTA_MU_G_SQ: float(self.delta_mu_g_sq_GeV2),
        }

    @property
    def c117_parameters(self) -> ExploratoryC1171Parameters:
        return ExploratoryC1171Parameters(
            resolution=_K9,
            residual_normalization=float(self.c117_residual_normalization),
            mixing_coefficient=float(self.c117_mixing_coefficient),
        )

    def record(self) -> Mapping[str, Any]:
        payload = {
            "point_id": self.point_id,
            "resolution": _K9,
            "c396_coefficients_GeV2": dict(self.c396_coefficients),
            "c117_residual_normalization": self.c117_residual_normalization,
            "c117_mixing_coefficient": self.c117_mixing_coefficient,
            "c117_coefficient": self.c117_coefficient,
            "purpose": self.purpose,
            "claim_tier": self.claim_tier,
            "physical": self.physical,
            "physical_fit": False,
        }
        return {**payload, "root": _root(payload)}


K9_EXPLORATORY_BASELINE = ExploratoryK9ParameterPoint(
    point_id="M2_K9_EXPLORATORY_BASELINE_V1",
    mu_q_sq_GeV2=0.20,
    delta_mu_g_sq_GeV2=0.10,
    c117_residual_normalization=0.50,
    c117_mixing_coefficient=0.80,
    c117_coefficient=0.07,
    purpose=(
        "Numerically well-conditioned nonzero Lane-A integration point; values are "
        "diagnostic inputs, not source-matched or fitted physical parameters"
    ),
)

K9_EXPLORATORY_SENSITIVITY_POINTS = (
    replace(
        K9_EXPLORATORY_BASELINE,
        point_id="M2_K9_EXPLORATORY_MU_Q_SQ_PLUS_0P05_GEV2_V1",
        mu_q_sq_GeV2=0.25,
        purpose="One-at-a-time +0.05 GeV^2 C401/C396 quark-mass-direction sensitivity",
    ),
    replace(
        K9_EXPLORATORY_BASELINE,
        point_id="M2_K9_EXPLORATORY_DELTA_MU_G_SQ_MINUS_0P05_GEV2_V1",
        delta_mu_g_sq_GeV2=0.05,
        purpose="One-at-a-time -0.05 GeV^2 C401/C396 gluon-mass-direction sensitivity",
    ),
    replace(
        K9_EXPLORATORY_BASELINE,
        point_id="M2_K9_EXPLORATORY_C117_COEFFICIENT_PLUS_0P03_V1",
        c117_coefficient=0.10,
        purpose="One-at-a-time +0.03 C411 exploratory-action coefficient sensitivity",
    ),
)


@dataclass(frozen=True)
class K9HamiltonianAssembly:
    """The named input point and exact sparse K9 bundle it constructs."""

    point: ExploratoryK9ParameterPoint
    h0_supply: KLocalH0Supply
    bundle: ExploratoryHamiltonian


@dataclass(frozen=True)
class ExploratoryK9LowEigenspace:
    """Lowest isolated eigenspace, intentionally not an individually selected state."""

    assembly: K9HamiltonianAssembly
    energies_GeV2: tuple[float, ...]
    basis: np.ndarray
    residual_norms: tuple[float, ...]
    degeneracy_tolerance_GeV2: float
    gap_after_cluster_GeV2: float
    symmetry_labels: Mapping[str, Any]
    sparse_matrix_free_max_abs_residual: float

    @property
    def multiplicity(self) -> int:
        return int(self.basis.shape[1])

    @property
    def energy_GeV2(self) -> float:
        return float(self.energies_GeV2[0])

    @property
    def degenerate(self) -> bool:
        return self.multiplicity > 1

    @property
    def projector(self) -> np.ndarray:
        """Return the invariant projector; callers must not infer a vector choice."""

        return np.asarray(self.basis @ self.basis.conj().T, dtype=np.complex128)


def build_parameter_explicit_k9_hamiltonian(
    point: ExploratoryK9ParameterPoint,
) -> K9HamiltonianAssembly:
    """Assemble an exact sparse K9 bundle without choosing a physical parameter."""

    if not isinstance(point, ExploratoryK9ParameterPoint):
        raise TypeError("point must be an ExploratoryK9ParameterPoint")
    supply = build_exploratory_k_local_h0(_K9)
    bundle = build_mapped_exploratory_hamiltonian(
        _K9,
        h0_supply=supply,
        c396_coefficients=point.c396_coefficients,
        c117_parameters=point.c117_parameters,
        c117_coefficient=point.c117_coefficient,
    )
    if not bundle.h0_is_sparse:
        raise RuntimeError("M2 K9 eigenspace study requires an exact sparse bundle")
    return K9HamiltonianAssembly(point=point, h0_supply=supply, bundle=bundle)


def _seed_vector(dimension: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    vector = generator.normal(size=dimension) + 1j * generator.normal(size=dimension)
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.complex128)


def _lowest_eigenpairs(
    bundle: ExploratoryHamiltonian,
    *,
    count: int,
    tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count < 2 or count >= bundle.dimension:
        raise ValueError("count must be between 2 and dimension - 1")
    matrix = bundle.matrix()
    values, vectors = eigsh(
        matrix,
        k=count,
        which="SA",
        v0=_seed_vector(bundle.dimension, seed),
        tol=tolerance,
        ncv=min(max(2 * count + 1, 32), bundle.dimension),
        maxiter=20000,
    )
    order = np.argsort(np.real(values), kind="stable")
    sorted_values = np.asarray(np.real(values[order]), dtype=float)
    sorted_vectors = np.asarray(vectors[:, order], dtype=np.complex128)
    residuals = np.asarray(
        [
            np.linalg.norm(bundle.apply(sorted_vectors[:, index]) - sorted_values[index] * sorted_vectors[:, index])
            for index in range(count)
        ],
        dtype=float,
    )
    return sorted_values, sorted_vectors, residuals


def _ground_cluster_size(values: np.ndarray, tolerance: float) -> int:
    threshold = max(tolerance, abs(float(values[0])) * tolerance)
    return int(np.count_nonzero(np.abs(values - values[0]) <= threshold))


def _projected_sector_weight(basis: np.ndarray, rows: slice) -> float:
    return float(np.sum(np.abs(basis[rows, :]) ** 2).real / basis.shape[1])


def _label_record(assembly: K9HamiltonianAssembly, basis: np.ndarray) -> Mapping[str, Any]:
    labels = assembly.h0_supply.target_basis_labels
    q_dimension = int(resolution_record(_K9)["q_dimension"])
    weights = np.sum(np.abs(basis) ** 2, axis=1)
    support = tuple(
        (labels[index], float(weights[index]))
        for index in range(q_dimension)
        if weights[index] > 1.0e-9
    )
    helicities = {
        str(target_helicity / 2.0): sum(
            weight
            for (_, label_helicity, _), weight in support
            if label_helicity == target_helicity
        )
        for target_helicity in (-1, 1)
    }
    colors = {
        str(target_color): sum(
            weight for (_, _, label_color), weight in support if label_color == target_color
        )
        for target_color in range(3)
    }
    matrix = assembly.bundle.matrix()
    q_to_qg = matrix[:q_dimension, q_dimension:]
    qg_to_q = matrix[q_dimension:, :q_dimension]
    cross_residual = max(
        float(np.max(np.abs(q_to_qg.data))) if q_to_qg.nnz else 0.0,
        float(np.max(np.abs(qg_to_q.data))) if qg_to_q.nnz else 0.0,
    )
    return {
        "Fock_sector": "q-only invariant ground subspace",
        "q_weight": _projected_sector_weight(basis, slice(0, q_dimension)),
        "qg_weight": _projected_sector_weight(basis, slice(q_dimension, None)),
        "q_to_qg_block_max_abs": cross_residual,
        "open_Jz_component_projector_trace": helicities,
        "open_triplet_color_component_projector_trace": colors,
        "basis_label_projector_diagonal": support,
        "physical_charge_or_flavor_selected": False,
        "physical_color_singlet_selected": False,
        "deuteron_Jz_selected": False,
        "physical_state_selected": False,
    }


def _matrix_route_residual(bundle: ExploratoryHamiltonian, seed: int = 9102) -> float:
    vector = _seed_vector(bundle.dimension, seed)
    sparse_result = np.asarray(bundle.matrix() @ vector, dtype=np.complex128)
    matrix_free_result = np.asarray(bundle.apply(vector), dtype=np.complex128)
    linear_result = np.asarray(bundle.linear_operator() @ vector, dtype=np.complex128)
    return float(max(np.max(np.abs(sparse_result - matrix_free_result)), np.max(np.abs(sparse_result - linear_result))))


def solve_low_k9_eigenspace(
    assembly: K9HamiltonianAssembly,
    *,
    initial_eigenpairs: int = 16,
    degeneracy_tolerance_GeV2: float = 1.0e-10,
    eigensolver_tolerance: float = 1.0e-12,
    seed: int = 410,
) -> ExploratoryK9LowEigenspace:
    """Solve until the lowest cluster is demonstrably separated from the next level."""

    if not isinstance(assembly, K9HamiltonianAssembly):
        raise TypeError("assembly must be a K9HamiltonianAssembly")
    if initial_eigenpairs < 2:
        raise ValueError("initial_eigenpairs must be at least two")
    threshold = _finite(degeneracy_tolerance_GeV2, "degeneracy_tolerance_GeV2")
    solver_tolerance = _finite(eigensolver_tolerance, "eigensolver_tolerance")
    if threshold <= 0.0 or solver_tolerance <= 0.0:
        raise ValueError("eigenspace tolerances must be positive")

    count = min(initial_eigenpairs, assembly.bundle.dimension - 1)
    while True:
        values, vectors, residuals = _lowest_eigenpairs(
            assembly.bundle, count=count, tolerance=solver_tolerance, seed=seed
        )
        multiplicity = _ground_cluster_size(values, threshold)
        if multiplicity < count:
            break
        if count >= min(64, assembly.bundle.dimension - 1):
            raise RuntimeError("lowest cluster is not separated within the declared eigenspace window")
        count = min(count * 2, 64, assembly.bundle.dimension - 1)

    # ARPACK can return a numerically nonorthogonal spanning set inside an
    # exactly degenerate cluster.  QR keeps the invariant span while removing
    # arbitrary solver-vector mixing before projector or derivative work.
    cluster, _ = np.linalg.qr(vectors[:, :multiplicity], mode="reduced")
    cluster_residuals = tuple(
        float(np.linalg.norm(assembly.bundle.apply(cluster[:, index]) - values[0] * cluster[:, index]))
        for index in range(multiplicity)
    )
    gap = float(values[multiplicity] - values[0])
    return ExploratoryK9LowEigenspace(
        assembly=assembly,
        energies_GeV2=tuple(float(item) for item in values[:multiplicity]),
        basis=cluster,
        residual_norms=cluster_residuals,
        degeneracy_tolerance_GeV2=threshold,
        gap_after_cluster_GeV2=gap,
        symmetry_labels=_label_record(assembly, cluster),
        sparse_matrix_free_max_abs_residual=_matrix_route_residual(assembly.bundle),
    )


def _subspace_distance(reference: np.ndarray, candidate: np.ndarray) -> Mapping[str, float]:
    singular_values = np.linalg.svd(reference.conj().T @ candidate, compute_uv=False)
    clipped = np.clip(np.real(singular_values), -1.0, 1.0)
    angles = np.arccos(clipped)
    projector_frobenius = np.sqrt(max(0.0, 2.0 * reference.shape[1] - 2.0 * np.sum(clipped**2)))
    return {
        "maximum_principal_angle_rad": float(np.max(angles)) if angles.size else 0.0,
        "projector_frobenius_distance": float(projector_frobenius),
    }


def stability_report(
    eigenspace: ExploratoryK9LowEigenspace,
    *,
    seeds: Sequence[int] = (101, 211, 307),
    tolerances: Sequence[float] = (1.0e-10, 1.0e-12),
) -> Mapping[str, Any]:
    """Compare invariant subspaces across independent Krylov starts and tolerances."""

    rows = []
    for seed in seeds:
        for tolerance in tolerances:
            candidate = solve_low_k9_eigenspace(
                eigenspace.assembly,
                degeneracy_tolerance_GeV2=eigenspace.degeneracy_tolerance_GeV2,
                eigensolver_tolerance=float(tolerance),
                seed=int(seed),
            )
            if candidate.multiplicity != eigenspace.multiplicity:
                raise RuntimeError("seed/tolerance route changed the identified ground-space multiplicity")
            distance = _subspace_distance(eigenspace.basis, candidate.basis)
            rows.append(
                {
                    "seed": int(seed),
                    "eigensolver_tolerance": float(tolerance),
                    "energy_max_abs_delta_GeV2": float(
                        np.max(np.abs(np.asarray(candidate.energies_GeV2) - np.asarray(eigenspace.energies_GeV2)))
                    ),
                    "residual_norm_max": max(candidate.residual_norms),
                    **distance,
                }
            )
    payload = {
        "rows": tuple(rows),
        "multiplicity": eigenspace.multiplicity,
        "individual_vector_tracking": "FORBIDDEN_DEGENERATE_SUBSPACE_TRACKED",
        "max_energy_delta_GeV2": max(row["energy_max_abs_delta_GeV2"] for row in rows),
        "max_residual_norm": max(row["residual_norm_max"] for row in rows),
        "max_principal_angle_rad": max(row["maximum_principal_angle_rad"] for row in rows),
        "max_projector_frobenius_distance": max(row["projector_frobenius_distance"] for row in rows),
    }
    return {**payload, "root": _root(payload)}


def _subspace_average(matrix: sparse.spmatrix, eigenspace: ExploratoryK9LowEigenspace) -> Mapping[str, Any]:
    projected = np.asarray(eigenspace.basis.conj().T @ (matrix @ eigenspace.basis), dtype=np.complex128)
    values = np.linalg.eigvalsh(projected)
    return {
        "average": float(np.trace(projected).real / eigenspace.multiplicity),
        "projected_eigenvalues": tuple(float(item) for item in values),
        "branch_independent": bool(np.max(np.abs(values - values[0])) <= 1.0e-10),
    }


def derivative_report(
    eigenspace: ExploratoryK9LowEigenspace,
    *,
    finite_difference_step: float = 1.0e-5,
) -> Mapping[str, Any]:
    """Use subspace averages, not an arbitrary degenerate eigenvector, for derivatives."""

    step = _finite(finite_difference_step, "finite_difference_step")
    if step <= 0.0:
        raise ValueError("finite_difference_step must be positive")
    rows = []
    point = eigenspace.assembly.point
    for parameter_id in eigenspace.assembly.bundle.parameter_ids:
        term = eigenspace.assembly.bundle.term(parameter_id)
        average = _subspace_average(term.matrix, eigenspace)
        if parameter_id == f"c396:{D_MU_Q_SQ}":
            plus_point = replace(point, mu_q_sq_GeV2=point.mu_q_sq_GeV2 + step)
            minus_point = replace(point, mu_q_sq_GeV2=point.mu_q_sq_GeV2 - step)
        elif parameter_id == f"c396:{D_DELTA_MU_G_SQ}":
            plus_point = replace(point, delta_mu_g_sq_GeV2=point.delta_mu_g_sq_GeV2 + step)
            minus_point = replace(point, delta_mu_g_sq_GeV2=point.delta_mu_g_sq_GeV2 - step)
        elif parameter_id == _C117_PARAMETER_ID:
            plus_point = replace(point, c117_coefficient=point.c117_coefficient + step)
            minus_point = replace(point, c117_coefficient=point.c117_coefficient - step)
        else:
            raise RuntimeError(f"unrecognized explicit M2 parameter {parameter_id!r}")
        plus = solve_low_k9_eigenspace(build_parameter_explicit_k9_hamiltonian(plus_point))
        minus = solve_low_k9_eigenspace(build_parameter_explicit_k9_hamiltonian(minus_point))
        if plus.multiplicity != eigenspace.multiplicity or minus.multiplicity != eigenspace.multiplicity:
            raise RuntimeError("finite difference changes the tracked eigenspace multiplicity")
        finite_difference = (plus.energy_GeV2 - minus.energy_GeV2) / (2.0 * step)
        rows.append(
            {
                "parameter_id": parameter_id,
                "subspace_average_hellmann_feynman": average["average"],
                "projected_derivative_eigenvalues": average["projected_eigenvalues"],
                "branch_independent": average["branch_independent"],
                "finite_difference_energy_derivative": finite_difference,
                "HF_minus_FD_abs": abs(average["average"] - finite_difference),
            }
        )
    payload = {
        "finite_difference_step": step,
        "rows": tuple(rows),
        "individual_eigenvector_derivative": "NOT_REPORTED_DEGENERATE_SUBSPACE_AVERAGE_USED",
        "max_HF_FD_abs": max(row["HF_minus_FD_abs"] for row in rows),
    }
    return {**payload, "root": _root(payload)}


def sensitivity_report(
    baseline: ExploratoryK9LowEigenspace,
    points: Sequence[ExploratoryK9ParameterPoint] = K9_EXPLORATORY_SENSITIVITY_POINTS,
) -> Mapping[str, Any]:
    """Evaluate the small declared one-at-a-time exploratory sensitivity set."""

    rows = []
    for point in points:
        study = solve_low_k9_eigenspace(build_parameter_explicit_k9_hamiltonian(point))
        rows.append(
            {
                "point": point.record(),
                "energy_GeV2": study.energy_GeV2,
                "energy_shift_GeV2": study.energy_GeV2 - baseline.energy_GeV2,
                "multiplicity": study.multiplicity,
                "gap_after_cluster_GeV2": study.gap_after_cluster_GeV2,
                "q_weight": study.symmetry_labels["q_weight"],
                "qg_weight": study.symmetry_labels["qg_weight"],
            }
        )
    payload = {
        "baseline_point_id": baseline.assembly.point.point_id,
        "rows": tuple(rows),
        "physical_parameter_inference": False,
    }
    return {**payload, "root": _root(payload)}


def q0_codec_report(eigenspace: ExploratoryK9LowEigenspace) -> Mapping[str, Any]:
    """Apply the recovered public Q0 coordinate codec without fixture injection."""

    basis = q0.basis_metadata(_K9)
    if basis.compact_dimension != eigenspace.assembly.bundle.dimension:
        raise RuntimeError("Q0 compact dimension and M2 K9 dimension disagree")
    rows = []
    for column in range(eigenspace.multiplicity):
        compact = eigenspace.basis[:, column]
        padded = q0.compact_to_padded_state(_K9, compact)
        recovered = q0.padded_to_compact_state(_K9, padded)
        sectors = q0.sector_leakage_diagnostics(_K9, padded)
        rows.append(
            {
                "compact_round_trip_max_abs": float(np.max(np.abs(recovered - compact))),
                "q_weight": sectors["q_weight"],
                "qg_weight": sectors["qg_weight"],
                "padded_leakage": sectors["padded_leakage"],
            }
        )
    payload = {
        "Q0_encoding": "COMPACT_INDEX_DIRECT_ORDER_V1",
        "Q0_compact_dimension": basis.compact_dimension,
        "Q0_padded_dimension": basis.padded_dimension,
        "Q0_qubits": basis.qubits,
        "rows": tuple(rows),
        "Q1_route": "FIXTURE_ONLY_NOT_APPLIED_TO_M2_EXTERNAL_HAMILTONIAN",
        "Q2_route": "FIXTURE_ONLY_NOT_APPLIED_TO_M2_EXTERNAL_HAMILTONIAN",
        "physical_state_selected": False,
    }
    return {**payload, "root": _root(payload)}


def q0_stateprep_subspace_report(eigenspace: ExploratoryK9LowEigenspace) -> Mapping[str, Any]:
    """Echo M2 subspace expectations through Q0's public compact encoding.

    This optional route uses the same exact ``lightning.qubit`` StatePrep and
    sparse-observable mechanism recovered by Q0/Q1, but supplies an explicitly
    padded M2 matrix rather than altering a frozen C144 fixture.  The trace
    average over the complete degenerate subspace avoids a vector selection.
    Q2 remains fixture-registry-specific and is therefore not injected here.
    """

    import pennylane as qml

    basis = q0.basis_metadata(_K9)
    dimension = eigenspace.assembly.bundle.dimension
    if basis.compact_dimension != dimension:
        raise RuntimeError("Q0 compact dimension and M2 K9 dimension disagree")
    tail = sparse.csr_matrix((basis.padded_dimension - dimension,) * 2, dtype=np.complex128)
    operators: list[tuple[str, sparse.csr_matrix, str]] = [
        ("total_hamiltonian", eigenspace.assembly.bundle.matrix(), "M2 sparse total"),
        *[
            (term.parameter_id, term.matrix, term.source)
            for term in eigenspace.assembly.bundle.terms
        ],
    ]
    rows = []
    for operator_id, compact_matrix, source in operators:
        padded_matrix = sparse.block_diag((compact_matrix, tail), format="csr")
        device = qml.device("lightning.qubit", wires=basis.qubits, shots=None, c_dtype=np.complex128)
        observable = qml.SparseHamiltonian(padded_matrix, wires=list(range(basis.qubits)))

        @qml.qnode(device)
        def expectation_route(state: np.ndarray) -> complex:
            qml.StatePrep(state, wires=list(range(basis.qubits)))
            return qml.expval(observable)

        qnode_values = []
        sparse_values = []
        matrix_free_values = []
        for column in range(eigenspace.multiplicity):
            compact = eigenspace.basis[:, column]
            padded = q0.compact_to_padded_state(_K9, compact)
            qnode_values.append(complex(expectation_route(padded)))
            sparse_values.append(complex(np.vdot(compact, compact_matrix @ compact)))
            if operator_id == "total_hamiltonian":
                action = eigenspace.assembly.bundle.apply(compact)
            else:
                action = compact_matrix @ compact
            matrix_free_values.append(complex(np.vdot(compact, action)))
        qnode_average = sum(qnode_values) / eigenspace.multiplicity
        sparse_average = sum(sparse_values) / eigenspace.multiplicity
        matrix_free_average = sum(matrix_free_values) / eigenspace.multiplicity
        rows.append(
            {
                "operator_id": operator_id,
                "source": source,
                "subspace_average_qnode": qnode_average,
                "subspace_average_sparse": sparse_average,
                "subspace_average_matrix_free": matrix_free_average,
                "qnode_sparse_abs_residual": abs(qnode_average - sparse_average),
                "sparse_matrix_free_abs_residual": abs(sparse_average - matrix_free_average),
            }
        )
    payload = {
        "route": "Q0_PUBLIC_COMPACT_CODEC_PLUS_Q1_STYLE_STATEPREP_SPARSE_OBSERVABLE",
        "subspace_average": "TRACE_OVER_DEGENERATE_GROUND_PROJECTOR_DIVIDED_BY_MULTIPLICITY",
        "rows": tuple(rows),
        "Q2_route": "FIXTURE_REGISTRY_ONLY_NOT_INJECTED_WITH_M2_EXTERNAL_HAMILTONIAN",
        "physical_state_selected": False,
    }
    return {**payload, "root": _root(payload)}


def exploratory_k9_state_record(
    eigenspace: ExploratoryK9LowEigenspace,
    *,
    include_stability: bool = True,
    include_derivatives: bool = True,
    include_sensitivity: bool = True,
) -> Mapping[str, Any]:
    """Return one compact M2 status record without serializing a large projector."""

    payload: dict[str, Any] = {
        "schema": "M2-EXPLORATORY-K9-LOW-EIGENSPACE-V1",
        "point": eigenspace.assembly.point.record(),
        "energy_GeV2": eigenspace.energy_GeV2,
        "multiplicity": eigenspace.multiplicity,
        "degenerate": eigenspace.degenerate,
        "eigenvalues_GeV2": eigenspace.energies_GeV2,
        "residual_norms": eigenspace.residual_norms,
        "gap_after_cluster_GeV2": eigenspace.gap_after_cluster_GeV2,
        "degeneracy_tolerance_GeV2": eigenspace.degeneracy_tolerance_GeV2,
        "symmetry_labels": eigenspace.symmetry_labels,
        "sparse_matrix_free_max_abs_residual": eigenspace.sparse_matrix_free_max_abs_residual,
        "Q0_codec": q0_codec_report(eigenspace),
        "claim_tier": CLAIM_TIER,
        "physical": False,
        "deuteron_claim": False,
        "current_response": "NOT_EVALUATED_STATE_TO_CURRENT_BASIS_INTERFACE_UNESTABLISHED",
        "hamiltonian_activation": False,
    }
    if include_stability:
        payload["stability"] = stability_report(eigenspace)
    if include_derivatives:
        payload["derivatives"] = derivative_report(eigenspace)
    if include_sensitivity:
        payload["sensitivity"] = sensitivity_report(eigenspace)
    return {**payload, "root": _root(payload)}


__all__ = [
    "CLAIM_TIER",
    "ExploratoryK9ParameterPoint",
    "K9_EXPLORATORY_BASELINE",
    "K9_EXPLORATORY_SENSITIVITY_POINTS",
    "K9HamiltonianAssembly",
    "ExploratoryK9LowEigenspace",
    "build_parameter_explicit_k9_hamiltonian",
    "solve_low_k9_eigenspace",
    "stability_report",
    "derivative_report",
    "sensitivity_report",
    "q0_codec_report",
    "q0_stateprep_subspace_report",
    "exploratory_k9_state_record",
]

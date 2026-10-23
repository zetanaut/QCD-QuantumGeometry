"""Exploratory C53-mixed color-singlet K9 helicity states.

This module is the first dynamical use of the exact M2 spectator color
completion.  It maps C53 from its native C47 shell-major order into the live
C401/C410 target order, descends every operator to the singlet multiplicity
space, and adds the source-derived canonical q <-> qg block with an explicit
caller-owned dimensionless coupling.

The resulting two Jz-resolved vectors are color-singlet helicity states, not
yet physical proton/neutron states: spectator spin, flavor, charge, baryon
number, and nuclear composition remain unconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from math import isfinite
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh

from deuteron_wigner.bridge.c401_c396_mass_directions import (
    D_DELTA_MU_G_SQ,
    D_MU_Q_SQ,
    resolution_record,
)
from deuteron_wigner.bridge.vertex3.core import (
    apply_physical_canonical_emission,
    generated_adjoint_and_block,
)

from .m2_color_singlet_bridge import (
    M2SingletOperator,
    c47_to_m2_target_embedding,
    complete_m2_operator_to_singlet,
    map_c47_operator_to_m2_target,
    m2_color_modules,
)
from .m2_k9_exploratory_state import (
    K9_EXPLORATORY_BASELINE,
    ExploratoryK9ParameterPoint,
    K9HamiltonianAssembly,
    build_parameter_explicit_k9_hamiltonian,
)


K9 = "K9"
C53_MIXED_CLAIM_TIER = "EXPLORATORY_COLOR_SINGLET_NUCLEON_HELICITY_STATE"
C53_PARAMETER_ID = "c53:g_s_canonical_q_qg"
_C117_PARAMETER_ID = "c117:I2_density_projector"
_HELICITIES_TWICE = (-1, 1)
_C53_DIAGNOSTIC_MASS_GEV = 1.2
_C53_DIAGNOSTIC_PPLUS_GEV = 3.0


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _max_abs(matrix: sparse.spmatrix) -> float:
    data = matrix.tocsr().data
    return float(np.max(np.abs(data))) if data.size else 0.0


def _hermiticity_residual(matrix: sparse.spmatrix) -> float:
    return _max_abs(matrix - matrix.getH())


@dataclass(frozen=True)
class ExploratoryC53K9Point:
    """A named nonphysical C53 coupling attached to an explicit M2 point."""

    point_id: str
    base_point: ExploratoryK9ParameterPoint
    c53_coupling: float
    purpose: str
    claim_tier: str = C53_MIXED_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.point_id, str) or not self.point_id:
            raise ValueError("point_id must be nonempty")
        if not isinstance(self.base_point, ExploratoryK9ParameterPoint):
            raise TypeError("base_point must be an ExploratoryK9ParameterPoint")
        _finite_real(self.c53_coupling, "c53_coupling")
        if not isinstance(self.purpose, str) or not self.purpose:
            raise ValueError("purpose must be nonempty")
        if (
            self.base_point.physical
            or self.claim_tier != C53_MIXED_CLAIM_TIER
            or self.physical
        ):
            raise ValueError("C53-mixed K9 points must remain exploratory and nonphysical")

    def record(self) -> Mapping[str, Any]:
        return {
            "point_id": self.point_id,
            "resolution": K9,
            "base_point": self.base_point.record(),
            "c53_parameter_id": C53_PARAMETER_ID,
            "c53_coupling": float(self.c53_coupling),
            "c53_coupling_units": "dimensionless",
            "c53_operator_units": "GeV^2",
            "c53_operator_parameters": {
                "mass_GeV": _C53_DIAGNOSTIC_MASS_GEV,
                "P_plus_GeV": _C53_DIAGNOSTIC_PPLUS_GEV,
                "g_s": "FACTORED_CALLER_OWNED",
            },
            "purpose": self.purpose,
            "claim_tier": self.claim_tier,
            "physical": self.physical,
            "physical_fit": False,
        }


C53_MIXED_K9_EXPLORATORY_BASELINE = ExploratoryC53K9Point(
    point_id="M2_C53_MIXED_K9_EXPLORATORY_BASELINE_V1",
    base_point=K9_EXPLORATORY_BASELINE,
    c53_coupling=0.05,
    purpose=(
        "Moderate nonzero qg-support diagnostic with a separated helicity pair; "
        "the coupling and C52 mass/Pplus inputs are not a physical fit"
    ),
)


@dataclass(frozen=True)
class ReducedOperatorTerm:
    parameter_id: str
    coefficient: float
    matrix: sparse.csr_matrix
    source: str
    coefficient_units: str
    operator_units: str


@dataclass(frozen=True)
class C53MixedK9Hamiltonian:
    """The 450-dimensional spectator-completed K9 Hamiltonian."""

    point: ExploratoryC53K9Point
    full_assembly: K9HamiltonianAssembly
    h0: M2SingletOperator
    base_terms: tuple[ReducedOperatorTerm, ...]
    c53: M2SingletOperator
    direct_c53_emission: sparse.csr_matrix
    validation: Mapping[str, Any]
    claim_tier: str = C53_MIXED_CLAIM_TIER
    physical: bool = False

    @property
    def dimension(self) -> int:
        return self.h0.singlet_dimension

    @property
    def q_dimension(self) -> int:
        return 2

    @property
    def parameter_ids(self) -> tuple[str, ...]:
        return tuple(term.parameter_id for term in self.base_terms) + (C53_PARAMETER_ID,)

    def term_matrix(self, parameter_id: str) -> sparse.csr_matrix:
        if parameter_id == C53_PARAMETER_ID:
            return self.c53.reduced_operator
        for term in self.base_terms:
            if term.parameter_id == parameter_id:
                return term.matrix
        raise KeyError(parameter_id)

    def matrix(self) -> sparse.csr_matrix:
        result = self.h0.reduced_operator.copy()
        for term in self.base_terms:
            result = (result + term.coefficient * term.matrix).tocsr()
        result = (
            result + float(self.point.c53_coupling) * self.c53.reduced_operator
        ).tocsr()
        result.eliminate_zeros()
        result.sort_indices()
        return result

    def apply(self, vector: Sequence[Any]) -> np.ndarray:
        values = np.asarray(vector, dtype=np.complex128)
        if values.shape != (self.dimension,):
            raise ValueError(f"vector must have shape ({self.dimension},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("vector contains nonfinite entries")
        result = np.asarray(self.h0.reduced_operator @ values, dtype=np.complex128)
        for term in self.base_terms:
            result += term.coefficient * (term.matrix @ values)
        q = values[: self.q_dimension]
        qg = values[self.q_dimension :]
        coupling = float(self.point.c53_coupling)
        result[: self.q_dimension] += coupling * (self.direct_c53_emission.getH() @ qg)
        result[self.q_dimension :] += coupling * (self.direct_c53_emission @ q)
        return result

    def linear_operator(self) -> LinearOperator:
        return LinearOperator(
            (self.dimension, self.dimension),
            matvec=self.apply,
            rmatvec=self.apply,
            dtype=np.complex128,
        )


def _reduced_jz_values() -> np.ndarray:
    values = []
    for label in m2_color_modules(K9).multiplicity_labels:
        if label[0] == "q":
            values.append(label[1] / 2.0)
        else:
            values.append(label[4] / 2.0 + label[5] + label[3])
    return np.asarray(values, dtype=float)


def _direct_reduced_c53_emission(route: str) -> sparse.csr_matrix:
    modules = m2_color_modules(K9)
    embedding = c47_to_m2_target_embedding(K9)
    full_dimension = modules.source_dimension
    q_full_dimension = int(resolution_record(K9)["q_dimension"])
    full_resolution = str(resolution_record(K9)["full_resolution_id"])
    q_modules = [
        index for index, label in enumerate(modules.multiplicity_labels) if label[0] == "q"
    ]
    qg_modules = [
        index for index, label in enumerate(modules.multiplicity_labels) if label[0] == "qg"
    ]
    if q_modules != [0, 1] or len(qg_modules) != modules.multiplicity - 2:
        raise RuntimeError("unexpected K9 reduced q/qg ordering")

    columns = []
    for module_index in q_modules:
        target_input = np.zeros(full_dimension, dtype=np.complex128)
        target_input[modules.source_indices_by_module[module_index][0]] = 1.0
        source_input = np.asarray(embedding.getH() @ target_input, dtype=np.complex128)
        source_emission = apply_physical_canonical_emission(
            source_input[:q_full_dimension], full_resolution, route=route
        )
        source_output = np.zeros(full_dimension, dtype=np.complex128)
        source_output[q_full_dimension:] = source_emission
        target_output = np.asarray(embedding @ source_output, dtype=np.complex128)
        columns.append(
            np.asarray(
                [
                    target_output[modules.source_indices_by_module[index][0]]
                    for index in qg_modules
                ],
                dtype=np.complex128,
            )
        )
    return sparse.csr_matrix(np.column_stack(columns))


@lru_cache(maxsize=1)
def _direct_c53_routes() -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    return (
        _direct_reduced_c53_emission("reduced"),
        _direct_reduced_c53_emission("full_product"),
    )


def build_c53_mixed_k9_hamiltonian(
    point: ExploratoryC53K9Point,
) -> C53MixedK9Hamiltonian:
    """Build the mapped and color-completed C53 K9 Hamiltonian."""

    if not isinstance(point, ExploratoryC53K9Point):
        raise TypeError("point must be an ExploratoryC53K9Point")
    full_assembly = build_parameter_explicit_k9_hamiltonian(point.base_point)
    h0 = complete_m2_operator_to_singlet(
        K9, full_assembly.bundle.h0, operator_id="M2_K9_H0"
    )
    base_terms = tuple(
        ReducedOperatorTerm(
            parameter_id=term.parameter_id,
            coefficient=float(term.coefficient),
            matrix=complete_m2_operator_to_singlet(
                K9, term.matrix, operator_id=term.parameter_id
            ).reduced_operator,
            source=term.source,
            coefficient_units=term.coefficient_units,
            operator_units=term.operator_units,
        )
        for term in full_assembly.bundle.terms
    )

    full_resolution = str(resolution_record(K9)["full_resolution_id"])
    c53_source = generated_adjoint_and_block(full_resolution)["block"]
    c53_target = map_c47_operator_to_m2_target(K9, c53_source)
    c53 = complete_m2_operator_to_singlet(
        K9, c53_target, operator_id="C53_CANONICAL_VERTEX_MAPPED_TO_M2_TARGET"
    )
    direct_reduced, direct_full_product = _direct_c53_routes()
    sparse_emission = c53.reduced_operator[2:, :2].tocsr()

    reduced_total_from_terms = h0.reduced_operator.copy()
    for term in base_terms:
        reduced_total_from_terms = (
            reduced_total_from_terms + term.coefficient * term.matrix
        ).tocsr()
    reduced_full_total = complete_m2_operator_to_singlet(
        K9,
        full_assembly.bundle.matrix(),
        operator_id="M2_K9_BASE_TOTAL",
    ).reduced_operator
    jz = sparse.diags(_reduced_jz_values(), format="csr", dtype=np.complex128)
    c117_term = next(
        term for term in base_terms if term.parameter_id == _C117_PARAMETER_ID
    )
    total = (
        reduced_total_from_terms
        + float(point.c53_coupling) * c53.reduced_operator
    ).tocsr()
    validation = {
        "source_basis": "C47_SHELL_MAJOR",
        "target_basis": "C401_C410_TARGET_ORDER",
        "basis_permutation_applied": True,
        "dimension": h0.singlet_dimension,
        "q_dimension": 2,
        "qg_dimension": h0.singlet_dimension - 2,
        "c53_source_nnz": int(c53.source_nnz),
        "c53_reduced_nnz": int(c53.reduced_nnz),
        "c53_sparse_vs_direct_reduced_max_abs": _max_abs(
            sparse_emission - direct_reduced
        ),
        "c53_direct_reduced_vs_full_product_max_abs": _max_abs(
            direct_reduced - direct_full_product
        ),
        "base_reconstruction_max_abs": _max_abs(
            reduced_total_from_terms - reduced_full_total
        ),
        "c53_jz_commutator_max_abs": _max_abs(
            jz @ c53.reduced_operator - c53.reduced_operator @ jz
        ),
        "c117_jz_commutator_max_abs": _max_abs(
            jz @ c117_term.matrix - c117_term.matrix @ jz
        ),
        "c117_baseline_coefficient": c117_term.coefficient,
        "c117_helicity_hamiltonian_status": "MAPPED_JZ_CONSERVING_EXPLORATORY_TERM",
        "total_jz_commutator_max_abs": _max_abs(jz @ total - total @ jz),
        "total_hermiticity_max_abs": _hermiticity_residual(total),
        "claim_tier": C53_MIXED_CLAIM_TIER,
        "physical": False,
    }
    residual_keys = (
        "c53_sparse_vs_direct_reduced_max_abs",
        "c53_direct_reduced_vs_full_product_max_abs",
        "base_reconstruction_max_abs",
        "c53_jz_commutator_max_abs",
        "c117_jz_commutator_max_abs",
        "total_jz_commutator_max_abs",
        "total_hermiticity_max_abs",
    )
    worst_key = max(residual_keys, key=lambda key: validation[key])
    if validation[worst_key] > 2.0e-11:
        raise RuntimeError(
            "C53-mixed K9 assembly failed exact-route validation: "
            f"{worst_key}={validation[worst_key]:.3e}"
        )
    return C53MixedK9Hamiltonian(
        point=point,
        full_assembly=full_assembly,
        h0=h0,
        base_terms=base_terms,
        c53=c53,
        direct_c53_emission=direct_reduced,
        validation=validation,
    )


@dataclass(frozen=True)
class C53MixedHelicityState:
    helicity_twice: int
    energy_GeV2: float
    state: np.ndarray
    residual_norm_GeV2: float
    q_weight: float
    qg_weight: float
    sector_gap_GeV2: float

    @property
    def helicity(self) -> float:
        return self.helicity_twice / 2.0


@dataclass(frozen=True)
class C53MixedK9HelicityPair:
    assembly: C53MixedK9Hamiltonian
    states: tuple[C53MixedHelicityState, C53MixedHelicityState]
    helicity_splitting_GeV2: float
    gap_after_pair_GeV2: float
    sparse_matrix_free_max_abs: float
    claim_tier: str = C53_MIXED_CLAIM_TIER
    physical: bool = False

    def state(self, helicity_twice: int) -> C53MixedHelicityState:
        for value in self.states:
            if value.helicity_twice == helicity_twice:
                return value
        raise KeyError(helicity_twice)

    @property
    def basis(self) -> np.ndarray:
        return np.column_stack([self.state(value).state for value in _HELICITIES_TWICE])

    @property
    def projector(self) -> np.ndarray:
        basis = self.basis
        return np.asarray(basis @ basis.conj().T, dtype=np.complex128)


def _phase_fix(state: np.ndarray, q_index: int) -> np.ndarray:
    result = np.asarray(state, dtype=np.complex128).copy()
    pivot = q_index if abs(result[q_index]) > 1.0e-14 else int(np.argmax(np.abs(result)))
    result *= np.conjugate(result[pivot]) / abs(result[pivot])
    return result


def solve_c53_mixed_k9_helicity_pair(
    assembly: C53MixedK9Hamiltonian,
) -> C53MixedK9HelicityPair:
    """Solve the lowest state separately in each exact Jz=+/-1/2 sector."""

    if not isinstance(assembly, C53MixedK9Hamiltonian):
        raise TypeError("assembly must be a C53MixedK9Hamiltonian")
    matrix = assembly.matrix()
    jz = _reduced_jz_values()
    states = []
    for helicity_twice in _HELICITIES_TWICE:
        helicity = helicity_twice / 2.0
        indices = np.flatnonzero(jz == helicity)
        sector = matrix[indices, :][:, indices].toarray()
        values, vectors = np.linalg.eigh(sector)
        full_state = np.zeros(assembly.dimension, dtype=np.complex128)
        full_state[indices] = vectors[:, 0]
        q_index = 0 if helicity_twice == -1 else 1
        full_state = _phase_fix(full_state, q_index)
        q_weight = float(np.vdot(full_state[:2], full_state[:2]).real)
        qg_weight = float(np.vdot(full_state[2:], full_state[2:]).real)
        states.append(
            C53MixedHelicityState(
                helicity_twice=helicity_twice,
                energy_GeV2=float(values[0]),
                state=full_state,
                residual_norm_GeV2=float(
                    np.linalg.norm(assembly.apply(full_state) - values[0] * full_state)
                ),
                q_weight=q_weight,
                qg_weight=qg_weight,
                sector_gap_GeV2=float(values[1] - values[0]),
            )
        )

    global_values = np.sort(
        eigsh(matrix, k=4, which="SA", return_eigenvectors=False, tol=1.0e-12)
    )
    pair_energies = sorted(state.energy_GeV2 for state in states)
    gap_after_pair = float(global_values[2] - pair_energies[-1])
    generator = np.random.default_rng(5309)
    probe = generator.normal(size=assembly.dimension) + 1j * generator.normal(
        size=assembly.dimension
    )
    probe = np.asarray(probe / np.linalg.norm(probe), dtype=np.complex128)
    sparse_action = np.asarray(matrix @ probe, dtype=np.complex128)
    direct_action = assembly.apply(probe)
    linear_action = np.asarray(assembly.linear_operator() @ probe, dtype=np.complex128)
    route_residual = float(
        max(
            np.max(np.abs(sparse_action - direct_action)),
            np.max(np.abs(sparse_action - linear_action)),
        )
    )
    return C53MixedK9HelicityPair(
        assembly=assembly,
        states=(states[0], states[1]),
        helicity_splitting_GeV2=float(states[1].energy_GeV2 - states[0].energy_GeV2),
        gap_after_pair_GeV2=gap_after_pair,
        sparse_matrix_free_max_abs=route_residual,
    )


def vary_c53_mixed_k9_point(
    point: ExploratoryC53K9Point,
    parameter_id: str,
    delta: float,
    *,
    point_id: str | None = None,
) -> ExploratoryC53K9Point:
    """Vary one owned coefficient while preserving the typed exploratory point.

    ``point_id`` lets downstream response calculations give the two finite-
    difference endpoints distinct provenance without duplicating the parameter
    routing owned here.
    """

    if not isinstance(point, ExploratoryC53K9Point):
        raise TypeError("point must be an ExploratoryC53K9Point")
    amount = _finite_real(delta, "delta")
    base = point.base_point
    if parameter_id == f"c396:{D_MU_Q_SQ}":
        varied = replace(
            point,
            base_point=replace(base, mu_q_sq_GeV2=base.mu_q_sq_GeV2 + amount),
        )
    elif parameter_id == f"c396:{D_DELTA_MU_G_SQ}":
        varied = replace(
            point,
            base_point=replace(
                base,
                delta_mu_g_sq_GeV2=base.delta_mu_g_sq_GeV2 + amount,
            ),
        )
    elif parameter_id == _C117_PARAMETER_ID:
        varied = replace(
            point,
            base_point=replace(
                base, c117_coefficient=base.c117_coefficient + amount
            ),
        )
    elif parameter_id == C53_PARAMETER_ID:
        varied = replace(point, c53_coupling=point.c53_coupling + amount)
    else:
        raise KeyError(parameter_id)
    if point_id is not None:
        varied = replace(varied, point_id=point_id)
    return varied


def helicity_derivative_report(
    pair: C53MixedK9HelicityPair,
    *,
    finite_difference_step: float = 1.0e-6,
) -> Mapping[str, Any]:
    """Compare branch-resolved Hellmann--Feynman and finite differences."""

    step = _finite_real(finite_difference_step, "finite_difference_step")
    if step <= 0.0:
        raise ValueError("finite_difference_step must be positive")
    rows = []
    for parameter_id in pair.assembly.parameter_ids:
        operator = pair.assembly.term_matrix(parameter_id)
        plus = solve_c53_mixed_k9_helicity_pair(
            build_c53_mixed_k9_hamiltonian(
                vary_c53_mixed_k9_point(pair.assembly.point, parameter_id, step)
            )
        )
        minus = solve_c53_mixed_k9_helicity_pair(
            build_c53_mixed_k9_hamiltonian(
                vary_c53_mixed_k9_point(pair.assembly.point, parameter_id, -step)
            )
        )
        branches = []
        for helicity_twice in _HELICITIES_TWICE:
            state = pair.state(helicity_twice).state
            hf = float(np.vdot(state, operator @ state).real)
            fd = (
                plus.state(helicity_twice).energy_GeV2
                - minus.state(helicity_twice).energy_GeV2
            ) / (2.0 * step)
            branches.append(
                {
                    "helicity_twice": helicity_twice,
                    "hellmann_feynman": hf,
                    "finite_difference": float(fd),
                    "HF_minus_FD_abs": abs(hf - fd),
                }
            )
        values = [branch["hellmann_feynman"] for branch in branches]
        rows.append(
            {
                "parameter_id": parameter_id,
                "branches": tuple(branches),
                "helicity_average": float(np.mean(values)),
                "helicity_spread": float(max(values) - min(values)),
            }
        )
    return {
        "finite_difference_step": step,
        "rows": tuple(rows),
        "max_HF_FD_abs": max(
            branch["HF_minus_FD_abs"]
            for row in rows
            for branch in row["branches"]
        ),
        "helicity_branches_tracked_by": "EXACT_JZ_SECTORS_NOT_SOLVER_VECTOR_ORDER",
        "physical_parameter_inference": False,
    }


def c53_coupling_scan(
    point: ExploratoryC53K9Point = C53_MIXED_K9_EXPLORATORY_BASELINE,
    couplings: Sequence[float] = (0.0, 0.01, 0.03, 0.05),
) -> Mapping[str, Any]:
    """Track the zero-coupling limit and qg support at named diagnostic values."""

    rows = []
    for coupling in couplings:
        value = _finite_real(coupling, "coupling")
        study = solve_c53_mixed_k9_helicity_pair(
            build_c53_mixed_k9_hamiltonian(replace(point, c53_coupling=value))
        )
        rows.append(
            {
                "c53_coupling": value,
                "energies_GeV2": tuple(state.energy_GeV2 for state in study.states),
                "qg_weights": tuple(state.qg_weight for state in study.states),
                "helicity_splitting_GeV2": study.helicity_splitting_GeV2,
                "gap_after_pair_GeV2": study.gap_after_pair_GeV2,
            }
        )
    return {
        "rows": tuple(rows),
        "zero_coupling_required": True,
        "physical_coupling_selected": False,
    }


def c117_response_report(
    pair: C53MixedK9HelicityPair,
) -> Mapping[str, Any]:
    """Show how C53-generated qg support changes the lawful C117 response."""

    operator = pair.assembly.term_matrix(_C117_PARAMETER_ID)
    basis = pair.basis
    projected = np.asarray(basis.conj().T @ (operator @ basis), dtype=np.complex128)
    slopes = np.linalg.eigvalsh(projected)
    outside = np.eye(pair.assembly.dimension, dtype=np.complex128) - pair.projector
    leakage = tuple(
        float(np.linalg.norm(outside @ (operator @ basis[:, column])))
        for column in range(basis.shape[1])
    )
    zero_pair = solve_c53_mixed_k9_helicity_pair(
        build_c53_mixed_k9_hamiltonian(
            replace(pair.assembly.point, c53_coupling=0.0)
        )
    )
    zero_basis = zero_pair.basis
    zero_projected = np.asarray(
        zero_basis.conj().T @ (operator @ zero_basis), dtype=np.complex128
    )
    zero_slopes = np.linalg.eigvalsh(zero_projected)
    return {
        "parameter_id": _C117_PARAMETER_ID,
        "status": "MAPPED_JZ_CONSERVING_EXPLORATORY_HAMILTONIAN_TERM",
        "jz_commutator_max_abs": pair.assembly.validation[
            "c117_jz_commutator_max_abs"
        ],
        "projected_pair_matrix": projected,
        "projected_first_order_slopes": tuple(float(value) for value in slopes),
        "outside_lowest_pair_action_norms": leakage,
        "zero_c53_projected_max_abs": float(np.max(np.abs(zero_projected))),
        "zero_c53_q_sector_first_order_slopes": tuple(
            float(value) for value in zero_slopes
        ),
        "qg_induced_first_order_slope_shift": tuple(
            float(value - baseline) for value, baseline in zip(slopes, zero_slopes)
        ),
        "nonzero_response_requires_qg_support": False,
        "q_sector_response_already_present": True,
        "qg_support_changes_c117_response": True,
        "finite_exploratory_coefficient_authorized": True,
        "physical_coefficient_authorized": False,
        "physical": False,
    }


def c53_mixed_state_record(
    pair: C53MixedK9HelicityPair,
    *,
    include_derivatives: bool = True,
) -> Mapping[str, Any]:
    payload = {
        "status": "M2_C53_MIXED_COLOR_SINGLET_K9_HELICITY_PAIR_READY",
        "point": pair.assembly.point.record(),
        "dimension": pair.assembly.dimension,
        "helicity_states": tuple(
            {
                "helicity_twice": state.helicity_twice,
                "energy_GeV2": state.energy_GeV2,
                "residual_norm_GeV2": state.residual_norm_GeV2,
                "q_weight": state.q_weight,
                "qg_weight": state.qg_weight,
                "sector_gap_GeV2": state.sector_gap_GeV2,
            }
            for state in pair.states
        ),
        "helicity_splitting_GeV2": pair.helicity_splitting_GeV2,
        "gap_after_pair_GeV2": pair.gap_after_pair_GeV2,
        "sparse_matrix_free_max_abs": pair.sparse_matrix_free_max_abs,
        "assembly_validation": dict(pair.assembly.validation),
        "c117_response": c117_response_report(pair),
        "color_singlet_completed": True,
        "spectator_spin_flavor_charge_constructed": False,
        "deuteron_H_D_constructed": False,
        "current_constructed": False,
        "claim_tier": pair.claim_tier,
        "physical": pair.physical,
    }
    if include_derivatives:
        payload["derivatives"] = helicity_derivative_report(pair)
    return payload


__all__ = [
    "K9",
    "C53_MIXED_CLAIM_TIER",
    "C53_PARAMETER_ID",
    "ExploratoryC53K9Point",
    "C53_MIXED_K9_EXPLORATORY_BASELINE",
    "ReducedOperatorTerm",
    "C53MixedK9Hamiltonian",
    "C53MixedHelicityState",
    "C53MixedK9HelicityPair",
    "build_c53_mixed_k9_hamiltonian",
    "solve_c53_mixed_k9_helicity_pair",
    "vary_c53_mixed_k9_point",
    "helicity_derivative_report",
    "c53_coupling_scan",
    "c117_response_report",
    "c53_mixed_state_record",
]

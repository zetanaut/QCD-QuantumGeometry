"""Explicit exploratory Hamiltonian bundles for the first K-local response.

This module is the main-line seam between the modern finite-basis operators and
the conditional quantum backends.  It deliberately requires the caller to
provide ``H0`` and all coefficients.  In particular, it never promotes a C144
diagnostic fixture to a C396 free Hamiltonian and never supplies a physical
default.

The bundle is useful in Lane A (exploratory) and Lane B (validated-model)
work.  It is not a physical deuteron Hamiltonian: the first C117 normalization
and mixing remain caller-owned exploratory parameters, the physical sector is
not selected, and current operators are not inferred from the Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh

from deuteron_wigner.bridge.c401_c396_mass_directions import (
    D_DELTA_MU_G_SQ,
    D_MU_Q_SQ,
    DIRECTIONS,
    coordinate_operator_csr,
    resolution_record,
)
from deuteron_wigner.bridge.c411_c117_i2_finite_c43_adapter import (
    ExploratoryC1171Parameters,
    exploratory_c117_1_csr,
    exploratory_parameter_record,
)
from deuteron_wigner.microscopic.h0.k_local import (
    KLocalH0Supply,
    k_local_h0_record,
)


CLAIM_TIER = "EXPLORATORY"
PARAMETER_IDS = (
    f"c396:{D_MU_Q_SQ}",
    f"c396:{D_DELTA_MU_G_SQ}",
    "c117:I2_density_projector",
)
_C117_PARAMETER_ID = PARAMETER_IDS[-1]


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _finite_vector(vector: Sequence[Any], dimension: int, name: str = "vector") -> np.ndarray:
    values = np.asarray(vector, dtype=np.complex128)
    if values.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains nonfinite entries")
    return values


def _matrix_or_operator(value: Any, dimension: int, name: str) -> sparse.csr_matrix | LinearOperator:
    if sparse.issparse(value):
        matrix = value.tocsr().astype(np.complex128)
        if matrix.shape != (dimension, dimension):
            raise ValueError(f"{name} must have shape {(dimension, dimension)}, got {matrix.shape}")
        if matrix.data.size and not np.all(np.isfinite(matrix.data)):
            raise ValueError(f"{name} contains nonfinite entries")
        return matrix
    if isinstance(value, LinearOperator):
        if value.shape != (dimension, dimension):
            raise ValueError(f"{name} must have shape {(dimension, dimension)}, got {value.shape}")
        return value
    try:
        matrix = sparse.csr_matrix(np.asarray(value, dtype=np.complex128))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be sparse, a LinearOperator, or array-like") from exc
    return _matrix_or_operator(matrix, dimension, name)


def _is_sparse(value: Any) -> bool:
    return sparse.issparse(value)


def _hermiticity_residual(value: sparse.spmatrix) -> float:
    defect = (value - value.getH()).tocsr()
    return float(np.max(np.abs(defect.data))) if defect.nnz else 0.0


def _root(value: Any) -> str:
    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): plain(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(val) for val in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, complex):
            return [float(item.real), float(item.imag)]
        return item

    return sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class OperatorTerm:
    """One explicitly owned operator direction in a bundle."""

    parameter_id: str
    coefficient: float
    matrix: sparse.csr_matrix
    source: str
    coefficient_units: str
    operator_units: str
    claim_tier: str = CLAIM_TIER


@dataclass(frozen=True)
class ExploratoryHamiltonian:
    """A K-local Hamiltonian with explicit exploratory provenance."""

    resolution: str
    h0: sparse.csr_matrix | LinearOperator
    h0_source: str
    h0_hermitian_certified: bool
    terms: tuple[OperatorTerm, ...]
    c117_parameters: ExploratoryC1171Parameters
    c117_coefficient: float
    claim_tier: str = CLAIM_TIER
    physical: bool = False
    h0_contract: Mapping[str, Any] | None = None

    @property
    def dimension(self) -> int:
        return int(resolution_record(self.resolution)["direct_sum_dimension"])

    @property
    def parameter_ids(self) -> tuple[str, ...]:
        return tuple(term.parameter_id for term in self.terms)

    @property
    def h0_is_sparse(self) -> bool:
        return _is_sparse(self.h0)

    def matrix(self) -> sparse.csr_matrix:
        """Materialize the sparse Hamiltonian when the supplied H0 permits it."""

        if not self.h0_is_sparse:
            raise TypeError("the supplied H0 is matrix-free; no sparse matrix is available")
        result = self.h0.tocsr().astype(np.complex128)
        for term in self.terms:
            result = (result + term.coefficient * term.matrix).tocsr()
        result.eliminate_zeros()
        return result

    def apply(self, vector: Sequence[Any]) -> np.ndarray:
        values = _finite_vector(vector, self.dimension)
        result = np.asarray(self.h0 @ values, dtype=np.complex128)
        for term in self.terms:
            result += term.coefficient * (term.matrix @ values)
        return result

    def linear_operator(self) -> LinearOperator:
        return LinearOperator(
            (self.dimension, self.dimension),
            matvec=self.apply,
            rmatvec=self.apply,
            dtype=np.complex128,
        )

    def term(self, parameter_id: str) -> OperatorTerm:
        for term in self.terms:
            if term.parameter_id == parameter_id:
                return term
        raise KeyError(parameter_id)

    def with_parameter(self, parameter_id: str, coefficient: float) -> "ExploratoryHamiltonian":
        value = _finite_real(coefficient, "coefficient")
        if parameter_id not in self.parameter_ids:
            raise KeyError(parameter_id)
        replacement = tuple(
            OperatorTerm(
                term.parameter_id,
                value if term.parameter_id == parameter_id else term.coefficient,
                term.matrix,
                term.source,
                term.coefficient_units,
                term.operator_units,
                term.claim_tier,
            )
            for term in self.terms
        )
        c117 = value if parameter_id == _C117_PARAMETER_ID else self.c117_coefficient
        return ExploratoryHamiltonian(
            resolution=self.resolution,
            h0=self.h0,
            h0_source=self.h0_source,
            h0_hermitian_certified=self.h0_hermitian_certified,
            terms=replacement,
            c117_parameters=self.c117_parameters,
            c117_coefficient=c117,
            claim_tier=self.claim_tier,
            physical=self.physical,
            h0_contract=self.h0_contract,
        )


@dataclass(frozen=True)
class ExactGroundState:
    resolution: str
    energy: float
    state: np.ndarray
    residual_norm: float
    q_weight: float
    qg_weight: float
    claim_tier: str = CLAIM_TIER
    physical: bool = False


def build_exploratory_hamiltonian(
    resolution: str,
    *,
    h0: Any,
    h0_source: str,
    c396_coefficients: Mapping[str, Any],
    c117_parameters: ExploratoryC1171Parameters,
    c117_coefficient: Any,
    h0_hermitian_certified: bool | None = None,
) -> ExploratoryHamiltonian:
    """Build an explicit C396+C117 exploratory Hamiltonian.

    ``h0`` is mandatory and may be sparse or matrix-free.  The two C396 mass
    coefficients and the C117 coefficient are also mandatory; omitted terms
    are rejected rather than silently interpreted as zero.  A matrix-free
    H0 requires the caller to certify Hermiticity because it cannot be checked
    from entries here.
    """

    record = resolution_record(resolution)
    if not isinstance(h0_source, str) or not h0_source.strip():
        raise ValueError("h0_source must identify the caller-supplied H0")
    h0_value = _matrix_or_operator(h0, int(record["direct_sum_dimension"]), "h0")
    if h0_hermitian_certified is None:
        h0_hermitian_certified = _is_sparse(h0_value)
    if not h0_hermitian_certified:
        raise ValueError("a matrix-free H0 requires h0_hermitian_certified=True")
    if _is_sparse(h0_value) and _hermiticity_residual(h0_value) != 0.0:
        raise ValueError("h0 must be exactly Hermitian")
    if not isinstance(c396_coefficients, Mapping):
        raise TypeError("c396_coefficients must be an explicit mapping")
    expected = set(DIRECTIONS)
    received = set(c396_coefficients)
    if received != expected:
        raise ValueError(f"c396_coefficients must contain exactly {tuple(DIRECTIONS)}")
    c117_value = _finite_real(c117_coefficient, "c117_coefficient")
    if c117_parameters.resolution != resolution:
        raise ValueError("C117 parameters and Hamiltonian resolution disagree")

    terms = (
        OperatorTerm(
            f"c396:{D_MU_Q_SQ}",
            _finite_real(c396_coefficients[D_MU_Q_SQ], f"coefficient {D_MU_Q_SQ}"),
            coordinate_operator_csr(resolution, D_MU_Q_SQ).astype(np.complex128),
            "C401 C396 quark mass-squared direction",
            "GeV^2",
            "dimensionless",
        ),
        OperatorTerm(
            f"c396:{D_DELTA_MU_G_SQ}",
            _finite_real(c396_coefficients[D_DELTA_MU_G_SQ], f"coefficient {D_DELTA_MU_G_SQ}"),
            coordinate_operator_csr(resolution, D_DELTA_MU_G_SQ).astype(np.complex128),
            "C401 C396 gluon mass-squared direction",
            "GeV^2",
            "dimensionless",
        ),
        OperatorTerm(
            _C117_PARAMETER_ID,
            c117_value,
            exploratory_c117_1_csr(c117_parameters).astype(np.complex128),
            "C411 Lane-A first C117 I2 exploratory action",
            "dimensionless",
            "GeV^2",
        ),
    )
    return ExploratoryHamiltonian(
        resolution,
        h0_value,
        h0_source,
        bool(h0_hermitian_certified),
        terms,
        c117_parameters,
        c117_value,
    )


def build_mapped_exploratory_hamiltonian(
    resolution: str,
    *,
    h0_supply: KLocalH0Supply,
    c396_coefficients: Mapping[str, Any],
    c117_parameters: ExploratoryC1171Parameters,
    c117_coefficient: Any,
) -> ExploratoryHamiltonian:
    """Bind a tested K-local H0 supply to the exploratory operator bundle.

    This is the M2 contract route.  It accepts only an explicitly nonphysical
    H0 whose source operator, basis map, embedded target operator, units, and
    omitted-sector treatment have already passed the K-local validation.  The
    mass terms remain owned by the two explicit C401/C396 coefficients.
    """

    if not isinstance(h0_supply, KLocalH0Supply):
        raise TypeError("h0_supply must be an explicit KLocalH0Supply")
    record = resolution_record(resolution)
    if h0_supply.resolution != record["resolution_label"]:
        raise ValueError("H0 supply and Hamiltonian resolution disagree")
    if h0_supply.dimension != int(record["direct_sum_dimension"]):
        raise ValueError("H0 supply and C401/C410 target dimensions disagree")
    if h0_supply.claim_tier != CLAIM_TIER or h0_supply.physical:
        raise ValueError("the M2 bundle accepts only an exploratory nonphysical H0 supply")
    if h0_supply.basis_map.claim_tier != CLAIM_TIER or h0_supply.basis_map.physical:
        raise ValueError("the H0 basis map must retain the exploratory claim tier")
    if (
        h0_supply.basis_map.source_units != "GeV^2"
        or h0_supply.basis_map.target_units != "GeV^2"
    ):
        raise ValueError("the H0 source and target operators must use GeV^2")
    if not bool(h0_supply.validation.get("pass")):
        raise ValueError("the H0 validation record is not positive")
    if bool(h0_supply.validation.get("mass_terms_in_h0", True)):
        raise ValueError("H0 must exclude mass terms owned by the C401/C396 directions")
    mapped = h0_supply.basis_map.embed_operator(h0_supply.source_operator)
    defect = (mapped - h0_supply.target_operator).tocsr()
    if _hermiticity_residual(h0_supply.target_operator) != 0.0 or defect.nnz:
        raise ValueError("the supplied H0 is not the exact Hermitian basis-map image")

    bundle = build_exploratory_hamiltonian(
        resolution,
        h0=h0_supply.target_operator,
        h0_source=h0_supply.source_id,
        c396_coefficients=c396_coefficients,
        c117_parameters=c117_parameters,
        c117_coefficient=c117_coefficient,
        h0_hermitian_certified=True,
    )
    # C410/C411 owns a C128/C112 q block followed by a native C47-order qg
    # block.  The mapped M2 Hamiltonian uses the C401/C410 target order, whose
    # transverse qg sub-order is the retained C128 order.  Equal dimensions do
    # not identify these bases: consume the same exact label permutation used
    # for H0 before adding C117 to the mapped bundle.
    embedding = h0_supply.basis_map.embedding
    mapped_terms = tuple(
        replace(
            term,
            matrix=(embedding @ term.matrix @ embedding.getH()).tocsr(),
            source=(
                term.source
                + "; exact C47-qg to C401/C410-target basis permutation applied"
            ),
        )
        if term.parameter_id == _C117_PARAMETER_ID
        else term
        for term in bundle.terms
    )
    contract = dict(k_local_h0_record(h0_supply))
    contract["c117_source_basis"] = "C128/C112 q plus C47 shell-major qg"
    contract["c117_target_basis"] = h0_supply.basis_map.target_basis_id
    contract["c117_basis_permutation_applied"] = True
    return replace(bundle, terms=mapped_terms, h0_contract=contract)


def bundle_record(bundle: ExploratoryHamiltonian) -> Mapping[str, Any]:
    """Return a compact provenance record suitable for handoff artifacts."""

    basis = resolution_record(bundle.resolution)
    c117 = exploratory_parameter_record(bundle.c117_parameters)
    terms = tuple(
        {
            "parameter_id": term.parameter_id,
            "coefficient": term.coefficient,
            "source": term.source,
            "coefficient_units": term.coefficient_units,
            "operator_units": term.operator_units,
            "shape": term.matrix.shape,
            "nnz": int(term.matrix.nnz),
            "hermiticity_residual": _hermiticity_residual(term.matrix),
            "claim_tier": term.claim_tier,
        }
        for term in bundle.terms
    )
    payload = {
        "schema": "MAINLINE-EXPLORATORY-OPERATOR-BUNDLE-V1",
        "resolution": bundle.resolution,
        "dimension": bundle.dimension,
        "basis_order": basis["basis_order"],
        "h0_source": bundle.h0_source,
        "h0_sparse": bundle.h0_is_sparse,
        "h0_hermitian_certified": bundle.h0_hermitian_certified,
        "h0_basis_map_supplied": bundle.h0_contract is not None,
        "h0_basis_map_id": (
            bundle.h0_contract.get("basis_map_id") if bundle.h0_contract is not None else None
        ),
        "h0_operator_units": (
            bundle.h0_contract.get("operator_units") if bundle.h0_contract is not None else None
        ),
        "h0_mass_terms_included": (
            bundle.h0_contract.get("mass_terms_in_h0") if bundle.h0_contract is not None else None
        ),
        "h0_omitted_sector_treatment": (
            bundle.h0_contract.get("omitted_sector_treatment")
            if bundle.h0_contract is not None
            else None
        ),
        "h0_basis_ordering": (
            bundle.h0_contract.get("basis_ordering") if bundle.h0_contract is not None else None
        ),
        "h0_normalization_ownership": (
            bundle.h0_contract.get("normalization_ownership")
            if bundle.h0_contract is not None
            else None
        ),
        "c117_source_basis": (
            bundle.h0_contract.get("c117_source_basis")
            if bundle.h0_contract is not None
            else None
        ),
        "c117_target_basis": (
            bundle.h0_contract.get("c117_target_basis")
            if bundle.h0_contract is not None
            else None
        ),
        "c117_basis_permutation_applied": (
            bundle.h0_contract.get("c117_basis_permutation_applied")
            if bundle.h0_contract is not None
            else None
        ),
        "h0_claim_tier": (
            bundle.h0_contract.get("claim_tier") if bundle.h0_contract is not None else None
        ),
        "h0_physical": (
            bundle.h0_contract.get("physical") if bundle.h0_contract is not None else None
        ),
        "terms": terms,
        "c117_parameter_root": c117["root"],
        "unavailable_C117_source_directions": c117["unavailable_source_directions"],
        "Pminus_to_M2_applied": c117["Pminus_to_M2"]["applied_to_C410_shape"],
        "claim_tier": bundle.claim_tier,
        "physical": bundle.physical,
        "physical_fit_authorized": False,
        "hamiltonian_activation": False,
    }
    return {**payload, "root": _root(payload)}


def exact_ground_state(bundle: ExploratoryHamiltonian) -> ExactGroundState:
    """Compute the lowest state through the sparse/matrix-free exact oracle."""

    initial = np.ones(bundle.dimension, dtype=np.complex128)
    initial /= np.linalg.norm(initial)
    values, vectors = eigsh(
        bundle.matrix() if bundle.h0_is_sparse else bundle.linear_operator(),
        k=1,
        which="SA",
        v0=initial,
        tol=1.0e-11,
        maxiter=20000,
    )
    state = np.asarray(vectors[:, 0], dtype=np.complex128)
    pivot = int(np.argmax(np.abs(state)))
    if abs(state[pivot]) > 0.0:
        state = state * np.conjugate(state[pivot]) / abs(state[pivot])
    energy = float(np.real(values[0]))
    residual = float(np.linalg.norm(bundle.apply(state) - energy * state))
    q_dimension = int(resolution_record(bundle.resolution)["q_dimension"])
    q_weight = float(np.vdot(state[:q_dimension], state[:q_dimension]).real)
    qg_weight = float(np.vdot(state[q_dimension:], state[q_dimension:]).real)
    return ExactGroundState(bundle.resolution, energy, state, residual, q_weight, qg_weight)


def expectation(matrix: sparse.spmatrix | LinearOperator, state: Sequence[Any]) -> complex:
    """Evaluate a supplied observable without changing or renormalizing state."""

    values = np.asarray(state, dtype=np.complex128)
    if values.ndim != 1:
        raise ValueError("state must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("state contains nonfinite entries")
    if matrix.shape != (values.size, values.size):
        raise ValueError("observable and state dimensions disagree")
    return complex(np.vdot(values, matrix @ values))


def exploratory_response_map(
    bundle: ExploratoryHamiltonian,
    observables: Mapping[str, sparse.spmatrix | LinearOperator],
    *,
    finite_difference_step: float = 1.0e-5,
) -> Mapping[str, Any]:
    """Map explicit coefficients to state observables and sensitivity.

    The returned Jacobian is a diagnostic response matrix.  Its singular
    spectrum is a sensitivity diagnostic only; it is not the physical C396
    rank and does not authorize fitting or activation.
    """

    step = _finite_real(finite_difference_step, "finite_difference_step")
    if step <= 0.0:
        raise ValueError("finite_difference_step must be positive")
    if not isinstance(observables, Mapping) or not observables:
        raise ValueError("at least one explicit observable is required")
    state = exact_ground_state(bundle)
    checked: dict[str, sparse.spmatrix | LinearOperator] = {}
    for name, matrix in observables.items():
        if not isinstance(name, str) or not name:
            raise ValueError("observable names must be nonempty strings")
        if matrix.shape != (bundle.dimension, bundle.dimension):
            raise ValueError(f"observable {name!r} has the wrong shape")
        if _is_sparse(matrix) and _hermiticity_residual(matrix) != 0.0:
            raise ValueError(f"observable {name!r} must be Hermitian")
        checked[name] = matrix

    observable_values = {
        name: float(np.real(expectation(matrix, state.state)))
        for name, matrix in checked.items()
    }
    rows = ("energy", *tuple(checked))
    columns = bundle.parameter_ids
    jacobian = np.zeros((len(rows), len(columns)), dtype=float)
    hf_energy = {}
    energy_fd = {}
    for column_index, parameter_id in enumerate(columns):
        term = bundle.term(parameter_id)
        hf = float(np.real(expectation(term.matrix, state.state)))
        hf_energy[parameter_id] = hf
        plus = bundle.with_parameter(parameter_id, term.coefficient + step)
        minus = bundle.with_parameter(parameter_id, term.coefficient - step)
        plus_state = exact_ground_state(plus)
        minus_state = exact_ground_state(minus)
        energy_derivative = (plus_state.energy - minus_state.energy) / (2.0 * step)
        energy_fd[parameter_id] = energy_derivative
        jacobian[0, column_index] = energy_derivative
        for row_index, name in enumerate(checked, start=1):
            plus_value = float(np.real(expectation(checked[name], plus_state.state)))
            minus_value = float(np.real(expectation(checked[name], minus_state.state)))
            jacobian[row_index, column_index] = (plus_value - minus_value) / (2.0 * step)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    rank_tolerance = max(float(singular_values[0]) * 1.0e-10, 1.0e-12) if singular_values.size else 1.0e-12
    diagnostic_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    return {
        "schema": "MAINLINE-EXPLORATORY-K9-RESPONSE-MAP-V1",
        "resolution": bundle.resolution,
        "claim_tier": CLAIM_TIER,
        "physical": False,
        "parameter_ids": columns,
        "observable_ids": tuple(checked),
        "energy": state.energy,
        "state_residual_norm": state.residual_norm,
        "sector_weights": {"q": state.q_weight, "qg": state.qg_weight},
        "observable_values": observable_values,
        "hellmann_feynman_energy_derivative": hf_energy,
        "finite_difference_energy_derivative": energy_fd,
        "energy_hf_fd_max_abs_residual": max(abs(hf_energy[key] - energy_fd[key]) for key in columns),
        "jacobian": jacobian.tolist(),
        "singular_values": singular_values.tolist(),
        "diagnostic_sensitivity_rank": diagnostic_rank,
        "rank_is_physical": False,
        "current_response": "NOT_PROVIDED",
        "physical_fit_authorized": False,
        "hamiltonian_activation": False,
        "bundle_root": bundle_record(bundle)["root"],
        "root": _root((bundle_record(bundle), rows, columns, jacobian.tolist(), observable_values)),
    }


__all__ = [
    "CLAIM_TIER",
    "PARAMETER_IDS",
    "OperatorTerm",
    "ExploratoryHamiltonian",
    "ExactGroundState",
    "build_exploratory_hamiltonian",
    "build_mapped_exploratory_hamiltonian",
    "bundle_record",
    "exact_ground_state",
    "expectation",
    "exploratory_response_map",
]

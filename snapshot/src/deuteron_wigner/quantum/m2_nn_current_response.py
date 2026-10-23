"""M2g response of the fixed NN pointlike current to microscopic directions.

The C396, C117, and C53 coefficients move the internal color-singlet K9
helicity pair.  The M2f pointlike one-body current, however, acts as the
identity on that normalized internal factor and consumes only the unchanged
nuclear wave grid, transfer fibers, and nucleon-current member.  This module
computes both sides of that statement: a gauge-aligned internal-state tangent
matrix and independently recomputed central finite differences of the current.

The resulting null current Jacobian is a diagnostic of the present factorized
current interface.  It is not a physical response rank and does not establish
that the omitted microscopic, exchange, pair, zero-mode, or nonnucleonic
currents vanish.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from typing import Any, Mapping

import numpy as np

from .m2_c53_mixed_singlet_state import (
    C53_PARAMETER_ID,
    C53MixedK9HelicityPair,
    build_c53_mixed_k9_hamiltonian,
    solve_c53_mixed_k9_helicity_pair,
    vary_c53_mixed_k9_point,
)
from .m2_nn_deuteron_state import (
    ExploratoryNNDeuteronState,
    compose_nn_deuteron_state,
)
from .m2_nn_transfer_current import (
    NN_POINTLIKE_CHARGE_CURRENT_V1,
    NNOneBodyCurrentMember,
    NNOneBodyDiagnosticCurrent,
    build_nn_transfer_fibers,
    evaluate_nn_one_body_diagnostic_current,
    nn_one_body_current_pair_record,
)


NN_CURRENT_RESPONSE_CLAIM_TIER = (
    "EXPLORATORY_FACTORIZED_POINTLIKE_MICROSCOPIC_RESPONSE_DIAGNOSTIC"
)
NN_CURRENT_RESPONSE_STATUS = (
    "M2_FACTORIZED_POINTLIKE_CURRENT_MICROSCOPIC_RESPONSE_NULL_READY"
)
HELICITY_CURRENT_IDS = ("I11", "I10", "I1m1", "I00")
EXPECTED_PARAMETER_IDS = (
    "c396:D_mu_q_sq",
    "c396:D_delta_mu_g_sq",
    "c117:I2_density_projector",
    C53_PARAMETER_ID,
)
DEFAULT_PARAMETER_SCALES = {
    "c396:D_mu_q_sq": 0.05,
    "c396:D_delta_mu_g_sq": 0.05,
    "c117:I2_density_projector": 0.03,
    C53_PARAMETER_ID: 0.01,
}
PARAMETER_UNITS = {
    "c396:D_mu_q_sq": "GeV^2",
    "c396:D_delta_mu_g_sq": "GeV^2",
    "c117:I2_density_projector": "dimensionless",
    C53_PARAMETER_ID: "dimensionless",
}


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _amplitude_vector(current: NNOneBodyDiagnosticCurrent) -> np.ndarray:
    spin = current.spin_one_plus_current()
    return np.asarray(
        (spin.plus_plus, spin.plus_zero, spin.plus_minus, spin.zero_zero),
        dtype=np.complex128,
    )


def _phase_aligned_basis(
    candidate: C53MixedK9HelicityPair,
    reference_basis: np.ndarray,
) -> np.ndarray:
    """Align each exact-Jz branch to the baseline parallel-transport gauge."""

    result = np.asarray(candidate.basis, dtype=np.complex128).copy()
    for column in range(result.shape[1]):
        overlap = complex(np.vdot(reference_basis[:, column], result[:, column]))
        if abs(overlap) < 0.5:
            raise RuntimeError("finite-difference state left the tracked helicity branch")
        result[:, column] *= np.conjugate(overlap) / abs(overlap)
    return result


def _real_embedding(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.complex128)
    return np.vstack((values.real, values.imag))


def _diagnostic_rank(singular_values: np.ndarray) -> int:
    values = np.asarray(singular_values, dtype=float)
    if values.size == 0:
        return 0
    tolerance = max(float(values[0]) * 1.0e-9, 1.0e-13)
    return int(np.count_nonzero(values > tolerance))


@dataclass(frozen=True)
class NNMicroscopicCurrentResponse:
    """Internal-state tangents and pointlike-current response at one M2 point."""

    state: ExploratoryNNDeuteronState
    current_member: NNOneBodyCurrentMember
    baseline_current: NNOneBodyDiagnosticCurrent
    baseline_reversal_record: Mapping[str, Any]
    finite_difference_step: float
    parameter_ids: tuple[str, ...]
    parameter_scales: Mapping[str, float]
    parameter_units: Mapping[str, str]
    finite_difference_current_response: np.ndarray
    analytic_identity_current_response: np.ndarray
    full_target_matrix_response: np.ndarray
    component_response_max_abs: Mapping[str, np.ndarray]
    angular_condition_response: np.ndarray
    internal_state_tangent_matrix: np.ndarray
    state_tangent_norms: np.ndarray
    internal_gram_derivative_residuals: np.ndarray
    internal_gram_finite_difference_residuals: np.ndarray
    energy_response_by_helicity: np.ndarray
    qg_weight_response_by_helicity: np.ndarray
    claim_tier: str = NN_CURRENT_RESPONSE_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, ExploratoryNNDeuteronState):
            raise TypeError("state must be an ExploratoryNNDeuteronState")
        if self.current_member != NN_POINTLIKE_CHARGE_CURRENT_V1:
            raise ValueError("M2g is restricted to the fixed pointlike current member")
        if self.parameter_ids != EXPECTED_PARAMETER_IDS:
            raise ValueError("M2g requires the owned C396/C117/C53 parameter order")
        if set(self.parameter_scales) != set(self.parameter_ids):
            raise ValueError("every response parameter requires one declared scale")
        if set(self.parameter_units) != set(self.parameter_ids):
            raise ValueError("every response parameter requires declared units")
        if self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")
        parameter_count = len(self.parameter_ids)
        expected_shapes = {
            "finite_difference_current_response": (len(HELICITY_CURRENT_IDS), parameter_count),
            "analytic_identity_current_response": (len(HELICITY_CURRENT_IDS), parameter_count),
            "full_target_matrix_response": (parameter_count, 3, 3),
            "angular_condition_response": (parameter_count,),
            "internal_state_tangent_matrix": (
                2 * self.state.pair.assembly.dimension,
                parameter_count,
            ),
            "state_tangent_norms": (2, parameter_count),
            "internal_gram_derivative_residuals": (parameter_count,),
            "internal_gram_finite_difference_residuals": (parameter_count,),
            "energy_response_by_helicity": (2, parameter_count),
            "qg_weight_response_by_helicity": (2, parameter_count),
        }
        for name, shape in expected_shapes.items():
            values = np.asarray(getattr(self, name))
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains nonfinite values")
        if set(self.component_response_max_abs) != {"SS", "SD", "DS", "DD"}:
            raise ValueError("component response must retain SS, SD, DS, and DD")
        for values in self.component_response_max_abs.values():
            if np.asarray(values).shape != (parameter_count,):
                raise ValueError("component response vectors have the wrong shape")
        if self.claim_tier != NN_CURRENT_RESPONSE_CLAIM_TIER or self.physical:
            raise ValueError("M2g must remain an exploratory response diagnostic")

    @property
    def standardized_current_response(self) -> np.ndarray:
        scales = np.asarray(
            [self.parameter_scales[name] for name in self.parameter_ids],
            dtype=float,
        )
        return np.asarray(self.finite_difference_current_response) * scales[None, :]

    @property
    def current_response_singular_values(self) -> np.ndarray:
        return np.linalg.svd(
            _real_embedding(self.standardized_current_response),
            compute_uv=False,
        )

    @property
    def diagnostic_current_response_rank(self) -> int:
        return _diagnostic_rank(self.current_response_singular_values)

    @property
    def standardized_internal_state_tangent(self) -> np.ndarray:
        scales = np.asarray(
            [self.parameter_scales[name] for name in self.parameter_ids],
            dtype=float,
        )
        return np.asarray(self.internal_state_tangent_matrix) * scales[None, :]

    @property
    def internal_state_tangent_singular_values(self) -> np.ndarray:
        return np.linalg.svd(
            _real_embedding(self.standardized_internal_state_tangent),
            compute_uv=False,
        )

    @property
    def diagnostic_internal_state_tangent_rank(self) -> int:
        return _diagnostic_rank(self.internal_state_tangent_singular_values)

    @property
    def central_difference_consistency_residual(self) -> float:
        return float(
            np.max(
                np.abs(
                    self.finite_difference_current_response
                    - self.analytic_identity_current_response
                )
            )
        )


def evaluate_nn_microscopic_current_response(
    state: ExploratoryNNDeuteronState,
    *,
    delta_x_GeV: float = 0.2,
    delta_y_GeV: float = 0.0,
    p_plus_GeV: float = 2.0,
    reference_deuteron_mass_GeV: float = 1.87561294257,
    current_member: NNOneBodyCurrentMember = NN_POINTLIKE_CHARGE_CURRENT_V1,
    finite_difference_step: float = 1.0e-6,
) -> NNMicroscopicCurrentResponse:
    """Recompute the fixed current at both endpoints of all four directions."""

    if not isinstance(state, ExploratoryNNDeuteronState):
        raise TypeError("state must be an ExploratoryNNDeuteronState")
    if current_member != NN_POINTLIKE_CHARGE_CURRENT_V1:
        raise ValueError("M2g is restricted to the fixed pointlike current member")
    delta_x = _finite_real(delta_x_GeV, "delta_x_GeV")
    delta_y = _finite_real(delta_y_GeV, "delta_y_GeV")
    p_plus = _finite_real(p_plus_GeV, "p_plus_GeV")
    reference_mass = _finite_real(
        reference_deuteron_mass_GeV,
        "reference_deuteron_mass_GeV",
    )
    step = _finite_real(finite_difference_step, "finite_difference_step")
    if delta_x <= 0.0 or delta_y != 0.0:
        raise ValueError("M2g requires the positive x-directed diagnostic transfer")
    if p_plus <= 0.0 or reference_mass <= 0.0 or step <= 0.0:
        raise ValueError("plus momentum, reference mass, and finite-difference step must be positive")

    parameter_ids = tuple(state.pair.assembly.parameter_ids)
    if parameter_ids != EXPECTED_PARAMETER_IDS:
        raise ValueError("unexpected microscopic parameter order")

    def current_for(candidate: ExploratoryNNDeuteronState, signed_delta: float) -> NNOneBodyDiagnosticCurrent:
        transfer = build_nn_transfer_fibers(
            candidate,
            delta_x_GeV=signed_delta,
            delta_y_GeV=delta_y,
            p_plus_GeV=p_plus,
            reference_deuteron_mass_GeV=reference_mass,
        )
        return evaluate_nn_one_body_diagnostic_current(
            transfer,
            current_member=current_member,
        )

    baseline_current = current_for(state, delta_x)
    baseline_reversed = current_for(state, -delta_x)
    baseline_record = nn_one_body_current_pair_record(
        baseline_current,
        baseline_reversed,
    )
    baseline_basis = np.asarray(state.pair.basis, dtype=np.complex128)
    parameter_count = len(parameter_ids)
    amplitude_response = np.zeros(
        (len(HELICITY_CURRENT_IDS), parameter_count),
        dtype=np.complex128,
    )
    full_matrix_response = np.zeros(
        (parameter_count, 3, 3),
        dtype=np.complex128,
    )
    angular_response = np.zeros(parameter_count, dtype=np.complex128)
    state_tangents = np.zeros(
        (2 * state.pair.assembly.dimension, parameter_count),
        dtype=np.complex128,
    )
    state_tangent_norms = np.zeros((2, parameter_count), dtype=float)
    gram_derivative_residuals = np.zeros(parameter_count, dtype=float)
    gram_fd_residuals = np.zeros(parameter_count, dtype=float)
    energy_response = np.zeros((2, parameter_count), dtype=float)
    qg_response = np.zeros((2, parameter_count), dtype=float)
    component_response = {
        label: np.zeros(parameter_count, dtype=float)
        for label in ("SS", "SD", "DS", "DD")
    }

    for column, parameter_id in enumerate(parameter_ids):
        safe_id = parameter_id.replace(":", "_")
        plus_point = vary_c53_mixed_k9_point(
            state.pair.assembly.point,
            parameter_id,
            step,
            point_id=f"{state.pair.assembly.point.point_id}__{safe_id}__PLUS_FD",
        )
        minus_point = vary_c53_mixed_k9_point(
            state.pair.assembly.point,
            parameter_id,
            -step,
            point_id=f"{state.pair.assembly.point.point_id}__{safe_id}__MINUS_FD",
        )
        plus_pair = solve_c53_mixed_k9_helicity_pair(
            build_c53_mixed_k9_hamiltonian(plus_point)
        )
        minus_pair = solve_c53_mixed_k9_helicity_pair(
            build_c53_mixed_k9_hamiltonian(minus_point)
        )
        plus_state = compose_nn_deuteron_state(
            plus_pair,
            wave_grid=state.wave_grid,
            channel=state.channel,
        )
        minus_state = compose_nn_deuteron_state(
            minus_pair,
            wave_grid=state.wave_grid,
            channel=state.channel,
        )
        plus_current = current_for(plus_state, delta_x)
        minus_current = current_for(minus_state, delta_x)

        amplitude_response[:, column] = (
            _amplitude_vector(plus_current) - _amplitude_vector(minus_current)
        ) / (2.0 * step)
        full_matrix_response[column] = (
            plus_current.target_matrix - minus_current.target_matrix
        ) / (2.0 * step)
        angular_response[column] = (
            plus_current.spin_one_plus_current().angular_condition(
                plus_current.transfer.eta
            )
            - minus_current.spin_one_plus_current().angular_condition(
                minus_current.transfer.eta
            )
        ) / (2.0 * step)
        for label in component_response:
            derivative = (
                plus_current.component_matrices[label]
                - minus_current.component_matrices[label]
            ) / (2.0 * step)
            component_response[label][column] = float(np.max(np.abs(derivative)))

        plus_basis = _phase_aligned_basis(plus_pair, baseline_basis)
        minus_basis = _phase_aligned_basis(minus_pair, baseline_basis)
        tangent = (plus_basis - minus_basis) / (2.0 * step)
        state_tangents[:, column] = tangent.reshape(-1)
        state_tangent_norms[:, column] = np.linalg.norm(tangent, axis=0)
        gram_derivative = tangent.conj().T @ baseline_basis + baseline_basis.conj().T @ tangent
        gram_derivative_residuals[column] = float(np.max(np.abs(gram_derivative)))
        plus_gram = plus_basis.conj().T @ plus_basis
        minus_gram = minus_basis.conj().T @ minus_basis
        gram_fd_residuals[column] = float(
            np.max(np.abs((plus_gram - minus_gram) / (2.0 * step)))
        )
        for row, helicity_twice in enumerate((-1, 1)):
            energy_response[row, column] = (
                plus_pair.state(helicity_twice).energy_GeV2
                - minus_pair.state(helicity_twice).energy_GeV2
            ) / (2.0 * step)
            qg_response[row, column] = (
                plus_pair.state(helicity_twice).qg_weight
                - minus_pair.state(helicity_twice).qg_weight
            ) / (2.0 * step)

    analytic_response = np.zeros_like(amplitude_response)
    return NNMicroscopicCurrentResponse(
        state=state,
        current_member=current_member,
        baseline_current=baseline_current,
        baseline_reversal_record=baseline_record,
        finite_difference_step=step,
        parameter_ids=parameter_ids,
        parameter_scales=dict(DEFAULT_PARAMETER_SCALES),
        parameter_units=dict(PARAMETER_UNITS),
        finite_difference_current_response=amplitude_response,
        analytic_identity_current_response=analytic_response,
        full_target_matrix_response=full_matrix_response,
        component_response_max_abs=component_response,
        angular_condition_response=angular_response,
        internal_state_tangent_matrix=state_tangents,
        state_tangent_norms=state_tangent_norms,
        internal_gram_derivative_residuals=gram_derivative_residuals,
        internal_gram_finite_difference_residuals=gram_fd_residuals,
        energy_response_by_helicity=energy_response,
        qg_weight_response_by_helicity=qg_response,
    )


def nn_microscopic_current_response_record(
    response: NNMicroscopicCurrentResponse,
) -> Mapping[str, Any]:
    """Return a JSON-safe M2g response ledger with explicit nonclaims."""

    if not isinstance(response, NNMicroscopicCurrentResponse):
        raise TypeError("response must be an NNMicroscopicCurrentResponse")
    current_response = np.asarray(
        response.finite_difference_current_response,
        dtype=np.complex128,
    )
    angular_response = np.asarray(response.angular_condition_response)
    baseline_amplitudes = _amplitude_vector(response.baseline_current)
    baseline_angular = response.baseline_current.spin_one_plus_current().angular_condition(
        response.baseline_current.transfer.eta
    )
    payload = {
        "status": NN_CURRENT_RESPONSE_STATUS,
        "claim_tier": response.claim_tier,
        "physical": False,
        "point_id": response.state.pair.assembly.point.point_id,
        "hilbert_space_id": response.state.hilbert_space_id,
        "nucleon_family": response.state.channel,
        "current_member_id": response.current_member.member_id,
        "current_dependency_scope": (
            "NUCLEAR_WAVE_GRID",
            "TRANSFER_FIBERS",
            "NUCLEON_CURRENT_MEMBER",
            "NORMALIZED_INTERNAL_IDENTITY_FACTOR",
        ),
        "parameter_ids": response.parameter_ids,
        "parameter_units": tuple(
            response.parameter_units[name] for name in response.parameter_ids
        ),
        "parameter_scales": tuple(
            response.parameter_scales[name] for name in response.parameter_ids
        ),
        "parameter_scale_role": "DIMENSIONLESS_DIAGNOSTIC_COORDINATE_NOT_UNCERTAINTY",
        "finite_difference_step_in_each_native_parameter_unit": response.finite_difference_step,
        "finite_difference_endpoint_current_evaluations": 2 * len(response.parameter_ids),
        "helicity_current_ids": HELICITY_CURRENT_IDS,
        "baseline_helicity_current_real": tuple(float(value.real) for value in baseline_amplitudes),
        "baseline_helicity_current_imag": tuple(float(value.imag) for value in baseline_amplitudes),
        "finite_difference_current_response_real": tuple(
            tuple(float(value) for value in row) for row in current_response.real
        ),
        "finite_difference_current_response_imag": tuple(
            tuple(float(value) for value in row) for row in current_response.imag
        ),
        "analytic_identity_current_response_max_abs": float(
            np.max(np.abs(response.analytic_identity_current_response))
        ),
        "central_difference_consistency_residual": response.central_difference_consistency_residual,
        "full_target_matrix_response_max_abs": float(
            np.max(np.abs(response.full_target_matrix_response))
        ),
        "component_response_max_abs": {
            label: tuple(float(value) for value in values)
            for label, values in response.component_response_max_abs.items()
        },
        "baseline_angular_condition": {
            "real": float(baseline_angular.real),
            "imag": float(baseline_angular.imag),
        },
        "angular_condition_response_real": tuple(
            float(value.real) for value in angular_response
        ),
        "angular_condition_response_imag": tuple(
            float(value.imag) for value in angular_response
        ),
        "angular_condition_response_max_abs": float(
            np.max(np.abs(angular_response))
        ),
        "standardized_current_response_singular_values": tuple(
            float(value) for value in response.current_response_singular_values
        ),
        "diagnostic_current_response_rank": response.diagnostic_current_response_rank,
        "rank_is_physical": False,
        "state_tangent_norms_by_helicity": tuple(
            tuple(float(value) for value in row)
            for row in response.state_tangent_norms
        ),
        "standardized_internal_state_tangent_singular_values": tuple(
            float(value) for value in response.internal_state_tangent_singular_values
        ),
        "diagnostic_internal_state_tangent_rank": response.diagnostic_internal_state_tangent_rank,
        "internal_gram_derivative_max_abs": float(
            np.max(response.internal_gram_derivative_residuals)
        ),
        "internal_gram_finite_difference_max_abs": float(
            np.max(response.internal_gram_finite_difference_residuals)
        ),
        "energy_response_by_helicity": tuple(
            tuple(float(value) for value in row)
            for row in response.energy_response_by_helicity
        ),
        "qg_weight_response_by_helicity": tuple(
            tuple(float(value) for value in row)
            for row in response.qg_weight_response_by_helicity
        ),
        "internal_state_variations_recomputed": True,
        "null_response_interpretation": (
            "FACTORIZED_POINTLIKE_CURRENT_IS_BLIND_TO_INTERNAL_M2_STATE_VARIATIONS"
        ),
        "source_qualified_microscopic_current_constructed": False,
        "hamiltonian_consistent_current_constructed": False,
        "production_current_selected": False,
        "physical_response_rank": None,
        "next_required_object": (
            "SOURCE_QUALIFIED_INTERNAL_OPERATOR_VALUED_CURRENT_OR_EXPLICIT_"
            "MICROSCOPIC_TO_NUCLEON_CURRENT_MATCHING"
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**payload, "root": sha256(encoded.encode()).hexdigest()}


__all__ = [
    "NN_CURRENT_RESPONSE_CLAIM_TIER",
    "NN_CURRENT_RESPONSE_STATUS",
    "HELICITY_CURRENT_IDS",
    "EXPECTED_PARAMETER_IDS",
    "DEFAULT_PARAMETER_SCALES",
    "PARAMETER_UNITS",
    "NNMicroscopicCurrentResponse",
    "evaluate_nn_microscopic_current_response",
    "nn_microscopic_current_response_record",
]

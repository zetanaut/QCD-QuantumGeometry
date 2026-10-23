"""M2h response through the finite-basis active-quark impulse current.

M2g established a rank-four internal-state tangent and an exactly null
pointlike-current response.  M2h holds the source-derived active-quark
operator fixed, recomputes the state at both endpoints of the same four
microscopic directions, and propagates the resulting current through the
existing NN grid.  The active-subsystem operator is source-qualified within
its stated Wilson-order-zero finite-basis scope.  Its pointlike-spectator
nucleon lift remains exploratory and is not a complete current matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from typing import Any, Mapping

import numpy as np

from .m2_active_quark_current import (
    CURRENT_COMPONENT_IDS,
    M2ActiveQuarkImpulseOperator,
    NNActiveImpulseDiagnosticCurrent,
    active_operator_pair_record,
    build_m2_active_quark_impulse_operator,
    evaluate_nn_active_impulse_diagnostic_current,
    nn_active_impulse_pair_record,
)
from .m2_c53_mixed_singlet_state import (
    C53_PARAMETER_ID,
    build_c53_mixed_k9_hamiltonian,
    solve_c53_mixed_k9_helicity_pair,
    vary_c53_mixed_k9_point,
)
from .m2_nn_current_response import (
    DEFAULT_PARAMETER_SCALES,
    EXPECTED_PARAMETER_IDS,
    PARAMETER_UNITS,
)
from .m2_nn_deuteron_state import (
    ExploratoryNNDeuteronState,
    compose_nn_deuteron_state,
)
from .m2_nn_transfer_current import (
    NN_POINTLIKE_CHARGE_CURRENT_V1,
    NNOneBodyDiagnosticCurrent,
    build_nn_transfer_fibers,
    evaluate_nn_one_body_diagnostic_current,
)


M2_INTERNAL_CURRENT_RESPONSE_STATUS = (
    "M2_ACTIVE_QUARK_CURRENT_MICROSCOPIC_RESPONSE_NONZERO_READY"
)
M2_INTERNAL_CURRENT_RESPONSE_CLAIM_TIER = (
    "EXPLORATORY_RESPONSE_THROUGH_SOURCE_DERIVED_ACTIVE_SUBSYSTEM_CURRENT"
)
PAIR_CURRENT_IDS = ("j--", "j-+", "j+-", "j++")
DIAGNOSTIC_RANK_RELATIVE_TOLERANCE = 1.0e-6


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _amplitude_vector(current: NNActiveImpulseDiagnosticCurrent) -> np.ndarray:
    spin = current.spin_one_plus_current()
    return np.asarray(
        (spin.plus_plus, spin.plus_zero, spin.plus_minus, spin.zero_zero),
        dtype=np.complex128,
    )


def _pair_vector(current: np.ndarray) -> np.ndarray:
    return np.asarray(current, dtype=np.complex128).reshape(-1)


def _phase_aligned_basis(candidate: Any, reference_basis: np.ndarray) -> np.ndarray:
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
    # The response uses a 1e-6 central difference followed by a 5376-node
    # nuclear quadrature.  Roundoff divided by the finite-difference step
    # appears at O(1e-10) in the final amplitudes.  A 1e-6 relative threshold
    # cleanly separates that floor from the second resolved singular value
    # (which is O(1e-2) relative to the first).  This remains a diagnostic
    # numerical rank, never a physical identifiability threshold.
    tolerance = max(
        float(values[0]) * DIAGNOSTIC_RANK_RELATIVE_TOLERANCE, 1.0e-12
    )
    return int(np.count_nonzero(values > tolerance))


@dataclass(frozen=True)
class NNInternalCurrentResponse:
    """Current and angular response to the four represented M2 directions."""

    state: ExploratoryNNDeuteronState
    operator: M2ActiveQuarkImpulseOperator
    reversed_operator: M2ActiveQuarkImpulseOperator
    forward_operator: M2ActiveQuarkImpulseOperator
    baseline_current: NNActiveImpulseDiagnosticCurrent
    reversed_current: NNActiveImpulseDiagnosticCurrent
    forward_current: NNActiveImpulseDiagnosticCurrent
    identity_control_current: NNOneBodyDiagnosticCurrent
    operator_record: Mapping[str, Any]
    current_record: Mapping[str, Any]
    finite_difference_step: float
    parameter_ids: tuple[str, ...]
    parameter_scales: Mapping[str, float]
    parameter_units: Mapping[str, str]
    pair_current_response: np.ndarray
    tangent_pair_current_response: np.ndarray
    finite_difference_current_response: np.ndarray
    full_target_matrix_response: np.ndarray
    component_response_max_abs: Mapping[str, np.ndarray]
    angular_condition_response: np.ndarray
    identity_control_response: np.ndarray
    state_tangent_matrix: np.ndarray
    claim_tier: str = M2_INTERNAL_CURRENT_RESPONSE_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, ExploratoryNNDeuteronState):
            raise TypeError("state must be an ExploratoryNNDeuteronState")
        if self.parameter_ids != EXPECTED_PARAMETER_IDS:
            raise ValueError("M2h requires the owned C396/C117/C53 parameter order")
        if set(self.parameter_scales) != set(self.parameter_ids):
            raise ValueError("every response parameter requires one scale")
        if set(self.parameter_units) != set(self.parameter_ids):
            raise ValueError("every response parameter requires declared units")
        if self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")
        parameter_count = len(self.parameter_ids)
        expected = {
            "pair_current_response": (len(PAIR_CURRENT_IDS), parameter_count),
            "tangent_pair_current_response": (len(PAIR_CURRENT_IDS), parameter_count),
            "finite_difference_current_response": (
                len(CURRENT_COMPONENT_IDS),
                parameter_count,
            ),
            "full_target_matrix_response": (parameter_count, 3, 3),
            "angular_condition_response": (parameter_count,),
            "identity_control_response": (len(CURRENT_COMPONENT_IDS), parameter_count),
            "state_tangent_matrix": (
                2 * self.state.pair.assembly.dimension,
                parameter_count,
            ),
        }
        for name, shape in expected.items():
            values = np.asarray(getattr(self, name))
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite with shape {shape}")
        if set(self.component_response_max_abs) != {"SS", "SD", "DS", "DD"}:
            raise ValueError("component response must retain SS, SD, DS, and DD")
        if any(np.asarray(value).shape != (parameter_count,) for value in self.component_response_max_abs.values()):
            raise ValueError("component response vectors have the wrong shape")
        if (
            self.claim_tier != M2_INTERNAL_CURRENT_RESPONSE_CLAIM_TIER
            or self.physical
        ):
            raise ValueError("M2h is an exploratory response, not a physical rank")

    @property
    def standardized_pair_response(self) -> np.ndarray:
        scales = np.asarray(
            [self.parameter_scales[name] for name in self.parameter_ids], dtype=float
        )
        return np.asarray(self.pair_current_response) * scales[None, :]

    @property
    def pair_response_singular_values(self) -> np.ndarray:
        return np.linalg.svd(
            _real_embedding(self.standardized_pair_response), compute_uv=False
        )

    @property
    def diagnostic_pair_response_rank(self) -> int:
        return _diagnostic_rank(self.pair_response_singular_values)

    @property
    def standardized_current_response(self) -> np.ndarray:
        scales = np.asarray(
            [self.parameter_scales[name] for name in self.parameter_ids], dtype=float
        )
        return np.asarray(self.finite_difference_current_response) * scales[None, :]

    @property
    def current_response_singular_values(self) -> np.ndarray:
        return np.linalg.svd(
            _real_embedding(self.standardized_current_response), compute_uv=False
        )

    @property
    def diagnostic_current_response_rank(self) -> int:
        return _diagnostic_rank(self.current_response_singular_values)

    @property
    def tangent_pair_consistency_residual(self) -> float:
        return float(
            np.max(
                np.abs(
                    self.pair_current_response
                    - self.tangent_pair_current_response
                )
            )
        )


def evaluate_nn_internal_current_response(
    state: ExploratoryNNDeuteronState,
    *,
    delta_x_GeV: float = 0.2,
    delta_y_GeV: float = 0.0,
    p_plus_GeV: float = 2.0,
    reference_deuteron_mass_GeV: float = 1.87561294257,
    finite_difference_step: float = 1.0e-6,
    quadrature_nodes: int = 28,
) -> NNInternalCurrentResponse:
    """Recompute all four M2 directions through the active impulse current."""

    if not isinstance(state, ExploratoryNNDeuteronState):
        raise TypeError("state must be an ExploratoryNNDeuteronState")
    dx = _finite_real(delta_x_GeV, "delta_x_GeV")
    dy = _finite_real(delta_y_GeV, "delta_y_GeV")
    p_plus = _finite_real(p_plus_GeV, "p_plus_GeV")
    reference_mass = _finite_real(
        reference_deuteron_mass_GeV, "reference_deuteron_mass_GeV"
    )
    step = _finite_real(finite_difference_step, "finite_difference_step")
    if dx <= 0.0 or dy != 0.0:
        raise ValueError("M2h requires the positive x-directed diagnostic transfer")
    if p_plus <= 0.0 or reference_mass <= 0.0 or step <= 0.0:
        raise ValueError("plus momentum, reference mass, and finite-difference step must be positive")
    parameter_ids = tuple(state.pair.assembly.parameter_ids)
    if parameter_ids != EXPECTED_PARAMETER_IDS:
        raise ValueError("unexpected microscopic parameter order")

    operator = build_m2_active_quark_impulse_operator(
        "K9",
        delta_x_GeV=dx,
        delta_y_GeV=dy,
        quadrature_nodes=quadrature_nodes,
    )
    reversed_operator = build_m2_active_quark_impulse_operator(
        "K9",
        delta_x_GeV=-dx,
        delta_y_GeV=-dy,
        quadrature_nodes=quadrature_nodes,
    )
    forward_operator = build_m2_active_quark_impulse_operator(
        "K9",
        delta_x_GeV=0.0,
        delta_y_GeV=0.0,
        quadrature_nodes=quadrature_nodes,
    )

    def current_for(
        candidate: ExploratoryNNDeuteronState,
        current_operator: M2ActiveQuarkImpulseOperator,
        pair_basis: np.ndarray | None = None,
    ) -> NNActiveImpulseDiagnosticCurrent:
        transfer = build_nn_transfer_fibers(
            candidate,
            delta_x_GeV=current_operator.delta_x_GeV,
            delta_y_GeV=current_operator.delta_y_GeV,
            p_plus_GeV=p_plus,
            reference_deuteron_mass_GeV=reference_mass,
        )
        return evaluate_nn_active_impulse_diagnostic_current(
            transfer, current_operator, pair_basis=pair_basis
        )

    baseline_current = current_for(state, operator)
    reversed_current = current_for(state, reversed_operator)
    forward_current = current_for(state, forward_operator)
    operator_record = active_operator_pair_record(
        operator, reversed_operator, forward_operator
    )
    current_record = nn_active_impulse_pair_record(
        baseline_current, reversed_current, forward_current
    )
    identity_transfer = build_nn_transfer_fibers(
        state,
        delta_x_GeV=dx,
        delta_y_GeV=dy,
        p_plus_GeV=p_plus,
        reference_deuteron_mass_GeV=reference_mass,
    )
    identity_control_current = evaluate_nn_one_body_diagnostic_current(
        identity_transfer, current_member=NN_POINTLIKE_CHARGE_CURRENT_V1
    )

    parameter_count = len(parameter_ids)
    pair_response = np.zeros((len(PAIR_CURRENT_IDS), parameter_count), complex)
    tangent_pair_response = np.zeros_like(pair_response)
    amplitude_response = np.zeros(
        (len(CURRENT_COMPONENT_IDS), parameter_count), complex
    )
    full_matrix_response = np.zeros((parameter_count, 3, 3), complex)
    angular_response = np.zeros(parameter_count, complex)
    state_tangents = np.zeros(
        (2 * state.pair.assembly.dimension, parameter_count), complex
    )
    component_response = {
        label: np.zeros(parameter_count, dtype=float)
        for label in ("SS", "SD", "DS", "DD")
    }
    baseline_basis = np.asarray(state.pair.basis, dtype=np.complex128)
    baseline_pair_current = np.asarray(baseline_current.pair_current)

    for column, parameter_id in enumerate(parameter_ids):
        safe_id = parameter_id.replace(":", "_")
        plus_point = vary_c53_mixed_k9_point(
            state.pair.assembly.point,
            parameter_id,
            step,
            point_id=f"{state.pair.assembly.point.point_id}__{safe_id}__M2H_PLUS_FD",
        )
        minus_point = vary_c53_mixed_k9_point(
            state.pair.assembly.point,
            parameter_id,
            -step,
            point_id=f"{state.pair.assembly.point.point_id}__{safe_id}__M2H_MINUS_FD",
        )
        plus_pair = solve_c53_mixed_k9_helicity_pair(
            build_c53_mixed_k9_hamiltonian(plus_point)
        )
        minus_pair = solve_c53_mixed_k9_helicity_pair(
            build_c53_mixed_k9_hamiltonian(minus_point)
        )
        plus_basis = _phase_aligned_basis(plus_pair, baseline_basis)
        minus_basis = _phase_aligned_basis(minus_pair, baseline_basis)
        tangent = (plus_basis - minus_basis) / (2.0 * step)
        state_tangents[:, column] = tangent.reshape(-1)
        plus_state = compose_nn_deuteron_state(
            plus_pair, wave_grid=state.wave_grid, channel=state.channel
        )
        minus_state = compose_nn_deuteron_state(
            minus_pair, wave_grid=state.wave_grid, channel=state.channel
        )
        plus_current = current_for(plus_state, operator, plus_basis)
        minus_current = current_for(minus_state, operator, minus_basis)

        pair_response[:, column] = _pair_vector(
            (plus_current.pair_current - minus_current.pair_current)
            / (2.0 * step)
        )
        tangent_pair_response[:, column] = _pair_vector(
            tangent.conj().T @ (operator.matrix @ baseline_basis)
            + baseline_basis.conj().T @ (operator.matrix @ tangent)
        )
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

    # The M2g identity operator result is analytic: differentiated
    # orthonormality makes its response exactly zero.  Keep it in the same
    # component/parameter shape as the new response for direct comparison.
    identity_response = np.zeros_like(amplitude_response)
    if np.max(np.abs(baseline_pair_current)) == 0.0:
        raise RuntimeError("active current unexpectedly annihilated the helicity pair")
    return NNInternalCurrentResponse(
        state=state,
        operator=operator,
        reversed_operator=reversed_operator,
        forward_operator=forward_operator,
        baseline_current=baseline_current,
        reversed_current=reversed_current,
        forward_current=forward_current,
        identity_control_current=identity_control_current,
        operator_record=operator_record,
        current_record=current_record,
        finite_difference_step=step,
        parameter_ids=parameter_ids,
        parameter_scales=dict(DEFAULT_PARAMETER_SCALES),
        parameter_units=dict(PARAMETER_UNITS),
        pair_current_response=pair_response,
        tangent_pair_current_response=tangent_pair_response,
        finite_difference_current_response=amplitude_response,
        full_target_matrix_response=full_matrix_response,
        component_response_max_abs=component_response,
        angular_condition_response=angular_response,
        identity_control_response=identity_response,
        state_tangent_matrix=state_tangents,
    )


def nn_internal_current_response_record(
    response: NNInternalCurrentResponse,
) -> Mapping[str, Any]:
    """Return a compact JSON-safe M2h scientific result and claim boundary."""

    if not isinstance(response, NNInternalCurrentResponse):
        raise TypeError("response must be an NNInternalCurrentResponse")
    baseline = _amplitude_vector(response.baseline_current)
    identity = np.asarray(
        (
            response.identity_control_current.spin_one_plus_current().plus_plus,
            response.identity_control_current.spin_one_plus_current().plus_zero,
            response.identity_control_current.spin_one_plus_current().plus_minus,
            response.identity_control_current.spin_one_plus_current().zero_zero,
        ),
        dtype=np.complex128,
    )
    angular = response.baseline_current.spin_one_plus_current().angular_condition(
        response.baseline_current.transfer.eta
    )
    identity_angular = (
        response.identity_control_current.spin_one_plus_current().angular_condition(
            response.identity_control_current.transfer.eta
        )
    )
    current_response = np.asarray(response.finite_difference_current_response)
    pair_response = np.asarray(response.pair_current_response)
    payload = {
        "status": M2_INTERNAL_CURRENT_RESPONSE_STATUS,
        "claim_tier": response.claim_tier,
        "physical": False,
        "point_id": response.state.pair.assembly.point.point_id,
        "hilbert_space_id": response.state.hilbert_space_id,
        "nucleon_family": response.state.channel,
        "operator_status": response.operator_record["status"],
        "operator_source_qualified_within_scope": True,
        "operator_complete_nucleon_current": False,
        "nucleon_lift_physical_matching": False,
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
        "pair_current_ids": PAIR_CURRENT_IDS,
        "helicity_current_ids": CURRENT_COMPONENT_IDS,
        "baseline_pair_current_real": tuple(
            tuple(float(value.real) for value in row)
            for row in response.baseline_current.pair_current
        ),
        "baseline_pair_current_imag": tuple(
            tuple(float(value.imag) for value in row)
            for row in response.baseline_current.pair_current
        ),
        "baseline_helicity_current_real": tuple(float(value.real) for value in baseline),
        "baseline_helicity_current_imag": tuple(float(value.imag) for value in baseline),
        "identity_control_helicity_current_real": tuple(float(value.real) for value in identity),
        "identity_control_helicity_current_imag": tuple(float(value.imag) for value in identity),
        "pair_current_response_real": tuple(
            tuple(float(value) for value in row) for row in pair_response.real
        ),
        "pair_current_response_imag": tuple(
            tuple(float(value) for value in row) for row in pair_response.imag
        ),
        "tangent_pair_consistency_residual": response.tangent_pair_consistency_residual,
        "finite_difference_current_response_real": tuple(
            tuple(float(value) for value in row) for row in current_response.real
        ),
        "finite_difference_current_response_imag": tuple(
            tuple(float(value) for value in row) for row in current_response.imag
        ),
        "full_target_matrix_response_max_abs": float(
            np.max(np.abs(response.full_target_matrix_response))
        ),
        "component_response_max_abs": {
            label: tuple(float(value) for value in values)
            for label, values in response.component_response_max_abs.items()
        },
        "baseline_angular_condition": {
            "real": float(angular.real),
            "imag": float(angular.imag),
        },
        "identity_control_angular_condition": {
            "real": float(identity_angular.real),
            "imag": float(identity_angular.imag),
        },
        "angular_condition_response_real": tuple(
            float(value.real) for value in response.angular_condition_response
        ),
        "angular_condition_response_imag": tuple(
            float(value.imag) for value in response.angular_condition_response
        ),
        "angular_condition_response_max_abs": float(
            np.max(np.abs(response.angular_condition_response))
        ),
        "identity_control_response_max_abs": float(
            np.max(np.abs(response.identity_control_response))
        ),
        "standardized_pair_response_singular_values": tuple(
            float(value) for value in response.pair_response_singular_values
        ),
        "diagnostic_pair_response_rank": response.diagnostic_pair_response_rank,
        "standardized_spin_one_response_singular_values": tuple(
            float(value) for value in response.current_response_singular_values
        ),
        "diagnostic_spin_one_response_rank": response.diagnostic_current_response_rank,
        "diagnostic_rank_relative_tolerance": DIAGNOSTIC_RANK_RELATIVE_TOLERANCE,
        "rank_is_physical": False,
        "physical_response_rank": None,
        "pointlike_identity_control_preserved": True,
        "source_qualified_internal_current_constructed": True,
        "complete_microscopic_to_nucleon_matching_constructed": False,
        "hamiltonian_consistent_current_constructed": False,
        "production_current_selected": False,
        "next_required_object": (
            "RESOLVE_SPECTATOR_MOMENTUM_DYNAMICS_AND_MATCH_ALL_ONE_BODY_"
            "CURRENT_COMPONENTS_BEFORE_PHYSICAL_CURRENT_OR_RANK"
        ),
        "operator_record_root": response.operator_record["root"],
        "current_record_root": response.current_record["root"],
        "c53_parameter_id": C53_PARAMETER_ID,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**payload, "root": sha256(encoded.encode()).hexdigest()}


__all__ = [
    "M2_INTERNAL_CURRENT_RESPONSE_STATUS",
    "M2_INTERNAL_CURRENT_RESPONSE_CLAIM_TIER",
    "PAIR_CURRENT_IDS",
    "DIAGNOSTIC_RANK_RELATIVE_TOLERANCE",
    "NNInternalCurrentResponse",
    "evaluate_nn_internal_current_response",
    "nn_internal_current_response_record",
]

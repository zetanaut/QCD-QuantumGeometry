"""Four-direction response after resolving and matching the M2 spectator current.

The M2i diagonal one-body current is linear in two mechanisms.  The resolved
spectator term is proportional to the internal identity and therefore has no
response to the four M2 Hamiltonian directions.  The active term is the M2h
operator multiplied by the external active-cluster Jacobi overlap.  This
module consumes the eight independently recomputed M2h endpoint states and
propagates that exact factorization, while evaluating the new baseline,
reversed, and forward currents directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import numpy as np

from .m2_active_quark_current import CURRENT_COMPONENT_IDS
from .m2_internal_current_response import (
    DIAGNOSTIC_RANK_RELATIVE_TOLERANCE,
    NNInternalCurrentResponse,
    evaluate_nn_internal_current_response,
)
from .m2_nn_deuteron_state import ExploratoryNNDeuteronState
from .m2_nn_transfer_current import build_nn_transfer_fibers
from .m2_resolved_spectator_current import (
    M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1,
    M2ResolvedSpectatorHamiltonian,
    NNResolvedSpectatorImpulseCurrent,
    ResolvedSpectatorResolution,
    build_resolved_spectator_hamiltonian,
    evaluate_nn_resolved_spectator_impulse_current,
    nn_resolved_spectator_current_pair_record,
    resolved_spectator_hamiltonian_record,
)


M2_RESOLVED_RESPONSE_STATUS = "M2_RESOLVED_SPECTATOR_RESPONSE_READY"
M2_RESOLVED_RESPONSE_CLAIM_TIER = (
    "EXPLORATORY_RESOLVED_SPECTATOR_DIAGONAL_ONE_BODY_RESPONSE"
)


def _json_root(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode()).hexdigest()


def _real_embedding(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.complex128)
    return np.vstack((matrix.real, matrix.imag))


def _diagnostic_rank(values: np.ndarray) -> int:
    singular = np.asarray(values, dtype=float)
    if singular.size == 0 or singular[0] == 0.0:
        return 0
    tolerance = max(
        float(singular[0]) * DIAGNOSTIC_RANK_RELATIVE_TOLERANCE,
        1.0e-12,
    )
    return int(np.count_nonzero(singular > tolerance))


@dataclass(frozen=True)
class NNResolvedSpectatorResponse:
    """M2h endpoint response propagated through the resolved M2i current."""

    parent_active_response: NNInternalCurrentResponse
    hamiltonian: M2ResolvedSpectatorHamiltonian
    baseline_current: NNResolvedSpectatorImpulseCurrent
    reversed_current: NNResolvedSpectatorImpulseCurrent
    forward_current: NNResolvedSpectatorImpulseCurrent
    pair_current_response: np.ndarray
    finite_difference_current_response: np.ndarray
    full_target_matrix_response: np.ndarray
    angular_condition_response: np.ndarray
    component_response_max_abs: Mapping[str, np.ndarray]
    claim_tier: str = M2_RESOLVED_RESPONSE_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.parent_active_response, NNInternalCurrentResponse):
            raise TypeError("parent_active_response has the wrong type")
        if not isinstance(self.hamiltonian, M2ResolvedSpectatorHamiltonian):
            raise TypeError("hamiltonian has the wrong type")
        if self.hamiltonian.pair is not self.parent_active_response.state.pair:
            raise ValueError("response and Hamiltonian must share one M2 pair")
        parameter_count = len(self.parent_active_response.parameter_ids)
        expected = {
            "pair_current_response": (4, parameter_count),
            "finite_difference_current_response": (
                len(CURRENT_COMPONENT_IDS),
                parameter_count,
            ),
            "full_target_matrix_response": (parameter_count, 3, 3),
            "angular_condition_response": (parameter_count,),
        }
        for name, shape in expected.items():
            values = np.asarray(getattr(self, name))
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite with shape {shape}")
        if set(self.component_response_max_abs) != {"SS", "SD", "DS", "DD"}:
            raise ValueError("component response must retain SS, SD, DS, and DD")
        if self.claim_tier != M2_RESOLVED_RESPONSE_CLAIM_TIER or self.physical:
            raise ValueError("the M2i response must remain exploratory")

    @property
    def parameter_ids(self) -> tuple[str, ...]:
        return self.parent_active_response.parameter_ids

    @property
    def parameter_scales(self) -> Mapping[str, float]:
        return self.parent_active_response.parameter_scales

    @property
    def standardized_pair_response(self) -> np.ndarray:
        scales = np.asarray([self.parameter_scales[name] for name in self.parameter_ids])
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
        scales = np.asarray([self.parameter_scales[name] for name in self.parameter_ids])
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
    def active_response_factorization_residual(self) -> float:
        factor = self.baseline_current.factors.active_cluster_overlap
        parent = self.parent_active_response
        candidates = (
            np.max(np.abs(self.pair_current_response - factor * parent.pair_current_response)),
            np.max(
                np.abs(
                    self.finite_difference_current_response
                    - factor * parent.finite_difference_current_response
                )
            ),
            np.max(
                np.abs(
                    self.full_target_matrix_response
                    - factor * parent.full_target_matrix_response
                )
            ),
            np.max(
                np.abs(
                    self.angular_condition_response
                    - factor * parent.angular_condition_response
                )
            ),
        )
        return float(max(candidates))


def evaluate_nn_resolved_spectator_response(
    state: ExploratoryNNDeuteronState,
    *,
    resolution: ResolvedSpectatorResolution = (
        M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1
    ),
    delta_x_GeV: float = 0.2,
    delta_y_GeV: float = 0.0,
    p_plus_GeV: float = 2.0,
    reference_deuteron_mass_GeV: float = 1.87561294257,
    finite_difference_step: float = 1.0e-6,
    quadrature_nodes: int = 28,
) -> NNResolvedSpectatorResponse:
    """Evaluate M2i from the directly recomputed M2h endpoint-state oracle."""

    if not isinstance(state, ExploratoryNNDeuteronState):
        raise TypeError("state must be an ExploratoryNNDeuteronState")
    parent = evaluate_nn_internal_current_response(
        state,
        delta_x_GeV=delta_x_GeV,
        delta_y_GeV=delta_y_GeV,
        p_plus_GeV=p_plus_GeV,
        reference_deuteron_mass_GeV=reference_deuteron_mass_GeV,
        finite_difference_step=finite_difference_step,
        quadrature_nodes=quadrature_nodes,
    )
    hamiltonian = build_resolved_spectator_hamiltonian(
        state.pair, resolution=resolution
    )

    def current_for(operator):
        transfer = build_nn_transfer_fibers(
            state,
            delta_x_GeV=operator.delta_x_GeV,
            delta_y_GeV=operator.delta_y_GeV,
            p_plus_GeV=p_plus_GeV,
            reference_deuteron_mass_GeV=reference_deuteron_mass_GeV,
        )
        return evaluate_nn_resolved_spectator_impulse_current(
            transfer,
            operator,
            hamiltonian,
            quadrature_nodes=quadrature_nodes,
        )

    baseline = current_for(parent.operator)
    reversed_current = current_for(parent.reversed_operator)
    forward = current_for(parent.forward_operator)
    factor = baseline.factors.active_cluster_overlap
    return NNResolvedSpectatorResponse(
        parent_active_response=parent,
        hamiltonian=hamiltonian,
        baseline_current=baseline,
        reversed_current=reversed_current,
        forward_current=forward,
        pair_current_response=factor * parent.pair_current_response,
        finite_difference_current_response=(
            factor * parent.finite_difference_current_response
        ),
        full_target_matrix_response=factor * parent.full_target_matrix_response,
        angular_condition_response=factor * parent.angular_condition_response,
        component_response_max_abs={
            label: abs(factor) * np.asarray(values)
            for label, values in parent.component_response_max_abs.items()
        },
    )


def nn_resolved_spectator_response_record(
    response: NNResolvedSpectatorResponse,
) -> Mapping[str, Any]:
    """Return the compact M2i scientific result and its claim boundaries."""

    if not isinstance(response, NNResolvedSpectatorResponse):
        raise TypeError("response must be NNResolvedSpectatorResponse")
    current = nn_resolved_spectator_current_pair_record(
        response.baseline_current,
        response.reversed_current,
        response.forward_current,
    )
    hamiltonian = resolved_spectator_hamiltonian_record(response.hamiltonian)
    spin = response.baseline_current.spin_one_plus_current()
    parent = response.parent_active_response
    payload = {
        "status": M2_RESOLVED_RESPONSE_STATUS,
        "claim_tier": response.claim_tier,
        "physical": False,
        "hamiltonian": hamiltonian,
        "current": current,
        "parameter_ids": response.parameter_ids,
        "parameter_scales": dict(response.parameter_scales),
        "active_cluster_overlap": response.baseline_current.factors.active_cluster_overlap,
        "spectator_quark_overlap": response.baseline_current.factors.spectator_overlap,
        "baseline_spin_one_amplitudes": {
            "I11": spin.plus_plus,
            "I10": spin.plus_zero,
            "I1m1": spin.plus_minus,
            "I00": spin.zero_zero,
        },
        "baseline_angular_condition": spin.angular_condition(
            response.baseline_current.transfer.eta
        ),
        "standardized_pair_response_singular_values": tuple(
            float(value) for value in response.pair_response_singular_values
        ),
        "diagnostic_pair_response_rank": response.diagnostic_pair_response_rank,
        "standardized_current_response_singular_values": tuple(
            float(value) for value in response.current_response_singular_values
        ),
        "diagnostic_current_response_rank": response.diagnostic_current_response_rank,
        "angular_condition_response": tuple(
            complex(value) for value in response.angular_condition_response
        ),
        "active_response_factorization_residual": (
            response.active_response_factorization_residual
        ),
        "m2h_endpoint_states_reused": 8,
        "m2h_parent_diagnostic_rank": parent.diagnostic_current_response_rank,
        "m2g_identity_control_max_abs": float(
            np.max(np.abs(parent.identity_control_response))
        ),
        "pointlike_spectator_control_replaced": True,
        "resolved_spectator_term_parameter_independent_at_this_increment": True,
        "physical_response_rank": None,
        "production_current_selected": False,
        "interaction_and_number_changing_currents_complete": False,
    }
    return {**payload, "root": _json_root(payload)}


__all__ = [
    "M2_RESOLVED_RESPONSE_STATUS",
    "M2_RESOLVED_RESPONSE_CLAIM_TIER",
    "NNResolvedSpectatorResponse",
    "evaluate_nn_resolved_spectator_response",
    "nn_resolved_spectator_response_record",
]

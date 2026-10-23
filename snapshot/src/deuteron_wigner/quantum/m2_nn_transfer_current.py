"""First finite-resolution transfer/current diagnostic for the M2 NN state.

The construction deliberately stops at a pointlike, spin-independent one-body
charge current.  It owns distinct symmetric-frame momentum fibers and consumes
the shared Volume-IV recoil authority in :mod:`deuteron_wigner.light_front`.
It is not a Hamiltonian-consistent production current and supplies no exchange,
pair, zero-mode, interaction, or nonnucleonic current terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from deuteron_wigner.kinematics import MomentumTransfer, TransverseVector
from deuteron_wigner.lf_current import (
    SpinOnePlusCurrent,
    nucleon_plus_current,
)
from deuteron_wigner.light_front import (
    LFNormalization,
    SpinRotation,
    off_forward_active_component_densities,
    spectator_preserving_recoil_coordinates,
)
from deuteron_wigner.pilot.fibers import (
    FiberRole,
    MomentumFiber,
    ZeroSkewnessFrame,
)

from .m2_nn_deuteron_state import ExploratoryNNDeuteronState


NN_TRANSFER_CURRENT_CLAIM_TIER = (
    "EXPLORATORY_FINITE_RESOLUTION_ONE_BODY_CHARGE_DIAGNOSTIC"
)
NN_RECOIL_AUTHORITY_ID = "VOLUME_IV_SPECTATOR_PRESERVING_XI0_RECOIL_V1"
REFERENCE_MASS_ROLE = "CALLER_KINEMATIC_REFERENCE_NOT_STATE_EIGENVALUE"
OMITTED_CURRENT_TERMS = (
    "TWO_BODY",
    "EXCHANGE",
    "PAIR",
    "ZERO_MODE",
    "INTERACTION",
    "PION",
    "DELTA",
    "SHORT_DISTANCE",
    "NONNUCLEONIC_TRANSITION",
)


@dataclass(frozen=True)
class NNOneBodyCurrentMember:
    """One explicitly named nucleon-current input for the NN diagnostic."""

    member_id: str
    proton_f1: float
    proton_f2: float
    neutron_f1: float
    neutron_f2: float
    q2_dependence: str
    physical: bool = False

    def __post_init__(self) -> None:
        if not self.member_id or not self.q2_dependence:
            raise ValueError("current member provenance must be nonempty")
        values = np.asarray(
            (
                self.proton_f1,
                self.proton_f2,
                self.neutron_f1,
                self.neutron_f2,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("current form factors must be finite")
        if self.physical:
            raise ValueError("the first one-body current member is diagnostic only")

    @property
    def isosinglet_f1(self) -> float:
        return self.proton_f1 + self.neutron_f1

    @property
    def isosinglet_f2(self) -> float:
        return self.proton_f2 + self.neutron_f2


NN_POINTLIKE_CHARGE_CURRENT_V1 = NNOneBodyCurrentMember(
    member_id="NN_POINTLIKE_PROTON_CHARGE_NEUTRON_NEUTRAL_V1",
    proton_f1=1.0,
    proton_f2=0.0,
    neutron_f1=0.0,
    neutron_f2=0.0,
    q2_dependence="CONSTANT_POINTLIKE_DIAGNOSTIC",
)


@dataclass(frozen=True)
class NNTransferFibers:
    """Distinct initial/final NN momentum fibers for one exploratory state."""

    state: ExploratoryNNDeuteronState
    frame: ZeroSkewnessFrame
    reference_mass_role: str = REFERENCE_MASS_ROLE
    recoil_authority_id: str = NN_RECOIL_AUTHORITY_ID
    claim_tier: str = NN_TRANSFER_CURRENT_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, ExploratoryNNDeuteronState):
            raise TypeError("state must be an ExploratoryNNDeuteronState")
        if self.reference_mass_role != REFERENCE_MASS_ROLE:
            raise ValueError("the frame mass must remain a caller kinematic reference")
        if self.recoil_authority_id != NN_RECOIL_AUTHORITY_ID:
            raise ValueError("the NN transfer must consume the Volume-IV recoil authority")
        if self.claim_tier != NN_TRANSFER_CURRENT_CLAIM_TIER or self.physical:
            raise ValueError("the first NN transfer fibers must remain exploratory")
        for fiber in (self.frame.average, self.frame.incoming, self.frame.outgoing):
            if fiber.hilbert_space_id != self.state.hilbert_space_id:
                raise ValueError("fiber Hilbert-space identity does not match the NN state")
            if fiber.sector_scope != "NN_ONLY":
                raise ValueError("the first transfer fibers are restricted to NN_ONLY")
            if fiber.regulator_or_pilot_id != self.state.wave_grid.resolution.resolution_id:
                raise ValueError("fiber nuclear resolution does not match the NN state")

    @property
    def q2_GeV2(self) -> float:
        return self.frame.delta_t.norm_squared()

    @property
    def eta(self) -> float:
        mass = self.frame.average.invariant_mass_gev
        return self.q2_GeV2 / (4.0 * mass**2)


def build_nn_transfer_fibers(
    state: ExploratoryNNDeuteronState,
    *,
    delta_x_GeV: float,
    delta_y_GeV: float,
    p_plus_GeV: float,
    reference_deuteron_mass_GeV: float,
) -> NNTransferFibers:
    """Build the typed symmetric-``xi=0`` frame for the exploratory NN state."""

    if not isinstance(state, ExploratoryNNDeuteronState):
        raise TypeError("state must be an ExploratoryNNDeuteronState")
    values = np.asarray(
        (
            delta_x_GeV,
            delta_y_GeV,
            p_plus_GeV,
            reference_deuteron_mass_GeV,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("fiber kinematics must be finite")
    if p_plus_GeV <= 0.0 or reference_deuteron_mass_GeV <= 0.0:
        raise ValueError("fiber plus momentum and reference mass must be positive")
    delta = MomentumTransfer(float(delta_x_GeV), float(delta_y_GeV))
    common = {
        "p_plus": float(p_plus_GeV),
        "invariant_mass_gev": float(reference_deuteron_mass_GeV),
        "light_front_convention": "vpm=(v0+-v3)/sqrt2",
        "normalization_id": LFNormalization.FLAT.value,
        "regulator_or_pilot_id": state.wave_grid.resolution.resolution_id,
        "hilbert_space_id": state.hilbert_space_id,
        "sector_scope": "NN_ONLY",
    }
    frame = ZeroSkewnessFrame(
        average=MomentumFiber(
            "m2-nn:fiber:average",
            FiberRole.AVERAGE,
            p_transverse=TransverseVector(0.0, 0.0),
            **common,
        ),
        incoming=MomentumFiber(
            "m2-nn:fiber:incoming",
            FiberRole.INCOMING,
            p_transverse=delta.scale(-0.5),
            **common,
        ),
        outgoing=MomentumFiber(
            "m2-nn:fiber:outgoing",
            FiberRole.OUTGOING,
            p_transverse=delta.scale(0.5),
            **common,
        ),
        delta_t=delta,
    )
    return NNTransferFibers(state=state, frame=frame)


@dataclass(frozen=True)
class NNOneBodyDiagnosticCurrent:
    """Finite-grid NN plus-current matrix and coherent S/D decomposition."""

    transfer: NNTransferFibers
    current_member: NNOneBodyCurrentMember
    target_matrix: np.ndarray
    component_matrices: Mapping[str, np.ndarray]
    claim_tier: str = NN_TRANSFER_CURRENT_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.transfer, NNTransferFibers):
            raise TypeError("transfer must be NNTransferFibers")
        if not isinstance(self.current_member, NNOneBodyCurrentMember):
            raise TypeError("current_member must be NNOneBodyCurrentMember")
        if np.asarray(self.target_matrix).shape != (3, 3):
            raise ValueError("target current matrix must have shape (3,3)")
        if set(self.component_matrices) != {"SS", "SD", "DS", "DD"}:
            raise ValueError("current must retain SS, SD, DS, and DD components")
        if any(np.asarray(value).shape != (3, 3) for value in self.component_matrices.values()):
            raise ValueError("all current components must have shape (3,3)")
        if self.claim_tier != NN_TRANSFER_CURRENT_CLAIM_TIER or self.physical:
            raise ValueError("the first NN current must remain diagnostic")

    def spin_one_plus_current(self) -> SpinOnePlusCurrent:
        values = np.asarray(self.target_matrix, dtype=np.complex128)
        return SpinOnePlusCurrent(
            plus_plus=complex(values[0, 0]),
            plus_zero=complex(values[0, 1]),
            plus_minus=complex(values[0, 2]),
            zero_zero=complex(values[1, 1]),
        )

    def component_reconstruction_residual(self) -> float:
        reconstructed = sum(
            np.asarray(value, dtype=np.complex128)
            for value in self.component_matrices.values()
        )
        return float(np.max(np.abs(self.target_matrix - reconstructed)))


def evaluate_nn_one_body_diagnostic_current(
    transfer: NNTransferFibers,
    *,
    current_member: NNOneBodyCurrentMember,
) -> NNOneBodyDiagnosticCurrent:
    """Integrate the pointlike one-body plus current on the retained NN grid."""

    if not isinstance(transfer, NNTransferFibers):
        raise TypeError("transfer must be NNTransferFibers")
    if not isinstance(current_member, NNOneBodyCurrentMember):
        raise TypeError("current_member must be NNOneBodyCurrentMember")
    grid = transfer.state.wave_grid
    delta = transfer.frame.delta_t
    nucleon_current = nucleon_plus_current(
        f1=current_member.isosinglet_f1,
        f2=current_member.isosinglet_f2,
        delta_x=delta.x,
        delta_y=delta.y,
        mass=grid.nucleon_mass_GeV,
    )
    components = {
        label: np.zeros((3, 3), dtype=np.complex128)
        for label in ("SS", "SD", "DS", "DD")
    }
    normalization_squared = grid.common_normalization_factor**2
    for y, p_x, p_y, weight in zip(
        grid.y,
        grid.p_x_GeV,
        grid.p_y_GeV,
        grid.flat_measure_weights,
    ):
        densities = off_forward_active_component_densities(
            y=float(y),
            p_x=float(p_x),
            p_y=float(p_y),
            delta_x=delta.x,
            delta_y=delta.y,
            mass=grid.nucleon_mass_GeV,
            radial=grid.wave_member.radial_GeV,
            normalization=LFNormalization.FLAT,
            spin_rotation=SpinRotation.MELOSH,
        )
        factor = float(weight) * normalization_squared
        for label, density in densities.items():
            components[label] += factor * np.einsum(
                "IHca,ca->IH", density, nucleon_current
            )
    total = sum(components.values())
    return NNOneBodyDiagnosticCurrent(
        transfer=transfer,
        current_member=current_member,
        target_matrix=total,
        component_matrices=components,
    )


def _spin_one_parity_residual(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.complex128)
    residuals = (
        values[2, 2] - values[0, 0],
        values[2, 1] + values[0, 1],
        values[1, 2] + values[1, 0],
        values[2, 0] - values[0, 2],
    )
    return float(max(abs(value) for value in residuals))


def nn_one_body_current_pair_record(
    positive: NNOneBodyDiagnosticCurrent,
    reversed_transfer: NNOneBodyDiagnosticCurrent,
) -> Mapping[str, Any]:
    """Report charge, reversal, parity, S/D, and angular-condition diagnostics."""

    if positive.transfer.state is not reversed_transfer.transfer.state:
        raise ValueError("reversal pair must act on the same NN state")
    if positive.current_member != reversed_transfer.current_member:
        raise ValueError("reversal pair must use one current member")
    delta = positive.transfer.frame.delta_t
    reverse_delta = reversed_transfer.transfer.frame.delta_t
    if delta.x <= 0.0 or delta.y != 0.0:
        raise ValueError("the spin-one diagnostic record requires positive x-directed transfer")
    if reverse_delta != delta.scale(-1.0):
        raise ValueError("reversal pair must use opposite transverse transfer")
    plus_matrix = np.asarray(positive.target_matrix, dtype=np.complex128)
    minus_matrix = np.asarray(reversed_transfer.target_matrix, dtype=np.complex128)
    reversal_partner = {"SS": "SS", "SD": "DS", "DS": "SD", "DD": "DD"}
    component_reversal = max(
        float(
            np.max(
                np.abs(
                    reversed_transfer.component_matrices[label]
                    - positive.component_matrices[partner].conj().T
                )
            )
        )
        for label, partner in reversal_partner.items()
    )
    forward_charge = (
        positive.current_member.isosinglet_f1
        * positive.transfer.state.wave_grid.helicity_gram()
    )
    recoil = spectator_preserving_recoil_coordinates(
        y=float(positive.transfer.state.wave_grid.y[0]),
        p_x=float(positive.transfer.state.wave_grid.p_x_GeV[0]),
        p_y=float(positive.transfer.state.wave_grid.p_y_GeV[0]),
        delta_x=delta.x,
        delta_y=delta.y,
    )
    spin_current = positive.spin_one_plus_current()
    zero_plus_relation = abs(plus_matrix[1, 0] + plus_matrix[0, 1])
    return {
        "status": "M2_EXPLORATORY_NN_ONE_BODY_TRANSFER_CURRENT_READY",
        "claim_tier": NN_TRANSFER_CURRENT_CLAIM_TIER,
        "physical": False,
        "hilbert_space_id": positive.transfer.state.hilbert_space_id,
        "nucleon_family": positive.transfer.state.channel,
        "internal_resolution": "K9",
        "nuclear_resolution": positive.transfer.state.wave_grid.resolution.resolution_id,
        "frame": "SYMMETRIC_XI0",
        "xi": positive.transfer.frame.xi,
        "delta_plus": positive.transfer.frame.delta_plus,
        "delta_t_GeV": (delta.x, delta.y),
        "q2_GeV2": positive.transfer.q2_GeV2,
        "reference_deuteron_mass_GeV": positive.transfer.frame.average.invariant_mass_gev,
        "reference_mass_role": positive.transfer.reference_mass_role,
        "p_plus_GeV": positive.transfer.frame.average.p_plus,
        "eta": positive.transfer.eta,
        "incoming_transverse_GeV": (
            positive.transfer.frame.incoming.p_transverse.x,
            positive.transfer.frame.incoming.p_transverse.y,
        ),
        "outgoing_transverse_GeV": (
            positive.transfer.frame.outgoing.p_transverse.x,
            positive.transfer.frame.outgoing.p_transverse.y,
        ),
        "recoil_authority_id": positive.transfer.recoil_authority_id,
        "recoil_jacobian": recoil.jacobian,
        "active_physical_transfer_residual": float(
            np.max(np.abs(np.asarray(recoil.active_physical_transfer) - delta.array()))
        ),
        "spectator_physical_transfer_residual": float(
            np.max(np.abs(np.asarray(recoil.spectator_physical_transfer)))
        ),
        "current_member_id": positive.current_member.member_id,
        "current_member_q2_dependence": positive.current_member.q2_dependence,
        "isosinglet_f1": positive.current_member.isosinglet_f1,
        "isosinglet_f2": positive.current_member.isosinglet_f2,
        "target_matrix": tuple(
            tuple(complex(value) for value in row) for row in plus_matrix
        ),
        "spin_one_amplitudes": {
            "I11": spin_current.plus_plus,
            "I10": spin_current.plus_zero,
            "I1m1": spin_current.plus_minus,
            "I00": spin_current.zero_zero,
        },
        "charge_q2_zero_residual": float(
            np.max(np.abs(forward_charge - np.eye(3)))
        ),
        "reversal_hermiticity_residual": float(
            np.max(np.abs(minus_matrix - plus_matrix.conj().T))
        ),
        "component_reversal_residual": component_reversal,
        "parity_residual": _spin_one_parity_residual(plus_matrix),
        "zero_plus_relation_residual": float(zero_plus_relation),
        "angular_condition_residual": spin_current.angular_condition(
            positive.transfer.eta
        ),
        "relative_angular_condition_violation": spin_current.relative_angular_violation(
            positive.transfer.eta
        ),
        "sd_component_reconstruction_residual": max(
            positive.component_reconstruction_residual(),
            reversed_transfer.component_reconstruction_residual(),
        ),
        "nuclear_hamiltonian_constructed": False,
        "deuteron_mass_eigenvalue_constructed": False,
        "hamiltonian_consistent_current_constructed": False,
        "production_current_selected": False,
        "omitted_current_terms": OMITTED_CURRENT_TERMS,
    }


__all__ = [
    "NN_TRANSFER_CURRENT_CLAIM_TIER",
    "NN_RECOIL_AUTHORITY_ID",
    "REFERENCE_MASS_ROLE",
    "OMITTED_CURRENT_TERMS",
    "NNOneBodyCurrentMember",
    "NN_POINTLIKE_CHARGE_CURRENT_V1",
    "NNTransferFibers",
    "NNOneBodyDiagnosticCurrent",
    "build_nn_transfer_fibers",
    "evaluate_nn_one_body_diagnostic_current",
    "nn_one_body_current_pair_record",
]

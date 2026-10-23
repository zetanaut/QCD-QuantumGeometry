"""Finite-basis active-quark impulse current for the M2 ``q + qg`` state.

The M2g current acted as the identity on the normalized internal state.  This
module constructs the first nontrivial internal current that is already fixed
by the represented source conventions.  At zero skewness and Wilson order
zero the canonical quark bilocal is diagonal in retained parton number.  In
the one-quark sector there is no resolved intrinsic coordinate.  In each
``qg`` longitudinal partition the C47 relative oscillator coordinate changes
by

    q_rel,out - q_rel,in = sqrt(x_g / x_q) Delta_T,

which follows from the Volume-XI active/spectator recoil map and C47's
``q_rel = k_q / sqrt(x_q x_g)`` convention.  The matrix elements are direct
C45 harmonic-oscillator overlaps evaluated with Gauss--Hermite quadrature.

This is a source-derived finite-basis *active-subsystem* operator.  It is not
a complete electromagnetic nucleon current: the two-quark spectator added by
the M2 color/spin/flavor completion has no resolved momentum wave function.
For diagnostics only, this module also supplies the unique current available
in the existing identity-spectator extension: the active quark carries its
exact flavor charge and the unresolved spectator charge is pointlike.  That
lift preserves target charge at zero transfer, but it is not a physical or
Hamiltonian-consistent matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from math import isfinite, pi, sqrt
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.special import roots_hermite

from deuteron_wigner.bridge.c401_c396_mass_directions import (
    canonical_partitions,
    resolution_record,
)
from deuteron_wigner.bridge.modes.core import ho_momentum
from deuteron_wigner.conventions import HELICITIES
from deuteron_wigner.lf_current import SpinOnePlusCurrent
from deuteron_wigner.light_front import (
    LFNormalization,
    SpinRotation,
    off_forward_active_component_densities,
)

from .m2_c53_mixed_singlet_state import C53MixedK9HelicityPair, K9
from .m2_color_singlet_bridge import m2_color_modules
from .m2_nn_deuteron_state import ExploratoryNNDeuteronState
from .m2_nn_transfer_current import NNTransferFibers
from .m2_nucleon_spectator_composition import NucleonSpectatorDoublet


M2_ACTIVE_CURRENT_STATUS = "M2_ACTIVE_QUARK_IMPULSE_OPERATOR_READY"
M2_ACTIVE_CURRENT_CLAIM_TIER = (
    "SOURCE_DERIVED_FINITE_BASIS_ACTIVE_SUBSYSTEM_OPERATOR"
)
NN_ACTIVE_IMPULSE_STATUS = "M2_ACTIVE_RESOLVED_NN_CURRENT_DIAGNOSTIC_READY"
NN_ACTIVE_IMPULSE_CLAIM_TIER = (
    "EXPLORATORY_ACTIVE_QUARK_IMPULSE_WITH_POINTLIKE_SPECTATOR_EXTENSION"
)
CURRENT_COMPONENT_IDS = ("I11", "I10", "I1m1", "I00")
ACTIVE_OPERATOR_SOURCES = (
    "C45_2D_HO_NORMALIZATION_AND_PHASE",
    "C47_INTRINSIC_QG_COORDINATE",
    "VOLUME_XI_SYMMETRIC_XI0_ACTIVE_SPECTATOR_RECOIL",
    "VOLUME_XI_WILSON_ORDER_ZERO_DIAGONAL_FOCK_OVERLAP",
)
UNAVAILABLE_INTERNAL_CURRENT_TERMS = (
    "Q_TO_QG_AND_QG_TO_Q_NUMBER_CHANGING_CURRENT",
    "INDUCED_EFFECTIVE_CURRENT",
    "CONSTRAINED_FIELD_CURRENT",
    "ZERO_MODE_CURRENT",
    "OPERATOR_RENORMALIZATION_AND_MATCHING",
)
UNAVAILABLE_NUCLEON_CURRENT_TERMS = (
    "RESOLVED_TWO_QUARK_SPECTATOR_MOMENTUM_CURRENT",
    "ACTIVE_SPECTATOR_RELATIVE_RECOIL",
    "HAMILTONIAN_CONSISTENT_INTERACTION_CURRENT",
    "PAIR_AND_ZERO_MODE_CURRENT",
    "NUCLEON_CURRENT_RENORMALIZATION_AND_MATCHING",
)


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _max_abs(matrix: sparse.spmatrix) -> float:
    values = matrix.tocsr().data
    return float(np.max(np.abs(values))) if values.size else 0.0


def _sparse_root(matrix: sparse.csr_matrix) -> str:
    value = matrix.tocsr()
    digest = sha256()
    for array in (value.indptr, value.indices, value.data):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _json_root(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode()).hexdigest()


@lru_cache(maxsize=None)
def displaced_ho_overlap(
    modes: tuple[tuple[int, int], ...],
    b_GeV: float,
    shift_x_GeV: float,
    shift_y_GeV: float,
    quadrature_nodes: int = 28,
) -> np.ndarray:
    """Return ``<out|in>`` for symmetrically displaced C45 HO modes.

    The returned element is

    ``integral d2q/(2pi)^2 phi_out(q+s/2)^* phi_in(q-s/2)``.

    The common Gaussian is integrated by tensor-product Gauss--Hermite
    quadrature.  No element is threshold-pruned.  Exactly zero transfer is
    returned as the identity so finite-basis charge normalization is exact.
    """

    b = _finite_real(b_GeV, "b_GeV")
    sx = _finite_real(shift_x_GeV, "shift_x_GeV")
    sy = _finite_real(shift_y_GeV, "shift_y_GeV")
    if b <= 0.0:
        raise ValueError("b_GeV must be positive")
    if isinstance(quadrature_nodes, bool) or int(quadrature_nodes) != quadrature_nodes:
        raise ValueError("quadrature_nodes must be an integer")
    nodes = int(quadrature_nodes)
    if nodes < 16:
        raise ValueError("quadrature_nodes must be at least 16")
    if not modes or len(modes) != len(set(modes)):
        raise ValueError("modes must be a nonempty unique tuple")
    if any(n < 0 or 2 * n + abs(m) < 0 for n, m in modes):
        raise ValueError("invalid transverse HO mode")
    if sx == 0.0 and sy == 0.0:
        result = np.eye(len(modes), dtype=np.complex128)
        result.setflags(write=False)
        return result

    points, weights = roots_hermite(nodes)
    x, y = np.meshgrid(points, points, indexing="ij")
    qx = b * x
    qy = b * y
    weight = np.outer(weights, weights)
    # roots_hermite integrates exp(-x^2-y^2).  The two shifted C45 modes
    # already contain that Gaussian, so remove it from the sampled integrand.
    undo_weight = np.exp(x * x + y * y)
    left = tuple(
        ho_momentum(n, m, qx + 0.5 * sx, qy + 0.5 * sy, b)
        for n, m in modes
    )
    right = tuple(
        ho_momentum(n, m, qx - 0.5 * sx, qy - 0.5 * sy, b)
        for n, m in modes
    )
    result = np.empty((len(modes), len(modes)), dtype=np.complex128)
    measure = b * b / (2.0 * pi) ** 2
    for row, phi_out in enumerate(left):
        for column, phi_in in enumerate(right):
            result[row, column] = measure * np.sum(
                weight * np.conjugate(phi_out) * phi_in * undo_weight
            )
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class M2ActiveQuarkImpulseOperator:
    """Canonical active-quark current on one reduced M2 singlet space."""

    resolution: str
    delta_x_GeV: float
    delta_y_GeV: float
    quadrature_nodes: int
    matrix: sparse.csr_matrix
    partition_records: tuple[Mapping[str, Any], ...]
    claim_tier: str = M2_ACTIVE_CURRENT_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        record = resolution_record(self.resolution)
        expected = int(record["direct_sum_dimension"]) // 3
        matrix = self.matrix.tocsr()
        if matrix.shape != (expected, expected):
            raise ValueError(f"active current must have shape ({expected},{expected})")
        if matrix.data.size and not np.all(np.isfinite(matrix.data)):
            raise ValueError("active current contains nonfinite entries")
        if len(self.partition_records) != len(canonical_partitions(self.resolution)):
            raise ValueError("every longitudinal partition requires a recoil record")
        if self.claim_tier != M2_ACTIVE_CURRENT_CLAIM_TIER or self.physical:
            raise ValueError("the M2 active current cannot be promoted to a physical current")

    @property
    def dimension(self) -> int:
        return self.matrix.shape[0]

    @property
    def q2_GeV2(self) -> float:
        return self.delta_x_GeV**2 + self.delta_y_GeV**2

    @property
    def operator_root(self) -> str:
        return _sparse_root(self.matrix)

    def apply(self, vector: Sequence[Any]) -> np.ndarray:
        values = np.asarray(vector, dtype=np.complex128)
        if values.shape != (self.dimension,):
            raise ValueError(f"vector must have shape ({self.dimension},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("vector contains nonfinite entries")
        return np.asarray(self.matrix @ values, dtype=np.complex128)


def build_m2_active_quark_impulse_operator(
    resolution: str,
    *,
    delta_x_GeV: float,
    delta_y_GeV: float,
    quadrature_nodes: int = 28,
) -> M2ActiveQuarkImpulseOperator:
    """Assemble the diagonal-Fock canonical quark current in M2 target order."""

    dx = _finite_real(delta_x_GeV, "delta_x_GeV")
    dy = _finite_real(delta_y_GeV, "delta_y_GeV")
    record = resolution_record(resolution)
    b = float(record["b_HO"])
    labels = m2_color_modules(resolution).multiplicity_labels
    rows: list[int] = []
    columns: list[int] = []
    data: list[complex] = []

    q_indices = [index for index, label in enumerate(labels) if label[0] == "q"]
    if len(q_indices) != 2:
        raise RuntimeError("M2 reduced one-quark sector must contain two helicities")
    for index in q_indices:
        rows.append(index)
        columns.append(index)
        data.append(1.0 + 0.0j)

    partition_records: list[Mapping[str, Any]] = []
    for partition in canonical_partitions(resolution):
        partition_indices = [
            index
            for index, label in enumerate(labels)
            if label[0] == "qg" and label[1] == partition.partition_id
        ]
        modes = tuple(dict.fromkeys((int(labels[index][2]), int(labels[index][3])) for index in partition_indices))
        expected_modes = int(partition.qg_state_count) // (3 * 4)
        if len(modes) != expected_modes:
            raise RuntimeError("qg partition does not close its transverse/helicity basis")
        shift_factor = sqrt(float(partition.xg / partition.xq))
        sx = shift_factor * dx
        sy = shift_factor * dy
        overlap = displaced_ho_overlap(modes, b, sx, sy, quadrature_nodes)
        mode_index = {mode: index for index, mode in enumerate(modes)}
        groups: dict[tuple[int, int], list[int]] = {}
        for index in partition_indices:
            label = labels[index]
            groups.setdefault((int(label[4]), int(label[5])), []).append(index)
        if set(groups) != {(-1, -1), (-1, 1), (1, -1), (1, 1)}:
            raise RuntimeError("qg helicity blocks are incomplete")
        for indices in groups.values():
            for row in indices:
                out_mode = mode_index[(int(labels[row][2]), int(labels[row][3]))]
                for column in indices:
                    in_mode = mode_index[(int(labels[column][2]), int(labels[column][3]))]
                    rows.append(row)
                    columns.append(column)
                    data.append(complex(overlap[out_mode, in_mode]))
        singular_values = np.linalg.svd(overlap, compute_uv=False)
        partition_records.append(
            {
                "partition_id": partition.partition_id,
                "xq": str(partition.xq),
                "xg": str(partition.xg),
                "relative_shift_factor": shift_factor,
                "relative_shift_GeV": (sx, sy),
                "formula": "delta_q_rel=sqrt(x_g/x_q)*Delta_T",
                "mode_count": len(modes),
                "helicity_blocks": 4,
                "largest_singular_value": float(singular_values[0]),
                "smallest_singular_value": float(singular_values[-1]),
                "finite_shell_leakage_retained": True,
            }
        )

    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=np.complex128), (rows, columns)),
        shape=(len(labels), len(labels)),
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return M2ActiveQuarkImpulseOperator(
        resolution=resolution,
        delta_x_GeV=dx,
        delta_y_GeV=dy,
        quadrature_nodes=int(quadrature_nodes),
        matrix=matrix,
        partition_records=tuple(partition_records),
    )


def active_operator_pair_record(
    positive: M2ActiveQuarkImpulseOperator,
    reversed_transfer: M2ActiveQuarkImpulseOperator,
    forward: M2ActiveQuarkImpulseOperator,
) -> Mapping[str, Any]:
    """Report source scope, charge, reversal, and finite-shell diagnostics."""

    if not (
        positive.resolution == reversed_transfer.resolution == forward.resolution
    ):
        raise ValueError("operator comparison requires one resolution")
    if (
        reversed_transfer.delta_x_GeV != -positive.delta_x_GeV
        or reversed_transfer.delta_y_GeV != -positive.delta_y_GeV
    ):
        raise ValueError("reversed operator must carry exactly opposite transfer")
    if forward.delta_x_GeV != 0.0 or forward.delta_y_GeV != 0.0:
        raise ValueError("forward operator must have zero transfer")
    identity = sparse.identity(
        positive.dimension, format="csr", dtype=np.complex128
    )
    payload = {
        "status": M2_ACTIVE_CURRENT_STATUS,
        "claim_tier": positive.claim_tier,
        "physical": False,
        "resolution": positive.resolution,
        "dimension": positive.dimension,
        "delta_t_GeV": (positive.delta_x_GeV, positive.delta_y_GeV),
        "q2_GeV2": positive.q2_GeV2,
        "quadrature_nodes": positive.quadrature_nodes,
        "basis": "M2_COLOR_SINGLET_MULTIPLICITY_SPACE_IN_C401_TARGET_ORDER",
        "sector_action": {
            "q_to_q": "EXACT_INTRINSIC_IDENTITY",
            "qg_to_qg": "PARTITION_DIAGONAL_HELICITY_DIAGONAL_DISPLACED_HO_OVERLAP",
            "q_to_qg": "UNAVAILABLE_SEPARATE_OPERATOR_NOT_PHYSICAL_ZERO",
            "qg_to_q": "UNAVAILABLE_SEPARATE_OPERATOR_NOT_PHYSICAL_ZERO",
        },
        "active_spin_kernel": "NORMALIZED_GAMMA_PLUS_HELICITY_IDENTITY_AT_XI0",
        "relative_recoil": "delta_q_rel=sqrt(x_g/x_q)*Delta_T",
        "operator_sources": ACTIVE_OPERATOR_SOURCES,
        "partition_records": positive.partition_records,
        "forward_charge_residual": _max_abs(forward.matrix - identity),
        "reversal_adjoint_residual": _max_abs(
            reversed_transfer.matrix - positive.matrix.getH()
        ),
        "maximum_partition_contraction_excess": max(
            max(0.0, float(row["largest_singular_value"]) - 1.0)
            for row in positive.partition_records
        ),
        "minimum_retained_partition_singular_value": min(
            float(row["smallest_singular_value"])
            for row in positive.partition_records
        ),
        "finite_shell_projection_is_unitary": False,
        "threshold_pruned": False,
        "wilson_order": 0,
        "source_qualified_operator_form": True,
        "complete_nucleon_current": False,
        "hamiltonian_consistent_current": False,
        "production_current": False,
        "unavailable_internal_terms": UNAVAILABLE_INTERNAL_CURRENT_TERMS,
        "operator_root": positive.operator_root,
    }
    return {**payload, "root": _json_root(payload)}


def pair_active_current_matrix(
    pair: C53MixedK9HelicityPair,
    operator: M2ActiveQuarkImpulseOperator,
    *,
    basis: np.ndarray | None = None,
) -> np.ndarray:
    """Project one active current between the two M2 helicity branches."""

    if not isinstance(pair, C53MixedK9HelicityPair):
        raise TypeError("pair must be a C53MixedK9HelicityPair")
    if not isinstance(operator, M2ActiveQuarkImpulseOperator):
        raise TypeError("operator must be an M2ActiveQuarkImpulseOperator")
    if operator.resolution != K9 or operator.dimension != pair.assembly.dimension:
        raise ValueError("operator and M2 helicity pair resolutions do not match")
    vectors = pair.basis if basis is None else np.asarray(basis, dtype=np.complex128)
    if vectors.shape != (operator.dimension, 2):
        raise ValueError("pair basis must have shape (M2 dimension,2)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("pair basis contains nonfinite entries")
    return np.asarray(vectors.conj().T @ (operator.matrix @ vectors), dtype=np.complex128)


def nucleon_active_impulse_matrix(
    doublet: NucleonSpectatorDoublet,
    pair_current: np.ndarray,
) -> np.ndarray:
    """Lift the active current through exact spectator charge tensors.

    The unresolved spectator charge acts as a pointlike identity.  This is the
    diagnostic extension of the current M2 spectator construction, not a
    source-qualified finite-size spectator current.
    """

    if not isinstance(doublet, NucleonSpectatorDoublet):
        raise TypeError("doublet must be a NucleonSpectatorDoublet")
    current = np.asarray(pair_current, dtype=np.complex128)
    if current.shape != (2, 2) or not np.all(np.isfinite(current)):
        raise ValueError("pair_current must be a finite 2-by-2 matrix")
    result = np.zeros((2, 2), dtype=np.complex128)
    pair_index = {-1: 0, 1: 1}
    for out_index, out_helicity in enumerate((-1, 1)):
        bra = doublet.state(out_helicity)
        for in_index, in_helicity in enumerate((-1, 1)):
            ket = doublet.state(in_helicity)
            value = 0.0 + 0.0j
            for bra_component in bra.components:
                for ket_component in ket.components:
                    external_bra = (
                        bra_component.active_flavor,
                        bra_component.spectator_spin_projection,
                        bra_component.spectator_isospin_projection,
                    )
                    external_ket = (
                        ket_component.active_flavor,
                        ket_component.spectator_spin_projection,
                        ket_component.spectator_isospin_projection,
                    )
                    if external_bra != external_ket:
                        continue
                    active_charge = float(bra_component.active_charge)
                    spectator_charge = float(
                        bra.target_charge - bra_component.active_charge
                    )
                    active = active_charge * current[
                        pair_index[bra_component.active_helicity_twice],
                        pair_index[ket_component.active_helicity_twice],
                    ]
                    spectator = (
                        spectator_charge
                        if bra_component.active_helicity_twice
                        == ket_component.active_helicity_twice
                        else 0.0
                    )
                    value += (
                        np.conjugate(bra_component.amplitude)
                        * ket_component.amplitude
                        * (active + spectator)
                    )
            result[out_index, in_index] = value
    return result


@dataclass(frozen=True)
class NNActiveImpulseDiagnosticCurrent:
    """NN current using the active M2 operator and pointlike spectator charge."""

    transfer: NNTransferFibers
    internal_operator: M2ActiveQuarkImpulseOperator
    pair_current: np.ndarray
    proton_current: np.ndarray
    neutron_current: np.ndarray
    target_matrix: np.ndarray
    component_matrices: Mapping[str, np.ndarray]
    claim_tier: str = NN_ACTIVE_IMPULSE_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.transfer, NNTransferFibers):
            raise TypeError("transfer must be NNTransferFibers")
        if not isinstance(self.internal_operator, M2ActiveQuarkImpulseOperator):
            raise TypeError("internal_operator has the wrong type")
        for name, matrix, shape in (
            ("pair_current", self.pair_current, (2, 2)),
            ("proton_current", self.proton_current, (2, 2)),
            ("neutron_current", self.neutron_current, (2, 2)),
            ("target_matrix", self.target_matrix, (3, 3)),
        ):
            values = np.asarray(matrix)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite {shape} matrix")
        if set(self.component_matrices) != {"SS", "SD", "DS", "DD"}:
            raise ValueError("current must retain SS, SD, DS, and DD")
        if any(np.asarray(value).shape != (3, 3) for value in self.component_matrices.values()):
            raise ValueError("all current components must have shape (3,3)")
        delta = self.transfer.frame.delta_t
        if (
            delta.x != self.internal_operator.delta_x_GeV
            or delta.y != self.internal_operator.delta_y_GeV
        ):
            raise ValueError("nuclear and internal currents must share one transfer")
        if self.claim_tier != NN_ACTIVE_IMPULSE_CLAIM_TIER or self.physical:
            raise ValueError("the active-resolved NN current must remain diagnostic")

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


def evaluate_nn_active_impulse_diagnostic_current(
    transfer: NNTransferFibers,
    operator: M2ActiveQuarkImpulseOperator,
    *,
    pair_basis: np.ndarray | None = None,
) -> NNActiveImpulseDiagnosticCurrent:
    """Insert the active-resolved current into the retained NN convolution."""

    if not isinstance(transfer, NNTransferFibers):
        raise TypeError("transfer must be NNTransferFibers")
    pair_current = pair_active_current_matrix(
        transfer.state.pair, operator, basis=pair_basis
    )
    proton_current = nucleon_active_impulse_matrix(
        transfer.state.proton, pair_current
    )
    neutron_current = nucleon_active_impulse_matrix(
        transfer.state.neutron, pair_current
    )
    isosinglet_current = proton_current + neutron_current
    grid = transfer.state.wave_grid
    delta = transfer.frame.delta_t
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
                "IHca,ca->IH", density, isosinglet_current
            )
    return NNActiveImpulseDiagnosticCurrent(
        transfer=transfer,
        internal_operator=operator,
        pair_current=pair_current,
        proton_current=proton_current,
        neutron_current=neutron_current,
        target_matrix=sum(components.values()),
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


def nn_active_impulse_pair_record(
    positive: NNActiveImpulseDiagnosticCurrent,
    reversed_transfer: NNActiveImpulseDiagnosticCurrent,
    forward: NNActiveImpulseDiagnosticCurrent,
) -> Mapping[str, Any]:
    """Report the diagnostic lift and keep all missing matching explicit."""

    if not (
        positive.transfer.state is reversed_transfer.transfer.state
        and positive.transfer.state is forward.transfer.state
    ):
        raise ValueError("current comparison requires one NN state")
    delta = positive.transfer.frame.delta_t
    reverse_delta = reversed_transfer.transfer.frame.delta_t
    forward_delta = forward.transfer.frame.delta_t
    if reverse_delta != delta.scale(-1.0):
        raise ValueError("reversed current must carry opposite transfer")
    if forward_delta.x != 0.0 or forward_delta.y != 0.0:
        raise ValueError("forward current must have zero transfer")
    plus = np.asarray(positive.target_matrix, dtype=np.complex128)
    minus = np.asarray(reversed_transfer.target_matrix, dtype=np.complex128)
    spin = positive.spin_one_plus_current()
    component_partner = {"SS": "SS", "SD": "DS", "DS": "SD", "DD": "DD"}
    component_reversal = max(
        float(
            np.max(
                np.abs(
                    reversed_transfer.component_matrices[label]
                    - positive.component_matrices[partner].conj().T
                )
            )
        )
        for label, partner in component_partner.items()
    )
    payload = {
        "status": NN_ACTIVE_IMPULSE_STATUS,
        "claim_tier": positive.claim_tier,
        "physical": False,
        "hilbert_space_id": positive.transfer.state.hilbert_space_id,
        "nucleon_family": positive.transfer.state.channel,
        "target_helicities": HELICITIES,
        "delta_t_GeV": (delta.x, delta.y),
        "q2_GeV2": positive.transfer.q2_GeV2,
        "internal_operator_status": M2_ACTIVE_CURRENT_STATUS,
        "internal_operator_source_qualified": True,
        "internal_operator_root": positive.internal_operator.operator_root,
        "nucleon_lift": (
            "EXACT_ACTIVE_FLAVOR_CHARGE_PLUS_POINTLIKE_UNRESOLVED_"
            "SPECTATOR_CHARGE_IDENTITY"
        ),
        "pair_current": tuple(tuple(complex(value) for value in row) for row in positive.pair_current),
        "proton_current": tuple(tuple(complex(value) for value in row) for row in positive.proton_current),
        "neutron_current": tuple(tuple(complex(value) for value in row) for row in positive.neutron_current),
        "spin_one_amplitudes": {
            "I11": spin.plus_plus,
            "I10": spin.plus_zero,
            "I1m1": spin.plus_minus,
            "I00": spin.zero_zero,
        },
        "forward_pair_charge_residual": float(
            np.max(np.abs(forward.pair_current - np.eye(2)))
        ),
        "forward_proton_charge_residual": float(
            np.max(np.abs(forward.proton_current - np.eye(2)))
        ),
        "forward_neutron_charge_residual": float(
            np.max(np.abs(forward.neutron_current))
        ),
        "forward_deuteron_charge_residual": float(
            np.max(np.abs(forward.target_matrix - np.eye(3)))
        ),
        "pair_reversal_adjoint_residual": float(
            np.max(np.abs(reversed_transfer.pair_current - positive.pair_current.conj().T))
        ),
        "target_reversal_adjoint_residual": float(
            np.max(np.abs(minus - plus.conj().T))
        ),
        "component_reversal_residual": component_reversal,
        "parity_residual": _spin_one_parity_residual(plus),
        "zero_plus_relation_residual": float(abs(plus[1, 0] + plus[0, 1])),
        "angular_condition_residual": spin.angular_condition(positive.transfer.eta),
        "relative_angular_condition_violation": spin.relative_angular_violation(
            positive.transfer.eta
        ),
        "sd_component_reconstruction_residual": max(
            positive.component_reconstruction_residual(),
            reversed_transfer.component_reconstruction_residual(),
            forward.component_reconstruction_residual(),
        ),
        "pointlike_spectator_is_physical_matching": False,
        "complete_nucleon_current": False,
        "hamiltonian_consistent_current": False,
        "production_current": False,
        "unavailable_nucleon_terms": UNAVAILABLE_NUCLEON_CURRENT_TERMS,
    }
    return {**payload, "root": _json_root(payload)}


__all__ = [
    "M2_ACTIVE_CURRENT_STATUS",
    "M2_ACTIVE_CURRENT_CLAIM_TIER",
    "NN_ACTIVE_IMPULSE_STATUS",
    "NN_ACTIVE_IMPULSE_CLAIM_TIER",
    "CURRENT_COMPONENT_IDS",
    "ACTIVE_OPERATOR_SOURCES",
    "UNAVAILABLE_INTERNAL_CURRENT_TERMS",
    "UNAVAILABLE_NUCLEON_CURRENT_TERMS",
    "displaced_ho_overlap",
    "M2ActiveQuarkImpulseOperator",
    "build_m2_active_quark_impulse_operator",
    "active_operator_pair_record",
    "pair_active_current_matrix",
    "nucleon_active_impulse_matrix",
    "NNActiveImpulseDiagnosticCurrent",
    "evaluate_nn_active_impulse_diagnostic_current",
    "nn_active_impulse_pair_record",
]

"""M2l symmetry-complete global-ground spin--flavor carrier.

M2j retained the complete C7 color-singlet multiplicity and M2k supplied the
exact emitter-tree recoupling.  For the one-ground-mode branch selected here,
``b_rho=sqrt(2/3)b_lambda`` makes the three-quark orbital wave function an
``S3`` scalar.  Fermi statistics can therefore be completed exactly in the
remaining color and spin--flavor factors.

The ordered one-quark spin--flavor basis is ``u-,u+,d-,d+``.  The exact
three-quark permutation representation acts on its 64-dimensional tensor
cube.  In the ``qqq`` sector, the color sign representation selects the
fully symmetric spin--flavor nucleon state with ``S=I=1/2``.  At fixed target
and helicity this state is unique.  Relative to a chosen active quark and
spectator pair, it has equal scalar-``(S,I)=(0,0)`` and axial-``(1,1)``
probabilities; this is a derived property of this two-flavor SU(4) restriction
of the usual SU(6)-symmetric quark-model regulator member, not a
phenomenological scalar/axial fit.

The source-owned C53 kernel is then applied to all three quark slots together
with the exact C7 color insertions.  Its spin-scalar part cancels by color
neutrality.  Every surviving channel lies in the total ``S3`` sign sector.
After tracing spin--flavor, the two bare C7 color coordinates have density
``I_2/2``.  Hence the M2j coordinate called ``orthogonal`` is populated with
weight one half, but it is not an independently coupled vertex: it is the
entangled partner required to combine the color and spin--flavor standard
representations into the fermion sign representation.

This module closes the finite equal-share global-ground symmetry algebra and
the source image.  It does not yet supply a complete H2 Hamiltonian, a gauged
interaction current, spin-dependent symmetry breaking, or a physical
nucleon/deuteron state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from math import fsum, sqrt
from typing import Any, Mapping

import numpy as np

from deuteron_wigner.microscopic.h0.color import emitted_gluon_color_amplitudes

from .m2_c53_mixed_singlet_state import (
    C53_MIXED_K9_EXPLORATORY_BASELINE,
    build_c53_mixed_k9_hamiltonian,
)
from .m2_color_singlet_bridge import m2_color_modules
from .m2_permutation_complete_h2 import build_quark_permutation_algebra
from .m2_resolved_spectator_current import (
    M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1,
)


M2L_SPIN_FLAVOR_STATUS = "M2L_GLOBAL_GROUND_SPIN_FLAVOR_S3_COMPLETION_READY"
M2L_C53_SOURCE_STATUS = "M2L_TOTAL_ANTISYMMETRIC_C53_SOURCE_IMAGE_READY"
M2L_CLAIM_TIER = "EXACT_FINITE_K_GLOBAL_GROUND_S3_SOURCE_AUDIT"
M2L_RESOLUTION_ID = "M2L_K27_GLOBAL_GROUND_SPIN_FLAVOR_S3_V1"

TARGETS = ("PROTON", "NEUTRON")
HELICITIES_TWICE = (-1, 1)
SINGLE_QUARK_STATES = (("u", -1), ("u", 1), ("d", -1), ("d", 1))
SPIN_FLAVOR_BASIS = tuple(product(range(4), repeat=3))

_TOLERANCE = 5.0e-12
_IDENTITY_4 = np.eye(4, dtype=np.complex128)
_IDENTITY_64 = np.eye(64, dtype=np.complex128)


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


def _kron3(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(left, middle), right)


def _slot_operator(operator: np.ndarray, slot: int) -> np.ndarray:
    if operator.shape != (4, 4):
        raise ValueError("one-quark operator must have shape (4,4)")
    if slot not in (0, 1, 2):
        raise IndexError(slot)
    factors = [_IDENTITY_4, _IDENTITY_4, _IDENTITY_4]
    factors[slot] = operator
    return _kron3(*factors)


def _total_one_body(operator: np.ndarray) -> np.ndarray:
    return sum(
        (_slot_operator(operator, slot) for slot in range(3)),
        np.zeros((64, 64), dtype=np.complex128),
    )


def _single_particle_generators() -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    spin_z = np.diag((-0.5, 0.5, -0.5, 0.5)).astype(np.complex128)
    spin_plus = np.zeros((4, 4), dtype=np.complex128)
    spin_plus[1, 0] = 1.0
    spin_plus[3, 2] = 1.0
    spin_x = 0.5 * (spin_plus + spin_plus.conj().T)
    spin_y = (spin_plus - spin_plus.conj().T) / (2.0j)

    isospin_z = np.diag((0.5, 0.5, -0.5, -0.5)).astype(np.complex128)
    isospin_plus = np.zeros((4, 4), dtype=np.complex128)
    isospin_plus[0, 2] = 1.0
    isospin_plus[1, 3] = 1.0
    isospin_x = 0.5 * (isospin_plus + isospin_plus.conj().T)
    isospin_y = (isospin_plus - isospin_plus.conj().T) / (2.0j)
    return (
        (spin_x, spin_y, spin_z),
        (isospin_x, isospin_y, isospin_z),
    )


def spin_flavor_permutation_representation(
    permutation: tuple[int, int, int],
) -> np.ndarray:
    """Return the ordered-slot permutation in the 64-state tensor cube."""

    if permutation not in tuple(build_quark_permutation_algebra().group):
        raise ValueError("permutation must be an element of S3")
    result = np.zeros((64, 64), dtype=np.complex128)
    for column, state in enumerate(SPIN_FLAVOR_BASIS):
        tensor = np.zeros((4, 4, 4), dtype=np.complex128)
        tensor[state] = 1.0
        result[:, column] = np.transpose(tensor, axes=permutation).reshape(-1)
    return result


@dataclass(frozen=True)
class SpinFlavorS3Algebra:
    group: tuple[tuple[int, int, int], ...]
    representations: tuple[np.ndarray, ...]
    symmetric_projector: np.ndarray
    spin_squared: np.ndarray
    isospin_squared: np.ndarray
    total_spin_z: np.ndarray
    total_isospin_z: np.ndarray
    qqqg_total_antisymmetrizer: np.ndarray
    representation_residual: float
    projector_residual: float
    qqqg_sign_rank: int


@lru_cache(maxsize=1)
def build_spin_flavor_s3_algebra() -> SpinFlavorS3Algebra:
    color = build_quark_permutation_algebra()
    group = tuple(color.group)
    representations = tuple(
        spin_flavor_permutation_representation(item) for item in group
    )
    symmetric = sum(representations) / float(len(group))
    spin_one, isospin_one = _single_particle_generators()
    spin = tuple(_total_one_body(item) for item in spin_one)
    isospin = tuple(_total_one_body(item) for item in isospin_one)
    spin_squared = sum(
        (item @ item for item in spin),
        np.zeros((64, 64), dtype=np.complex128),
    )
    isospin_squared = sum(
        (item @ item for item in isospin),
        np.zeros((64, 64), dtype=np.complex128),
    )
    antisymmetrizer = sum(
        (
            _permutation_sign(item)
            * np.kron(color.qqqg_color_representations[index], representations[index])
            for index, item in enumerate(group)
        ),
        np.zeros((128, 128), dtype=np.complex128),
    ) / float(len(group))

    representation_residual = 0.0
    for left_index, left in enumerate(group):
        for right_index, right in enumerate(group):
            product_value = _compose(right, left)
            product_index = group.index(product_value)
            representation_residual = max(
                representation_residual,
                float(
                    np.max(
                        np.abs(
                            representations[left_index]
                            @ representations[right_index]
                            - representations[product_index]
                        )
                    )
                ),
            )
    projector_residual = max(
        float(np.max(np.abs(symmetric - symmetric.conj().T))),
        float(np.max(np.abs(symmetric @ symmetric - symmetric))),
        float(np.max(np.abs(antisymmetrizer - antisymmetrizer.conj().T))),
        float(np.max(np.abs(antisymmetrizer @ antisymmetrizer - antisymmetrizer))),
    )
    result = SpinFlavorS3Algebra(
        group=group,
        representations=representations,
        symmetric_projector=symmetric,
        spin_squared=spin_squared,
        isospin_squared=isospin_squared,
        total_spin_z=spin[2],
        total_isospin_z=isospin[2],
        qqqg_total_antisymmetrizer=antisymmetrizer,
        representation_residual=representation_residual,
        projector_residual=projector_residual,
        qqqg_sign_rank=int(np.linalg.matrix_rank(antisymmetrizer, tol=1.0e-10)),
    )
    if (
        result.qqqg_sign_rank != 20
        or max(result.representation_residual, result.projector_residual) > _TOLERANCE
    ):
        raise RuntimeError("spin-flavor S3 algebra did not close")
    return result


def _weight_indices(target: str, helicity_twice: int) -> tuple[int, ...]:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if helicity_twice not in HELICITIES_TWICE:
        raise ValueError("helicity_twice must be -1 or 1")
    required_u = 2 if target == "PROTON" else 1
    return tuple(
        index
        for index, state in enumerate(SPIN_FLAVOR_BASIS)
        if sum(SINGLE_QUARK_STATES[item][0] == "u" for item in state) == required_u
        and sum(SINGLE_QUARK_STATES[item][1] for item in state) == helicity_twice
    )


def _canonical_projector_vector(projector: np.ndarray) -> np.ndarray:
    basis = _canonical_projector_basis(projector)
    if basis.shape[1] != 1:
        raise RuntimeError("the requested projector image is not one-dimensional")
    return basis[:, 0]


def _canonical_projector_basis(projector: np.ndarray) -> np.ndarray:
    """Return a deterministic orthonormal basis for a projector image."""

    matrix = np.asarray(0.5 * (projector + projector.conj().T), dtype=np.complex128)
    expected_rank = int(round(float(np.trace(matrix).real)))
    columns: list[np.ndarray] = []
    for index in range(matrix.shape[1]):
        vector = matrix[:, index].copy()
        for previous in columns:
            vector -= previous * np.vdot(previous, vector)
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-10:
            continue
        vector /= norm
        pivot = int(np.flatnonzero(np.abs(vector) > 1.0e-10)[0])
        vector *= np.exp(-1j * np.angle(vector[pivot]))
        columns.append(vector)
        if len(columns) == expected_rank:
            break
    if len(columns) != expected_rank:
        raise RuntimeError("could not construct the complete projector image")
    return np.column_stack(columns)


def _spectator_pair_channel_projectors() -> Mapping[str, np.ndarray]:
    spin_one, isospin_one = _single_particle_generators()
    pair_spin = tuple(
        _slot_operator(item, 1) + _slot_operator(item, 2) for item in spin_one
    )
    pair_isospin = tuple(
        _slot_operator(item, 1) + _slot_operator(item, 2) for item in isospin_one
    )
    pair_spin_squared = sum(
        (item @ item for item in pair_spin),
        np.zeros((64, 64), dtype=np.complex128),
    )
    pair_isospin_squared = sum(
        (item @ item for item in pair_isospin),
        np.zeros((64, 64), dtype=np.complex128),
    )
    spin_one_projector = pair_spin_squared / 2.0
    spin_zero_projector = _IDENTITY_64 - spin_one_projector
    isospin_one_projector = pair_isospin_squared / 2.0
    isospin_zero_projector = _IDENTITY_64 - isospin_one_projector
    return {
        "SCALAR_I0_S0_L0": spin_zero_projector @ isospin_zero_projector,
        "AXIAL_I1_S1_L0": spin_one_projector @ isospin_one_projector,
        "MIXED_I1_S0_L0": spin_zero_projector @ isospin_one_projector,
        "MIXED_I0_S1_L0": spin_one_projector @ isospin_zero_projector,
    }


@dataclass(frozen=True)
class GlobalGroundNucleonSpinFlavorState:
    target: str
    helicity_twice: int
    vector: np.ndarray
    scalar_spectator_weight: float
    axial_spectator_weight: float
    mixed_spectator_weight: float
    normalization_residual: float
    symmetric_residual: float
    spin_half_residual: float
    isospin_half_residual: float
    qqq_total_sign_residual: float


@lru_cache(maxsize=4)
def build_global_ground_nucleon_spin_flavor_state(
    target: str, helicity_twice: int
) -> GlobalGroundNucleonSpinFlavorState:
    algebra = build_spin_flavor_s3_algebra()
    indices = _weight_indices(target, helicity_twice)
    weight = np.zeros((64, 64), dtype=np.complex128)
    weight[indices, indices] = 1.0
    spin_half = (3.75 * _IDENTITY_64 - algebra.spin_squared) / 3.0
    isospin_half = (3.75 * _IDENTITY_64 - algebra.isospin_squared) / 3.0
    projector = (
        weight @ algebra.symmetric_projector @ spin_half @ isospin_half @ weight
    )
    vector = _canonical_projector_vector(projector)
    channels = _spectator_pair_channel_projectors()
    weights = {
        key: float(np.vdot(vector, value @ vector).real)
        for key, value in channels.items()
    }
    color = build_quark_permutation_algebra()
    qqq_sign_residual = max(
        float(
            np.linalg.norm(
                color.qqq_color_representations[index][0, 0]
                * algebra.representations[index]
                @ vector
                - _permutation_sign(item) * vector
            )
        )
        for index, item in enumerate(algebra.group)
    )
    result = GlobalGroundNucleonSpinFlavorState(
        target=target,
        helicity_twice=helicity_twice,
        vector=vector,
        scalar_spectator_weight=weights["SCALAR_I0_S0_L0"],
        axial_spectator_weight=weights["AXIAL_I1_S1_L0"],
        mixed_spectator_weight=(
            weights["MIXED_I1_S0_L0"] + weights["MIXED_I0_S1_L0"]
        ),
        normalization_residual=float(abs(np.vdot(vector, vector) - 1.0)),
        symmetric_residual=float(
            np.linalg.norm(algebra.symmetric_projector @ vector - vector)
        ),
        spin_half_residual=float(
            np.linalg.norm(algebra.spin_squared @ vector - 0.75 * vector)
        ),
        isospin_half_residual=float(
            np.linalg.norm(algebra.isospin_squared @ vector - 0.75 * vector)
        ),
        qqq_total_sign_residual=qqq_sign_residual,
    )
    if max(
        result.normalization_residual,
        result.symmetric_residual,
        result.spin_half_residual,
        result.isospin_half_residual,
        result.qqq_total_sign_residual,
        abs(result.scalar_spectator_weight - 0.5),
        abs(result.axial_spectator_weight - 0.5),
        result.mixed_spectator_weight,
    ) > _TOLERANCE:
        raise RuntimeError("global-ground nucleon spin-flavor state did not close")
    return result


@dataclass(frozen=True)
class TotalSignSpinFlavorCarrier:
    """Exact qqqg total-sign carrier at fixed flavor weight and quark Sz."""

    target: str
    quark_spin_twice: int
    basis: np.ndarray
    rank: int
    projector_residual: float
    orthonormality_residual: float
    total_sign_residual: float
    isospin_half_residual: float
    spin_projection_residual: float


@lru_cache(maxsize=8)
def build_total_sign_spin_flavor_carrier(
    target: str, quark_spin_twice: int
) -> TotalSignSpinFlavorCarrier:
    """Resolve the full allowed internal carrier, not only the C53 source ray.

    The gluon is flavorless and is external to this 128-state color times
    three-quark spin--flavor carrier.  Thus proton/neutron fixes the quark
    isospin projection while ``quark_spin_twice`` fixes the sum of the three
    quark helicities.  The image is then restricted to total isospin one half
    and to the exact color--spin-flavor sign representation.
    """

    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if quark_spin_twice not in (-3, -1, 1, 3):
        raise ValueError("quark_spin_twice must be one of -3, -1, 1, 3")
    algebra = build_spin_flavor_s3_algebra()
    required_u = 2 if target == "PROTON" else 1
    mask = np.asarray(
        [
            sum(SINGLE_QUARK_STATES[item][0] == "u" for item in state)
            == required_u
            and sum(SINGLE_QUARK_STATES[item][1] for item in state)
            == quark_spin_twice
            for state in SPIN_FLAVOR_BASIS
        ],
        dtype=float,
    )
    weight = np.diag(np.tile(mask, 2)).astype(np.complex128)
    identity_color = np.eye(2, dtype=np.complex128)
    isospin_half = np.kron(
        identity_color,
        (3.75 * _IDENTITY_64 - algebra.isospin_squared) / 3.0,
    )
    quark_spin_z = np.kron(identity_color, algebra.total_spin_z)
    projector = (
        weight
        @ algebra.qqqg_total_antisymmetrizer
        @ isospin_half
        @ weight
    )
    projector = np.asarray(0.5 * (projector + projector.conj().T))
    basis = _canonical_projector_basis(projector)
    reconstructed = basis @ basis.conj().T
    expected_spin = quark_spin_twice / 2.0
    result = TotalSignSpinFlavorCarrier(
        target=target,
        quark_spin_twice=quark_spin_twice,
        basis=basis,
        rank=basis.shape[1],
        projector_residual=max(
            float(np.max(np.abs(projector @ projector - projector))),
            float(np.max(np.abs(projector - reconstructed))),
        ),
        orthonormality_residual=float(
            np.max(np.abs(basis.conj().T @ basis - np.eye(basis.shape[1])))
        ),
        total_sign_residual=float(
            np.max(
                np.abs(algebra.qqqg_total_antisymmetrizer @ basis - basis)
            )
        ),
        isospin_half_residual=float(
            np.max(np.abs(isospin_half @ basis - basis))
        ),
        spin_projection_residual=float(
            np.max(np.abs(quark_spin_z @ basis - expected_spin * basis))
        ),
    )
    expected_rank = 1 if abs(quark_spin_twice) == 3 else 2
    if result.rank != expected_rank or max(
        result.projector_residual,
        result.orthonormality_residual,
        result.total_sign_residual,
        result.isospin_half_residual,
        result.spin_projection_residual,
    ) > _TOLERANCE:
        raise RuntimeError("fixed-weight total-sign carrier did not close")
    return result


def _spin_kernel_on_slot(kernel: np.ndarray, slot: int) -> np.ndarray:
    if kernel.shape != (2, 2):
        raise ValueError("spin kernel must have shape (2,2)")
    spin_flavor = np.zeros((4, 4), dtype=np.complex128)
    spin_flavor[:2, :2] = kernel
    spin_flavor[2:, 2:] = kernel
    return _slot_operator(spin_flavor, slot)


def _c53_channel_kernels() -> tuple[
    tuple[tuple[int, int, int, int], np.ndarray], ...
]:
    assembly = build_c53_mixed_k9_hamiltonian(
        C53_MIXED_K9_EXPLORATORY_BASELINE
    )
    direct = assembly.direct_c53_emission.toarray()
    color_norm = float(np.linalg.norm(emitted_gluon_color_amplitudes(0)))
    colorless = direct / color_norm
    labels = m2_color_modules("K9").multiplicity_labels[2:]
    grouped: dict[tuple[int, int, int, int], dict[int, int]] = {}
    for row, label in enumerate(labels):
        _, partition, n_value, m_value, quark_helicity, gluon_helicity = label
        key = (
            int(partition),
            int(n_value),
            int(m_value),
            int(gluon_helicity),
        )
        grouped.setdefault(key, {})[int(quark_helicity)] = row
    result = []
    for key, rows in grouped.items():
        if tuple(sorted(rows)) != HELICITIES_TWICE:
            raise RuntimeError("C53 channel does not contain both final helicities")
        kernel = np.zeros((2, 2), dtype=np.complex128)
        for final_index, final_helicity in enumerate(HELICITIES_TWICE):
            kernel[final_index, :] = colorless[rows[final_helicity], :]
        result.append((key, kernel))
    reconstruction_residual = float(
        np.max(np.abs(colorless * color_norm - direct))
    )
    if reconstruction_residual > _TOLERANCE:
        raise RuntimeError("C53 colorless-kernel factorization failed")
    return tuple(result)


@dataclass(frozen=True)
class C53SpinFlavorSourceChannel:
    mode_key: tuple[int, int, int, int]
    target: str
    helicity_twice: int
    spin_kernel: np.ndarray
    source_vector: np.ndarray
    norm: float

    @property
    def normalized_vector(self) -> np.ndarray:
        return self.source_vector / self.norm


@lru_cache(maxsize=4)
def build_c53_spin_flavor_source_channels(
    target: str, helicity_twice: int
) -> tuple[C53SpinFlavorSourceChannel, ...]:
    """Return every nonzero emitter-covariant C53 source component."""

    source = build_global_ground_nucleon_spin_flavor_state(
        target, helicity_twice
    ).vector
    color_vectors = tuple(emitted_gluon_color_amplitudes(slot) for slot in range(3))
    result = []
    for key, kernel in _c53_channel_kernels():
        if np.linalg.norm(kernel) <= 1.0e-13:
            continue
        full = sum(
            (
                np.kron(
                    color_vectors[slot],
                    _spin_kernel_on_slot(kernel, slot) @ source,
                )
                for slot in range(3)
            ),
            np.zeros(128, dtype=np.complex128),
        )
        norm = float(np.linalg.norm(full))
        if norm <= 1.0e-13:
            continue
        result.append(
            C53SpinFlavorSourceChannel(
                mode_key=key,
                target=target,
                helicity_twice=helicity_twice,
                spin_kernel=kernel,
                source_vector=full,
                norm=norm,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class C53SpinFlavorOrbitAudit:
    nonzero_colorless_matrix_entries: int
    nonzero_spin_channels: int
    source_reachable_channels: int
    qqqg_total_sign_rank: int
    color_insertion_norm: float
    color_neutrality_residual: float
    scalar_emission_cancellation_residual: float
    traceless_spin_reconstruction_residual: float
    total_sign_projection_residual: float
    isospin_half_residual: float
    total_jz_residual: float
    reduced_color_isotropy_residual: float
    active_color_weight_range: tuple[float, float]
    orthogonal_color_weight_range: tuple[float, float]
    source_norm_squared_by_state: Mapping[str, float]
    proton_neutron_source_norm_residual: float
    helicity_source_norm_residual: float


@lru_cache(maxsize=1)
def build_c53_spin_flavor_orbit_audit() -> C53SpinFlavorOrbitAudit:
    algebra = build_spin_flavor_s3_algebra()
    color = build_quark_permutation_algebra()
    color_vectors = tuple(emitted_gluon_color_amplitudes(slot) for slot in range(3))
    color_norms = tuple(float(np.linalg.norm(item)) for item in color_vectors)
    color_norm = color_norms[0]
    color_neutrality = float(np.linalg.norm(sum(color_vectors)))
    kernels = _c53_channel_kernels()
    nonzero_matrix_entries = sum(
        int(np.count_nonzero(np.abs(kernel) > 1.0e-13)) for _, kernel in kernels
    )

    scalar_residual = 0.0
    traceless_residual = 0.0
    sign_residual = 0.0
    isospin_residual = 0.0
    jz_residual = 0.0
    isotropy_residual = 0.0
    active_weights: list[float] = []
    orthogonal_weights: list[float] = []
    source_norms: dict[str, float] = {}
    reachable_counts: list[int] = []
    identity_color = np.eye(2, dtype=np.complex128)
    combined_isospin_squared = np.kron(identity_color, algebra.isospin_squared)
    combined_spin_z = np.kron(identity_color, algebra.total_spin_z)

    for target in TARGETS:
        for helicity_twice in HELICITIES_TWICE:
            source = build_global_ground_nucleon_spin_flavor_state(
                target, helicity_twice
            ).vector
            channel_norms_squared: list[float] = []
            reachable = 0
            for key, kernel in kernels:
                if np.linalg.norm(kernel) <= 1.0e-13:
                    continue
                scalar_kernel = 0.5 * np.trace(kernel) * np.eye(2)
                traceless_kernel = kernel - scalar_kernel
                full = sum(
                    (
                        np.kron(
                            color_vectors[slot],
                            _spin_kernel_on_slot(kernel, slot) @ source,
                        )
                        for slot in range(3)
                    ),
                    np.zeros(128, dtype=np.complex128),
                )
                scalar_part = sum(
                    (
                        np.kron(
                            color_vectors[slot],
                            _spin_kernel_on_slot(scalar_kernel, slot) @ source,
                        )
                        for slot in range(3)
                    ),
                    np.zeros(128, dtype=np.complex128),
                )
                traceless_part = sum(
                    (
                        np.kron(
                            color_vectors[slot],
                            _spin_kernel_on_slot(traceless_kernel, slot) @ source,
                        )
                        for slot in range(3)
                    ),
                    np.zeros(128, dtype=np.complex128),
                )
                scalar_residual = max(scalar_residual, float(np.linalg.norm(scalar_part)))
                traceless_residual = max(
                    traceless_residual, float(np.linalg.norm(full - traceless_part))
                )
                if np.linalg.norm(full) <= 1.0e-13:
                    continue
                reachable += 1
                projected = algebra.qqqg_total_antisymmetrizer @ full
                sign_residual = max(
                    sign_residual, float(np.linalg.norm(projected - full))
                )
                isospin_residual = max(
                    isospin_residual,
                    float(
                        np.linalg.norm(
                            combined_isospin_squared @ full - 0.75 * full
                        )
                    ),
                )
                _, _, m_value, gluon_helicity = key
                expected_quark_spin_z = (
                    helicity_twice / 2.0 - gluon_helicity - m_value
                )
                jz_residual = max(
                    jz_residual,
                    float(
                        np.linalg.norm(
                            combined_spin_z @ full - expected_quark_spin_z * full
                        )
                    ),
                )
                shaped = full.reshape(2, 64)
                density = shaped @ shaped.conj().T
                norm_squared = float(np.trace(density).real)
                normalized_density = density / norm_squared
                isotropy_residual = max(
                    isotropy_residual,
                    float(np.max(np.abs(normalized_density - 0.5 * identity_color))),
                )
                active_weights.append(
                    float(
                        np.vdot(
                            color.active_emitter_coordinates,
                            normalized_density @ color.active_emitter_coordinates,
                        ).real
                    )
                )
                orthogonal_weights.append(
                    float(
                        np.vdot(
                            color.orthogonal_emitter_coordinates,
                            normalized_density
                            @ color.orthogonal_emitter_coordinates,
                        ).real
                    )
                )
                channel_norms_squared.append(norm_squared)
            total_norm_squared = fsum(channel_norms_squared)
            source_norms[f"{target}:JZ{helicity_twice:+d}/2"] = total_norm_squared
            reachable_counts.append(reachable)

    source_values = tuple(source_norms.values())
    proton_neutron_residual = max(
        abs(source_norms[f"PROTON:JZ{h:+d}/2"] - source_norms[f"NEUTRON:JZ{h:+d}/2"])
        for h in HELICITIES_TWICE
    )
    helicity_residual = max(
        abs(source_norms[f"{target}:JZ-1/2"] - source_norms[f"{target}:JZ+1/2"])
        for target in TARGETS
    )
    result = C53SpinFlavorOrbitAudit(
        nonzero_colorless_matrix_entries=nonzero_matrix_entries,
        nonzero_spin_channels=sum(
            np.linalg.norm(kernel) > 1.0e-13 for _, kernel in kernels
        ),
        source_reachable_channels=min(reachable_counts),
        qqqg_total_sign_rank=algebra.qqqg_sign_rank,
        color_insertion_norm=color_norm,
        color_neutrality_residual=color_neutrality,
        scalar_emission_cancellation_residual=scalar_residual,
        traceless_spin_reconstruction_residual=traceless_residual,
        total_sign_projection_residual=sign_residual,
        isospin_half_residual=isospin_residual,
        total_jz_residual=jz_residual,
        reduced_color_isotropy_residual=isotropy_residual,
        active_color_weight_range=(min(active_weights), max(active_weights)),
        orthogonal_color_weight_range=(
            min(orthogonal_weights),
            max(orthogonal_weights),
        ),
        source_norm_squared_by_state=source_norms,
        proton_neutron_source_norm_residual=float(proton_neutron_residual),
        helicity_source_norm_residual=float(helicity_residual),
    )
    if (
        result.nonzero_colorless_matrix_entries != 104
        or result.nonzero_spin_channels != 80
        or result.source_reachable_channels != 80
        or result.qqqg_total_sign_rank != 20
        or abs(result.color_insertion_norm - sqrt(4.0 / 3.0)) > _TOLERANCE
        or max(
            result.color_neutrality_residual,
            result.scalar_emission_cancellation_residual,
            result.traceless_spin_reconstruction_residual,
            result.total_sign_projection_residual,
            result.isospin_half_residual,
            result.total_jz_residual,
            result.reduced_color_isotropy_residual,
            result.proton_neutron_source_norm_residual,
            result.helicity_source_norm_residual,
            max(abs(value - 0.5) for value in result.active_color_weight_range),
            max(abs(value - 0.5) for value in result.orthogonal_color_weight_range),
        )
        > _TOLERANCE
    ):
        raise RuntimeError("C53 spin-flavor emitter orbit did not close")
    if min(source_values) <= 0.0:
        raise RuntimeError("C53 spin-flavor source image is empty")
    return result


def spin_flavor_s3_completion_record() -> Mapping[str, Any]:
    algebra = build_spin_flavor_s3_algebra()
    states = tuple(
        build_global_ground_nucleon_spin_flavor_state(target, helicity)
        for target in TARGETS
        for helicity in HELICITIES_TWICE
    )
    orbit = build_c53_spin_flavor_orbit_audit()
    resolution = M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1
    payload = {
        "status": M2L_SPIN_FLAVOR_STATUS,
        "c53_source_status": M2L_C53_SOURCE_STATUS,
        "claim_tier": M2L_CLAIM_TIER,
        "physical": False,
        "resolution_id": M2L_RESOLUTION_ID,
        "active_resolution": resolution.active_resolution,
        "nucleon_K2": resolution.nucleon_K2,
        "orbital_member": "S3_SCALAR_GLOBAL_JACOBI_GROUND",
        "b_rho_GeV": sqrt(resolution.spectator_cluster_fraction)
        * resolution.b_lambda_GeV,
        "b_lambda_GeV": resolution.b_lambda_GeV,
        "single_quark_spin_flavor_order": SINGLE_QUARK_STATES,
        "ordered_three_quark_spin_flavor_dimension": 64,
        "qqqg_color_times_spin_flavor_dimension": 128,
        "qqqg_total_sign_rank": algebra.qqqg_sign_rank,
        "qqqg_fixed_target_spin_carrier_ranks": {
            f"{target}:2Sz={spin_twice:+d}": build_total_sign_spin_flavor_carrier(
                target, spin_twice
            ).rank
            for target in TARGETS
            for spin_twice in (-3, -1, 1, 3)
        },
        "s3_representation_residual": algebra.representation_residual,
        "s3_projector_residual": algebra.projector_residual,
        "nucleon_states": tuple(
            {
                "target": state.target,
                "helicity_twice": state.helicity_twice,
                "scalar_spectator_weight": state.scalar_spectator_weight,
                "axial_spectator_weight": state.axial_spectator_weight,
                "mixed_spectator_weight": state.mixed_spectator_weight,
                "normalization_residual": state.normalization_residual,
                "symmetric_residual": state.symmetric_residual,
                "spin_half_residual": state.spin_half_residual,
                "isospin_half_residual": state.isospin_half_residual,
                "qqq_total_sign_residual": state.qqq_total_sign_residual,
            }
            for state in states
        ),
        "scalar_axial_interpretation": (
            "equal coherent scalar/axial recoupling is derived only for this "
            "common-orbital two-flavor SU(4) restriction of the usual "
            "SU(6)-symmetric global-ground member"
        ),
        "c53_nonzero_colorless_matrix_entries": (
            orbit.nonzero_colorless_matrix_entries
        ),
        "c53_nonzero_spin_channels": orbit.nonzero_spin_channels,
        "c53_source_reachable_channels": orbit.source_reachable_channels,
        "c53_color_insertion_norm": orbit.color_insertion_norm,
        "c53_color_neutrality_residual": orbit.color_neutrality_residual,
        "c53_scalar_emission_cancellation_residual": (
            orbit.scalar_emission_cancellation_residual
        ),
        "c53_traceless_spin_reconstruction_residual": (
            orbit.traceless_spin_reconstruction_residual
        ),
        "c53_total_sign_projection_residual": (
            orbit.total_sign_projection_residual
        ),
        "c53_isospin_half_residual": orbit.isospin_half_residual,
        "c53_total_jz_residual": orbit.total_jz_residual,
        "c53_reduced_color_isotropy_residual": (
            orbit.reduced_color_isotropy_residual
        ),
        "c53_active_color_weight_range": orbit.active_color_weight_range,
        "c53_orthogonal_color_weight_range": (
            orbit.orthogonal_color_weight_range
        ),
        "c53_source_norm_squared_by_state": orbit.source_norm_squared_by_state,
        "orthogonal_qqqg_singlet_status": (
            "POPULATED_AS_REQUIRED_ENTANGLED_COLOR_COORDINATE_NOT_AN_"
            "INDEPENDENT_VERTEX"
        ),
        "independent_orthogonal_c53_coupling": False,
        "complete_noncolor_quark_antisymmetrizer_applied": True,
        "complete_h2_hamiltonian_constructed": False,
        "hamiltonian_gauged_current_constructed": False,
        "spin_flavor_symmetry_breaking_constructed": False,
        "physical_nucleon_or_deuteron_state_constructed": False,
        "production_observable_claimed": False,
        "next_required_object": (
            "represent the source-supported H2 Hamiltonian on this total-sign "
            "carrier, then gauge the identical finite Hamiltonian and regulator"
        ),
    }
    return {**_plain(payload), "root": _json_root(payload)}


__all__ = [
    "M2L_SPIN_FLAVOR_STATUS",
    "M2L_C53_SOURCE_STATUS",
    "M2L_CLAIM_TIER",
    "M2L_RESOLUTION_ID",
    "TARGETS",
    "HELICITIES_TWICE",
    "SINGLE_QUARK_STATES",
    "SPIN_FLAVOR_BASIS",
    "SpinFlavorS3Algebra",
    "GlobalGroundNucleonSpinFlavorState",
    "TotalSignSpinFlavorCarrier",
    "C53SpinFlavorSourceChannel",
    "C53SpinFlavorOrbitAudit",
    "spin_flavor_permutation_representation",
    "build_spin_flavor_s3_algebra",
    "build_global_ground_nucleon_spin_flavor_state",
    "build_total_sign_spin_flavor_carrier",
    "build_c53_spin_flavor_source_channels",
    "build_c53_spin_flavor_orbit_audit",
    "spin_flavor_s3_completion_record",
]

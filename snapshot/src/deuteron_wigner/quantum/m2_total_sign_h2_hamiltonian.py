"""Source-supported total-sign H2 Hamiltonian for the M2 global ground.

M2l identified the exact global-ground nucleon spin--flavor states and the
total-antisymmetric C53 source image.  This module supplies the next finite
Hamiltonian object.  It embeds the represented K9 active-cluster operator at
``y_A=1/3``, uses the permutation-symmetric Jacobi width
``b_rho=sqrt(2/3)b_lambda``, and retains every external qg mode reached by
C53.

The one C53 source ray per external mode is not, by itself, an invariant
carrier when the fixed-spin total-sign space has rank two.  The Hamiltonian
therefore uses the complete color times spin--flavor sign carrier at fixed
target isospin projection and quark spin projection.  Each Jz block has one
qqq state and 132 qqqg states.  The represented H0, C396, and C117 terms are
spin/color scalars on this carrier and close the 80-mode C53 support exactly.

This is an exact finite-matrix realization of the *represented,
source-supported* H2 terms.  It is not the complete physical H2 Hamiltonian:
spectator/full-orbit C117 interactions, instantaneous and constrained-field
terms, induced operators, zero modes, counterterms, and spin-dependent
symmetry breaking remain unavailable.  No fitted nucleon mass or production
observable is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
from math import sqrt
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from .m2_c53_mixed_singlet_state import (
    C53_PARAMETER_ID,
    C53_MIXED_K9_EXPLORATORY_BASELINE,
    ExploratoryC53K9Point,
    build_c53_mixed_k9_hamiltonian,
)
from .m2_color_singlet_bridge import m2_color_modules
from .m2_resolved_spectator_current import (
    M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1,
)
from .m2_spin_flavor_s3_completion import (
    HELICITIES_TWICE,
    TARGETS,
    build_c53_spin_flavor_source_channels,
    build_total_sign_spin_flavor_carrier,
)


M2M_HAMILTONIAN_STATUS = "M2M_SOURCE_SUPPORTED_TOTAL_SIGN_H2_HAMILTONIAN_READY"
M2M_CLAIM_TIER = "EXACT_FINITE_K_REPRESENTED_SOURCE_SUPPORTED_H2"
M2M_RESOLUTION_ID = "M2M_K27_GLOBAL_GROUND_TOTAL_SIGN_H2_V1"

UNAVAILABLE_COMPLETE_H2_TERMS = (
    "FULL_SPECTATOR_AND_ALL_EMITTER_C117_INTERACTION",
    "INSTANTANEOUS_INTERACTION",
    "CONSTRAINED_FIELD_INTERACTION",
    "INDUCED_EFFECTIVE_INTERACTION",
    "ZERO_MODE_INTERACTION",
    "COUNTERTERM_AND_OPERATOR_RENORMALIZATION_MATCHING",
    "SPIN_DEPENDENT_GLOBAL_GROUND_SYMMETRY_BREAKING",
)

_C117_PARAMETER_ID = "c117:I2_density_projector"
_TOLERANCE = 5.0e-11


def _max_abs(matrix: sparse.spmatrix | np.ndarray) -> float:
    if sparse.issparse(matrix):
        values = matrix.tocsr().data
        return float(np.max(np.abs(values))) if values.size else 0.0
    values = np.asarray(matrix)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _sparse_root(matrix: sparse.csr_matrix) -> str:
    value = matrix.tocsr()
    digest = sha256()
    for array in (value.indptr, value.indices, value.data):
        contiguous = np.ascontiguousarray(array)
        if np.issubdtype(contiguous.dtype, np.inexact):
            contiguous = np.round(contiguous, 14)
            contiguous[contiguous == 0] = 0
        digest.update(contiguous.dtype.str.encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


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


@dataclass(frozen=True)
class GlobalGroundMassEmbedding:
    active_fraction: float
    spectator_fraction: float
    spectator_internal_fractions: tuple[float, float]
    b_rho_GeV: float
    b_lambda_GeV: float
    rho_k2_expectation_GeV2: float
    lambda_k2_expectation_GeV2: float
    spectator_cluster_mass_sq_GeV2: float
    active_mass_scale: float
    additive_identity_shift_GeV2: float


def _global_ground_mass_embedding(point: ExploratoryC53K9Point) -> GlobalGroundMassEmbedding:
    resolution = M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1
    y_active = float(resolution.active_cluster_fraction)
    y_spectator = float(resolution.spectator_cluster_fraction)
    z_values = tuple(float(value) for value in resolution.spectator_internal_fractions)
    b_lambda = float(resolution.b_lambda_GeV)
    b_rho = sqrt(y_spectator) * b_lambda
    rho_k2 = b_rho**2
    lambda_k2 = b_lambda**2
    mass_sq = float(point.base_point.mu_q_sq_GeV2)
    spectator_mass_sq = sum(mass_sq / value for value in z_values) + rho_k2
    shift = spectator_mass_sq / y_spectator + lambda_k2
    return GlobalGroundMassEmbedding(
        active_fraction=y_active,
        spectator_fraction=y_spectator,
        spectator_internal_fractions=z_values,
        b_rho_GeV=b_rho,
        b_lambda_GeV=b_lambda,
        rho_k2_expectation_GeV2=rho_k2,
        lambda_k2_expectation_GeV2=lambda_k2,
        spectator_cluster_mass_sq_GeV2=spectator_mass_sq,
        active_mass_scale=1.0 / y_active,
        additive_identity_shift_GeV2=shift,
    )


@dataclass(frozen=True)
class SpinScalarExternalReduction:
    q_block: np.ndarray
    all_mode_keys: tuple[tuple[int, int, int, int], ...]
    external_operator: np.ndarray
    retained_operator: np.ndarray
    q_spin_scalar_residual: float
    q_qg_cross_residual: float
    qg_spin_scalar_residual: float
    retained_support_leakage_residual: float


def reduce_spin_scalar_external_operator(
    matrix: sparse.csr_matrix,
    retained_keys: tuple[tuple[int, int, int, int], ...],
) -> SpinScalarExternalReduction:
    """Strip the final active-quark helicity from a scalar K9 operator."""

    labels = m2_color_modules("K9").multiplicity_labels
    grouped: dict[tuple[int, int, int, int], dict[int, int]] = {}
    q_indices: dict[int, int] = {}
    for index, label in enumerate(labels):
        if label[0] == "q":
            q_indices[int(label[1])] = index
            continue
        _, partition, n_value, m_value, quark_helicity, gluon_helicity = label
        key = (int(partition), int(n_value), int(m_value), int(gluon_helicity))
        grouped.setdefault(key, {})[int(quark_helicity)] = index
    if tuple(sorted(q_indices)) != HELICITIES_TWICE:
        raise RuntimeError("unexpected K9 q-helicity ordering")
    all_keys = tuple(grouped)
    if any(tuple(sorted(grouped[key])) != HELICITIES_TWICE for key in all_keys):
        raise RuntimeError("a K9 external mode lacks one quark helicity")
    missing = set(retained_keys) - set(all_keys)
    if missing:
        raise RuntimeError(f"C53 support contains unknown K9 modes: {sorted(missing)}")

    dense = np.asarray(matrix.toarray(), dtype=np.complex128)
    minus = np.asarray([grouped[key][-1] for key in all_keys], dtype=int)
    plus = np.asarray([grouped[key][1] for key in all_keys], dtype=int)
    minus_minus = dense[np.ix_(minus, minus)]
    plus_plus = dense[np.ix_(plus, plus)]
    minus_plus = dense[np.ix_(minus, plus)]
    plus_minus = dense[np.ix_(plus, minus)]
    external = 0.5 * (minus_minus + plus_plus)
    q_order = np.asarray([q_indices[-1], q_indices[1]], dtype=int)
    q_block = dense[np.ix_(q_order, q_order)]
    qg_order = np.concatenate((minus, plus))
    retained_indices = np.asarray([all_keys.index(key) for key in retained_keys], dtype=int)
    excluded_indices = np.asarray(
        [index for index, key in enumerate(all_keys) if key not in set(retained_keys)],
        dtype=int,
    )
    support_leakage = 0.0
    if excluded_indices.size:
        support_leakage = max(
            _max_abs(external[np.ix_(retained_indices, excluded_indices)]),
            _max_abs(external[np.ix_(excluded_indices, retained_indices)]),
        )
    return SpinScalarExternalReduction(
        q_block=q_block,
        all_mode_keys=all_keys,
        external_operator=external,
        retained_operator=external[np.ix_(retained_indices, retained_indices)],
        q_spin_scalar_residual=max(
            abs(q_block[0, 1]),
            abs(q_block[1, 0]),
            abs(q_block[0, 0] - q_block[1, 1]),
        ),
        q_qg_cross_residual=max(
            _max_abs(dense[np.ix_(q_order, qg_order)]),
            _max_abs(dense[np.ix_(qg_order, q_order)]),
        ),
        qg_spin_scalar_residual=max(
            _max_abs(minus_minus - plus_plus),
            _max_abs(minus_plus),
            _max_abs(plus_minus),
        ),
        retained_support_leakage_residual=support_leakage,
    )


@dataclass(frozen=True)
class TotalSignH2BasisState:
    index: int
    sector: str
    target: str
    helicity_twice: int
    mode_key: tuple[int, int, int, int] | None
    quark_spin_twice: int
    carrier_index: int | None


@dataclass(frozen=True)
class M2TotalSignH2Hamiltonian:
    point: ExploratoryC53K9Point
    target: str
    mass_embedding: GlobalGroundMassEmbedding
    basis_states: tuple[TotalSignH2BasisState, ...]
    retained_mode_keys: tuple[tuple[int, int, int, int], ...]
    fock_diagonal_matrix: sparse.csr_matrix
    c53_emission: sparse.csr_matrix
    matrix: sparse.csr_matrix
    validation: Mapping[str, Any]
    claim_tier: str = M2M_CLAIM_TIER
    physical: bool = False

    @property
    def dimension(self) -> int:
        return self.matrix.shape[0]

    @property
    def q_dimension(self) -> int:
        return 2

    @property
    def qg_dimension(self) -> int:
        return self.dimension - self.q_dimension

    @property
    def hamiltonian_root(self) -> str:
        return _sparse_root(self.matrix)

    @property
    def jz_values(self) -> np.ndarray:
        return np.asarray(
            [state.helicity_twice / 2.0 for state in self.basis_states], dtype=float
        )


@lru_cache(maxsize=8)
def build_total_sign_h2_hamiltonian(
    target: str,
    point: ExploratoryC53K9Point = C53_MIXED_K9_EXPLORATORY_BASELINE,
) -> M2TotalSignH2Hamiltonian:
    """Build the exact represented H2 matrix on the complete sign carrier."""

    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if not isinstance(point, ExploratoryC53K9Point):
        raise TypeError("point must be an ExploratoryC53K9Point")
    source_assembly = build_c53_mixed_k9_hamiltonian(point)
    embedding = _global_ground_mass_embedding(point)
    active_diagonal = source_assembly.h0.reduced_operator.copy()
    for term in source_assembly.base_terms:
        active_diagonal = (
            active_diagonal + float(term.coefficient) * term.matrix
        ).tocsr()
    source_identity = sparse.identity(
        active_diagonal.shape[0], format="csr", dtype=np.complex128
    )
    embedded_diagonal = (
        embedding.active_mass_scale * active_diagonal
        + embedding.additive_identity_shift_GeV2 * source_identity
    ).tocsr()

    source_channels = {
        helicity: build_c53_spin_flavor_source_channels(target, helicity)
        for helicity in HELICITIES_TWICE
    }
    retained_keys = tuple(channel.mode_key for channel in source_channels[-1])
    if tuple(channel.mode_key for channel in source_channels[1]) != retained_keys:
        raise RuntimeError("C53 support differs between helicity sectors")
    reduction = reduce_spin_scalar_external_operator(
        embedded_diagonal, retained_keys
    )
    c117_term = next(
        term
        for term in source_assembly.base_terms
        if term.parameter_id == _C117_PARAMETER_ID
    )
    c117_embedded = (
        embedding.active_mass_scale * float(c117_term.coefficient) * c117_term.matrix
    ).tocsr()
    c117_reduction = reduce_spin_scalar_external_operator(
        c117_embedded, retained_keys
    )

    basis_states: list[TotalSignH2BasisState] = [
        TotalSignH2BasisState(0, "qqq", target, -1, None, -1, None),
        TotalSignH2BasisState(1, "qqq", target, 1, None, 1, None),
    ]
    carrier_bases: dict[tuple[int, tuple[int, int, int, int]], np.ndarray] = {}
    basis_indices: dict[tuple[int, tuple[int, int, int, int]], tuple[int, ...]] = {}
    for helicity in HELICITIES_TWICE:
        for key in retained_keys:
            quark_spin_twice = helicity - 2 * (key[2] + key[3])
            carrier = build_total_sign_spin_flavor_carrier(
                target, quark_spin_twice
            )
            carrier_bases[(helicity, key)] = carrier.basis
            indices = []
            for carrier_index in range(carrier.rank):
                index = len(basis_states)
                indices.append(index)
                basis_states.append(
                    TotalSignH2BasisState(
                        index=index,
                        sector="qqqg",
                        target=target,
                        helicity_twice=helicity,
                        mode_key=key,
                        quark_spin_twice=quark_spin_twice,
                        carrier_index=carrier_index,
                    )
                )
            basis_indices[(helicity, key)] = tuple(indices)

    dimension = len(basis_states)
    fock_dense = np.zeros((dimension, dimension), dtype=np.complex128)
    fock_dense[:2, :2] = reduction.q_block
    for helicity in HELICITIES_TWICE:
        for alpha_index, alpha in enumerate(retained_keys):
            alpha_basis = carrier_bases[(helicity, alpha)]
            alpha_rows = basis_indices[(helicity, alpha)]
            for beta_index, beta in enumerate(retained_keys):
                coefficient = reduction.retained_operator[alpha_index, beta_index]
                if abs(coefficient) <= 1.0e-14:
                    continue
                beta_basis = carrier_bases[(helicity, beta)]
                beta_columns = basis_indices[(helicity, beta)]
                overlap = alpha_basis.conj().T @ beta_basis
                fock_dense[np.ix_(alpha_rows, beta_columns)] = coefficient * overlap

    emission = np.zeros((dimension - 2, 2), dtype=np.complex128)
    source_projection_residual = 0.0
    source_norm_residual = 0.0
    for helicity in HELICITIES_TWICE:
        q_column = 0 if helicity == -1 else 1
        channel_by_key = {
            channel.mode_key: channel for channel in source_channels[helicity]
        }
        for key in retained_keys:
            channel = channel_by_key[key]
            carrier_basis = carrier_bases[(helicity, key)]
            coefficients = carrier_basis.conj().T @ channel.source_vector
            reconstructed = carrier_basis @ coefficients
            source_projection_residual = max(
                source_projection_residual,
                float(np.linalg.norm(reconstructed - channel.source_vector)),
            )
            source_norm_residual = max(
                source_norm_residual,
                abs(float(np.vdot(coefficients, coefficients).real) - channel.norm**2),
            )
            rows = tuple(index - 2 for index in basis_indices[(helicity, key)])
            emission[np.asarray(rows, dtype=int), q_column] = (
                embedding.active_mass_scale
                * float(point.c53_coupling)
                * coefficients
            )

    fock_diagonal = sparse.csr_matrix(fock_dense)
    c53_emission = sparse.csr_matrix(emission)
    c53_block = sparse.bmat(
        [
            [sparse.csr_matrix((2, 2)), c53_emission.getH()],
            [c53_emission, sparse.csr_matrix((dimension - 2, dimension - 2))],
        ],
        format="csr",
    )
    matrix = (fock_diagonal + c53_block).tocsr()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    jz = sparse.diags(
        [state.helicity_twice / 2.0 for state in basis_states],
        format="csr",
        dtype=np.complex128,
    )
    carrier_ranks = tuple(
        len(basis_indices[(helicity, key)])
        for helicity in HELICITIES_TWICE
        for key in retained_keys
    )
    source_norms = tuple(
        sum(channel.norm**2 for channel in source_channels[helicity])
        for helicity in HELICITIES_TWICE
    )
    expected_emission_norms = tuple(
        embedding.active_mass_scale
        * abs(float(point.c53_coupling))
        * sqrt(value)
        for value in source_norms
    )
    actual_emission_norms = tuple(
        float(np.linalg.norm(emission[:, column])) for column in range(2)
    )
    validation = {
        "source_k9_dimension": source_assembly.dimension,
        "all_external_qg_modes": len(reduction.all_mode_keys),
        "retained_c53_external_modes": len(retained_keys),
        "discarded_unreached_external_modes": (
            len(reduction.all_mode_keys) - len(retained_keys)
        ),
        "basis_dimension": dimension,
        "q_dimension": 2,
        "qg_dimension": dimension - 2,
        "jz_block_dimensions": {
            f"JZ{helicity:+d}/2": sum(
                state.helicity_twice == helicity for state in basis_states
            )
            for helicity in HELICITIES_TWICE
        },
        "fixed_mode_carrier_rank_counts": {
            "rank_1": carrier_ranks.count(1),
            "rank_2": carrier_ranks.count(2),
        },
        "one_source_ray_per_mode_dimension": 2 + 2 * len(retained_keys),
        "complete_carrier_extra_dimensions": dimension - (2 + 2 * len(retained_keys)),
        "active_mass_scale": embedding.active_mass_scale,
        "additive_identity_shift_GeV2": embedding.additive_identity_shift_GeV2,
        "q_spin_scalar_residual": reduction.q_spin_scalar_residual,
        "q_qg_cross_residual": reduction.q_qg_cross_residual,
        "qg_spin_scalar_residual": reduction.qg_spin_scalar_residual,
        "retained_support_leakage_residual": (
            reduction.retained_support_leakage_residual
        ),
        "c117_qg_spin_scalar_residual": c117_reduction.qg_spin_scalar_residual,
        "c117_retained_support_leakage_residual": (
            c117_reduction.retained_support_leakage_residual
        ),
        "c53_source_projection_residual": source_projection_residual,
        "c53_source_norm_residual": source_norm_residual,
        "c53_source_norm_squared_by_helicity": source_norms,
        "c53_emission_norms": actual_emission_norms,
        "c53_expected_emission_norms": expected_emission_norms,
        "c53_emission_norm_residual": max(
            abs(actual - expected)
            for actual, expected in zip(actual_emission_norms, expected_emission_norms)
        ),
        "fock_diagonal_hermiticity_residual": _max_abs(
            fock_diagonal - fock_diagonal.getH()
        ),
        "hamiltonian_hermiticity_residual": _max_abs(matrix - matrix.getH()),
        "hamiltonian_jz_commutator_residual": _max_abs(jz @ matrix - matrix @ jz),
        "c53_opposite_helicity_residual": max(
            _max_abs(emission[
                [index - 2 for index, state in enumerate(basis_states[2:], start=2)
                 if state.helicity_twice == 1],
                0,
            ]),
            _max_abs(emission[
                [index - 2 for index, state in enumerate(basis_states[2:], start=2)
                 if state.helicity_twice == -1],
                1,
            ]),
        ),
        "represented_c117_included": True,
        "complete_h2_terms_included": False,
        "complete_h2_missing_terms": UNAVAILABLE_COMPLETE_H2_TERMS,
        "claim_tier": M2M_CLAIM_TIER,
        "physical": False,
    }
    residual_keys = tuple(
        key for key in validation if key.endswith("residual")
    )
    worst_key = max(residual_keys, key=lambda key: float(validation[key]))
    if (
        dimension != 266
        or validation["jz_block_dimensions"] != {"JZ-1/2": 133, "JZ+1/2": 133}
        or validation["fixed_mode_carrier_rank_counts"] != {"rank_1": 56, "rank_2": 104}
        or float(validation[worst_key]) > _TOLERANCE
    ):
        raise RuntimeError(
            "source-supported total-sign H2 assembly did not close: "
            f"{worst_key}={float(validation[worst_key]):.3e}"
        )
    return M2TotalSignH2Hamiltonian(
        point=point,
        target=target,
        mass_embedding=embedding,
        basis_states=tuple(basis_states),
        retained_mode_keys=retained_keys,
        fock_diagonal_matrix=fock_diagonal,
        c53_emission=c53_emission,
        matrix=matrix,
        validation=validation,
    )


@dataclass(frozen=True)
class TotalSignH2HelicityState:
    helicity_twice: int
    energy_GeV2: float
    state: np.ndarray
    residual_norm_GeV2: float
    q_weight: float
    qg_weight: float
    sector_gap_GeV2: float


@dataclass(frozen=True)
class TotalSignH2HelicityPair:
    assembly: M2TotalSignH2Hamiltonian
    states: tuple[TotalSignH2HelicityState, TotalSignH2HelicityState]
    helicity_splitting_GeV2: float

    def state(self, helicity_twice: int) -> TotalSignH2HelicityState:
        for value in self.states:
            if value.helicity_twice == helicity_twice:
                return value
        raise KeyError(helicity_twice)


def _phase_fix(vector: np.ndarray, pivot_index: int) -> np.ndarray:
    result = np.asarray(vector, dtype=np.complex128).copy()
    pivot = pivot_index if abs(result[pivot_index]) > 1.0e-14 else int(np.argmax(np.abs(result)))
    result *= np.conjugate(result[pivot]) / abs(result[pivot])
    return result


def solve_total_sign_h2_helicity_pair(
    assembly: M2TotalSignH2Hamiltonian,
) -> TotalSignH2HelicityPair:
    """Solve the lowest state in each exact Jz block."""

    if not isinstance(assembly, M2TotalSignH2Hamiltonian):
        raise TypeError("assembly must be an M2TotalSignH2Hamiltonian")
    states = []
    for helicity in HELICITIES_TWICE:
        indices = np.asarray(
            [
                state.index
                for state in assembly.basis_states
                if state.helicity_twice == helicity
            ],
            dtype=int,
        )
        sector = assembly.matrix[indices, :][:, indices].toarray()
        values, vectors = np.linalg.eigh(sector)
        full = np.zeros(assembly.dimension, dtype=np.complex128)
        full[indices] = vectors[:, 0]
        q_index = 0 if helicity == -1 else 1
        full = _phase_fix(full, q_index)
        q_weight = float(np.vdot(full[:2], full[:2]).real)
        states.append(
            TotalSignH2HelicityState(
                helicity_twice=helicity,
                energy_GeV2=float(values[0]),
                state=full,
                residual_norm_GeV2=float(
                    np.linalg.norm(assembly.matrix @ full - values[0] * full)
                ),
                q_weight=q_weight,
                qg_weight=1.0 - q_weight,
                sector_gap_GeV2=float(values[1] - values[0]),
            )
        )
    return TotalSignH2HelicityPair(
        assembly=assembly,
        states=(states[0], states[1]),
        helicity_splitting_GeV2=float(states[1].energy_GeV2 - states[0].energy_GeV2),
    )


def total_sign_h2_hamiltonian_record(
    point: ExploratoryC53K9Point = C53_MIXED_K9_EXPLORATORY_BASELINE,
) -> Mapping[str, Any]:
    """Serialize both isospin partners and the exact claim boundary."""

    pairs = {
        target: solve_total_sign_h2_helicity_pair(
            build_total_sign_h2_hamiltonian(target, point)
        )
        for target in TARGETS
    }
    reference = pairs["PROTON"].assembly
    proton_energies = tuple(state.energy_GeV2 for state in pairs["PROTON"].states)
    neutron_energies = tuple(state.energy_GeV2 for state in pairs["NEUTRON"].states)
    payload = {
        "status": M2M_HAMILTONIAN_STATUS,
        "claim_tier": M2M_CLAIM_TIER,
        "physical": False,
        "resolution_id": M2M_RESOLUTION_ID,
        "point_id": point.point_id,
        "basis_dimension_per_target": reference.dimension,
        "jz_block_dimensions": reference.validation["jz_block_dimensions"],
        "retained_c53_external_modes": len(reference.retained_mode_keys),
        "global_ground_mass_embedding": reference.mass_embedding.__dict__,
        "target_results": {
            target: {
                "hamiltonian_root": pair.assembly.hamiltonian_root,
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
            }
            for target, pair in pairs.items()
        },
        "proton_neutron_energy_residual_GeV2": max(
            abs(left - right) for left, right in zip(proton_energies, neutron_energies)
        ),
        "assembly_validation": dict(reference.validation),
        "zero_c53_limit_available": True,
        "one_source_ray_per_mode_rejected_as_incomplete_carrier": True,
        "complete_fixed_weight_total_sign_carrier_used": True,
        "represented_c117_included": True,
        "complete_h2_hamiltonian_constructed": False,
        "complete_h2_missing_terms": UNAVAILABLE_COMPLETE_H2_TERMS,
        "hamiltonian_gauged_current_constructed": False,
        "pennylane_implementation_ready": False,
        "next_required_object": (
            "gauge this identical represented Hamiltonian and its finite Jacobi "
            "regulator to construct the number-changing current before encoding"
        ),
        "production_observable_claimed": False,
    }
    root_payload = {
        "status": payload["status"],
        "claim_tier": payload["claim_tier"],
        "resolution_id": payload["resolution_id"],
        "point_root": point.record()["base_point"]["root"],
        "basis_dimension_per_target": payload["basis_dimension_per_target"],
        "jz_block_dimensions": payload["jz_block_dimensions"],
        "retained_c53_external_modes": payload["retained_c53_external_modes"],
        "global_ground_mass_embedding": payload["global_ground_mass_embedding"],
        "target_hamiltonian_roots": {
            target: pair.assembly.hamiltonian_root for target, pair in pairs.items()
        },
        "complete_h2_missing_terms": payload["complete_h2_missing_terms"],
    }
    return {**_plain(payload), "root": _json_root(root_payload)}


def zero_c53_point(
    point: ExploratoryC53K9Point = C53_MIXED_K9_EXPLORATORY_BASELINE,
) -> ExploratoryC53K9Point:
    return replace(
        point,
        point_id=f"{point.point_id}_ZERO_C53",
        c53_coupling=0.0,
        purpose=f"Zero-C53 limit of {point.point_id}",
    )


__all__ = [
    "M2M_HAMILTONIAN_STATUS",
    "M2M_CLAIM_TIER",
    "M2M_RESOLUTION_ID",
    "UNAVAILABLE_COMPLETE_H2_TERMS",
    "GlobalGroundMassEmbedding",
    "SpinScalarExternalReduction",
    "reduce_spin_scalar_external_operator",
    "TotalSignH2BasisState",
    "M2TotalSignH2Hamiltonian",
    "TotalSignH2HelicityState",
    "TotalSignH2HelicityPair",
    "build_total_sign_h2_hamiltonian",
    "solve_total_sign_h2_helicity_pair",
    "total_sign_h2_hamiltonian_record",
    "zero_c53_point",
]

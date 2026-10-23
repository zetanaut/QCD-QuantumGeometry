"""Abelianized finite-basis continuity current for the M2m Hamiltonian.

The first gauge benchmark uses normalized baryon number, assigning charge
``1/3`` to each quark and zero to the gluon.  This choice avoids importing an
unresolved flavor-dependent electromagnetic matching while retaining a
nontrivial conserved U(1) probe for proton and neutron alike.

The charge density uses exactly the M2m Jacobi regulator.  A hit on the
active quark carries the K9 qg oscillator displacement and the outer-lambda
ground-state overlap.  Each spectator hit carries the outer-lambda and
spectator-rho overlaps.  The three contributions are weighted ``1/3`` and
the forward density is the identity on the complete 266-state carrier.

For nonzero transverse transfer, the longitudinal minimum-norm current is
defined by the finite Hamiltonian continuity equation

``Delta_x J_x + Delta_y J_y = H rho(Delta) - rho(Delta) H``.

At the named nonzero-coupling benchmark, gauging the C53 off-diagonal block
produces a nonzero qqq--qqqg number-changing current. It vanishes continuously
with the explicitly supplied C53 coupling. The construction is exact for this finite
represented Hamiltonian and density, including transfer reversal.  It does
not determine divergence-free transverse pieces or supply the missing
instantaneous, constrained, induced, zero-mode, counterterm, or physical
electromagnetic currents.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import sqrt
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from .m2_active_quark_current import (
    build_m2_active_quark_impulse_operator,
    displaced_ho_overlap,
)
from .m2_spin_flavor_s3_completion import (
    HELICITIES_TWICE,
    build_total_sign_spin_flavor_carrier,
)
from .m2_total_sign_h2_hamiltonian import (
    M2TotalSignH2Hamiltonian,
    reduce_spin_scalar_external_operator,
)


M2N_DENSITY_STATUS = "M2N_MATCHED_GLOBAL_GROUND_BARYON_DENSITY_READY"
M2N_CURRENT_STATUS = "M2N_ABELIANIZED_HAMILTONIAN_CONTINUITY_CURRENT_READY"
M2N_CLAIM_TIER = "EXACT_FINITE_BASIS_REPRESENTED_HAMILTONIAN_U1_BENCHMARK"
M2N_PROBE_ID = "NORMALIZED_BARYON_NUMBER_U1"

UNAVAILABLE_COMPLETE_CURRENT_TERMS = (
    "PHYSICAL_FLAVOR_DEPENDENT_ELECTROMAGNETIC_CURRENT",
    "DIVERGENCE_FREE_TRANSVERSE_CURRENT",
    "INSTANTANEOUS_AND_PAIR_CURRENT",
    "CONSTRAINED_FIELD_CURRENT",
    "INDUCED_EFFECTIVE_CURRENT",
    "ZERO_MODE_CURRENT",
    "CURRENT_COUNTERTERM_AND_RENORMALIZATION_MATCHING",
    "FULL_NONABELIAN_SLAVNOV_TAYLOR_COMPLETION",
)

_TOLERANCE = 8.0e-11


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


def _ground_overlap(
    b_GeV: float,
    shift_x_GeV: float,
    shift_y_GeV: float,
    quadrature_nodes: int,
) -> complex:
    return complex(
        displaced_ho_overlap(
            ((0, 0),),
            b_GeV,
            shift_x_GeV,
            shift_y_GeV,
            quadrature_nodes,
        )[0, 0]
    )


def _carrier_basis(
    assembly: M2TotalSignH2Hamiltonian,
    helicity_twice: int,
    key: tuple[int, int, int, int],
) -> np.ndarray:
    quark_spin_twice = helicity_twice - 2 * (key[2] + key[3])
    return build_total_sign_spin_flavor_carrier(
        assembly.target, quark_spin_twice
    ).basis


def _basis_rows(
    assembly: M2TotalSignH2Hamiltonian,
) -> Mapping[tuple[int, tuple[int, int, int, int]], tuple[int, ...]]:
    rows: dict[tuple[int, tuple[int, int, int, int]], list[int]] = {}
    for state in assembly.basis_states:
        if state.sector == "qqqg":
            if state.mode_key is None:
                raise RuntimeError("qqqg basis state lacks its mode key")
            rows.setdefault((state.helicity_twice, state.mode_key), []).append(
                state.index
            )
    return {key: tuple(value) for key, value in rows.items()}


@dataclass(frozen=True)
class M2BaryonDensityOperator:
    assembly: M2TotalSignH2Hamiltonian
    delta_x_GeV: float
    delta_y_GeV: float
    quadrature_nodes: int
    active_lambda_overlap: complex
    spectator_lambda_overlap: complex
    spectator_rho_overlap: complex
    spectator_total_overlap: complex
    matrix: sparse.csr_matrix
    validation: Mapping[str, Any]
    probe_id: str = M2N_PROBE_ID
    claim_tier: str = M2N_CLAIM_TIER
    physical: bool = False

    @property
    def q2_GeV2(self) -> float:
        return self.delta_x_GeV**2 + self.delta_y_GeV**2

    @property
    def operator_root(self) -> str:
        return _sparse_root(self.matrix)


def build_m2_baryon_density_operator(
    assembly: M2TotalSignH2Hamiltonian,
    *,
    delta_x_GeV: float,
    delta_y_GeV: float,
    quadrature_nodes: int = 28,
) -> M2BaryonDensityOperator:
    """Project the matched three-quark U(1) density into the M2m basis."""

    if not isinstance(assembly, M2TotalSignH2Hamiltonian):
        raise TypeError("assembly must be an M2TotalSignH2Hamiltonian")
    dx = float(delta_x_GeV)
    dy = float(delta_y_GeV)
    if not np.all(np.isfinite((dx, dy))) or quadrature_nodes < 8:
        raise ValueError("transfer must be finite and quadrature_nodes must be at least 8")
    embedding = assembly.mass_embedding
    active_factor = sqrt(embedding.spectator_fraction / embedding.active_fraction)
    spectator_factor = sqrt(embedding.active_fraction / embedding.spectator_fraction)
    z1, z2 = embedding.spectator_internal_fractions
    active_lambda = _ground_overlap(
        embedding.b_lambda_GeV,
        active_factor * dx,
        active_factor * dy,
        quadrature_nodes,
    )
    spectator_lambda = _ground_overlap(
        embedding.b_lambda_GeV,
        -spectator_factor * dx,
        -spectator_factor * dy,
        quadrature_nodes,
    )
    spectator_rho = _ground_overlap(
        embedding.b_rho_GeV,
        sqrt(z2 / z1) * dx,
        sqrt(z2 / z1) * dy,
        quadrature_nodes,
    )
    spectator_total = spectator_lambda * spectator_rho

    active_source = build_m2_active_quark_impulse_operator(
        "K9",
        delta_x_GeV=dx,
        delta_y_GeV=dy,
        quadrature_nodes=quadrature_nodes,
    )
    reduction = reduce_spin_scalar_external_operator(
        active_source.matrix, assembly.retained_mode_keys
    )
    rows = _basis_rows(assembly)
    density = np.zeros((assembly.dimension, assembly.dimension), dtype=np.complex128)
    q_ground_form_factor = (active_lambda + 2.0 * spectator_total) / 3.0
    density[:2, :2] = q_ground_form_factor * np.eye(2, dtype=np.complex128)

    for out_helicity in HELICITIES_TWICE:
        for in_helicity in HELICITIES_TWICE:
            for alpha_index, alpha in enumerate(assembly.retained_mode_keys):
                out_basis = _carrier_basis(assembly, out_helicity, alpha)
                out_rows = rows[(out_helicity, alpha)]
                for beta_index, beta in enumerate(assembly.retained_mode_keys):
                    active_value = reduction.retained_operator[alpha_index, beta_index]
                    spectator_value = (
                        1.0 if alpha_index == beta_index else 0.0
                    )
                    coefficient = (
                        active_lambda * active_value / 3.0
                        + 2.0 * spectator_total * spectator_value / 3.0
                    )
                    if abs(coefficient) <= 1.0e-15:
                        continue
                    in_basis = _carrier_basis(assembly, in_helicity, beta)
                    in_columns = rows[(in_helicity, beta)]
                    overlap = out_basis.conj().T @ in_basis
                    density[np.ix_(out_rows, in_columns)] = coefficient * overlap

    matrix = sparse.csr_matrix(density)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    q = assembly.q_dimension
    forward = dx == 0.0 and dy == 0.0
    identity_residual = _max_abs(
        matrix - sparse.identity(assembly.dimension, format="csr", dtype=np.complex128)
    )
    validation = {
        "dimension": assembly.dimension,
        "probe_id": M2N_PROBE_ID,
        "quark_probe_charge": "1/3",
        "gluon_probe_charge": "0",
        "forward": forward,
        "q_to_qg_density_residual": max(
            _max_abs(matrix[:q, q:]), _max_abs(matrix[q:, :q])
        ),
        "source_active_q_spin_scalar_residual": reduction.q_spin_scalar_residual,
        "source_active_q_qg_cross_residual": reduction.q_qg_cross_residual,
        "source_active_qg_spin_scalar_residual": reduction.qg_spin_scalar_residual,
        "source_active_excluded_mode_leakage": (
            reduction.retained_support_leakage_residual
        ),
        "forward_identity_residual": identity_residual if forward else None,
        "finite_shell_projection_explicit": True,
        "physical_electromagnetic_current": False,
    }
    checked = tuple(
        float(value)
        for key, value in validation.items()
        if key.endswith("residual") and value is not None
    )
    if checked and max(checked) > _TOLERANCE:
        raise RuntimeError("matched M2 baryon density failed its exact checks")
    return M2BaryonDensityOperator(
        assembly=assembly,
        delta_x_GeV=dx,
        delta_y_GeV=dy,
        quadrature_nodes=int(quadrature_nodes),
        active_lambda_overlap=active_lambda,
        spectator_lambda_overlap=spectator_lambda,
        spectator_rho_overlap=spectator_rho,
        spectator_total_overlap=spectator_total,
        matrix=matrix,
        validation=validation,
    )


@dataclass(frozen=True)
class M2HamiltonianGaugedBaryonCurrent:
    density: M2BaryonDensityOperator
    current_x: sparse.csr_matrix
    current_y: sparse.csr_matrix
    fock_diagonal_current_x: sparse.csr_matrix
    fock_diagonal_current_y: sparse.csr_matrix
    c53_number_changing_current_x: sparse.csr_matrix
    c53_number_changing_current_y: sparse.csr_matrix
    continuity_numerator: sparse.csr_matrix
    validation: Mapping[str, Any]
    claim_tier: str = M2N_CLAIM_TIER
    physical: bool = False

    @property
    def current_root(self) -> str:
        digest = sha256()
        digest.update(_sparse_root(self.current_x).encode())
        digest.update(_sparse_root(self.current_y).encode())
        return digest.hexdigest()


def build_hamiltonian_gauged_baryon_current(
    density: M2BaryonDensityOperator,
) -> M2HamiltonianGaugedBaryonCurrent:
    """Construct the longitudinal current, including the explicit zero-C53 limit.

    A nonzero number-changing block is a property of a benchmark, not a
    general validity condition. Continuity and block ancestry are enforced
    for zero and nonzero caller-supplied couplings alike.
    """

    if not isinstance(density, M2BaryonDensityOperator):
        raise TypeError("density must be an M2BaryonDensityOperator")
    if density.q2_GeV2 <= 0.0:
        raise ValueError("a nonzero transverse transfer is required")
    assembly = density.assembly
    rho = density.matrix
    hamiltonian = assembly.matrix
    fock = assembly.fock_diagonal_matrix
    c53 = (hamiltonian - fock).tocsr()
    fock_numerator = (fock @ rho - rho @ fock).tocsr()
    c53_numerator = (c53 @ rho - rho @ c53).tocsr()
    numerator = (fock_numerator + c53_numerator).tocsr()
    x_factor = density.delta_x_GeV / density.q2_GeV2
    y_factor = density.delta_y_GeV / density.q2_GeV2
    fock_x = (x_factor * fock_numerator).tocsr()
    fock_y = (y_factor * fock_numerator).tocsr()
    c53_x = (x_factor * c53_numerator).tocsr()
    c53_y = (y_factor * c53_numerator).tocsr()
    current_x = (fock_x + c53_x).tocsr()
    current_y = (fock_y + c53_y).tocsr()
    divergence = (
        density.delta_x_GeV * current_x + density.delta_y_GeV * current_y
    ).tocsr()
    q = assembly.q_dimension
    validation = {
        "dimension": assembly.dimension,
        "q2_GeV2": density.q2_GeV2,
        "continuity_residual_GeV2": _max_abs(divergence - numerator),
        "hamiltonian_commutator_reconstruction_residual_GeV2": _max_abs(
            numerator - (hamiltonian @ rho - rho @ hamiltonian)
        ),
        "current_decomposition_residual_GeV": max(
            _max_abs(current_x - fock_x - c53_x),
            _max_abs(current_y - fock_y - c53_y),
        ),
        "fock_current_number_changing_residual_GeV": max(
            _max_abs(fock_x[:q, q:]),
            _max_abs(fock_x[q:, :q]),
            _max_abs(fock_y[:q, q:]),
            _max_abs(fock_y[q:, :q]),
        ),
        "c53_current_fock_diagonal_residual_GeV": max(
            _max_abs(c53_x[:q, :q]),
            _max_abs(c53_x[q:, q:]),
            _max_abs(c53_y[:q, :q]),
            _max_abs(c53_y[q:, q:]),
        ),
        "c53_number_changing_current_max_abs_GeV": max(
            _max_abs(c53_x[:q, q:]),
            _max_abs(c53_x[q:, :q]),
            _max_abs(c53_y[:q, q:]),
            _max_abs(c53_y[q:, :q]),
        ),
        "longitudinal_minimum_norm_choice": True,
        "divergence_free_transverse_piece_fixed": False,
        "complete_current": False,
        "physical_electromagnetic_current": False,
    }
    residuals = tuple(
        float(value)
        for key, value in validation.items()
        if "residual" in key
    )
    if max(residuals) > _TOLERANCE:
        raise RuntimeError("M2 Hamiltonian continuity current did not close")
    return M2HamiltonianGaugedBaryonCurrent(
        density=density,
        current_x=current_x,
        current_y=current_y,
        fock_diagonal_current_x=fock_x,
        fock_diagonal_current_y=fock_y,
        c53_number_changing_current_x=c53_x,
        c53_number_changing_current_y=c53_y,
        continuity_numerator=numerator,
        validation=validation,
    )


def hamiltonian_gauged_current_record(
    positive: M2HamiltonianGaugedBaryonCurrent,
    reversed_transfer: M2HamiltonianGaugedBaryonCurrent,
    forward_density: M2BaryonDensityOperator,
) -> Mapping[str, Any]:
    """Record continuity, reversal, charge, and completeness boundaries."""

    if not all(
        isinstance(item, M2HamiltonianGaugedBaryonCurrent)
        for item in (positive, reversed_transfer)
    ) or not isinstance(forward_density, M2BaryonDensityOperator):
        raise TypeError("current pair and forward density have the wrong types")
    if positive.density.assembly is not reversed_transfer.density.assembly:
        raise ValueError("transfer reversal requires one Hamiltonian assembly")
    if forward_density.assembly is not positive.density.assembly:
        raise ValueError("forward density requires the same Hamiltonian assembly")
    if (
        reversed_transfer.density.delta_x_GeV != -positive.density.delta_x_GeV
        or reversed_transfer.density.delta_y_GeV != -positive.density.delta_y_GeV
    ):
        raise ValueError("reversed current must carry the opposite transfer")
    if forward_density.q2_GeV2 != 0.0:
        raise ValueError("forward density must carry zero transfer")
    identity = sparse.identity(
        positive.density.assembly.dimension,
        format="csr",
        dtype=np.complex128,
    )
    density_reversal = _max_abs(
        reversed_transfer.density.matrix - positive.density.matrix.getH()
    )
    current_reversal = max(
        _max_abs(reversed_transfer.current_x - positive.current_x.getH()),
        _max_abs(reversed_transfer.current_y - positive.current_y.getH()),
    )
    payload = {
        "status": M2N_CURRENT_STATUS,
        "density_status": M2N_DENSITY_STATUS,
        "claim_tier": M2N_CLAIM_TIER,
        "physical": False,
        "probe_id": M2N_PROBE_ID,
        "target": positive.density.assembly.target,
        "dimension": positive.density.assembly.dimension,
        "delta_t_GeV": (
            positive.density.delta_x_GeV,
            positive.density.delta_y_GeV,
        ),
        "density_root": positive.density.operator_root,
        "current_root": positive.current_root,
        "active_lambda_overlap": positive.density.active_lambda_overlap,
        "spectator_lambda_overlap": positive.density.spectator_lambda_overlap,
        "spectator_rho_overlap": positive.density.spectator_rho_overlap,
        "spectator_total_overlap": positive.density.spectator_total_overlap,
        "forward_density_identity_residual": _max_abs(
            forward_density.matrix - identity
        ),
        "density_transfer_reversal_residual": density_reversal,
        "current_transfer_reversal_residual": current_reversal,
        "positive_current_validation": dict(positive.validation),
        "number_changing_current_constructed": True,
        "number_changing_current_ancestry": "M2M_C53_BLOCK",
        "finite_hamiltonian_continuity_closed": True,
        "longitudinal_current_fixed": True,
        "divergence_free_transverse_current_fixed": False,
        "physical_electromagnetic_current_constructed": False,
        "complete_hamiltonian_gauged_current_constructed": False,
        "unavailable_current_terms": UNAVAILABLE_COMPLETE_CURRENT_TERMS,
        "pennylane_observable_contract_ready": True,
        "production_observable_claimed": False,
        "next_required_object": (
            "publish a sparse-to-qubit encoding and Hermitian measurement "
            "contract for the M2m Hamiltonian and M2n reversal-paired operators"
        ),
    }
    worst = max(
        density_reversal,
        current_reversal,
        float(payload["forward_density_identity_residual"]),
    )
    if worst > _TOLERANCE:
        raise RuntimeError("M2 current pair failed transfer reversal or charge")
    return {**_plain(payload), "root": _json_root(payload)}


__all__ = [
    "M2N_DENSITY_STATUS",
    "M2N_CURRENT_STATUS",
    "M2N_CLAIM_TIER",
    "M2N_PROBE_ID",
    "UNAVAILABLE_COMPLETE_CURRENT_TERMS",
    "M2BaryonDensityOperator",
    "M2HamiltonianGaugedBaryonCurrent",
    "build_m2_baryon_density_operator",
    "build_hamiltonian_gauged_baryon_current",
    "hamiltonian_gauged_current_record",
]

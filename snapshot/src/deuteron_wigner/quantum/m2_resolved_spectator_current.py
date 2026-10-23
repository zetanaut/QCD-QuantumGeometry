"""Resolved two-quark spectator embedding and matched M2 one-body current.

M2h supplied a source-derived current for the active ``q + qg`` subsystem but
represented the two remaining quarks by a momentum-independent identity.  The
smallest finite-K continuation that preserves the existing K9 subsystem is a
nested Jacobi member:

* the active cluster keeps ``K2=9``;
* each exchange-related spectator quark has ``k2=9``;
* the nucleon therefore has ``K2=27`` and exact valence shares ``1/3``;
* the spectator-pair and active-cluster/spectator-cluster Jacobi coordinates
  are retained in the C45 ``(n,m)=(0,0)`` mode at the same ``b=0.4 GeV``.

The projected free invariant-mass operator and all three diagonal one-body
quark currents then use the same longitudinal fractions, oscillator modes,
and Volume-XI recoil map.  The nested transverse variables are the normalized
C45 pair coordinates: their kinetic contributions occur once at the level at
which the coordinate is introduced.  In particular, the pair coordinate is
not divided by each daughter fraction a second time and the outer coordinate
is not divided by both cluster fractions.  This is an explicit exploratory
regulator member.  It is not the complete antisymmetrized C7/H2 basis,
contains no spectator or cluster interaction, and is not a physical
scalar/axial diquark model.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from math import isfinite, sqrt
from numbers import Real
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from deuteron_wigner.bridge.c401_c396_mass_directions import (
    canonical_partitions,
    resolution_record,
)
from deuteron_wigner.conventions import HELICITIES
from deuteron_wigner.lf_current import SpinOnePlusCurrent
from deuteron_wigner.light_front import (
    LFNormalization,
    SpinRotation,
    off_forward_active_component_densities,
)

from .m2_active_quark_current import (
    M2ActiveQuarkImpulseOperator,
    displaced_ho_overlap,
    pair_active_current_matrix,
)
from .m2_c53_mixed_singlet_state import C53MixedK9HelicityPair, K9
from .m2_nn_transfer_current import NNTransferFibers
from .m2_nucleon_spectator_composition import NucleonSpectatorDoublet


M2_RESOLVED_SPECTATOR_STATUS = (
    "M2_RESOLVED_TWO_QUARK_SPECTATOR_HAMILTONIAN_READY"
)
M2_RESOLVED_CURRENT_STATUS = (
    "M2_RESOLVED_ACTIVE_AND_SPECTATOR_ONE_BODY_CURRENT_READY"
)
M2_RESOLVED_SPECTATOR_CLAIM_TIER = (
    "EXPLORATORY_FINITE_K_NESTED_JACOBI_SPECTATOR_MEMBER"
)
M2_RESOLVED_CURRENT_CLAIM_TIER = (
    "EXPLORATORY_MATCHED_DIAGONAL_ONE_BODY_CURRENT"
)

RESOLVED_SPECTATOR_SOURCES = (
    "VOLUME_VII_FREE_INVARIANT_MASS",
    "VOLUME_VII_EXACT_LONGITUDINAL_MODES",
    "C45_2D_HO_NORMALIZATION_AND_PHASE",
    "C47_K9_ACTIVE_CLUSTER_BASIS",
    "VOLUME_XI_SYMMETRIC_XI0_ACTIVE_SPECTATOR_RECOIL",
)

UNAVAILABLE_MATCHED_CURRENT_TERMS = (
    "Q_TO_QG_AND_QG_TO_Q_NUMBER_CHANGING_CURRENT",
    "SPECTATOR_AND_CLUSTER_INTERACTION_CURRENT",
    "INDUCED_EFFECTIVE_CURRENT",
    "CONSTRAINED_FIELD_CURRENT",
    "PAIR_CURRENT",
    "ZERO_MODE_CURRENT",
    "OPERATOR_RENORMALIZATION_AND_MATCHING",
    "FULL_ACTIVE_SPECTATOR_ANTISYMMETRIZATION",
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


def _json_root(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode()).hexdigest()


def _sparse_root(matrix: sparse.csr_matrix) -> str:
    value = matrix.tocsr()
    digest = sha256()
    for array in (value.indptr, value.indices, value.data):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ResolvedSpectatorResolution:
    """One explicit nested-Jacobi finite-K spectator regulator member."""

    resolution_id: str
    active_resolution: str
    active_cluster_K2: int
    spectator_quark_K2: tuple[int, int]
    nucleon_K2: int
    rho_mode: tuple[int, int]
    lambda_mode: tuple[int, int]
    b_rho_GeV: float
    b_lambda_GeV: float
    purpose: str
    claim_tier: str = M2_RESOLVED_SPECTATOR_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if not self.resolution_id or not self.purpose:
            raise ValueError("spectator resolution identity and purpose are required")
        if self.active_resolution != K9:
            raise ValueError("the first resolved spectator member preserves K9")
        if self.active_cluster_K2 != int(resolution_record(K9)["K2"]):
            raise ValueError("active_cluster_K2 must equal the represented K9 value")
        if len(self.spectator_quark_K2) != 2:
            raise ValueError("exactly two spectator quark modes are required")
        modes = (self.active_cluster_K2, *self.spectator_quark_K2)
        if any(mode <= 0 or mode % 2 != 1 for mode in modes):
            raise ValueError("all retained fermion K2 modes must be positive odd integers")
        if sum(modes) != self.nucleon_K2 or self.nucleon_K2 % 2 != 1:
            raise ValueError("finite-K longitudinal modes do not close the nucleon K2")
        if self.spectator_quark_K2[0] != self.spectator_quark_K2[1]:
            raise ValueError("the S-wave spectator pair requires exchange-equal modes")
        if self.rho_mode != (0, 0) or self.lambda_mode != (0, 0):
            raise ValueError("the first resolved member retains only Jacobi ground modes")
        for name in ("b_rho_GeV", "b_lambda_GeV"):
            if _finite_real(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        active_b = float(resolution_record(K9)["b_HO"])
        if self.b_rho_GeV != active_b or self.b_lambda_GeV != active_b:
            raise ValueError("all M2i Jacobi coordinates must share the K9 C45 scale")
        if self.claim_tier != M2_RESOLVED_SPECTATOR_CLAIM_TIER or self.physical:
            raise ValueError("the resolved spectator member must remain exploratory")

    @property
    def active_cluster_fraction(self) -> Fraction:
        return Fraction(self.active_cluster_K2, self.nucleon_K2)

    @property
    def spectator_quark_fractions(self) -> tuple[Fraction, Fraction]:
        return tuple(Fraction(value, self.nucleon_K2) for value in self.spectator_quark_K2)

    @property
    def spectator_cluster_fraction(self) -> Fraction:
        return sum(self.spectator_quark_fractions, Fraction(0))

    @property
    def spectator_internal_fractions(self) -> tuple[Fraction, Fraction]:
        total = sum(self.spectator_quark_K2)
        return tuple(Fraction(value, total) for value in self.spectator_quark_K2)

    @property
    def nucleon_K(self) -> Fraction:
        return Fraction(self.nucleon_K2, 2)


M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1 = ResolvedSpectatorResolution(
    resolution_id="M2I_K9_CLUSTER_K27_EQUAL_SHARE_RHO0_LAMBDA0_B0P40_V1",
    active_resolution=K9,
    active_cluster_K2=9,
    spectator_quark_K2=(9, 9),
    nucleon_K2=27,
    rho_mode=(0, 0),
    lambda_mode=(0, 0),
    b_rho_GeV=0.4,
    b_lambda_GeV=0.4,
    purpose=(
        "Smallest exchange-symmetric equal-share finite-K embedding that preserves "
        "the complete K9 active cluster; diagnostic regulator member, not a fit"
    ),
)


def resolved_sector_longitudinal_records(
    resolution: ResolvedSpectatorResolution,
) -> tuple[Mapping[str, Any], ...]:
    """Return exact qqq and qqqg longitudinal ledgers at the lifted K."""

    if not isinstance(resolution, ResolvedSpectatorResolution):
        raise TypeError("resolution must be ResolvedSpectatorResolution")
    denominator = resolution.nucleon_K2

    def row(sector: str, partition: str, modes: tuple[int, ...]) -> Mapping[str, Any]:
        fractions = tuple(Fraction(mode, denominator) for mode in modes)
        return {
            "sector": sector,
            "partition": partition,
            "K2_modes": modes,
            "fractions": tuple(str(value) for value in fractions),
            "fraction_sum_exact": str(sum(fractions, Fraction(0))),
            "K2_sum_residual": sum(modes) - denominator,
        }

    records = [
        row(
            "qqq",
            "q",
            (
                resolution.active_cluster_K2,
                *resolution.spectator_quark_K2,
            ),
        )
    ]
    for partition in canonical_partitions(resolution.active_resolution):
        kq2 = int(2 * partition.kq)
        kg2 = int(2 * partition.kg)
        records.append(
            row(
                "qqqg",
                partition.partition_id,
                (kq2, kg2, *resolution.spectator_quark_K2),
            )
        )
    if any(
        item["fraction_sum_exact"] != "1" or item["K2_sum_residual"] != 0
        for item in records
    ):
        raise RuntimeError("resolved sector longitudinal ledger did not close")
    return tuple(records)


@dataclass(frozen=True)
class M2ResolvedSpectatorHamiltonian:
    """Projected free-mass embedding of M2 and two resolved spectator quarks."""

    pair: C53MixedK9HelicityPair
    resolution: ResolvedSpectatorResolution
    spectator_quark_mass_sq_GeV2: float
    rho_k2_expectation_GeV2: float
    lambda_k2_expectation_GeV2: float
    spectator_cluster_mass_sq_GeV2: float
    additive_identity_shift_GeV2: float
    matrix: sparse.csr_matrix
    claim_tier: str = M2_RESOLVED_SPECTATOR_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.pair, C53MixedK9HelicityPair):
            raise TypeError("pair must be a C53MixedK9HelicityPair")
        if not isinstance(self.resolution, ResolvedSpectatorResolution):
            raise TypeError("resolution must be ResolvedSpectatorResolution")
        for name in (
            "spectator_quark_mass_sq_GeV2",
            "rho_k2_expectation_GeV2",
            "lambda_k2_expectation_GeV2",
            "spectator_cluster_mass_sq_GeV2",
            "additive_identity_shift_GeV2",
        ):
            if _finite_real(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        matrix = self.matrix.tocsr()
        if matrix.shape != (self.pair.assembly.dimension,) * 2:
            raise ValueError("resolved Hamiltonian has the wrong active-space shape")
        if _max_abs(matrix - matrix.getH()) > 2.0e-13:
            raise ValueError("resolved Hamiltonian must be Hermitian")
        if self.claim_tier != M2_RESOLVED_SPECTATOR_CLAIM_TIER or self.physical:
            raise ValueError("the resolved Hamiltonian must remain exploratory")

    @property
    def active_mass_scale(self) -> float:
        return 1.0 / float(self.resolution.active_cluster_fraction)

    @property
    def hamiltonian_root(self) -> str:
        return _sparse_root(self.matrix)

    @property
    def embedded_pair_energies_GeV2(self) -> tuple[float, float]:
        return tuple(
            self.active_mass_scale * self.pair.state(helicity).energy_GeV2
            + self.additive_identity_shift_GeV2
            for helicity in (-1, 1)
        )

    @property
    def embedded_pair_residual_GeV2(self) -> float:
        basis = self.pair.basis
        energies = np.asarray(self.embedded_pair_energies_GeV2)
        return float(np.max(np.abs(self.matrix @ basis - basis * energies[None, :])))


def build_resolved_spectator_hamiltonian(
    pair: C53MixedK9HelicityPair,
    *,
    resolution: ResolvedSpectatorResolution = (
        M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1
    ),
) -> M2ResolvedSpectatorHamiltonian:
    """Embed the K9 mass operator using the nested free-invariant-mass rule."""

    if not isinstance(pair, C53MixedK9HelicityPair):
        raise TypeError("pair must be a C53MixedK9HelicityPair")
    if not isinstance(resolution, ResolvedSpectatorResolution):
        raise TypeError("resolution must be ResolvedSpectatorResolution")
    y_active = float(resolution.active_cluster_fraction)
    y_spectator = float(resolution.spectator_cluster_fraction)
    z1, z2 = (float(value) for value in resolution.spectator_internal_fractions)
    mass_sq = float(pair.assembly.point.base_point.mu_q_sq_GeV2)
    rho_k2 = resolution.b_rho_GeV**2
    lambda_k2 = resolution.b_lambda_GeV**2
    # For the normalized C45 coordinate
    # q_rho=ell_1/sqrt(z_1 z_2), the exact pair identity is
    # M_S^2=m_1^2/z_1+m_2^2/z_2+q_rho^2.  Likewise the outer normalized
    # coordinate q_lambda=kappa_A/sqrt(y_A y_S) contributes q_lambda^2
    # once to M_N^2=M_A^2/y_A+M_S^2/y_S+q_lambda^2.  Dividing either
    # normalized transverse term by the daughter fractions again would
    # double-count the x scaling already built into the Jacobi variable.
    spectator_mass_sq = mass_sq / z1 + mass_sq / z2 + rho_k2
    shift = spectator_mass_sq / y_spectator + lambda_k2
    active = pair.assembly.matrix().tocsr()
    identity = sparse.identity(active.shape[0], format="csr", dtype=np.complex128)
    matrix = (active / y_active + shift * identity).tocsr()
    matrix.sort_indices()
    return M2ResolvedSpectatorHamiltonian(
        pair=pair,
        resolution=resolution,
        spectator_quark_mass_sq_GeV2=mass_sq,
        rho_k2_expectation_GeV2=rho_k2,
        lambda_k2_expectation_GeV2=lambda_k2,
        spectator_cluster_mass_sq_GeV2=spectator_mass_sq,
        additive_identity_shift_GeV2=shift,
        matrix=matrix,
    )


def resolved_spectator_hamiltonian_record(
    embedding: M2ResolvedSpectatorHamiltonian,
) -> Mapping[str, Any]:
    """Serialize the exact momentum ledger and projected Hamiltonian boundary."""

    if not isinstance(embedding, M2ResolvedSpectatorHamiltonian):
        raise TypeError("embedding must be M2ResolvedSpectatorHamiltonian")
    resolution = embedding.resolution
    longitudinal_sum = (
        resolution.active_cluster_fraction + resolution.spectator_cluster_fraction
    )
    payload = {
        "status": M2_RESOLVED_SPECTATOR_STATUS,
        "claim_tier": embedding.claim_tier,
        "physical": False,
        "resolution_id": resolution.resolution_id,
        "active_resolution": resolution.active_resolution,
        "active_cluster_K2": resolution.active_cluster_K2,
        "spectator_quark_K2": resolution.spectator_quark_K2,
        "nucleon_K2": resolution.nucleon_K2,
        "nucleon_K": str(resolution.nucleon_K),
        "active_cluster_fraction": str(resolution.active_cluster_fraction),
        "spectator_quark_fractions": tuple(
            str(value) for value in resolution.spectator_quark_fractions
        ),
        "spectator_cluster_fraction": str(resolution.spectator_cluster_fraction),
        "spectator_internal_fractions": tuple(
            str(value) for value in resolution.spectator_internal_fractions
        ),
        "sector_longitudinal_records": resolved_sector_longitudinal_records(
            resolution
        ),
        "longitudinal_fraction_sum_exact": str(longitudinal_sum),
        "longitudinal_fraction_sum_residual": float(longitudinal_sum - 1),
        "fermion_boundary_condition": "ANTIPERIODIC_HALF_INTEGER",
        "rho_mode": resolution.rho_mode,
        "lambda_mode": resolution.lambda_mode,
        "b_rho_GeV": resolution.b_rho_GeV,
        "b_lambda_GeV": resolution.b_lambda_GeV,
        "same_c45_scale_as_active_K9": True,
        "spectator_quark_mass_sq_GeV2": embedding.spectator_quark_mass_sq_GeV2,
        "rho_k2_expectation_GeV2": embedding.rho_k2_expectation_GeV2,
        "lambda_k2_expectation_GeV2": embedding.lambda_k2_expectation_GeV2,
        "spectator_cluster_mass_sq_GeV2": embedding.spectator_cluster_mass_sq_GeV2,
        "active_mass_scale": embedding.active_mass_scale,
        "additive_identity_shift_GeV2": embedding.additive_identity_shift_GeV2,
        "embedded_pair_energies_GeV2": embedding.embedded_pair_energies_GeV2,
        "embedded_pair_residual_GeV2": embedding.embedded_pair_residual_GeV2,
        "hermiticity_residual": _max_abs(embedding.matrix - embedding.matrix.getH()),
        "hamiltonian_root": embedding.hamiltonian_root,
        "hamiltonian_sources": RESOLVED_SPECTATOR_SOURCES,
        "jacobi_scaling_convention": (
            "q_rho=ell_1/sqrt(z_1*z_2); "
            "q_lambda=kappa_A/sqrt(y_A*y_S)"
        ),
        "projected_free_mass_formula": (
            "M_N^2=M_A^2/y_A+M_S^2/y_S+q_lambda^2; "
            "M_S^2=sum_i(m_q^2/z_i)+q_rho^2"
        ),
        "normalized_transverse_fraction_weights_consumed_once": True,
        "spectator_interaction_constructed": False,
        "full_active_spectator_antisymmetrization_constructed": False,
        "complete_c7_h2_basis_constructed": False,
        "physical_nucleon_mass_claimed": False,
    }
    return {**payload, "root": _json_root(payload)}


@dataclass(frozen=True)
class ResolvedSpectatorCurrentFactors:
    """Jacobi overlap factors for active-cluster and spectator-quark hits."""

    resolution: ResolvedSpectatorResolution
    delta_x_GeV: float
    delta_y_GeV: float
    quadrature_nodes: int
    active_lambda_shift_GeV: tuple[float, float]
    spectator_lambda_shift_GeV: tuple[float, float]
    spectator_rho_shift_GeV: tuple[float, float]
    active_cluster_overlap: complex
    spectator_quark_overlaps: tuple[complex, complex]

    def __post_init__(self) -> None:
        if not isinstance(self.resolution, ResolvedSpectatorResolution):
            raise TypeError("resolution must be ResolvedSpectatorResolution")
        values = np.asarray(
            (
                self.delta_x_GeV,
                self.delta_y_GeV,
                *self.active_lambda_shift_GeV,
                *self.spectator_lambda_shift_GeV,
                *self.spectator_rho_shift_GeV,
                self.active_cluster_overlap.real,
                self.active_cluster_overlap.imag,
                *(part for value in self.spectator_quark_overlaps for part in (value.real, value.imag)),
            )
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("resolved current factors contain nonfinite values")
        if len(self.spectator_quark_overlaps) != 2:
            raise ValueError("two spectator-quark overlaps are required")
        if abs(self.spectator_quark_overlaps[0] - self.spectator_quark_overlaps[1]) > 2.0e-15:
            raise ValueError("exchange-related spectator quarks must share one overlap")

    @property
    def spectator_overlap(self) -> complex:
        return self.spectator_quark_overlaps[0]

    @property
    def q2_GeV2(self) -> float:
        return self.delta_x_GeV**2 + self.delta_y_GeV**2


def build_resolved_spectator_current_factors(
    resolution: ResolvedSpectatorResolution,
    *,
    delta_x_GeV: float,
    delta_y_GeV: float,
    quadrature_nodes: int = 28,
) -> ResolvedSpectatorCurrentFactors:
    """Evaluate the common-basis recoil overlaps for all diagonal quark hits."""

    if not isinstance(resolution, ResolvedSpectatorResolution):
        raise TypeError("resolution must be ResolvedSpectatorResolution")
    dx = _finite_real(delta_x_GeV, "delta_x_GeV")
    dy = _finite_real(delta_y_GeV, "delta_y_GeV")
    y_active = float(resolution.active_cluster_fraction)
    y_spectator = float(resolution.spectator_cluster_fraction)
    z1, z2 = (float(value) for value in resolution.spectator_internal_fractions)

    active_factor = sqrt(y_spectator / y_active)
    spectator_factor = sqrt(y_active / y_spectator)
    rho_factor_1 = sqrt(z2 / z1)
    rho_factor_2 = sqrt(z1 / z2)
    active_shift = (active_factor * dx, active_factor * dy)
    # lambda is oriented from the spectator cluster to the active cluster.
    # A struck spectator quark leaves the active cluster physically unchanged,
    # hence delta(kappa_A)=-y_A Delta and the signed lambda shift is negative.
    spectator_shift = (-spectator_factor * dx, -spectator_factor * dy)
    rho_shift_1 = (rho_factor_1 * dx, rho_factor_1 * dy)
    rho_shift_2 = (-rho_factor_2 * dx, -rho_factor_2 * dy)
    mode = (resolution.lambda_mode,)
    active_overlap = complex(
        displaced_ho_overlap(
            mode,
            resolution.b_lambda_GeV,
            active_shift[0],
            active_shift[1],
            quadrature_nodes,
        )[0, 0]
    )
    spectator_lambda = complex(
        displaced_ho_overlap(
            mode,
            resolution.b_lambda_GeV,
            spectator_shift[0],
            spectator_shift[1],
            quadrature_nodes,
        )[0, 0]
    )
    rho_mode = (resolution.rho_mode,)
    spectator_rho_1 = complex(
        displaced_ho_overlap(
            rho_mode,
            resolution.b_rho_GeV,
            rho_shift_1[0],
            rho_shift_1[1],
            quadrature_nodes,
        )[0, 0]
    )
    spectator_rho_2 = complex(
        displaced_ho_overlap(
            rho_mode,
            resolution.b_rho_GeV,
            rho_shift_2[0],
            rho_shift_2[1],
            quadrature_nodes,
        )[0, 0]
    )
    return ResolvedSpectatorCurrentFactors(
        resolution=resolution,
        delta_x_GeV=dx,
        delta_y_GeV=dy,
        quadrature_nodes=int(quadrature_nodes),
        active_lambda_shift_GeV=active_shift,
        spectator_lambda_shift_GeV=spectator_shift,
        spectator_rho_shift_GeV=rho_shift_1,
        active_cluster_overlap=active_overlap,
        spectator_quark_overlaps=(
            spectator_lambda * spectator_rho_1,
            spectator_lambda * spectator_rho_2,
        ),
    )


def nucleon_resolved_impulse_components(
    doublet: NucleonSpectatorDoublet,
    pair_current: np.ndarray,
    factors: ResolvedSpectatorCurrentFactors,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return active, spectator, and total currents for one nucleon doublet."""

    if not isinstance(doublet, NucleonSpectatorDoublet):
        raise TypeError("doublet must be a NucleonSpectatorDoublet")
    if not isinstance(factors, ResolvedSpectatorCurrentFactors):
        raise TypeError("factors must be ResolvedSpectatorCurrentFactors")
    current = np.asarray(pair_current, dtype=np.complex128)
    if current.shape != (2, 2) or not np.all(np.isfinite(current)):
        raise ValueError("pair_current must be a finite 2-by-2 matrix")
    active_result = np.zeros((2, 2), dtype=np.complex128)
    spectator_result = np.zeros((2, 2), dtype=np.complex128)
    pair_index = {-1: 0, 1: 1}
    for out_index, out_helicity in enumerate((-1, 1)):
        bra = doublet.state(out_helicity)
        for in_index, in_helicity in enumerate((-1, 1)):
            ket = doublet.state(in_helicity)
            active_value = 0.0 + 0.0j
            spectator_value = 0.0 + 0.0j
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
                    coefficient = (
                        np.conjugate(bra_component.amplitude)
                        * ket_component.amplitude
                    )
                    active_value += (
                        coefficient
                        * float(bra_component.active_charge)
                        * factors.active_cluster_overlap
                        * current[
                            pair_index[bra_component.active_helicity_twice],
                            pair_index[ket_component.active_helicity_twice],
                        ]
                    )
                    if (
                        bra_component.active_helicity_twice
                        == ket_component.active_helicity_twice
                    ):
                        spectator_charge = float(
                            bra.target_charge - bra_component.active_charge
                        )
                        spectator_value += (
                            coefficient
                            * spectator_charge
                            * factors.spectator_overlap
                        )
            active_result[out_index, in_index] = active_value
            spectator_result[out_index, in_index] = spectator_value
    return active_result, spectator_result, active_result + spectator_result


@dataclass(frozen=True)
class NNResolvedSpectatorImpulseCurrent:
    """NN convolution of the matched active and resolved-spectator current."""

    transfer: NNTransferFibers
    hamiltonian: M2ResolvedSpectatorHamiltonian
    internal_operator: M2ActiveQuarkImpulseOperator
    factors: ResolvedSpectatorCurrentFactors
    pair_current: np.ndarray
    proton_current: np.ndarray
    neutron_current: np.ndarray
    active_target_matrix: np.ndarray
    spectator_target_matrix: np.ndarray
    target_matrix: np.ndarray
    component_matrices: Mapping[str, np.ndarray]
    claim_tier: str = M2_RESOLVED_CURRENT_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.transfer, NNTransferFibers):
            raise TypeError("transfer must be NNTransferFibers")
        if not isinstance(self.hamiltonian, M2ResolvedSpectatorHamiltonian):
            raise TypeError("hamiltonian has the wrong type")
        if self.hamiltonian.pair is not self.transfer.state.pair:
            raise ValueError("Hamiltonian and NN state must share one M2 pair")
        if not isinstance(self.internal_operator, M2ActiveQuarkImpulseOperator):
            raise TypeError("internal_operator has the wrong type")
        if not isinstance(self.factors, ResolvedSpectatorCurrentFactors):
            raise TypeError("factors have the wrong type")
        for name, matrix, shape in (
            ("pair_current", self.pair_current, (2, 2)),
            ("proton_current", self.proton_current, (2, 2)),
            ("neutron_current", self.neutron_current, (2, 2)),
            ("active_target_matrix", self.active_target_matrix, (3, 3)),
            ("spectator_target_matrix", self.spectator_target_matrix, (3, 3)),
            ("target_matrix", self.target_matrix, (3, 3)),
        ):
            values = np.asarray(matrix)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite {shape} matrix")
        if set(self.component_matrices) != {"SS", "SD", "DS", "DD"}:
            raise ValueError("current must retain SS, SD, DS, and DD")
        delta = self.transfer.frame.delta_t
        if (delta.x, delta.y) != (
            self.internal_operator.delta_x_GeV,
            self.internal_operator.delta_y_GeV,
        ):
            raise ValueError("nuclear and active currents must share one transfer")
        if (delta.x, delta.y) != (
            self.factors.delta_x_GeV,
            self.factors.delta_y_GeV,
        ):
            raise ValueError("nuclear and spectator currents must share one transfer")
        if self.claim_tier != M2_RESOLVED_CURRENT_CLAIM_TIER or self.physical:
            raise ValueError("the resolved one-body current must remain exploratory")

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

    def mechanism_reconstruction_residual(self) -> float:
        return float(
            np.max(
                np.abs(
                    self.target_matrix
                    - self.active_target_matrix
                    - self.spectator_target_matrix
                )
            )
        )


def evaluate_nn_resolved_spectator_impulse_current(
    transfer: NNTransferFibers,
    operator: M2ActiveQuarkImpulseOperator,
    hamiltonian: M2ResolvedSpectatorHamiltonian,
    *,
    pair_basis: np.ndarray | None = None,
    quadrature_nodes: int = 28,
) -> NNResolvedSpectatorImpulseCurrent:
    """Insert all three resolved diagonal quark currents into the NN overlap."""

    if not isinstance(transfer, NNTransferFibers):
        raise TypeError("transfer must be NNTransferFibers")
    if not isinstance(hamiltonian, M2ResolvedSpectatorHamiltonian):
        raise TypeError("hamiltonian must be M2ResolvedSpectatorHamiltonian")
    pair_current = pair_active_current_matrix(
        transfer.state.pair, operator, basis=pair_basis
    )
    factors = build_resolved_spectator_current_factors(
        hamiltonian.resolution,
        delta_x_GeV=operator.delta_x_GeV,
        delta_y_GeV=operator.delta_y_GeV,
        quadrature_nodes=quadrature_nodes,
    )
    proton_active, proton_spectator, proton = nucleon_resolved_impulse_components(
        transfer.state.proton, pair_current, factors
    )
    neutron_active, neutron_spectator, neutron = nucleon_resolved_impulse_components(
        transfer.state.neutron, pair_current, factors
    )
    active_isosinglet = proton_active + neutron_active
    spectator_isosinglet = proton_spectator + neutron_spectator
    total_isosinglet = active_isosinglet + spectator_isosinglet
    grid = transfer.state.wave_grid
    delta = transfer.frame.delta_t
    active_target = np.zeros((3, 3), dtype=np.complex128)
    spectator_target = np.zeros((3, 3), dtype=np.complex128)
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
            active_piece = factor * np.einsum(
                "IHca,ca->IH", density, active_isosinglet
            )
            spectator_piece = factor * np.einsum(
                "IHca,ca->IH", density, spectator_isosinglet
            )
            total_piece = factor * np.einsum(
                "IHca,ca->IH", density, total_isosinglet
            )
            active_target += active_piece
            spectator_target += spectator_piece
            components[label] += total_piece
    target = sum(components.values())
    return NNResolvedSpectatorImpulseCurrent(
        transfer=transfer,
        hamiltonian=hamiltonian,
        internal_operator=operator,
        factors=factors,
        pair_current=pair_current,
        proton_current=proton,
        neutron_current=neutron,
        active_target_matrix=active_target,
        spectator_target_matrix=spectator_target,
        target_matrix=target,
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


def nn_resolved_spectator_current_pair_record(
    positive: NNResolvedSpectatorImpulseCurrent,
    reversed_transfer: NNResolvedSpectatorImpulseCurrent,
    forward: NNResolvedSpectatorImpulseCurrent,
) -> Mapping[str, Any]:
    """Report one-body matching closure and preserve all open current terms."""

    if not (
        positive.transfer.state is reversed_transfer.transfer.state
        and positive.transfer.state is forward.transfer.state
    ):
        raise ValueError("current comparison requires one NN state")
    if not (
        positive.hamiltonian is reversed_transfer.hamiltonian
        and positive.hamiltonian is forward.hamiltonian
    ):
        raise ValueError("current comparison requires one Hamiltonian embedding")
    delta = positive.transfer.frame.delta_t
    reversed_delta = reversed_transfer.transfer.frame.delta_t
    forward_delta = forward.transfer.frame.delta_t
    if reversed_delta != delta.scale(-1.0):
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
        "status": M2_RESOLVED_CURRENT_STATUS,
        "claim_tier": positive.claim_tier,
        "physical": False,
        "hamiltonian_status": M2_RESOLVED_SPECTATOR_STATUS,
        "hamiltonian_root": positive.hamiltonian.hamiltonian_root,
        "resolution_id": positive.hamiltonian.resolution.resolution_id,
        "internal_resolution": K9,
        "nuclear_resolution": positive.transfer.state.wave_grid.resolution.resolution_id,
        "delta_t_GeV": (delta.x, delta.y),
        "q2_GeV2": positive.factors.q2_GeV2,
        "active_lambda_shift_GeV": positive.factors.active_lambda_shift_GeV,
        "spectator_lambda_shift_GeV": positive.factors.spectator_lambda_shift_GeV,
        "spectator_rho_shift_GeV": positive.factors.spectator_rho_shift_GeV,
        "active_cluster_overlap": positive.factors.active_cluster_overlap,
        "spectator_quark_overlap": positive.factors.spectator_overlap,
        "pair_current": tuple(
            tuple(complex(value) for value in row) for row in positive.pair_current
        ),
        "proton_current": tuple(
            tuple(complex(value) for value in row) for row in positive.proton_current
        ),
        "neutron_current": tuple(
            tuple(complex(value) for value in row) for row in positive.neutron_current
        ),
        "active_target_matrix": tuple(
            tuple(complex(value) for value in row)
            for row in positive.active_target_matrix
        ),
        "spectator_target_matrix": tuple(
            tuple(complex(value) for value in row)
            for row in positive.spectator_target_matrix
        ),
        "target_matrix": tuple(
            tuple(complex(value) for value in row) for row in plus
        ),
        "spin_one_amplitudes": {
            "I11": spin.plus_plus,
            "I10": spin.plus_zero,
            "I1m1": spin.plus_minus,
            "I00": spin.zero_zero,
        },
        "forward_active_overlap_residual": abs(
            forward.factors.active_cluster_overlap - 1.0
        ),
        "forward_spectator_overlap_residual": abs(
            forward.factors.spectator_overlap - 1.0
        ),
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
        "active_spectator_reconstruction_residual": max(
            positive.mechanism_reconstruction_residual(),
            reversed_transfer.mechanism_reconstruction_residual(),
            forward.mechanism_reconstruction_residual(),
        ),
        "active_and_spectator_one_body_blocks_matched": True,
        "same_longitudinal_transverse_regulator": True,
        "pointlike_spectator_identity_used": False,
        "complete_diagonal_one_body_current_in_represented_factorization": True,
        "hamiltonian_consistent_interaction_current_constructed": False,
        "production_current": False,
        "unavailable_current_terms": UNAVAILABLE_MATCHED_CURRENT_TERMS,
    }
    return {**payload, "root": _json_root(payload)}


__all__ = [
    "M2_RESOLVED_SPECTATOR_STATUS",
    "M2_RESOLVED_CURRENT_STATUS",
    "M2_RESOLVED_SPECTATOR_CLAIM_TIER",
    "M2_RESOLVED_CURRENT_CLAIM_TIER",
    "RESOLVED_SPECTATOR_SOURCES",
    "UNAVAILABLE_MATCHED_CURRENT_TERMS",
    "ResolvedSpectatorResolution",
    "M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1",
    "resolved_sector_longitudinal_records",
    "M2ResolvedSpectatorHamiltonian",
    "build_resolved_spectator_hamiltonian",
    "resolved_spectator_hamiltonian_record",
    "ResolvedSpectatorCurrentFactors",
    "build_resolved_spectator_current_factors",
    "nucleon_resolved_impulse_components",
    "NNResolvedSpectatorImpulseCurrent",
    "evaluate_nn_resolved_spectator_impulse_current",
    "nn_resolved_spectator_current_pair_record",
]

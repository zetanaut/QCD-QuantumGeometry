"""First separately resolved NN spin-one state over the typed M2 nucleons.

The state is deliberately factorized into three independently owned pieces:

* the K9 C53-mixed internal active state and typed proton/neutron spectator
  family;
* a separate nuclear quadrature resolution ``r_D``;
* the existing light-front S/D and Melosh-rotation convention evaluated with
  one named AV18 momentum-space wave-function member.

This constructs a normalized exploratory NN state in
``H_D^(r_D;K9)``.  It does not construct a nuclear Hamiltonian, select a
scalar/axial mixture, add nonnucleonic sectors, or define a current.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from deuteron_wigner.conventions import HELICITIES
from deuteron_wigner.light_front import (
    InternalMomentum,
    LFNormalization,
    SpinRotation,
    light_front_wave_function,
)
from deuteron_wigner.wavefunctions.av18 import load_av18_momentum

from .m2_c53_mixed_singlet_state import K9, C53MixedK9HelicityPair
from .m2_nucleon_spectator_composition import (
    SPECTATOR_CHANNELS,
    NucleonSpectatorDoublet,
    build_nucleon_spectator_doublet,
)


HBARC_GEV_FM = 0.1973269804
NN_DEUTERON_CLAIM_TIER = "EXPLORATORY_SEPARATELY_RESOLVED_NN_SPIN1_STATE"
OMITTED_NUCLEAR_SECTORS = (
    "NNPI",
    "DELTADELTA",
    "SIX_QUARK",
    "HIDDEN_COLOR",
    "OTHER_NONNUCLEONIC",
)
_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class NNNuclearResolution:
    """Nuclear quadrature data kept distinct from the internal M2 K label."""

    resolution_id: str
    n_k: int
    n_cos_theta: int
    n_phi: int
    k_max_GeV: float
    l_max: int = 2

    def __post_init__(self) -> None:
        if not self.resolution_id or self.resolution_id in {"K9", "K11", "K13"}:
            raise ValueError("nuclear resolution_id must be nonempty and distinct from M2 K")
        if min(self.n_k, self.n_cos_theta, self.n_phi) < 2:
            raise ValueError("all nuclear quadrature orders must be at least two")
        if self.n_phi % 2:
            raise ValueError("n_phi must be even for the symmetric azimuthal grid")
        if not np.isfinite(self.k_max_GeV) or self.k_max_GeV <= 0.0:
            raise ValueError("k_max_GeV must be positive and finite")
        if self.l_max < 2:
            raise ValueError("the first NN state must retain S and D waves")


NN_AV18_NUCLEAR_R1 = NNNuclearResolution(
    resolution_id="NN_AV18_R1_NK24_NC14_NP16_KMAX2GEV",
    n_k=24,
    n_cos_theta=14,
    n_phi=16,
    k_max_GeV=2.0,
)


@dataclass(frozen=True)
class NNWaveFunctionMember:
    member_id: str
    label: str
    source: str
    source_sha256: str
    radial_GeV: Callable[[float], tuple[float, float]]
    source_norm: float
    source_d_probability: float
    source_momentum_max_GeV: float
    physical_wave_input: bool

    def __post_init__(self) -> None:
        if not all((self.member_id, self.label, self.source, self.source_sha256)):
            raise ValueError("wave-function provenance fields must be nonempty")
        if self.source_norm <= 0.0 or not np.isfinite(self.source_norm):
            raise ValueError("source_norm must be positive and finite")
        if not 0.0 <= self.source_d_probability <= self.source_norm:
            raise ValueError("source D probability lies outside the source norm")


def build_av18_wave_member() -> NNWaveFunctionMember:
    """Load the repository AV18 momentum member with an explicit unit adapter."""

    relative = Path("data/raw/av18/deut.wfk")
    source = _ROOT / relative
    table = load_av18_momentum(source)
    radial_fm = table.radial_callable()
    scale = HBARC_GEV_FM ** (-1.5)

    def radial_GeV(momentum_GeV: float) -> tuple[float, float]:
        if momentum_GeV < 0.0:
            raise ValueError("momentum_GeV must be nonnegative")
        u_value, w_value = radial_fm(momentum_GeV / HBARC_GEV_FM)
        return scale * u_value, scale * w_value

    return NNWaveFunctionMember(
        member_id="NN_WAVE_AV18_MOMENTUM_TABLE_GEV_ADAPTER_V1",
        label="AV18",
        source=str(relative),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        radial_GeV=radial_GeV,
        source_norm=table.norm(),
        source_d_probability=table.d_state_probability(),
        source_momentum_max_GeV=float(table.grid[-1] * HBARC_GEV_FM),
        physical_wave_input=True,
    )


@dataclass(frozen=True)
class NNLightFrontWaveGrid:
    """One normalized finite-grid realization of the Volume-IV NN amplitude."""

    resolution: NNNuclearResolution
    wave_member: NNWaveFunctionMember
    nucleon_mass_GeV: float
    y: np.ndarray
    p_x_GeV: np.ndarray
    p_y_GeV: np.ndarray
    flat_measure_weights: np.ndarray
    amplitudes: np.ndarray
    s_amplitudes: np.ndarray
    d_amplitudes: np.ndarray
    raw_helicity_norms: np.ndarray
    common_normalization_factor: float
    claim_tier: str = NN_DEUTERON_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        node_count = len(self.y)
        if any(
            np.asarray(values).shape != (node_count,)
            for values in (
                self.p_x_GeV,
                self.p_y_GeV,
                self.flat_measure_weights,
            )
        ):
            raise ValueError("nuclear node vectors must share one shape")
        expected = (3, node_count, 2, 2)
        if any(
            np.asarray(values).shape != expected
            for values in (self.amplitudes, self.s_amplitudes, self.d_amplitudes)
        ):
            raise ValueError(f"nuclear amplitudes must have shape {expected}")
        if np.asarray(self.raw_helicity_norms).shape != (3,):
            raise ValueError("raw_helicity_norms must have shape (3,)")
        if not np.all((np.asarray(self.y) > 0.0) & (np.asarray(self.y) < 1.0)):
            raise ValueError("nuclear y nodes must lie in the open unit interval")
        if np.any(np.asarray(self.flat_measure_weights) <= 0.0):
            raise ValueError("nuclear quadrature weights must be positive")
        if self.physical or self.claim_tier != NN_DEUTERON_CLAIM_TIER:
            raise ValueError("the first NN grid must remain exploratory")

    @property
    def node_count(self) -> int:
        return len(self.y)

    def helicity_gram(self) -> np.ndarray:
        flattened = self.amplitudes.reshape(3, -1)
        return flattened.conj() @ flattened.T

    def component_norms(self) -> tuple[np.ndarray, np.ndarray]:
        s_norms = np.sum(np.abs(self.s_amplitudes) ** 2, axis=(1, 2, 3))
        d_norms = np.sum(np.abs(self.d_amplitudes) ** 2, axis=(1, 2, 3))
        return s_norms, d_norms

    def signed_sd_interference(self) -> np.ndarray:
        return 2.0 * np.real(
            np.sum(
                self.s_amplitudes.conj() * self.d_amplitudes,
                axis=(1, 2, 3),
            )
        )


def build_nn_light_front_wave_grid(
    *,
    resolution: NNNuclearResolution,
    wave_member: NNWaveFunctionMember,
    nucleon_mass_GeV: float,
) -> NNLightFrontWaveGrid:
    """Evaluate and commonly normalize one named S/D member on ``r_D``."""

    if not isinstance(resolution, NNNuclearResolution):
        raise TypeError("resolution must be an NNNuclearResolution")
    if not isinstance(wave_member, NNWaveFunctionMember):
        raise TypeError("wave_member must be an NNWaveFunctionMember")
    if not np.isfinite(nucleon_mass_GeV) or nucleon_mass_GeV <= 0.0:
        raise ValueError("nucleon_mass_GeV must be positive and finite")
    if resolution.k_max_GeV > wave_member.source_momentum_max_GeV:
        raise ValueError("nuclear k_max exceeds the named wave-function domain")

    k_nodes, k_weights = np.polynomial.legendre.leggauss(resolution.n_k)
    k_nodes = 0.5 * resolution.k_max_GeV * (k_nodes + 1.0)
    k_weights = 0.5 * resolution.k_max_GeV * k_weights
    cos_nodes, cos_weights = np.polynomial.legendre.leggauss(
        resolution.n_cos_theta
    )
    phi_nodes = (
        2.0 * np.pi * np.arange(resolution.n_phi, dtype=float) / resolution.n_phi
    )
    phi_weight = 2.0 * np.pi / resolution.n_phi

    y_values: list[float] = []
    px_values: list[float] = []
    py_values: list[float] = []
    flat_weights: list[float] = []
    full_values: list[np.ndarray] = []
    s_values: list[np.ndarray] = []
    d_values: list[np.ndarray] = []
    for momentum, radial_weight in zip(k_nodes, k_weights):
        u_value, w_value = wave_member.radial_GeV(float(momentum))
        for cos_theta, angular_weight in zip(cos_nodes, cos_weights):
            transverse = momentum * sqrt(max(0.0, 1.0 - cos_theta**2))
            k_z = float(momentum * cos_theta)
            d3_weight = float(
                radial_weight * momentum**2 * angular_weight * phi_weight
            )
            for phi in phi_nodes:
                p_x = float(transverse * np.cos(phi))
                p_y = float(transverse * np.sin(phi))
                internal = InternalMomentum.from_cartesian(
                    k_z=k_z, p_x=p_x, p_y=p_y, mass=nucleon_mass_GeV
                )
                flat_weight = d3_weight / internal.dkz_dy
                arguments = {
                    "y": internal.y,
                    "p_x": p_x,
                    "p_y": p_y,
                    "mass": nucleon_mass_GeV,
                    "normalization": LFNormalization.FLAT,
                    "spin_rotation": SpinRotation.MELOSH,
                }
                full = light_front_wave_function(
                    **arguments,
                    radial=lambda _k, u=u_value, w=w_value: (u, w),
                )
                s_only = light_front_wave_function(
                    **arguments,
                    radial=lambda _k, u=u_value: (u, 0.0),
                )
                d_only = light_front_wave_function(
                    **arguments,
                    radial=lambda _k, w=w_value: (0.0, w),
                )
                scale = sqrt(flat_weight)
                y_values.append(internal.y)
                px_values.append(p_x)
                py_values.append(p_y)
                flat_weights.append(flat_weight)
                full_values.append(scale * full)
                s_values.append(scale * s_only)
                d_values.append(scale * d_only)

    full_array = np.stack(full_values, axis=1)
    s_array = np.stack(s_values, axis=1)
    d_array = np.stack(d_values, axis=1)
    raw_norms = np.sum(np.abs(full_array) ** 2, axis=(1, 2, 3))
    if float(np.max(raw_norms) - np.min(raw_norms)) > 2.0e-12:
        raise RuntimeError("one common normalization cannot close all helicities")
    normalization = 1.0 / sqrt(float(np.mean(raw_norms)))
    full_array *= normalization
    s_array *= normalization
    d_array *= normalization
    reconstruction = float(np.max(np.abs(full_array - s_array - d_array)))
    if reconstruction > 2.0e-14:
        raise RuntimeError("S/D amplitudes do not reconstruct the NN wave grid")
    return NNLightFrontWaveGrid(
        resolution=resolution,
        wave_member=wave_member,
        nucleon_mass_GeV=float(nucleon_mass_GeV),
        y=np.asarray(y_values),
        p_x_GeV=np.asarray(px_values),
        p_y_GeV=np.asarray(py_values),
        flat_measure_weights=np.asarray(flat_weights),
        amplitudes=full_array,
        s_amplitudes=s_array,
        d_amplitudes=d_array,
        raw_helicity_norms=raw_norms,
        common_normalization_factor=normalization,
    )


def isospin_singlet_report() -> Mapping[str, Any]:
    """Return the explicit ``(pn-np)/sqrt(2)`` isospin-zero factor."""

    vector = np.asarray((0.0, 1.0, -1.0, 0.0), dtype=np.complex128) / sqrt(2.0)
    exchange = np.asarray(
        ((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)),
        dtype=np.complex128,
    )
    pauli = (
        np.asarray(((0, 1), (1, 0)), dtype=np.complex128),
        np.asarray(((0, -1j), (1j, 0)), dtype=np.complex128),
        np.asarray(((1, 0), (0, -1)), dtype=np.complex128),
    )
    identity = np.eye(2, dtype=np.complex128)
    total = tuple(
        0.5 * (np.kron(axis, identity) + np.kron(identity, axis))
        for axis in pauli
    )
    total_squared = sum(axis @ axis for axis in total)
    return {
        "basis_order": ("pp", "pn", "np", "nn"),
        "amplitudes": tuple(float(value.real) for value in vector),
        "normalization_residual": abs(float(np.vdot(vector, vector).real) - 1.0),
        "exchange_eigenvalue": -1.0,
        "exchange_residual": float(np.linalg.norm(exchange @ vector + vector)),
        "I_total": 0,
        "I_squared_residual": float(np.linalg.norm(total_squared @ vector)),
        "component_charge": 1,
        "component_baryon_number": 2,
    }


@dataclass(frozen=True)
class ExploratoryNNDeuteronState:
    """Factorized NN state in ``H_D^(r_D;K9)`` for one spectator family."""

    pair: C53MixedK9HelicityPair
    proton: NucleonSpectatorDoublet
    neutron: NucleonSpectatorDoublet
    wave_grid: NNLightFrontWaveGrid
    channel: str
    claim_tier: str = NN_DEUTERON_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        if self.channel not in SPECTATOR_CHANNELS:
            raise ValueError("channel must be an explicitly named spectator family")
        if self.proton.target != "PROTON" or self.neutron.target != "NEUTRON":
            raise ValueError("the NN state requires correlated proton/neutron inputs")
        if self.proton.channel != self.channel or self.neutron.channel != self.channel:
            raise ValueError("proton and neutron must use the same named family")
        if self.proton.pair is not self.pair or self.neutron.pair is not self.pair:
            raise ValueError("proton and neutron must share one internal M2 member")
        if self.physical or self.claim_tier != NN_DEUTERON_CLAIM_TIER:
            raise ValueError("the first NN state must remain exploratory")

    @property
    def hilbert_space_id(self) -> str:
        return (
            f"H_D^({self.wave_grid.resolution.resolution_id};"
            f"{K9})::NN_ONLY"
        )


def compose_nn_deuteron_state(
    pair: C53MixedK9HelicityPair,
    *,
    wave_grid: NNLightFrontWaveGrid,
    channel: str,
) -> ExploratoryNNDeuteronState:
    """Bind one named nucleon family to the separate NN light-front grid."""

    if not isinstance(pair, C53MixedK9HelicityPair):
        raise TypeError("pair must be a C53MixedK9HelicityPair")
    if not isinstance(wave_grid, NNLightFrontWaveGrid):
        raise TypeError("wave_grid must be an NNLightFrontWaveGrid")
    if channel not in SPECTATOR_CHANNELS:
        raise ValueError("channel must be an explicitly named spectator family")
    proton = build_nucleon_spectator_doublet(
        pair, target="PROTON", channel=channel
    )
    neutron = build_nucleon_spectator_doublet(
        pair, target="NEUTRON", channel=channel
    )
    return ExploratoryNNDeuteronState(
        pair=pair,
        proton=proton,
        neutron=neutron,
        wave_grid=wave_grid,
        channel=channel,
    )


def nn_deuteron_state_record(state: ExploratoryNNDeuteronState) -> Mapping[str, Any]:
    """Report exact state ledgers and the remaining Hamiltonian/current gap."""

    grid = state.wave_grid
    gram = grid.helicity_gram()
    s_norms, d_norms = grid.component_norms()
    interference = grid.signed_sd_interference()
    isospin = isospin_singlet_report()
    return {
        "status": "M2_EXPLORATORY_SEPARATELY_RESOLVED_NN_SPIN1_STATE_READY",
        "hilbert_space_id": state.hilbert_space_id,
        "internal_resolution": K9,
        "nuclear_resolution": grid.resolution.resolution_id,
        "resolution_labels_independent": True,
        "sector": "NN_ONLY",
        "target_helicities": HELICITIES,
        "helicity_gram_residual": float(np.max(np.abs(gram - np.eye(3)))),
        "raw_helicity_norm_spread": float(
            np.max(grid.raw_helicity_norms) - np.min(grid.raw_helicity_norms)
        ),
        "common_normalization_factor": grid.common_normalization_factor,
        "node_count": grid.node_count,
        "wave_member_id": grid.wave_member.member_id,
        "wave_label": grid.wave_member.label,
        "wave_source": grid.wave_member.source,
        "wave_source_sha256": grid.wave_member.source_sha256,
        "wave_source_norm": grid.wave_member.source_norm,
        "wave_source_d_probability": grid.wave_member.source_d_probability,
        "discrete_s_probabilities": tuple(float(value) for value in s_norms),
        "discrete_d_probabilities": tuple(float(value) for value in d_norms),
        "signed_integrated_sd_interference": tuple(
            float(value) for value in interference
        ),
        "local_sd_interference_max_abs": float(
            np.max(
                np.abs(
                    2.0
                    * np.real(grid.s_amplitudes.conj() * grid.d_amplitudes)
                )
            )
        ),
        "sd_reconstruction_residual": float(
            np.max(
                np.abs(
                    grid.amplitudes - grid.s_amplitudes - grid.d_amplitudes
                )
            )
        ),
        "spin_rotation": SpinRotation.MELOSH.value,
        "lf_normalization": LFNormalization.FLAT.value,
        "isospin": isospin,
        "charge": 1,
        "baryon_number": 2,
        "parity": 1,
        "J": 1,
        "nucleon_family": state.channel,
        "scalar_axial_mixture_selected": False,
        "internal_qg_weight": state.pair.state(-1).qg_weight,
        "nuclear_hamiltonian_constructed": False,
        "deuteron_mass_eigenvalue_constructed": False,
        "initial_final_transfer_fibers_constructed": False,
        "current_constructed": False,
        "omitted_nuclear_sectors": OMITTED_NUCLEAR_SECTORS,
        "claim_tier": state.claim_tier,
        "physical": state.physical,
    }


def nn_deuteron_family_record(
    pair: C53MixedK9HelicityPair,
    *,
    wave_grid: NNLightFrontWaveGrid,
) -> Mapping[str, Any]:
    """Report scalar and axial NN members without selecting between them."""

    members = tuple(
        nn_deuteron_state_record(
            compose_nn_deuteron_state(pair, wave_grid=wave_grid, channel=channel)
        )
        for channel in SPECTATOR_CHANNELS
    )
    return {
        "status": "M2_EXPLORATORY_NN_SPIN1_SCALAR_AXIAL_FAMILIES_READY",
        "members": members,
        "member_count": len(members),
        "scalar_axial_mixture_selected": False,
        "nuclear_hamiltonian_constructed": False,
        "current_constructed": False,
        "physical": False,
    }


__all__ = [
    "HBARC_GEV_FM",
    "NN_DEUTERON_CLAIM_TIER",
    "OMITTED_NUCLEAR_SECTORS",
    "NNNuclearResolution",
    "NN_AV18_NUCLEAR_R1",
    "NNWaveFunctionMember",
    "NNLightFrontWaveGrid",
    "ExploratoryNNDeuteronState",
    "build_av18_wave_member",
    "build_nn_light_front_wave_grid",
    "isospin_singlet_report",
    "compose_nn_deuteron_state",
    "nn_deuteron_state_record",
    "nn_deuteron_family_record",
]

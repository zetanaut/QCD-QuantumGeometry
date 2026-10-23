"""Typed spectator spin/flavor completion of the exploratory M2 helicity pair.

The M2 color bridge supplies an anti-fundamental spectator color, and the C53
launch supplies a two-dimensional active ``Jz=+/-1/2`` state space.  This
module adds the missing two-quark spectator spin and isospin tensors and
couples them to explicit proton and neutron doublets.

Two inequivalent S-wave families are kept separate:

* ``SCALAR_I0_S0_L0`` has spectator spin/isospin ``(0,0)``;
* ``AXIAL_I1_S1_L0`` has spectator spin/isospin ``(1,1)`` and uses explicit
  Condon--Shortley Clebsch--Gordan coefficients.

No scalar/axial mixture is selected.  The construction extends the active
Hamiltonian by the identity on spectator spin/flavor only; it contains no
spectator kinetic or interaction operator and is therefore a typed
exploratory composition, not a physical proton/neutron solution or deuteron.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
from typing import Any, Mapping

import numpy as np

from .m2_c53_mixed_singlet_state import C53MixedK9HelicityPair


SCALAR_SPECTATOR = "SCALAR_I0_S0_L0"
AXIAL_SPECTATOR = "AXIAL_I1_S1_L0"
SPECTATOR_CHANNELS = (SCALAR_SPECTATOR, AXIAL_SPECTATOR)
TARGETS = ("PROTON", "NEUTRON")
SPECTATOR_COMPOSITION_CLAIM_TIER = (
    "EXPLORATORY_TYPED_SPECTATOR_SPIN_FLAVOR_COMPOSITION"
)

_FLAVOR_CHARGE = {"u": Fraction(2, 3), "d": Fraction(-1, 3)}
_FLAVOR_ISOSPIN_TWICE = {"u": 1, "d": -1}
_TARGET_CHARGE = {"PROTON": Fraction(1), "NEUTRON": Fraction(0)}
_TARGET_ISOSPIN_TWICE = {"PROTON": 1, "NEUTRON": -1}
_PAIR_BASIS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_FLAVOR_PAIR_BASIS = (("u", "u"), ("u", "d"), ("d", "u"), ("d", "d"))


def _validate_target(target: str) -> str:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    return target


def _validate_channel(channel: str) -> str:
    if channel not in SPECTATOR_CHANNELS:
        raise ValueError(f"channel must be one of {SPECTATOR_CHANNELS}")
    return channel


def spectator_spin_wavefunction(channel: str, projection: int) -> np.ndarray:
    """Return the normalized ordered-pair spin tensor in ``++,+-,-+,--``."""

    _validate_channel(channel)
    vector = np.zeros(4, dtype=np.complex128)
    if channel == SCALAR_SPECTATOR:
        if projection != 0:
            raise ValueError("the scalar spectator has spin projection zero")
        vector[1:3] = (1.0 / sqrt(2.0), -1.0 / sqrt(2.0))
        return vector
    if projection == 1:
        vector[0] = 1.0
    elif projection == 0:
        vector[1:3] = 1.0 / sqrt(2.0)
    elif projection == -1:
        vector[3] = 1.0
    else:
        raise ValueError("the axial spectator spin projection must be -1, 0, or 1")
    return vector


def spectator_flavor_wavefunction(
    channel: str, isospin_projection: int
) -> np.ndarray:
    """Return the normalized ordered-pair flavor tensor in ``uu,ud,du,dd``."""

    _validate_channel(channel)
    vector = np.zeros(4, dtype=np.complex128)
    if channel == SCALAR_SPECTATOR:
        if isospin_projection != 0:
            raise ValueError("the scalar spectator has isospin projection zero")
        vector[1:3] = (1.0 / sqrt(2.0), -1.0 / sqrt(2.0))
        return vector
    if isospin_projection == 1:
        vector[0] = 1.0
    elif isospin_projection == 0:
        vector[1:3] = 1.0 / sqrt(2.0)
    elif isospin_projection == -1:
        vector[3] = 1.0
    else:
        raise ValueError(
            "the axial spectator isospin projection must be -1, 0, or 1"
        )
    return vector


def _pair_exchange_matrix() -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    for index, pair in enumerate(_PAIR_BASIS):
        matrix[_PAIR_BASIS.index((pair[1], pair[0])), index] = 1.0
    return matrix


def _flavor_exchange_matrix() -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    for index, pair in enumerate(_FLAVOR_PAIR_BASIS):
        matrix[_FLAVOR_PAIR_BASIS.index((pair[1], pair[0])), index] = 1.0
    return matrix


def spectator_exchange_report(channel: str) -> Mapping[str, Any]:
    """Verify spectator permutation symmetry and total fermion antisymmetry."""

    _validate_channel(channel)
    projections = (0,) if channel == SCALAR_SPECTATOR else (-1, 0, 1)
    expected = -1.0 if channel == SCALAR_SPECTATOR else 1.0
    spin_exchange = _pair_exchange_matrix()
    flavor_exchange = _flavor_exchange_matrix()
    spin_residual = max(
        float(
            np.linalg.norm(
                spin_exchange @ spectator_spin_wavefunction(channel, projection)
                - expected * spectator_spin_wavefunction(channel, projection)
            )
        )
        for projection in projections
    )
    flavor_residual = max(
        float(
            np.linalg.norm(
                flavor_exchange
                @ spectator_flavor_wavefunction(channel, projection)
                - expected * spectator_flavor_wavefunction(channel, projection)
            )
        )
        for projection in projections
    )
    return {
        "channel": channel,
        "spin_exchange_eigenvalue": expected,
        "flavor_exchange_eigenvalue": expected,
        "orbital_exchange_eigenvalue": 1.0,
        "color_exchange_eigenvalue": -1.0,
        "total_spectator_quark_exchange_eigenvalue": -1.0,
        "spin_exchange_residual": spin_residual,
        "flavor_exchange_residual": flavor_residual,
        "color_tensor": "C7_MATCHED_ANTITRIPLET_3BAR",
    }


def _spin_terms(channel: str, helicity_twice: int) -> tuple[tuple[int, int, float], ...]:
    """Return ``(active_helicity_twice, spectator_m, CG)`` terms."""

    _validate_channel(channel)
    if helicity_twice not in (-1, 1):
        raise ValueError("helicity_twice must be -1 or 1")
    if channel == SCALAR_SPECTATOR:
        return ((helicity_twice, 0, 1.0),)
    if helicity_twice == 1:
        return (
            (-1, 1, sqrt(2.0 / 3.0)),
            (1, 0, -sqrt(1.0 / 3.0)),
        )
    return (
        (-1, 0, sqrt(1.0 / 3.0)),
        (1, -1, -sqrt(2.0 / 3.0)),
    )


def _isospin_terms(
    channel: str, target: str
) -> tuple[tuple[str, int, float], ...]:
    """Return ``(active_flavor, spectator_Iz, CG)`` terms."""

    _validate_channel(channel)
    _validate_target(target)
    if channel == SCALAR_SPECTATOR:
        return (("u" if target == "PROTON" else "d", 0, 1.0),)
    if target == "PROTON":
        return (
            ("d", 1, sqrt(2.0 / 3.0)),
            ("u", 0, -sqrt(1.0 / 3.0)),
        )
    return (
        ("d", 0, sqrt(1.0 / 3.0)),
        ("u", -1, -sqrt(2.0 / 3.0)),
    )


def _spectator_charge(channel: str, isospin_projection: int) -> Fraction:
    amplitudes = spectator_flavor_wavefunction(channel, isospin_projection)
    values = tuple(
        _FLAVOR_CHARGE[first] + _FLAVOR_CHARGE[second]
        for first, second in _FLAVOR_PAIR_BASIS
    )
    nonzero = [values[index] for index, value in enumerate(amplitudes) if value]
    if not nonzero or len(set(nonzero)) != 1:
        raise RuntimeError("spectator flavor tensor does not have definite charge")
    return nonzero[0]


@dataclass(frozen=True)
class NucleonCompositionComponent:
    """One orthogonal active/spectator term in a nucleon composition."""

    active_helicity_twice: int
    active_flavor: str
    spectator_spin_projection: int
    spectator_isospin_projection: int
    amplitude: complex

    def __post_init__(self) -> None:
        if self.active_helicity_twice not in (-1, 1):
            raise ValueError("active_helicity_twice must be -1 or 1")
        if self.active_flavor not in _FLAVOR_CHARGE:
            raise ValueError("active_flavor must be u or d")
        if self.spectator_spin_projection not in (-1, 0, 1):
            raise ValueError("invalid spectator spin projection")
        if self.spectator_isospin_projection not in (-1, 0, 1):
            raise ValueError("invalid spectator isospin projection")
        if not np.isfinite(self.amplitude):
            raise ValueError("amplitude must be finite")

    @property
    def active_charge(self) -> Fraction:
        return _FLAVOR_CHARGE[self.active_flavor]

    @property
    def baryon_number(self) -> Fraction:
        return Fraction(1, 3) + Fraction(2, 3)

    @property
    def active_isospin_twice(self) -> int:
        return _FLAVOR_ISOSPIN_TWICE[self.active_flavor]


@dataclass(frozen=True)
class NucleonSpectatorState:
    """One target and helicity in a named spectator channel."""

    pair: C53MixedK9HelicityPair
    target: str
    helicity_twice: int
    channel: str
    components: tuple[NucleonCompositionComponent, ...]
    claim_tier: str = SPECTATOR_COMPOSITION_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        _validate_target(self.target)
        _validate_channel(self.channel)
        if self.helicity_twice not in (-1, 1):
            raise ValueError("helicity_twice must be -1 or 1")
        if self.physical or self.claim_tier != SPECTATOR_COMPOSITION_CLAIM_TIER:
            raise ValueError("spectator compositions must remain exploratory")
        if not self.components:
            raise ValueError("components must be nonempty")
        keys = tuple(
            (
                item.active_helicity_twice,
                item.active_flavor,
                item.spectator_spin_projection,
                item.spectator_isospin_projection,
            )
            for item in self.components
        )
        if len(keys) != len(set(keys)):
            raise ValueError("spectator composition contains duplicate components")
        if abs(self.norm - 1.0) > 2.0e-14:
            raise ValueError("spectator composition is not normalized")
        if self.max_component_ledger_residual != 0.0:
            raise ValueError("a spectator component violates an exact quantum ledger")

    @property
    def norm(self) -> float:
        return float(sum(abs(item.amplitude) ** 2 for item in self.components))

    @property
    def target_charge(self) -> Fraction:
        return _TARGET_CHARGE[self.target]

    @property
    def target_isospin_twice(self) -> int:
        return _TARGET_ISOSPIN_TWICE[self.target]

    def component_ledger(self, item: NucleonCompositionComponent) -> Mapping[str, Any]:
        spectator_charge = _spectator_charge(
            self.channel, item.spectator_isospin_projection
        )
        return {
            "charge": item.active_charge + spectator_charge,
            "target_charge": self.target_charge,
            "baryon_number": item.baryon_number,
            "target_baryon_number": Fraction(1),
            "Jz_twice": (
                item.active_helicity_twice + 2 * item.spectator_spin_projection
            ),
            "target_Jz_twice": self.helicity_twice,
            "isospin_z_twice": (
                item.active_isospin_twice
                + 2 * item.spectator_isospin_projection
            ),
            "target_isospin_z_twice": self.target_isospin_twice,
            "spectator_orbital_Lz": 0,
            "spectator_parity": 1,
        }

    @property
    def max_component_ledger_residual(self) -> float:
        residuals = []
        for item in self.components:
            ledger = self.component_ledger(item)
            residuals.extend(
                (
                    abs(float(ledger["charge"] - ledger["target_charge"])),
                    abs(
                        float(
                            ledger["baryon_number"]
                            - ledger["target_baryon_number"]
                        )
                    ),
                    abs(ledger["Jz_twice"] - ledger["target_Jz_twice"]),
                    abs(
                        ledger["isospin_z_twice"]
                        - ledger["target_isospin_z_twice"]
                    ),
                )
            )
        return float(max(residuals, default=0.0))

    def active_pair_density(self) -> np.ndarray:
        """Trace the added spin/flavor labels, returning a 2-by-2 pair density."""

        density = np.zeros((2, 2), dtype=np.complex128)
        groups: dict[tuple[object, ...], np.ndarray] = {}
        for item in self.components:
            external = (
                item.active_flavor,
                item.spectator_spin_projection,
                item.spectator_isospin_projection,
                self.channel,
            )
            vector = groups.setdefault(external, np.zeros(2, dtype=np.complex128))
            index = 0 if item.active_helicity_twice == -1 else 1
            vector[index] += item.amplitude
        for vector in groups.values():
            density += np.outer(vector, vector.conj())
        return density

    @property
    def energy_GeV2(self) -> float:
        density = self.active_pair_density()
        energies = np.asarray(
            [self.pair.state(-1).energy_GeV2, self.pair.state(1).energy_GeV2]
        )
        return float(np.dot(np.diag(density).real, energies))

    @property
    def qg_weight(self) -> float:
        density = self.active_pair_density()
        weights = np.asarray(
            [self.pair.state(-1).qg_weight, self.pair.state(1).qg_weight]
        )
        return float(np.dot(np.diag(density).real, weights))


@dataclass(frozen=True)
class NucleonSpectatorDoublet:
    """The two exact-Jz branches for one target and spectator family."""

    pair: C53MixedK9HelicityPair
    target: str
    channel: str
    states: tuple[NucleonSpectatorState, NucleonSpectatorState]
    claim_tier: str = SPECTATOR_COMPOSITION_CLAIM_TIER
    physical: bool = False

    def __post_init__(self) -> None:
        _validate_target(self.target)
        _validate_channel(self.channel)
        if self.physical or self.claim_tier != SPECTATOR_COMPOSITION_CLAIM_TIER:
            raise ValueError("spectator doublets must remain exploratory")
        if tuple(state.helicity_twice for state in self.states) != (-1, 1):
            raise ValueError("doublet states must be ordered as Jz=-1/2,+1/2")
        if any(
            state.pair is not self.pair
            or state.target != self.target
            or state.channel != self.channel
            for state in self.states
        ):
            raise ValueError("doublet states do not share one pair/target/channel")

    def state(self, helicity_twice: int) -> NucleonSpectatorState:
        for value in self.states:
            if value.helicity_twice == helicity_twice:
                return value
        raise KeyError(helicity_twice)

    def unpolarized_active_pair_density(self) -> np.ndarray:
        return 0.5 * (
            self.state(-1).active_pair_density()
            + self.state(1).active_pair_density()
        )


def build_nucleon_spectator_doublet(
    pair: C53MixedK9HelicityPair,
    *,
    target: str,
    channel: str,
) -> NucleonSpectatorDoublet:
    """Attach one explicitly named spectator family to the C53 helicity pair."""

    if not isinstance(pair, C53MixedK9HelicityPair):
        raise TypeError("pair must be a C53MixedK9HelicityPair")
    _validate_target(target)
    _validate_channel(channel)
    states = []
    for helicity_twice in (-1, 1):
        components = tuple(
            NucleonCompositionComponent(
                active_helicity_twice=active_helicity_twice,
                active_flavor=active_flavor,
                spectator_spin_projection=spectator_spin_projection,
                spectator_isospin_projection=spectator_isospin_projection,
                amplitude=complex(spin_coefficient * isospin_coefficient),
            )
            for active_helicity_twice, spectator_spin_projection, spin_coefficient
            in _spin_terms(channel, helicity_twice)
            for active_flavor, spectator_isospin_projection, isospin_coefficient
            in _isospin_terms(channel, target)
        )
        states.append(
            NucleonSpectatorState(
                pair=pair,
                target=target,
                helicity_twice=helicity_twice,
                channel=channel,
                components=components,
            )
        )
    return NucleonSpectatorDoublet(
        pair=pair,
        target=target,
        channel=channel,
        states=(states[0], states[1]),
    )


def _angular_momentum_matrices(j: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projections = np.arange(j, -j - 1.0, -1.0)
    dimension = len(projections)
    jz = np.diag(projections).astype(np.complex128)
    raising = np.zeros((dimension, dimension), dtype=np.complex128)
    for column, projection in enumerate(projections):
        raised = projection + 1.0
        matches = np.flatnonzero(np.isclose(projections, raised))
        if matches.size:
            raising[matches[0], column] = sqrt(
                j * (j + 1.0) - projection * raised
            )
    lowering = raising.conj().T
    jx = 0.5 * (raising + lowering)
    jy = (raising - lowering) / (2.0j)
    return jx, jy, jz


def _cg_vector(
    spectator_j: int,
    terms: tuple[tuple[float, float, float], ...],
) -> np.ndarray:
    spectator_projections = tuple(np.arange(spectator_j, -spectator_j - 1, -1))
    active_projections = (0.5, -0.5)
    vector = np.zeros((2 * spectator_j + 1) * 2, dtype=np.complex128)
    for active_projection, spectator_projection, coefficient in terms:
        spectator_index = spectator_projections.index(spectator_projection)
        active_index = active_projections.index(active_projection)
        vector[2 * spectator_index + active_index] = coefficient
    return vector


def _j_half_coupling_residual(
    spectator_j: int,
    terms_by_projection: tuple[tuple[tuple[float, float, float], ...], ...],
) -> float:
    spectator = _angular_momentum_matrices(float(spectator_j))
    active = _angular_momentum_matrices(0.5)
    identity_s = np.eye(2 * spectator_j + 1, dtype=np.complex128)
    identity_a = np.eye(2, dtype=np.complex128)
    total = tuple(
        np.kron(axis_s, identity_a) + np.kron(identity_s, axis_a)
        for axis_s, axis_a in zip(spectator, active)
    )
    total_squared = sum(axis @ axis for axis in total)
    return max(
        float(np.linalg.norm(total_squared @ vector - 0.75 * vector))
        for vector in (
            _cg_vector(spectator_j, terms) for terms in terms_by_projection
        )
    )


def spectator_coupling_report(channel: str, target: str) -> Mapping[str, Any]:
    """Check the abstract spin/isospin-1/2 CG carrier used by the composition."""

    _validate_channel(channel)
    _validate_target(target)
    spectator_j = 0 if channel == SCALAR_SPECTATOR else 1
    spin_terms = tuple(
        tuple(
            (active / 2.0, spectator, coefficient)
            for active, spectator, coefficient in _spin_terms(channel, helicity)
        )
        for helicity in (-1, 1)
    )
    isospin_terms = []
    for current_target in TARGETS:
        isospin_terms.append(
            tuple(
                (
                    _FLAVOR_ISOSPIN_TWICE[flavor] / 2.0,
                    spectator,
                    coefficient,
                )
                for flavor, spectator, coefficient in _isospin_terms(
                    channel, current_target
                )
            )
        )
    return {
        "channel": channel,
        "target": target,
        "spin_J_half_residual": _j_half_coupling_residual(
            spectator_j, spin_terms
        ),
        "isospin_I_half_residual": _j_half_coupling_residual(
            spectator_j, tuple(isospin_terms)
        ),
        "active_helicity_carrier_assumption": (
            "C53_JZ_DOUBLET_TREATED_AS_ABSTRACT_J_HALF_FOR_SPECTATOR_RECOUPLING"
        ),
        "full_light_front_rotational_covariance_claimed": False,
    }


def _component_record(
    state: NucleonSpectatorState, item: NucleonCompositionComponent
) -> Mapping[str, Any]:
    ledger = state.component_ledger(item)
    return {
        "active_helicity_twice": item.active_helicity_twice,
        "active_flavor": item.active_flavor,
        "spectator_spin_projection": item.spectator_spin_projection,
        "spectator_isospin_projection": item.spectator_isospin_projection,
        "amplitude": (float(item.amplitude.real), float(item.amplitude.imag)),
        "probability": float(abs(item.amplitude) ** 2),
        "charge": str(ledger["charge"]),
        "baryon_number": str(ledger["baryon_number"]),
        "Jz_twice": ledger["Jz_twice"],
        "isospin_z_twice": ledger["isospin_z_twice"],
        "spectator_orbital_Lz": ledger["spectator_orbital_Lz"],
    }


def _state_overlap(
    left: NucleonSpectatorState, right: NucleonSpectatorState
) -> complex:
    left_terms = {
        (
            item.active_helicity_twice,
            item.active_flavor,
            item.spectator_spin_projection,
            item.spectator_isospin_projection,
            left.channel,
        ): item.amplitude
        for item in left.components
    }
    right_terms = {
        (
            item.active_helicity_twice,
            item.active_flavor,
            item.spectator_spin_projection,
            item.spectator_isospin_projection,
            right.channel,
        ): item.amplitude
        for item in right.components
    }
    return sum(
        np.conjugate(value) * right_terms.get(key, 0.0j)
        for key, value in left_terms.items()
    )


def _complex_matrix_record(matrix: np.ndarray) -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(
        tuple((float(value.real), float(value.imag)) for value in row)
        for row in np.asarray(matrix, dtype=np.complex128)
    )


def nucleon_spectator_doublet_record(
    doublet: NucleonSpectatorDoublet,
) -> Mapping[str, Any]:
    """Return a compact machine/handoff record with strict nonclaims."""

    coupling = spectator_coupling_report(doublet.channel, doublet.target)
    exchange = spectator_exchange_report(doublet.channel)
    active_density = doublet.unpolarized_active_pair_density()
    states = tuple(
        {
            "helicity_twice": state.helicity_twice,
            "normalization_residual": abs(state.norm - 1.0),
            "max_component_ledger_residual": state.max_component_ledger_residual,
            "energy_GeV2": state.energy_GeV2,
            "qg_weight": state.qg_weight,
            "active_pair_density": _complex_matrix_record(
                state.active_pair_density()
            ),
            "components": tuple(
                _component_record(state, item) for item in state.components
            ),
        }
        for state in doublet.states
    )
    return {
        "status": "M2_TYPED_NUCLEON_SPECTATOR_DOUBLET_READY",
        "target": doublet.target,
        "channel": doublet.channel,
        "states": states,
        "doublet_orthogonality_residual": float(
            abs(_state_overlap(doublet.state(-1), doublet.state(1)))
        ),
        "unpolarized_active_pair_density": _complex_matrix_record(active_density),
        "unpolarized_active_pair_reduction_residual": float(
            np.max(np.abs(active_density - 0.5 * np.eye(2)))
        ),
        "coupling_validation": coupling,
        "exchange_validation": exchange,
        "spectator_color": "EXACT_ANTITRIPLET_COMPLETED_TO_SINGLET",
        "spectator_relative_orbital": "EXPLORATORY_L0_S_WAVE_MEMBER",
        "spectator_hamiltonian": "IDENTITY_EXTENSION_ONLY_INTERACTIONS_UNAVAILABLE",
        "scalar_axial_mixture_selected": False,
        "proton_neutron_correlated_by": "STRONG_ISOSPIN_PARTNER_CONVENTION",
        "deuteron_H_D_constructed": False,
        "current_constructed": False,
        "claim_tier": doublet.claim_tier,
        "physical": doublet.physical,
    }


def spectator_composition_family_record(
    pair: C53MixedK9HelicityPair,
) -> Mapping[str, Any]:
    """Report all four target/channel members without selecting a mixture."""

    records = tuple(
        nucleon_spectator_doublet_record(
            build_nucleon_spectator_doublet(pair, target=target, channel=channel)
        )
        for channel in SPECTATOR_CHANNELS
        for target in TARGETS
    )
    return {
        "status": "M2_TYPED_SCALAR_AND_AXIAL_PROTON_NEUTRON_FAMILIES_READY",
        "members": records,
        "member_count": len(records),
        "scalar_axial_overlap": 0.0,
        "scalar_axial_mixture_selected": False,
        "active_pair_point_id": pair.assembly.point.point_id,
        "active_pair_claim_tier": pair.claim_tier,
        "spectator_dynamics_supplied": False,
        "physical_nucleon_selected": False,
        "deuteron_H_D_constructed": False,
        "current_constructed": False,
        "claim_tier": SPECTATOR_COMPOSITION_CLAIM_TIER,
        "physical": False,
    }


__all__ = [
    "SCALAR_SPECTATOR",
    "AXIAL_SPECTATOR",
    "SPECTATOR_CHANNELS",
    "TARGETS",
    "SPECTATOR_COMPOSITION_CLAIM_TIER",
    "NucleonCompositionComponent",
    "NucleonSpectatorState",
    "NucleonSpectatorDoublet",
    "spectator_spin_wavefunction",
    "spectator_flavor_wavefunction",
    "spectator_exchange_report",
    "spectator_coupling_report",
    "build_nucleon_spectator_doublet",
    "nucleon_spectator_doublet_record",
    "spectator_composition_family_record",
]

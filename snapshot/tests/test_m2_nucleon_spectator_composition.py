from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_c53_mixed_singlet_state import (
    C53_MIXED_K9_EXPLORATORY_BASELINE,
    build_c53_mixed_k9_hamiltonian,
    solve_c53_mixed_k9_helicity_pair,
)
from deuteron_wigner.quantum.m2_nucleon_spectator_composition import (
    AXIAL_SPECTATOR,
    SCALAR_SPECTATOR,
    build_nucleon_spectator_doublet,
    nucleon_spectator_doublet_record,
    spectator_composition_family_record,
    spectator_coupling_report,
    spectator_exchange_report,
    spectator_flavor_wavefunction,
    spectator_spin_wavefunction,
)


@lru_cache(maxsize=1)
def _pair():
    return solve_c53_mixed_k9_helicity_pair(
        build_c53_mixed_k9_hamiltonian(C53_MIXED_K9_EXPLORATORY_BASELINE)
    )


def test_spectator_pair_tensors_are_normalized_with_required_exchange_symmetry() -> None:
    for channel, projections, expected in (
        (SCALAR_SPECTATOR, (0,), -1.0),
        (AXIAL_SPECTATOR, (-1, 0, 1), 1.0),
    ):
        for projection in projections:
            assert np.linalg.norm(spectator_spin_wavefunction(channel, projection)) == pytest.approx(1.0)
            assert np.linalg.norm(spectator_flavor_wavefunction(channel, projection)) == pytest.approx(1.0)
        report = spectator_exchange_report(channel)
        assert report["spin_exchange_eigenvalue"] == expected
        assert report["flavor_exchange_eigenvalue"] == expected
        assert report["total_spectator_quark_exchange_eigenvalue"] == -1.0
        assert report["spin_exchange_residual"] < 1.0e-15
        assert report["flavor_exchange_residual"] < 1.0e-15


@pytest.mark.parametrize("target,charge,isospin_twice", [("PROTON", 1, 1), ("NEUTRON", 0, -1)])
def test_scalar_doublets_have_exact_componentwise_quantum_ledgers(
    target: str, charge: int, isospin_twice: int
) -> None:
    doublet = build_nucleon_spectator_doublet(
        _pair(), target=target, channel=SCALAR_SPECTATOR
    )
    for state in doublet.states:
        assert state.norm == pytest.approx(1.0)
        assert state.max_component_ledger_residual == 0.0
        assert len(state.components) == 1
        ledger = state.component_ledger(state.components[0])
        assert ledger["charge"] == charge
        assert ledger["baryon_number"] == 1
        assert ledger["Jz_twice"] == state.helicity_twice
        assert ledger["isospin_z_twice"] == isospin_twice


@pytest.mark.parametrize("target,charge,isospin_twice", [("PROTON", 1, 1), ("NEUTRON", 0, -1)])
def test_axial_doublets_have_exact_componentwise_quantum_ledgers(
    target: str, charge: int, isospin_twice: int
) -> None:
    doublet = build_nucleon_spectator_doublet(
        _pair(), target=target, channel=AXIAL_SPECTATOR
    )
    for state in doublet.states:
        assert state.norm == pytest.approx(1.0)
        assert state.max_component_ledger_residual == 0.0
        assert len(state.components) == 4
        for component in state.components:
            ledger = state.component_ledger(component)
            assert ledger["charge"] == charge
            assert ledger["baryon_number"] == 1
            assert ledger["Jz_twice"] == state.helicity_twice
            assert ledger["isospin_z_twice"] == isospin_twice


def test_scalar_and_axial_clebsch_gordan_tensors_are_exact_j_and_i_half() -> None:
    for channel in (SCALAR_SPECTATOR, AXIAL_SPECTATOR):
        report = spectator_coupling_report(channel, "PROTON")
        assert report["spin_J_half_residual"] < 1.0e-14
        assert report["isospin_I_half_residual"] < 1.0e-14
        assert report["full_light_front_rotational_covariance_claimed"] is False


def test_unpolarized_doublet_trace_returns_half_the_active_pair_projector() -> None:
    for target in ("PROTON", "NEUTRON"):
        for channel in (SCALAR_SPECTATOR, AXIAL_SPECTATOR):
            doublet = build_nucleon_spectator_doublet(
                _pair(), target=target, channel=channel
            )
            np.testing.assert_allclose(
                doublet.unpolarized_active_pair_density(),
                0.5 * np.eye(2),
                rtol=0.0,
                atol=2.0e-15,
            )
            report = nucleon_spectator_doublet_record(doublet)
            assert report["unpolarized_active_pair_reduction_residual"] < 2.0e-15
            assert report["doublet_orthogonality_residual"] == 0.0


def test_composition_inherits_active_energy_and_qg_weight_without_new_dynamics() -> None:
    for channel in (SCALAR_SPECTATOR, AXIAL_SPECTATOR):
        doublet = build_nucleon_spectator_doublet(
            _pair(), target="PROTON", channel=channel
        )
        for state in doublet.states:
            assert state.energy_GeV2 == pytest.approx(
                _pair().state(state.helicity_twice).energy_GeV2, abs=2.0e-14
            )
            assert state.qg_weight == pytest.approx(
                0.0819939207777390, abs=2.0e-13
            )
        record = nucleon_spectator_doublet_record(doublet)
        assert record["spectator_hamiltonian"].startswith("IDENTITY_EXTENSION")


def test_family_record_keeps_scalar_and_axial_members_separate_and_nonphysical() -> None:
    record = spectator_composition_family_record(_pair())

    assert record["member_count"] == 4
    assert {(item["target"], item["channel"]) for item in record["members"]} == {
        ("PROTON", SCALAR_SPECTATOR),
        ("NEUTRON", SCALAR_SPECTATOR),
        ("PROTON", AXIAL_SPECTATOR),
        ("NEUTRON", AXIAL_SPECTATOR),
    }
    assert record["scalar_axial_overlap"] == 0.0
    assert record["scalar_axial_mixture_selected"] is False
    assert record["spectator_dynamics_supplied"] is False
    assert record["physical_nucleon_selected"] is False
    assert record["deuteron_H_D_constructed"] is False
    assert record["physical"] is False


def test_channel_and_target_are_mandatory_and_invalid_labels_fail_closed() -> None:
    with pytest.raises(TypeError):
        build_nucleon_spectator_doublet(_pair(), target="PROTON")
    with pytest.raises(ValueError, match="target"):
        build_nucleon_spectator_doublet(
            _pair(), target="DEUTERON", channel=SCALAR_SPECTATOR
        )
    with pytest.raises(ValueError, match="channel"):
        build_nucleon_spectator_doublet(
            _pair(), target="PROTON", channel="DEFAULT_DIQUARK"
        )

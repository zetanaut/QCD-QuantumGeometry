from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_c53_mixed_singlet_state import (
    C53_MIXED_K9_EXPLORATORY_BASELINE,
    build_c53_mixed_k9_hamiltonian,
    solve_c53_mixed_k9_helicity_pair,
)
from deuteron_wigner.quantum.m2_nn_deuteron_state import (
    NN_AV18_NUCLEAR_R1,
    NNNuclearResolution,
    build_av18_wave_member,
    build_nn_light_front_wave_grid,
    compose_nn_deuteron_state,
    isospin_singlet_report,
    nn_deuteron_family_record,
    nn_deuteron_state_record,
)
from deuteron_wigner.quantum.m2_nucleon_spectator_composition import (
    AXIAL_SPECTATOR,
    SCALAR_SPECTATOR,
)


@lru_cache(maxsize=1)
def _pair():
    return solve_c53_mixed_k9_helicity_pair(
        build_c53_mixed_k9_hamiltonian(C53_MIXED_K9_EXPLORATORY_BASELINE)
    )


@lru_cache(maxsize=1)
def _wave_member():
    return build_av18_wave_member()


@lru_cache(maxsize=1)
def _grid():
    return build_nn_light_front_wave_grid(
        resolution=NN_AV18_NUCLEAR_R1,
        wave_member=_wave_member(),
        nucleon_mass_GeV=0.9389,
    )


def test_av18_member_has_source_hash_normalization_and_explicit_gev_adapter() -> None:
    member = _wave_member()

    assert member.member_id == "NN_WAVE_AV18_MOMENTUM_TABLE_GEV_ADAPTER_V1"
    assert member.source == "data/raw/av18/deut.wfk"
    assert member.source_sha256 == (
        "6bfe39d084e692c75a1bbbb921e13a7941c6ee5b7de7b71a24e010da9ab35523"
    )
    assert member.source_norm == pytest.approx(1.0, abs=2.0e-5)
    assert member.source_d_probability == pytest.approx(0.057599, abs=2.0e-6)
    assert member.source_momentum_max_GeV > NN_AV18_NUCLEAR_R1.k_max_GeV
    assert sum(abs(value) for value in member.radial_GeV(0.2)) > 0.0


def test_nuclear_resolution_is_separate_from_k9_and_grid_closes_dimensions() -> None:
    grid = _grid()

    assert grid.resolution.resolution_id != "K9"
    assert grid.node_count == 24 * 14 * 16
    assert grid.amplitudes.shape == (3, grid.node_count, 2, 2)
    assert np.all((grid.y > 0.0) & (grid.y < 1.0))
    assert np.all(grid.flat_measure_weights > 0.0)
    assert np.max(grid.raw_helicity_norms) - np.min(grid.raw_helicity_norms) < 1.0e-14
    assert np.mean(grid.raw_helicity_norms) == pytest.approx(1.0, abs=3.0e-4)


def test_one_common_normalization_gives_an_orthonormal_spin_one_triplet() -> None:
    grid = _grid()

    np.testing.assert_allclose(grid.helicity_gram(), np.eye(3), atol=1.0e-14)
    assert grid.common_normalization_factor == pytest.approx(
        0.9998736629799589, abs=2.0e-13
    )


def test_s_and_d_amplitudes_reconstruct_coherently_and_retain_local_interference() -> None:
    grid = _grid()
    s_norms, d_norms = grid.component_norms()

    np.testing.assert_allclose(
        grid.amplitudes, grid.s_amplitudes + grid.d_amplitudes, atol=3.0e-17
    )
    np.testing.assert_allclose(s_norms + d_norms, np.ones(3), atol=2.0e-14)
    np.testing.assert_allclose(
        d_norms, np.full(3, 0.05758717871968042), atol=2.0e-13
    )
    assert np.max(np.abs(grid.signed_sd_interference())) < 2.0e-15
    assert np.max(
        np.abs(2.0 * np.real(grid.s_amplitudes.conj() * grid.d_amplitudes))
    ) > 1.0e-4


def test_isospin_zero_factor_is_normalized_antisymmetric_and_has_B2_Q1() -> None:
    report = isospin_singlet_report()

    assert report["amplitudes"] == pytest.approx(
        (0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0)
    )
    assert report["normalization_residual"] < 3.0e-16
    assert report["exchange_eigenvalue"] == -1.0
    assert report["exchange_residual"] == 0.0
    assert report["I_total"] == 0
    assert report["I_squared_residual"] == 0.0
    assert report["component_charge"] == 1
    assert report["component_baryon_number"] == 2


@pytest.mark.parametrize("channel", [SCALAR_SPECTATOR, AXIAL_SPECTATOR])
def test_each_spectator_family_binds_to_a_lawful_exploratory_nn_state(channel: str) -> None:
    state = compose_nn_deuteron_state(_pair(), wave_grid=_grid(), channel=channel)
    record = nn_deuteron_state_record(state)

    assert state.hilbert_space_id.startswith("H_D^(NN_AV18_R1")
    assert ";K9)" in state.hilbert_space_id
    assert record["internal_resolution"] == "K9"
    assert record["nuclear_resolution"] == NN_AV18_NUCLEAR_R1.resolution_id
    assert record["resolution_labels_independent"] is True
    assert record["helicity_gram_residual"] < 1.0e-14
    assert record["charge"] == 1
    assert record["baryon_number"] == 2
    assert record["parity"] == 1
    assert record["J"] == 1
    assert record["nucleon_family"] == channel
    assert record["internal_qg_weight"] == pytest.approx(0.081993920777739)


def test_scalar_and_axial_nn_members_remain_separate_without_a_default_mixture() -> None:
    record = nn_deuteron_family_record(_pair(), wave_grid=_grid())

    assert record["member_count"] == 2
    assert {item["nucleon_family"] for item in record["members"]} == {
        SCALAR_SPECTATOR,
        AXIAL_SPECTATOR,
    }
    assert record["scalar_axial_mixture_selected"] is False
    assert record["nuclear_hamiltonian_constructed"] is False
    assert record["current_constructed"] is False
    assert record["physical"] is False


def test_state_stops_before_hamiltonian_mass_transfer_current_and_omitted_sectors() -> None:
    state = compose_nn_deuteron_state(
        _pair(), wave_grid=_grid(), channel=SCALAR_SPECTATOR
    )
    record = nn_deuteron_state_record(state)

    assert record["nuclear_hamiltonian_constructed"] is False
    assert record["deuteron_mass_eigenvalue_constructed"] is False
    assert record["initial_final_transfer_fibers_constructed"] is False
    assert record["current_constructed"] is False
    assert set(record["omitted_nuclear_sectors"]) == {
        "NNPI",
        "DELTADELTA",
        "SIX_QUARK",
        "HIDDEN_COLOR",
        "OTHER_NONNUCLEONIC",
    }
    assert record["physical"] is False


def test_nuclear_inputs_and_spectator_channel_are_mandatory() -> None:
    with pytest.raises(TypeError):
        compose_nn_deuteron_state(_pair(), wave_grid=_grid())
    with pytest.raises(ValueError, match="channel"):
        compose_nn_deuteron_state(
            _pair(), wave_grid=_grid(), channel="DEFAULT_MIXTURE"
        )
    with pytest.raises(ValueError, match="distinct from M2 K"):
        NNNuclearResolution("K9", 4, 4, 4, 1.0)
    with pytest.raises(ValueError, match="wave-function domain"):
        build_nn_light_front_wave_grid(
            resolution=NNNuclearResolution("NN_TOO_WIDE", 4, 4, 4, 4.0),
            wave_member=_wave_member(),
            nucleon_mass_GeV=0.9389,
        )

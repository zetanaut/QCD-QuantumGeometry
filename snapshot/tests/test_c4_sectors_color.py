"""C4 Benchmark E sector, selector, color, zero, and ledger tests."""

from dataclasses import replace

import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.gtmd import Species
from deuteron_wigner.pilot.active import PositiveXActiveSelector
from deuteron_wigner.pilot.c4_benchmarks import (
    exact_structural_zero, integrated_parent_ledger, parents_from_state,
    require_diagonal_core_label,
)
from deuteron_wigner.pilot.color import (
    GluonColorSinglet, SeaColorSinglet, reject_singlet_times_free_gluon,
)
from deuteron_wigner.pilot.sectors import (
    SectorSuperposition, gluon_state, sea_state,
)


@pytest.mark.parametrize("probability", (0.0, 1e-8, 0.2, 0.65))
def test_explicit_positive_x_sea_zero_and_linear_scaling(probability):
    state = sea_state(probability)
    parents = parents_from_state(state, Species.ANTIQUARK, flavor="d")
    assert all(parent.species == Species.ANTIQUARK for parent in parents)
    assert exact_structural_zero(state, Species.ANTIQUARK) == probability
    assert (parents == ()) is (probability == 0)
    state.require_proton_ledger()


@pytest.mark.parametrize("probability", (0.0, 1e-8, 0.25, 0.7))
def test_explicit_gluon_zero_and_linear_probability_density(probability):
    state = gluon_state(probability)
    parents = parents_from_state(state, Species.GLUON)
    expected_momentum = probability * 0.2
    assert exact_structural_zero(state, Species.GLUON) == expected_momentum
    assert (parents == ()) is (probability == 0)
    state.require_proton_ledger()


def test_sea_and_gluon_ledgers_preserve_proton_quantum_numbers():
    sea = sea_state(0.3).ledger()
    assert sea["net_flavor"] == {"d": pytest.approx(1), "u": pytest.approx(2)}
    assert sea["occupation"]["d"] == pytest.approx(1.6)
    assert sea["baryon_number"] == pytest.approx(1)
    assert sea["electric_charge"] == pytest.approx(1)
    assert sea["plus_momentum"] == pytest.approx(1)
    gluon = gluon_state(0.4).ledger()
    assert gluon["net_flavor"] == {"d": pytest.approx(1), "u": pytest.approx(2)}
    assert gluon["plus_momentum"] == pytest.approx(1)
    for state in (sea_state(0.3), gluon_state(0.4)):
        overlap = integrated_parent_ledger(state)
        state_ledger = state.ledger()
        assert overlap["net_flavor"] == pytest.approx(
            state_ledger["net_flavor"]
        )
        assert overlap["occupation"] == pytest.approx(
            state_ledger["occupation"]
        )
        assert overlap["plus_momentum"] == pytest.approx(
            state_ledger["plus_momentum"], abs=2e-15
        )


def test_sea_color_cluster_is_normalized_and_exact_singlet():
    color = SeaColorSinglet()
    assert color.basis_status == "CLUSTER_BASIS_NOT_FULLY_ANTISYMMETRIZED"
    assert color.norm() == pytest.approx(1, abs=1e-15)
    assert color.generator_residual() == 0
    color.validate()
    assert color.generator_residual(antiquark_sign=1) > 0.2


def test_qqqg_octet_adjoint_color_is_normalized_singlet():
    color = GluonColorSinglet()
    assert "rho-octet" in color.multiplicity_channel
    assert color.norm() == pytest.approx(1, abs=3e-16)
    assert color.generator_residual() < 6e-17
    assert color.generator_residual(include_adjoint=False) > 0.1
    color.validate()
    with pytest.raises(ArchitectureError, match="C4.GLUON_COLOR.FREE_GLUON"):
        reject_singlet_times_free_gluon()


def test_positive_x_active_selectors_and_wrong_species_failure():
    state = sea_state(0.2)
    config = state.sectors[1].configuration
    selector = PositiveXActiveSelector()
    antiquarks = selector.select(config, Species.ANTIQUARK)
    assert len(antiquarks) == 1
    assert antiquarks[0].x_domain == "POSITIVE_X"
    with pytest.raises(ArchitectureError, match="C4.ACTIVE.SPECIES"):
        selector.require_compatible(config, antiquarks[0], Species.GLUON)
    with pytest.raises(ArchitectureError, match="C4.ACTIVE.DUPLICATE"):
        selector.require_unique((antiquarks[0], antiquarks[0]))


def test_probability_and_todd_label_injections_fail_closed():
    base = sea_state(0.2)
    with pytest.raises(ArchitectureError, match="C4.STATE.PROBABILITY"):
        SectorSuperposition(
            "bad", (replace(base.sectors[0], probability=.5),
                    replace(base.sectors[1], probability=.4)),
        )
    with pytest.raises(ArchitectureError, match="C4.GLUON.TODD"):
        require_diagonal_core_label("physical f_type Sivers")


def test_internal_transverse_states_have_analytic_unit_norm():
    for state in (sea_state(.2), gluon_state(.3)):
        for sector in state.sectors:
            count = len(sector.configuration.constituents)
            assert sector.state.transverse_norm_oracle(count) == pytest.approx(
                1, abs=3e-16
            )

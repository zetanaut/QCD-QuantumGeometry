from __future__ import annotations

import numpy as np
from scipy import sparse

from deuteron_wigner.bridge.c401_c396_mass_directions import (
    D_DELTA_MU_G_SQ,
    D_MU_Q_SQ,
    resolution_record,
)
from deuteron_wigner.bridge.c411_c117_i2_finite_c43_adapter import ExploratoryC1171Parameters
from deuteron_wigner.quantum.operator_bundle import (
    build_exploratory_hamiltonian,
    bundle_record,
    exact_ground_state,
    exploratory_response_map,
)


def _bundle():
    dimension = resolution_record("K9")["direct_sum_dimension"]
    # Deliberately named test H0.  It is not a C144 fixture or a physics input.
    h0 = sparse.diags(np.linspace(0.0, 4.0, dimension), format="csr")
    return build_exploratory_hamiltonian(
        "K9",
        h0=h0,
        h0_source="TEST_DIAGONAL_H0_NOT_PHYSICAL",
        c396_coefficients={D_MU_Q_SQ: 0.20, D_DELTA_MU_G_SQ: 0.10},
        c117_parameters=ExploratoryC1171Parameters("K9", 0.50, 0.80),
        c117_coefficient=0.07,
    )


def test_bundle_uses_actual_c396_and_c117_shapes_without_c144_proxy():
    bundle = _bundle()
    assert bundle.dimension == 1350
    assert bundle.parameter_ids == (
        "c396:D_mu_q_sq",
        "c396:D_delta_mu_g_sq",
        "c117:I2_density_projector",
    )
    assert bundle.term("c117:I2_density_projector").matrix.shape == (1350, 1350)
    assert bundle.term("c117:I2_density_projector").source.startswith("C411")
    assert bundle_record(bundle)["h0_source"] == "TEST_DIAGONAL_H0_NOT_PHYSICAL"
    assert bundle_record(bundle)["physical"] is False


def test_bundle_preserves_all_three_k_local_dimensions():
    for resolution, dimension in (("K9", 1350), ("K11", 2706), ("K13", 4758)):
        bundle = build_exploratory_hamiltonian(
            resolution,
            h0=sparse.identity(dimension, format="csr"),
            h0_source="TEST_IDENTITY_H0_NOT_PHYSICAL",
            c396_coefficients={D_MU_Q_SQ: 0.0, D_DELTA_MU_G_SQ: 0.0},
            c117_parameters=ExploratoryC1171Parameters(resolution, 1.0, 1.0),
            c117_coefficient=0.0,
        )
        assert bundle.dimension == dimension
        assert bundle.matrix().shape == (dimension, dimension)


def test_sparse_and_matrix_free_bundle_actions_agree():
    bundle = _bundle()
    rng = np.random.default_rng(410)
    vector = rng.normal(size=bundle.dimension) + 1j * rng.normal(size=bundle.dimension)
    assert np.max(np.abs((bundle.matrix() @ vector) - bundle.apply(vector))) < 1.0e-12


def test_exact_state_and_response_map_remain_diagnostic():
    bundle = _bundle()
    state = exact_ground_state(bundle)
    assert state.residual_norm < 1.0e-7
    assert np.isclose(state.q_weight + state.qg_weight, 1.0)
    response = exploratory_response_map(
        bundle,
        {
            "identity": sparse.identity(bundle.dimension, format="csr"),
            "quark_mass_direction": bundle.term("c396:D_mu_q_sq").matrix,
        },
        finite_difference_step=1.0e-4,
    )
    assert response["claim_tier"] == "EXPLORATORY"
    assert response["physical"] is False
    assert response["rank_is_physical"] is False
    assert response["current_response"] == "NOT_PROVIDED"
    assert response["energy_hf_fd_max_abs_residual"] < 1.0e-5
    assert len(response["singular_values"]) == 3

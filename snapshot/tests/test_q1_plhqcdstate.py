"""Q1/PLHQCDSTATE acceptance tests."""

from functools import lru_cache

import numpy as np

from deuteron_wigner.bridge import plhqcd0 as q0
from deuteron_wigner.bridge import plhqcdstate as q1


@lru_cache(maxsize=1)
def report():
    return q1.build_q1_report()


def test_q0_boundary_fixture_order_and_k9_encoding():
    contract = q1.q0_contract()
    assert contract["direct_q0_import_only"] is True
    assert contract["later_authority_imports"] == 0
    assert contract["fixtures"] == q1.FIXTURE_SEQUENCE
    basis = q0.basis_metadata("K9")
    assert basis.compact_dimension == 1350
    assert basis.qubits == 11
    assert q1.ENCODING == "COMPACT_INDEX_DIRECT_ORDER_V1"


def test_adapt_pool_and_ordinary_decomposition_preserve_padding():
    selection = q1.select_adapt_layers("K9", "FIXTURE-INTERACTING-A")
    assert selection["hardware_efficient_fallback"] is False
    assert [(x.edge_id, x.kind) for x in selection["selected"]] == [("EDGE-00-0-6", "real")]
    for edge in q1.authenticated_hamiltonian_edges("K9", "FIXTURE-INTERACTING-A"):
        for kind in ("real", "imaginary"):
            gates = q1.ordinary_two_level_rotation("K9", edge.left, edge.right, kind, 0.123)
            assert gates
            assert all(type(gate).__name__ != "QubitUnitary" for gate in gates)
    state = q1.trainable_state("K9", "FIXTURE-INTERACTING-A", selection["selected"], [0.123])
    np.testing.assert_allclose(np.linalg.norm(state), 1.0, rtol=0.0, atol=1.0e-10)
    leakage = q0.sector_leakage_diagnostics("K9", state)["padded_leakage"]
    assert leakage <= q1.TOLERANCES["padding"]


def test_q1_positive_acceptance_and_observable_vector():
    value = report()
    assert value["status"] == "Q1_PLHQCDSTATE_COMPLETE"
    assert value["positive_gate"] is True
    assert value["next"] == "Q2/PLHQCDOBS"
    assert value["hardware_execution"] is False
    assert value["physical_parameter_selected"] is False
    assert value["physical_state_created"] is False
    assert value["production_object_created"] is False
    assert value["stateprep_rows"]["FIXTURE-INTERACTING-A"]["validation_only"] is True
    assert all(row["oracle_residual"] <= q1.TOLERANCES["observable"] for row in value["stateprep_rows"].values())
    for row in value["trainable_rows"].values():
        assert row["energy_residual"] <= q1.TOLERANCES["energy"]
        assert row["eigenstate_residual_norm"] <= q1.TOLERANCES["residual_norm"]
        assert row["P_padding"] <= q1.TOLERANCES["padding"]
        assert row["observable_residual"] <= q1.TOLERANCES["observable"]
        assert row["padding_preserving_by_construction"] is True


def test_q1_holdouts_derivatives_and_responses():
    value = report()
    assert len(value["holdout_checks"]) == 40
    assert all(value["holdout_checks"])
    assert all(row["maximum_residual"] <= q1.TOLERANCES["derivative"] for row in value["derivative_checks"].values())
    assert np.isfinite(value["continuation"]["null_shift"]["energy_delta_B_minus_A"])
    assert np.isfinite(value["continuation"]["mass_sign"]["energy_delta_MASS_minus_A"])
    for resolution in q1.HOLDOUT_RESOLUTIONS:
        rows = [value["holdout_rows"][f"{resolution}:{fixture}"] for fixture in q1.FIXTURE_SEQUENCE]
        assert {row["compact_dimension"] for row in rows} == ({2706} if resolution == "K11" else {4758})

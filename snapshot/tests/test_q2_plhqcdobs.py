"""Q2/PLHQCDOBS acceptance and boundary tests."""

import numpy as np
import pytest

from deuteron_wigner.quantum import plhqcdobs as q2


def test_q2_authority_and_q0_q1_preservation():
    ancestry = q2.q0_ancestry_report()
    preservation = q2.q2_preservation_report()
    freeze = q2.q2_input_freeze()
    assert ancestry["positive_gate"] is True
    assert preservation["positive_gate"] is True
    assert freeze["q1_status"] == "Q1_PLHQCDSTATE_COMPLETE"
    assert freeze["q1_encoding"] == q2.ENCODING
    assert freeze["q1_basis_order"] == q2.BASIS_ORDER
    assert freeze["hardware_execution"] is False


def test_registry_is_source_structured_and_padded():
    records = q2.observable_registry()
    assert len(records) == 471
    assert all(record["qubit_space_dimension"] >= record["physical_space_dimension"] for record in records)
    assert all(record["measurement_compiler_route"] in {"source_structured", "ACTION_ONLY", "DERIVED_MATRIX_FREE"} for record in records)
    assert all(record["claim_tier"] == "CONDITIONAL_HAMILTONIAN_DIAGNOSTIC" for record in records)
    assert any(record["operator_family"] == "Q1_ADAPT_GRADIENT_OBSERVABLE" for record in records)


def test_compiler_and_measurement_leakage():
    record = next(row for row in q2.observable_registry() if row["operator_family"] == "TOTAL_HAMILTONIAN")
    terms = q2.measurement_term_manifest(record["observable_id"])
    groups = q2.measurement_group_manifest(record["observable_id"])
    compiled = q2.compile_observable_measurement(record["observable_id"])
    assert terms["term_count"] > 0
    assert groups["group_count"] > 0
    assert compiled["ordinary_gate_only"] is True
    assert compiled["production_qubitunitary_count"] == 0
    leakage = q2.measurement_leakage_validation()
    assert leakage["all_pass"] is True
    assert leakage["max_padding_leakage"] == 0.0


@pytest.mark.parametrize("route", q2.STATE_ROUTES)
@pytest.mark.parametrize("fixture", q2.FIXTURES)
def test_primary_expectation_routes(route, fixture):
    observable = f"K2:K9:{fixture}:total"
    state = f"{route}:{fixture}"
    sparse_value = q2.evaluate_sparse_expectation(state, observable)
    qnode_value = q2.evaluate_qnode_expectation(state, observable)
    compiled_value = q2.evaluate_compiled_expectation(state, observable)
    assert abs(sparse_value - qnode_value) <= q2.TOLERANCES["route"]
    assert abs(sparse_value - compiled_value) <= q2.TOLERANCES["compiler"]


def test_derivative_and_state_diagnostics():
    hf = q2.hellmann_feynman_report("FIXTURE-INTERACTING-A", "phi_mass", "EXACT_STATEPREP_ORACLE_STATE")
    residual = q2.eigenstate_residual_report("FIXTURE-INTERACTING-A", "EXACT_STATEPREP_ORACLE_STATE")
    overlap = q2.source_overlap_report("FIXTURE-INTERACTING-A", "Q1_VARIATIONAL_STATE")
    assert hf["qnode_residual"] <= q2.TOLERANCES["hf"]
    assert hf["compiled_residual"] <= q2.TOLERANCES["hf"]
    assert residual["residual_norm_matrix_free"] <= q2.TOLERANCES["residual"]
    assert 0.0 <= overlap["exact_state_overlap"] <= 1.0 + 3.0e-12


def test_variance_shot_plan_and_mutation_boundary():
    variance = q2.variance_manifest("FIXTURE-FREE", "EXACT_STATEPREP_ORACLE_STATE", "K2:K9:FIXTURE-FREE:total")
    assert variance["rows"]
    plan = q2.build_shot_plan({"shot_budget": 1000, "groups": variance["rows"][:3]})
    assert sum(plan["allocation"]) == 1000
    mutations = q2.focused_live_mutations()
    assert mutations["positive_gate"] is True
    assert mutations["pass_count"] == 384


def test_completion_certificate_and_forbidden_defaults():
    cert = q2.q2_completeness_certificate()
    assert cert["ancestry"] is True
    assert cert["preservation"] is True
    assert cert["no_dense_pauli"] is True
    assert cert["no_production_qubitunitary"] is True
    with pytest.raises(ValueError, match="explicit"):
        q2.build_shot_plan({"groups": ({"group_id": "g", "variance": 1.0},)})

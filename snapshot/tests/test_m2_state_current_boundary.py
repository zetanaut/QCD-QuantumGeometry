from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from deuteron_wigner.bridge.basis1 import core as c47
from deuteron_wigner.bridge.c401_c396_mass_directions import resolution_record
from deuteron_wigner.quantum.m2_k9_exploratory_state import (
    K9_EXPLORATORY_BASELINE,
    build_parameter_explicit_k9_hamiltonian,
    solve_low_k9_eigenspace,
)
from deuteron_wigner.quantum.m2_state_current_boundary import (
    M2StateCurrentInterfaceError,
    m2_k9_state_current_interface_audit,
    require_lawful_m2_k9_current_interface,
)


@lru_cache(maxsize=1)
def _study():
    return solve_low_k9_eigenspace(
        build_parameter_explicit_k9_hamiltonian(K9_EXPLORATORY_BASELINE)
    )


def test_m2_k9_audit_proves_the_c47_fundamental_only_color_obstruction() -> None:
    audit = m2_k9_state_current_interface_audit(_study())
    support = audit["domain"]["support"]
    color = audit["representation_obstruction"]["color_decomposition"]

    assert audit["claim_tier"] == "EXPLORATORY"
    assert audit["physical"] is False
    assert audit["state_selection"] == "INVARIANT_PROJECTOR_ONLY_NO_EIGENVECTOR_SELECTED"
    assert audit["domain"]["state_object"] == "Ran(P_K9)"
    assert audit["domain"]["state_dimension"] == 6
    assert support["q_block_identity_residual"] < 1.0e-10
    assert support["qg_row_frobenius_norm"] < 1.0e-10
    assert support["open_quark_Jz"] == (-0.5, 0.5)
    assert support["open_triplet_colors"] == (0, 1, 2)
    assert support["color_singlet_selected"] is False
    assert audit["required_codomain"]["lps_input"]["shape"] == (4, 3, 3)
    assert audit["required_codomain"]["initial_final_status"].startswith("M2 supplies one invariant subspace")
    assert audit["required_codomain"]["light_front_input"]["amplitude_order"] == (
        "I++",
        "I+0",
        "I+-",
        "I00",
    )
    assert audit["map_classification"]["existing_map"] == "NONE_SOURCE_QUALIFIED"
    assert audit["map_classification"]["primary_obstruction"] == "C47_COLOR_SINGLET_INTERTWINER_ZERO"
    assert audit["map_classification"]["color_singlet_map_into_H_M2,K9"] == (
        "ZERO_BY_SU3_REPRESENTATION_CONTENT"
    )
    equations = audit["required_defining_equations"]
    assert "H_D,K (not H_M2,K9)" in equations["color_singlet_target_composition"]
    assert equations["finite_K_current_after_enlargement"] == "J_D,K^mu: H_D,K -> H_D,K"
    assert "colored-subsystem diagnostic, never a deuteron target current" in equations[
        "colored_subsystem_diagnostic"
    ]
    assert "rules out an isomorphism only" in audit["map_classification"]["dimension_note"]
    assert color["q_fundamental_triplet_count"] == 2
    assert color["qg_fundamental_triplet_count"] == 448
    assert color["total_fundamental_triplet_count"] == 450
    assert color["full_space_decomposition"] == "H_M2,K9 = (450) * 3"
    assert color["direct_sum_dimension"] == 1350
    assert color["qg_triplet_isometry_shape"] == (24, 3)
    assert color["qg_triplet_isometry_residual"] < 1.0e-12
    assert color["qg_triplet_image_residual"] < 1.0e-12
    assert color["qg_triplet_intertwining_residual"] < 1.0e-12
    assert color["fundamental_casimir"] == "4/3"
    assert color["fundamental_casimir_residual"] < 1.0e-12
    assert color["color_singlet_subrepresentation_present"] is False
    assert color["singlet_intertwiner_space"] == "Hom_SU(3)(1, H_M2,K9) = {0}"
    assert color["M2_basis_map_preserves_C47_color_modules"] is True
    assert audit["representation_obstruction"]["missing_target_helicity_zero"] is True
    assert "color-singlet projection" in audit["representation_obstruction"]["missing_information_not_zero"]
    assert audit["finite_K_coordinate_correspondence"]["C405_axis_matches"].startswith("same K9")
    assert audit["separate_historical_current"]["tower_dimensions"] == (4, 7, 10)
    assert audit["separate_historical_current"]["usable_for_M2"] is False


def test_m2_current_obstruction_preserves_c405_c114_missing_blocks_as_not_zero() -> None:
    audit = m2_k9_state_current_interface_audit(_study())
    current = audit["current_operator_obstruction"]

    assert current["C405_same_direct_sum_axis"]["dimension_matches_M2"] is True
    assert current["C405_same_direct_sum_axis"]["q_diagonal_block"] == (
        "UNAVAILABLE_NOT_ZERO_FOR_C117_I2"
    )
    assert current["C405_complete_C117_action"] is False
    assert current["C114_complete_instantaneous_current"] is False
    assert set(current["C114_missing_products"]) == {
        "J_qJ_q",
        "J_qJ_g",
        "J_gJ_q",
        "J_gJ_g",
    }
    assert audit["adapter_use"] == "NOT_STARTED_NO_LAWFUL_INPUT_OBJECT"
    assert audit["current_response"] == "NOT_EVALUATED_UNAVAILABLE_NOT_ZERO"
    assert "enlarged many-body/hadronic" in audit["next_executable_construction"]

    with pytest.raises(M2StateCurrentInterfaceError, match=r"Hom_SU\(3\)\(1, H_M2,K9\) = \{0\}"):
        require_lawful_m2_k9_current_interface(_study())


def test_m2_current_boundary_is_invariant_under_degenerate_basis_rotation() -> None:
    study = _study()
    generator = np.random.default_rng(9004)
    raw = generator.normal(size=(study.multiplicity, study.multiplicity)) + 1j * generator.normal(
        size=(study.multiplicity, study.multiplicity)
    )
    unitary, _ = np.linalg.qr(raw)
    rotated = replace(study, basis=np.asarray(study.basis @ unitary, dtype=np.complex128))

    np.testing.assert_allclose(rotated.projector, study.projector, rtol=0.0, atol=2.0e-14)
    original = m2_k9_state_current_interface_audit(study)
    transformed = m2_k9_state_current_interface_audit(rotated)
    assert transformed["state_selection"] == original["state_selection"]
    assert transformed["domain"]["state_dimension"] == original["domain"]["state_dimension"]
    assert transformed["domain"]["support"]["q_labels"] == original["domain"]["support"]["q_labels"]
    assert transformed["representation_obstruction"] == original["representation_obstruction"]
    assert transformed["current_operator_obstruction"] == original["current_operator_obstruction"]


def test_m2_current_boundary_rejects_qg_contamination_with_unchanged_shape_and_multiplicity() -> None:
    study = _study()
    q_dimension = int(resolution_record("K9")["q_dimension"])
    epsilon = 1.0e-6
    contaminated_basis = np.array(study.basis, copy=True)
    contaminated_basis[:, 0] *= np.sqrt(1.0 - epsilon**2)
    contaminated_basis[q_dimension, 0] = epsilon
    contaminated = replace(study, basis=contaminated_basis)

    assert contaminated.basis.shape == study.basis.shape
    assert contaminated.multiplicity == study.multiplicity
    with pytest.raises(RuntimeError, match="M2 K9 projector qg leakage norm"):
        m2_k9_state_current_interface_audit(contaminated)


def test_m2_current_boundary_rejects_a_defective_c47_triplet_isometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = c47.triplet_isometry

    def defective_triplet_isometry() -> np.ndarray:
        result = np.array(original(), copy=True)
        result[0, 0] += 1.0e-3
        return result

    monkeypatch.setattr(c47, "triplet_isometry", defective_triplet_isometry)
    with pytest.raises(RuntimeError, match="C47 triplet isometry residual"):
        m2_k9_state_current_interface_audit(_study())

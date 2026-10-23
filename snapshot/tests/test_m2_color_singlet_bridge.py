from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import eigsh

from deuteron_wigner.bridge.c401_c396_mass_directions import (
    DIRECTIONS,
    coordinate_operator_csr,
    resolution_record,
)
from deuteron_wigner.bridge.c411_c117_i2_finite_c43_adapter import (
    ExploratoryC1171Parameters,
    exploratory_c117_1_csr,
)
from deuteron_wigner.microscopic.h0 import build_exploratory_k_local_h0
from deuteron_wigner.bridge.vertex3.core import generated_adjoint_and_block
from deuteron_wigner.quantum.m2_color_singlet_bridge import (
    M2ColorBridgeError,
    c47_to_m2_target_embedding,
    color_completion_algebra_record,
    color_completion_record,
    color_singlet_embedding,
    complete_m2_operator_to_singlet,
    map_c47_operator_to_m2_target,
    m2_color_modules,
)
from deuteron_wigner.quantum.m2_k9_exploratory_state import (
    K9_EXPLORATORY_BASELINE,
    build_parameter_explicit_k9_hamiltonian,
)


EXPECTED = {"K9": (1350, 450), "K11": (2706, 902), "K13": (4758, 1586)}
EXPECTED_C53_NNZ = {"K9": (624, 208), "K11": (1020, 340), "K13": (1512, 504)}


@pytest.mark.parametrize("resolution", tuple(EXPECTED))
def test_m2_color_modules_and_singlet_embedding_close_every_resolution(
    resolution: str,
) -> None:
    source_dimension, multiplicity = EXPECTED[resolution]
    modules = m2_color_modules(resolution)
    embedding = color_singlet_embedding(resolution)
    record = color_completion_record(resolution)

    assert modules.source_dimension == source_dimension
    assert modules.multiplicity == multiplicity
    assert all(len(row) == 3 for row in modules.source_indices_by_module)
    assert embedding.shape == (3 * source_dimension, multiplicity)
    assert embedding.nnz == source_dimension
    np.testing.assert_allclose(
        (embedding.getH() @ embedding).toarray(),
        np.eye(multiplicity),
        rtol=0.0,
        atol=3.0e-16,
    )
    assert record["status"] == "M2_COLOR_SINGLET_NUCLEON_COMPLETION_READY"
    assert record["physical"] is False
    assert "nuclear S/D composition" in record["not_yet_H_D"]


def test_color_completion_matches_c7_qqq_and_active_emitter_qqqg_singlets() -> None:
    record = color_completion_algebra_record()
    qqq_overlap = record["C7_qqq_overlap"]
    qqqg_coordinates = np.asarray(record["C7_qqqg_coordinates"])

    assert record["claim_tier"] == "EXACT_REPRESENTATION_BRIDGE_NONPHYSICAL"
    assert record["singlet_generator_residual"] < 1.0e-14
    assert record["diquark_isometry_residual"] < 1.0e-14
    assert record["diquark_intertwining_residual"] < 1.0e-14
    assert abs(abs(qqq_overlap) - 1.0) < 1.0e-14
    assert record["C7_qqq_projection_residual"] < 1.0e-14
    assert record["C7_qqqg_projection_residual"] < 2.0e-14
    assert record["C7_active_emitter_coordinate_residual"] < 2.0e-14
    np.testing.assert_allclose(
        qqqg_coordinates,
        qqq_overlap * np.asarray((np.sqrt(3.0) / 2.0, -0.5)),
        rtol=0.0,
        atol=3.0e-14,
    )
    assert record["C7_qqqg_orthogonal_singlet_status"] == "UNMAPPED_NOT_ZERO"
    assert record["deuteron_H_D_constructed"] is False


@pytest.mark.parametrize("resolution", tuple(EXPECTED))
def test_all_present_m2_operators_descend_to_the_color_singlet(
    resolution: str,
) -> None:
    supply = build_exploratory_k_local_h0(resolution)
    c53_source = generated_adjoint_and_block(
        str(resolution_record(resolution)["full_resolution_id"])
    )["block"]
    matrices = {
        "H0": supply.target_operator,
        "C53_CANONICAL_VERTEX": map_c47_operator_to_m2_target(
            resolution, c53_source
        ),
        **{
            f"c396:{direction}": coordinate_operator_csr(resolution, direction)
            for direction in DIRECTIONS
        },
        "c117:I2_density_projector": map_c47_operator_to_m2_target(
            resolution,
            exploratory_c117_1_csr(
                ExploratoryC1171Parameters(resolution, 1.0, 1.0)
            ),
        ),
    }
    _, expected_singlet_dimension = EXPECTED[resolution]
    for operator_id, matrix in matrices.items():
        completed = complete_m2_operator_to_singlet(
            resolution, matrix, operator_id=operator_id
        )
        assert completed.singlet_dimension == expected_singlet_dimension
        assert completed.cross_color_residual == 0.0
        assert completed.equal_color_block_residual < 2.0e-14
        assert completed.reconstruction_residual < 2.0e-14
        assert completed.singlet_projection_residual < 2.0e-14
        assert completed.physical is False
        if operator_id == "C53_CANONICAL_VERTEX":
            assert (completed.source_nnz, completed.reduced_nnz) == EXPECTED_C53_NNZ[
                resolution
            ]


def test_c53_and_c117_consume_exact_c47_to_m2_permutation_before_descent() -> None:
    resolution = "K9"
    source = generated_adjoint_and_block(
        str(resolution_record(resolution)["full_resolution_id"])
    )["block"]
    embedding = c47_to_m2_target_embedding(resolution)
    mapped = map_c47_operator_to_m2_target(resolution, source)
    recovered = (embedding.getH() @ mapped @ embedding).tocsr()
    assert (recovered - source).nnz == 0
    assert (mapped - source).nnz > 0

    labels = m2_color_modules(resolution).source_labels
    jz = np.asarray(
        [
            label[1] / 2.0
            if label[0] == "q"
            else label[4] / 2.0 + label[5] + label[3]
            for label in labels
        ]
    )
    generator = sparse.diags(jz, format="csr", dtype=np.complex128)
    mapped_defect = (generator @ mapped - mapped @ generator).tocsr()
    raw_defect = (generator @ source - source @ generator).tocsr()
    mapped_residual = (
        float(np.max(np.abs(mapped_defect.data))) if mapped_defect.nnz else 0.0
    )
    assert mapped_residual == 0.0
    assert np.max(np.abs(raw_defect.data)) > 1.0e-6

    c117_source = exploratory_c117_1_csr(
        ExploratoryC1171Parameters(resolution, 1.0, 1.0)
    )
    c117_mapped = map_c47_operator_to_m2_target(resolution, c117_source)
    c117_mapped_defect = (
        generator @ c117_mapped - c117_mapped @ generator
    ).tocsr()
    c117_raw_defect = (generator @ c117_source - c117_source @ generator).tocsr()
    assert c117_mapped_defect.nnz == 0
    assert np.max(np.abs(c117_raw_defect.data)) > 1.0e-6


def test_k9_total_operator_loses_only_the_open_color_degeneracy() -> None:
    assembly = build_parameter_explicit_k9_hamiltonian(K9_EXPLORATORY_BASELINE)
    full = assembly.bundle.matrix()
    completed = complete_m2_operator_to_singlet(
        "K9", full, operator_id="M2_K9_EXPLORATORY_BASELINE_TOTAL"
    )
    full_values = np.sort(eigsh(full, k=8, which="SA", return_eigenvectors=False))
    singlet_values = np.sort(
        eigsh(
            completed.reduced_operator,
            k=5,
            which="SA",
            return_eigenvectors=False,
        )
    )

    # The accepted sixfold M2 ground cluster is two helicities times three
    # open colors.  Exact singlet completion removes the color copy only.
    assert np.count_nonzero(abs(full_values - full_values[0]) < 1.0e-10) == 6
    assert np.count_nonzero(abs(singlet_values - singlet_values[0]) < 1.0e-10) == 2
    np.testing.assert_allclose(singlet_values[0], full_values[0], rtol=0.0, atol=2.0e-12)


def test_color_breaking_and_color_mixing_operators_fail_closed() -> None:
    modules = m2_color_modules("K9")
    diagonal_breaking = sparse.eye(modules.source_dimension, format="lil")
    diagonal_breaking[modules.source_indices_by_module[0][1], modules.source_indices_by_module[0][1]] = 2.0
    with pytest.raises(M2ColorBridgeError, match="not fundamental-color equivariant"):
        complete_m2_operator_to_singlet(
            "K9", diagonal_breaking.tocsr(), operator_id="BROKEN_COLOR_DIAGONAL"
        )

    color_mixing = sparse.eye(modules.source_dimension, format="lil")
    row = modules.source_indices_by_module[0]
    color_mixing[row[0], row[1]] = 0.25
    with pytest.raises(M2ColorBridgeError, match="not fundamental-color equivariant"):
        complete_m2_operator_to_singlet(
            "K9", color_mixing.tocsr(), operator_id="BROKEN_COLOR_MIXING"
        )

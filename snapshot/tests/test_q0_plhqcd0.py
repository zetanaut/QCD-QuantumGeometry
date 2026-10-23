"""Q0/PLHQCD0 acceptance tests."""

import numpy as np
import pytest

from deuteron_wigner.bridge import plhqcd0 as q0


def _probe(resolution: str, *, seed: int = 11) -> np.ndarray:
    basis = q0.basis_metadata(resolution)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=basis.compact_dimension) + 1j * rng.normal(size=basis.compact_dimension)
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.complex128)


def test_public_authority_import_is_positive_and_nonphysical():
    report = q0.verify_public_authorities()
    assert report["positive_gate"] is True
    assert report["loaded"] == ("C131", "C142", "C144", "C149", "C150")
    assert report["physical_parameter_selected"] is False
    assert report["C150_scheme_selected"] is False
    assert report["forbidden_layers_constructed"] == 0


@pytest.mark.parametrize("resolution", q0.RESOLUTIONS)
def test_compact_basis_encoding_and_physical_projector(resolution):
    basis = q0.basis_metadata(resolution)
    indices = (0, basis.q_dimension - 1, basis.q_dimension, basis.compact_dimension - 1)
    for index in indices:
        bitstring = q0.encode_index(resolution, index)
        assert len(bitstring) == basis.qubits
        assert q0.decode_bitstring(resolution, bitstring) == index
    if basis.padded_dimension > basis.compact_dimension:
        with pytest.raises(ValueError, match="padded leakage"):
            q0.decode_bitstring(resolution, format(basis.padded_dimension - 1, f"0{basis.qubits}b"))
    projector = q0.physical_subspace_projector(resolution)
    assert np.array_equal((projector @ projector).toarray(), projector.toarray())
    sectors = q0.sector_projector(resolution, "q") + q0.sector_projector(resolution, "qg")
    assert np.array_equal(sectors.toarray(), projector.toarray())


@pytest.mark.parametrize("resolution", q0.RESOLUTIONS)
def test_certified_sparse_matrix_free_and_encoded_parity(resolution):
    basis = q0.basis_metadata(resolution)
    hamiltonian = q0.certified_padded_hamiltonian(resolution, fixture_id="FIXTURE-INTERACTING-A")
    vector = _probe(resolution)
    padded = q0.compact_to_padded_state(resolution, vector)
    assert hamiltonian.certificate["physical_support"] is True
    assert hamiltonian.certificate["physical_parameter_selected"] is False
    assert hamiltonian.certificate["C150_scheme_selected"] is False
    assert hamiltonian.compact.shape == (basis.compact_dimension, basis.compact_dimension)
    assert hamiltonian.padded.shape == (basis.padded_dimension, basis.padded_dimension)
    assert hamiltonian.compact.nnz == hamiltonian.padded.nnz
    assert np.allclose(q0.sparse_action(resolution, vector, fixture_id="FIXTURE-INTERACTING-A"),
                       q0.matrix_free_action(resolution, vector, fixture_id="FIXTURE-INTERACTING-A"),
                       rtol=1e-13, atol=2e-12)
    encoded = q0.encoded_action(resolution, padded, fixture_id="FIXTURE-INTERACTING-A")
    assert np.allclose(encoded[: basis.compact_dimension], q0.sparse_action(resolution, vector, fixture_id="FIXTURE-INTERACTING-A"),
                       rtol=0.0, atol=0.0)
    assert np.count_nonzero(encoded[basis.compact_dimension :]) == 0


@pytest.mark.parametrize("resolution", q0.RESOLUTIONS)
def test_lightning_sparse_hamiltonian_qnode_parity(resolution):
    basis = q0.basis_metadata(resolution)
    state = np.zeros(basis.compact_dimension, dtype=np.complex128)
    state[0] = 1.0
    classical = q0.expectation(resolution, state, fixture_id="FIXTURE-INTERACTING-A")
    qnode = q0.qnode_expectation(resolution, state, fixture_id="FIXTURE-INTERACTING-A")
    assert np.allclose(qnode, classical, rtol=0.0, atol=1e-12)


def test_sector_and_leakage_diagnostics():
    basis = q0.basis_metadata("K9")
    state = np.zeros(basis.padded_dimension, dtype=np.complex128)
    state[0] = 0.5
    state[basis.q_dimension] = 0.5
    state[basis.compact_dimension] = np.sqrt(0.5)
    report = q0.sector_leakage_diagnostics("K9", state)
    assert report["q_weight"] == pytest.approx(0.25)
    assert report["qg_weight"] == pytest.approx(0.25)
    assert report["padded_leakage"] == pytest.approx(0.5)
    assert report["physical_weight"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="padded leakage"):
        q0.padded_to_compact_state("K9", state)


@pytest.mark.parametrize("resolution", q0.RESOLUTIONS)
def test_exact_derivative_parity(resolution):
    vector = _probe(resolution, seed=19)
    for direction in ("phi_mass", "phi_coupling", *[f"eta_{i}" for i in range(9)]):
        report = q0.derivative_parity(resolution, direction, vector)
        assert report["exact"] is True
        assert report["max_abs_residual"] == 0.0
    derivative_qnode = q0.derivative_qnode_expectation(resolution, "phi_coupling", vector)
    derivative_classical = np.vdot(vector, q0.derivative_matrix_free_action(resolution, "phi_coupling", vector))
    assert np.allclose(derivative_qnode, derivative_classical, rtol=0.0, atol=1e-12)


def test_resource_and_pauli_boundaries():
    for resolution in q0.RESOLUTIONS:
        report = q0.resource_report(resolution)
        assert report["sparse_production_path"] is True
        assert report["generic_full_pauli_decomposition"].endswith("FORBIDDEN_IN_Q0")
        assert report["vqe_or_ansatz"] == "NOT_STARTED_IN_Q0"
        assert report["C150_Z_q"] == "NOT_CONSUMED"
    with pytest.raises(NotImplementedError, match="FORBIDDEN_IN_Q0"):
        q0.pauli_decomposition("K9")

"""Independent compact-SciPy vs measured-PennyLane scientific regressions."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

qml = pytest.importorskip("pennylane")
pytest.importorskip("pennylane_lightning")

from deuteron_wigner.quantum.m2_hamiltonian_gauged_current import (
    build_hamiltonian_gauged_baryon_current,
    build_m2_baryon_density_operator,
)
from deuteron_wigner.quantum.m2_pennylane_contract import build_m2_pennylane_contract
from deuteron_wigner.quantum.m2_pennylane_oracle import M2PennyLaneOracle
from deuteron_wigner.quantum.m2_total_sign_h2_hamiltonian import (
    build_total_sign_h2_hamiltonian,
)

ATOL = 8e-13


@pytest.fixture(scope="module", params=[
    ("PROTON", 0.2, 0.0), ("PROTON", 0.12, 0.16),
    ("NEUTRON", 0.2, 0.0), ("NEUTRON", 0.12, 0.16),
])
def oracle(request):
    target, dx, dy = request.param
    assembly = build_total_sign_h2_hamiltonian(target)
    currents = [
        build_hamiltonian_gauged_baryon_current(build_m2_baryon_density_operator(
            assembly, delta_x_GeV=sign * dx, delta_y_GeV=sign * dy,
        ))
        for sign in (1, -1)
    ]
    return M2PennyLaneOracle(build_m2_pennylane_contract(assembly, *currents))


@pytest.fixture(scope="module")
def probes():
    rng = np.random.default_rng(20260905)
    vectors = rng.normal(size=(2, 266)) + 1j * rng.normal(size=(2, 266))
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def compact_operators(oracle):
    # Derive quadratures directly from the original directed compact
    # matrices and their adjoints, not from the encoded observable dictionary.
    current = oracle.contract.positive_current
    result = {"hamiltonian": oracle.contract.assembly.matrix}
    for name, operator in (
        ("density", current.density.matrix),
        ("current_x", current.current_x), ("current_y", current.current_y),
    ):
        result[name + "_cosine"] = (operator + operator.getH()) / 2
        result[name + "_sine"] = (operator - operator.getH()) / (2j)
    result["padding_probability"] = sparse.csr_matrix((266, 266))
    return result


def test_backend_state_round_trip_and_explicit_basis_order(oracle, probes):
    with qml.Tracker(oracle.device) as tracker:
        actual = oracle.prepare(probes[0])
    assert tracker.totals["executions"] > 0
    np.testing.assert_allclose(actual[:266], probes[0], rtol=0, atol=2e-15)
    np.testing.assert_array_equal(actual[266:], 0)
    for index in (0, 1, 128, 265):
        vector = np.eye(1, 266, k=index, dtype=complex)[0]
        actual = oracle.prepare(vector)
        assert np.argmax(np.abs(actual)) == index
        assert actual[index] == 1
    assert oracle.wires == tuple(range(9))


def test_complex_non_eigenstate_expectations_use_all_quadratures(oracle, probes):
    operators = compact_operators(oracle)
    for probe in probes:
        assert np.linalg.norm(
            operators["hamiltonian"] @ probe
            - np.vdot(probe, operators["hamiltonian"] @ probe) * probe
        ) > 1.0
        actual = oracle.expectations(probe)
        for name, matrix in operators.items():
            np.testing.assert_allclose(
                actual[name], np.vdot(probe, matrix @ probe), rtol=0, atol=ATOL
            )
        for reverse, current in (
            (False, oracle.contract.positive_current),
            (True, oracle.contract.reversed_current),
        ):
            directed = oracle.directed(actual, reverse=reverse)
            for name, matrix in (
                ("density", current.density.matrix),
                ("current_x", current.current_x), ("current_y", current.current_y),
            ):
                np.testing.assert_allclose(
                    directed[name], np.vdot(probe, matrix @ probe), rtol=0, atol=ATOL
                )


def test_full_actions_are_reconstructed_from_backend_measurements(oracle, probes):
    actual = oracle.actions(probes[0])
    for name, matrix in compact_operators(oracle).items():
        expected = matrix @ probes[0]
        np.testing.assert_allclose(actual[name][:266], expected, rtol=0, atol=ATOL)
        np.testing.assert_allclose(actual[name][266:], 0, rtol=0, atol=ATOL)
    for reverse, current in (
        (False, oracle.contract.positive_current),
        (True, oracle.contract.reversed_current),
    ):
        directed = oracle.directed(actual, reverse=reverse)
        for name, matrix in (
            ("density", current.density.matrix),
            ("current_x", current.current_x), ("current_y", current.current_y),
        ):
            np.testing.assert_allclose(
                directed[name][:266], matrix @ probes[0], rtol=0, atol=ATOL
            )
        lhs = (current.density.delta_x_GeV * directed["current_x"]
               + current.density.delta_y_GeV * directed["current_y"])
        h, rho = oracle.contract.assembly.matrix, current.density.matrix
        np.testing.assert_allclose(
            lhs[:266], h @ (rho @ probes[0]) - rho @ (h @ probes[0]),
            rtol=0, atol=ATOL,
        )
    if oracle.contract.positive_current.density.delta_y_GeV:
        for name in ("current_x", "current_y"):
            assert np.linalg.norm(oracle.directed(actual)[name]) > 0.01
    else:
        np.testing.assert_array_equal(actual["current_y_cosine"], 0)
        np.testing.assert_array_equal(actual["current_y_sine"], 0)


def test_complex_nonorthogonal_and_phase_transitions(oracle, probes):
    operators = compact_operators(oracle)
    for bra, ket in (
        (probes[0], probes[1]), (probes[0], probes[0]),
        (probes[0], 1j * probes[0]), (probes[0], -probes[0]),
    ):
        actual = oracle.transitions(bra, ket)
        for name, matrix in operators.items():
            np.testing.assert_allclose(
                actual[name], np.vdot(bra, matrix @ ket), rtol=0, atol=ATOL
            )
    # Prevent a real-only or zero-transition implementation from passing.
    assert abs(np.vdot(probes[0], operators["hamiltonian"] @ probes[1]).imag) > .01


def test_nonzero_spectral_transitions_obey_hamiltonian_continuity(oracle):
    h = oracle.contract.assembly.matrix
    current = oracle.contract.positive_current
    energies, states = np.linalg.eigh(h.toarray())
    transitions = states.conj().T @ (current.current_x @ states)
    np.fill_diagonal(transitions, 0)
    m, n = np.unravel_index(np.argmax(np.abs(transitions)), transitions.shape)
    assert m != n and abs(transitions[m, n]) > .1
    actual = oracle.directed(oracle.transitions(states[:, m], states[:, n]))
    rho_mn = np.vdot(states[:, m], current.density.matrix @ states[:, n])
    q2 = current.density.delta_x_GeV**2 + current.density.delta_y_GeV**2
    for name, delta in (("current_x", current.density.delta_x_GeV),
                        ("current_y", current.density.delta_y_GeV)):
        expected = delta / q2 * (energies[m] - energies[n]) * rho_mn
        np.testing.assert_allclose(actual[name], expected, rtol=0, atol=ATOL)
    for state in oracle.contract.helicity_pair.states:
        measured = oracle.expectations(state.state)
        np.testing.assert_allclose(measured["hamiltonian"], state.energy_GeV2,
                                   rtol=0, atol=ATOL)
        directed = oracle.directed(measured)
        np.testing.assert_allclose([directed["current_x"], directed["current_y"]],
                                   0, rtol=0, atol=ATOL)


def test_padding_states_and_coherent_leakage_measure_the_penalty(oracle, probes):
    physical = np.r_[probes[0], np.zeros(246)]
    padding = np.zeros(512, dtype=complex)
    padding[266], padding[511] = 1 / np.sqrt(2), 1j / np.sqrt(2)
    pure = oracle.expectations_padded(padding)
    assert pure["padding_probability"] == pytest.approx(1, abs=ATOL)
    assert pure["hamiltonian"] == pytest.approx(
        oracle.contract.padding_penalty_GeV2, abs=ATOL
    )
    for name in oracle.contract.hermitian_observables:
        assert pure[name] == 0
    weight = .23
    mixed = np.sqrt(1 - weight) * physical + np.sqrt(weight) * padding
    np.testing.assert_allclose(oracle.prepare_padded(mixed), mixed, rtol=0, atol=2e-15)
    actual, base = oracle.expectations_padded(mixed), oracle.expectations(probes[0])
    for name in oracle.observable_names:
        np.testing.assert_allclose(actual[name], (1 - weight) * base[name] + weight * pure[name],
                                   rtol=0, atol=ATOL)


def test_invalid_states_are_rejected_without_normalization_or_truncation(oracle, probes):
    for invalid in (np.zeros(266), 2 * probes[0], np.ones(265),
                    np.ones((266, 1)), np.full(266, np.nan), np.full(266, np.inf)):
        with pytest.raises(ValueError, match="state must"):
            oracle.prepare(invalid)
        with pytest.raises(ValueError, match="state must"):
            oracle.expectations(invalid)
    with pytest.raises(ValueError, match="shape"):
        oracle.expectations_padded(probes[0])
    metadata = oracle.execution_metadata()
    assert metadata["device"] == "lightning.qubit"
    assert metadata["shots"] is None
    assert metadata["physical_activation"] is False
    assert metadata["variational_algorithm_selected"] is False
    with pytest.raises(TypeError, match="contract"):
        M2PennyLaneOracle(None)

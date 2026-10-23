import numpy as np
import pytest

from deuteron_wigner.quantum.modular_baryon_algebra import (
    add, baryon, color_action, coupled_cluster, isospin_raise, norm2,
    quantum_numbers, spin_action, spin_squared, wedge, ClusterEmbedding,
)
from deuteron_wigner.quantum.modular_gauge_network import generators


@pytest.mark.parametrize("species,charge", [("p", 1), ("n", 0)])
@pytest.mark.parametrize("spin", [-1, 1])
def test_generated_interpolator_color_spin_flavor_and_charge(species, charge, spin):
    state = baryon(0, species, spin)
    assert norm2(state) == pytest.approx(1, abs=1e-14)
    assert {tuple(quantum_numbers(k).values()) for k in state} == {(3, 1, charge)}
    for t in generators():
        assert norm2(color_action(state, 0, t)) < 1e-27
    assert norm2(add(((1, spin_action(state, [0], "z")), (-spin/2, state)))) < 1e-27
    assert norm2(add(((1, spin_squared(state, [0])), (-.75, state)))) < 1e-27
    raised = isospin_raise(state, [0])
    if species == "p":
        assert norm2(raised) < 1e-27
    else:
        proton = baryon(0, "p", spin)
        # Relative isospin phase is convention dependent; compare both signs.
        assert min(norm2(add(((1, raised), (s, proton)))) for s in (-1, 1)) < 1e-27


def test_odd_exchange_nilpotency_and_associative_modularity():
    a, b, c = baryon(0, "p", 1), baryon(1, "n", -1), baryon(2, "p", -1)
    assert norm2(add(((1, wedge(a, b)), (1, wedge(b, a))))) == 0
    assert norm2(wedge(a, a)) < 1e-27
    assert norm2(add(((1, wedge(wedge(a, b), c)), (-1, wedge(a, wedge(b, c)))))) < 1e-27
    assert norm2(wedge(a, b)) == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize("species,intermediate,j,m,z", [
    (("p",), (), .5, 1, 1),
    (("n",), (), .5, -1, 0),
    (("p", "n"), (2,), 1., 0, 1),
    (("p", "n"), (0,), 0., 0, 1),
    (("p", "n", "n"), (0, 1), .5, 1, 1),
    (("p", "p", "n"), (2, 3), 1.5, 1, 2),
    (("p", "p", "n", "n"), (0, 1, 0), 0., 0, 2),
])
def test_variable_A_Z_and_true_SU2_coupling(species, intermediate, j, m, z):
    sites = tuple(range(len(species)))
    state = coupled_cluster(species, sites, intermediate, m)
    assert norm2(state) == pytest.approx(1, abs=2e-13)
    assert norm2(add(((1, spin_squared(state, sites)), (-j*(j+1), state)))) < 1e-25
    assert norm2(add(((1, spin_action(state, sites, "z")), (-m/2, state)))) < 1e-25
    for mask in state:
        q = quantum_numbers(mask)
        assert q["baryon_number"] == len(species) and q["charge"] == z


def test_overlapping_composites_are_not_independent_normalized_nucleons():
    separated = wedge(baryon(0, "p", 1), baryon(1, "n", 1))
    overlap = wedge(baryon(0, "p", 1), baryon(0, "n", 1))
    assert norm2(separated) == pytest.approx(1, abs=1e-14)
    assert abs(norm2(overlap)-1) > .01


def test_forbidden_spin_path_rejected_without_dynamical_assumptions():
    with pytest.raises(ValueError):
        coupled_cluster(("p", "n"), (0, 1), (1,), 0)
    with pytest.raises(ValueError):
        coupled_cluster(("p",), (0,), (), 3)


@pytest.mark.parametrize("twice_spin", [0, 2])
def test_common_Gram_removes_cluster_order_redundancy_without_charge_refit(twice_spin):
    pn = coupled_cluster(("p", "n"), (0, 0), (twice_spin,), 0)
    np_state = coupled_cluster(("n", "p"), (0, 0), (twice_spin,), 0)
    embedding = ClusterEmbedding((pn, np_state), relative_tolerance=1e-12)
    assert embedding.rank == 1
    assert np.linalg.norm(embedding.matrix@embedding.null_vectors) < 1e-13
    assert np.linalg.norm(embedding.isometry.conj().T@embedding.isometry-np.eye(1)) < 1e-13
    def charge(state):
        return {mask: quantum_numbers(mask)['charge']*v for mask, v in state.items()}
    assert np.max(abs(embedding.kernel(charge)-embedding.gram)) < 1e-13
    assert np.max(abs(embedding.operator(charge)-np.eye(1))) < 1e-13
    assert abs(embedding.kernel(charge)[0, 0]-1) > .01
    with pytest.raises(ValueError):
        embedding.full_state(embedding.null_vectors[:, 0])


def test_common_map_full_carrier_and_nontrivial_transition_matrix_agree():
    a = coupled_cluster(("p", "n"), (0, 1), (2,), 0)
    b = coupled_cluster(("p", "n"), (1, 0), (2,), 0)
    c = add(((.6, a), (.8j, b)))
    embedding = ClusterEmbedding((a, b, c), relative_tolerance=1e-12)
    assert embedding.rank == 2
    def proton_at_first_orbital(state):
        # Resolved quark charge in orbital zero, not a physical nuclear current.
        return {mask: quantum_numbers(mask & ((1 << 12)-1))['charge']*v for mask, v in state.items()}
    values = np.array([proton_at_first_orbital({mask: 1})[mask] for mask in embedding.occupations])
    expected = embedding.isometry.conj().T@(values[:, None]*embedding.isometry)
    assert np.max(abs(expected-embedding.operator(proton_at_first_orbital))) < 1e-13
    assert np.linalg.norm(expected-np.diag(np.diag(expected))) > .1
    coefficients = np.array([.3, -.2j, .7])
    state = embedding.full_state(coefficients)
    full_value = np.vdot(state, values*state)
    cluster_value = (coefficients.conj()@embedding.kernel(proton_at_first_orbital)@coefficients)/(coefficients.conj()@embedding.gram@coefficients)
    assert abs(full_value-cluster_value) < 1e-13


def test_all_pauli_forbidden_basis_is_not_regularized_into_a_state():
    zero = wedge(baryon(0, 'p', 1), baryon(0, 'p', 1))
    with pytest.raises(ValueError):
        ClusterEmbedding((zero,), relative_tolerance=1e-12)

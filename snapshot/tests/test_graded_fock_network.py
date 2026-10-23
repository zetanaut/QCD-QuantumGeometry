from itertools import product
import numpy as np
import pytest

from deuteron_wigner.quantum.graded_fock_network import Mode, Ladder, Word, FockRegister
from deuteron_wigner.quantum.modular_baryon_algebra import baryon, wedge


def vector(register, state):
    result = np.zeros(register.dimension, complex)
    for occupations, amplitude in state.items():
        result[np.ravel_multi_index(occupations, register.dimensions)] = amplitude
    return result


def test_tensor_compiler_matches_independent_full_occupation_matrices():
    register = FockRegister((Mode('q', 'fermion', 1), Mode('g', 'boson', 2),
                             Mode('qbar', 'fermion', 1), Mode('spectator', 'fermion', 1)))
    basis = list(product(*[range(d) for d in register.dimensions]))
    rng = np.random.default_rng(103)
    words = [Word(complex(*rng.normal(size=2)), tuple(Ladder(int(rng.integers(4)), bool(rng.integers(2)))
                                                    for _ in range(length)))
             for length in range(1, 6) for _ in range(6)]
    dense = register.dense(words)
    reference = np.column_stack([vector(register, register.occupation_action(words, {state: 1})) for state in basis])
    np.testing.assert_allclose(dense, reference, atol=1e-14)
    state = {basis[i]: complex(*rng.normal(size=2)) for i in (0, 3, 7, 11, 16, 23)}
    np.testing.assert_allclose(vector(register, register.tensor_action(words, state)), dense@vector(register, state), atol=1e-14)
    np.testing.assert_allclose(register.dense([w.adjoint() for w in words]), dense.conj().T, atol=1e-14)


@pytest.mark.parametrize('i,j', product(range(3), repeat=2))
def test_CAR_survives_nonadjacent_modes_and_boson_interleaving(i, j):
    register = FockRegister((Mode('q0', 'fermion', 1), Mode('g', 'boson', 2),
                             Mode('q1', 'fermion', 1), Mode('qbar', 'fermion', 1)))
    ii, jj = (0, 2, 3)[i], (0, 2, 3)[j]
    a, b = Ladder(ii, False), Ladder(jj, True)
    result = register.dense([Word(1, (a, b)), Word(1, (b, a))])
    np.testing.assert_allclose(result, np.eye(register.dimension)*(i == j), atol=1e-14)


def test_boson_capacity_boundary_is_not_silently_called_a_canonical_commutator():
    register = FockRegister((Mode('g', 'boson', 2),))
    a = Ladder(0, False)
    commutator = (Word(1, (a, a.adjoint())), Word(-1, (a.adjoint(), a)))
    np.testing.assert_allclose(register.dense(commutator), np.diag([1, 1, -2]), atol=1e-14)
    top = {(2,): 1}
    full_then_project = register.project(register.occupation_action(commutator, top, truncate_each_ladder=False))
    np.testing.assert_allclose(vector(register, full_then_project), [0, 0, 1], atol=1e-14)
    assert np.linalg.norm(vector(register, register.tensor_action(commutator, top))-vector(register, full_then_project)) > 2.9


def test_existing_composite_baryon_exterior_algebra_is_reused_as_independent_oracle():
    register = FockRegister(tuple(Mode(f'q{i}', 'fermion', 1) for i in range(24)))
    proton, neutron = baryon(0, 'p', 1), baryon(1, 'n', -1)
    words = [Word(value, tuple(Ladder(i, True) for i in range(24) if mask & (1 << i)))
             for mask, value in proton.items()]
    incoming = {tuple(int(bool(mask & (1 << i))) for i in range(24)): value for mask, value in neutron.items()}
    actual = register.tensor_action(words, incoming)
    expected = {tuple(int(bool(mask & (1 << i))) for i in range(24)): value for mask, value in wedge(proton, neutron).items()}
    assert actual.keys() == expected.keys()
    assert max(abs(actual[k]-expected[k]) for k in expected) < 1e-14
    assert len(actual) == 81


def test_malformed_modes_states_and_dense_expansion_are_rejected():
    with pytest.raises(ValueError):
        Mode('q', 'fermion', 2)
    register = FockRegister(tuple(Mode(str(i), 'fermion', 1) for i in range(12)))
    with pytest.raises(ValueError, match='dimension'):
        register.dense([])
    with pytest.raises(ValueError):
        register.tensor_action([Word(1, (Ladder(99, True),))], {(0,)*12: 1})

import numpy as np
import pytest
from scipy.linalg import expm
from deuteron_wigner.quantum.chiral_soft_matching import (
    ETA, GAMMA, G5, PionPole, PolynomialQuarkDressing, square,
    pion_charge_generator, static_one_pion_exchange,
)

PION = PionPole(.14, .0924)  # Representative scales, not a precision input fit.
K0 = np.array([.04, .08, -.03, .02])
K1 = np.array([-.02, .02, .1, -.04])


@pytest.mark.parametrize("mass", [0., .14])
def test_explicit_pion_matches_nonlocal_response_and_PCAC(mass):
    pion = PionPole(mass, .0924)
    for k in (K0, K1):
        response = pion.axial_response(k)
        np.testing.assert_allclose(response, pion.explicit_response(k), atol=5e-18)
        rhs = pion.decay_constant**2 * pion.mass**2 * k * pion.propagator(square(k))
        np.testing.assert_allclose(k @ ETA @ response, rhs, atol=5e-19)
        pole_only = response-pion.decay_constant**2*ETA
        assert np.linalg.norm(k @ ETA @ pole_only-rhs) > 5e-4


@pytest.mark.parametrize("order", [None, 0, 1, 3])
def test_vector_Ward_identity_includes_contacts_and_gauges_same_expansion(order):
    response = PION.axial_vector_source(K1, K0, order)
    lhs = np.einsum("m,mab->ab", (K1-K0) @ ETA, response)
    rhs = PION.axial_response(K1, order)-PION.axial_response(K0, order)
    np.testing.assert_allclose(lhs, rhs, atol=6e-18)
    only_flight = PION.axial_vector_source(K1, K0, order, include_contacts=False)
    assert np.linalg.norm(np.einsum("m,mab->ab", (K1-K0) @ ETA, only_flight)-rhs) > .001
    np.testing.assert_allclose(response, PION.axial_vector_source(K0, K1, order).transpose(0, 2, 1), atol=1e-17)


@pytest.mark.parametrize("component", range(4))
def test_independent_minimally_gauged_action_derivative(component):
    eps = 2e-6
    derivative = (PION.explicit_response_in_vector_source(K1, K0, component, eps) -
                  PION.explicit_response_in_vector_source(K1, K0, component, -eps))/(2*eps)
    np.testing.assert_allclose(derivative, PION.axial_vector_source(K1, K0)[component], atol=3e-11)


@pytest.mark.parametrize("axis", range(3))
def test_full_tensor_Lorentz_covariance_and_signed_LF_propagator(axis):
    generator = np.zeros((4, 4))
    generator[0, axis+1] = generator[axis+1, 0] = .53
    transform = expm(generator)
    np.testing.assert_allclose(PION.axial_response(transform @ K0), transform @ PION.axial_response(K0) @ transform.T, atol=1e-17)
    current = PION.axial_vector_source(K1, K0)
    transported = np.einsum("mi,aj,bk,ijk->mab", transform, transform, transform, current)
    np.testing.assert_allclose(PION.axial_vector_source(transform @ K1, transform @ K0), transported, atol=2e-16)
    for k in (K0, K1, -K0, transform @ K1):
        assert PION.light_front_propagator(k) == pytest.approx(PION.propagator(square(k)), rel=5e-15)


def test_exact_expansion_error_PCAC_remainder_and_zero_plus_limit():
    for ratio in (.25, .5, 1., 1.5, 2.):
        k = np.array([0., ratio*PION.mass, 0., 0.])
        exact = PION.propagator(square(k))
        for order in (0, 1, 2, 4):
            approximate = PION.propagator(square(k), order)
            assert abs(approximate/exact-1) == pytest.approx(ratio**(2*(order+1)), abs=1e-12)
            residual = k @ ETA @ PION.axial_response(k, order) - PION.decay_constant**2*PION.mass**2*k*approximate
            expected = PION.decay_constant**2*k*(square(k)/PION.mass**2)**(order+1)
            np.testing.assert_allclose(residual, expected, atol=1e-14)
    zero = np.array([0., .12, -.04, 0.])
    with pytest.raises(ValueError): PION.light_front_propagator(zero)
    for eps in (1e-4, 1e-7, -1e-4, -1e-7):
        shifted = zero.copy()
        shifted[0] = shifted[3] = eps/np.sqrt(2)
        assert PION.light_front_propagator(shifted) == pytest.approx(PION.propagator(square(zero)), rel=1e-14)
    # Coincident-invariant divided differences require no artificial epsilon.
    for order in (0, 1, 3):
        s = -.4*PION.mass**2
        analytic = sum(n*s**(n-1)/PION.mass**(2*n+2) for n in range(1, order+1))
        assert PION.divided_difference(s, s, order) == pytest.approx(analytic)


@pytest.mark.parametrize("current_mass,pion_mass", [(.0035, .14), (0., 0.)])
def test_dressed_quark_axial_Ward_constraint_and_bare_vertex_negative(current_mass, pion_mass):
    pion = PionPole(pion_mass, .0924)
    dressing = PolynomialQuarkDressing((1., .12), (.28, .025), .8)
    p0, p1 = np.array([.1, .3, -.1, .2]), np.array([.12, -.1, .2, .1])
    axial, weighted_p = dressing.axial_vertices(p1, p0, current_mass, pion)
    rhs = dressing.inverse(p1, current_mass) @ G5 + G5 @ dressing.inverse(p0, current_mass) + weighted_p
    lhs = np.einsum("m,mij->ij", (p1-p0) @ ETA, axial)
    np.testing.assert_allclose(lhs, rhs, atol=5e-16)
    bare = np.array([g @ G5 for g in GAMMA])
    bare_rhs = dressing.inverse(p1, current_mass) @ G5 + G5 @ dressing.inverse(p0, current_mass) + 2*current_mass*G5
    assert np.linalg.norm(np.einsum("m,mij->ij", (p1-p0) @ ETA, bare)-bare_rhs) > 1.
    # Field representation is reused; no instant-spin replacement of LF phases.
    rapidity = .41
    generator = np.zeros((4, 4)); generator[0, 1] = generator[1, 0] = rapidity
    transform = expm(generator)
    spinor = expm(rapidity/2 * GAMMA[0] @ GAMMA[1])
    moved, moved_p = dressing.axial_vertices(transform @ p1, transform @ p0, current_mass, pion)
    expected = np.einsum("mn,nij->mij", transform, spinor @ axial @ np.linalg.inv(spinor))
    np.testing.assert_allclose(moved, expected, atol=5e-16)
    np.testing.assert_allclose(moved_p, spinor @ weighted_p @ np.linalg.inv(spinor), atol=2e-16)


def test_free_quark_limit_and_unfixed_transverse_axial_response():
    dressing = PolynomialQuarkDressing((1.,), (0.,), 1.)
    axial, weighted = dressing.axial_vertices(K1, K0, .0035, PION)
    np.testing.assert_allclose(axial, np.array([g @ G5 for g in GAMMA]), atol=1e-15)
    np.testing.assert_allclose(weighted, .007*G5, atol=1e-15)
    q = K1-K0
    transverse = np.array([(square(q)*g-q[mu]*sum(q[nu]*ETA[nu,nu]*GAMMA[nu] for nu in range(4))) @ G5 for mu, g in enumerate(GAMMA)])
    assert np.linalg.norm(transverse) > .01
    np.testing.assert_allclose(np.einsum("m,mij->ij", q @ ETA, transverse), 0, atol=1e-18)


def test_pion_charge_and_OPE_spin_isospin_information():
    np.testing.assert_allclose(np.linalg.eigvalsh(pion_charge_generator()), [-1, 0, 1])
    np.testing.assert_array_equal(pion_charge_generator()[2], 0)
    iso0 = np.array([0, 1, -1, 0])/np.sqrt(2)
    triplet = (np.array([1, 0, 0, 0]), np.array([0, 1, 1, 0])/np.sqrt(2), np.array([0, 0, 0, 1]))
    k = np.array([0., 0., .12])
    potential = static_one_pion_exchange(k, PION, 1.27)
    np.testing.assert_allclose(potential, potential.conj().T, atol=1e-14)
    subspace = np.column_stack([np.kron(iso0, s) for s in triplet])
    coefficient = 3*1.27**2/(4*PION.decay_constant**2)*(.12**2/(.12**2+PION.mass**2))
    np.testing.assert_allclose(np.diag(subspace.conj().T @ potential @ subspace)/coefficient, [1, -1, 1], atol=1e-15)
    assert np.linalg.norm(potential-static_one_pion_exchange([.12, 0., 0.], PION, 1.27)) > 10.
    np.testing.assert_array_equal(static_one_pion_exchange([0, 0, 0], PION, 1.27), 0)


def test_invalid_inputs_and_no_silent_pole_or_chiral_assignment():
    with pytest.raises(ValueError): PionPole(-.14, .09)
    with pytest.raises(ValueError): PION.propagator(PION.mass**2)
    with pytest.raises(ValueError): PION.propagator(-.01, -1)
    with pytest.raises(ValueError): PionPole(0., .09).propagator(-.01, 2)
    with pytest.raises(ValueError): PolynomialQuarkDressing((1.,), (.3,), 1.).axial_vertices(K1, K0, 0., PION)

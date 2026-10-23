"""State/current accuracy, counterexamples and representation scope."""

import numpy as np
import pytest

from deuteron_wigner.quantum.scalar_composite_reference import (
    longitudinal_form_factor, triangle_form_factor,
)
from deuteron_wigner.quantum.scalar_joint_representation import (
    A_MIN,A_MAX,ExactCore,PolynomialCore,build_core,dy_current,
    longitudinal_current,state_samples,
)


@pytest.fixture(scope="module")
def dense():
    return build_core(56)


@pytest.fixture(scope="module")
def reduced():
    return build_core(56,5)


@pytest.mark.parametrize("mass,q2", [(1.,0.),(1.,4.),(1.5,1.),(1.9,4.),(1.7,.5625)])
def test_dense_and_exact_routes_against_independent_triangle(dense,mass,q2):
    f = triangle_form_factor(mass,q2,128)
    old = longitudinal_form_factor(mass,q2,128)
    for core in (ExactCore(),dense):
        assert abs(dy_current(core,mass,q2)-f) < 1e-8
        lp = longitudinal_current(core,mass,q2)
        assert abs(lp.total-f) < 1e-8
        assert abs(lp.valence-old.valence) < 1e-8
        assert abs(lp.pair-old.pair) < 1e-8


@pytest.mark.parametrize("mass", [1.,1.5,1.9,1.7])
def test_whitened_state_and_shared_charge_no_renormalization(dense,reduced,mass):
    exact = state_samples(ExactCore(),mass)
    assert np.linalg.norm(exact)**2 == pytest.approx(1,abs=1e-10)
    assert np.linalg.norm(state_samples(dense,mass)-exact) < 1e-8
    b = state_samples(reduced,mass)
    assert np.linalg.norm(b-exact) < 1e-3
    # Three independent evaluations of the same represented, NOT rescaled, norm.
    norm = np.linalg.norm(b)**2
    assert dy_current(reduced,mass,0) == pytest.approx(norm,abs=1e-12)
    lp = longitudinal_current(reduced,mass,0)
    assert lp.valence == pytest.approx(norm,abs=1e-12)
    assert lp.pair == 0j


@pytest.mark.parametrize("mass,q2", [(1.,4.),(1.5,1.),(1.9,.25),
    (1.,.0625),(1.5,.5625),(1.9,2.25),(1.7,.0625),(1.7,2.25)])
def test_reduced_currents_and_separate_pair_error(reduced,mass,q2):
    reference = triangle_form_factor(mass,q2,128)
    old = longitudinal_form_factor(mass,q2,128)
    dy = dy_current(reduced,mass,q2)
    lp = longitudinal_current(reduced,mass,q2)
    assert abs(dy-reference) < 1e-4
    assert abs(lp.total-reference) < 1e-4
    assert abs(lp.valence-old.valence) < 1e-4
    assert abs(lp.pair-old.pair) < 1e-4
    assert abs(dy-lp.total) < 1e-4


def test_pair_is_not_exact_result_pasted_onto_compressed_valence():
    coarse = build_core(56,1)
    lp = longitudinal_current(coarse,1.5,1.)
    reference = longitudinal_form_factor(1.5,1.,128)
    assert abs(lp.pair-reference.pair) > 1e-2
    assert abs(lp.total-reference.total) > 1e-2


def test_rank_four_is_not_promoted(dense):
    coarse = PolynomialCore(dense.coefficients,4)
    error = abs(dy_current(coarse,1.5,4)-triangle_form_factor(1.5,4,128))
    assert error > 1e-4
    assert np.linalg.norm(state_samples(coarse,1.5)-state_samples(ExactCore(),1.5)) > 1e-3


def test_numerical_degree_and_quadrature_are_separate(reduced):
    low = build_core(24)
    # Rank cannot repair an unresolved interpolant: full degree 24 already errs.
    assert abs(dy_current(low,1.,4)-triangle_form_factor(1.,4,128)) > 1e-8
    assert abs(dy_current(reduced,1.9,4,64)-dy_current(reduced,1.9,4,96)) < 1e-8


def test_reduced_storage_is_real_not_dense_matrix_in_disguise(dense,reduced):
    assert dense.stored_coefficients == 57**2
    assert reduced.coefficients is None
    assert reduced.left.shape == (57,5)
    assert reduced.right.shape == (5,57)
    assert reduced.stored_coefficients == 570
    assert dense.stored_bytes/reduced.stored_bytes == pytest.approx(5.7)


def test_complex_core_and_full_rank_factorization(dense):
    phase = np.exp(.73j)
    complex_core = PolynomialCore(phase*dense.coefficients,57)
    assert complex_core.coefficients is None
    b = state_samples(complex_core,1.5)
    assert np.linalg.norm(b-phase*state_samples(dense,1.5)) < 1e-12
    assert abs(dy_current(complex_core,1.5,1)-dy_current(dense,1.5,1)) < 1e-12
    assert abs(longitudinal_current(complex_core,1.5,1).pair-
               longitudinal_current(dense,1.5,1).pair) < 1e-12


def test_basis_phases_need_matching_operator_transport(reduced):
    phase = lambda a,s: .8*a+.6*s/(1+s)
    dy = dy_current(reduced,1.5,1)
    lp = longitudinal_current(reduced,1.5,1)
    transported = longitudinal_current(reduced,1.5,1,phase=phase)
    assert abs(dy_current(reduced,1.5,1,phase=phase)-dy) < 1e-12
    assert abs(transported.valence-lp.valence) < 1e-12
    assert abs(transported.pair-lp.pair) < 1e-12
    wrong = longitudinal_current(reduced,1.5,1,phase=phase,transport_operator=False)
    assert abs(wrong.pair-lp.pair) > 1e-2


def test_ultraviolet_falloff_is_not_exact_tail_coefficient(dense,reduced):
    points = np.linspace(A_MIN,A_MAX,121)
    assert max(abs(dense.bind(a).g(1)-1) for a in points) < 1e-12
    # Preserve this limitation. The finite-grid success is NOT an exact UV test.
    drift = max(abs(reduced.bind(a).g(1)-1) for a in points)
    assert 1e-4 < drift < 2e-3


@pytest.mark.parametrize("degree,rank", [(4,None),(12,0),(12,14),(12,1.5)])
def test_bad_representation_settings(degree,rank):
    with pytest.raises(ValueError):
        build_core(degree,rank)


def test_no_virtuality_extrapolation_or_new_physical_domain(dense):
    for a in (A_MIN/2,2.1,np.nan):
        with pytest.raises(ValueError):
            dense.bind(a)
    for m,q in ((1.95,0),(1.5,4.1),(1.5,-1)):
        with pytest.raises(ValueError):
            dy_current(dense,m,q)
    with pytest.raises(ValueError):
        dense.bind(1).h(-.1)

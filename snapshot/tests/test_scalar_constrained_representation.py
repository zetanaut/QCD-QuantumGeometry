"""Independent algebra, boundary and state/current checks of the scalar lift."""

import numpy as np
import pytest

from deuteron_wigner.quantum.scalar_joint_representation import (
    A_MIN,A_MAX,ExactCore,PolynomialCore,build_core,dy_current,
    longitudinal_current,state_samples,
)
from deuteron_wigner.quantum.scalar_constrained_representation import ConstrainedCore
from deuteron_wigner.quantum.scalar_composite_reference import triangle_form_factor,longitudinal_form_factor


@pytest.fixture(scope="module")
def selected():
    return ConstrainedCore(build_core(56,5),"lift")


@pytest.mark.parametrize("mode", ["lift","projection"])
def test_constraints_for_arbitrary_complex_residuals(mode):
    rng = np.random.default_rng(927)
    coefficients = rng.normal(size=(9,9))+1j*rng.normal(size=(9,9))
    core = ConstrainedCore(PolynomialCore(coefficients,3),mode)
    for a in np.linspace(A_MIN,A_MAX,23):
        assert core.bind(a).g(0) == pytest.approx(1/a,abs=1e-12)
        assert core.bind(a).g(1) == pytest.approx(1,abs=1e-12)
    assert np.max(abs(core.bind(1).g(np.linspace(0,1,37))-1)) < 1e-12


def test_exact_lift_identity_and_absolute_error_bound():
    a = np.linspace(A_MIN,A_MAX,53)[:,None]
    t = np.linspace(0,1,59)[None,:]
    exact = 1/(a+(1-a)*t)
    line = (1-t)/a+t
    weight = (1-a)**2/a*t*(1-t)
    assert np.max(abs(line-weight*exact-exact)) < 1e-14
    epsilon = 1e-4*(1+.5j)
    assert np.max(abs((line-weight*(exact+epsilon)-exact)+weight*epsilon)) < 1e-14
    assert np.max(weight) <= (1-A_MIN)**2/(4*A_MIN)


def test_projection_matches_continuous_boundary_formula():
    rng = np.random.default_rng(319)
    raw = PolynomialCore(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)),3)
    projected = ConstrainedCore(raw,"projection")
    t = np.linspace(0,1,43)
    f1 = raw.bind(1)
    for a in (.2,.7,1.,1.6):
        f = raw.bind(a)
        expected = ((1-t)/a+t + f.g(t)-(1-t)*f.g(0)-t*f.g(1)
                    -f1.g(t)+(1-t)*f1.g(0)+t*f1.g(1))
        assert np.max(abs(projected.bind(a).g(t)-expected)) < 1e-12


def test_lift_preserves_first_virtuality_derivative_at_unit_line(selected):
    t = np.linspace(0,1,31)
    h = 1e-5
    derivative = (selected.bind(1+h).g(t)-selected.bind(1-h).g(t))/(2*h)
    assert np.max(abs(derivative+(1-t))) < 1e-8


@pytest.mark.parametrize("mass,q2", [(1.,0.),(1.5,1.),(1.9,4.),
    (1.2,.16),(1.4,.81),(1.6,1.69),(1.8,3.24)])
def test_selected_currents_and_explicit_pair(selected,mass,q2):
    f = triangle_form_factor(mass,q2,128)
    reference = longitudinal_form_factor(mass,q2,128)
    dy = dy_current(selected,mass,q2)
    lp = longitudinal_current(selected,mass,q2)
    assert abs(dy-f) < 1e-4
    assert abs(lp.total-f) < 1e-4
    assert abs(lp.valence-reference.valence) < 1e-4
    assert abs(lp.pair-reference.pair) < 1e-4
    assert abs(dy-lp.total) < 1e-4


@pytest.mark.parametrize("mass", [1.,1.5,1.9,1.2,1.4,1.6,1.8])
def test_states_and_no_post_compression_normalization(selected,mass):
    b = state_samples(selected,mass)
    assert np.linalg.norm(b-state_samples(ExactCore(),mass)) < 1e-3
    norm = np.linalg.norm(b)**2
    assert dy_current(selected,mass,0) == pytest.approx(norm,abs=1e-12)
    assert abs(norm-1) < 1e-4


def test_selected_rank_is_not_just_an_observable_pass():
    coarse = ConstrainedCore(build_core(56,4),"lift")
    # Rank four meets sampled currents, but loses more state information than allowed.
    assert abs(dy_current(coarse,1.9,4)-triangle_form_factor(1.9,4,128)) < 1e-4
    assert np.linalg.norm(state_samples(coarse,1.9)-state_samples(ExactCore(),1.9)) > 1e-3


def test_exact_boundaries_are_not_sufficient_for_good_physics():
    bad = ConstrainedCore(PolynomialCore(np.zeros((9,9))),"lift")
    assert bad.bind(.2).g(0) == 5
    assert bad.bind(.2).g(1) == 1
    assert bad.bind(1).g(.4) == 1
    assert abs(dy_current(bad,1.9,1)-triangle_form_factor(1.9,1,128)) > .01


def test_phase_transport_and_missing_pair_remain_decisive(selected):
    phase = lambda a,s: .8*a+.6*s/(1+s)
    base = longitudinal_current(selected,1.5,1)
    correct = longitudinal_current(selected,1.5,1,phase=phase)
    wrong = longitudinal_current(selected,1.5,1,phase=phase,transport_operator=False)
    assert abs(base.total-correct.total) < 1e-12
    assert abs(base.pair-wrong.pair) > .01
    assert abs(base.valence/triangle_form_factor(1.5,1,128)-1) > .1


def test_density_resolution_and_large_momentum_tail(selected):
    assert abs(dy_current(selected,1.9,4,64)-dy_current(selected,1.9,4,96)) < 1e-8
    for a in np.linspace(A_MIN,A_MAX,17):
        assert a*selected.bind(a).h(0) == pytest.approx(1,abs=1e-15)
        assert 1e14*selected.bind(a).h(1e14) == pytest.approx(1,abs=3e-13)


def test_storage_counts_include_projection_anchor(selected):
    assert selected.stored_coefficients == 570
    projected = ConstrainedCore(build_core(56,8),"projection")
    assert projected.stored_coefficients == 2*57*8+57
    assert projected.stored_bytes == projected.stored_coefficients*8


def test_no_exact_reference_queries_inside_constrained_current(monkeypatch,selected):
    def forbidden(*args,**kwargs):
        raise AssertionError("ExactCore is not a production dependency of the represented current.")
    monkeypatch.setattr(ExactCore,"bind",forbidden)
    assert np.isfinite(dy_current(selected,1.5,1))
    assert np.isfinite(longitudinal_current(selected,1.5,1).pair)


def test_domain_and_mode_validation():
    raw = build_core(16,3)
    with pytest.raises(ValueError):
        ConstrainedCore(raw,"automatic")
    with pytest.raises(TypeError):
        ConstrainedCore(ExactCore())
    constrained = ConstrainedCore(raw)
    with pytest.raises(ValueError):
        constrained.bind(A_MIN/2)
    with pytest.raises(ValueError):
        constrained.bind(1).g(1.1)

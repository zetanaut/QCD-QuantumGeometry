import numpy as np
import pytest

from deuteron_wigner.quantum.m2_constrained_propagators import (
    METRIC, NULL, slash, fermion_propagator_parts, gluon_propagator_parts,
)
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("mass", [0., .35, 1.2])
def test_fermion_signed_plus_decomposition_and_independent_inverse(sign, mass):
    p = sign*np.array([1.1, .23, -.31, 1.7])
    cov, dyn, instant = fermion_propagator_parts(p, mass)
    np.testing.assert_allclose(cov, dyn+instant, atol=3e-15)
    np.testing.assert_allclose(cov, np.linalg.inv(slash(p)-mass*np.eye(4)), atol=3e-15)
    assert np.linalg.norm(instant) > .1


@pytest.mark.parametrize("mass", [0., .35])
def test_positive_plus_propagating_numerator_is_spinor_completeness(mass):
    p = np.array([2.2, .27, -.35, .8])
    pp = NULL@METRIC@p
    _, dyn, _ = fermion_propagator_parts(p, mass)
    u = [lf_spinor(pp, p[1], p[2], mass, h) for h in (-1, 1)]
    from deuteron_wigner.bridge.modes.core import GAMMA
    numerator = sum(np.outer(v, v.conj())@GAMMA[0] for v in u)
    np.testing.assert_allclose(dyn, numerator/(p@METRIC@p-mass**2), atol=3e-15)


@pytest.mark.parametrize("sign", [-1, 1])
def test_doubly_transverse_and_instantaneous_gluon_exchange(sign):
    p = sign*np.array([1.1, .23, -.31, 1.7])
    axial, dt, instant = gluon_propagator_parts(p)
    np.testing.assert_allclose(dt@METRIC@p, 0., atol=3e-15)
    np.testing.assert_allclose(dt@METRIC@NULL, 0., atol=3e-15)
    np.testing.assert_allclose(dt+instant, axial, atol=3e-15)
    rng = np.random.default_rng(1704)
    a, b = rng.normal(size=(2, 4))
    a -= p*(a@METRIC@p)/(p@METRIC@p)
    b -= p*(b@METRIC@p)/(p@METRIC@p)
    current_contraction = a@METRIC@(dt+instant)@METRIC@b/(p@METRIC@p)
    covariant = -a@METRIC@b/(p@METRIC@p)
    assert current_contraction == pytest.approx(covariant, abs=2e-14)
    assert abs(a@METRIC@instant@METRIC@b) > .01


@pytest.mark.parametrize("p,mass", [([1., 0., 0., -1.], .35), ([1., 0., 0., 0.], 1.), ([1., 0., 0., 0.], -.1)])
def test_fermion_rejects_undefined_points(p, mass):
    with pytest.raises(ValueError):
        fermion_propagator_parts(p, mass)


def test_gluon_rejects_zero_plus_no_invented_prescription():
    with pytest.raises(ValueError):
        gluon_propagator_parts([1., 0., 0., -1.])

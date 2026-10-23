"""Direct differential geometry/current controls, not physical activation tests."""

import numpy as np
import pytest

from deuteron_wigner.quantum.scalar_joint_representation import (
    PolynomialCore,build_core,state_samples,dy_current,longitudinal_current,
)
from deuteron_wigner.quantum.scalar_constrained_representation import ConstrainedCore
from deuteron_wigner.quantum.scalar_representation_geometry import (
    TangentChart,tangent_chart,state_jacobian,horizontal_tangent,current_gradients,
    coefficient_matrix,displaced_core,weak_response_direction,
)


@pytest.fixture(scope="module")
def setup():
    core = ConstrainedCore(build_core(56,5))
    return core,tangent_chart(core.core)


def vector(core,mass,q2,order=96):
    lp = longitudinal_current(core,mass,q2,order)
    return np.array([dy_current(core,mass,q2,order),lp.valence,lp.pair])


def test_orthonormal_coefficient_chart_and_actual_dimension(setup):
    core,chart = setup
    gram = (chart.left.T@chart.left)*(chart.right.T@chart.right)
    assert chart.dimension == 5*(57+57-5) == 545
    assert chart.fixed_right_count == 285
    assert np.max(abs(gram-np.eye(545))) < 1e-13
    rng = np.random.default_rng(831)
    coordinates = rng.normal(size=545)+1j*rng.normal(size=545)
    assert np.linalg.norm(chart.matrix(coordinates)) == pytest.approx(np.linalg.norm(coordinates),rel=1e-13)


def test_chart_spans_raw_factor_tangents_and_removes_only_factor_gauge(setup):
    core,chart = setup
    rng = np.random.default_rng(323)
    l,r = core.core.left,core.core.right
    dl = rng.normal(size=l.shape)+1j*rng.normal(size=l.shape)
    dr = rng.normal(size=r.shape)+1j*rng.normal(size=r.shape)
    delta = dl@r+l@dr
    assert np.linalg.norm(chart.matrix(chart.contract(delta))-delta)/np.linalg.norm(delta) < 1e-13
    x = rng.normal(size=(5,5))+1j*rng.normal(size=(5,5))
    zero = (l@x)@r+l@(-x@r)
    assert np.linalg.norm(zero) < 1e-13
    change = np.diag(np.linspace(.5,3.,5))+.1*rng.normal(size=(5,5))
    assert np.linalg.norm((l@change)@np.linalg.solve(change,r)-l@r) < 1e-13


@pytest.mark.parametrize("imaginary", [False,True])
def test_raw_and_projective_state_derivatives(setup,imaginary):
    core,chart = setup
    rng = np.random.default_rng(421)
    v = rng.normal(size=545)
    v /= np.linalg.norm(v)
    if imaginary:
        v = 1j*v
    b,j,a = state_jacobian(core,chart,1.5,96)
    eps = 1e-4
    bp = state_samples(displaced_core(core,chart,v,eps),1.5,96).ravel()
    bm = state_samples(displaced_core(core,chart,v,-eps),1.5,96).ravel()
    assert np.linalg.norm((bp-bm)/(2*eps)-j@v) < 1e-10
    normalized_fd = (bp/np.linalg.norm(bp)-bm/np.linalg.norm(bm))/(2*eps)
    psi = b/np.linalg.norm(b)
    projected_fd = normalized_fd-psi*np.vdot(psi,normalized_fd)
    assert np.linalg.norm(projected_fd-a@v) < 1e-9
    assert abs(np.vdot(psi,a@v)) < 1e-14


def test_global_phase_is_external_ray_gauge_not_free_boundary_parameter(setup):
    core,chart = setup
    b,j,a = state_jacobian(core,chart,1.5,64)
    assert np.linalg.norm(horizontal_tangent(b,1j*b)) < 1e-14
    assert np.linalg.norm(horizontal_tangent(b,b)) < 1e-14
    assert np.linalg.norm(1j*b/np.linalg.norm(b)) == pytest.approx(1)
    # Multiplying the whole core by a phase violates its fixed endpoint.
    assert abs(np.exp(.2j)*core.bind(.2).g(1)-1) > .1


@pytest.mark.parametrize("mass,q2", [(1.5,0.),(1.5,1.),(1.2,.81),(1.8,3.24)])
def test_both_complex_current_leg_derivatives_match_independent_differences(setup,mass,q2):
    core,chart = setup
    rng = np.random.default_rng(72)
    v = rng.normal(size=545)+1j*rng.normal(size=545)
    v /= np.linalg.norm(v)
    g = current_gradients(core,chart,mass,q2,96)
    eps = 1e-4
    fd = (vector(displaced_core(core,chart,v,eps),mass,q2)-
          vector(displaced_core(core,chart,v,-eps),mass,q2))/(2*eps)
    analytic = g["real"]@v.real+g["imaginary"]@v.imag
    assert np.max(abs(fd-analytic)) < 1e-8
    assert np.max(abs(analytic-(g["outgoing"]@v.conj()+g["incoming"]@v))) < 1e-14
    if q2 > 0:
        assert np.linalg.norm(analytic-g["incoming"]@v) > 1e-5


def test_complex_projective_tensor_has_imaginary_cross_block(setup):
    core,chart = setup
    b,j,a = state_jacobian(core,chart,1.5,64)
    sub = a[:,:4]
    complex_j = np.column_stack((sub,1j*sub))
    q = complex_j.conj().T@complex_j
    gram = sub.T@sub
    assert np.max(abs(q[:4,4:]-1j*gram)) < 1e-14
    assert np.max(abs(q.real[:4,4:])) < 1e-14
    assert np.linalg.eigvalsh(q.real).min() > -1e-14
    assert np.linalg.norm(q.imag) > 1e-5


def test_normalization_differentiated_without_changing_reference_current(setup):
    core,chart = setup
    b,j,a = state_jacobian(core,chart,1.5,128)
    norm_gradient = 2*b@j
    v = norm_gradient/np.linalg.norm(norm_gradient)
    gp = current_gradients(core,chart,1.5,0,96)["real"][0]@v
    assert abs(gp-norm_gradient@v) < 1e-12
    n = np.linalg.norm(b)**2
    f0 = dy_current(core,1.5,0,96)
    assert abs(f0-n) < 1e-12
    assert abs(gp/n-f0*(norm_gradient@v)/n**2) < 1e-12
    assert abs(gp/n) > .01  # omitting dn would wrongly retain a charge response


def test_coordinate_covariance_on_identified_small_chart(setup):
    core,chart = setup
    a = state_jacobian(core,chart,1.5,96)[2]
    rng = np.random.default_rng(83)
    z,_ = np.linalg.qr(rng.normal(size=(545,7)))
    d = a@z
    h = np.diag(np.linspace(.4,2.,7))+.05*rng.normal(size=(7,7))
    metric = d.T@d
    covector = rng.normal(size=7)
    delta = d@np.linalg.solve(metric,covector)
    transformed = (d@h)@np.linalg.solve((d@h).T@(d@h),h.T@covector)
    assert np.linalg.norm(transformed-delta)/np.linalg.norm(delta) < 1e-10
    assert np.linalg.norm(d@covector-d@h@h.T@covector) > .01*np.linalg.norm(d@covector)


def test_fixed_right_affine_path_preserves_rank_and_all_boundaries(setup):
    core,full = setup
    n = full.fixed_right_count
    chart = TangentChart(full.left[:,:n],full.right[:,:n],n)
    v = np.random.default_rng(661).normal(size=n)
    v /= np.linalg.norm(v)
    changed = displaced_core(core,chart,v,.25)
    assert np.linalg.svd(coefficient_matrix(changed.core),compute_uv=False)[5] < 1e-13
    for a in (.1,.5,1.,1.5,2.):
        assert changed.bind(a).g(0) == pytest.approx(1/a,abs=1e-14)
        assert changed.bind(a).g(1) == 1
    assert np.max(abs(changed.bind(1).g(np.linspace(0,1,39))-1)) == 0


def test_state_weak_ensemble_direction_can_change_pair_current(setup):
    core,full = setup
    n = full.fixed_right_count
    chart = TangentChart(full.left[:,:n],full.right[:,:n],n)
    a = np.vstack([state_jacobian(core,chart,m,128)[2]/np.sqrt(3) for m in (1.,1.5,1.9)])
    pair = current_gradients(core,chart,1.5,1.,96)["real"][2]
    v,_,count = weak_response_direction(a,pair,1e-8)
    assert count > 0
    assert np.linalg.norm(a@v) < 1e-8
    assert abs(pair@v) > 1e-4
    assert np.linalg.norm(state_jacobian(core,chart,1.8,160)[2]@v) < 1e-8
    changed = displaced_core(core,chart,v,.25)
    assert np.linalg.norm(state_samples(changed,1.5,160)-state_samples(core,1.5,160)) < 1e-8
    assert abs(longitudinal_current(changed,1.5,1,96).pair-longitudinal_current(core,1.5,1,96).pair) > 1e-4


def test_scope_and_invalid_geometry_inputs(setup):
    core,chart = setup
    with pytest.raises(ValueError):
        tangent_chart(PolynomialCore(np.eye(8)))
    with pytest.raises(ValueError):
        chart.matrix(np.zeros(3))
    with pytest.raises(ValueError):
        horizontal_tangent(np.zeros(3),np.ones(3))
    with pytest.raises(ValueError):
        weak_response_direction(np.eye(4),np.ones(4),0)
    with pytest.raises(ValueError):
        weak_response_direction(np.eye(4),np.ones(4),1e-8)
    with pytest.raises(ValueError):
        state_jacobian(ConstrainedCore(core.core,"projection"),chart,1.5)

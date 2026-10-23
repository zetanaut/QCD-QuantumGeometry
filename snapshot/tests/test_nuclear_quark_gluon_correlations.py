"""Direct operator tests; reference profiles are not physical nuclear PDFs."""
import numpy as np
import pytest
from scipy.integrate import quad, quad_vec

from deuteron_wigner.quantum.nuclear_quark_gluon_correlations import (
    binding_scalar_match, embed_nucleon_pdfs, evolve_moment,
    kernel_moment_by_quadrature, lo_x_action, moment_generator,
    moment_null_shape, nonsinglet_moment,
)


def reference_pdf(x):
    # Distinct smooth shapes in all six q/qbar channels and the gluon channel.
    return np.array([(j+1)*x**(1+j % 2)*(1-x)**(2+j % 3) for j in range(7)])


@pytest.mark.parametrize("nf,n", [(1,2), (2,3), (3,2), (3,4), (4,5), (6,7)])
def test_analytic_mellin_matches_distribution_integral(nf, n):
    np.testing.assert_allclose(moment_generator(n,nf), kernel_moment_by_quadrature(n,nf),
                               atol=7e-14, rtol=2e-14)


@pytest.mark.parametrize("nf", [1,2,3,6])
def test_momentum_and_valence_conservation(nf):
    np.testing.assert_allclose(moment_generator(2,nf).sum(axis=0), 0, atol=3e-15)
    assert nonsinglet_moment(1) == 0
    assert np.all(moment_generator(2,nf)[-1,:-1] > 0)
    assert np.all(moment_generator(2,nf)[:-1,-1] > 0)


@pytest.mark.parametrize("n", [2,3])
def test_actual_x_convolution_has_independent_mellin_moment(n):
    initial, _ = quad_vec(lambda x: x**(n-1)*reference_pdf(x), 0, 1,
                          epsabs=1e-11, epsrel=1e-11)
    actual, _ = quad_vec(lambda x: x**(n-1)*lo_x_action(reference_pdf,x,3), 0, 1,
                         epsabs=3e-9, epsrel=3e-9)
    np.testing.assert_allclose(actual, moment_generator(n,3)@initial,
                               atol=5e-9, rtol=2e-8)


def test_valence_number_from_actual_x_convolution():
    integral, _ = quad_vec(lambda x: (lo_x_action(reference_pdf,x,3)[:-1]
                                     .reshape(3,2)@np.array([1.,-1.])),
                          0, 1, epsabs=3e-9, epsrel=3e-9)
    np.testing.assert_allclose(integral, 0, atol=5e-9)


def test_source_tensor_axes_and_complex_coherences_survive():
    source = np.array([[1., .2+.3j], [.2-.3j, -.4]])
    scalar = lo_x_action(reference_pdf,.23,3)
    tensor = lo_x_action(lambda x: reference_pdf(x)[:,None,None]*source,.23,3)
    np.testing.assert_allclose(tensor, scalar[:,None,None]*source, atol=2e-11)
    moment = np.arange(1.,8.)[:,None,None]*source
    result, _ = evolve_moment(moment,2,3,.25,2.,7.)
    np.testing.assert_allclose(result.sum(axis=0), moment.sum(axis=0), atol=3e-14)


def test_running_moment_transport_and_two_channel_closed_form():
    initial = np.array([.2,.03,.15,.04,.04,.04,.5])
    final, alpha = evolve_moment(initial,2,3,.25,2.,8.)
    mid, alpha_mid = evolve_moment(initial,2,3,.25,2.,3.)
    chained, alpha_chain = evolve_moment(mid,2,3,alpha_mid,3.,8.)
    reverse, alpha_back = evolve_moment(final,2,3,alpha,8.,2.)
    np.testing.assert_allclose(chained, final, atol=3e-16)
    np.testing.assert_allclose(reverse, initial, atol=3e-16)
    assert abs(alpha-alpha_chain) < 1e-15 and abs(alpha_back-.25) < 1e-15
    equilibrium_q = 9/25
    decay = (alpha/.25)**(50/81)
    assert abs(final[:-1].sum()-(equilibrium_q+(.5-equilibrium_q)*decay)) < 3e-16
    assert abs(final.sum()-1) < 3e-16


def proton_pdf(x):
    # Each valence beta(2,8) has mean x=.2. q valence carries .6;
    # a .8*beta(2,2) gluon has momentum .4. nf=2 reference, sea zero.
    v = 72*x*(1-x)**7
    return np.array([2*v,0.,v,0.,.8*6*x*(1-x)])


def neutron_pdf(x):
    p = proton_pdf(x)
    return p[[2,3,0,1,4]]


@pytest.mark.parametrize("a,z", [(1,1),(1,0),(2,1),(3,1),(4,2),(12,6)])
def test_whole_target_embedding_number_momentum_and_support(a,z):
    fractions = np.arange(1,a+1,dtype=float)
    fractions /= fractions.sum()
    pdfs = [proton_pdf]*z + [neutron_pdf]*(a-z)
    f = lambda x: embed_nucleon_pdfs(pdfs,fractions,x,2)
    # Split at all known support boundaries; no interpolation/extrapolation.
    points = np.unique(np.r_[0.,fractions,1.])
    number = sum(quad_vec(f,lo,hi,epsabs=2e-10)[0]
                 for lo,hi in zip(points[:-1],points[1:]))
    momentum = sum(quad_vec(lambda x:x*f(x),lo,hi,epsabs=2e-10)[0]
                   for lo,hi in zip(points[:-1],points[1:]))
    assert abs(number[0]-number[1]-(a+z)) < 2e-11
    assert abs(number[2]-number[3]-(2*a-z)) < 2e-11
    assert abs(momentum.sum()-1) < 2e-11
    if a > 1:
        np.testing.assert_array_equal(f((1+fractions.max())/2), np.zeros(5))


@pytest.mark.parametrize("x", [.07,.22,.48])
def test_embedding_and_actual_x_evolution_commute(x):
    fractions = [.37,.63]
    nuclear = lambda t: embed_nucleon_pdfs([proton_pdf,neutron_pdf],fractions,t,2)
    left = lo_x_action(nuclear,x,2)
    right = sum(lo_x_action(pdf,x/y,2)/y for pdf,y in
                zip([proton_pdf,neutron_pdf],fractions) if x < y)
    np.testing.assert_allclose(left,right,atol=3e-8,rtol=3e-8)


def test_moment_null_shape_changes_x_and_higher_moments_after_evolution():
    moments = [quad(lambda x:x**(n-1)*moment_null_shape(x),0,1)[0] for n in [1,2,3]]
    np.testing.assert_allclose(moments,[0.,0.,1/420],atol=2e-16)
    direction = np.array([1.,1.,-.3,-.3,.4,.4,-.5])
    deformation = lambda x: direction*moment_null_shape(x)
    assert np.linalg.norm(lo_x_action(deformation,.18,3)) > .01
    evolved, _ = evolve_moment(direction/420,3,3,.25,2,6)
    assert np.linalg.norm(evolved) > .001
    # No valence change even pointwise; all momentum moments remain zero.
    np.testing.assert_array_equal(direction[:-1].reshape(3,2)@np.array([1.,-1.]),0)


def test_scalar_binding_matches_independent_common_mass_source_difference():
    m0 = np.array([.004,.007,.08])
    sp, sn = np.array([.020,.028,.015]), np.array([.023,.025,.016])
    db = np.array([.0006,-.0002,.0001])
    gamma = np.array([.18,.18,.18])
    a,z,b0 = 4,2,.025
    def mass(m):
        logs = np.log(m/m0)
        return z*(.94+sp@logs)+(a-z)*(.95+sn@logs)-(b0+db@logs)
    match = binding_scalar_match(a,z,.94,.95,sp,sn,b0,db,gamma)
    numerical = []
    for j in range(3):
        step = np.zeros(3); step[j] = m0[j]*1e-4
        numerical.append(m0[j]*(mass(m0+step)-mass(m0-step))/(2*step[j]))
    np.testing.assert_allclose(match.sigma,numerical,atol=5e-10)
    assert abs(match.mass-mass(m0)) < 1e-15
    assert abs(match.anomaly+(1+gamma)@match.sigma-match.mass) < 1e-15
    baseline_anomaly = z*(.94-(1+gamma)@sp)+(a-z)*(.95-(1+gamma)@sn)
    assert abs(match.anomaly-baseline_anomaly-match.connected_anomaly) < 5e-16
    assert abs(match.connected_anomaly+(1+gamma)@match.connected_sigma+b0) < 1e-16


def test_same_binding_does_not_fix_condensate_and_cluster_limit_is_additive():
    args = (2,1,.94,.95,[.02,.03],[.03,.02])
    one = binding_scalar_match(*args,.002,[.0004,-.0001],[.2,.2])
    two = binding_scalar_match(*args,.002,[-.001,.003],[.2,.2])
    assert one.mass == two.mass
    assert np.linalg.norm(one.sigma-two.sigma) > .003
    isolated = binding_scalar_match(*args,0.,[0.,0.],[.2,.2])
    np.testing.assert_array_equal(isolated.connected_sigma,0)
    assert isolated.connected_anomaly == 0


@pytest.mark.parametrize("call", [
    lambda: moment_generator(1,3), lambda: moment_generator(2,0),
    lambda: lo_x_action(reference_pdf,0,3),
    lambda: evolve_moment(np.ones(7),2,3,.5,2,1e-10),
    lambda: embed_nucleon_pdfs([proton_pdf],[.5],.2,2),
    lambda: binding_scalar_match(2,3,.94,.95,[.1],[.1],.002,[.001],[.2]),
])
def test_invalid_scope_is_not_silently_reinterpreted(call):
    with pytest.raises(ValueError):
        call()

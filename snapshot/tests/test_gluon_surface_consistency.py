import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from deuteron_wigner.quantum.gluon_surface_consistency import (
    Patch, Path, baryon_wilson, box_action_exact, closed_box, cube_graph,
    cube_rewrites, factored_loop, interval_integral, quadratic_action,
    rectangle, rewrite_boundary, static_potential, static_slope, surface_pairing,
)
from deuteron_wigner.quantum.modular_gauge_network import generators, su3


@pytest.mark.parametrize('size,ell', [(1e-6,.8), (.4,1.), (1.,.7), (4.,1.3)])
def test_interval_against_separation_quadrature(size, ell):
    independent = 2*quad(lambda z: (size-z)*np.exp(-(z/ell)**2), 0, size,
                         epsabs=1e-15, epsrel=2e-12)[0]
    assert interval_integral(size, ell) == pytest.approx(independent, rel=3e-12, abs=1e-26)


@pytest.mark.parametrize('a,b,ell', [(.4,.8,1.), (1.,2.,.7), (3.,4.,1.2)])
def test_rectangle_independent_surface_quadrature(a, b, ell):
    d = .6
    exact = .5*d*interval_integral(a,ell)*interval_integral(b,ell)
    assert quadratic_action([rectangle(a,b)],d,ell,22) == pytest.approx(exact, abs=3e-12)


@pytest.mark.parametrize('lengths,ell', [((.2,.3,.5),1.), ((1.,1.,1.),1.), ((.7,1.1,1.4),.6)])
def test_closed_surface_defect_is_finite_not_quadrature_error(lengths, ell):
    predicted = box_action_exact(lengths,.7,ell)
    assert predicted > 0
    for order in (10,16):
        assert quadratic_action(closed_box(lengths),.7,ell,order) == pytest.approx(predicted, abs=2e-10)
    # A true no-boundary Wilson identity equals 1, not exp(-positive action).
    assert np.exp(-predicted) < 1


def test_same_boundary_distant_closed_bubble_does_not_drop_out():
    sheet = [rectangle(1.,1.5)]
    bubble = [p.translated([0.,12.,12.,0.]) for p in closed_box([1.,1.,1.])]
    base = quadratic_action(sheet)
    together = quadratic_action(sheet+bubble)
    assert together-base == pytest.approx(box_action_exact([1.,1.,1.]), abs=3e-12)
    assert together > base+.5


def test_orientation_translation_and_euclidean_tensor_rotation():
    original = list(closed_box([.7,.9,1.2]))
    rng = np.random.default_rng(9)
    q, _ = np.linalg.qr(rng.normal(size=(4,4)))
    rotated = [Patch(q@p.origin+np.array([2.,-.5,.3,1.]),q@p.first,q@p.second)
               for p in original]
    value = quadratic_action(original,order=10)
    assert quadratic_action(rotated,order=10) == pytest.approx(value, abs=2e-12)
    assert quadratic_action([p.reversed() for p in original],order=10) == pytest.approx(value, abs=2e-12)


def test_separated_surfaces_cluster_despite_closed_surface_failure():
    one = [rectangle(.8,1.2)]
    for distance in (0.,1.,3.,5.):
        two = [one[0].translated([0.,distance,0.,0.])]
        cross = surface_pairing(one,two)
        assert cross == pytest.approx(2*quadratic_action(one)*np.exp(-distance**2), rel=3e-12)
        assert surface_pairing(one,[two[0].reversed()]) == pytest.approx(-cross, abs=2e-12)


def test_static_binding_slope_and_independent_finite_time_limit():
    d, ell, r, time = .7,.8,1.3,8.
    finite_time = quadratic_action([rectangle(r,time)],d,ell,36)/time
    expected_ratio = interval_integral(time,ell)/(time*np.sqrt(np.pi)*ell)
    assert finite_time/static_potential(r,d,ell) == pytest.approx(expected_ratio, abs=3e-11)
    h=1e-5
    derivative = (static_potential(r+h,d,ell)-static_potential(r-h,d,ell))/(2*h)
    assert derivative == pytest.approx(static_slope(r,d,ell), abs=3e-10)
    assert static_slope(5*ell,d,ell) == pytest.approx(np.pi*d*ell**2/2, abs=3e-11)
    assert static_potential(0,d,ell) == 0


def test_strength_switch_off_and_no_universal_bubble_normalization():
    assert quadratic_action(closed_box([1.,1.,1.]),strength=0.) == 0.
    small, large = [box_action_exact([l,l,l]) for l in (.5,2.)]
    assert large > 20*small > 0
    # A constant scalar normalization cannot set both arbitrary bubbles to 1.
    assert np.exp(-large+small) < .01


@pytest.mark.parametrize('seed', [2,17,41])
def test_nonabelian_two_surfaces_exact_same_boundary(seed):
    graph = cube_graph(seed)
    old = Path((0,1,2,3,0))
    for step in cube_rewrites():
        assert np.linalg.norm(graph.transport(old)
                              -graph.transport(step.based_face)@graph.transport(step.boundary)) < 4e-15
        old = step.boundary
    bottom = graph.transport(Path((0,1,2,3,0)))
    product = factored_loop(graph)
    assert np.linalg.norm(bottom-product) < 8e-15
    assert np.linalg.norm(product@bottom.conj().T-np.eye(3)) < 9e-15


def test_individual_face_traces_are_not_a_replacement_for_ordered_product():
    graph = cube_graph()
    steps = cube_rewrites()
    scalarized = np.prod([graph.loop(s.based_face) for s in steps])*graph.loop(steps[-1].boundary)
    assert abs(graph.loop(Path((0,1,2,3,0)))-scalarized) > .05


def test_local_gauge_covariance_and_full_baryon_color_contraction():
    graph = cube_graph()
    rng = np.random.default_rng(8)
    gauges = {v:su3(rng.normal(size=8)) for v in graph.nodes}
    changed = graph.transformed(gauges)
    direct = graph.transport(Path((0,1,2,3,0)))
    assert np.linalg.norm(factored_loop(changed)-gauges[0]@direct@gauges[0].conj().T) < 1e-14
    partners = [graph.transport(s.based_face) for s in cube_rewrites()[1:3]]
    b = baryon_wilson([direct]+partners)
    assert baryon_wilson([factored_loop(graph)]+partners) == pytest.approx(b, abs=3e-15)
    assert baryon_wilson([gauges[0]@u@gauges[0].conj().T for u in [direct]+partners]) == pytest.approx(b, abs=4e-15)
    assert baryon_wilson([direct]*3) == pytest.approx(1., abs=4e-15)
    assert baryon_wilson([direct,np.eye(3),np.eye(3)]) == pytest.approx(np.trace(direct)/3, abs=2e-15)


def test_all_connection_tangents_and_independent_link_perturbation():
    graph = cube_graph()
    bottom = Path((0,1,2,3,0))
    for edge in graph.links:
        for generator in generators():
            computed = factored_loop(graph,edge,generator)[1]
            assert np.linalg.norm(computed-graph.path_tangent(bottom,edge,generator)) < 1e-14
    edge, generator, h = (0,1), generators()[3], 1e-5
    displaced = []
    for sign in (1,-1):
        links = graph.links.copy()
        links[edge] = expm(sign*1j*h*generator)@links[edge]
        from deuteron_wigner.quantum.modular_gauge_network import GaugeGraph
        displaced.append(factored_loop(GaugeGraph(links)))
    assert np.linalg.norm((displaced[0]-displaced[1])/(2*h)
                          -factored_loop(graph,edge,generator)[1]) < 1e-9


def test_freezing_auxiliary_connectors_creates_spurious_source_response():
    graph = cube_graph()
    edge, generator = (0,4), generators()[1]
    exact = factored_loop(graph,edge,generator)[1]
    frozen = factored_loop(graph,edge,generator,freeze_connectors=True)[1]
    assert np.linalg.norm(exact) < 1e-14  # edge is absent from the bottom boundary
    assert np.linalg.norm(frozen) > .1


@pytest.mark.parametrize('call', [
    lambda: closed_box([1.,0.,1.]),
    lambda: Patch([0.]*4,[1.,0.,0.,0.],[2.,0.,0.,0.]),
    lambda: quadratic_action([rectangle(1.,1.)],strength=-1),
    lambda: quadratic_action([rectangle(1.,1.)],length=0),
    lambda: quadratic_action([rectangle(1.,1.)],order=1),
    lambda: rewrite_boundary(Path((0,1,0)),0,(0,2,3)),
    lambda: baryon_wilson([np.eye(3)]),
])
def test_invalid_scope_rejected(call):
    with pytest.raises(ValueError):
        call()

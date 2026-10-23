import numpy as np
import pytest

from deuteron_wigner.quantum.modular_gauge_network import (
    GaugeGraph, Path, SharedLoopKernel, adjoint, baryon_color, commutator_loop,
    gauge_divergence, generators, su3, normalized_real_characters,
    positive_character_kernel, dress_spatial_potential,
)


def fixture(seed=21):
    rng = np.random.default_rng(seed)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5), (5, 2))
    graph = GaugeGraph({e: su3(rng.normal(size=8)*.8) for e in edges})
    loops = (Path((1, 2, 3, 0, 1)), Path((1, 4, 5, 2, 1)))
    full = loops+(commutator_loop(*loops),)
    kernel = SharedLoopKernel([.6, 1.2, -1.8, 2.4], [.1, -.2, .3, -.1],
                             rng.normal(size=(4, 4))*.5, [.2, -.1, .05, .3], [.2, -.3, .4, .1])
    return graph, full, kernel


def test_su3_and_adjoint():
    t = generators()
    assert np.max(abs(2*np.einsum("aij,bji->ab", t, t)-np.eye(8))) < 1e-14
    u = su3(np.arange(8)*.12)
    a = adjoint(u)
    assert np.linalg.norm(a.T@a-np.eye(8)) < 1e-13
    assert np.max(abs(u@t@u.conj().T-np.einsum("ab,aij->bij", a, t))) < 1e-13


@pytest.mark.parametrize("seed", range(4))
def test_local_gauge_path_current_and_baryon_covariance(seed):
    graph, loops, kernel = fixture(seed)
    rng = np.random.default_rng(seed+45)
    gauges = {n: su3(rng.normal(size=8)) for n in graph.nodes}
    changed = graph.transformed(gauges)
    path = Path((0, 1, 4, 5), "FUTURE")
    assert np.max(abs(changed.transport(path)-gauges[0]@graph.transport(path)@gauges[5].conj().T)) < 2e-14
    assert np.max(abs(graph.transport(path.reversed())-graph.transport(path).conj().T)) < 1e-14
    assert path.reversed().process == "FUTURE"
    assert kernel.value(changed, loops) == pytest.approx(kernel.value(graph, loops), abs=1e-14)
    old, new = kernel.current(graph, loops), kernel.current(changed, loops)
    for i, (a, _) in enumerate(graph.links):
        assert np.max(abs(new[i]-adjoint(gauges[a])@old[i])) < 1e-14
    assert max(np.linalg.norm(x) for x in gauge_divergence(graph, old).values()) < 1e-14
    paths = [Path((0, 1, 2)), Path((0, 3)), Path((0, 1, 4, 5))]
    q = rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3))
    transformed = [gauges[p.vertices[-1]]@column for p, column in zip(paths, q)]
    assert abs(baryon_color(graph, paths, q)-baryon_color(changed, paths, transformed)) < 1e-13


@pytest.mark.parametrize("epsilon", [1e-4, 1e-5, 1e-6])
def test_complete_shared_current_by_independent_link_differences(epsilon):
    graph, loops, kernel = fixture()
    current = kernel.current(graph, loops)
    for i, edge in enumerate(graph.links):
        for a in range(8):
            plus, minus = graph.links.copy(), graph.links.copy()
            direction = np.eye(8)[a]*epsilon
            plus[edge] = su3(direction)@graph.links[edge]
            minus[edge] = su3(-direction)@graph.links[edge]
            numerical = (kernel.value(GaugeGraph(plus), loops)-kernel.value(GaugeGraph(minus), loops))/(2*epsilon)
            assert numerical == pytest.approx(current[i, a], abs=5e-9)


def test_tree_gauge_retains_holonomy_but_erasure_does_not():
    graph, loops, kernel = fixture()
    tree = ((0, 1), (1, 2), (2, 3), (1, 4), (4, 5))
    fixed, _ = graph.tree_gauge(tree, 0)
    assert max(np.linalg.norm(fixed.link(*e)-np.eye(3)) for e in tree) < 2e-14
    assert max(abs(graph.loop(p)-fixed.loop(p)) for p in loops) < 1e-14
    assert kernel.value(graph, loops) == pytest.approx(kernel.value(fixed, loops), abs=1e-14)
    erased = GaugeGraph({e: np.eye(3) for e in graph.links})
    assert max(abs(erased.loop(p)-graph.loop(p)) for p in loops) > .1
    assert abs(kernel.value(erased, loops)-kernel.value(graph, loops)) > .001
    assert np.linalg.norm(kernel.current(erased, loops)) < 1e-14


def test_matrix_loop_orientation_is_not_determined_by_individual_traces():
    # Both elementary holonomies have identical spectra; relative orientation differs.
    graph, loops, _ = fixture()
    identity = {e: np.eye(3) for e in graph.links}
    a = su3(np.array([0, 0, 0, 0, 0, 0, .8, .3]))
    b = su3(np.array([0, 0, 0, 0, 0, 0, -.4, .9]))
    rotation = su3([1.3, .4, -.7, .8, .2, .1, 0., 0.])
    first, second = identity.copy(), identity.copy()
    first[3, 0], first[5, 2] = a, b
    second[3, 0], second[5, 2] = a, rotation@b@rotation.conj().T
    left, right = GaugeGraph(first), GaugeGraph(second)
    assert all(abs(left.loop(p)-right.loop(p)) < 1e-14 for p in loops[:2])
    assert abs(left.loop(loops[2])-right.loop(loops[2])) > .001


def test_value_zero_cannot_qualify_a_wrong_current():
    graph, loops, kernel = fixture()
    full = kernel.current(graph, loops)
    omitted = full.copy()
    omitted[0] = 0
    assert max(np.linalg.norm(x) for x in gauge_divergence(graph, omitted).values()) > .001


def test_path_validation_and_order():
    graph, loops, _ = fixture()
    assert abs(graph.loop(loops[0].reversed())-graph.loop(loops[0]).conjugate()) < 1e-14
    with pytest.raises(ValueError):
        graph.loop(Path((0, 1)))
    with pytest.raises(ValueError):
        graph.transport(Path((0, 5)))
    with pytest.raises(ValueError):
        graph.tree_gauge(((0, 1),)*5, 0)
    with pytest.raises(ValueError):
        GaugeGraph({(0, 1): 2*np.eye(3)})


@pytest.mark.parametrize("seed,size", [(81, 40), (192, 32), (293, 64)])
def test_positive_time_kernel_with_unrestricted_real_spatial_neural_potential(seed, size):
    from scripts.evaluate_modular_gauge_architecture import kernel
    rng = np.random.default_rng(seed)
    u = np.array([su3(rng.normal(size=8)) for _ in range(size)])
    net = kernel()
    relative = u[:, None]@u[None, :].conj().swapaxes(-1, -2)
    z = 1-np.trace(relative, axis1=-2, axis2=-1).real/3
    unrestricted_time = np.exp(-net.features(z)@net.coefficients)
    assert np.linalg.eigvalsh(unrestricted_time)[0] < -.01
    positive = positive_character_kernel(u, [.5, .2, .1])
    assert np.linalg.eigvalsh(positive)[0] > -1e-12
    potential = net.features(1-np.trace(u, axis1=-2, axis2=-1).real/3)@net.coefficients
    dressed, offset = dress_spatial_potential(positive, potential, .2)
    assert np.linalg.eigvalsh(dressed)[0] > -1e-12
    d = np.exp(-.2*potential/2)
    assert np.max(abs(d[:, None]*positive*d[None, :]-np.exp(-.2*offset)*dressed)) < 1e-13
    g, h = su3(rng.normal(size=8)), su3(rng.normal(size=8))
    assert np.max(abs(positive_character_kernel(g@u@h.conj().T, [.5, .2, .1])-positive)) < 2e-14


def test_character_identities_and_signed_weight_negative():
    assert np.max(abs(normalized_real_characters(np.eye(3))-1)) < 1e-14
    u = np.array([np.eye(3), su3(np.arange(8)*.1)])
    assert np.max(abs(positive_character_kernel(u, [0, 0, 0])-1)) == 0
    with pytest.raises(ValueError):
        positive_character_kernel(u, [-.1, .2, .3])


def test_shared_neural_inference_reserved_group_configurations():
    from scripts.evaluate_modular_gauge_architecture import inference
    result = inference()
    assert not result['physical_data_used'] and not result['hidden_layers_trained']
    first, joint = result['results'].values()
    assert max(first['exact_coefficient_error'], joint['exact_coefficient_error']) < 1e-10
    assert joint['reserved_current_RMSE'] < first['reserved_current_RMSE']

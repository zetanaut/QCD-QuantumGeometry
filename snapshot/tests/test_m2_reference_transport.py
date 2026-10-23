import numpy as np
import pytest
from scipy import linalg

from deuteron_wigner.quantum.m2_matched_h2_reduction import FiniteParentReduction
from deuteron_wigner.quantum.m2_reference_transport import FiniteReferenceTransport
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source
from deuteron_wigner.quantum.m2_excluded_fock_matching import ExcludedFockReference


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(8601)
    parents = []
    for n in (7, 9):
        b = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n))
        h = (b+b.conj().T)/12 + np.diag([1., 1.3, 1.7]+[5.+i for i in range(n-3)])
        parents.append(FiniteParentReduction(h, np.eye(n)[:, :3], scope="independent complex finite-reference test"))
    unitary = linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))[0]
    return parents, unitary, FiniteReferenceTransport(*parents, map_to_reference=unitary)


@pytest.mark.parametrize("energy", [.7, 1.2, 2.])
def test_exact_hamiltonian_and_metric_transport_in_rotated_coordinates(toy, energy):
    (low, high), t, flow = toy
    np.testing.assert_allclose(low.effective_hamiltonian(energy)+flow.hamiltonian_counterterm(energy),
                               t.conj().T@high.effective_hamiltonian(energy)@t, atol=2e-15)
    np.testing.assert_allclose(low.norm_metric(energy)+flow.norm_counterterm(energy),
                               t.conj().T@high.norm_metric(energy)@t, atol=3e-15)
    step = 1e-5
    derivative = (flow.hamiltonian_counterterm(energy+step)-flow.hamiltonian_counterterm(energy-step))/(2*step)
    np.testing.assert_allclose(derivative, flow.slope_counterterm(energy), atol=4e-11)


@pytest.mark.parametrize("out,inc", [(.7, 1.2), (1.2, .7), (1.2, 1.2)])
def test_two_energy_current_transport_and_reversal(toy, out, inc):
    (low, high), t, flow = toy
    rng = np.random.default_rng(724)
    ops = [rng.normal(size=r.h.shape)+1j*rng.normal(size=r.h.shape) for r in (low, high)]
    delta = flow.operator_counterterm(*ops, out, inc)
    actual = low.matched_operator(ops[0], out, inc)+delta
    expected = t.conj().T@high.matched_operator(ops[1], out, inc)@t
    np.testing.assert_allclose(actual, expected, atol=2e-15)
    reverse = flow.operator_counterterm(*(o.conj().T for o in ops), inc, out)
    np.testing.assert_allclose(reverse, delta.conj().T, atol=2e-15)


def test_charge_counterterm_is_metric_counterterm_not_zero(toy):
    (low, high), _, flow = toy
    delta = flow.operator_counterterm(np.eye(len(low.h)), np.eye(len(high.h)), 1.2, 1.2)
    np.testing.assert_allclose(delta, flow.norm_counterterm(1.2), atol=2e-15)
    assert np.linalg.norm(delta) > .001


def test_independent_operator_energy_derivatives(toy):
    (low, high), _, flow = toy
    rng = np.random.default_rng(201)
    ops = [rng.normal(size=r.h.shape)+1j*rng.normal(size=r.h.shape) for r in (low, high)]
    anchor, step = 1.2, 1e-5
    value, left, right = flow.operator_jet_counterterm(*ops, anchor)
    np.testing.assert_allclose(value, flow.operator_counterterm(*ops, anchor, anchor), atol=2e-15)
    for axis, expected in ((0, left), (1, right)):
        a = [anchor, anchor]; b = a.copy(); a[axis] += step; b[axis] -= step
        fd = (flow.operator_counterterm(*ops, *a)-flow.operator_counterterm(*ops, *b))/(2*step)
        np.testing.assert_allclose(fd, expected, atol=9e-11)
    assert np.linalg.norm(left-right) > .001


@pytest.mark.parametrize("shift", [.02, .1, -.1])
def test_local_counterterm_remainder_and_generalized_schur_complement(toy, shift):
    (low, _), _, flow = toy
    anchor, energy = 1.2, 1.2+shift
    ct = flow.local_counterterm(anchor)
    error = np.linalg.norm(flow.hamiltonian_counterterm(energy)-ct.at(energy), 2)
    assert error <= flow.hamiltonian_remainder_bound(energy, anchor)
    a, b = flow.generalized_lower_parent(anchor)
    p, q = low.p, low.q
    pencil = a-energy*b
    schur = p.conj().T@pencil@p - p.conj().T@pencil@q@linalg.solve(q.conj().T@pencil@q, q.conj().T@pencil@p)
    expected = low.effective_hamiltonian(energy)+ct.at(energy)-energy*np.eye(3)
    np.testing.assert_allclose(schur, expected, atol=3e-15)


def test_coordinate_map_required_and_not_guessed(toy):
    parents, _, _ = toy
    with pytest.raises(TypeError):
        FiniteReferenceTransport(*parents)
    with pytest.raises(ValueError):
        FiniteReferenceTransport(*parents, map_to_reference=np.ones((3, 3)))


@pytest.fixture(scope="module")
def actual():
    pilot = build_coupled_pilot(total_k2=9, cutoff=0, target="PROTON", jz_twice=1,
                               b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)
    parents = [ExcludedFockReference(pilot, [build_excluded_source(pilot, sector, out_cutoff=cut)
               for sector in ("qqqgg", "qqqq-qbar")]) for cut in (0, 1)]
    return pilot, parents, FiniteReferenceTransport(*(p.reduction for p in parents),
                                                    map_to_reference=np.eye(len(pilot.matrix)))


def test_actual_excluded_shell_flow_has_attractive_h_and_positive_added_norm(actual):
    _, (low, high), flow = actual
    e = 1.4
    dh, dn = flow.hamiltonian_counterterm(e), flow.norm_counterterm(e)
    assert np.linalg.eigvalsh(dh)[-1] < 1e-13
    assert np.linalg.eigvalsh(dn)[0] > -1e-13
    assert np.linalg.norm(dh) > 1e-4
    assert np.linalg.norm(dn) > 1e-5
    a, b = flow.generalized_lower_parent(e)
    transported = linalg.eigvalsh(a, b)[0]
    reference = linalg.eigvalsh(high.matrix)[0]
    assert abs(transported-reference) < 3e-10
    assert abs(linalg.eigvalsh(low.matrix)[0]-reference) > 5e-7


@pytest.mark.parametrize("charge", ["baryon", "electromagnetic", "up_number", "down_number"])
def test_actual_current_and_metric_corrections_travel_together(actual, charge):
    _, (low, high), flow = actual
    e = 1.4
    zero = [p.density((0., 0.), charge=charge) for p in (low, high)]
    charge_value = dict(baryon=1., electromagnetic=1., up_number=2., down_number=1.)[charge]
    np.testing.assert_allclose(flow.operator_counterterm(*zero, e, e),
                               charge_value*flow.norm_counterterm(e), atol=2e-14)
    ops = [p.density((.12, .16), charge=charge) for p in (low, high)]
    correction = flow.operator_counterterm(*ops, e, e)
    assert np.linalg.norm(correction) > 1e-5
    np.testing.assert_allclose(low.reduction.matched_operator(ops[0], e, e)+correction,
                               high.reduction.matched_operator(ops[1], e, e), atol=3e-15)


def test_actual_charge_is_wrong_if_only_energy_and_metric_are_transported(actual):
    _, (low, high), flow = actual
    state = high.reduction.lowest_state()
    e, v = state.energy_GeV2, state.retained
    wrong = np.vdot(v, low.reduction.norm_metric(e)@v).real
    correction = np.vdot(v, flow.norm_counterterm(e)@v).real
    assert abs(wrong-1) > 1e-8
    assert wrong+correction == pytest.approx(1., abs=2e-14)


def test_actual_two_energy_flow_current_transport_with_boundary_terms(actual):
    _, (low, high), flow = actual
    out, inc = 1.47, 1.31
    delta = np.array([.12, .16])
    currents = [r.flow_current(delta, charge="electromagnetic") for r in (low, high)]
    matched = []
    for ref, (rho, jx, jy) in zip((low, high), currents):
        r = ref.reduction
        wf, wi = r.wave_operator(out), r.wave_operator(inc)
        density = r.matched_operator(rho, out, inc)
        current = r.matched_operator(delta[0]*jx+delta[1]*jy, out, inc)
        boundary = ((r.effective_hamiltonian(out)-out*np.eye(len(density)))@r.p.conj().T@rho@wi
                    -wf.conj().T@rho@r.p@(r.effective_hamiltonian(inc)-inc*np.eye(len(density))))
        np.testing.assert_allclose(current, (out-inc)*density+boundary, atol=2e-14)
        matched.append(current)
    correction = flow.operator_counterterm(*(delta[0]*c[1]+delta[1]*c[2] for c in currents), out, inc)
    np.testing.assert_allclose(matched[0]+correction, matched[1], atol=2e-14)

"""Independent Lorentz/source, cell quadrature and H2 leakage checks."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from scipy.special import roots_legendre

from deuteron_wigner.bridge.modes.core import GAMMA, polarization, polarization_cartesian, ho_momentum
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import jacobi_frame, oscillator_shell
from deuteron_wigner.quantum.m2_coupled_pair_action import circular_to_c45
from deuteron_wigner.quantum.m2_free_spin_generators import (
    SIGMA, free_generator, free_jacobi_tangent, free_pauli_lubanski,
    longitudinal_cell_operators, projected_commutator, free_cell_galerkin,
    moving_frame_connection,
)
from deuteron_wigner.quantum.m2_matched_h2_reduction import FiniteParentReduction


def lorentz_pair(name):
    """Independent active Cartesian four-vector and Dirac transformations."""
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    basis = {}
    for axis, label in enumerate("xyz"):
        rotation = np.zeros((4, 4))
        rotation[1:, 1:] = np.column_stack([np.cross(np.eye(3)[axis], e) for e in np.eye(3)])
        sigma = np.kron(np.eye(2), pauli[axis])
        boost = np.zeros((4, 4))
        boost[0, axis+1] = boost[axis+1, 0] = 1
        basis["J"+label] = (rotation, -0.5j*sigma)
        basis["B"+label] = (boost, 0.5*GAMMA[0]@GAMMA[axis+1])
    if name in basis:
        return basis[name]
    a, b, sign = {"F1": ("Bx", "Jy", -1), "F2": ("By", "Jx", 1),
                  "E1": ("Bx", "Jy", 1), "E2": ("By", "Jx", -1)}[name]
    return tuple((x+sign*y)/np.sqrt(2) for x, y in zip(basis[a], basis[b]))


def cartesian(p, m):
    pm = (m*m+p[1]**2+p[2]**2)/(2*p[0])
    return np.array([(p[0]+pm)/np.sqrt(2), p[1], p[2], (p[0]-pm)/np.sqrt(2)])


def moved(p, m, name, t):
    q = expm(t*lorentz_pair(name)[0])@cartesian(p, m)
    return np.array([(q[0]+q[3])/np.sqrt(2), q[1], q[2]])


def source_wigner(p, mass, name, t, particle):
    q = moved(p, mass, name, t)
    v, s = lorentz_pair(name)
    if particle in ("quark", "antiquark"):
        spinor = (lambda k, h: lf_spinor(*k, mass, h)) if particle == "quark" else (
            lambda k, h: antiquark_spinor(*k, mass, h))
        u = np.column_stack([spinor(p, h) for h in (-1, 1)])
        out = np.column_stack([spinor(q, h) for h in (-1, 1)])
        transformed = expm(t*s)@u
        d = out.conj().T@transformed/(2*cartesian(q, mass)[0])
        assert_allclose(out@d, transformed, atol=3e-14)
        # The v field multiplies d^dagger: its Wigner matrix is conjugated
        # when read as an active matrix on physical antiparticle states.
        return d.conj() if particle == "antiquark" else d
    u = np.column_stack([polarization_cartesian(polarization(*p, h)) for h in (-1, 1)])
    out = np.column_stack([polarization_cartesian(polarization(*q, h)) for h in (-1, 1)])
    transformed = expm(t*v)@u
    # Restore A+=0 with a pure-gauge longitudinal polarization; no phase fit.
    plus = (transformed[0]+transformed[3])/np.sqrt(2)
    transformed -= cartesian(q, 0)[:, None]*plus[None, :]/q[0]
    d = -out.conj().T@np.diag([1, -1, -1, -1])@transformed
    assert_allclose(out@d, transformed, atol=3e-14)
    return d


@pytest.mark.parametrize("name", ["F1", "F2", "E1", "E2", "Jx", "Jy", "Jz", "Bx", "By", "Bz"])
@pytest.mark.parametrize("particle,mass", [("quark", .31), ("quark", 0.), ("antiquark", .31), ("antiquark", 0.), ("gluon", 0.)])
def test_source_spinor_and_gauge_restored_gluon_lorentz_action(name, particle, mass):
    p = np.array([1.37, .23, -.41])
    g = free_generator(p, mass, name, particle=particle)
    dt = 2e-6
    derivative = (source_wigner(p, mass, name, dt, particle)
                  - source_wigner(p, mass, name, -dt, particle))/(2*dt)
    assert_allclose(1j*derivative, g.spin, atol=2e-10)
    assert_allclose((moved(p, mass, name, dt)-moved(p, mass, name, -dt))/(2*dt), g.velocity, atol=2e-10)
    d = source_wigner(p, mass, name, .17, particle)
    assert_allclose(d.conj().T@d, np.eye(2), atol=4e-14)
    divergence = 0
    for k in range(3):
        step = np.eye(3)[k]*dt
        divergence += (free_generator(p+step, mass, name, particle=particle).velocity[k]
                       - free_generator(p-step, mass, name, particle=particle).velocity[k])/(2*dt)
    assert_allclose(divergence, g.divergence, atol=1e-10)
    # Invariant-measure divergence is zero; flat amplitudes need div/2.
    assert_allclose(g.divergence-g.velocity[0]/p[0], 0, atol=1e-14)


@pytest.mark.parametrize("particle,mass", [("quark", .31), ("quark", 0.), ("antiquark", .31), ("antiquark", 0.), ("gluon", 0.)])
@pytest.mark.parametrize("a,b,c,sign", [
    ("F1", "F2", "Jz", 0), ("Jz", "F1", "F2", 1), ("Jz", "F2", "F1", -1),
    ("Jx", "Jy", "Jz", 1), ("Jy", "Jz", "Jx", 1), ("Jz", "Jx", "Jy", 1),
    ("Bx", "By", "Jz", -1), ("By", "Bz", "Jx", -1), ("Bz", "Bx", "Jy", -1),
    ("Jx", "By", "Bz", 1), ("Jy", "Bz", "Bx", 1), ("Jz", "Bx", "By", 1)])
def test_differential_lorentz_algebra(particle, mass, a, b, c, sign):
    p, dt = np.array([.87, -.29, .13]), 3e-6
    def at(name, q=p):
        return free_generator(q, mass, name, particle=particle)
    ga, gb, gc = at(a), at(b), at(c)
    def derivative(name, field):
        return np.array([(getattr(at(name, p+dt*e), field)-getattr(at(name, p-dt*e), field))/(2*dt) for e in np.eye(3)])
    bracket_v = ga.velocity@derivative(b, "velocity")-gb.velocity@derivative(a, "velocity")
    bracket_a = (-1j*np.einsum("k,kij->ij", ga.velocity, derivative(b, "spin"))
                 +1j*np.einsum("k,kij->ij", gb.velocity, derivative(a, "spin"))
                 +ga.spin@gb.spin-gb.spin@ga.spin)
    assert_allclose(bracket_v, -sign*gc.velocity, atol=2e-10)
    assert_allclose(bracket_a, 1j*sign*gc.spin, atol=2e-10)


@pytest.mark.parametrize("mass,particle", [(.31, "quark"), (0., "quark"), (.31, "antiquark"), (0., "antiquark"), (0., "gluon")])
def test_pauli_lubanski_orbital_cancellation_casimir_and_intrinsic_spin(mass, particle):
    p = np.array([.73, .27, -.16])
    cart = cartesian(p, mass)
    j = np.array([free_generator(p, mass, "J"+a, particle=particle).velocity for a in "xyz"])
    b = np.array([free_generator(p, mass, "B"+a, particle=particle).velocity for a in "xyz"])
    assert_allclose(cart[1:]@j, 0, atol=2e-16)
    cross = np.array([cart[2]*b[2]-cart[3]*b[1], cart[3]*b[0]-cart[1]*b[2], cart[1]*b[1]-cart[2]*b[0]])
    assert_allclose(cart[0]*j-cross, 0, atol=2e-16)
    # The half-density scalar terms cancel too.
    jd = np.array([free_generator(p, mass, "J"+a, particle=particle).divergence for a in "xyz"])
    bd = np.array([free_generator(p, mass, "B"+a, particle=particle).divergence for a in "xyz"])
    assert_allclose(cart[1:]@jd, 0, atol=2e-16)
    assert_allclose(cart[0]*jd-np.cross(cart[1:], bd), 0, atol=2e-16)
    wp, wm, wx, wy = free_pauli_lubanski(p, mass, particle=particle)
    pm = (mass*mass+p[1]**2+p[2]**2)/(2*p[0])
    assert_allclose(p[0]*wm+pm*wp-p[1]*wx-p[2]*wy, 0, atol=2e-16)
    expected = -3*mass*mass/4 if particle != "gluon" else 0.
    assert_allclose(wp@wm+wm@wp-wx@wx-wy@wy, expected*np.eye(2), atol=2e-16)
    if mass:
        spin = np.array([(wx-p[1]*wp/p[0])/mass, (wy-p[2]*wp/p[0])/mass, wp/p[0]])
        assert_allclose(spin, SIGMA/2, atol=5e-16)
        assert_allclose(spin[0]@spin[1]-spin[1]@spin[0], 1j*spin[2], atol=5e-16)
    else:
        h = SIGMA[2]*(.5 if particle != "gluon" else 1.)
        assert_allclose(np.array([wp, wm, wx, wy]), np.array([p[0], pm, p[1], p[2]])[:, None, None]*h, atol=3e-16)


@pytest.mark.parametrize("modes", [(1, 3, 7, 9), (6, 2, 4, 10)])
def test_cell_coordinate_independent_quadrature_boundary_and_localized_state(modes):
    length = 2.7
    x, p, boundary = longitudinal_cell_operators(modes, length)
    t, w = roots_legendre(96)
    phi = np.exp(1j*np.pi*np.outer(t, np.array(modes)/2))/np.sqrt(2*length)
    exact = phi.conj().T@((length*length*w*t)[:, None]*phi)
    assert_allclose(x, exact, atol=2e-14)
    assert_allclose(x, x.conj().T, atol=1e-15)
    assert_allclose(x@p-p@x, 1j*(np.eye(len(modes))-boundary), atol=2e-15)
    assert_allclose(np.linalg.eigvalsh(boundary), [0, 0, 0, len(modes)], atol=2e-15)
    face = np.exp(-.5j*np.pi*np.array(modes))
    assert_allclose(boundary, np.outer(face, face.conj()), atol=2e-15)
    state = np.array([1., 2j, -.3, .8j])
    state -= face*np.vdot(face, state)/len(modes)
    state /= np.linalg.norm(state)
    assert_allclose(np.vdot(state, (x@p-p@x)@state), 1j, atol=1e-15)
    assert abs(np.trace(x@p-p@x)) < 1e-15  # never assert full finite CCR=iI


@pytest.mark.parametrize("modes", [(1, 3, 5), (1, 1, 3, 4)])
@pytest.mark.parametrize("axis", [0, 1])
def test_source_jacobi_tangent_includes_moving_longitudinal_frame(modes, axis):
    frame = jacobi_frame(modes)
    q = np.arange(len(modes)*2).reshape(-1, 2)*.031-.07
    q[-1] = (.12, -.18)
    masses = np.array([.3]*3+([0.] if len(modes) == 4 else []))
    pp = 2.1
    dp, dx, dq = free_jacobi_tangent(modes, q, pp, masses, axis)
    physical = frame.from_jacobi(q)
    def flow(t):
        new = np.array([moved(np.array([pp*x, *pt]), m, "F"+str(axis+1), t)
                        for x, pt, m in zip(frame.fractions, physical, masses)])
        total = new[:, 0].sum()
        x = new[:, 0]/total
        # Independent definition Q=(XB pA-XA pB)/sqrt(XA XB (XA+XB)).
        rows = []
        def visit(tree):
            if not isinstance(tree, tuple):
                return [tree]
            a, b = visit(tree[0]), visit(tree[1])
            xa, xb = x[a].sum(), x[b].sum()
            rows.append((xb*new[a, 1:].sum(axis=0)-xa*new[b, 1:].sum(axis=0))/np.sqrt(xa*xb*(xa+xb)))
            return a+b
        visit(frame.tree)
        return total, x, np.array(rows+[new[:, 1:].sum(axis=0)])
    dt = 2e-6
    numeric = [(a-b)/(2*dt) for a, b in zip(flow(dt), flow(-dt))]
    for exact, approx in zip((dp, dx, dq), numeric):
        assert_allclose(exact, approx, atol=3e-10)
    assert_allclose(dx.sum(), 0, atol=1e-16)
    assert np.linalg.norm(dx) > .005  # not tangent to a fixed grid point


def test_projection_distinguishes_complete_and_cut_spin_multiplets():
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])/np.sqrt(2)
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])/np.sqrt(2)
    full = projected_commutator(np.eye(3), sx, sy)
    assert_allclose(full["commutator"], 1j*np.diag([1, 0, -1]), atol=3e-16)
    cut = projected_commutator(np.eye(3)[:, [0, 2]], sx, sy)
    assert_allclose(cut["commutator"], 0)
    assert_allclose(cut["leakage"], -1j*np.diag([1, -1]), atol=3e-16)
    assert cut["identity_residual"] < 1e-15
    assert cut["first_leakage_norm"] > .9


@pytest.mark.parametrize("particle,mass,modes", [("quark", .3, (1, 3, 5)), ("antiquark", .3, (1, 3, 5)), ("gluon", 0., (2, 4, 6))])
def test_source_cell_galerkin_boundary_and_exact_shell_leakage(particle, mass, modes):
    low = free_cell_galerkin(modes, 2.7, 1, .4, mass, particle=particle)
    high = free_cell_galerkin(modes, 2.7, 4, .4, mass, particle=particle)
    indices = [((k*15+n)*2+s) for k in range(3) for n in range(3) for s in range(2)]
    u = np.eye(90)[:, indices]
    for name in ("F1", "F2", "Pplus", "Pminus", "Px", "Py", "Jz"):
        assert_allclose(low[name], low[name].conj().T, atol=5e-16)
        assert_allclose(u.conj().T@high[name]@u, low[name], atol=8e-16)
    for axis in ("x", "y"):
        f = low["F1" if axis == "x" else "F2"]
        p = low["Pplus"]
        assert_allclose(f@p-p@f, -1j*low["P"+axis]+1j*low["cell_boundary_"+axis], atol=5e-16)
    # Exact residual split: parent cell defect plus finite orbital leakage.
    report = projected_commutator(u, high["F1"], high["F2"])
    assert report["identity_residual"] < 3e-15
    assert np.linalg.norm(report["parent_commutator"]) > .1
    assert np.linalg.norm(report["leakage"]) > .1
    assert_allclose(low["Jz"]@low["F1"]-low["F1"]@low["Jz"], 1j*low["F2"], atol=8e-16)


def test_projected_free_generator_against_source_ho_quadrature():
    from scipy.special import roots_hermite
    b, m, pp, dt = .37, .3, .91, 2e-6
    shell = oscillator_shell(1, 2)
    nodes, weights = roots_hermite(16)
    x, y = np.meshgrid(b*nodes, b*nodes, indexing="ij")
    weight = (b*b*np.outer(weights, weights)*np.exp((x*x+y*y)/(b*b))).ravel()
    def basis(x, y):
        return np.array([phase*ho_momentum(n, angular, x, y, b)/(2*np.pi)
                         for n, angular, phase in (circular_to_c45(*state) for state in shell)]).reshape(len(shell), -1).T
    phi = basis(x, y)
    assert_allclose(phi.conj().T@(weight[:, None]*phi), np.eye(len(shell)), atol=3e-15)
    dx = (basis(x+dt, y)-basis(x-dt, y))/(2*dt)
    pm = (m*m+x*x+y*y).ravel()/(2*pp)
    orbital = phi.conj().T@(weight[:, None]*(-1j*pm[:, None]*dx-.5j*x.ravel()[:, None]*phi/pp))
    ky = phi.conj().T@((weight*y.ravel())[:, None]*phi)
    expected = np.kron(orbital, np.eye(2))-np.kron(ky, SIGMA[2])/(2*pp)-np.kron(np.eye(len(shell)), m*SIGMA[1])/(2*pp)
    # One mode: <X>=0, so this quadrature is precisely the transverse block.
    actual = free_cell_galerkin((1,), np.pi/(2*pp), 2, b, m)["F1"]
    assert_allclose(actual, expected, atol=2e-10)


def test_h2_same_norm_transport_and_raw_metric_product():
    # Explicit known spin-half tensor-product reference, NOT the QCD parent.
    h = np.kron(np.array([[1., .4], [.4, 3.]]), np.eye(2))
    p = np.eye(4)[:, :2]
    model = FiniteParentReduction(h, p, scope="free-spin algebra transport reference")
    energy = .8
    u = model.linearized_model(energy).parent_isometry
    a, b = (np.kron(np.eye(2), s/2) for s in SIGMA[:2])
    result = projected_commutator(u, a, b)
    assert_allclose(result["commutator"], 1j*SIGMA[2]/2, atol=4e-16)
    assert result["first_leakage_norm"] < 1e-15
    am, bm = [model.matched_operator(o, energy, energy) for o in (a, b)]
    raw = model.matched_operator(a@b-b@a, energy, energy)
    metric = model.norm_metric(energy)
    assert_allclose(am@np.linalg.solve(metric, bm)-bm@np.linalg.solve(metric, am), raw, atol=4e-16)
    assert np.linalg.norm(am@bm-bm@am-raw) > .02
    # A generic supplied Hermitian parent need not preserve the H2 image.
    rng = np.random.default_rng(23)
    a, b = [(lambda z: z+z.conj().T)(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))) for _ in range(2)]
    result = projected_commutator(u, a, b)
    assert result["identity_residual"] < 2e-14
    assert np.linalg.norm(result["leakage"]) > 1


def test_moving_h2_frame_product_rule_and_internal_basis_gauge():
    h = np.kron(np.array([[1., .4], [.4, 3.]]), np.eye(2))
    model = FiniteParentReduction(h, np.eye(4)[:, :2], scope="moving-frame reference")
    base = model.linearized_model(.8).parent_isometry
    parent_motion = np.kron(SIGMA[0], SIGMA[2])*.3
    a = np.kron(np.eye(2), SIGMA[0]/2)
    t, dt = .19, 2e-6
    frame = lambda t: expm(-1j*t*parent_motion)@base
    u, du = frame(t), -1j*parent_motion@frame(t)
    eff, tail = moving_frame_connection(u, du, a)
    assert_allclose(eff, eff.conj().T, atol=3e-16)
    f, df = np.array([.4+.2j, -.3j]), np.array([.1j, -.2])
    full = -1j*(frame(t+dt)@(f+dt*df)-frame(t-dt)@(f-dt*df))/(2*dt)+a@u@f
    assert_allclose(full, u@(-1j*df+eff@f)+tail@f, atol=3e-11)
    assert np.linalg.norm(eff-u.conj().T@a@u) > .01
    # A momentum-dependent retained basis adds its OWN connection, not a fit.
    r = expm(-1j*t*SIGMA[1]/2)
    dr = -1j*SIGMA[1]@r/2
    rotated, _ = moving_frame_connection(u@r, du@r+u@dr, a)
    assert_allclose(rotated, r.conj().T@eff@r-1j*r.conj().T@dr, atol=4e-16)
    with pytest.raises(ValueError):
        moving_frame_connection(u, u, a)


def test_saved_free_spin_report_is_replayable():
    import json
    from pathlib import Path
    from scripts.analyze_m2_free_spin_generators import analyze
    saved = json.loads((Path(__file__).resolve().parents[1]/"docs/next_level/m2_free_spin_generators_results.json").read_text())
    current = analyze()
    assert not saved["interacting_spin_qualified"]
    for particle in ("quark", "gluon"):
        for key, value in saved["cases"][particle].items():
            assert_allclose(current["cases"][particle][key], value, rtol=2e-13, atol=1e-14)
    assert_allclose(current["qqq_K2_9_zero_total_transverse_F1_tangent"]["fraction_derivative"],
                    saved["qqq_K2_9_zero_total_transverse_F1_tangent"]["fraction_derivative"], atol=1e-15)


@pytest.mark.parametrize("modes", [(), (1, 2), (1, 1), (0, 2), (1.5, 3), (True, 3)])
def test_invalid_cell_domains_rejected(modes):
    with pytest.raises(ValueError):
        longitudinal_cell_operators(modes, 1.)


def test_no_massive_two_helicity_gluon_or_arbitrary_generator():
    with pytest.raises(ValueError):
        free_generator([1, 0, 0], .1, "F1", particle="gluon")
    with pytest.raises(ValueError):
        free_generator([1, 0, 0], .1, "Sx")
    with pytest.raises(ValueError):
        projected_commutator(np.ones((3, 2)), np.eye(3), np.eye(3))
    g = free_generator([1, .2, .3], .3, "F1")
    assert_allclose(g.apply(np.ones(2), np.zeros((3, 2)))-g.apply(np.ones(2), np.zeros((3, 2)), flat_measure=False), -.1j*np.ones(2))

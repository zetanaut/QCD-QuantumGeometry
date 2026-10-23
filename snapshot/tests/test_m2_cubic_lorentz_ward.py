"""Independent field-density, finite-Lorentz and source-vertex comparisons."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from deuteron_wigner.bridge.modes.core import GAMMA, GAMMA_PLUS, GAMMA_MINUS, polarization
from deuteron_wigner.quantum.m2_canonical_emission import emission_momenta, emission_mass_kernel_per_gs
from deuteron_wigner.quantum.m2_constrained_fermion_current import GOOD_AXES, bad_field_kernel
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import emission_routes
from deuteron_wigner.quantum.m2_free_spin_generators import free_generator
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianClusterRegulator
from deuteron_wigner.quantum.m2_gaussian_reference_shape import GaussianReferenceShape
from deuteron_wigner.quantum.m2_cubic_lorentz_ward import (
    free_fermion_noether_spin, free_gluon_noether_spin, fermion_noether_one_gluon,
    emission_minus_vertex, emission_lorentz_contact, emission_regular_ward,
    cluster_quadratic_lorentz, source_regulator_lorentz,
    retained_fermion_spin_orbital, apply_retained_fermion_spin,
)


def vector_generator(name):
    basis = {}
    for n, label in enumerate("xyz"):
        r = np.zeros((4, 4))
        r[1:, 1:] = np.column_stack([np.cross(np.eye(3)[n], e) for e in np.eye(3)])
        b = np.zeros((4, 4))
        b[0, n+1] = b[n+1, 0] = 1.
        basis["J"+label], basis["B"+label] = r, b
    if name in basis:
        return basis[name]
    a, b, s = {"F1": ("Bx", "Jy", -1), "F2": ("By", "Jx", 1),
               "E1": ("Bx", "Jy", 1), "E2": ("By", "Jx", -1)}[name]
    return (basis[a]+s*basis[b])/np.sqrt(2)


def moved(p, mass, name, t):
    pm = (mass*mass+p[1:]@p[1:])/(2*p[0])
    four = np.array([(p[0]+pm)/np.sqrt(2), p[1], p[2], (p[0]-pm)/np.sqrt(2)])
    q = expm(t*vector_generator(name))@four
    return np.array([(q[0]+q[3])/np.sqrt(2), q[1], q[2]])


def derivative(f, step=1e-4):
    return (-f(2*step)+8*f(step)-8*f(-step)+f(-2*step))/(12*step)


@pytest.mark.parametrize("hg", [-1, 1])
@pytest.mark.parametrize("axis", [0, 1])
def test_projected_fermion_spin_ground_radial_integral_and_units(hg, axis):
    reg = GaussianClusterRegulator(width_GeV=.8, boson_modes=8, fermion_modes=8)
    b, pp, mass = .3, 1.9, .31
    for route in emission_routes(9)[::5]:
        xf, xg = np.array([route.outgoing[route.emitter], route.outgoing[3]])/9
        xp = xf+xg
        eps = polarization(xg*pp, 0., 0., hg).conj()[2:]
        density = fermion_noether_one_gluon([xf*pp, 0., 0.], [xp*pp, 0., 0.], mass,
                    eps, right_plus=xf*pp, left_plus=xp*pp)[axis]
        for ho in (-1, 1):
            for hi in (-1, 1):
                args = (route, 0, b, mass, ho, hi, hg)
                value = retained_fermion_spin_orbital(*args, axis=axis, regulator=reg, total_pplus_GeV=pp)
                # Independent one-dimensional radial integral; spectators are
                # normalized identities, Q_T=(xf+xg)q_rel^2 on this slice.
                field = 1/np.sqrt(2*np.pi*route.outgoing[3])
                orbital = np.sqrt(xf*xg/xp)*b/np.sqrt(np.pi)/(1+xp*b*b/reg.width_GeV**2)
                assert_allclose(value, density[(ho+1)//2, (hi+1)//2]*field*orbital, atol=3e-17)
                scaled = retained_fermion_spin_orbital(*args, axis=axis, regulator=reg, total_pplus_GeV=2*pp)
                assert_allclose(scaled, value/2, atol=2e-17)
                assert not value.flags.writeable


@pytest.mark.parametrize("total,cutoff", [(9, 0), (7, 1)])
@pytest.mark.parametrize("target", ["PROTON", "NEUTRON"])
@pytest.mark.parametrize("jz", [-1, 1])
@pytest.mark.parametrize("hg", [-1, 1])
def test_actual_fock_fermion_spin_preserves_sign_isospin_and_vector_weight(total, cutoff, target, jz, hg):
    from deuteron_wigner.quantum.m2_coupled_pilot import symmetry_basis
    reg = GaussianClusterRegulator(width_GeV=.8, boson_modes=8, fermion_modes=8)
    inc = symmetry_basis("qqq", total, cutoff, target, jz, 0).toarray()
    out = symmetry_basis("qqqg", total, cutoff, target, jz+2*hg, hg).toarray()
    images = []
    for a in (0, 1):
        images.append(np.column_stack([apply_retained_fermion_spin(
            u.reshape(-1, 1, 64), total, cutoff, .3, .31, hg,
            axis=a, regulator=reg, total_pplus_GeV=1.9).ravel() for u in inc.T]))
    assert np.linalg.norm(images[0]) > 1e-4
    # Direct raw-space leakage checks both the joint S3 sign and I=1/2
    # carrier, plus the declared vector Jz change; no state is chosen.
    for v in images:
        assert_allclose(v, out@(out.conj().T@v), atol=3e-16)
    # [Jz,F1]=i F2 with Delta Jz=hg in the source active convention.
    assert_allclose(hg*images[0], 1j*images[1], atol=3e-16)
    # The paired absorption block is exactly the adjoint of this same
    # labelled density. This is an assembly check, not full Lorentz closure.
    c = out.conj().T@images[0]
    operator = np.block([[np.zeros((inc.shape[1], inc.shape[1])), c.conj().T],
                         [c, np.zeros((out.shape[1], out.shape[1]))]])
    assert_allclose(operator, operator.conj().T, atol=1e-16)


@pytest.mark.parametrize("mass", [0., .3, .8])
@pytest.mark.parametrize("p", [[.9, .2, -.1], [1.7, -.4, .3]])
def test_canonical_noether_density_recovers_actual_free_generators(mass, p):
    quark = free_fermion_noether_spin(p, p, mass)
    gluon = free_gluon_noether_spin(p, p)
    for a in range(2):
        assert_allclose(quark[a], free_generator(p, mass, "F"+str(a+1)).spin, atol=3e-16)
        assert_allclose(gluon[a], free_generator(p, 0., "F"+str(a+1), particle="gluon").spin, atol=3e-16)
    q = np.array(p)+[.2, -.1, .05]
    for fn, args in ((free_fermion_noether_spin, (mass,)), (free_gluon_noether_spin, ())):
        assert_allclose(fn(p, q, *args), fn(q, p, *args).conj().transpose(0, 2, 1), atol=4e-16)


@pytest.mark.parametrize("right,left", [(.7, 1.1), (-.7, 1.1), (.7, -1.1), (-.7, -1.1)])
@pytest.mark.parametrize("hg", [-1, 1])
def test_linear_noether_constraint_contact_and_absent_quadratic_spin_term(right, left, hg):
    pf, pi, mass = np.array([.7, .13, -.2]), np.array([1.1, .17, -.13]), .3
    eps = polarization(.4, .04, .07, hg).conj()[2:]
    value = fermion_noether_one_gluon(pf, pi, mass, eps, right_plus=right, left_plus=left)
    # Independent four-component constrained fields and anticommutator density.
    alpha = [GAMMA[0]@GAMMA[j] for j in (1, 2)]
    right_b = sum(a*e for a, e in zip(alpha, eps))/(np.sqrt(2)*right)
    left_b = sum(a*e.conjugate() for a, e in zip(alpha, eps))/(np.sqrt(2)*left)
    def full(g, axis):
        sigma = .5j*(GAMMA_MINUS@GAMMA[axis+1]-GAMMA[axis+1]@GAMMA_MINUS)
        op = GAMMA[0]@(GAMMA_PLUS@sigma+sigma@GAMMA_PLUS)/4
        out = (np.eye(4)+bad_field_kernel(pf, mass)+g*left_b)@GOOD_AXES/2**.25
        inc = (np.eye(4)+bad_field_kernel(pi, mass)+g*right_b)@GOOD_AXES/2**.25
        return out.conj().T@op@inc
    for axis in (0, 1):
        assert_allclose((full(.3, axis)-full(-.3, axis))/.6, value[axis], atol=3e-16)
        assert_allclose(full(.3, axis)+full(-.3, axis)-2*full(0, axis), 0, atol=3e-16)
    reverse = fermion_noether_one_gluon(pi, pf, mass, eps.conj(), right_plus=left, left_plus=right)
    assert_allclose(value, reverse.conj().transpose(0, 2, 1), atol=3e-16)
    assert_allclose(value, fermion_noether_one_gluon(pf, pi, .91, eps, right_plus=right, left_plus=left), atol=3e-16)


@pytest.mark.parametrize("mass", [0., .3])
@pytest.mark.parametrize("h", [-1, 1])
def test_continuum_vertex_matches_discrete_source_with_exact_cell_conversion(mass, h):
    total = 9
    for route in emission_routes(total)[::3]:
        pp = 1.9
        pf, k, pi = emission_momenta(route, (.11, -.17), pp, (.14, -.09))
        c = emission_minus_vertex(pf, k, pi, mass, h)
        cell_length = np.pi*total/pp
        expected = np.array([[emission_mass_kernel_per_gs(route, (.11, -.17), mass, hf, hi, h,
                              total_pplus=pp, parent_transverse=(.14, -.09)) for hi in (-1, 1)] for hf in (-1, 1)])
        assert_allclose(2*pp/np.sqrt(cell_length)*c, expected, rtol=4e-15, atol=2e-15)


@pytest.mark.parametrize("mass", [0., .3, .8])
@pytest.mark.parametrize("hg", [-1, 1])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("frame", [(1., 0., 0.), (1.8, .19, -.27)])
def test_bare_cubic_lorentz_ward_from_independent_finite_flow(mass, hg, axis, frame):
    scale, vx, vy = frame
    pf, k = np.array([.7*scale, .13, -.2]), np.array([.4*scale, .04, .07])
    for p in (pf, k):
        p[1:] += p[0]*np.array([vx, vy])
    pi = pf+k
    name = "F"+str(axis+1)
    def flow(t):
        return emission_minus_vertex(moved(pf, mass, name, t), moved(k, 0., name, t),
                                     moved(pi, mass, name, t), mass, hg)
    r = emission_regular_ward(pf, k, pi, mass, hg, axis, derivative(flow))
    assert_allclose(r["residual"], 0., atol=3e-10)
    assert_allclose(r["smooth_regulator_defect"], 0.)
    # This contact is NOT the local fermion spin-density contribution alone.
    local = fermion_noether_one_gluon(pf, pi, mass, polarization(*k, hg).conj()[2:],
                                     right_plus=pf[0], left_plus=pi[0])[axis]/np.sqrt(2*k[0])
    assert np.linalg.norm(local-r["contact"]) > .01


@pytest.mark.parametrize("name", ["F1", "F2", "Jx", "Jy", "Jz", "Bx", "By", "Bz", "E1", "E2"])
@pytest.mark.parametrize("n", [3, 4, 5])
def test_source_transverse_quadratic_drift_and_scalar_comparison(name, n):
    rng = np.random.default_rng(32+n)
    p = rng.uniform(.2, .9, (n, 3))
    p[:, 1:] -= .4
    masses = np.array([.3, 0., .3, .5, 0.])[:n]
    r = cluster_quadratic_lorentz(p, masses, name)
    reg = GaussianClusterRegulator(width_GeV=.8, fermion_modes=8, boson_modes=8)
    assert_allclose(r["transverse_GeV2"], np.sum((reg.leg_transform(p[:, 0])@p[:, 1:])**2), atol=5e-16)
    def flow(t):
        return cluster_quadratic_lorentz(np.array([moved(q, m, name, t) for q, m in zip(p, masses)]), masses, name)["transverse_GeV2"]
    assert_allclose(derivative(flow), r["transverse_derivative_GeV2"], atol=8e-12)
    roundoff = 32*np.finfo(float).eps*max(1., r["scalar_comparison_GeV2"])
    assert_allclose(r["scalar_comparison_derivative_GeV2"], 0., atol=roundoff)
    q = np.array([moved(q, m, name, .17) for q, m in zip(p, masses)])
    assert_allclose(cluster_quadratic_lorentz(q, masses, name)["scalar_comparison_GeV2"], r["scalar_comparison_GeV2"], atol=2e-15)
    if name in ("Jz", "Bz", "E1", "E2"):
        assert_allclose(r["transverse_derivative_GeV2"], 0., atol=4e-16)
    else:
        assert abs(r["transverse_derivative_GeV2"]) > .001


@pytest.mark.parametrize("degree", [None, 0, 1, 2])
@pytest.mark.parametrize("mass", [0., .3])
@pytest.mark.parametrize("axis", [0, 1])
def test_common_regulated_cubic_defect_is_product_rule_not_zero(degree, mass, axis):
    shape = None if degree is None else GaussianReferenceShape(base_width_GeV=.8, reference_width_GeV=1.2, degree=degree)
    reg = GaussianClusterRegulator(width_GeV=.8, boson_modes=8, fermion_modes=8, reference_shape=shape)
    pf, k = np.array([.7, .13, -.2]), np.array([.4, .04, .07])
    pi, name, hg = pf+k, "F"+str(axis+1), 1
    def flow(t):
        ps = [moved(q, m, name, t) for q, m in zip((pf, k, pi), (mass, 0., mass))]
        value, _ = source_regulator_lorentz(ps, [mass, 0., mass], name, reg)
        return value*emission_minus_vertex(*ps, mass, hg)
    r = emission_regular_ward(pf, k, pi, mass, hg, axis, derivative(flow), regulator=reg)
    assert_allclose(r["residual"], r["smooth_regulator_defect"], atol=2e-11)
    if mass:
        # Distinguish a genuine drift from the finite-difference error; no
        # physical lower bound on its magnitude is imposed across shapes.
        assert np.linalg.norm(r["residual"]) > 1e4*2e-11
    else:
        assert_allclose(r["smooth_regulator_defect"], 0, atol=2e-16)


def test_collinear_massless_limit_has_no_division_by_energy_gap():
    pf, k = np.array([.7, .14, -.07]), np.array([.4, .08, -.04])
    pi = pf+k
    for axis in (0, 1):
        name = "F"+str(axis+1)
        def flow(t):
            return emission_minus_vertex(moved(pf, 0, name, t), moved(k, 0, name, t), moved(pi, 0, name, t), 0, 1)
        r = emission_regular_ward(pf, k, pi, 0., 1, axis, derivative(flow))
        assert abs(r["gap_GeV"]) < 1e-17
        assert np.all(np.isfinite(r["contact"]))
        assert_allclose(r["residual"], 0., atol=3e-12)


def test_invalid_or_undeclared_kinematics_rejected():
    with pytest.raises(ValueError):
        emission_regular_ward([1, .1, .2], [.3, .1, 0], [1.4, .2, .2], .3, 1, 0, np.eye(2))
    for bad in (True, 0, 2):
        with pytest.raises(ValueError):
            emission_lorentz_contact([1, 0, 0], bad, 0)
    with pytest.raises(ValueError):
        emission_lorentz_contact([1, 0, 0], 1, 2)
    with pytest.raises(ValueError):
        cluster_quadratic_lorentz([[1, 0, 0], [0, 1, 1]], [.3, 0], "F1")
    with pytest.raises(ValueError):
        fermion_noether_one_gluon([1, 0, 0], [1, 0, 0], .3, [1, 0], right_plus=0., left_plus=1.)

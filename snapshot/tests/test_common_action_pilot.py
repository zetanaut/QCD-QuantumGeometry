import numpy as np
import pytest
from scipy.linalg import expm

from deuteron_wigner.quantum.common_action_pilot import (
    ChiralGraphAction, calibrated_action, scalar_elimination, gauged_kinetic,
    minimal_vector_current, HBARC,
)
from deuteron_wigner.quantum.canonical_lf_fields import GAMMA, slash
from deuteron_wigner.quantum.chiral_soft_matching import G5, square
from deuteron_wigner.quantum.modular_gauge_network import generators
from deuteron_wigner.quantum.causal_spectral_matching import SpectralResponse


def candidate(angle=.6, ratio=1.4):
    return calibrated_action(.1348, .0922, .515, .004, angle, ratio, .12, [3., 1.])


@pytest.mark.parametrize("mass", [0., .002, .004, .008])
def test_vacuum_stability_PCAC_and_generated_mass(mass):
    a = candidate(); s = a.solve(mass)
    v = s["vacuum"]; hp, hs = a.matrices(v)
    assert s["gap_norm"] < 2e-11
    np.testing.assert_allclose(s["pion_mass_squared"]*s["decay_constants"],
                               2*mass*s["pseudoscalar_residues"], atol=2e-11)
    assert np.min(s["scalar_mass_squared"]) > 0
    assert s["quark_mass"] == pytest.approx(mass+a.yukawa@v)
    if mass == 0:
        assert abs(s["pion_mass_squared"][0]) < 1e-10
        assert s["pion_mass_squared"][1] > .1
        np.testing.assert_allclose(hp@v, 0, atol=2e-11)


@pytest.mark.parametrize("start", [[.01, .01], [.2, .04], [.3, .3]])
def test_unique_positive_vacuum_from_independent_starts(start):
    a = candidate()
    np.testing.assert_allclose(a.solve(.004, start)["vacuum"], a.solve(.004)["vacuum"], atol=2e-10)


def test_potential_Hessian_and_third_vertex_are_actual_Torch_derivatives():
    import torch
    a = candidate(); s = a.solve(.004); v = s["vacuum"]; hp, hs = a.matrices(v)
    f = np.zeros((2, 4)); f[:, 0] = v
    def potential(x):
        return (torch.tensor(a.quartic)/4*(torch.sum(x*x, dim=1)-torch.tensor(a.radial_squared))**2).sum()+a.coupling/2*((x[0]-x[1])**2).sum()-.004*torch.tensor(a.scalar_source)@x[:, 0]
    x = torch.tensor(f, dtype=torch.float64, requires_grad=True)
    hessian = torch.autograd.functional.hessian(potential, x).reshape(8, 8).detach().numpy()
    np.testing.assert_allclose(hessian[np.ix_([0, 4], [0, 4])], hs, atol=1e-14)
    for axis in [1, 2, 3]:
        np.testing.assert_allclose(hessian[np.ix_([axis, axis+4], [axis, axis+4])], hp, atol=1e-14)
    e = torch.tensor(s["pion_modes"][:, 0], dtype=torch.float64)
    def pion_quadratic(y):
        hh = torch.autograd.functional.hessian(potential, y, create_graph=True).reshape(8, 8)
        return e@hh[[1, 5]][:, [1, 5]]@e
    third = torch.autograd.functional.jacobian(pion_quadratic, x)[:, 0].detach().numpy()
    np.testing.assert_allclose(third, a.scalar_vertex(v, e.numpy()), atol=1e-13)


def test_global_O4_invariance_of_interactions_not_explicit_source():
    a = candidate(); rng = np.random.default_rng(20)
    f = rng.normal(0, .1, (2, 4)); z = rng.normal(size=(4, 4)); rotation = expm(z-z.T)
    assert a.potential(f, 0.) == pytest.approx(a.potential(f@rotation.T, 0.), abs=1e-14)
    assert abs(a.potential(f, .004)-a.potential(f@rotation.T, .004)) > 1e-6


@pytest.mark.parametrize("mass", [.002, .004, .008])
def test_scalar_Feynman_Hellmann_from_independent_resolved_vacua(mass):
    a = candidate(); s = a.solve(mass); h = 2e-5
    def m2(x): return a.solve(x)["pion_mass_squared"][0]
    fd = (-m2(mass+2*h)+8*m2(mass+h)-8*m2(mass-h)+m2(mass-2*h))/(12*h)
    assert a.scalar_form_factor(s["vacuum"], s["pion_modes"][:, 0], 0.) == pytest.approx(fd, rel=2e-7)


def test_scalar_radius_calibration_and_no_source_normalization_invention():
    a = candidate(); s = a.solve(.004); v, e = s["vacuum"], s["pion_modes"][:, 0]
    assert np.sqrt(s["pion_mass_squared"][0]) == pytest.approx(.1348, abs=2e-10)
    assert s["decay_constants"][0] == pytest.approx(.0922, abs=2e-10)
    assert a.scalar_radius_squared(v, e) == pytest.approx(.515, abs=2e-9)
    q = 1e-4
    f0 = a.scalar_form_factor(v, e, 0.)
    slope = (-3*f0+4*a.scalar_form_factor(v, e, q)-a.scalar_form_factor(v, e, 2*q))/(2*q)
    assert -6*HBARC**2*slope/f0 == pytest.approx(.515, rel=3e-7)
    # Rescale the source mass coordinate and c inversely: same BOSONIC
    # potential/vacuum and normalized slope, not the same bare quark mass.
    b = ChiralGraphAction(a.quartic, a.radial_squared, a.coupling, 2*a.scalar_source, a.yukawa)
    t = b.solve(.002)
    np.testing.assert_allclose(t["vacuum"], v, atol=1e-12)
    assert b.scalar_radius_squared(v, e) == pytest.approx(a.scalar_radius_squared(v, e))
    assert b.scalar_form_factor(v, e, 0.) == pytest.approx(2*f0)


@pytest.mark.parametrize("z", [-.02, -.1, .25+.04j])
def test_generated_scalar_exact_elimination_keeps_both_sources_and_direct_term(z):
    a = candidate(); s = a.solve(.004); v = s["vacuum"]
    hs = a.matrices(v)[1]; c = a.scalar_source; g = a.scalar_vertex(v, s["pion_modes"][:, 0])
    direct = c@np.linalg.solve(hs-z*np.eye(2), g)
    np.testing.assert_allclose(scalar_elimination(hs, c, g, z), direct, atol=2e-13)
    kernel = hs-z*np.eye(2)
    wrong = c[0]*g[0]/(kernel[0, 0]-kernel[0, 1]*kernel[1, 0]/kernel[1, 1])
    assert abs(wrong-direct)/abs(direct) > .05


def test_generated_linear_modes_have_positive_causal_responses():
    a = candidate(); s = a.solve(.004); hs = a.matrices(s["vacuum"])[1]
    masses, modes = np.linalg.eigh(hs)
    residues = np.array([np.outer(modes[:, i], modes[:, i]) for i in range(2)])
    response = SpectralResponse(masses, residues, "physical_response")
    np.testing.assert_allclose(response.analytic(.1+.03j), np.linalg.inv(hs-(.1+.03j)*np.eye(2)), atol=1e-13)
    assert np.min(np.linalg.eigvalsh(response.reflection_matrix([.1, .5, 1.]))) > -1e-13


def test_generated_quark_sources_satisfy_full_tree_axial_identity():
    a = candidate(); s = a.solve(.008); v, mass = s["vacuum"], s["quark_mass"]
    p0, p1 = np.array([.2, .02, .07, .03]), np.array([.21, .32, .15, -.1])
    axial, weighted_p = a.quark_sources(v, .008, p1, p0); q = p1-p0
    lhs = q[0]*axial[0]-sum(q[i]*axial[i] for i in [1, 2, 3])
    rhs = (slash(p1)-mass*np.eye(4))@G5+G5@(slash(p0)-mass*np.eye(4))+weighted_p
    np.testing.assert_allclose(lhs, rhs, atol=5e-11)
    # The bare-current failure diagnoses omitted induced partners, not a
    # reason to rescale the current.
    assert np.linalg.norm(slash(q)@G5-rhs) > .01


def test_resolved_Yukawa_color_flavor_and_Dirac_Hermiticity():
    a = candidate(); f = np.array([[.09, .01, -.02, .03], [.04, -.01, .02, .01]])
    y = a.quark_mass_matrix(f, .004)
    gamma0 = np.kron(GAMMA[0], np.eye(6))
    np.testing.assert_allclose(gamma0@y.conj().T@gamma0, y, atol=1e-15)
    for t in generators():
        color = np.kron(np.eye(8), t)
        np.testing.assert_allclose(color@y-y@color, 0, atol=1e-15)
    # No collapse of the independent isovector pseudoscalar components.
    assert np.linalg.norm(y-np.trace(y)/24*np.eye(24)) > .1


def test_gauged_kinetic_actual_one_two_source_vertices():
    rng = np.random.default_rng(24)
    fields = rng.normal(0, .1, (2, 4)); d = rng.normal(0, .2, (4, 2, 4))
    a = rng.normal(0, .1, (4, 3)); zero = np.zeros((4, 3))
    # Axial current sigma*dpi - pi*dsigma follows from the common action.
    expected = sum(sign*np.sum(a[mu]*(fields[:, :1]*d[mu, :, 1:]-fields[:, 1:]*d[mu, :, :1])) for mu, sign in enumerate([1., -1., -1., -1.]))
    h = .05  # exactly quadratic in the source: no higher-order FD bias
    first = (gauged_kinetic(fields, d, zero, h*a)-gauged_kinetic(fields, d, zero, -h*a))/(2*h)
    assert first == pytest.approx(expected, abs=1e-13)
    second = (gauged_kinetic(fields, d, zero, h*a)+gauged_kinetic(fields, d, zero, -h*a)-2*gauged_kinetic(fields, d, zero, zero))/h**2
    expected2 = sum(sign*np.sum((fields[:, 1:]@a[mu])**2+np.sum((a[mu]*fields[:, :1])**2, axis=1)) for mu, sign in enumerate([1., -1., -1., -1.]))
    assert second == pytest.approx(expected2, abs=2e-9)
    v = rng.normal(0, .1, (4, 3))
    metric = [1., -1., -1., -1.]
    expected_v = sum(sign*np.sum(d[mu, :, 1:]*np.cross(v[mu], fields[:, 1:])) for mu, sign in enumerate(metric))
    first_v = (gauged_kinetic(fields, d, h*v, zero)-gauged_kinetic(fields, d, -h*v, zero))/(2*h)
    assert first_v == pytest.approx(expected_v, abs=1e-13)
    expected_vv = sum(sign*np.sum(np.cross(v[mu], fields[:, 1:])**2) for mu, sign in enumerate(metric))
    second_v = (gauged_kinetic(fields, d, h*v, zero)+gauged_kinetic(fields, d, -h*v, zero)-2*gauged_kinetic(fields, d, zero, zero))/h**2
    assert second_v == pytest.approx(expected_vv, abs=2e-9)
    expected_av = sum(sign*np.sum(a[mu]*fields[:, :1]*np.cross(v[mu], fields[:, 1:])) for mu, sign in enumerate(metric))
    mixed = sum(sa*sv*gauged_kinetic(fields, d, sv*h*v, sa*h*a) for sa in [-1, 1] for sv in [-1, 1])/(4*h**2)
    assert mixed == pytest.approx(expected_av, abs=2e-9)


def test_vector_vertex_is_independent_third_derivative_of_gauged_action():
    # Analytic continuation of real pion components to charged plane waves.
    # Differentiate L with respect to incoming/outgoing amplitudes and v_mu^3;
    # do not use the proposed current formula as its own action oracle.
    a = candidate(); s = a.solve(.004); mode = s["pion_modes"][:, 0]
    incoming = np.array([0., 1., 1j, 0.])/np.sqrt(2)
    outgoing = incoming.conj()
    mass2 = s["pion_mass_squared"][0]
    p0 = np.array([np.sqrt(mass2+.08), .2, 0., .2])
    p1 = np.array([np.sqrt(mass2+.11), -.1, .3, .1])
    metric = np.array([1., -1., -1., -1.]); zero = np.zeros((4, 3))
    def action(x, y, source):
        fields = mode[:, None]*(x*incoming+y*outgoing)
        d = mode[None, :, None]*(-1j*x*(metric*p0)[:, None, None]*incoming
                               +1j*y*(metric*p1)[:, None, None]*outgoing)
        return gauged_kinetic(fields, d, source, zero)
    h = .05; vertex = []
    for mu in range(4):
        source = np.zeros((4, 3)); source[mu, 2] = h
        derivative = sum(x*y*z*action(x*h, y*h, z*source)
                         for x in [-1, 1] for y in [-1, 1] for z in [-1, 1])/(8*h**3)
        vertex.append(derivative)
    np.testing.assert_allclose(vertex, minimal_vector_current(p1, p0, mode, mode), atol=2e-13)


@pytest.mark.parametrize("angle", [.35, .65, 1.0])
def test_minimal_current_pointlike_for_changed_potential_and_mixing(angle):
    a = candidate(angle); s = a.solve(.004); e = s["pion_modes"][:, 0]
    mass = np.sqrt(s["pion_mass_squared"][0])
    for q in [.03, .2, .4]:
        energy = np.sqrt(mass**2+q*q/4)
        p0 = np.array([energy, -q/2, 0., 0.]); p1 = np.array([energy, q/2, 0., 0.])
        j = minimal_vector_current(p1, p0, e, e)
        assert j[0]/(2*energy) == pytest.approx(1., abs=1e-14)
        assert square(p1) == pytest.approx(square(p0))
        q4 = p1-p0
        assert abs(q4[0]*j[0]-q4[1:]@j[1:]) < 1e-15


def test_no_invalid_unstable_action_defaults():
    with pytest.raises(ValueError):
        ChiralGraphAction([-1, 2], [.01, .02], .1, [1, 2], [1, 1])


def test_scalar_quark_source_follows_same_solved_vacuum():
    a = candidate(); m = .004; s = a.solve(m); h = 2e-5
    fd = (a.solve(m+h)["quark_mass"]-a.solve(m-h)["quark_mass"])/(2*h)
    assert a.quark_scalar_vertex(s["vacuum"], 0.) == pytest.approx(fd, rel=1e-7)
    f = np.zeros((2, 4)); f[:, 0] = s["vacuum"]; f[:, 1] = [.01, .03]
    direction = np.array([1., 0., 0.])
    fd_p = (a.potential(f, m, pseudoscalar_source=h*direction)-a.potential(f, m, pseudoscalar_source=-h*direction))/(2*h)
    assert fd_p == pytest.approx(-a.scalar_source@f[:, 1], abs=1e-14)


def test_local_O4_connection_covariance_including_mixed_vector_axial_sources():
    rng = np.random.default_rng(100)
    f = rng.normal(size=(2, 4)); d = rng.normal(size=(4, 2, 4))
    v, a = rng.normal(size=(2, 4, 3))
    z = rng.normal(size=(4, 4)); rotation = expm(.2*(z-z.T))
    slopes = rng.normal(size=(4, 4, 4)); slopes = slopes-slopes.transpose(0, 2, 1)
    connections = np.zeros((4, 4, 4))
    for mu in range(4):
        connections[mu, 0, 1:] = -a[mu]; connections[mu, 1:, 0] = a[mu]
        x, y, z = v[mu]
        connections[mu, 1:, 1:] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    moved_f = f@rotation.T
    moved_d = np.einsum('ab,mib->mia', rotation, d)+np.einsum('mab,ib->mia', slopes, moved_f)
    moved_c = rotation@connections@rotation.T-slopes
    moved_a = moved_c[:, 1:, 0]
    moved_v = moved_c[:, [3, 1, 2], [2, 3, 1]]
    np.testing.assert_allclose(gauged_kinetic(moved_f, moved_d, moved_v, moved_a), gauged_kinetic(f, d, v, a), atol=2e-13)


def test_generated_quark_mass_can_use_preserved_canonical_LF_exchange_without_H2():
    from deuteron_wigner.quantum.momentum_chiral_matching import two_quark_exchange
    a = candidate(); mass = a.solve(.004)["quark_mass"]
    k = np.array([.1, .2, .4]); r = np.array([.4, .2, .1]); energy = np.sqrt(mass**2+k@k)
    momenta = [np.r_[energy, r], np.r_[energy, -r], np.r_[energy, k], np.r_[energy, -k]]
    lf = lambda p: np.array([(p[0]+p[3])/np.sqrt(2), p[1], p[2]])
    result = two_quark_exchange(*map(lf, momenta), mass, mass, 1.2)
    np.testing.assert_allclose(result["light_front_combined"], result["covariant"], atol=2e-12)
    # This is canonical portability at a supplied tree mass, NOT matching
    # the additional collective scalar interaction to QCD.


def test_serializable_replay_preserves_calibration_and_failure_boundary():
    import json
    from scripts.evaluate_common_action_pilot import evaluate
    result = evaluate(); json.dumps(result, allow_nan=False)
    for c in result["cases"]:
        assert c["scalar_radius_squared_fm2"] == pytest.approx(.515, abs=1e-9)
        assert c["vector_radius_squared_fm2"] == 0
        assert c["exact_elimination_max_error_GeV"] < 1e-12
    assert result["targets"]["withheld_vector_radius_fm"] > .6

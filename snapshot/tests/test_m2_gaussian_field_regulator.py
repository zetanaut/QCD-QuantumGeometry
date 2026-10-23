"""Independent polar-wave quadrature and bare-kernel checks for common fields."""
from dataclasses import replace
from itertools import product
from math import pi
import numpy as np
import pytest
from scipy.special import roots_hermite
from scipy.integrate import quad

from deuteron_wigner.bridge.modes.core import ho_momentum
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_shell
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianFieldRegulator, gaussian_shell_integral
from deuteron_wigner.quantum.m2_gaussian_field_vertices import (
    cubic_field_orbital, pair_field_orbital, direct_field_orbital)
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import split_routes, longitudinal_states
from deuteron_wigner.quantum.m2_excluded_fock_vertices import cubic_orbital
from deuteron_wigner.quantum.m2_instantaneous_pair_exit import pair_contact_orbital
from deuteron_wigner.quantum.m2_six_body_quartic import COMPONENTS, direct_routes, contact_orbital


def regulator(width=.8):
    return GaussianFieldRegulator(width_GeV=width, boson_modes=8, fermion_modes=8,
                                  reference_transverse_velocity=(0., 0.))


def controls(r=None):
    return dict(regulator=r, total_pplus_GeV=1.3, total_transverse_GeV=(0., 0.))


@pytest.mark.parametrize("d", [1, 2, 3])
def test_unregulated_orthonormality_in_actual_circular_shell_convention(d):
    a = gaussian_shell_integral(out_cutoff=2, in_cutoff=2, out_map=np.eye(d), in_map=np.eye(d),
        b_GeV=.4, field_precision=0., field_maps=np.zeros((0, d)), field_shifts=np.zeros((0, 2)))
    np.testing.assert_allclose(a, np.eye(len(a)), atol=8e-15)


def test_unregulated_contact_measure_and_all_radial_signs():
    a = gaussian_shell_integral(out_cutoff=4, in_cutoff=2, out_map=np.eye(2), in_map=np.array([[1., 0.]]),
        b_GeV=.4, field_precision=0., field_maps=np.zeros((0, 2)), field_shifts=np.zeros((0, 2)))
    expected = np.array([[.4/np.sqrt(pi)*(o[0] == i[0] and o[2] == i[1] and o[1] == o[3])
        for i in oscillator_shell(1, 2)] for o in oscillator_shell(2, 4)])
    np.testing.assert_allclose(a, expected, atol=2e-15)


@pytest.mark.parametrize("d", [1, 2])
def test_shifted_gaussian_and_complex_quadratic_against_direct_polar_quadrature(d):
    rng = np.random.default_rng(729+d)
    lo, li = np.eye(d), np.linalg.qr(rng.normal(size=(d, d)))[0]
    so, si = rng.normal(size=(d, 2))*.07, rng.normal(size=(d, 2))*.05
    fields, shifts = rng.normal(size=(3, d)), rng.normal(size=(3, 2))*.1
    b, precision = .43, 1/.8**2
    linear = rng.normal(size=(2, d))+1j*rng.normal(size=(2, d))
    quadratic = rng.normal(size=(2*d, 2*d))+1j*rng.normal(size=(2*d, 2*d))
    quadratic = (quadratic+quadratic.T)/2
    kwargs = dict(out_cutoff=2, in_cutoff=2, out_map=lo, in_map=li, out_shift=so, in_shift=si,
        b_GeV=b, field_precision=precision, field_maps=fields, field_shifts=shifts,
        constant=.31+.2j, linear=linear, quadratic=quadratic)
    answer = gaussian_shell_integral(**kwargs)
    c = (lo.T@lo+li.T@li)/b**2+precision*fields.T@fields
    h = -(lo.T@so+li.T@si)/b**2-precision*fields.T@shifts
    covariance = np.linalg.inv(c)
    nodes, weights = roots_hermite(8)
    ids = np.array(list(product(range(8), repeat=2*d)))
    z = nodes[ids]
    q = (np.sqrt(2)*z.reshape(-1, 2, d)@np.linalg.cholesky(covariance).T
         +(covariance@h).T).transpose(0, 2, 1)
    measure = (np.prod(weights[ids], axis=1)*np.exp(np.sum(z*z, axis=1))
               *2**d/np.linalg.det(c)/(2*pi)**(2*d))
    qo, qi, legs = np.einsum("ad,ndj->naj", lo, q)+so, np.einsum("ad,ndj->naj", li, q)+si, np.einsum("ad,ndj->naj", fields, q)+shifts
    field = np.exp(-precision*np.sum(legs**2, axis=(1, 2))/2)
    flat = q.transpose(0, 2, 1).reshape(len(q), -1)
    polynomial = .31+.2j+flat@linear.reshape(-1)+np.einsum("na,ab,nb->n", flat, quadratic, flat)/2
    def wave(state, points):
        value = np.ones(len(points), complex)
        for j in range(d):
            plus, minus = state[j], state[d+j]
            radial, angular = min(plus, minus), plus-minus
            value *= (-1)**radial*ho_momentum(radial, angular, points[:, j, 0], points[:, j, 1], b)
        return value
    modes = oscillator_shell(d, 2)
    out, inc = np.array([wave(s, qo) for s in modes]), np.array([wave(s, qi) for s in modes])
    expected = (out.conj()*(measure*field*polynomial))@inc.T
    np.testing.assert_allclose(answer, expected, atol=2e-14, rtol=3e-13)
    reverse = gaussian_shell_integral(**{**kwargs, "out_map":li, "in_map":lo,
        "out_shift":si, "in_shift":so, "constant":(.31+.2j).conjugate(),
        "linear":linear.conj(), "quadratic":quadratic.conj()})
    np.testing.assert_allclose(answer, reverse.conj().T, atol=2e-14)


@pytest.mark.parametrize("width", [.3, .8, 1.7])
def test_exact_same_field_trace_and_joint_frame_boost(width):
    r = regulator(width)
    value = quad(lambda k: k*r.amplitude("g", 2, np.array([k, 0.]), pplus_GeV=.5)**2/(2*pi),
                 0., np.inf, epsabs=1e-13)[0]
    assert value == pytest.approx(r.coincident_trace_GeV2, abs=2e-14)
    k = np.array([[.13, -.42], [.24, .52]])
    beta = np.array([.2, -.3])
    shifted = replace(r, reference_transverse_velocity=tuple(beta))
    np.testing.assert_allclose(r.amplitude("q", 3, k, pplus_GeV=.7),
        shifted.amplitude("q", 3, k+.7*beta, pplus_GeV=.7), atol=2e-16)
    assert r.amplitude("q", 17, k[0], pplus_GeV=.7) == 0
    assert r.amplitude("g", 18, k[0], pplus_GeV=.7) == 0


@pytest.mark.parametrize("channel", ["q_to_qg", "g_to_gg", "g_to_pair"])
@pytest.mark.parametrize("cutoffs", [(0, 0), (1, 2), (1, 0)])
def test_all_cubic_orbitals_recover_the_canonical_bare_limit(channel, cutoffs):
    routes = split_routes(9, channel)
    for route in (routes[0], routes[-1]):
        for hs in product((-1, 1), repeat=3):
            a = cubic_field_orbital(route, *cutoffs, .4, .35, hs, **controls())
            b = cubic_orbital(route, *cutoffs, .4, .35, hs)
            np.testing.assert_allclose(a, b, atol=2e-15, rtol=3e-13)


@pytest.mark.parametrize("sector", ["qqqgg", "qqqq-qbar"])
def test_quartic_pair_orbitals_recover_bare_measure_and_radial_states(sector):
    for outgoing in (longitudinal_states(sector, 9)[0], longitudinal_states(sector, 9)[-1]):
        for emitter in (0, 2):
            incoming = list(outgoing[:3])
            incoming[emitter] += outgoing[3]+outgoing[4]
            incoming = tuple(incoming)
            a = pair_field_orbital(incoming, outgoing, emitter, 1, 2, .4, sector=sector, **controls())
            b = pair_contact_orbital(incoming, outgoing, emitter, 1, 2, .4, sector=sector)
            np.testing.assert_allclose(a, b, atol=2e-15, rtol=3e-13)


@pytest.mark.parametrize("component", COMPONENTS)
def test_every_direct_six_body_family_reproduces_its_bare_orbital(component):
    routes = direct_routes(9, component)
    for route in (routes[0], routes[-1]):
        a = direct_field_orbital(route, 1, 2, .4, **controls())
        b = contact_orbital(route, 1, 2, .4)
        np.testing.assert_allclose(a, b, atol=2e-15, rtol=3e-13)


def test_field_regulation_changes_inherited_shell_selection_and_has_explicit_frame():
    route = split_routes(9, "q_to_qg")[0]
    values = [cubic_field_orbital(route, 1, 0, .4, .35, hs, **controls(regulator()))
              for hs in product((-1, 1), repeat=3)]
    assert max(np.linalg.norm(a[:, 1:]) for a in values) > 1e-6
    for hs in product((-1, 1), repeat=3):
        np.testing.assert_allclose(cubic_orbital(route, 1, 0, .4, .35, hs)[:, 1:], 0., atol=1e-17)
    r = regulator()
    beta = np.array([.2, -.1])
    a = cubic_field_orbital(route, 1, 2, .4, .35, (1, 1, 1), **controls(r))
    args = {**controls(replace(r, reference_transverse_velocity=tuple(beta))),
            "total_transverse_GeV":1.3*beta}
    b = cubic_field_orbital(route, 1, 2, .4, .35, (1, 1, 1), **args)
    np.testing.assert_allclose(a, b, atol=2e-15)


@pytest.mark.parametrize("kwargs", [
    dict(width_GeV=0.), dict(width_GeV=np.inf), dict(boson_modes=True),
    dict(fermion_modes=0), dict(reference_transverse_velocity=(0., np.nan))])
def test_invalid_field_parameters_rejected(kwargs):
    args = dict(width_GeV=.8, boson_modes=8, fermion_modes=8,
                reference_transverse_velocity=(0., 0.))
    with pytest.raises(ValueError):
        GaussianFieldRegulator(**{**args, **kwargs})

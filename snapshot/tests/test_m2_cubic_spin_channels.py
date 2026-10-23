"""Crossed Dirac/YM source checks, independent density moments and Ward flows."""
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose

from deuteron_wigner.bridge.modes.core import GAMMA, polarization, lf_dot
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_constrained_fermion_current import GOOD_AXES, ANTI_AXES, bad_field_kernel
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor, _slash_epsilon, three_gluon_numerator
from deuteron_wigner.quantum.m2_canonical_cubic_spin import canonical_emission_density, canonical_density_contact
from deuteron_wigner.quantum.m2_cubic_spin_channels import (
    FERMION_CHANNELS, fermionic_cubic_density, fermionic_cubic_contact,
    fermionic_gauge_contact, gluonic_cubic_density, gluonic_cubic_contact,
    gluonic_gauge_contact, cubic_regular_ward,
)
from deuteron_wigner.quantum.m2_cubic_lorentz_ward import source_regulator_lorentz
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianClusterRegulator
from deuteron_wigner.quantum.m2_gaussian_reference_shape import GaussianReferenceShape
from deuteron_wigner.quantum import m2_excluded_fock_vertices as five_vertices
from deuteron_wigner.quantum import m2_six_body_vertices as six_vertices
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import split_routes as five_routes


def momenta(boost=(1., 0., 0.)):
    scale, vx, vy = boost
    p = np.array([[.7*scale, .13, -.2], [.4*scale, .04, .07], [1.1*scale, .17, -.13]])
    p[:, 1:] += p[:, :1]*np.array([vx, vy])
    return p


def derivative(f, h=1e-4):
    return (-f(2*h)+8*f(h)-8*f(-h)+f(-2*h))/(12*h)


def flow(p, mass, axis, t):
    q = p.copy()
    pm = (mass*mass+p[1:]@p[1:])/(2*p[0])
    q[0] += t*p[axis+1]+t*t*pm/2
    q[axis+1] += t*pm
    return q


def source_dirac(channel, p, mass, hg):
    """Original physical u/v bilinears, no signed-field/density routines."""
    u = lambda q: np.column_stack([lf_spinor(*q, mass, h) for h in (-1, 1)])
    v = lambda q: np.column_stack([antiquark_spinor(*q, mass, h) for h in (-1, 1)])
    if channel == "q_to_qg":
        value = -u(p[0]).conj().T@GAMMA[0]@_slash_epsilon(polarization(*p[1], hg).conj())@u(p[2])
    elif channel == "anti_to_antig":
        value = (-v(p[2]).conj().T@GAMMA[0]@_slash_epsilon(polarization(*p[1], hg).conj())@v(p[0])).T
    else:
        value = -u(p[0]).conj().T@GAMMA[0]@_slash_epsilon(polarization(*p[2], hg))@v(p[1])
    return value/np.sqrt(8*np.prod(p[:, 0]))


@pytest.mark.parametrize("mass", [0., .3, .8])
def test_signed_creation_field_matches_independent_charge_conjugated_spinors(mass):
    p = momenta()[0]
    for axes, sign, spinor in ((GOOD_AXES, 1, lf_spinor), (ANTI_AXES, -1, antiquark_spinor)):
        expected = np.column_stack([spinor(*p, mass, h) for h in (-1, 1)])/np.sqrt(2*p[0])
        assert_allclose((np.eye(4)+bad_field_kernel(sign*p, mass))@axes/2**.25, expected, atol=5e-16)


@pytest.mark.parametrize("channel", FERMION_CHANNELS)
@pytest.mark.parametrize("mass", [0., .3, .8])
@pytest.mark.parametrize("hg", [-1, 1])
@pytest.mark.parametrize("offset", [[0., 0., 0.], [.07, -.03, .06]])
def test_all_signed_fermion_channels_match_actual_physical_source(channel, mass, hg, offset):
    p = momenta()
    p[2] -= offset
    d = fermionic_cubic_density(channel, p, mass, hg)
    assert_allclose(d["Dirac"], source_dirac(channel, p, mass, hg), atol=3e-15)
    if not np.any(offset):
        assert_allclose(d["fermion_density_difference"], 0., atol=1e-15)
        assert_allclose(d["gluon_density_difference"], 0., atol=1e-15)
    else:
        assert np.linalg.norm(d["canonical"]-d["Dirac"]) > .001
    if channel == "q_to_qg":
        old = canonical_emission_density(*p, mass, hg)
        for key in old:
            assert_allclose(d[key], old[key], atol=3e-15)


@pytest.mark.parametrize("channel", FERMION_CHANNELS)
@pytest.mark.parametrize("mass", [0., .3, .8])
@pytest.mark.parametrize("hg", [-1, 1])
@pytest.mark.parametrize("boost", [(1., 0., 0.), (1.8, .2, -.3)])
def test_crossed_noether_density_first_moments_without_energy_gap_inversion(channel, mass, hg, boost):
    p = momenta(boost)
    for axis in (0, 1):
        result = fermionic_cubic_contact(channel, p, mass, hg, axis)
        # Independent choice of normal derivative: ALWAYS move the physical
        # parent. It need not equal the implementation's signed-field path.
        def density(t, part):
            q = p.copy()
            q[2, axis+1] -= t
            return fermionic_cubic_density(channel, q, mass, hg)[part+"_density_difference"]
        for part in ("fermion", "gluon"):
            assert_allclose(derivative(lambda t: density(t, part)), result[part+"_density_derivative"], atol=8e-12)
        z = fermionic_gauge_contact(channel, p, hg, axis)
        assert_allclose(result["combined_contact"], z, atol=8e-15)
        assert_allclose(result["fermion_improved_contact"], 0., atol=5e-15)
        if channel == "q_to_qg":
            previous = canonical_density_contact(*p, mass, hg, axis)
            for key in previous:
                assert_allclose(result[key], previous[key], atol=4e-15)


@pytest.mark.parametrize("h", list(product((-1, 1), repeat=3)))
@pytest.mark.parametrize("offset", [[0., 0., 0.], [.07, -.03, .06]])
def test_local_ym_hamiltonian_matches_source_and_bose_color_antisymmetry(h, offset):
    p = momenta()
    p[2] -= offset
    d = gluonic_cubic_density(p, h)
    source = three_gluon_numerator(*p, *h)/np.sqrt(8*np.prod(p[:, 0]))
    assert_allclose(d["local_YM"], source, atol=3e-15)
    swapped = gluonic_cubic_density(p[[1, 0, 2]], (h[1], h[0], h[2]))
    for key in d:
        assert_allclose(swapped[key], -d[key], atol=3e-15)
    if not np.any(offset):
        assert_allclose(d["gluon_density_difference"], 0., atol=4e-16)


@pytest.mark.parametrize("h", list(product((-1, 1), repeat=3)))
@pytest.mark.parametrize("boost", [(1., 0., 0.), (1.8, .2, -.3)])
def test_ym_noether_spin_and_density_improvement_match_gauge_contact(h, boost):
    p = momenta(boost)
    for axis in (0, 1):
        result = gluonic_cubic_contact(p, h, axis)
        def density(t):
            q = p.copy()
            q[2, axis+1] -= t
            return gluonic_cubic_density(q, h)["gluon_density_difference"]
        z = gluonic_gauge_contact(p, h, axis)
        assert_allclose(derivative(density), result["gluon_density_derivative"], atol=1e-11)
        assert_allclose(result["gluon_spin"], 2*z, atol=8e-15)
        assert_allclose(result["combined_contact"], z, atol=8e-15)


@pytest.mark.parametrize("h", list(product((-1, 1), repeat=3)))
def test_three_gluon_contracted_ward_identity_from_covariant_tensor(h):
    p = momenta()
    four = lambda q: np.array([q[0], q[1:]@q[1:]/(2*q[0]), *q[1:]])
    k = np.array([-four(p[2]), four(p[0]), four(p[1])])
    eps = np.array([polarization(*p[2], h[2]), polarization(*p[0], h[0]).conj(), polarization(*p[1], h[1]).conj()])
    gap = k[:, 1].sum()
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        e = eps.copy()
        e[a] = k[a]
        contracted = sum(lf_dot(k[i]-k[j], e[l])*lf_dot(e[i], e[j])
                         for i, j, l in ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
        assert_allclose(contracted, gap*(k[b, 0]-k[c, 0])*lf_dot(eps[b], eps[c]), atol=7e-16)


@pytest.mark.parametrize("sector,channel", [
    (None, "q_to_qg"), (None, "g_to_pair"), (None, "g_to_gg"),
    ("qqqgg", "q_to_qg"), ("qqqgg", "g_to_pair"), ("qqqgg", "g_to_gg"),
    ("qqqq-qbar", "q_to_qg"), ("qqqq-qbar", "anti_to_antig")])
@pytest.mark.parametrize("pp", [.9, 2.7, 6.2])
def test_actual_five_and_six_body_source_finite_cell_factor_once(sector, channel, pp):
    source = five_vertices if sector is None else six_vertices
    routes = five_routes(11, channel) if sector is None else source.split_routes(11, sector, channel)
    relative, pt, mass = (.13, -.21), (.34, .16), .35
    for route in (routes[0], routes[-1]):
        p = np.array(source.daughter_momenta(route, relative, pp, pt))
        for h in product((-1, 1), repeat=3):
            if channel == "g_to_gg":
                c = gluonic_cubic_density(p, h)["canonical"]
            else:
                d = fermionic_cubic_density(channel, p, mass, h[2] if channel == "g_to_pair" else h[1])
                row, col = (h[0]+1)//2, (h[1 if channel == "g_to_pair" else 2]+1)//2
                c = d["canonical"][row, col]
            expected = source.cubic_mass_kernel(route, relative, mass, h, total_pplus=pp, parent_transverse=pt)
            assert_allclose(2*pp/np.sqrt(np.pi*11/pp)*c, expected, atol=8e-15)


@pytest.mark.parametrize("channel", (*FERMION_CHANNELS, "g_to_gg"))
@pytest.mark.parametrize("mass", [0., .3])
@pytest.mark.parametrize("degree", [None, 0, 2])
def test_all_bare_ward_and_existing_regulator_product_rules(channel, mass, degree):
    p = momenta()
    mass = 0. if channel == "g_to_gg" else mass
    masses = (mass, mass, 0.) if channel == "g_to_pair" else (mass, 0., mass)
    hs = list(product((-1, 1), repeat=3)) if channel == "g_to_gg" else (-1, 1)
    shape = None if degree is None else GaussianReferenceShape(base_width_GeV=.8, reference_width_GeV=1.2, degree=degree)
    reg = GaussianClusterRegulator(width_GeV=.8, boson_modes=8, fermion_modes=8, reference_shape=shape)
    for h, axis in product(hs, (0, 1)):
        def vertex(t, regulated):
            q = np.array([flow(leg, m, axis, t) for leg, m in zip(p, masses)])
            c = (three_gluon_numerator(*q, *h)/np.sqrt(8*np.prod(q[:, 0])) if channel == "g_to_gg"
                 else source_dirac(channel, q, mass, h))
            r = source_regulator_lorentz(q, masses, "F"+str(axis+1), reg)[0] if regulated else 1.
            return r*c
        for regulated in (False, True):
            result = cubic_regular_ward(channel, p, mass, h, axis,
                derivative(lambda t: vertex(t, regulated)), regulator=reg if regulated else None)
            assert_allclose(result["residual"], result["smooth_regulator_defect"], atol=2e-11)
            if mass == 0:
                assert_allclose(result["smooth_regulator_defect"], 0., atol=1e-15)


@pytest.mark.parametrize("channel", (*FERMION_CHANNELS, "g_to_gg"))
def test_massless_collinear_zero_gap_contacts_remain_regular(channel):
    p = np.array([[.7, .14, -.07], [.4, .08, -.04], [1.1, .22, -.11]])
    hs = list(product((-1, 1), repeat=3)) if channel == "g_to_gg" else (-1, 1)
    for h, axis in product(hs, (0, 1)):
        c = (gluonic_cubic_contact(p, h, axis) if channel == "g_to_gg"
             else fermionic_cubic_contact(channel, p, 0., h, axis))
        z = (gluonic_gauge_contact(p, h, axis) if channel == "g_to_gg"
             else fermionic_gauge_contact(channel, p, h, axis))
        assert_allclose(c["combined_contact"], z, atol=8e-15)
        assert np.all(np.isfinite(z))


def test_invalid_channels_momenta_and_missing_constraint_domains_are_explicit():
    p = momenta()
    with pytest.raises(ValueError, match="channel"):
        fermionic_cubic_density("guess", p, .3, 1)
    with pytest.raises(ValueError, match="nonnegative"):
        fermionic_cubic_density("g_to_pair", p, -.3, 1)
    with pytest.raises(ValueError, match="physical"):
        fermionic_cubic_density("g_to_pair", -p, .3, 1)
    q = p.copy()
    q[2, 0] = q[0, 0]
    with pytest.raises(ValueError, match="zero mode"):
        gluonic_cubic_density(q, (1, -1, 1))
    q = p.copy()
    q[2, 1] += .02
    with pytest.raises(ValueError, match="conservation"):
        fermionic_cubic_contact("anti_to_antig", q, .3, 1, 0)

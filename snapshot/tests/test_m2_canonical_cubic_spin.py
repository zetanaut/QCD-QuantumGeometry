"""Independent Legendre-density expansion and delta-contact reduction."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from deuteron_wigner.bridge.modes.core import GAMMA, GAMMA_MINUS, polarization, polarization_cartesian
from deuteron_wigner.quantum.m2_constrained_fermion_current import GOOD_AXES, bad_field_kernel
from deuteron_wigner.quantum.m2_cubic_lorentz_ward import (
    emission_lorentz_contact, emission_regular_ward, emission_minus_vertex,
    source_regulator_lorentz,
)
from deuteron_wigner.quantum.m2_canonical_cubic_spin import (
    canonical_emission_density, quark_gluon_noether_spin, canonical_density_contact,
)
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianClusterRegulator
from deuteron_wigner.quantum.m2_gaussian_reference_shape import GaussianReferenceShape


def derivative(f, h=1e-4):
    return (-f(2*h)+8*f(h)-8*f(-h)+f(-2*h))/(12*h)


def scalar_legendre_expansion(pf, k, pi, mass, h, row, col, strength):
    """Fourier field values and derivatives inserted directly into H=-L_space.

    Extract the g coefficient separately; the fields include their explicit
    first-order constraint corrections and their DIFFERENT Fourier momenta.
    No production spatial-H or density-difference function is used here.
    """
    eps = polarization(*k, h).conj()
    cart = polarization_cartesian(eps)
    slash = sum(a*e*(1 if i == 0 else -1) for i, (a, e) in enumerate(zip(GAMMA, cart)))
    alpha = [GAMMA[0]@GAMMA[i] for i in (1, 2)]
    vertex = sum(a*e for a, e in zip(alpha, eps[2:]))
    norm = GOOD_AXES/2**.25
    ui = (np.eye(4)+bad_field_kernel(pi, mass))@norm[:, col]
    uf = ((np.eye(4)+bad_field_kernel(pf, mass))@norm[:, row]).conj()
    r, l = pi-k, pf+k
    ri = vertex@norm[:, col]/(np.sqrt(2)*r[0])
    lf = norm[:, row].conj()@vertex/(np.sqrt(2)*l[0])
    right, left = ui+strength*ri, uf+strength*lf
    # Coordinates x-,x,y: incoming exp(-ip+x-+ip_T.x_T), outgoing
    # conjugate phases. Right correction carries pi-k, left pf+k.
    signs = np.array([-1., 1., 1.])
    dr = [1j*s*(p*ui+strength*q*ri) for s,p,q in zip(signs,pi,r)]
    dl = [-1j*s*(p*uf+strength*q*lf) for s,p,q in zip(signs,pf,l)]
    hfree = mass*(left@GAMMA[0]@right)
    for gamma, a, b in zip((GAMMA_MINUS, GAMMA[1], GAMMA[2]), dr, dl):
        hfree += -.5j*(left@GAMMA[0]@gamma@a-b@GAMMA[0]@gamma@right)
    interaction = -strength*left@GAMMA[0]@slash@right
    return (hfree+interaction)/np.sqrt(2*k[0])


@pytest.mark.parametrize("mass", [0., .3, .8])
@pytest.mark.parametrize("hg", [-1, 1])
@pytest.mark.parametrize("offset", [[.0, .0, .0], [.07, -.03, .06], [-.08, .02, -.05]])
def test_canonical_fermion_legendre_density_independent_field_expansion(mass, hg, offset):
    pf, k = np.array([.7, .13, -.2]), np.array([.4, .04, .07])
    pi = pf+k-np.array(offset)
    result = canonical_emission_density(pf, k, pi, mass, hg)
    expected = np.array([[derivative(lambda g: scalar_legendre_expansion(pf, k, pi, mass, hg, r, c, g), .2)
                         for c in (0, 1)] for r in (0, 1)])
    assert_allclose(result["Dirac"]+result["fermion_density_difference"], expected, atol=3e-15)
    if not np.any(offset):
        assert_allclose(result["fermion_density_difference"], 0., atol=4e-16)
        assert_allclose(result["gluon_density_difference"], 0., atol=4e-16)
    else:
        assert np.linalg.norm(result["canonical"]-result["Dirac"]) > .001


@pytest.mark.parametrize("hg", [-1, 1])
@pytest.mark.parametrize("delta_plus", [-.1, 0., .1])
def test_gluon_gauss_spin_and_legendre_terms_from_fourier_field_derivatives(hg, delta_plus):
    pf, k = np.array([.7, .13, -.2]), np.array([.4, .04, .07])
    pi = pf+k-[delta_plus, .05, -.03]
    q = pi-pf
    # Source D=partial-ig A: (partial+)² A^-_q = g J+.
    aq = -np.eye(2)/q[0]**2
    assert_allclose(-q[0]**2*aq, np.eye(2))
    eps = polarization(*k, hg).conj()/np.sqrt(2*k[0])
    dplus_gluon = 1j*k[0]*eps
    dplus_aq = -1j*q[0]*aq
    dtransverse_aq = [1j*q[a+1]*aq for a in (0, 1)]
    gauge_h = -dplus_gluon[1]*dplus_aq-sum(dplus_gluon[a+2]*dtransverse_aq[a] for a in (0, 1))
    spin = np.array([dplus_gluon[a+2]*aq-dplus_aq*eps[a+2] for a in (0, 1)])
    assert_allclose(canonical_emission_density(pf, k, pi, .3, hg)["gluon_density_difference"], gauge_h, atol=3e-16)
    assert_allclose(quark_gluon_noether_spin(pf, k, pi, .3, hg)["gluon"], spin, atol=4e-15)


@pytest.mark.parametrize("mass", [0., .3, .8])
@pytest.mark.parametrize("hg", [-1, 1])
@pytest.mark.parametrize("boost", [(1., .0, .0), (1.8, .2, -.3)])
def test_density_first_moment_and_local_noether_spin_give_ward_representative(mass, hg, boost):
    scale, vx, vy = boost
    pf, k = np.array([.7*scale, .13, -.2]), np.array([.4*scale, .04, .07])
    for p in (pf, k):
        p[1:] += p[0]*np.array([vx, vy])
    pi = pf+k
    for axis in (0, 1):
        out = canonical_density_contact(pf, k, pi, mass, hg, axis)
        unit = np.eye(3)[axis+1]
        for part in ("fermion", "gluon"):
            fd = derivative(lambda t: canonical_emission_density(pf, k, pi-t*unit, mass, hg)[part+"_density_difference"])
            assert_allclose(fd, out[part+"_density_derivative"], atol=5e-12)
        assert_allclose(out["fermion_improved_contact"], 0., atol=1e-15)
        expected = emission_lorentz_contact(k, hg, axis)
        assert_allclose(out["combined_contact"], expected, atol=4e-15)
        assert_allclose(out["gluon_improved_contact"], expected, atol=4e-15)
        assert np.linalg.norm(out["fermion_spin"]+out["gluon_spin"]-expected) > .1


@pytest.mark.parametrize("degree", [0, 2])
@pytest.mark.parametrize("axis", [0, 1])
def test_common_kernel_times_complete_bare_noether_still_has_smooth_lorentz_defect(degree, axis):
    shape = GaussianReferenceShape(base_width_GeV=.8, reference_width_GeV=1.2, degree=degree)
    reg = GaussianClusterRegulator(width_GeV=.8, boson_modes=8, fermion_modes=8, reference_shape=shape)
    pf, k = np.array([.7, .13, -.2]), np.array([.4, .04, .07])
    pi, mass, hg = pf+k, .3, 1
    name = "F"+str(axis+1)
    r0 = source_regulator_lorentz([pf, k, pi], [mass, 0., mass], name, reg)[0]
    unit = np.eye(3)[axis+1]
    def density_difference(t):
        incoming = pi-t*unit
        d = canonical_emission_density(pf, k, incoming, mass, hg)
        r = source_regulator_lorentz([pf, k, incoming], [mass, 0., mass], name, reg)[0]
        return r*(d["fermion_density_difference"]+d["gluon_density_difference"])
    s = quark_gluon_noether_spin(pf, k, pi, mass, hg)
    contact = r0*(s["fermion"][axis]+s["gluon"][axis])+1j*derivative(density_difference)
    assert_allclose(contact, r0*emission_lorentz_contact(k, hg, axis), atol=3e-12)
    def flow(p, m, t):
        pm = (m*m+p[1:]@p[1:])/(2*p[0])
        q = p.copy()
        q[0] += t*p[axis+1]+t*t*pm/2
        q[axis+1] += t*pm
        return q
    def moved_vertex(t):
        f, g, i = [flow(p, m, t) for p, m in zip((pf, k, pi), (mass, 0., mass))]
        r = source_regulator_lorentz([f, g, i], [mass, 0., mass], name, reg)[0]
        return r*emission_minus_vertex(f, g, i, mass, hg)
    ward = emission_regular_ward(pf, k, pi, mass, hg, axis, derivative(moved_vertex), regulator=reg)
    assert np.linalg.norm(ward["smooth_regulator_defect"]) > 1e-4
    assert_allclose(ward["residual"], ward["smooth_regulator_defect"], atol=1e-11)


@pytest.mark.parametrize("incoming", [[.4, .0, .0], [.7, .0, .0]])
def test_excluded_constraint_zero_modes_are_not_manufactured(incoming):
    with pytest.raises(ValueError, match="zero modes"):
        canonical_emission_density([.7, .13, -.2], [.4, .04, .07], incoming, .3, 1)


def test_contact_derivative_requires_declared_conservation_slice():
    with pytest.raises(ValueError, match="spatial conservation"):
        canonical_density_contact([.7, .13, -.2], [.4, .04, .07], [1.1, .2, -.13], .3, 1, 0)

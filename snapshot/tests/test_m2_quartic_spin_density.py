"""Independent local-field expansion, projector proof and source normalization."""
from itertools import product
from math import pi, sqrt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from deuteron_wigner.bridge.modes.core import GAMMA, GAMMA_PLUS, GAMMA_MINUS, polarization
from deuteron_wigner.quantum.m2_constrained_fermion_current import GOOD_AXES, ANTI_AXES, bad_field_kernel
from deuteron_wigner.quantum.m2_excluded_fock_vertices import _slash_epsilon
from deuteron_wigner.quantum.m2_canonical_instantaneous_fermion import transverse_spin_factor
from deuteron_wigner.quantum.m2_instantaneous_pair_exit import pair_spin_factor
from deuteron_wigner.quantum.m2_six_body_quartic import pair_scattering_spin
from deuteron_wigner.quantum.m2_quartic_spin_density import (
    bad_projected_noether, instantaneous_fermion_density, instantaneous_gluon_density,
    four_gluon_density,
)


def field_coefficients(pf, pi_, kl, kr, el, er, mass, oa, ia, a, b):
    """Insert Fourier fields/derivatives into -L_space, no production K."""
    axes = {"q": GOOD_AXES, "qbar": ANTI_AXES}
    go, gi = axes[oa]/2**.25, axes[ia]/2**.25
    uf = ((np.eye(4)+bad_field_kernel(pf, mass))@go).conj().T
    ui = (np.eye(4)+bad_field_kernel(pi_, mass))@gi
    alpha = [GAMMA[0]@GAMMA[j] for j in (1, 2)]
    vl, vr = [sum(m*e for m, e in zip(alpha, eps)) for eps in (el, er)]
    l, r = pf+kl, pi_-kr
    left1, right1 = go.conj().T@vl/(sqrt(2)*l[0]), vr@gi/(sqrt(2)*r[0])
    left, right = uf+a*left1, ui+b*right1
    h = mass*left@GAMMA[0]@right
    for j, (gamma, sign) in enumerate(zip((GAMMA_MINUS, GAMMA[1], GAMMA[2]), (-1, 1, 1))):
        dl = -1j*sign*(pf[j]*uf+a*l[j]*left1)
        dr = 1j*sign*(pi_[j]*ui+b*r[j]*right1)
        h += -.5j*(left@GAMMA[0]@gamma@dr-dl@GAMMA[0]@gamma@right)
    eps = [np.array([0., k[1:]@e/k[0], *e]) for k, e in ((kl, el), (kr, er))]
    # Fix the declared color ordering (left gluon, right gluon); its
    # reversed order is a separate coefficient, not an implicit factor two.
    h -= a*uf@GAMMA[0]@_slash_epsilon(eps[0])@right
    h -= b*left@GAMMA[0]@_slash_epsilon(eps[1])@ui
    spin = []
    for j in (1, 2):
        sigma = .5j*(GAMMA_MINUS@GAMMA[j]-GAMMA[j]@GAMMA_MINUS)
        hj = GAMMA[0]@(GAMMA_PLUS@sigma+sigma@GAMMA_PLUS)/4
        spin.append(left@hj@right)
    norm = sqrt(4*abs(kl[0]*kr[0]))
    return np.array([h, *spin])/norm


@pytest.mark.parametrize("axes,signs", [
    (("q", "q"), (1, 1)), (("qbar", "qbar"), (-1, -1)),
    (("q", "qbar"), (1, -1)), (("qbar", "q"), (-1, 1))])
@pytest.mark.parametrize("mass", [0., .3, .8])
@pytest.mark.parametrize("h", list(product((-1, 1), repeat=2)))
def test_ordered_quartic_from_fourier_legendre_and_spin_fields(axes, signs, mass, h):
    pf = signs[0]*np.array([.7, .13, -.2])
    pi_ = signs[1]*np.array([1.3, .23, -.16])
    kl, kr = np.array([.4, .04, .07]), np.array([-.2, -.1, .03])
    el, er = [polarization(abs(k[0]), k[1], k[2], hh)[2:] for k, hh in zip((kl, kr), h)]
    t = .2
    raw = sum(a*b*field_coefficients(pf, pi_, kl, kr, el, er, mass, *axes, a*t, b*t)
              for a, b in product((-1, 1), repeat=2))/(4*t*t)
    d = instantaneous_fermion_density(pf, pi_, kl, kr, mass, el, er, out_axes=axes[0], in_axes=axes[1])
    assert_allclose(d["canonical"], raw[0], atol=8e-14)
    assert_allclose(d["local_spin"], raw[1:], atol=3e-14)
    assert_allclose(d["local_spin"], 0., atol=1e-16)
    assert_allclose(d["canonical"]-d["source"], d["density_difference"], atol=4e-15)
    for axis in (1, 2):
        q = pi_.copy()
        q[axis] += .41
        moved = instantaneous_fermion_density(pf, q, kl, kr, mass, el, er, out_axes=axes[0], in_axes=axes[1])
        assert_allclose(moved["canonical"], d["canonical"], atol=3e-15)
        assert_allclose(moved["density_difference"], d["density_difference"], atol=3e-15)


def test_bad_projector_noether_zero_and_no_quartic_gluon_spin_from_field_degree():
    bad = (np.eye(4)-GAMMA[0]@GAMMA[3])/2
    for j in (1, 2):
        sigma = .5j*(GAMMA_MINUS@GAMMA[j]-GAMMA[j]@GAMMA_MINUS)
        h = GAMMA[0]@(GAMMA_PLUS@sigma+sigma@GAMMA_PLUS)/4
        assert_allclose(bad@h@bad, 0., atol=1e-16)
        assert_allclose(bad_projected_noether(j-1), bad@h@bad, atol=1e-16)
    # A^-=A0^-+g A1^- exactly for good fields. Sg=partial+Ai A^-
    # -partial+A^- Ai has no g² term, unlike its quadratic H density.
    at, dat, am0, am1, dam0, dam1 = .2+.3j, -.7j, .6, -.8+.1j, .4j, .5-.2j
    spin = lambda g: dat*(am0+g*am1)-(dam0+g*dam1)*at
    ham = lambda g: -.5*(dam0+g*dam1)**2
    assert_allclose(spin(1)+spin(-1)-2*spin(0), 0., atol=1e-15)
    assert abs(ham(1)+ham(-1)-2*ham(0)) > .1


@pytest.mark.parametrize("crossed", [False, True])
@pytest.mark.parametrize("pp", [.9, 2.7, 6.2])
@pytest.mark.parametrize("hg", [-1, 1])
def test_qg_IF_actual_cell_normalization_and_signed_crossed_inverse(crossed, pp, hg):
    # q_out,g_out,q_in,g_in: APBC/PBC at K2=11, crossed inverse <0.
    modes = np.array([1, 6, 3, 4])
    pf, ko, pi_, ki = np.column_stack((pp*modes/11, np.array([[.1,.2],[.2,-.3],[.4,-.2],[-.1,.1]])))
    eo, ei = polarization(*ko, hg).conj()[2:], polarization(*ki, hg)[2:]
    kl, kr, el, er = (-ki, ko, ei, eo) if crossed else (ko, -ki, eo, ei)
    d = instantaneous_fermion_density(pf, pi_, kl, kr, .35, el, er, out_axes="q", in_axes="q")
    r2 = modes[2]-modes[1] if crossed else modes[2]+modes[3]
    expected = np.diag([11/(2*pi*sqrt(modes[1]*modes[3])*r2)*transverse_spin_factor(hq,hg,crossed=crossed)
                        for hq in (-1, 1)])
    assert_allclose(2*pp/(pi*11/pp)*d["canonical"], expected, atol=3e-15)
    assert_allclose(d["density_difference"], 0., atol=3e-15)


@pytest.mark.parametrize("h", list(product((-1, 1), repeat=4)))
def test_pair_emission_and_gluon_pair_IF_source_spin_and_cell_factor(h):
    pp, K = 2.7, 11
    # q -> q gg, one distinguished color order; other order is independent.
    po, k1, k2 = np.array([3, 2, 4])*pp/K
    pf, kl, kr = np.array([po,.1,.2]), np.array([k1,.2,-.3]), np.array([k2,.12,.05])
    pi_ = pf+kl+kr
    el, er = polarization(*kl,h[2]).conj()[2:], polarization(*kr,h[3]).conj()[2:]
    d = instantaneous_fermion_density(pf,pi_,kl,kr,.35,el,er,out_axes="q",in_axes="q")
    source = K/(2*pi*sqrt(2*4)*5)*pair_spin_factor(*h)
    assert_allclose(2*pp/(pi*K/pp)*d["canonical"][(h[0]+1)//2,(h[1]+1)//2], source, atol=2e-15)
    # g -> q anti g: field pf=q, pi=-anti, kL=out g, kR=-in g.
    pf, pa, kg = np.array([3*pp/K,.1,.2]),np.array([pp/K,.2,-.3]),np.array([2*pp/K,.12,.05])
    parent = pf+pa+kg
    el, er = polarization(*kg,h[2]).conj()[2:],polarization(*parent,h[3])[2:]
    direct, reverse = pair_scattering_spin(*h)
    for kl, kr, a, b, expected in ((kg,-parent,el,er,direct/(parent[0]-pa[0])),
                                  (-parent,kg,er,el,reverse/(-kg[0]-pa[0]))):
        d = instantaneous_fermion_density(pf,-pa,kl,kr,.35,a,b,out_axes="q",in_axes="qbar")
        value = d["canonical"][(h[0]+1)//2,(h[1]+1)//2]
        assert_allclose(value, expected/(2*sqrt(4*kg[0]*parent[0])), atol=3e-15)


@pytest.mark.parametrize("qa,qb", [(.7,-.7), (-.3,.3), (.7,-.4), (-.3,-.2)])
def test_gauss_density_direct_fourier_derivatives_and_longitudinal_improvement(qa,qb):
    ja, jb = np.array([[.3,.2j],[.7j,-.6]]), np.array([[.8,-.1j],[.2j,.4]])
    aa, ab = -ja/qa**2, -jb/qb**2
    expected = -.5*(1j*qa*aa)@(1j*qb*ab)-ja@ab
    result = instantaneous_gluon_density(qa,qb,ja@jb)
    assert_allclose(result["canonical"], expected, atol=3e-15)
    assert_allclose(result["canonical"]-result["source"],result["density_difference"],atol=3e-15)
    if qa == -qb:
        assert_allclose(result["density_difference"],0.,atol=1e-16)
    else:
        assert np.linalg.norm(result["density_difference"]) > .1


@pytest.mark.parametrize("pp", [.9,2.7,6.2])
def test_Gauss_current_source_cell_normalization_qq_qg_gg_and_pair(pp):
    from deuteron_wigner.quantum.m2_canonical_mixed_current import canonical_gluon_current_factor
    from deuteron_wigner.quantum.m2_instantaneous_sea_exit import sea_pair_spin_factor
    K, q2 = 11, 2
    currents = [1., canonical_gluon_current_factor(4,2),canonical_gluon_current_factor(6,4),sea_pair_spin_factor(1,-1)]
    for ja,jb in product(currents,repeat=2):
        # Both physically distinct current orders, before Wick or Fock.
        q = pp*q2/K
        c = instantaneous_gluon_density(q,-q,ja*jb)["canonical"]+instantaneous_gluon_density(-q,q,jb*ja)["canonical"]
        assert_allclose(2*pp/(pi*K/pp)*c, 2*K/(pi*q2**2)*ja*jb, atol=5e-15)


@pytest.mark.parametrize("h", list(product((-1,1),repeat=4)))
def test_A4_ordered_density_matches_existing_24_assignment_source(h):
    from deuteron_wigner.quantum.m2_excluded_four_gluon import ORDERS, quartic_ordered_color, quartic_spin_color
    modes,pp,K=np.array([2,4,4,2]),2.7,11
    eps=np.array([polarization(1.,0.,0.,v)[2:] for v in h]);eps[:2]=eps[:2].conj()
    c=sum(four_gluon_density(pp*modes[list(order)]/K,eps[list(order)],quartic_ordered_color(order))["canonical"] for order in ORDERS)
    # Divide by two only here, from the two normalized identical-boson states.
    assert_allclose(2*pp/(pi*K/pp)*c/2, K/(16*pi*sqrt(np.prod(modes)))*quartic_spin_color(*h),atol=3e-15)


def test_unrepresented_zero_modes_and_invalid_fields_are_not_silently_removed():
    with pytest.raises(ValueError,match="nonzero"):
        instantaneous_gluon_density(0.,.2,1.)
    with pytest.raises(ValueError,match="positive"):
        four_gluon_density([1.,2.,3.,-4.],np.ones((4,2)),1.)
    with pytest.raises(ValueError,match="zero constrained"):
        instantaneous_fermion_density([.7,0.,0.],[.4,0.,0.],[.2,0.,0.],[.4,0.,0.],.3,[1.,1j],[1.,-1j],out_axes="q",in_axes="q")

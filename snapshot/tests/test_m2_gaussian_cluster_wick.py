"""Common quartic-kernel trace, analytic derivatives and exact H subtraction."""
from itertools import product
from math import pi
import numpy as np
import pytest
from scipy.special import roots_hermite
from deuteron_wigner.quantum.m2_gaussian_cluster_wick import (
    GaussianClusterWick, cluster_wick_summands)
from deuteron_wigner.quantum.m2_canonical_quark_self import (
    CF, quark_wick_sums, antiquark_wick_sums, quark_mass_squared_direction)
from deuteron_wigner.quantum.m2_canonical_gluon_self import (
    gluon_wick_sums, gluon_mass_squared_direction)
from deuteron_wigner.quantum.m2_wick_one_body import WickOneBody
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density
from deuteron_wigner.quantum.m2_gaussian_field_sources import apply_field_source
from test_m2_gaussian_cluster_regulator import regulator
from test_m2_interacting_projector_current import pilot
from test_m2_source_vectors import probes


@pytest.mark.parametrize("p,k", [(1., 2.), (5., 2.), (2., 7.)])
@pytest.mark.parametrize("delta", [(0., 0.), (.12, .16), (.8, -.6)])
def test_off_conservation_trace_and_derivatives_by_differentiation_under_fixed_momentum_integral(p, k, delta):
    reg = regulator()
    delta, center = np.array(delta), np.array([.23, -.17])
    pf, pi_ = center+delta/2, center-delta/2
    b = reg.leg_transform((p, k, p, k))
    nodes, weights = roots_hermite(14)
    ids = np.array(list(product(range(len(nodes)), repeat=2)))
    z, width = nodes[ids], reg.width_GeV
    loop = k/p*center+width*np.sqrt(k/p)*z
    legs = np.stack((np.broadcast_to(pf, loop.shape), loop,
                     np.broadcast_to(pi_, loop.shape), loop), axis=1)
    form = np.exp(-np.sum(np.einsum("ab,nbj->naj", b, legs)**2, axis=(1, 2))/(2*width**2))
    measure = np.prod(weights[ids], axis=1)*np.exp(np.sum(z*z, axis=1))*width**2*k/p/(2*pi)**2
    kernel = measure*form
    args = dict(external_mode=p, loop_mode=k,
                outgoing_transverse_GeV=pf, incoming_transverse_GeV=pi_)
    np.testing.assert_allclose(kernel.sum(), reg.contracted_kernel_trace(**args), atol=3e-15)
    deriv = reg.contracted_kernel_derivatives(**args)
    # Differentiate Q at FIXED physical momenta, before contracting kT.
    dq_dp = 2*np.sum(loop**2, axis=1)/k-k*(pf@pf+pi_@pi_)/p**2
    dq_dk = -2*p*np.sum(loop**2, axis=1)/k**2+(pf@pf+pi_@pi_)/p
    dq_df = 2*(p+k)*pf/p-(pf+pi_+2*loop)
    dq_di = 2*(p+k)*pi_/p-(pf+pi_+2*loop)
    for key, dq in (("external_mode", dq_dp), ("loop_mode", dq_dk),
                    ("outgoing_transverse", dq_df), ("incoming_transverse", dq_di)):
        expected = np.einsum("n,n...->...", kernel, -dq/(2*width**2))
        np.testing.assert_allclose(deriv[key], expected, atol=3e-15)
    np.testing.assert_array_equal(deriv["outgoing_transverse"]+deriv["incoming_transverse"], 0.)
    for key in ("external_mode", "loop_mode"):
        eps = 1e-4
        def value(t):
            return reg.contracted_kernel_trace(**{**args, key:args[key]+t})
        finite = (-value(2*eps)+8*value(eps)-8*value(-eps)+value(-2*eps))/(12*eps)
        assert finite == pytest.approx(deriv[key], abs=2e-12)


@pytest.mark.parametrize("species,p", [("q", 1), ("qbar", 1), ("g", 2)])
@pytest.mark.parametrize("width", [.3, .8, 1.7])
def test_weighted_terms_against_independently_resolved_legacy_ordered_algebra(species, p, width):
    reg = regulator(width)
    got = cluster_wick_summands(species, p, regulator=reg, quark_flavors=2)
    # Recover each bare Wick term by increments of the independently
    # Fock-verified legacy sums, then integrate the new quartic kernel.
    names = ("IF", "IG", "A4") if species == "g" else ("IF", "IG")
    func = gluon_wick_sums if species == "g" else (
        quark_wick_sums if species == "q" else antiquark_wick_sums)
    for j, name in enumerate(names):
        modes, values = got[name]
        fermion_loop = (species == "g" and name == "IF") or (species != "g" and name == "IG")
        previous = 0.
        independent = []
        for n, mode in enumerate(modes, 1):
            kwargs = dict(boson_modes=8, fermion_modes=8)
            kwargs["fermion_modes" if fermion_loop else "boson_modes"] = n
            if species == "g":
                kwargs["quark_flavors"] = 2
            cumulative = func(p, **kwargs)[j]*(1 if species == "g" else CF)
            bare = cumulative-previous
            previous = cumulative
            # Independent radial Gaussian moment of F(p,k;p,k).
            independent.append(bare*width**2*mode/(4*pi*p))
        np.testing.assert_allclose(values, independent, atol=3e-15, rtol=2e-12)


def test_signed_large_modes_and_zero_transfer_subtraction_are_not_discarded():
    q = cluster_wick_summands("q", 1, regulator=regulator(), quark_flavors=2)
    a = cluster_wick_summands("qbar", 1, regulator=regulator(), quark_flavors=2)
    assert np.all(q["IF"][1] < 0.) and np.all(a["IF"][1] > 0.)
    assert q["IG"][1][0] < 0.  # zero direct transfer does not remove pair term
    q5 = cluster_wick_summands("q", 5, regulator=regulator(), quark_flavors=2)["IF"]
    assert np.any(q5[1] > 0.) and np.any(q5[1] < 0.)
    q_extended = cluster_wick_summands("q", 1, regulator=type(regulator())(
        width_GeV=.8, boson_modes=32, fermion_modes=8), quark_flavors=2)
    assert sum(q_extended["IF"][1]) < sum(q["IF"][1])  # no false UV limit


@pytest.mark.parametrize("species,p", [("q", 1), ("q", 5), ("qbar", 3)])
def test_IF_longitudinal_kernel_derivative_is_nonzero_and_needed_beside_canonical_squared_denominator(species, p):
    r = regulator()
    k = np.arange(2., 17., 2.)
    sign = -1 if species == "q" else 1
    def h(x):
        return CF*sum(r.contraction_trace_GeV2(external_mode=x, loop_mode=z)/
                      (z*(x+sign*z)) for z in k)
    eps = 1e-4
    finite = -(-h(p+2*eps)+8*h(p+eps)-8*h(p-eps)+h(p-2*eps))/(12*eps)
    trace = np.array([r.contraction_trace_GeV2(external_mode=p, loop_mode=z) for z in k])
    canonical = CF*sum(trace/(k*(p+sign*k)**2))
    regulator_term = CF*sum(trace/(p*k*(p+sign*k)))
    assert abs(regulator_term) > 1e-4
    assert finite == pytest.approx(canonical+regulator_term, abs=3e-12)
    assert abs(finite-canonical) > 1e-4


def operators(reg, *, subtract=True, q=.011, g=-.007):
    return GaussianClusterWick(regulator=reg, quark_flavors=2,
        normal_order_subtraction=subtract, finite_quark_mass2_per_gs2_GeV2=q,
        finite_gluon_mass2_per_gs2_GeV2=g)


@pytest.mark.parametrize("target", ["PROTON", "NEUTRON"])
def test_explicit_common_trace_subtraction_on_actual_retained_and_all_raw_sectors(target):
    p = pilot(1, target)
    expected = .011*quark_mass_squared_direction(p)-.007*gluon_mass_squared_direction(p)
    for width in (.3, .8, 1.7):
        op = operators(regulator(width))
        assert not isinstance(op, WickOneBody) and not op.complete_regulated_H2_matching
        np.testing.assert_allclose(op.retained_mass_coefficient(p), expected, atol=3e-15)
        raw = operators(regulator(width), subtract=False, q=0., g=0.).retained_mass_coefficient(p)
        np.testing.assert_allclose(raw, raw.conj().T, atol=3e-14)
        density = good_current_density(p, (.12, .16), charge="electromagnetic")
        np.testing.assert_allclose(raw@density, density@raw, atol=3e-13)
    small = pilot(0, target)
    for kind in ("qqqgg", "sea", "valence_IF", "valence_sea"):
        image = apply_field_source(small, probes(small, 1), kind=kind, out_cutoff=0,
            regulator=regulator(), total_pplus_GeV=1.3, total_transverse_GeV=(0., 0.))
        from deuteron_wigner.quantum.m2_excluded_fock_kinematics import SPECIES as FIVE
        from deuteron_wigner.quantum.m2_six_body_kinematics import SPECIES as SIX
        species = FIVE.get(image.space.sector, SIX.get(image.space.sector))
        values = np.array([sum((-.007 if s == "g" else .011)*9/m
            for s, m in zip(species, modes)) for modes in image.space.longitudinal])
        np.testing.assert_allclose(operators(regulator()).shell_mass_diagonal(image.space),
                                   values[image.space.coordinates[:, 0]], atol=3e-16)


@pytest.mark.parametrize("bad", [dict(normal_order_subtraction=1),
    dict(finite_quark_mass2_per_gs2_GeV2=np.nan), dict(quark_flavors=True), dict(regulator=None)])
def test_invalid_or_missing_finite_choices_are_not_defaulted(bad):
    args = dict(regulator=regulator(), quark_flavors=2, normal_order_subtraction=True,
        finite_quark_mass2_per_gs2_GeV2=0., finite_gluon_mass2_per_gs2_GeV2=0.)
    with pytest.raises(ValueError):
        GaussianClusterWick(**{**args, **bad})
    with pytest.raises(TypeError):
        GaussianClusterWick(regulator=regulator(), quark_flavors=2)

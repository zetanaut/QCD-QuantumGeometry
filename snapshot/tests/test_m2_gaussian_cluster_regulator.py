"""Boost-invariant correlated fields and their Wick/current building blocks."""
from itertools import product
from dataclasses import replace
from math import pi
import numpy as np
import pytest
from scipy.special import roots_hermite

from deuteron_wigner.quantum.m2_gaussian_field_regulator import (
    GaussianClusterRegulator, gaussian_shell_integral, gaussian_field_direction_vertex)
from deuteron_wigner.quantum.m2_gaussian_field_vertices import cubic_field_orbital, _field_data
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import split_routes, split_frame, frame, SPECIES
from deuteron_wigner.quantum.m2_gaussian_field_sources import apply_field_source
from deuteron_wigner.quantum.m2_excluded_fock_vertices import cubic_mass_kernel
from deuteron_wigner.quantum.m2_matched_h2_reduction import nested_pilot_isometry
from test_m2_gaussian_field_current import controls
from test_m2_gaussian_field_regulator import regulator as field_regulator
from test_m2_source_vectors import CASES, source, probes
from test_m2_interacting_projector_current import pilot


def regulator(width=.8):
    return GaussianClusterRegulator(width_GeV=width, boson_modes=8, fermion_modes=8)


@pytest.mark.parametrize("modes", [(3, 2, 5), (1, 2, 2, 5), (3, 2, 3, 2), (3, 5, 1, 7)])
def test_correlated_kernel_is_variance_and_preserves_both_kinematic_boosts(modes):
    w = np.array(modes)
    b = regulator().leg_transform(w)
    p = np.random.default_rng(sum(w)).normal(size=(len(w), 2))
    expected = (w.sum()*np.sum(p*p/w[:, None])-np.sum(p.sum(axis=0)**2))/2
    np.testing.assert_allclose(np.sum((b@p)**2), expected, atol=2e-14)
    np.testing.assert_allclose(b@w, 0., atol=4e-15)
    np.testing.assert_allclose(b@(p+w[:, None]*np.array([.35, -.21])), b@p, atol=3e-15)
    np.testing.assert_allclose(regulator().leg_transform(3.7*w), b, atol=1e-15)


@pytest.mark.parametrize("p,k", [(1, 2), (5, 2), (2, 7), (4, 6)])
@pytest.mark.parametrize("width", [.3, .8, 1.7])
def test_cluster_Wick_trace_from_actual_four_field_kernel_with_oblique_external_momentum(p, k, width):
    r = regulator(width)
    b = r.leg_transform((p, k, p, k))
    external = np.array([.23, -.17])
    nodes, weights = roots_hermite(12)
    ids = np.array(list(product(range(len(nodes)), repeat=2)))
    z = nodes[ids]
    loop = k/p*external+width*np.sqrt(k/p)*z
    legs = np.stack((np.broadcast_to(external, loop.shape), loop,
                     np.broadcast_to(external, loop.shape), loop), axis=1)
    form = np.exp(-np.sum(np.einsum("ab,nbj->naj", b, legs)**2, axis=(1, 2))/(2*width**2))
    measured = np.sum(np.prod(weights[ids], axis=1)*np.exp(np.sum(z*z, axis=1))*form)*width**2*k/p/(2*pi)**2
    expected = r.contraction_trace_GeV2(external_mode=p, loop_mode=k)
    np.testing.assert_allclose(measured, expected, rtol=5e-14)
    assert abs(expected-width**2/(4*pi)) > .01*expected
    # The constant field-leg trace would change each signed Wick summand.
    assert np.isclose(expected/(width**2/(4*pi)), k/p)


@pytest.mark.parametrize("channel", ["q_to_qg", "g_to_gg", "g_to_pair"])
def test_actual_source_maps_reduce_to_active_invariant_mass_and_keep_inherited_shells(channel):
    route = split_routes(9, channel)[0]
    f, tree = frame(route.sector, route.outgoing), split_frame(route)
    fields, shifts, _, eligible = _field_data(f, route.daughters, route.incoming[route.parent],
        "q" if channel == "q_to_qg" else "g", regulator=regulator(),
        total_pplus_GeV=1.3, total_transverse_GeV=(.23, -.17), species=SPECIES[route.sector])
    x = route.incoming[route.parent]/9
    assert eligible
    np.testing.assert_allclose(fields.T@fields, x*np.outer(tree[-1], tree[-1]), atol=8e-16)
    np.testing.assert_allclose(shifts, 0., atol=8e-17)
    hs = (1, 1, 1) if channel != "g_to_pair" else (1, -1, 1)
    args = dict(regulator=regulator(), total_pplus_GeV=1.3, total_transverse_GeV=(0., 0.))
    a = cubic_field_orbital(route, 1, 1, .4, .35, hs, **args)
    moved = cubic_field_orbital(route, 1, 1, .4, .35, hs,
        **{**args, "total_pplus_GeV":2.6, "total_transverse_GeV":(.23, -.17)})
    np.testing.assert_allclose(a, moved, atol=3e-16)
    low = cubic_field_orbital(route, 1, 0, .4, .35, hs, **args)
    np.testing.assert_allclose(low[:, 1:], 0., atol=3e-16)


@pytest.mark.parametrize("target,jz", [("PROTON", 1), ("NEUTRON", -1)])
@pytest.mark.parametrize("kind", CASES)
def test_actual_cluster_source_full_statistics_boost_and_bare_limit(kind, target, jz):
    p = pilot(0, target, jz)
    v = probes(p, 1)
    args = dict(kind=kind, out_cutoff=1, regulator=regulator(),
                total_pplus_GeV=1.3, total_transverse_GeV=(0., 0.))
    a = apply_field_source(p, v, **args)
    moved = apply_field_source(p, v, **{**args, "total_pplus_GeV":2.6,
                                        "total_transverse_GeV":(.23, -.17)})
    near_bare = apply_field_source(p, v, **{**args, "regulator":regulator(1e7)})
    np.testing.assert_allclose(moved.values, a.values, atol=3e-15)
    np.testing.assert_allclose(a.space.statistics_project(a.values), a.values, atol=3e-14)
    np.testing.assert_allclose(near_bare.values, source(p, kind, 1, v).values, atol=3e-14)
    assert a.field_regulator == regulator()
    assert not a.complete_regulated_H2_matching


def test_cluster_keeps_bare_parent_selection_and_field_leg_off_frame_rejects_incomplete_Jz():
    p, small = pilot(1), pilot(0)
    e = nested_pilot_isometry(small, p)
    v = probes(p, 1)
    v -= e@(e.conj().T@v)
    args = dict(kind="qqqgg", out_cutoff=0, regulator=regulator(),
                total_pplus_GeV=1.3, total_transverse_GeV=(.23, -.17))
    np.testing.assert_allclose(apply_field_source(p, v, **args).values, 0., atol=3e-15)
    with pytest.raises(ValueError, match="additional Jz"):
        apply_field_source(p, v, **{**args, "regulator":field_regulator()})
    r = replace(field_regulator(), reference_transverse_velocity=(.23/1.3, -.17/1.3))
    field = apply_field_source(p, v, **{**args, "regulator":r})
    assert np.linalg.norm(field.values) > 1e-6


@pytest.mark.parametrize("leg", range(3))
@pytest.mark.parametrize("delta", [(0., 0.), (.12, .16), (.8, -.6)])
def test_physical_cluster_leg_current_moves_all_correlated_rows(leg, delta):
    args = controls()
    modes = (3, 2, 5)
    b = regulator().leg_transform(modes)
    original_maps, original_shifts = args["field_maps"], args["field_shifts"]
    args.update(field_maps=b@original_maps, field_shifts=b@original_shifts)
    direction = b[:, leg]
    current = gaussian_field_direction_vertex(**args, direction=direction,
                                               transfer_GeV=delta, quadrature_order=24)
    higher = gaussian_field_direction_vertex(**args, direction=direction,
                                              transfer_GeV=delta, quadrature_order=40)
    np.testing.assert_allclose(current, higher, atol=4e-15)
    final = args["field_shifts"]+direction[:, None]*delta
    change = gaussian_shell_integral(**{**args, "field_shifts":final})-gaussian_shell_integral(**args)
    np.testing.assert_allclose(np.einsum("j,joi->oi", delta, current), change, atol=3e-15)
    if not np.any(delta):
        for j in range(2):
            step = np.zeros(2)
            step[j] = 1e-5
            plus, minus = (gaussian_shell_integral(**{**args, "field_shifts":
                args["field_shifts"]+sign*direction[:, None]*step}) for sign in (1, -1))
            np.testing.assert_allclose(current[j], (plus-minus)/2e-5, atol=3e-10)
    else:
        wrong = np.eye(3)[:, leg]
        omitted_correlations = gaussian_field_direction_vertex(**args, direction=wrong,
            transfer_GeV=delta, quadrature_order=24)
        assert np.linalg.norm(np.einsum("j,joi->oi", delta, omitted_correlations)-change) > 1e-4


def test_correlated_kernel_collective_boost_direction_has_no_regulator_current():
    args = controls()
    w = np.array([3, 2, 5])
    b = regulator().leg_transform(w)
    args.update(field_maps=b@args["field_maps"], field_shifts=b@args["field_shifts"])
    current = gaussian_field_direction_vertex(**args, direction=b@w,
        transfer_GeV=(.12, .16), quadrature_order=16)
    np.testing.assert_allclose(current, 0., atol=2e-15)


@pytest.mark.parametrize("channel", ["q_to_qg", "g_to_gg", "g_to_pair"])
@pytest.mark.parametrize("width", [.3, .8, 1.7])
def test_actual_cubic_source_shell_resolution_against_independent_radial_Parseval_sum(channel, width):
    route = split_routes(9, channel)[0]
    f = frame(route.sector, route.outgoing)
    x, y = f.fractions[list(route.daughters)]
    active = x+y
    b = .4
    hs = (1, 1, 1) if channel != "g_to_pair" else (1, -1, 1)
    c = cubic_mass_kernel(route, (0., 0.), .35, hs, total_pplus=1.3)
    fx = cubic_mass_kernel(route, (1., 0.), .35, hs, total_pplus=1.3)-c
    fy = cubic_mass_kernel(route, (0., 1.), .35, hs, total_pplus=1.3)-c
    t = active*b*b/width**2
    ratio2 = ((1-t)/(1+t))**2
    norm_inf = x*y/active*(abs(c)**2*width**2/(4*pi*active)
                +(abs(fx)**2+abs(fy)**2)*width**4/(8*pi*active**2))
    previous = 0.
    for cutoff in range(5):
        # One relative circular mode, with inherited parent in its ground.
        # Complete-shell Jacobi recoupling must preserve this norm exactly.
        n0 = cutoff//2
        s0 = b*b/pi/(1+t)**2*sum(ratio2**n for n in range(n0+1))
        n1 = (cutoff-1)//2
        s1 = 2*b**4/pi/(1+t)**4*sum((n+1)*ratio2**n for n in range(n1+1))
        reference = x*y/active*(abs(c)**2*s0+(abs(fx)**2+abs(fy)**2)*s1)
        orbital = cubic_field_orbital(route, 0, cutoff, b, .35, hs,
            regulator=regulator(width), total_pplus_GeV=1.3, total_transverse_GeV=(.23, -.17))
        actual = np.vdot(orbital[:, 0], orbital[:, 0]).real
        assert actual == pytest.approx(reference, abs=3e-15, rel=3e-13)
        assert actual+3e-15 >= previous
        assert actual <= norm_inf+3e-15
        previous = actual
    # This convergent SOURCE norm is not a loop/current/physical UV claim.
    assert norm_inf > 0

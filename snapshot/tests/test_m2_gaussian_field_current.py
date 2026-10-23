"""Regulator currents are differentiated integrals, not fitted Ward repairs."""
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_gaussian_field_regulator import (
    gaussian_shell_integral, gaussian_field_leg_vertex)
from deuteron_wigner.quantum.m2_gaussian_field_vertices import _field_data
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import (
    split_routes, split_frame, frame, SPECIES)
from deuteron_wigner.quantum.m2_excluded_fock_vertices import cubic_mass_kernel
from test_m2_gaussian_field_regulator import regulator


def controls():
    return dict(out_cutoff=2, in_cutoff=2, out_map=np.eye(2), in_map=np.eye(2),
        b_GeV=.4, field_precision=1/.8**2,
        field_maps=np.array([[.3, -.4], [.7, .2], [.2, .5]]),
        field_shifts=np.array([[.1, .2], [-.15, .1], [.13, -.22]]),
        out_shift=np.array([[.05, .12], [-.2, .07]]), in_shift=np.zeros((2, 2)),
        constant=.2+.13j, linear=np.array([[.17+.1j, -.21], [.11, .24j]]))


@pytest.mark.parametrize("leg", [0, 1, 2])
@pytest.mark.parametrize("delta", [(0., 0.), (1e-9, -2e-9), (.12, .16), (.8, -.6)])
def test_leg_Ward_and_path_convergence_on_shifted_quadratic_integrals(leg, delta):
    args = controls()
    a = gaussian_field_leg_vertex(**args, leg=leg, transfer_GeV=delta, quadrature_order=16)
    b = gaussian_field_leg_vertex(**args, leg=leg, transfer_GeV=delta, quadrature_order=32)
    np.testing.assert_allclose(a, b, atol=3e-15, rtol=3e-12)
    changed = args["field_shifts"].copy()
    changed[leg] += delta
    expected = gaussian_shell_integral(**{**args, "field_shifts":changed})-gaussian_shell_integral(**args)
    np.testing.assert_allclose(np.einsum("j,joi->oi", delta, a), expected, atol=2e-15, rtol=3e-11)
    # Reverse the path, matrices and complex canonical polynomial.
    reverse = {**args, "out_map":args["in_map"], "in_map":args["out_map"],
        "out_shift":args["in_shift"], "in_shift":args["out_shift"],
        "constant":args["constant"].conjugate(), "linear":args["linear"].conj(),
        "field_shifts":changed}
    rev = gaussian_field_leg_vertex(**reverse, leg=leg, transfer_GeV=-np.array(delta),
                                     quadrature_order=16)
    np.testing.assert_allclose(a, rev.conj().swapaxes(1, 2), atol=2e-15)


def test_zero_transfer_is_the_direct_nonzero_derivative_not_division_by_transfer():
    args = controls()
    a = gaussian_field_leg_vertex(**args, leg=1, transfer_GeV=(0., 0.), quadrature_order=8)
    assert np.linalg.norm(a) > 1e-3
    eps = 1e-5
    for j in (0, 1):
        plus, minus = args["field_shifts"].copy(), args["field_shifts"].copy()
        plus[1, j] += eps
        minus[1, j] -= eps
        finite = (gaussian_shell_integral(**{**args, "field_shifts":plus})
                 -gaussian_shell_integral(**{**args, "field_shifts":minus}))/(2*eps)
        np.testing.assert_allclose(a[j], finite, atol=3e-11, rtol=3e-8)


@pytest.mark.parametrize("channel", ["q_to_qg", "g_to_gg", "g_to_pair"])
def test_actual_cubic_source_polynomial_and_global_Jacobi_maps(channel):
    route = split_routes(9, channel)[0]
    f, tree = frame(route.sector, route.outgoing), split_frame(route)
    fields, shifts, precision, eligible = _field_data(f, route.daughters, route.incoming[route.parent],
        "q" if channel == "q_to_qg" else "g", regulator=regulator(),
        total_pplus_GeV=1.3, total_transverse_GeV=(.04, -.03), species=SPECIES[route.sector])
    assert eligible
    hs = (1, 1, 1) if channel != "g_to_pair" else (1, -1, 1)
    c0 = cubic_mass_kernel(route, (0., 0.), .35, hs)
    cx = cubic_mass_kernel(route, (1., 0.), .35, hs)-c0
    cy = cubic_mass_kernel(route, (0., 1.), .35, hs)-c0
    args = dict(out_cutoff=1, in_cutoff=1, out_map=np.eye(4), in_map=tree[:3],
        b_GeV=.4, field_maps=fields, field_shifts=shifts, field_precision=precision,
        constant=c0, linear=np.array([cx*tree[-1], cy*tree[-1]]))
    delta = np.array([.12, .16])
    for leg in range(3):
        current = gaussian_field_leg_vertex(**args, leg=leg, transfer_GeV=delta, quadrature_order=16)
        shifted = shifts.copy()
        shifted[leg] += delta
        expected = gaussian_shell_integral(**{**args, "field_shifts":shifted})-gaussian_shell_integral(**args)
        np.testing.assert_allclose(np.einsum("j,joi->oi", delta, current), expected, atol=3e-16)
        assert np.linalg.norm(current) > 1e-6


def test_two_field_product_rule_keeps_both_endpoints_and_cannot_double_count():
    args = controls()
    args["field_maps"] = np.array([[.3, -.4], [.3, -.4]])
    args["field_shifts"] = np.array([[.1, .2], [.1, .2]])
    delta = np.array([.12, .16])
    # (r r)' = r' r_i + r_f r': each field occurrence appears once.
    one = gaussian_field_leg_vertex(**args, leg=0, transfer_GeV=delta, quadrature_order=16)
    middle = args["field_shifts"].copy()
    middle[0] += delta
    two = gaussian_field_leg_vertex(**{**args, "field_shifts":middle},
                                    leg=1, transfer_GeV=delta, quadrature_order=16)
    final = middle.copy()
    final[1] += delta
    expected = gaussian_shell_integral(**{**args, "field_shifts":final})-gaussian_shell_integral(**args)
    np.testing.assert_allclose(np.einsum("j,joi->oi", delta, one+two), expected, atol=2e-15)
    assert np.linalg.norm(np.einsum("j,joi->oi", delta, one)-expected) > 1e-4
    assert np.linalg.norm(np.einsum("j,joi->oi", delta, 2*(one+two))-expected) > 1e-4


def test_bare_field_limit_has_no_extra_regulator_vertex():
    args = {**controls(), "field_precision":0.}
    np.testing.assert_array_equal(gaussian_field_leg_vertex(**args, leg=0,
        transfer_GeV=(.12, .16), quadrature_order=8), 0.)

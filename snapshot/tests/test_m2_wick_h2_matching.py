"""Actual-source H2 Wick matching, raw backflow, finite subtraction and currents."""
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_wick_h2_matching import (
    WickMatchedFourthOrderH2, direct_minus_coefficients_between,
    minus_coefficients_between, minus_current_between, plus_current_between,
    transverse_current_between,
)
from deuteron_wigner.quantum.m2_fourth_order_minus_current import minus_coefficients_between as bare_coefficients
from deuteron_wigner.quantum.m2_fourth_order_minus_current import minus_current_between as bare_current
from deuteron_wigner.quantum.m2_shell_minus_current import shell_minus_current_between
from deuteron_wigner.quantum.m2_excluded_minus_contact import build_excluded_minus_contact
from test_m2_fourth_order_transverse_current import model
from test_m2_wick_one_body import operators


@pytest.mark.parametrize("subtract", [False, True])
def test_H_norm_and_full_raw_extra_wave(subtract):
    base = model("PROTON", 1)
    matched = WickMatchedFourthOrderH2(base, operators=operators(subtract=subtract))
    old, s = base.snapshot(1.4), matched.snapshot(1.4)
    extra = {n: s.canonical_snapshot.five_wave_coefficients[2][n]-old.five_wave_coefficients[2][n]
             for n in base.spaces5}
    h4 = np.zeros_like(base.pilot.matrix)
    n4 = np.zeros_like(h4)
    for n, space in base.spaces5.items():
        a1 = old.five_wave_coefficients[0][n]
        source = base.qq.sources_by_power[1][n]
        free = space.free_energies(.4, .35, 0.)
        expected = matched.qq[n][:, None]*source/(1.4-free[:, None])**2
        np.testing.assert_allclose(extra[n], expected, atol=2e-17)
        h4 += source.conj().T@expected
        n4 += a1.conj().T@expected+expected.conj().T@a1
        assert np.linalg.norm(expected) > 1e-6
    np.testing.assert_allclose(s.effective_hamiltonian,
        old.effective_hamiltonian+.09*matched.pp+.3**4*h4, atol=5e-16)
    np.testing.assert_allclose(s.metric, old.metric+.3**4*n4, atol=3e-16)
    eps = 2e-5
    derivative = (matched.snapshot(1.4+eps).effective_hamiltonian
                  -matched.snapshot(1.4-eps).effective_hamiltonian)/(2*eps)
    np.testing.assert_allclose(derivative, np.eye(len(s.matrix))-s.metric, atol=3e-11)
    w = s.canonical_snapshot.whitener
    np.testing.assert_allclose(w@s.metric@w, np.eye(len(s.matrix)), atol=6e-15)
    # Q6 one-body excursions first enter at g^6, not g^4.
    for n in base.spaces6:
        np.testing.assert_array_equal(s.canonical_snapshot.six_wave_coefficient[n], old.six_wave_coefficient[n])


@pytest.mark.parametrize("charge,q", [("electromagnetic", 1), ("baryon", 1), ("up_number", 2), ("down_number", 1)])
def test_normalized_charge_includes_the_new_induced_metric(charge, q):
    m = WickMatchedFourthOrderH2(model("PROTON", 1), operators=operators())
    s = m.snapshot(1.4)
    np.testing.assert_allclose(s.current((0., 0.), charge=charge), q*np.eye(len(s.matrix)), atol=8e-15)


@pytest.fixture(scope="module", params=[("PROTON", 1), ("PROTON", -1), ("NEUTRON", -1)])
def endpoints(request):
    target, jz = request.param
    left, right = model(target, jz), model(target, 1)
    op = operators()
    sf = WickMatchedFourthOrderH2(left, operators=op).snapshot(1.39)
    si = WickMatchedFourthOrderH2(right, operators=op).snapshot(1.47)
    delta, P = (.12, .16), 1.3
    args = dict(total_pplus_GeV=P, charge="electromagnetic", loop_flavor_charges=[2/3, -1/3])
    coefficients = minus_coefficients_between(sf, si, delta, **args)
    return sf, si, delta, args, coefficients


def test_minus_order_changes_against_independent_raw_backflow_products(endpoints):
    f, i, delta, args, result = endpoints
    basef, basei = f.owner.base.snapshot(f.energy_GeV2), i.owner.base.snapshot(i.energy_GeV2)
    original = bare_coefficients(basef, basei, delta, total_pplus_GeV=args["total_pplus_GeV"], charge=args["charge"])
    direct = direct_minus_coefficients_between(f, i, delta, **args)
    for order in (0, 1, 3):
        np.testing.assert_array_equal(result[order], original[order])
    np.testing.assert_allclose(result[2]-original[2], direct[2], atol=2e-15)
    expected = direct[4].copy()
    for n, space in basei.owner.spaces5.items():
        df = f.canonical_snapshot.five_wave_coefficients[2][n]-basef.five_wave_coefficients[2][n]
        di = i.canonical_snapshot.five_wave_coefficients[2][n]-basei.five_wave_coefficients[2][n]
        af, ai = basef.five_wave_coefficients[0][n], basei.five_wave_coefficients[0][n]
        joined = shell_minus_current_between(basef.owner.spaces5[n], space,
            np.column_stack((ai, di)), delta, b_GeV=.4, mass_GeV=.35,
            total_pplus_GeV=1.3, charge="electromagnetic")
        size = len(i.matrix)
        expected += df.conj().T@joined[:, :size]+af.conj().T@joined[:, size:]
        forward = build_excluded_minus_contact(basei.owner.pilot, basef.owner.spaces5[n], delta,
            total_pplus_GeV=1.3, charge="electromagnetic")
        backward = build_excluded_minus_contact(basef.owner.pilot, space, tuple(-x for x in delta),
            total_pplus_GeV=1.3, charge="electromagnetic")
        expected[:, forward.retained_columns] += df.conj().T@forward.current_per_gs
        expected[backward.retained_columns, :] += backward.current_per_gs.conj().T@di
    np.testing.assert_allclose(result[4]-original[4], expected, atol=2e-15)
    # The raw-wave change is not redundant with a direct local counterterm.
    assert np.linalg.norm(expected-direct[4]) > 1e-9


def test_two_endpoint_reversal_normalization_and_kinematic_boost(endpoints):
    f, i, delta, args, c = endpoints
    reverse = minus_coefficients_between(i, f, tuple(-x for x in delta), **args)
    for n in c:
        np.testing.assert_allclose(c[n], reverse[n].conj().T, atol=2e-14)
    raw = sum(.3**n*a for n, a in c.items())
    full = minus_current_between(f, i, delta, **args, normalized=True)
    np.testing.assert_allclose(full, f.canonical_snapshot.whitener@raw@i.canonical_snapshot.whitener, atol=0.)
    beta = np.array([.06, -.09])/1.3
    boosted = minus_current_between(f, i, delta, **args, normalized=False, mean_total_transverse=beta*1.3)
    jt = transverse_current_between(f, i, delta, total_pplus_GeV=1.3,
                                     charge="electromagnetic", normalized=False)
    jp = plus_current_between(f, i, delta, charge="electromagnetic", normalized=False)
    np.testing.assert_allclose(boosted-raw, np.einsum("j,joi->oi", beta, jt)+(beta@beta)/2*jp, atol=2e-14)


def test_zero_finite_normal_order_recovers_entire_previous_model():
    base = model("PROTON", 1)
    old = base.snapshot(1.4)
    op = operators(subtract=True, muq=0., mug=0.)
    new = WickMatchedFourthOrderH2(base, operators=op).snapshot(1.4)
    for attr in ("effective_hamiltonian", "metric", "matrix"):
        np.testing.assert_array_equal(getattr(new, attr), getattr(old, attr))
    args = dict(total_pplus_GeV=1.3, charge="electromagnetic")
    actual = minus_current_between(new, new, (.12, .16), **args,
        loop_flavor_charges=[2/3, -1/3], normalized=True)
    expected = bare_current(old, old, (.12, .16), **args, normalized=True)
    np.testing.assert_allclose(actual, expected, atol=0.)


def test_subtracted_finite_model_is_trace_independent_but_not_zero():
    base = model("PROTON", 1)
    left = WickMatchedFourthOrderH2(base, operators=operators(subtract=True, width=.4)).snapshot(1.4)
    right = WickMatchedFourthOrderH2(base, operators=operators(subtract=True, width=1.6)).snapshot(1.4)
    old = base.snapshot(1.4)
    for name in ("matrix", "metric", "effective_hamiltonian"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    assert np.linalg.norm(left.matrix-old.matrix) > .01
    args = dict(total_pplus_GeV=1.3, charge="electromagnetic", loop_flavor_charges=[2/3, -1/3], normalized=True)
    np.testing.assert_allclose(minus_current_between(left, left, (.12, .16), **args),
                               minus_current_between(right, right, (.12, .16), **args), atol=0.)


def test_explicit_counterterms_and_complete_current_interface_required():
    base = model("PROTON", 1)
    f = WickMatchedFourthOrderH2(base, operators=operators()).snapshot(1.4)
    i = WickMatchedFourthOrderH2(base, operators=operators(subtract=True)).snapshot(1.4)
    with pytest.raises(ValueError):
        plus_current_between(f, i, (0., 0.), charge="baryon", normalized=True)
    with pytest.raises(ValueError):
        bare_current(f, f, (.12, .16), total_pplus_GeV=1.3, charge="electromagnetic", normalized=True)
    assert not f.owner.physical and not f.owner.common_regulator_renormalization_complete


def test_snapshot_reuse_preserves_the_original_and_requires_exact_source():
    base = model("PROTON", 1)
    old = base.snapshot(1.4)
    arrays = [old.matrix, old.metric, old.effective_hamiltonian,
              *(v for order in old.five_wave_coefficients for v in order.values())]
    copies = [a.copy() for a in arrays]
    matched = WickMatchedFourthOrderH2(base, operators=operators())
    reused = matched.match_snapshot(old)
    fresh = matched.snapshot(1.4)
    for key in ("matrix", "metric", "effective_hamiltonian"):
        np.testing.assert_array_equal(getattr(reused, key), getattr(fresh, key))
    for a, before in zip(arrays, copies):
        np.testing.assert_array_equal(a, before)
    with pytest.raises(ValueError, match="exact canonical base"):
        matched.match_snapshot(model("NEUTRON", 1).snapshot(1.4))
    with pytest.raises(ValueError, match="exact canonical base"):
        matched.match_snapshot(reused)


def test_zero_coupling_leaves_H_norm_and_full_current_unchanged():
    from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
    from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
    from deuteron_wigner.quantum.m2_canonical_qq import CanonicalQQSourceAction, CanonicalQQInsertion
    from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
    from deuteron_wigner.quantum.m2_fourth_order_h2 import FourthOrderH2
    p = build_coupled_pilot(total_k2=9, cutoff=0, target="PROTON", jz_twice=1,
        b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=0.)
    ref = build_order_resolved_reference(p, out_cutoff=0)
    qq = CanonicalQQInsertion(ref, action=CanonicalQQSourceAction(ref))
    base = FourthOrderH2(ref, QQ_insertion=qq,
        six_body=CanonicalSixBodyExcursion(ref, out_cutoff=0))
    old = base.snapshot(1.4)
    new = WickMatchedFourthOrderH2(base, operators=operators()).match_snapshot(old)
    for key in ("matrix", "metric", "effective_hamiltonian"):
        np.testing.assert_array_equal(getattr(new, key), getattr(old, key))
    args = dict(total_pplus_GeV=1.3, charge="electromagnetic", normalized=True)
    np.testing.assert_array_equal(
        minus_current_between(new, new, (.12, .16), **args, loop_flavor_charges=[2/3, -1/3]),
        bare_current(old, old, (.12, .16), **args))

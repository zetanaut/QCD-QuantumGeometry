"""Retained-only resolution at fixed actual Q spaces and two H2 metrics."""
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
import pytest

from scripts.benchmark_m2_gaussian_cluster_four_current import build_model
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import cluster_plus_density_between
from deuteron_wigner.quantum.m2_gaussian_retained_resolution import retained_metric_injection, retained_ward_boundary
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum import m2_excluded_fock_kinematics as five
from deuteron_wigner.quantum import m2_six_body_kinematics as six


@pytest.fixture(scope='module')
def endpoints():
    lower, lv = build_model('Q5_Q6_shell1')
    larger, hv = build_model('retained_shell1')
    assert hv == {**lv, 'retained': 1}
    result = []
    for energy in (1.39, 1.47):
        a, b = lower.snapshot(energy), larger.snapshot(energy)
        result.append((a, b, retained_metric_injection(a, b)))
    return result


def test_exact_raw_wave_H_and_norm_restrictions_and_metric_isometry(endpoints):
    for a, b, mapping in endpoints:
        e, t = mapping.raw, mapping.normalized
        assert e.shape == t.shape == (204, 37)
        np.testing.assert_allclose(e.conj().T@e, np.eye(37), atol=5e-13, rtol=0.)
        np.testing.assert_allclose(t.conj().T@t, np.eye(37), atol=5e-13, rtol=0.)
        w, u = np.linalg.eigh(b.metric)
        assert min(w) > 0
        sqrt_high = (u*np.sqrt(w))@u.conj().T
        w, u = np.linalg.eigh(a.metric)
        assert min(w) > 0
        invsqrt_low = (u/np.sqrt(w))@u.conj().T
        np.testing.assert_allclose(t, sqrt_high@e@invsqrt_low, atol=5e-14, rtol=3e-12)
        np.testing.assert_allclose(b.whitener@t, e@a.whitener, atol=5e-14, rtol=3e-12)
        np.testing.assert_allclose(t.conj().T@b.matrix@t, a.matrix, atol=8e-14, rtol=3e-12)


def test_both_energy_metric_injections_preserve_actual_matched_good_current(endpoints):
    (af, bf, ef), (ai, bi, ei) = endpoints
    delta = (.12, .16)
    for normalized in (False, True):
        low = cluster_plus_density_between(af, ai, delta, charge='electromagnetic', normalized=normalized)
        high = cluster_plus_density_between(bf, bi, delta, charge='electromagnetic', normalized=normalized)
        mf, mi = (ef.normalized, ei.normalized) if normalized else (ef.raw, ei.raw)
        np.testing.assert_allclose(mf.conj().T@high@mi, low, atol=8e-14, rtol=3e-12)
        if normalized:
            naive = np.linalg.norm(ef.raw.conj().T@high@ei.raw-low)
            assert 1e-10 < naive < 1e-7
            print('naive raw-map normalized-current mismatch', naive, flush=True)
    separation = np.linalg.norm(ef.normalized-ei.normalized)
    assert separation > 1e-6
    print('endpoint metric-map difference', separation, flush=True)


def test_mixed_energy_or_Q_resolution_is_not_mislabeled_retained_only(endpoints):
    (low, high, _), _ = endpoints
    with pytest.raises(ValueError, match='equal energy'):
        retained_metric_injection(low, replace(high, energy_GeV2=1.47))
    baseline, _ = build_model('baseline')
    with pytest.raises(ValueError, match='fixed Q coordinates'):
        retained_metric_injection(baseline.snapshot(1.39), high)


def test_retained_boundary_is_independent_compressed_commutator_difference(endpoints):
    (af, bf, mf), (ai, bi, mi) = endpoints
    np.testing.assert_array_equal(mf.raw, mi.raw)
    e = mf.raw
    rho = good_current_density_between(bf.owner.pilot, bi.owner.pilot, (.12, .16), charge='electromagnetic')
    lowrho = good_current_density_between(af.owner.pilot, ai.owner.pilot, (.12, .16), charge='electromagnetic')
    np.testing.assert_allclose(e.conj().T@rho@e, lowrho, atol=4e-14, rtol=3e-12)
    low = af.effective_hamiltonian@lowrho-lowrho@ai.effective_hamiltonian
    high = bf.effective_hamiltonian@rho-rho@bi.effective_hamiltonian
    boundary = retained_ward_boundary(bf.effective_hamiltonian, bi.effective_hamiltonian, rho, e)
    np.testing.assert_allclose(low-e.conj().T@high@e, boundary, atol=5e-14, rtol=3e-12)
    # At equal retained Jz, density preserves every constituent spin.
    # The newly added N=1 orbitals have Lz=+/-1, so this density cannot
    # connect them to the N=0 columns inside the same Jz block. This is a
    # selection zero of the Ward contraction, not four-current equality.
    q = np.eye(len(e))-e@e.conj().T
    np.testing.assert_allclose(q@rho@e, 0., atol=4e-14, rtol=0.)
    np.testing.assert_allclose(e.conj().T@rho@q, 0., atol=4e-14, rtol=0.)
    np.testing.assert_allclose(boundary, 0., atol=4e-14, rtol=0.)
    # Both endpoint maps turn this raw difference into the corresponding
    # normalized contraction; neither changes the finite projector physics.
    np.testing.assert_allclose(af.whitener@boundary@ai.whitener,
        af.whitener@low@ai.whitener-mf.normalized.conj().T@bf.whitener@high@bi.whitener@mi.normalized,
        atol=5e-14, rtol=3e-12)
    print('retained-only Ward-boundary norm', np.linalg.norm(boundary), flush=True)


def test_boundary_formula_for_independent_complex_rectangular_coordinates():
    rng = np.random.default_rng(42)
    e = np.linalg.qr(rng.normal(size=(7, 3))+1j*rng.normal(size=(7, 3)))[0]
    hf, hi, rho = [rng.normal(size=(7, 7))+1j*rng.normal(size=(7, 7)) for _ in range(3)]
    low = [e.conj().T@a@e for a in (hf, hi, rho)]
    expected = low[0]@low[2]-low[2]@low[1]-e.conj().T@(hf@rho-rho@hi)@e
    assert np.linalg.norm(expected) > 1.
    np.testing.assert_allclose(retained_ward_boundary(hf, hi, rho, e), expected, atol=2e-14, rtol=3e-12)
    with pytest.raises(ValueError, match='isometry'):
        retained_ward_boundary(hf, hi, rho, 2*e)
    with pytest.raises(ValueError, match='high-space matrices'):
        retained_ward_boundary(hf[:-1], hi, rho, e)


def test_existing_baseline_parameters_Hamiltonians_and_norms_are_unchanged():
    root = Path(__file__).resolve().parents[1]/'docs/next_level'
    info = json.loads((root/'m2_gaussian_cluster_four_current_baseline.json').read_text())
    model, values = build_model('baseline')
    assert values == info['parameters']
    with np.load(root/info['full_matrix_file']) as arrays:
        for label, e in (('out', 1.39), ('in', 1.47)):
            s = model.snapshot(e)
            np.testing.assert_allclose(s.effective_hamiltonian, arrays['H_'+label], atol=3e-14, rtol=3e-12)
            np.testing.assert_allclose(s.metric, arrays['N_'+label], atol=3e-15, rtol=3e-12)


def test_saved_retained_shell_one_four_current_has_independent_H2_Ward_coefficients_and_both_norms(endpoints):
    from test_m2_gaussian_reference_shape_four_current import saved, verify_saved_current
    info, arrays = saved('retained_shell1')
    low, _ = saved('Q5_Q6_shell1')
    assert info['parameters'] == {**low['parameters'], 'retained':1}
    for key in ('total_k2', 'jz_twice', 'b_GeV', 'quark_mass_GeV', 'gluon_mass_GeV',
                'coupling', 'total_pplus_GeV', 'energy_out_GeV2', 'energy_in_GeV2',
                'transfer_GeV', 'component_order', 'current_normalization',
                'physical_leg_prescription', 'smooth_prescription',
                'field_envelope_prescription', 'isospin_prescription', 'raw_Q_dimensions'):
        assert info[key] == low[key]
    assert arrays['raw_current'].shape == (4, 204, 204)
    assert not info['physical'] and not info['continuum_convergence_established']
    (_, high_out, _), (_, high_in, _) = endpoints
    # This reconstructs each H coefficient from the independent H2 wave
    # recurrence, rather than merely rechecking the saved total Ward norm.
    verify_saved_current((high_out, high_in, arrays))


def retained_comparison_data():
    root = Path(__file__).resolve().parents[1]/'docs/next_level'
    info = json.loads((root/'m2_gaussian_retained_resolution_P0_P1.json').read_text())
    with np.load(root/info['full_matrix_file'], allow_pickle=False) as archive:
        arrays = dict(archive)
    assert len(arrays) == 27
    assert all(np.all(np.isfinite(a)) for a in arrays.values())
    return info, arrays


def test_saved_retained_comparison_uses_physical_embedding_and_independent_two_norm_maps():
    from test_m2_gaussian_reference_shape_four_current import saved
    from deuteron_wigner.quantum.m2_matched_h2_reduction import nested_pilot_isometry
    _, low = saved('Q5_Q6_shell1')
    _, high = saved('retained_shell1')
    info, a = retained_comparison_data()
    lp, hp = [build_model(c)[0].pilot for c in ('Q5_Q6_shell1', 'retained_shell1')]
    e = nested_pilot_isometry(lp, hp)
    np.testing.assert_array_equal(a['E'], e)

    def metric_power(value, power):
        w, u = np.linalg.eigh(value)
        assert min(w) > 0
        return (u*w**power)@u.conj().T

    maps = {}
    for end in ('out', 'in'):
        maps[end] = metric_power(high['N_'+end], .5)@e@metric_power(low['N_'+end], -.5)
        np.testing.assert_allclose(a['T_'+end], maps[end], atol=6e-14, rtol=3e-12)
        np.testing.assert_allclose(maps[end].conj().T@maps[end], np.eye(37), atol=5e-13, rtol=0.)
        for stem in ('H_', 'N_'):
            key = stem+end
            np.testing.assert_array_equal(a['lower_'+key], low[key])
            np.testing.assert_allclose(a['compressed_'+key], e.conj().T@high[key]@e, atol=4e-14, rtol=3e-12)
            np.testing.assert_allclose(a['compressed_'+key], low[key], atol=4e-14, rtol=3e-12)
    for key in ('raw_current', 'normalized_current', *('current_power_'+str(n) for n in range(5))):
        f, i = (maps['out'], maps['in']) if key == 'normalized_current' else (e, e)
        np.testing.assert_array_equal(a['lower_'+key], low[key])
        expected = np.stack([f.conj().T@v@i for v in high[key]])
        np.testing.assert_allclose(a['compressed_'+key], expected, atol=8e-14, rtol=3e-12)
    summed = sum(info['coupling']**n*a['compressed_current_power_'+str(n)] for n in range(5))
    np.testing.assert_allclose(summed, a['compressed_raw_current'], atol=5e-14, rtol=3e-12)
    normalized = np.stack([metric_power(low['N_out'], -.5)@v@metric_power(low['N_in'], -.5) for v in summed])
    np.testing.assert_allclose(normalized, a['compressed_normalized_current'], atol=8e-14, rtol=3e-12)
    naive = e.conj().T@high['normalized_current']@e
    error = np.linalg.norm(naive-a['compressed_normalized_current'])
    np.testing.assert_allclose(error, info['naive_normalized_raw_embedding_error_Frobenius'], rtol=3e-9, atol=2e-14)
    assert error > 1e-5 and np.linalg.norm(maps['out']-maps['in']) > 1e-5
    rho = good_current_density_between(hp, hp, info['transfer_GeV'], charge='electromagnetic')
    lowrho = good_current_density_between(lp, lp, info['transfer_GeV'], charge='electromagnetic')
    # Independent difference of products, not the boundary helper used
    # to generate the report. The unequal-energy plus term is retained.
    boundary = low['H_out']@lowrho-lowrho@low['H_in']-e.conj().T@(high['H_out']@rho-rho@high['H_in'])@e
    change = a['lower_raw_current']-a['compressed_raw_current']
    contraction = 2*hp.total_pplus_GeV*np.einsum('m,mab->ab', info['transfer_GeV'], change[2:])
    residual = contraction-boundary-(info['energy_out_GeV2']-info['energy_in_GeV2'])*change[0]
    np.testing.assert_allclose(a['Ward_boundary'], boundary, atol=5e-14, rtol=3e-12)
    np.testing.assert_allclose(residual, 0., atol=8e-14, rtol=0.)
    np.testing.assert_allclose(a['Ward_difference_residual'], residual, atol=8e-14, rtol=0.)


def test_retained_minus_change_and_every_saved_component_statistic():
    info, a = retained_comparison_data()
    assert info['retained_dimensions'] == [37, 204] and info['fixed_Q_shells'] == [1, 1]
    assert not info['physical'] and not info['continuum_convergence_established']
    assert 'near-zero reference' in info['diagnostic_projection_note']
    for key, record in info['comparisons'].items():
        low, high = a['lower_'+key], a['compressed_'+key]
        np.testing.assert_allclose(low[[0, 2, 3]], high[[0, 2, 3]], atol=4e-14, rtol=3e-12)
        pairs = [(record['stacked'], low, high)]
        pairs += [(record['components'][label], x, y) for label, x, y in zip(info['component_order'], low, high)]
        for name, direction in (('transverse_along_transfer', np.array([.6, .8])),
                                ('transverse_orthogonal_to_transfer', np.array([-.8, .6]))):
            pairs.append((record[name], np.einsum('m,mab->ab', direction, low[2:]),
                           np.einsum('m,mab->ab', direction, high[2:])))
        for entry, x, y in pairs:
            norm, error = np.linalg.norm(x), np.linalg.norm(y-x)
            np.testing.assert_allclose(entry['reference_Frobenius'], norm, atol=1e-14, rtol=3e-12)
            np.testing.assert_allclose(entry['candidate_Frobenius'], np.linalg.norm(y), atol=1e-14, rtol=3e-12)
            np.testing.assert_allclose(entry['difference_Frobenius'], error, atol=1e-14, rtol=3e-12)
            if norm > 1e-12:
                np.testing.assert_allclose(entry['relative_difference'], error/norm, atol=5e-11, rtol=3e-12)
            elif norm == 0:
                assert entry['relative_difference'] is None
            # At near-zero reference projections, retain the absolute
            # numbers without claiming that roundoff ratios are errors.
    differences = [np.linalg.norm(a['compressed_current_power_'+str(n)][1]-a['lower_current_power_'+str(n)][1]) for n in range(5)]
    np.testing.assert_allclose(differences[:3], [.1590693481938184, .0026483513674489876, .0013439000832377774], rtol=3e-12)
    assert max(differences[3:]) < 1e-15
    np.testing.assert_allclose(info['comparisons']['normalized_current']['components']['minus']['relative_difference'], .020677646198150483, rtol=3e-12)
    np.testing.assert_allclose(info['comparisons']['normalized_current']['stacked']['relative_difference'], .01749187518954279, rtol=3e-12)


def test_retained_comparison_diagnostic_does_not_infer_minus_or_orthogonal_current_from_contraction():
    from scripts.benchmark_m2_gaussian_retained_resolution import current_differences, pullback
    low = np.ones((4, 3, 3), complex)
    high = low.copy()
    high[1] += np.eye(3)
    high[2] += .8*np.eye(3)
    high[3] -= .6*np.eye(3)
    result = current_differences(low, high, ['plus', 'minus', 'x', 'y'], [.12, .16])
    assert result['components']['plus']['difference_Frobenius'] == 0
    assert result['components']['minus']['difference_Frobenius'] > 1
    assert result['transverse_along_transfer']['difference_Frobenius'] < 1e-14
    assert result['transverse_orthogonal_to_transfer']['difference_Frobenius'] > 1
    with pytest.raises(ValueError, match='transfer'):
        current_differences(low, high, ['plus', 'minus', 'x', 'y'], [0, 0])
    with pytest.raises(ValueError, match='dimensions'):
        pullback(low, np.eye(2), np.eye(3))


def test_retained_minus_change_is_the_independently_evaluated_P_projector_boundary():
    from deuteron_wigner.quantum.m2_gaussian_cluster_retained_projector import exact_retained_smooth_minus_projector_coefficients
    from deuteron_wigner.quantum.m2_smooth_longitudinal_projector import SmoothMinusProjectorContext, SMOOTH_PRESCRIPTION
    from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION
    from deuteron_wigner.quantum.m2_matched_h2_reduction import nested_pilot_isometry
    info, a = retained_comparison_data()
    lp, hp = [build_model(c)[0].pilot for c in ('Q5_Q6_shell1', 'retained_shell1')]
    e = nested_pilot_isometry(lp, hp)
    context = SmoothMinusProjectorContext(mean_total_transverse_GeV=(0., 0.), prescription=SMOOTH_PRESCRIPTION)
    args = dict(charge='electromagnetic', prescription=PRESCRIPTION, quadrature_order=8, minus_context=context)
    lower = exact_retained_smooth_minus_projector_coefficients(lp, lp, info['transfer_GeV'], **args)
    higher = exact_retained_smooth_minus_projector_coefficients(hp, hp, info['transfer_GeV'], **args)
    assert set(lower) == set(higher) == {0, 1, 2}
    for n in (0, 1, 2):
        observed = a['compressed_current_power_'+str(n)][1]-a['lower_current_power_'+str(n)][1]
        boundary = e.conj().T@higher[n][0]@e-lower[n][0]
        np.testing.assert_allclose(observed, boundary, atol=3e-14, rtol=3e-12)
        assert np.linalg.norm(boundary) > 1e-3
    # No g3/g4 term is manufactured to fit the difference. Those full
    # coefficients agree separately, and Delta+=0 Ward closure never
    # determines this independently derived minus boundary.


@pytest.mark.parametrize('sector', ['qqqgg', 'qqqq-qbar', 'qqqggg', 'qqqq-qbar-g'])
def test_K11_extension_has_an_independently_enumerated_off_pole_mass_floor(sector):
    module = five if sector in five.SPECIES else six
    species = module.SPECIES[sector]
    floors = [11*sum((Fraction(49, 400*p) for s, p in zip(species, modes) if s != 'g'), Fraction())
              for modes in module.longitudinal_states(sector, 11)]
    assert min(floors) > Fraction(147, 100)
    # This is a K11 statement, not extrapolation to arbitrary longitudinal
    # resolution. The compressed transverse kinetic energy is nonnegative.
    assert min(floors)-Fraction(147, 100) >= Fraction(931, 1200)

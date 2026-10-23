"""Independent finite H2 checks of nested residual propagation and bounds."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest

from scripts.benchmark_m2_gaussian_cluster_four_current import build_model
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import GaussianClusterH2, cluster_plus_density_between
from deuteron_wigner.quantum.m2_gaussian_resolution_defect import (
    nested_rows, wave_defect, plus_and_metric_bounds, _charge_bound)
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_shell_current import shell_density_between

D = np.array([.12, .16])


@pytest.fixture(scope='module', params=[(0, 0, 1, 1), (1, 0, 2, 0)])
def pair(request):
    base, _ = build_model('baseline')
    a, b, c, d = request.param
    low = base if (a, b) == (0, 0) else GaussianClusterH2(base.pilot, excluded_cutoff=a, six_cutoff=b)
    ref = GaussianClusterH2(base.pilot, excluded_cutoff=c, six_cutoff=d)
    result = []
    for energy in (1.39, 1.47):
        lower, reference = low.snapshot(energy), ref.snapshot(energy)
        result.append((wave_defect(lower, ref), reference))
    return result


def test_triangular_defects_reconstruct_every_actual_reference_wave_and_bound_error(pair):
    for audit, reference in pair:
        target = dict(zip(('A1', 'A2', 'C3'), reference.five_wave_coefficients))
        target['A6'] = reference.six_wave_coefficient
        for key, waves in target.items():
            for sector, value in waves.items():
                error = value-audit.embedded[key][sector]
                np.testing.assert_allclose(audit.errors[key][sector], error, atol=4e-15, rtol=3e-12)
                assert np.linalg.norm(error) <= audit.error_bounds[key][sector]+3e-14
                propagation = (reference.owner.six.propagation6 if key == 'A6'
                               else reference.owner.qq.propagation)[sector]
                np.testing.assert_allclose(audit.effective_defects[key][sector],
                    reference.energy_GeV2*error-propagation.apply(error), atol=3e-14, rtol=3e-12)
        # Propagating preceding errors matters. The bare higher-wave residual
        # is not in general the complete error source in the triangular system.
        assert sum(np.linalg.norm(audit.effective_defects['C3'][n]-audit.raw_defects['C3'][n])
                   for n in target['C3']) > 1e-5
        for key, sectors in audit.leading_defect_parts.items():
            for n, parts in sectors.items():
                np.testing.assert_allclose(sum(parts.values()), audit.raw_defects[key][n],
                                           atol=3e-14, rtol=3e-12)
                rows = nested_rows(audit.lower.owner.spaces5[n], reference.owner.spaces5[n])
                np.testing.assert_allclose(audit.raw_defects[key][n][rows], 0., atol=3e-14)
                np.testing.assert_allclose(parts['lower_equation'], 0., atol=3e-14)


def _plus_coefficients(f, i):
    m = i.owner
    result = {n: np.zeros_like(m.pilot.matrix) for n in range(5)}
    result[0] = good_current_density_between(m.pilot, m.pilot, D, charge='electromagnetic')
    for sector, space in m.spaces5.items():
        for a, left in enumerate(f.five_wave_coefficients):
            for b, right in enumerate(i.five_wave_coefficients):
                if a+b+2 > 4:
                    continue
                acted = shell_density_between(space, space, right[sector], D,
                    b_GeV=m.pilot.width_GeV, charge='electromagnetic')
                result[a+b+2] += left[sector].conj().T@acted
    for sector, space in m.spaces6.items():
        acted = shell_density_between(space, space, i.six_wave_coefficient[sector], D,
            b_GeV=m.pilot.width_GeV, charge='electromagnetic')
        result[4] += f.six_wave_coefficient[sector].conj().T@acted
    return result


def test_each_plus_order_full_metric_and_both_norm_maps_obey_residual_bounds(pair):
    (f, rf), (i, ri) = pair
    result = plus_and_metric_bounds(f, i, D, charge='electromagnetic')
    a, b = _plus_coefficients(f.lower, i.lower), _plus_coefficients(rf, ri)
    for n in range(5):
        np.testing.assert_allclose(result['reference_raw_coefficients'][n], b[n], atol=3e-15, rtol=3e-12)
        np.testing.assert_allclose(result['lower_raw_coefficients'][n], a[n], atol=3e-15, rtol=3e-12)
        assert np.linalg.norm(b[n]-a[n]) <= result['raw_coefficient_bounds'][n]+3e-14
    for k, (lower, reference) in enumerate(((f.lower, rf), (i.lower, ri))):
        np.testing.assert_allclose(result['reference_metrics'][k], reference.metric, atol=3e-15, rtol=3e-12)
        assert np.linalg.norm(reference.metric-lower.metric) <= result['endpoint_metric_bounds'][k]+3e-14
    for normalized, key in ((False, 'raw_total_bound'), (True, 'normalized_total_bound')):
        lo = cluster_plus_density_between(f.lower, i.lower, D, charge='electromagnetic', normalized=normalized)
        hi = cluster_plus_density_between(rf, ri, D, charge='electromagnetic', normalized=normalized)
        assert 0 < np.linalg.norm(hi-lo) <= result[key]+3e-14


def test_actual_coordinate_injection_and_charge_contraction(pair):
    audit, ref = pair[0]
    low = audit.lower.owner
    for n, space in {**low.spaces5, **low.spaces6}.items():
        large = {**ref.owner.spaces5, **ref.owner.spaces6}[n]
        rows = nested_rows(space, large)
        assert len(np.unique(rows)) == space.dimension
        rng = np.random.default_rng(501)
        v = rng.normal(size=(space.dimension, 2))+1j*rng.normal(size=(space.dimension, 2))
        embedded = np.zeros((large.dimension, 2), complex)
        embedded[rows] = v
        acted = shell_density_between(large, large, embedded, D, b_GeV=.4, charge='electromagnetic')
        small = shell_density_between(space, space, v, D, b_GeV=.4, charge='electromagnetic')
        np.testing.assert_allclose(acted[rows], small, atol=4e-15, rtol=3e-12)
        assert np.linalg.norm(acted) <= _charge_bound(large, 'electromagnetic')*np.linalg.norm(v)
        with pytest.raises(ValueError, match='disagree'):
            nested_rows(replace(space, target='NEUTRON'), large)


def test_shared_retained_pilot_is_required(pair):
    audit, ref = pair[0]
    other, _ = build_model('baseline')
    with pytest.raises(ValueError, match='unchanged retained'):
        wave_defect(other.snapshot(1.39), ref.owner)


def test_poles_and_invalid_endpoint_or_transfer_are_not_silently_repaired(pair):
    (f, rf), (i, _) = pair
    propagation = next(iter(rf.owner.qq.propagation.values()))
    pole = propagation.blocks[0].mass+propagation.blocks[0].eigenvalues[0]
    with pytest.raises(ValueError, match='off-pole'):
        wave_defect(replace(f.lower, energy_GeV2=pole), rf.owner)
    with pytest.raises(ValueError, match='off-pole'):
        wave_defect(replace(f.lower, energy_GeV2=float('nan')), rf.owner)
    with pytest.raises(ValueError, match='two-component'):
        plus_and_metric_bounds(f, i, [0.], charge='electromagnetic')
    with pytest.raises(ValueError, match='model pair'):
        plus_and_metric_bounds(f, replace(i, reference_model=i.lower.owner), D, charge='electromagnetic')


def test_saved_shell_three_reference_against_independent_direct_H2_construction():
    root = Path(__file__).resolve().parents[1]/'docs/next_level'
    info = json.loads((root/'m2_gaussian_resolution_defect_Q5_shell2_to3.json').read_text())
    with np.load(root/info['full_matrix_file']) as saved:
        arrays = {k: saved[k] for k in saved}
    base, _ = build_model('baseline')
    ref = GaussianClusterH2(base.pilot, excluded_cutoff=3, six_cutoff=0)
    f, i = (ref.snapshot(e) for e in info['endpoint_energies_GeV2'])
    direct = _plus_coefficients(f, i)
    for n, v in direct.items():
        np.testing.assert_allclose(arrays['reference_plus_g'+str(n)], v, atol=4e-15, rtol=3e-12)
        change = np.linalg.norm(v-arrays['lower_plus_g'+str(n)])
        assert change <= info['plus_coefficients'][str(n)]['analytic_bound_Frobenius']+3e-14
    for label, endpoint in (('out', f), ('in', i)):
        np.testing.assert_allclose(arrays['reference_N_'+label], endpoint.metric, atol=4e-15, rtol=3e-12)
    for normalized, key in ((False, 'raw'), (True, 'normalized')):
        v = cluster_plus_density_between(f, i, D, charge='electromagnetic', normalized=normalized)
        np.testing.assert_allclose(arrays['reference_'+key+'_plus'], v, atol=4e-15, rtol=3e-12)
    with np.load(root/'m2_gaussian_plus_resolution_Q5_shell2.npz') as earlier:
        np.testing.assert_allclose(arrays['lower_plus_g4'], earlier['raw_plus_g4'], atol=3e-17, rtol=3e-12)
    assert not info['continuum_convergence_established'] and not info['physical']

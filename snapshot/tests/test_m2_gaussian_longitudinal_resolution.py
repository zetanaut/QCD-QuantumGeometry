"""Common physical-label probes, not column matching across different K grids."""
from dataclasses import fields, replace
import json
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import block_diag

from scripts.benchmark_m2_gaussian_cluster_four_current import build_model
from deuteron_wigner.quantum.m2_gaussian_longitudinal_resolution import longitudinal_probe_frame, probe_current
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import cluster_plus_density_between
from deuteron_wigner.quantum.m2_source_vectors import source_tensor
from deuteron_wigner.quantum.m2_c117_pair_current import color_permutation_matrix
from scripts.benchmark_m2_gaussian_longitudinal_resolution import load_current, raw_change_attribution


@pytest.fixture(scope='module', params=['baseline', 'longitudinal_K11'])
def model(request):
    m, _ = build_model(request.param)
    return m, m.snapshot(1.39), m.snapshot(1.47)


@pytest.mark.parametrize('power', [1., 2.])
def test_physical_seed_labels_helicities_and_normalization_are_preserved(model, power):
    m, _, _ = model
    frame = longitudinal_probe_frame(m.pilot, longitudinal_power=power)
    assert frame.labels == ('qqq', 'qqqg_minus', 'qqqg_plus')
    assert frame.vectors.shape == (len(m.pilot.matrix), 3)
    np.testing.assert_allclose(frame.vectors.conj().T@frame.vectors, np.eye(3), atol=3e-14, rtol=0.)
    qqq = source_tensor(m.pilot, 'qqq', frame.vectors)
    qqqg = source_tensor(m.pilot, 'qqqg', frame.vectors)
    np.testing.assert_array_equal(qqq[..., 1:], 0.)
    np.testing.assert_array_equal(qqqg[..., 0], 0.)
    np.testing.assert_array_equal(qqqg[..., 1, 1], 0.)  # minus-gluon seed has no plus-gluon support
    np.testing.assert_array_equal(qqqg[..., 0, 2], 0.)
    spin = np.array(list(np.ndindex(4, 4, 4)))
    sz = np.sum(2*(spin % 2)-1, axis=1)
    np.testing.assert_allclose(qqq[:, :, :, sz != 1, 0], 0., atol=3e-15)
    np.testing.assert_allclose(qqqg[:, :, :, sz != 3, 0, 1], 0., atol=3e-15)
    np.testing.assert_allclose(qqqg[:, :, :, sz != -1, 1, 2], 0., atol=3e-15)


def test_same_probe_is_invariant_under_compact_symmetry_basis_rotations(model):
    m, _, _ = model
    p = m.pilot
    frame = longitudinal_probe_frame(p, longitudinal_power=2.)
    rng = np.random.default_rng(781)
    rotations = [np.linalg.qr(rng.normal(size=(b.shape[1], b.shape[1]))
                 +1j*rng.normal(size=(b.shape[1], b.shape[1])))[0] for b in p.bases]
    total = block_diag(*rotations)
    changes = {f.name: total.conj().T@getattr(p, f.name)@total for f in fields(p)
               if isinstance(getattr(p, f.name), np.ndarray) and getattr(p, f.name).shape == p.matrix.shape}
    changes['bases'] = tuple(sparse.csr_matrix(b@r) for b, r in zip(p.bases, rotations))
    rotated = replace(p, **changes)
    transformed = longitudinal_probe_frame(rotated, longitudinal_power=2.)
    np.testing.assert_allclose(transformed.vectors, total.conj().T@frame.vectors, atol=4e-14, rtol=3e-12)
    for sector in ('qqq', 'qqqg'):
        np.testing.assert_allclose(source_tensor(rotated, sector, transformed.vectors),
                                   source_tensor(p, sector, frame.vectors), atol=4e-14, rtol=3e-12)


def test_raw_color_choice_has_the_required_identical_quark_exchange_sign():
    # Both gluon-sector seeds have identical first/second u spins and
    # symmetric prod(x)^p orbitals. Their color must be odd under (12).
    exchange = color_permutation_matrix('qqqg', (1, 0, 2))
    np.testing.assert_allclose(exchange, np.diag([1., -1.]), atol=2e-15, rtol=0.)


def test_zero_projected_probe_is_rejected_not_normalized(model):
    p = model[0].pilot
    empty = replace(p, bases=tuple(sparse.csr_matrix(b.shape, dtype=complex) for b in p.bases))
    with pytest.raises(ArithmeticError, match='numerically resolved projected support'):
        longitudinal_probe_frame(empty, longitudinal_power=2.)


def test_full_two_probe_metrics_agree_with_independent_full_space_pullback(model):
    m, f, i = model
    frame = longitudinal_probe_frame(m.pilot, longitudinal_power=2.)
    raw = cluster_plus_density_between(f, i, (.12, .16), charge='electromagnetic', normalized=False)
    compressed, normalized, norms = probe_current(raw, f, i, frame)
    maps = []
    for s, g in zip((f, i), norms):
        eigen, u = np.linalg.eigh(g)
        assert min(eigen) > 0
        invsqrt = (u/np.sqrt(eigen))@u.conj().T
        t = np.linalg.solve(s.whitener, frame.vectors@invsqrt)
        np.testing.assert_allclose(t.conj().T@t, np.eye(3), atol=4e-14, rtol=0.)
        maps.append(t)
    high = cluster_plus_density_between(f, i, (.12, .16), charge='electromagnetic', normalized=True)
    np.testing.assert_allclose(normalized, maps[0].conj().T@high@maps[1], atol=4e-14, rtol=3e-12)
    np.testing.assert_allclose(compressed, frame.vectors.conj().T@raw@frame.vectors, atol=4e-14, rtol=3e-12)
    zero = cluster_plus_density_between(i, i, (0., 0.), charge='electromagnetic', normalized=False)
    np.testing.assert_allclose(zero, i.metric, atol=4e-14, rtol=3e-12)
    _, charge, _ = probe_current(zero, i, i, frame)
    np.testing.assert_allclose(charge, np.eye(3), atol=4e-14, rtol=0.)


def test_field_caps_are_inactive_on_physical_legs_and_paired_Wick_tails(model):
    m, _, _ = model
    reg = m.field_regulator
    assert 2*reg.boson_modes > m.pilot.total_k2
    assert 2*reg.fermion_modes-1 > m.pilot.total_k2
    wider = replace(m.wick_operators, regulator=replace(reg, boson_modes=16, fermion_modes=16))
    np.testing.assert_array_equal(wider.retained_mass_coefficient(m.pilot),
                                  m.wick_operators.retained_mass_coefficient(m.pilot))
    for space in {**m.spaces5, **m.spaces6}.values():
        np.testing.assert_array_equal(wider.shell_mass_diagonal(space), m.wick_operators.shell_mass_diagonal(space))
    # Fixed P+ implies L proportional to K2. Counts held fixed do NOT mean
    # fixed physical p+ caps; this finite subtracted control is insensitive
    # because those caps exceed every physical leg and the trace is paired.


@pytest.mark.parametrize('bad', [True, -1., float('nan')])
def test_no_implicit_invalid_probe_exponent(model, bad):
    with pytest.raises(ValueError, match='explicit finite'):
        longitudinal_probe_frame(model[0].pilot, longitudinal_power=bad)


def test_wrong_pilot_or_current_is_not_silently_reinterpreted(model):
    m, f, i = model
    frame = longitudinal_probe_frame(m.pilot, longitudinal_power=2.)
    with pytest.raises(ValueError, match='named probe pilot'):
        probe_current(np.eye(len(m.pilot.matrix)), f, i, replace(frame, pilot=replace(m.pilot)))
    with pytest.raises(ValueError, match='raw current'):
        probe_current(np.eye(2), f, i, frame)


def saved_probes():
    root = Path(__file__).resolve().parents[1]/'docs/next_level'
    info = json.loads((root/'m2_gaussian_longitudinal_resolution_K9_K11.json').read_text())
    with np.load(root/info['full_matrix_file']) as archive:
        arrays = {k:archive[k] for k in archive}
    return info, arrays


@pytest.mark.parametrize('power', [1, 2])
def test_saved_four_current_probes_have_independent_full_space_two_metric_pullbacks(model, power):
    m, f, i = model
    case = 'baseline' if m.pilot.total_k2 == 9 else 'longitudinal_K11'
    info, full = load_current(case)
    _, data = saved_probes()
    stem = 'K'+str(m.pilot.total_k2)+'_p'+str(power)+'_'
    frame = longitudinal_probe_frame(m.pilot, longitudinal_power=power)
    u = frame.vectors
    np.testing.assert_array_equal(data[stem+'U'], u)
    maps = []
    for label, snapshot in (('out', f), ('in', i)):
        g = u.conj().T@snapshot.metric@u
        np.testing.assert_allclose(data[stem+'N_'+label], g, atol=4e-14, rtol=3e-12)
        np.testing.assert_allclose(data[stem+'H_'+label],
            u.conj().T@snapshot.effective_hamiltonian@u, atol=4e-14, rtol=3e-12)
        w, v = np.linalg.eigh(g)
        assert min(w) > 0
        maps.append(np.linalg.solve(snapshot.whitener, u@((v/np.sqrt(w))@v.conj().T)))
    for name in ('raw_current', *('current_power_'+str(n) for n in range(5))):
        independent = np.array([u.conj().T@a@u for a in full[name]])
        np.testing.assert_allclose(data[stem+name], independent, atol=4e-14, rtol=3e-12)
    normalized = np.array([maps[0].conj().T@a@maps[1] for a in full['normalized_current']])
    np.testing.assert_allclose(data[stem+'normalized_current'], normalized, atol=4e-14, rtol=3e-12)
    assert info['total_k2'] == m.pilot.total_k2


def test_saved_longitudinal_report_preserves_all_channels_orders_and_large_finite_changes():
    info, data = saved_probes()
    assert len(data) == 48
    assert info['probe_labels'] == ['qqq', 'qqqg_minus', 'qqqg_plus']
    assert info['raw_color_coordinates'] == [0, 1, 1]
    assert info['full_retained_dimensions'] == [37, 69]
    assert not info['different_K_compact_column_alignment']
    assert not info['physical'] and not info['continuum_convergence_established']
    for power in (1, 2):
        left, right = ('K'+str(k)+'_p'+str(power)+'_' for k in (9, 11))
        for name, entry in info['comparisons'][str(power)].items():
            a, b = data[left+name], data[right+name]
            values = entry.get('stacked', entry)
            np.testing.assert_allclose(values['difference_Frobenius'], np.linalg.norm(b-a), atol=2e-15, rtol=3e-12)
            if 'components' in entry:
                for mu, label in enumerate(info['component_order']):
                    np.testing.assert_allclose(entry['components'][label]['difference_Frobenius'],
                        np.linalg.norm(b[mu]-a[mu]), atol=2e-15, rtol=3e-12)
        # Measured finite-grid sensitivity, not an asymptotic error bound.
        assert info['comparisons'][str(power)]['current_power_3']['stacked']['relative_difference'] > .6
        assert info['comparisons'][str(power)]['current_power_4']['stacked']['relative_difference'] > .4
    np.testing.assert_allclose([info['comparisons'][str(p)]['normalized_current']['stacked']['relative_difference']
                               for p in (1, 2)], [.009832650766062425, .07749789985091234], rtol=3e-12)


def test_raw_longitudinal_change_is_coherently_resolved_not_a_Wick_uncertainty_fraction():
    info, data = saved_probes()
    actual = raw_change_attribution(data, info['coupling'])
    assert not info['signed_projections_are_probabilities_or_physical_uncertainties']
    for power, parts in actual.items():
        signed = []
        for n, entry in parts.items():
            recorded = info['raw_change_attribution'][power][n]
            for key, value in entry.items():
                np.testing.assert_allclose(recorded[key], value, atol=2e-15, rtol=3e-12)
            signed.append(entry['signed_projection_on_total_change'])
        np.testing.assert_allclose(sum(signed), 1., atol=2e-13, rtol=0.)
        assert .98 < signed[0] < 1.
        assert any(value < 0 for value in signed)
    # The dominant raw total displacement is already present at g0;
    # it contains no Wick loop or chosen finite mass counterterm.

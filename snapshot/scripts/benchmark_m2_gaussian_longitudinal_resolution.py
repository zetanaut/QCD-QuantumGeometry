"""Compare saved K9/K11 currents on named physical-label probe families.

Both full current archives remain unchanged. This evaluates separate
three-channel diagnostics, not a physical state, fit or continuum claim.
"""
import argparse
import json
from pathlib import Path
from time import monotonic
import numpy as np

from scripts.benchmark_m2_gaussian_cluster_four_current import build_model
from deuteron_wigner.quantum.m2_gaussian_longitudinal_resolution import longitudinal_probe_frame, probe_current

ROOT = Path(__file__).resolve().parents[1]/'docs/next_level'
PREFIX = 'm2_gaussian_cluster_four_current_'
CASES = ('baseline', 'longitudinal_K11')


def load_current(case):
    info = json.loads((ROOT/(PREFIX+case+'.json')).read_text())
    if info['full_matrix_file'] != PREFIX+case+'.npz':
        raise ValueError('explicit named full-current archive required')
    with np.load(ROOT/info['full_matrix_file']) as data:
        arrays = {name:data[name] for name in data}
    expected = {'raw_current', 'normalized_current', 'H_out', 'H_in', 'N_out', 'N_in'}
    expected.update('current_power_'+str(n) for n in range(5))
    if set(arrays) != expected or any(not np.all(np.isfinite(v)) for v in arrays.values()):
        raise ValueError('all eleven finite full-current matrices required')
    return info, arrays


def difference(reference, candidate):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        raise ValueError('same named probe coordinates required')
    norm = float(np.linalg.norm(a))
    error = float(np.linalg.norm(b-a))
    return dict(reference_Frobenius=norm, candidate_Frobenius=float(np.linalg.norm(b)),
                difference_Frobenius=error, relative_difference=error/norm if norm else None)


def raw_change_attribution(arrays, coupling):
    """Coherent signed projection onto the raw total grid difference.

    These are matrix amplitudes, not probabilities or physical uncertainty
    fractions. Keeping the raw order avoids conflating metric transport
    with a coefficient. In particular g0 does not contain a Wick loop.
    """
    result = {}
    for power in (1, 2):
        left, right = ('K'+str(k)+'_p'+str(power)+'_' for k in (9, 11))
        total = arrays[right+'raw_current']-arrays[left+'raw_current']
        changes = {n:coupling**n*(arrays[right+'current_power_'+str(n)]
                                  -arrays[left+'current_power_'+str(n)]) for n in range(5)}
        np.testing.assert_allclose(sum(changes.values()), total, atol=4e-14, rtol=3e-12)
        norm2 = float(np.vdot(total, total).real)
        result[str(power)] = {str(n):dict(weighted_difference_Frobenius=float(np.linalg.norm(v)),
            signed_projection_on_total_change=float(np.vdot(total, v).real)/norm2 if norm2 else None)
            for n, v in changes.items()}
    return result


def evaluate():
    start = monotonic()
    records = [load_current(case) for case in CASES]
    base, candidate = (r[0] for r in records)
    assert candidate['parameters'] == {**base['parameters'], 'k2':11}
    for key in ('retained_cutoff', 'jz_twice', 'b_GeV', 'quark_mass_GeV', 'gluon_mass_GeV',
                'coupling', 'total_pplus_GeV', 'energy_out_GeV2', 'energy_in_GeV2', 'transfer_GeV',
                'component_order', 'physical_leg_prescription', 'smooth_prescription',
                'field_envelope_prescription', 'isospin_prescription'):
        if candidate[key] != base[key]:
            raise ValueError('only the declared longitudinal grid may change: '+key)
    arrays = {}
    for case, (info, current) in zip(CASES, records):
        model, parameters = build_model(case)
        assert parameters == info['parameters'] and model.pilot.total_k2 == info['total_k2']
        f, i = (model.snapshot(info['energy_'+a+'_GeV2']) for a in ('out', 'in'))
        for label, snapshot in (('out', f), ('in', i)):
            np.testing.assert_allclose(snapshot.effective_hamiltonian, current['H_'+label], atol=4e-14, rtol=3e-12)
            np.testing.assert_allclose(snapshot.metric, current['N_'+label], atol=4e-14, rtol=3e-12)
        for power in (1, 2):
            frame = longitudinal_probe_frame(model.pilot, longitudinal_power=power)
            stem = 'K'+str(info['total_k2'])+'_p'+str(power)+'_'
            compressed, normalized, norms = probe_current(current['raw_current'], f, i, frame)
            arrays[stem+'raw_current'] = compressed
            arrays[stem+'normalized_current'] = normalized
            arrays[stem+'U'] = frame.vectors
            for n in range(5):
                values, _, _ = probe_current(current['current_power_'+str(n)], f, i, frame)
                arrays[stem+'current_power_'+str(n)] = values
            for label, norm, snapshot in zip(('out', 'in'), norms, (f, i)):
                arrays[stem+'N_'+label] = norm
                arrays[stem+'H_'+label] = frame.vectors.conj().T@snapshot.effective_hamiltonian@frame.vectors
            raw_sum = sum(info['coupling']**n*arrays[stem+'current_power_'+str(n)] for n in range(5))
            np.testing.assert_allclose(raw_sum, compressed, atol=4e-14, rtol=3e-12)
    comparisons = {}
    for power in (1, 2):
        left, right = ('K'+str(k)+'_p'+str(power)+'_' for k in (9, 11))
        entries = {}
        for key in ('raw_current', 'normalized_current', *('current_power_'+str(n) for n in range(5))):
            a, b = arrays[left+key], arrays[right+key]
            entries[key] = dict(stacked=difference(a, b),
                components={label:difference(x, y) for label, x, y in zip(base['component_order'], a, b)})
        for key in ('H_out', 'H_in', 'N_out', 'N_in'):
            entries[key] = difference(arrays[left+key], arrays[right+key])
        comparisons[str(power)] = entries
    report = dict(longitudinal_K2=[9, 11], full_retained_dimensions=[37, 69],
        source_full_current_reports=[PREFIX+case+'.json' for case in CASES],
        component_order=base['component_order'], coupling=base['coupling'],
        energy_out_GeV2=base['energy_out_GeV2'], energy_in_GeV2=base['energy_in_GeV2'],
        transfer_GeV=base['transfer_GeV'], probe_powers=[1, 2],
        probe_labels=['qqq', 'qqqg_minus', 'qqqg_plus'], raw_color_coordinates=[0, 1, 1],
        quark_labels=[[0, 1, 3], [1, 1, 3], [0, 0, 3]],
        raw_spin_flavor_legend=['u_down', 'u_up', 'd_down', 'd_up'],
        both_full_three_probe_Gram_metrics_retained=True,
        normalization='Common raw order g4 selected before both three-probe Gram inverse square roots.',
        total_pplus_fixed_GeV=base['total_pplus_GeV'],
        finite_conditions_are_uncalibrated_test_values=True,
        different_K_compact_column_alignment=False, physical=False,
        continuum_convergence_established=False, comparisons=comparisons,
        raw_change_attribution=raw_change_attribution(arrays, base['coupling']),
        signed_projections_are_probabilities_or_physical_uncertainties=False,
        seconds=monotonic()-start)
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    matrices = args.output.with_suffix('.npz')
    if args.output.exists() or matrices.exists():
        parser.error('select new outputs to preserve previous evidence')
    report, arrays = evaluate()
    report['full_matrix_file'] = matrices.name
    np.savez_compressed(matrices, **arrays)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(seconds=report['seconds'], matrices=len(arrays),
        comparisons={p:{k:v['stacked'] for k, v in data.items() if 'stacked' in v}
                     for p, data in report['comparisons'].items()})), flush=True)


if __name__ == '__main__':
    main()

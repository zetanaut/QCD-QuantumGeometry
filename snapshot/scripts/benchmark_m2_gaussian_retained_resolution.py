"""Compare two completed retained-shell currents without recomputing them.

The outputs keep every coupling coefficient and both endpoint metrics.
No equality of minus or transfer-orthogonal currents is assumed from Ward
closure. Invoke as python -m scripts.benchmark_m2_gaussian_retained_resolution
with two explicit saved reports and a fresh output. Originals remain unchanged.
"""
import argparse
import json
from pathlib import Path
from time import monotonic
import numpy as np

from scripts.benchmark_m2_gaussian_cluster_four_current import build_model
from scripts.benchmark_m2_gaussian_longitudinal_resolution import difference
from deuteron_wigner.quantum.m2_gaussian_retained_resolution import retained_metric_injection, retained_ward_boundary
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between


def load_report(path):
    info = json.loads(path.read_text())
    with np.load(path.parent/info['full_matrix_file'], allow_pickle=False) as archive:
        arrays = {name:archive[name] for name in archive}
    expected = {'raw_current', 'normalized_current', 'H_out', 'H_in', 'N_out', 'N_in'}
    expected.update('current_power_'+str(n) for n in range(5))
    if set(arrays) != expected or any(not np.all(np.isfinite(a)) for a in arrays.values()):
        raise ValueError('all eleven finite full-current matrices required')
    return info, arrays


def pullback(current, left, right):
    a = np.asarray(current)
    if a.ndim not in (2, 3) or a.shape[-2:] != (len(left), len(right)):
        raise ValueError('current dimensions must match the two endpoint maps')
    return left.conj().T@a@right


def current_differences(low, high, component_order, transfer):
    result = dict(stacked=difference(low, high),
        components={label:difference(a, b) for label, a, b in zip(component_order, low, high)})
    delta = np.asarray(transfer, float)
    if delta.shape != (2,) or not np.all(np.isfinite(delta)) or np.linalg.norm(delta) == 0:
        raise ValueError('nonzero finite transverse transfer required for diagnostic projections')
    along = delta/np.linalg.norm(delta)
    orthogonal = np.array([-along[1], along[0]])
    for name, direction in (('transverse_along_transfer', along), ('transverse_orthogonal_to_transfer', orthogonal)):
        result[name] = difference(np.einsum('m,mab->ab', direction, low[2:]),
                                  np.einsum('m,mab->ab', direction, high[2:]))
    return result


def evaluate(lower_path, larger_path):
    start = monotonic()
    (li, la), (hi, ha) = load_report(lower_path), load_report(larger_path)
    if (li['case'], hi['case']) != ('Q5_Q6_shell1', 'retained_shell1'):
        raise ValueError('explicit named P0/P1 pair with fixed Q5/Q6 shell one required')
    if hi['parameters'] != {**li['parameters'], 'retained':1}:
        raise ValueError('only the retained cutoff may change')
    for key in ('total_k2', 'jz_twice', 'b_GeV', 'quark_mass_GeV', 'gluon_mass_GeV',
                'coupling', 'total_pplus_GeV', 'energy_out_GeV2', 'energy_in_GeV2',
                'transfer_GeV', 'component_order', 'current_normalization',
                'physical_leg_prescription', 'smooth_prescription',
                'field_envelope_prescription', 'isospin_prescription', 'raw_Q_dimensions'):
        if li[key] != hi[key]:
            raise ValueError('fixed physical definitions required: '+key)
    lm, lp = build_model(li['case'])
    hm, hp = build_model(hi['case'])
    if lp != li['parameters'] or hp != hi['parameters']:
        raise ValueError('saved parameters do not match the explicitly reconstructed models')
    endpoints = {}
    output = {}
    for label in ('out', 'in'):
        ls = lm.snapshot(li['energy_'+label+'_GeV2'])
        hs = hm.snapshot(hi['energy_'+label+'_GeV2'])
        mapping = retained_metric_injection(ls, hs)
        endpoints[label] = ls, hs, mapping
        output['T_'+label] = mapping.normalized
        for key, attribute in (('H_', 'effective_hamiltonian'), ('N_', 'metric')):
            np.testing.assert_allclose(la[key+label], getattr(ls, attribute), atol=4e-14, rtol=3e-12)
            np.testing.assert_allclose(ha[key+label], getattr(hs, attribute), atol=4e-14, rtol=3e-12)
            output['lower_'+key+label] = la[key+label]
            output['compressed_'+key+label] = pullback(ha[key+label], mapping.raw, mapping.raw)
    lf, hf, mf = endpoints['out']
    lin, hin, mi = endpoints['in']
    np.testing.assert_array_equal(mf.raw, mi.raw)
    e = mf.raw
    output['E'] = e
    comparisons = {}
    for key in ('raw_current', 'normalized_current', *('current_power_'+str(n) for n in range(5))):
        left, right = (mf.normalized, mi.normalized) if key == 'normalized_current' else (e, e)
        compressed = pullback(ha[key], left, right)
        output['lower_'+key] = la[key]
        output['compressed_'+key] = compressed
        comparisons[key] = current_differences(la[key], compressed, li['component_order'], li['transfer_GeV'])
    raw = sum(li['coupling']**n*output['compressed_current_power_'+str(n)] for n in range(5))
    np.testing.assert_allclose(raw, output['compressed_raw_current'], atol=5e-14, rtol=3e-12)
    np.testing.assert_allclose(pullback(raw, lf.whitener, lin.whitener),
        output['compressed_normalized_current'], atol=8e-14, rtol=3e-12)
    rho_high = good_current_density_between(hm.pilot, hm.pilot, li['transfer_GeV'], charge='electromagnetic')
    rho_low = good_current_density_between(lm.pilot, lm.pilot, li['transfer_GeV'], charge='electromagnetic')
    np.testing.assert_allclose(pullback(rho_high, e, e), rho_low, atol=4e-14, rtol=3e-12)
    boundary = retained_ward_boundary(hf.effective_hamiltonian, hin.effective_hamiltonian, rho_high, e)
    change = output['lower_raw_current']-output['compressed_raw_current']
    # The general difference includes the unequal-energy good-current
    # term; do not drop it just because exact restriction is anticipated.
    contraction = 2*li['total_pplus_GeV']*np.einsum('m,mab->ab', li['transfer_GeV'], change[2:])
    predicted = boundary+(lf.energy_GeV2-lin.energy_GeV2)*change[0]
    np.testing.assert_allclose(contraction, predicted, atol=8e-14, rtol=3e-12)
    output['Ward_boundary'] = boundary
    output['Ward_difference_residual'] = contraction-predicted
    report = dict(source_reports=[str(lower_path), str(larger_path)],
        retained_dimensions=[e.shape[1], e.shape[0]], fixed_Q_shells=[1, 1], total_k2=li['total_k2'],
        coupling=li['coupling'], energy_out_GeV2=lf.energy_GeV2, energy_in_GeV2=lin.energy_GeV2,
        transfer_GeV=li['transfer_GeV'], component_order=li['component_order'],
        endpoint_metric_map_difference_Frobenius=float(np.linalg.norm(mf.normalized-mi.normalized)),
        naive_normalized_raw_embedding_error_Frobenius=float(np.linalg.norm(
            pullback(ha['normalized_current'], e, e)-output['compressed_normalized_current'])),
        Ward_boundary_Frobenius=float(np.linalg.norm(boundary)),
        Ward_difference_residual_Frobenius=float(np.linalg.norm(contraction-predicted)),
        comparisons=comparisons, finite_conditions_are_uncalibrated_test_values=True,
        physical=False, continuum_convergence_established=False,
        diagnostic_projection_note='Relative ratios with near-zero reference projections are ill-conditioned; inspect the reported absolute norms. No matrix entries or small coefficients are discarded.',
        normalization='Raw occupation-label E for H/N and each g coefficient; separate T_out/T_in for normalized current.',
        seconds=monotonic()-start)
    return report, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lower', type=Path, required=True)
    parser.add_argument('--larger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    matrices = args.output.with_suffix('.npz')
    if args.output.exists() or matrices.exists():
        parser.error('select new outputs to preserve prior evidence')
    report, arrays = evaluate(args.lower, args.larger)
    report['full_matrix_file'] = matrices.name
    np.savez_compressed(matrices, **arrays)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

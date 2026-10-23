"""Finite Q-reference residual, wave and good-current error study.

The retained pilot, finite conditions and field kernel stay unchanged.
This does not compare longitudinal/P resolutions or certify the continuum.
"""
import argparse
import json
from pathlib import Path
from time import monotonic
import numpy as np
from scripts.benchmark_m2_gaussian_cluster_four_current import build_model
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import GaussianClusterH2
from deuteron_wigner.quantum.m2_gaussian_resolution_defect import wave_defect, plus_and_metric_bounds
from deuteron_wigner.quantum.m2_fourth_order_h2 import _norm_inverse_sqrt


def evaluate(q5_lower, q5_reference, q6_lower, q6_reference):
    if not (0 <= q5_lower <= q5_reference and 0 <= q6_lower <= q6_reference):
        raise ValueError('ordered nonnegative lower/reference Q cutoffs required')
    start = monotonic()
    baseline, parameters = build_model('baseline')
    lower = baseline if (q5_lower, q6_lower) == (0, 0) else GaussianClusterH2(
        baseline.pilot, excluded_cutoff=q5_lower, six_cutoff=q6_lower)
    reference = GaussianClusterH2(baseline.pilot, excluded_cutoff=q5_reference, six_cutoff=q6_reference)
    audits = []
    for energy in (1.39, 1.47):
        print(f'Q5 {q5_lower}->{q5_reference}, Q6 {q6_lower}->{q6_reference}, E={energy}: residual transport', flush=True)
        audits.append(wave_defect(lower.snapshot(energy), reference))
    bounds = plus_and_metric_bounds(*audits, (.12, .16), charge='electromagnetic')
    arrays = {}
    for label in ('lower', 'reference'):
        coefficients = bounds[label+'_raw_coefficients']
        arrays.update({label+'_plus_g'+str(n): v for n, v in coefficients.items()})
        total = sum(lower.pilot.coupling**n*v for n, v in coefficients.items())
        norms = [a.lower.metric for a in audits] if label == 'lower' else bounds['reference_metrics']
        arrays.update({label+'_N_'+key: v for key, v in zip(('out', 'in'), norms)})
        arrays[label+'_raw_plus'] = total
        arrays[label+'_normalized_plus'] = _norm_inverse_sqrt(norms[0])@total@_norm_inverse_sqrt(norms[1])
    changes = {}
    for n in range(5):
        a, b = (arrays[label+'_plus_g'+str(n)] for label in ('lower', 'reference'))
        diff, scale = float(np.linalg.norm(b-a)), float(np.linalg.norm(a))
        bound = bounds['raw_coefficient_bounds'][n]
        if diff > bound+3e-13:
            raise ArithmeticError('finite-reference good-current bound failed')
        changes[n] = dict(difference_Frobenius=diff, relative_to_lower=diff/scale if scale else None,
                          analytic_bound_Frobenius=bound)
    actual = {}
    for kind in ('raw', 'normalized'):
        a, b = (arrays[label+'_'+kind+'_plus'] for label in ('lower', 'reference'))
        diff = float(np.linalg.norm(b-a))
        bound = bounds[kind+'_total_bound']
        if diff > bound+3e-13:
            raise ArithmeticError('finite-reference total current bound failed')
        actual[kind] = dict(difference_Frobenius=diff, analytic_bound_Frobenius=bound)
    waves = []
    for audit in audits:
        value = {}
        for key in audit.errors:
            value[key] = {n: dict(error_Frobenius=float(np.linalg.norm(v)),
                raw_defect_Frobenius=float(np.linalg.norm(audit.raw_defects[key][n])),
                propagated_defect_Frobenius=float(np.linalg.norm(audit.effective_defects[key][n])),
                error_bound_Frobenius=audit.error_bounds[key][n],
                reference_resolvent_gap_GeV2=audit.gaps[n]) for n, v in audit.errors[key].items()}
            if key in audit.leading_defect_parts:
                for n, parts in audit.leading_defect_parts[key].items():
                    value[key][n]['coherent_leading_defect_parts_Frobenius'] = {
                        label: float(np.linalg.norm(v)) for label, v in parts.items()}
        waves.append(value)
    parameters.pop('q5'); parameters.pop('q6')
    report = dict(shared_benchmark_parameters=parameters,
        reference_is_finite_and_not_continuum=True,
        Q5_cutoffs=[q5_lower, q5_reference], Q6_cutoffs=[q6_lower, q6_reference],
        total_k2=9, retained_cutoff=0, b_GeV=.4, quark_mass_GeV=.35,
        gluon_mass_GeV=0., coupling=.3, total_pplus_GeV=1.3,
        endpoint_energies_GeV2=[1.39, 1.47], transfer_GeV=[.12, .16],
        charge='electromagnetic', current_normalization='Jplus/(2Pplus)',
        current_powers_selected_before_both_norms=True,
        wave_defects=waves, plus_coefficients=changes, plus_totals=actual,
        endpoint_metric_bound_Frobenius=bounds['endpoint_metric_bounds'],
        minimum_endpoint_metric_eigenvalue=bounds['metric_gap'],
        physical=False, physical_conditions_selected=False,
        full_four_current_bound=False, continuum_convergence_established=False,
        seconds=monotonic()-start)
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('q5-lower', 'q5-reference', 'q6-lower', 'q6-reference'):
        parser.add_argument('--'+name, type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    matrices = args.output.with_suffix('.npz')
    if args.output.exists() or matrices.exists():
        parser.error('select new outputs to preserve prior evidence')
    info, arrays = evaluate(args.q5_lower, args.q5_reference, args.q6_lower, args.q6_reference)
    info['full_matrix_file'] = matrices.name
    np.savez_compressed(matrices, **arrays)
    args.output.write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()

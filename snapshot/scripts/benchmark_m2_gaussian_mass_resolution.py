"""Parameter-resolved finite-mass contribution to the observed Q5 sensitivity.

The common-order raw H2 Hamiltonian/norm/current is affine in the two
explicit finite kinetic-mass coefficients. Evaluate its intercept and
directions without changing the saved benchmark or selecting physical values.
Both intercept norm maps are independently diagonalized, not linearized.
"""
import argparse
import json
from pathlib import Path
from time import monotonic
import numpy as np
from scripts.benchmark_m2_gaussian_cluster_four_current import (
    build_model,PRESCRIPTION,SmoothMinusProjectorContext,SMOOTH_PRESCRIPTION,
    SPECTRAL_PRESCRIPTION,H2_ISOSPIN_PRESCRIPTION,good_current_density_between)
from deuteron_wigner.quantum.m2_gaussian_cluster_finite_response import (
    finite_mass_snapshot_response,finite_mass_current_coefficients)
from deuteron_wigner.quantum.m2_fourth_order_h2 import _norm_inverse_sqrt


def evaluate(case):
    start=monotonic();root=Path(__file__).resolve().parents[1]/'docs/next_level'
    info=json.loads((root/('m2_gaussian_cluster_four_current_'+case+'.json')).read_text())
    with np.load(root/info['full_matrix_file']) as archive:base={k:archive[k] for k in archive}
    m,parameters=build_model(case)
    if parameters!=info['parameters']:raise ValueError('saved benchmark parameters disagree')
    f,i=m.snapshot(info['energy_out_GeV2']),m.snapshot(info['energy_in_GeV2'])
    for label,s in (('out',f),('in',i)):
        np.testing.assert_allclose(base['H_'+label],s.effective_hamiltonian,atol=2e-14,rtol=2e-12)
        np.testing.assert_allclose(base['N_'+label],s.metric,atol=2e-14,rtol=2e-12)
    zeroH={'out':base['H_out'].copy(),'in':base['H_in'].copy()}
    zeroN={'out':base['N_out'].copy(),'in':base['N_in'].copy()}
    finite={n:np.zeros_like(base['current_power_0']) for n in range(5)}
    args=dict(charge='electromagnetic',loop_flavor_charges=(2/3,-1/3),
        prescription=PRESCRIPTION,quadrature_order=8,
        minus_context=SmoothMinusProjectorContext(mean_total_transverse_GeV=(0.,0.),prescription=SMOOTH_PRESCRIPTION),
        field_envelope_prescription=SPECTRAL_PRESCRIPTION,isospin_prescription=H2_ISOSPIN_PRESCRIPTION)
    arrays={}
    for parameter,key in (('quark','muq'),('gluon','mug')):
        print(case+': independent '+parameter+' finite-mass current direction',flush=True)
        current=finite_mass_current_coefficients(f,i,info['transfer_GeV'],parameter=parameter,**args)
        for n,v in current.items():
            if n not in (2,4):np.testing.assert_array_equal(v,0.)
            else:arrays[parameter+'_current_g'+str(n)]=v
            finite[n]+=parameters[key]*v
        for label,s in (('out',f),('in',i)):
            response=finite_mass_snapshot_response(s,parameter=parameter)
            arrays[parameter+'_H_'+label]=response.effective_hamiltonian
            arrays[parameter+'_N_'+label]=response.metric
            zeroH[label]-=parameters[key]*response.effective_hamiltonian
            zeroN[label]-=parameters[key]*response.metric
    zero={n:base['current_power_'+str(n)]-finite[n] for n in range(5)}
    raw=sum(info['coupling']**n*v for n,v in zero.items())
    wf,wi=(_norm_inverse_sqrt(zeroN[label]) for label in ('out','in'))
    normalized=np.einsum('ab,mbc,cd->mad',wf,raw,wi,optimize=True)
    rho=good_current_density_between(m.pilot,m.pilot,info['transfer_GeV'],charge='electromagnetic')
    expected=zeroH['out']@rho-rho@zeroH['in']+(f.energy_GeV2-i.energy_GeV2)*(raw[0]-rho)
    ward=2*info['total_pplus_GeV']*np.einsum('j,jab->ab',info['transfer_GeV'],raw[2:])-expected
    if np.linalg.norm(ward)>2e-11:raise ArithmeticError('finite-family intercept Ward identity failed')
    arrays.update({**{'finite_current_g'+str(n):finite[n] for n in (2,4)},
        **{'intercept_current_g'+str(n):v for n,v in zero.items()},
        **{'intercept_H_'+k:v for k,v in zeroH.items()},
        **{'intercept_N_'+k:v for k,v in zeroN.items()},
        'intercept_raw_current':raw,'intercept_normalized_current':normalized})
    report=dict(case=case,parameters=parameters,energy_out_GeV2=f.energy_GeV2,
        energy_in_GeV2=i.energy_GeV2,transfer_GeV=info['transfer_GeV'],
        intercept_means='Affine finite-parameter decomposition at muq=mug=0; no selected physical condition or replacement of the benchmark.',
        original_benchmark_unchanged=True,intercept_metrics_rediagonalized_without_clipping=True,
        intercept_Ward_Frobenius=float(np.linalg.norm(ward)),
        intercept_normalized_Ward_Frobenius=float(np.linalg.norm(wf@ward@wi)),
        intercept_metric_minimum_eigenvalues=[float(np.linalg.eigvalsh(zeroN[k])[0]) for k in ('out','in')],
        physical=False,continuum_convergence_established=False,seconds=monotonic()-start)
    return report,arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('baseline','Q5_shell1','Q6_shell1','Q5_Q6_shell1'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();matrices=args.output.with_suffix('.npz')
    if args.output.exists() or matrices.exists():parser.error('select new outputs to preserve prior evidence')
    report,arrays=evaluate(args.case);report['full_matrix_file']=matrices.name
    np.savez_compressed(matrices,**arrays);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

"""Resolve the finite H2 g^4 plus current into coherent loop words.

Uses the same finite benchmark and actual raw source waves as the full
four-current. These are matrix amplitudes, not sector probabilities or fits.
No four-current contribution or normalization map is replaced by this audit.
"""
import argparse
import json
from pathlib import Path
from time import monotonic
import numpy as np
from scripts.benchmark_m2_gaussian_cluster_four_current import build_model
from deuteron_wigner.quantum.m2_shell_current import shell_density_between


def fourth_order_plus_words(outgoing,incoming,transfer,*,charge):
    """Resolve A2^dag rho5 A2, A1/C3 backflow, and A6^dag rho6 A6.

    For this same-basis diagnostic the endpoints share one model, but retain
    different energies. C3 is split into R5 W55 A1, R5 U56 R6 W6P,
    and R5 U56 R6 U65 A1, with no change to the model's auxiliary projection.
    """
    m=incoming.owner
    if outgoing.owner is not m:
        raise ValueError('this same-basis loop diagnostic requires one shared model')
    delta=np.asarray(transfer,float)
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)):
        raise ValueError('finite two-component transverse transfer required')
    def resolve(propagators,e,values):
        return {n:propagators[n].resolve(e,v) for n,v in values.items()}
    def endpoint(snapshot):
        e=snapshot.energy_GeV2;a1,a2,c3=snapshot.five_wave_coefficients
        direct=resolve(m.six.propagation6,e,m.six.W)
        iterated=resolve(m.six.propagation6,e,m.six.U.apply(a1))
        for n in direct:
            np.testing.assert_allclose(direct[n]+iterated[n],snapshot.six_wave_coefficient[n],
                                       atol=2e-14,rtol=2e-12)
        tails={
            'QQ_backflow':resolve(m.qq.propagation,e,m.qq.action.apply(a1)),
            'direct6_backflow':resolve(m.six.propagation5,e,m.six.U.adjoint(direct)),
            'iterated6_backflow':resolve(m.six.propagation5,e,m.six.U.adjoint(iterated)),
        }
        for n in a1:
            np.testing.assert_allclose(sum(t[n] for t in tails.values()),c3[n],atol=2e-14,rtol=2e-12)
        return a1,a2,tails,direct,iterated
    af,bf,tf,df,uf=endpoint(outgoing)
    ai,bi,ti,di,ui=endpoint(incoming)
    columns=len(m.pilot.matrix);result={}
    for n,space in m.spaces5.items():
        values=[ai[n],bi[n],*(v[n] for v in ti.values())]
        acted=shell_density_between(space,space,np.column_stack(values),delta,
            b_GeV=m.pilot.width_GeV,charge=charge)
        blocks=[acted[:,k*columns:(k+1)*columns] for k in range(len(values))]
        result['Q5_'+n+'_quartic_square']=bf[n].conj().T@blocks[1]
        for k,label in enumerate(ti):
            result['Q5_'+n+'_'+label]=af[n].conj().T@blocks[k+2]+tf[label][n].conj().T@blocks[0]
    for n,space in m.spaces6.items():
        acted=shell_density_between(space,space,np.column_stack((di[n],ui[n])),delta,
            b_GeV=m.pilot.width_GeV,charge=charge)
        direct,iterated=acted[:,:columns],acted[:,columns:]
        result['Q6_'+n+'_direct_square']=df[n].conj().T@direct
        result['Q6_'+n+'_iterated_square']=uf[n].conj().T@iterated
        result['Q6_'+n+'_direct_iterated_interference']=df[n].conj().T@iterated+uf[n].conj().T@direct
    return result


def QQ_backflow_plus_parts(outgoing,incoming,transfer,*,charge):
    """Resolve W55 into physical contact, pair-current and finite Wick H.

    These six matrices subdivide, not augment, the two QQ_backflow words.
    They carry the unchanged real source/projector action in each sector.
    """
    m=incoming.owner
    if outgoing.owner is not m:
        raise ValueError('this same-basis loop diagnostic requires one shared model')
    def endpoint(s):
        a=s.five_wave_coefficients[0];action=m.qq.action
        acted={'contact':action.contacts.apply(a),'pair':action.pair_current.apply(a),
               'finite_Wick':{n:action.wick_diagonal[n][:,None]*a[n] for n in a}}
        return {label:{n:m.qq.propagation[n].resolve(s.energy_GeV2,v) for n,v in values.items()}
                for label,values in acted.items()}
    tf,ti=endpoint(outgoing),endpoint(incoming)
    af,ai=outgoing.five_wave_coefficients[0],incoming.five_wave_coefficients[0]
    columns=len(m.pilot.matrix);result={}
    for n,space in m.spaces5.items():
        joined=np.column_stack([ai[n],*(t[n] for t in ti.values())])
        acted=shell_density_between(space,space,joined,transfer,b_GeV=m.pilot.width_GeV,charge=charge)
        leading=acted[:,:columns]
        for k,label in enumerate(ti):
            result['resolved_Q5_'+n+'_QQ_'+label]=(af[n].conj().T@acted[:,(k+1)*columns:(k+2)*columns]
                                                  +tf[label][n].conj().T@leading)
    return result


def evaluate(case,*,resolve_QQ=False):
    start=monotonic();m,parameters=build_model(case)
    f,i=m.snapshot(1.39),m.snapshot(1.47)
    print(case+': resolving fourteen coherent fourth-order plus-current words',flush=True)
    words=fourth_order_plus_words(f,i,(.12,.16),charge='electromagnetic')
    raw=sum(words.values())
    normalized=f.whitener@raw@i.whitener
    parts=QQ_backflow_plus_parts(f,i,(.12,.16),charge='electromagnetic') if resolve_QQ else {}
    for n in m.spaces5:
        if parts:
            np.testing.assert_allclose(sum(v for k,v in parts.items() if k.startswith('resolved_Q5_'+n+'_QQ_')),
                words['Q5_'+n+'_QQ_backflow'],atol=2e-14,rtol=2e-12)
    arrays={**words,**parts,'raw_plus_g4':raw,'normalized_plus_g4':normalized}
    report=dict(case=case,parameters=parameters,energy_out_GeV2=1.39,energy_in_GeV2=1.47,
        transfer_GeV=[.12,.16],current_normalization='g4 coefficient of Jplus/(2Pplus), before coupling multiplication',
        words_Frobenius={k:float(np.linalg.norm(v)) for k,v in words.items()},
        QQ_subdivision_Frobenius={k:float(np.linalg.norm(v)) for k,v in parts.items()},
        QQ_subdivision_is_not_added_again_to_the_raw_current=True,
        paired_normal_order_subtraction=bool(m.wick_operators.normal_order_subtraction),
        finite_Wick_label_means='Residual explicit mu_q^2 D_q + mu_g^2 D_g after exact paired normal-order subtraction; not the divergent Wick trace.' if m.wick_operators.normal_order_subtraction else 'Unsubtracted ordered Wick mass plus explicit finite terms.',
        raw_plus_g4_Frobenius=float(np.linalg.norm(raw)),
        normalized_plus_g4_Frobenius=float(np.linalg.norm(normalized)),
        terms_are_coherent_matrix_amplitudes_not_probabilities=True,
        finite_conditions_are_uncalibrated_test_values=True,physical=False,
        continuum_convergence_established=False,seconds=monotonic()-start)
    return report,arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('baseline','Q5_shell1','Q6_shell1','Q5_Q6_shell1','Q5_shell2'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resolve-QQ',action='store_true',help='Also subdivide the existing QQ terms; never add these subterms a second time')
    args=parser.parse_args();matrices=args.output.with_suffix('.npz')
    if args.output.exists() or matrices.exists():parser.error('select new outputs to preserve prior evidence')
    result,arrays=evaluate(args.case,resolve_QQ=args.resolve_QQ);result['full_matrix_file']=matrices.name
    np.savez_compressed(matrices,**arrays)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

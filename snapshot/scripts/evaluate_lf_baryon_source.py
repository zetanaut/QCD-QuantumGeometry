"""Replay complete mixed source/emission baryon waves; no physical fit."""
import json
from math import sqrt
import numpy as np

from deuteron_wigner.quantum.finite_lf_coupling import reference_inputs
from deuteron_wigner.quantum.lf_baryon_emission import norm2,difference,occupied
from deuteron_wigner.quantum.lf_baryon_source import BaryonSource


def evaluate():
    cell,m,g,_=reference_inputs();c=np.array([[.14,.037],[.037,-.06]])
    p=(.375,.2,0.);q=np.array([0,-.2,.2])
    def build(species='p',sign=1,*,gauge=False,representation='good',shell=1):
        return BaryonSource(cell,m,c,g,.43,.27,[p],sign*q,shell,12,(.5,0,0,0),
            lambda l:l.interpolator(p,species,1),pure_gauge=gauge,representation=representation)
    cases=[];original_wave=None;physical_small=None
    def physical(obj,wave):
        return {(kind,k,tuple((obj.layer.momenta[i//12],i%12) for i in occupied(mask))):v
                for (kind,k,mask),v in wave.items()}
    for species in ('p','n'):
        for sign in (1,-1):
            obj=build(species,sign);wave=obj.response()
            entry=dict(species=species,source_sign=sign,dimensions=obj.diagnostics(),
                mixed_wave_norm_squared_GeV_minus2=norm2(wave),
                phi_norm_squared_GeV_minus2=norm2({k:v for k,v in wave.items() if k[0]=='phi'}),
                gluon_norm_squared_GeV_minus2=norm2({k:v for k,v in wave.items() if k[0]=='g'}))
            if species=='p' and sign==1:
                entry['quadrature_refinement_error_GeV_minus1']=difference(wave,obj.response(nodes=96))
                entry['omission_wave_errors_GeV_minus1']={name:difference(wave,obj.response(omit=name))
                    for name in ('before','after_quark','collective','fermion_contact','collective_endpoint','outside_intermediates','spectators')}
                original_wave=wave;physical_small=physical(obj,wave)
            cases.append(entry)
    ref=build(representation='dirac')
    coarse,nc=ref.ode_response();fine,nf=ref.ode_response(rtol=3e-12,atol=3e-14)
    ode=dict(coarse_wave_error_GeV_minus1=difference(original_wave,coarse),
             refined_wave_error_GeV_minus1=difference(original_wave,fine),
             coarse_rhs_evaluations=nc,refined_rhs_evaluations=nf)
    pure=[]
    for sign in (1,-1):
        obj=build(sign=sign,gauge=True);wave=obj.response()
        pure.append(dict(source_sign=sign,whole_wave_norm=sqrt(norm2(wave)),
            spectator_omitted_norm=sqrt(norm2(obj.response(omit='spectators'))),
            outside_intermediates_omitted_norm=sqrt(norm2(obj.response(omit='outside_intermediates'))),
            fermion_contact_omitted_norm=sqrt(norm2(obj.response(omit='fermion_contact')))))
    larger=build(shell=2);wave=larger.response(nodes=96);physical_large=physical(larger,wave)
    assert physical_small.keys()<=physical_large.keys()
    scan=dict(larger_dimensions=larger.diagnostics(),
        old_amplitude_error_GeV_minus1=difference(physical_small,{k:physical_large[k] for k in physical_small}),
        newly_represented_wave_norm_GeV_minus1=sqrt(norm2({k:v for k,v in physical_large.items() if k not in physical_small})),
        larger_norm_squared_GeV_minus2=norm2(wave))
    return dict(scope='one emission and one classical q+=0 EM insertion, full free-reference baryon state, not bound nucleons',
        inputs=dict(cell_lengths_GeV_minus1=cell.lengths,mass_squared_GeV2=m.tolist(),C_GeV_minus2=c.tolist(),
            quark_mass_GeV=.27,Yukawas=g.tolist(),gs=.43,quark_momentum_GeV=p,
            transfer_GeV=q.tolist(),duration_GeV_minus1=12,each_real_source_Fourier_amplitude=.5),
        ordinary_cases=cases,independent_Dirac_ODE=ode,pure_gauge=pure,window_enlargement=scan,
        interpretation=dict(real_source_squared_norm_is_sum_of_both_orthogonal_transfer_branches=True,
            mixed_norm_is_derivative_coefficient_not_probability=True,spectator_omission_can_pass_pure_gauge=True,
            global_Fock_cutoff_or_continuum_completion=False,full_bound_nucleon_current=False,
            physical_covariance_defect_repaired=False,production_model_or_physical_fit_selected=False,
            old_loops_requantized=False,old_BLFQ_H2_rebuilt=False,public_push=False))


if __name__=='__main__':
    print(json.dumps(evaluate(),indent=2))

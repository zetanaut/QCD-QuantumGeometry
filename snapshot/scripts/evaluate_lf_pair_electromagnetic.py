"""Replay normalized one-Yukawa/one-EM LF transition and source controls."""
import json
import numpy as np
import torch
from deuteron_wigner.quantum.finite_lf_coupling import reference_inputs
from deuteron_wigner.quantum.lf_pair_electromagnetic import (
    tree_response,PulseTransition,pulse_parameter_features,torch_pulse_response,
)


def seed_for(charge,m,*,eigen=False):
    if charge==0:
        if not eigen: return reference_inputs()[3]
        _,u=np.linalg.eigh(m[0]);v=np.zeros((2,4),complex);v[:,0]=u[:,0];return v
    v=np.zeros((2,4),complex)
    v[:,1]=np.linalg.eigh(m[1])[1][:,0] if eigen else [1,.3j]
    v[:,2]=1j*charge*v[:,1]
    return v/np.linalg.norm(v)


def evaluate():
    cell,m,g,_=reference_inputs();c=np.array([[.14,.037],[.037,-.06]])
    a=np.array([.125,.4,.2]);b=np.array([.375,-.2,-.4]);p=np.array([.5,0,0])
    errors=[];wards=[]
    for charge in (-1,0,1):
        seed=seed_for(charge,m,eigen=True)
        for omega in (.11,.17,.27):
            columns=[]
            for e in np.eye(4):
                r=tree_response(a,b,p,.27,m,c,g,seed,e,omega,cell.volume)
                errors.append(float(np.linalg.norm(r['covariant']-r['light_front'])))
                columns.append(r['light_front'])
            wards.append(float(np.linalg.norm(omega*columns[1]+.2*columns[2]-.2*columns[3])))
    parent=np.array([.5,.2,0]);transfer=np.array([0,-.2,.2]);duration=12.
    waves=[];gauges=[]
    for charge in (-1,0,1):
        seed=seed_for(charge,m)
        for sign in (-1,1):
            # Both half-amplitude branches make one REAL A+ cosine source.
            args=(cell,m,c,g,.27,parent,sign*transfer,1,duration,[.5,0,0,0])
            model=PulseTransition(*args);wave=model.response(seed)
            independent=PulseTransition(*args,representation='dirac')
            reference,nfev=independent.ode_response(seed)
            waves.append({'charge':charge,'source_fourier_sign':sign,
                          'final_pair_routes':len(model.routes),
                          'resolved_complex_amplitudes':int(wave.size),
                          'outside_window_intermediates':model.outside_intermediates(),
                          'mixed_wave_norm_GeV_minus1':float(np.linalg.norm(wave)),
                          'independent_Dirac_ODE_error_GeV_minus1':float(np.linalg.norm(wave-reference)),
                          'ode_function_evaluations':nfev})
            pure=PulseTransition(*args,pure_gauge=True)
            gauges.append({'charge':charge,'source_fourier_sign':sign,
                           'endpoint_zero_gauge_wave_norm':float(np.linalg.norm(pure.response(seed))),
                           'omitted_fermion_contact_wave_norm':float(np.linalg.norm(pure.response(seed,omit='fermion_contact'))),
                           'omitted_outside_intermediate_wave_norm':float(np.linalg.norm(pure.response(seed,omit='outside_intermediates')))})
    seed=seed_for(1,m)
    model=PulseTransition(cell,m,c,g,.27,parent,transfer,1,duration,[1,0,0,0])
    full=model.response(seed)
    negative={key:float(np.linalg.norm(full-model.response(seed,omit=key))) for key in (
        'quark','anti','collective','fermion_contact','collective_endpoint','source_derivatives','outside_intermediates')}
    scan=[]
    for shell in (1,2,3,4,5):
        norm2=0.;count=0;diff2=0.
        for sign in (-1,1):
            model=PulseTransition(cell,m,c,g,.27,parent,sign*transfer,shell,duration,[.5,0,0,0])
            wave=model.response(seed,nodes=96)
            norm2+=float(np.linalg.norm(wave)**2);count+=len(model.routes)
            diff2+=float(np.linalg.norm(wave-model.response(seed,nodes=128))**2)
        scan.append({'final_transverse_shell':shell,'both_source_sign_pair_routes':count,
                     'mixed_wave_norm_squared_GeV_minus2_NOT_probability':norm2,
                     'quadrature_refinement_error_GeV_minus1':np.sqrt(diff2)})
    epsilon=[.5,.2,-.1,.3]
    features=pulse_parameter_features(cell,m,.27,parent,transfer,1,duration,epsilon,seed,nodes=48)
    x=torch.tensor([*g,c[0,0],c[0,1],c[1,1]],dtype=torch.float64,requires_grad=True)
    wave=torch_pulse_response(features,x[:2],x[2:])
    probe=np.random.default_rng(37).normal(size=wave.shape)+1j*np.random.default_rng(38).normal(size=wave.shape)
    probe/=np.linalg.norm(probe)
    objective=torch.real(torch.sum(torch.tensor(probe).conj()*wave))
    grad=torch.autograd.grad(objective,x)[0].numpy();fd=[];step=2e-4;base=x.detach().numpy()
    def value(par):
        cc=np.array([[par[2],par[3]],[par[3],par[4]]])
        z=PulseTransition(cell,m,cc,par[:2],.27,parent,transfer,1,duration,epsilon).response(seed,nodes=48)
        return np.vdot(probe,z).real
    for i in range(5):
        d=step*np.eye(5)[i];fd.append((value(base+d)-value(base-d))/(2*step))
    return {'scope':'one bare Yukawa and one classical q+=0 EM insertion; full resolved final-window Fock response',
            'preedit_commit':'28fd253fb3b09c878d0f874c02ca6f48576f5506',
            'cell_volume_GeV_minus3':cell.volume,'quark_mass_GeV':.27,
            'mass_squared_blocks_GeV2':m.tolist(),'yukawa_couplings':g.tolist(),
            'C_GeV_minus2':c.tolist(),'D_E_begin_at_two_photons_not_set_to_zero':True,
            'time_parent_plus_transverse_GeV':parent.tolist(),'time_source_transfer_plus_transverse_GeV':transfer.tolist(),
            'duration_GeV_minus1':duration,'all_component_frequency_cases':36,
            'max_covariant_LF_normalized_tree_error':max(errors),'max_tree_Ward_norm_GeV':max(wards),
            'real_cosine_source_wave_runs':waves,'pure_gauge_runs':gauges,
            'unit_positive_Fourier_Aplus_omission_errors_GeV_minus1':negative,
            'final_window_scan':scan,'shared_g1_g2_C00_C01_C11_derivatives':grad.tolist(),
            'independent_centered_difference_derivatives':fd,'gradient_max_error':float(np.max(abs(grad-fd))),
            'mass_derivatives_in_new_Torch_feature_map':False,
            'all_four_component_quantum_loop_current':False,'nonzero_plus_source_pair_sectors':False,
            'global_Fock_or_continuum_completion':False,'physical_covariance_defect_repaired':False,
            'bound_nucleon_or_nuclear_model_generated':False,'old_loops_requantized':False,
            'production_model_selected_or_parameters_refitted':False,'public_push_or_review_export':False}


if __name__=='__main__':
    print(json.dumps(evaluate(),indent=2,sort_keys=True))

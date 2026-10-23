"""Replay the normalized first-order LF Fock coupling reference (prints JSON)."""
import json
import numpy as np
import torch

from deuteron_wigner.quantum.finite_lf_coupling import (
    PairReference, direct_time_wave, first_order_wave, reference_inputs,
)


def evaluate():
    cell,m,g,v=reference_inputs();duration=12.
    scans=[]
    for n in (0,1,2,3,5,8):
        ref=PairReference(cell,(.5,0,0),.27,n)
        parent,w=first_order_wave(ref,m,g,v,duration)
        a=w.numpy();r=a.shape[0]
        scans.append({'transverse_shell':n,'pair_momentum_routes':r,
                      'resolved_pair_amplitudes':int(a.size),
                      'leading_pair_norm_squared':float(np.linalg.norm(a)**2),
                      'missing_reversed_assignment_norm':float(np.linalg.norm(a[r//2:])),
                      'parent_norm_error':abs(float(torch.linalg.vector_norm(parent))-1)})
    ref=PairReference(cell,(.5,0,0),.27,3)
    _,w=first_order_wave(ref,m,g,v,duration)
    direct,error=direct_time_wave(ref,m,g,v,duration)
    direction=np.tile(np.array([[.2,1],[1,-.3]]),(4,1,1))
    x=torch.tensor([0.,g[0],g[1]],dtype=torch.float64,requires_grad=True)
    probe=torch.tensor(np.random.default_rng(42).normal(size=w.shape)
                       +1j*np.random.default_rng(43).normal(size=w.shape))
    probe=probe/torch.linalg.vector_norm(probe)
    def response(par):
        _,wave=first_order_wave(ref,torch.tensor(m)+par[0]*torch.tensor(direction),par[1:],v,duration)
        return torch.real(torch.sum(probe.conj()*wave))
    grad=torch.autograd.grad(response(x),x)[0].numpy()
    h=2e-5;fd=[]
    for i in range(3):
        e=np.eye(3)[i]*h
        vals=[response(torch.tensor(x.detach().numpy()+k*e)).item() for k in (-2,-1,1,2)]
        fd.append((vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*h))
    charges=[]
    q=np.tile(np.repeat([2/3,-1/3],3),2)
    qpair=(q[:,None]-q[None,:]).reshape(144)
    for charge in (-1,0,1):
        seed=v.copy()
        if charge:
            seed=np.zeros((2,4),complex);seed[:,1]=[1,.3j]
            seed[:,2]=1j*charge*seed[:,1];seed/=np.linalg.norm(seed)
        _,wave=first_order_wave(ref,m,g,seed,duration)
        charges.append({'electric_charge':charge,
                        'total_charge_residual':float(np.linalg.norm((qpair[None,:]-charge)*wave.numpy())),
                        'leading_pair_norm_squared':float(torch.linalg.vector_norm(wave)**2)})
    return {
        'scope':'first order in a smoothly switched bare Yukawa interaction; finite AP/periodic LF cell',
        'cell_lengths_GeV_minus1':list(cell.lengths),'cell_volume_GeV_minus3':cell.volume,
        'spacing_GeV':cell.spacing.tolist(),'parent_plus_transverse_GeV':[.5,0,0],
        'quark_mass_GeV':.27,'mass_squared_blocks_GeV2':m.tolist(),
        'yukawa_couplings':g.tolist(),'duration_GeV_minus1':duration,
        'seed_real':v.real.tolist(),'seed_imag':v.imag.tolist(),
        'pulse':'lambda*sin(pi*x_plus/T)^4 for 0<x_plus<T',
        'order':'psi=psi_0+lambda*psi_1+O(lambda^2); norm(psi_1)^2 is the leading transition coefficient',
        'domain_scan':scans,
        'fixed_charge_runs':charges,
        'shell3_full_complex_wave_error':float(np.linalg.norm(w.numpy()-direct)),
        'independent_time_integral_error_estimate':error,
        'shell3_scalar_mass_squared_source_and_yukawa_gradients':grad.tolist(),
        'independent_fourth_order_finite_difference_gradients':fd,
        'gradient_max_error':float(np.max(abs(grad-fd))),
        'source_is_electromagnetic_current':False,
        'first_order_infinite_transverse_limit_proved':False,
        'second_order_state_or_unitarity_completion':False,
        'old_loops_used_as_bare_vertices':False,
        'physical_model_or_coefficients_selected_or_refitted':False,
        'physical_nucleon_or_nucleus_generated':False,
        'public_push_or_review_export':False,
    }


if __name__=='__main__':
    print(json.dumps(evaluate(),indent=2,sort_keys=True))

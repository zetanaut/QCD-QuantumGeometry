"""Replay nonstationary LF source and weak-wave controls, with no fit."""
import json
import numpy as np
import torch
from scipy.linalg import expm
from deuteron_wigner.quantum.nonstationary_lf_source import (
    SourceFrame,inverse_jet,mapped_jet,evolve_weak_wave,canonical_wave_jet,
    tensor,plus,minus,commutator,
)


def cart(P,w,x,y): return np.array([(P+w)/np.sqrt(2),x,y,(P-w)/np.sqrt(2)])
def pairs(values): return np.stack([values.real,values.imag],axis=-1).tolist()
def size(x): return float(torch.linalg.norm(tensor(x)))


def evaluate():
    f=SourceFrame(cart(.8,0,.1,.2),cart(.13,.07,.05,-.03),cart(.07,-.04,-.02,.04),
        np.array([.2,.4,-.1,.3]),np.array([.1,-.2,.35,-.15]))
    args=(np.array([[.12,.02],[.02,.25]]),np.array([[.4,.05],[.05,.2]]),
        np.array([[-.1,.03],[.03,.06]]),np.array([[.7,.1],[.1,.4]]))
    omega=-.15;m=f.mapping(*args);p=f.frequency(2,omega)
    actual=f.inverse_jets(*args,omega,actual=True);direct=inverse_jet(actual);result=mapped_jet(m,p)
    omissions={name:size(direct[3]-mapped_jet(m,p,omit=name)[3]) for name in ['connection','endpoints','direct','one_order','right_adjoint']}
    all_components=[]
    for mu in range(4):
        for nu in range(4):
            s=SourceFrame(f.p0,f.qa,f.qb,np.eye(4)[mu],np.eye(4)[nu])
            a=inverse_jet(s.inverse_jets(*args,omega,actual=True));b=mapped_jet(s.mapping(*args),s.frequency(2,omega))
            all_components.append(dict(Cartesian_polarization_indices=[mu,nu],mixed_error=size(a[3]-b[3]),mixed_response_norm=size(a[3])))
    ward=[]
    for which in range(2):
        eps=[f.epsa,f.epsb];eps[which]=[f.qa,f.qb][which];s=SourceFrame(f.p0,f.qa,f.qb,*eps)
        v=[s.unnormalize_response(x).numpy() for x in mapped_jet(s.mapping(*args),s.frequency(2,omega))]
        generator=np.zeros((4,4))
        for i,j in ([(1,0),(3,2)] if which==0 else [(2,0),(3,1)]): generator[i,j]=1;generator[j,i]=-1
        difference=commutator(v[2 if which==0 else 1],np.kron(generator,np.eye(2)))
        ward.append(dict(photon=which,offshell_Ward_error=float(np.linalg.norm(v[3]-difference)),nonzero_endpoint_difference_norm=float(np.linalg.norm(difference))))
    boosts=[]
    for beta in [[.17,-.12,.08],[-.13,.09,.19]]:
        a=np.zeros((4,4));a[0,1:]=a[1:,0]=beta;L=expm(a)
        s=SourceFrame(L@f.momenta(omega)[0],L@f.qa,L@f.qb,L@f.epsa,L@f.epsb)
        v=mapped_jet(s.mapping(*args),s.frequency(2,0))[3]
        boosts.append(dict(boost_generator=beta,scalar_unnormalized_error=size(s.unnormalize_response(v)-f.unnormalize_response(result[3])),
            omitting_external_kinetic_factors_error=size(v-result[3])))
    expanded=f.weak_extension();rng=np.random.default_rng(39);seed=rng.normal(size=8)+1j*rng.normal(size=8);seed/=np.linalg.norm(seed)
    u=np.zeros(32,complex);u[:8]=seed
    state=evolve_weak_wave(expanded.matrices(*args),11.,u)
    exact=canonical_wave_jet(expanded.mapping(*args),expanded.frequency(2,0),11.,u)
    small=canonical_wave_jet(m,f.frequency(2,0),11.,seed)
    z=torch.zeros(3,dtype=torch.float64)
    def variation(theta,which):
        h,c,d,e=[tensor(x) for x in args];direction=tensor([[1.,.2],[.2,0.]])
        carrier=f if which=='response' else expanded
        model=carrier.mapping(h+theta[0]*direction,c+theta[1]*direction,d,e+theta[2]*direction)
        if which=='response': return mapped_jet(model,p)[3][6,0].real
        v=canonical_wave_jet(model,expanded.frequency(2,0),11.,u)[3,6]
        return v.real+.3*v.imag
    gradients={which:torch.autograd.functional.jacobian(lambda x:variation(x,which),z).tolist() for which in ['response','state']}
    return dict(scope='Charged minimal/C/D/E reference: exact mixed a,b source coefficients in positive-plus LF channels; nonstationary canonical map and weak one-charge WAVE amplitudes. A 16-route carrier retains all paths from the four seeds at this order. Not a complete sourced Fock state, vacuum/pair-sector solution or physical hadron.',
        parameters=dict(mass_squared_GeV2=args[0].tolist(),C_GeV_minus2=args[1].tolist(),D_GeV_minus2=args[2].tolist(),E_GeV_minus4=args[3].tolist(),
            p0_cartesian_GeV=f.p0.tolist(),qa_cartesian_GeV=f.qa.tolist(),qb_cartesian_GeV=f.qb.tolist(),epsa=f.epsa.tolist(),epsb=f.epsb.tolist(),
            common_minus_shift_GeV=omega,route_plus_GeV=plus(f.momenta(omega)).tolist(),route_minus_offsets_GeV=minus(f.momenta(0)).tolist()),
        response=dict(canonical_jet_norm_error=[size(a-b) for a,b in zip(direct,result)],
            units_by_order=['GeV^-1','GeV^-2','GeV^-2','GeV^-3'],
            original_scalar_two_source_block_GeV_minus4=f.unnormalize_response(result[3])[6:,:2].real.tolist(),
            time_dependent_connection_norm_GeV_minus1=size(m['connection']),negative_control_errors_GeV_minus3=omissions,
            all_16_Cartesian_polarization_pairs=all_components,offshell_Ward=ward,boosts=boosts),
        weak_wave_response=dict(source_order='0,a,b,a*b connected one-charge wave amplitudes; not complete sourced Fock-state derivatives',
            seed_routes=list(map(list,f.routes)),all_weak_routes=list(map(list,expanded.routes)),
            carrier_momenta_cartesian_GeV=expanded.momenta(0).tolist(),
            spatial_mode_step_GeV=[.01,.01,.01],periodic_box_lengths_GeV_minus1=[2*np.pi/.01]*3,
            normalization='Periodic scalar modes exp(-ip.x)/sqrt(2p+ V); volume cancels for external Fourier-source amplitudes a,b in GeV. The box is a reference convention, not a QCD matching scale.',
            all_retained_four_corner_amplitude_errors_by_order=torch.linalg.norm(small-exact[:,:8],dim=1).tolist(),
            previously_unrepresented_wave_amplitude_norms_by_order=torch.linalg.norm(exact[:,8:],dim=1).tolist(),
            times_GeV_minus1=state['times'].tolist(),initial_normalized_complex_amplitudes=pairs(u),
            error_units_by_source_order=['dimensionless','GeV^-1','GeV^-1','GeV^-2'],
            original_vs_mapped_errors_by_order=np.max(abs(state['original']-state['mapped_original']),axis=(0,2)).tolist(),
            direct_ODE_vs_differentiable_exponential_errors_by_order=torch.linalg.norm(exact-tensor(state['canonical'][-1]),dim=1).tolist(),
            final_mixed_canonical_amplitudes_GeV_minus2=pairs(state['canonical'][-1,3]),
            canonical_mixed_norm_residual_GeV_minus2=float(max(abs(state['canonical_ab_norm']))),
            original_Noether_mixed_norm_residual_GeV_minus2=float(max(abs(state['original_ab_Noether_norm']))),
            omitting_D_velocity_norm_defect_GeV_minus2=float(max(abs(state['omitting_D_velocity_norm'])))),
        differentiable_shared_map=dict(direction_matrix=[[1.,.2],[.2,0.]],parameter_units=['GeV^2','GeV^-2','GeV^-4'],
            response_units='GeV^-3 before parameter differentiation',state_projection_units='GeV^-2 before parameter differentiation',gradients=gradients),
        reassessment=dict(time_dependent_canonical_frame_and_direct_terms_required=True,all_declared_source_matrices_and_complex_phases_retained=True,
            mixed_two_source_order_complete_on_declared_routes=True,independent_quark_loop_LF_minus_transverse_currents_supplied=False,
            zero_plus_or_pair_creation_crossings_resolved=False,exact_finite_source_or_nonquadratic_LF_Fock_dynamics_supplied=False,
            source_free_physics_or_fitted_matrices_changed=False,known_loops_requantized=False,physical_nucleon_or_nucleus_generated=False,
            production_model_selected=False,public_push=False,
            next='Rejoin preserved graded quark/color and canonical LF operator leaves: pre-evaluate their normalized finite wavepacket/state coupling and linked source response, retaining all spin/flavor/color and physically defined connectivity. Do not invent a mode measure, substitute the reference into nucleon binding or rebuild BLFQ/H2.'))


if __name__=='__main__': print(json.dumps(evaluate(),indent=2,allow_nan=False))

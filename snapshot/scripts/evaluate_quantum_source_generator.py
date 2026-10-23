"""Replay canonical nonzero-mode quantum controls; no physical refit."""
import json
import numpy as np
from deuteron_wigner.quantum.quantum_source_generator import (
    ChargedModes,JointSourceAction,action_mode_lagrangian,pulse,phase_matrix,
    phase_evolution,initial_covariance,charged_excitation,gaussian_state,
    gaussian_wavefunction,gaussian_fock_coefficients,FockSourceControl,
    lf_weak_source,lf_response_parts,
)
from deuteron_wigner.quantum.joint_source_matching import action_contact


def complex_pair(z): return [float(z.real),float(z.imag)]


def model(nodes):
    if nodes==1:
        return ChargedModes(np.array([[.12]]),JointSourceAction(*[np.array([[x]]) for x in [.4,-.1,.7]]),.23)
    return ChargedModes(np.array([[.12,.02],[.02,.25]]),JointSourceAction(
        np.array([[.4,.05],[.05,.2]]),np.array([[-.1,.03],[.03,.06]]),np.array([[.7,.1],[.1,.4]])),.23)


def quantum_control(nodes,amplitude,shells):
    s=model(nodes);m=s.dimension;w,u=s.normal_basis();t=np.block([[u,u*0],[u*0,u]])
    p=phase_evolution(s,8,amplitude);normal=t.T@p@t;scale=np.r_[np.sqrt(w),1/np.sqrt(w)]
    def scaled_matrix(matrix): return scale[:,None]*(t.T@matrix@t)/scale[None,:]
    q,v=np.random.default_rng(17).normal(0,.3,(2,m));z,b,V,*_=s.matrices(.21,.11)
    density_error=abs(float(action_mode_lagrangian(s,q,v,.21,.11))-(.5*v@z@v+v@b@q-.5*q@V@q))
    classical=dict(original_density_error_GeV=density_error,
        endpoint_canonical_vs_Lagrange_dimensionless_error=float(np.linalg.norm(scaled_matrix(p-phase_evolution(s,8,amplitude,method='lagrangian')))),
        naive_minus_interaction_L_dimensionless_error=float(np.linalg.norm(scaled_matrix(p-phase_evolution(s,8,amplitude,method='naive')))),
        frozen_source_derivatives_dimensionless_error=float(np.linalg.norm(scaled_matrix(p-phase_evolution(s,8,amplitude,method='frozen_source_derivatives')))),
        symplectic_error=float(np.linalg.norm(p.T@phase_matrix(m)@p-phase_matrix(m))))
    minima=[]
    for time in np.linspace(0,8,201):
        a,ad,add=pulse(time,8,amplitude);z,_,V,*_=s.matrices(a,ad,add)
        minima.append([np.linalg.eigvalsh(z).min(),np.linalg.eigvalsh(V).min()])
    g=gaussian_state(s,8,amplitude);cov=[normal@initial_covariance(w,exc)@normal.T for exc in [None,charged_excitation(s)]]
    coordinates=np.random.default_rng(28).normal(0,.6,(5,m))/np.sqrt(w)
    refinements=[]
    for shell in shells:
        f=FockSourceControl(s,shell);state=f.evolve(8,amplitude);exact=gaussian_fock_coefficients(g,f.states)
        overlap=np.diag(f.initial.conj().T@state);norm=np.sum(abs(state)**2,axis=0)
        wf=[];covariance=[]
        for j in range(2):
            wf.append(max(abs(f.wavefunction(state[:,j],q)-gaussian_wavefunction(g,q,one_particle=bool(j))) for q in coordinates))
            covariance.append(np.linalg.norm(scale[:,None]*(f.covariance(state[:,j])-cov[j])*scale[None,:]))
        refinements.append(dict(shell=shell,states=f.size,
            vacuum_and_charged_complex_survival=[complex_pair(x) for x in overlap],
            survival_errors=abs(overlap-[g['vacuum_overlap'],g['one_particle_overlap']]).tolist(),
            norm_errors=abs(norm-1).tolist(),
            projected_full_amplitude_vector_errors=np.linalg.norm(state-exact,axis=0).tolist(),
            infinite_Gaussian_outside_shell_norm_deficits=(1-np.sum(abs(exact)**2,axis=0)).tolist(),
            outside_shell_deficit_precision_note='Raw subtraction; values near 1e-10 can have either sign from ODE/roundoff, not negative physical probability.',
            wavefunction_five_probe_absolute_errors=wf,
            wavefunction_units=f'GeV^{m/4:g} in the normalized Q representation',
            dimensionless_covariance_errors=covariance,
            charge_expectations=[float(np.vdot(state[:,j],f.charge@state[:,j]).real) for j in range(2)],
            momentum_expectations_GeV=[float(np.vdot(state[:,j],f.momentum@state[:,j]).real) for j in range(2)],
            pair_excitation_probabilities=[float(sum(abs(state[f.total_excitation>j,j])**2)) for j in range(2)]))
    return dict(nodes=nodes,real_oscillators=m,source_free_mass_squared_GeV2=s.mass_squared.tolist(),
        C_GeV_minus2=s.source.c.tolist(),D_GeV_minus2=s.source.d.tolist(),E_GeV_minus4=s.source.e.tolist(),
        momentum_GeV=s.momentum,pulse_duration_GeV_minus1=8.,pulse_amplitude_GeV=amplitude,
        normalized_spatial_modes='sqrt(2/V) cos(kx), sqrt(2/V) sin(kx); V cancels in quadratic slice',
        minimum_sampled_kinetic_eigenvalue=float(np.min(minima,axis=0)[0]),minimum_sampled_V_eigenvalue_GeV2=float(np.min(minima,axis=0)[1]),
        gaussian_complex_survival=[complex_pair(g[x]) for x in ['vacuum_overlap','one_particle_overlap']],
        classical_control=classical,refinements=refinements)


def evaluate():
    h=np.array([[.2,.02+.03j],[.02-.03j,.7]]);d=np.array([[.7,.1j],[-.1j,.4]])
    b=np.array([[.1,.2],[.2,-.1]]);c=np.array([[.1,.03j],[-.03j,.1]])
    shift,endpoint,contact=lf_weak_source(h,d,b,c);omega=.08
    exact,parts=lf_response_parts(h,d,omega,b,c);errors=[]
    for eps in [1e-3,5e-4,2.5e-4]:
        W=np.eye(2)+eps*endpoint
        direct=np.linalg.inv(h-omega*np.eye(2)-eps*(d*omega**2+b*omega+c))
        mapped=W@np.linalg.inv(h+eps*shift-omega*np.eye(2))@W.conj().T+eps*contact
        values,u=np.linalg.eigh(h+eps*shift);psi=W@u;norm=np.empty((2,2),complex);residual=[]
        for n in range(2):
            residual.append(np.linalg.norm((h-values[n]*np.eye(2)-eps*(d*values[n]**2+b*values[n]+c))@psi[:,n]))
            for m in range(2): norm[n,m]=np.vdot(psi[:,n],(np.eye(2)+eps*((values[n]+values[m])*d+b))@psi[:,m])
        errors.append(dict(epsilon=eps,Green_map_error=float(np.linalg.norm(mapped-direct)),pole_equation_residual=float(max(residual)),pole_norm_error=float(np.linalg.norm(norm-np.eye(2)))))
    s=model(2);E=s.source.e;source=JointSourceAction(E*0,E*0,E);P=.8;qp=.3;omega=.2
    p=np.array([(P+omega)/np.sqrt(2),.1,.2,(P-omega)/np.sqrt(2)]);q=np.array([qp/np.sqrt(2),0,0,qp/np.sqrt(2)]);pol=np.array([0.,1,0,0])
    actual=np.array([[action_contact(source,p,p,q,-q,pol,pol,np.eye(2)[i],np.eye(2)[j]).real for j in range(2)] for i in range(2)])
    f=FockSourceControl(model(1),10);quantum=[];independent=[];step=1e-4
    for amp in [.08-step,.08+step]:
        quantum.append(np.vdot(f.initial[:,1],f.evolve(8,amp)[:,1]));independent.append(gaussian_state(model(1),8,amp)['one_particle_overlap'])
    return dict(scope='Actual regular-time charged quadratic reference states and linked pulse response; bounded stationary LF one-insertion map. Not physical pion/nucleon states or complete LF quantization.',
        quantum_controls=[quantum_control(1,.15,[4,8,12,16]),quantum_control(2,.1,[4,6,8])],
        dynamic_source_derivative=dict(amplitude_GeV=.08,step_GeV=step,units='GeV^-1',
            quantum_complex_derivative=complex_pair((quantum[1]-quantum[0])/(2*step)),independent_complex_derivative=complex_pair((independent[1]-independent[0])/(2*step))),
        stationary_LF=dict(source_order='first epsilon coefficient; nonzero positive plus, zero net source minus/plus transfer in actual E example',
            generic_matrices_units='h GeV, d GeV^-1, b dimensionless, c GeV; epsilon dimensionless, omega GeV',
            generic_h=[list(map(complex_pair,row)) for row in h],generic_d=[list(map(complex_pair,row)) for row in d],generic_b=b.tolist(),generic_c=[list(map(complex_pair,row)) for row in c],
            generic_omega_GeV=.08,first_coefficient_error=float(np.linalg.norm(exact-sum(parts))),
            omitting_generator_endpoint_contact_errors=[float(np.linalg.norm(exact-sum(x for j,x in enumerate(parts) if j!=i))) for i in range(3)],
            weak_order_refinement=errors,
            actual_E_example=dict(plus_GeV=P,photon_plus_GeV=qp,minus_GeV=omega,transverse_GeV=[.1,.2],action_contact_dimensionless=actual.tolist(),
                independent_contact_dimensionless=(4*E*qp**2*omega**2).tolist(),external_kinetic_metric_GeV=2*P,
                canonical_d_GeV_minus3=(4*E*qp**2/(2*P)).tolist(),
                weak_bookkeeper_units='Here epsilon is the product of two source amplitudes, GeV^2; d has GeV^-3, unlike the dimensionless-epsilon generic control.')),
        reassessment=dict(source_free_model_or_fitted_C_D_E_changed=False,retained_modes_quantized_once=True,
            already_computed_quark_loops_requantized=False,complete_LF_time_dependent_or_zero_plus_map=False,
            generated_physical_proton_neutron_or_nucleus=False,production_binding_model_selected=False,public_push=False,
            next='Pre-evaluate nonstationary LF source transport at a declared weak order and positive-plus domain, with explicit endpoints, direct terms and independent action response; then reconnect the preserved graded quark/color and physical network modules. Do not infer zero-plus, all-order source matching, binding or covariance completion from this reference.'))


if __name__=='__main__': print(json.dumps(evaluate(),indent=2,allow_nan=False))

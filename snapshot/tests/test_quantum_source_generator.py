import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.linalg import expm

from deuteron_wigner.quantum.quantum_source_generator import (
    ChargedModes,JointSourceAction,action_mode_lagrangian,pulse,phase_matrix,
    phase_evolution,initial_covariance,charged_excitation,gaussian_state,
    gaussian_wavefunction,gaussian_fock_coefficients,FockSourceControl,quadratic_family,lf_weak_source,
    lf_response_parts,
)
from deuteron_wigner.quantum.joint_source_matching import action_contact,stress_contact


def model(nodes=1):
    if nodes==1:
        return ChargedModes(np.array([[.12]]),JointSourceAction(*[np.array([[x]]) for x in [.4,-.1,.7]]),.23)
    return ChargedModes(np.array([[.12,.02],[.02,.25]]),JointSourceAction(
        np.array([[.4,.05],[.05,.2]]),np.array([[-.1,.03],[.03,.06]]),np.array([[.7,.1],[.1,.4]])),.23)


def normal_transport(system,amplitude):
    w,u=system.normal_basis();t=np.block([[u,u*0],[u*0,u]])
    return w,t.T@phase_evolution(system,8.,amplitude)@t


@pytest.mark.parametrize('nodes',[1,2])
def test_original_density_integral_and_canonical_momentum(nodes):
    s=model(nodes);m=s.dimension;q,v=np.random.default_rng(nodes).normal(0,.3,(2,m))
    a,ad,add=.21,.11,.03;z,b,V,zdot,bdot=s.matrices(a,ad,add)
    assert action_mode_lagrangian(s,q,v,a,ad)==pytest.approx(.5*v@z@v+v@b@q-.5*q@V@q,abs=2e-14)
    vel=torch.tensor(v,requires_grad=True)
    actual=action_mode_lagrangian(s,q,vel,a,ad)
    p=torch.autograd.grad(actual,vel,create_graph=True)[0]
    np.testing.assert_allclose(p.detach(),z@v+b@q,atol=3e-14)
    hess=torch.autograd.functional.hessian(lambda x:action_mode_lagrangian(s,q,x,a,ad),vel)
    np.testing.assert_allclose(hess,z,atol=3e-14)
    phase=np.r_[q,p.detach().numpy()]
    assert .5*phase@s.hessian(a,ad)@phase==pytest.approx(p.detach().numpy()@v-actual.item(),abs=2e-14)
    h=1e-5;plus=s.matrices(a+h*ad,ad+h*add);minus=s.matrices(a-h*ad,ad-h*add)
    np.testing.assert_allclose((plus[0]-minus[0])/(2*h),zdot,atol=1e-11)
    np.testing.assert_allclose((plus[1]-minus[1])/(2*h),bdot,atol=1e-11)


@pytest.mark.parametrize('time',[.4,2.1,5.2,7.7])
def test_actual_source_time_derivatives(time):
    dt=1e-5;a,ad,add=pulse(time,8.,.25)
    p,m=pulse(time+dt,8.,.25),pulse(time-dt,8.,.25)
    assert ad==pytest.approx((p[0]-m[0])/(2*dt),abs=2e-11)
    assert add==pytest.approx((p[1]-m[1])/(2*dt),abs=2e-11)
    assert pulse(0.,8.,.25)==pulse(8.,8.,.25)==(0.,0.,0.)


@pytest.mark.parametrize('nodes',[1,2])
def test_second_order_equations_symplectic_evolution_and_wrong_generators(nodes):
    s=model(nodes);p=phase_evolution(s,8,.25);j=phase_matrix(s.dimension)
    np.testing.assert_allclose(p,phase_evolution(s,8,.25,method='lagrangian'),atol=2e-9)
    np.testing.assert_allclose(p.T@j@p,j,atol=2e-9)
    assert np.linalg.norm(p-phase_evolution(s,8,.25,method='naive'))>1e-3
    assert np.linalg.norm(p-phase_evolution(s,8,.25,method='frozen_source_derivatives'))>.1


def test_legendre_inverse_and_C_squared_contact_are_not_minus_L_int():
    s=model();a,ad=.21,.11;z,b,V,*_=s.matrices(a,ad)
    good,bad=s.hessian(a,ad),s.hessian(a,ad,naive=True);m=s.dimension
    np.testing.assert_allclose(good[:m,:m]-bad[:m,:m],b.T@np.linalg.solve(z,b),atol=1e-15)
    assert np.linalg.norm(good-bad)>1e-4
    np.testing.assert_array_equal(s.hessian(0,0),s.hessian(0,0,naive=True))


@pytest.mark.parametrize('nodes',[1,2])
def test_exact_generator_family_and_noncommuting_node_rotation(nodes):
    s=model(nodes);family=quadratic_family(nodes);gram=np.einsum('aij,bij->ab',family,family)
    np.testing.assert_allclose(gram,np.eye(len(family)),atol=1e-15)
    f=FockSourceControl(s,2);h=s.hessian(.15,.1);normal=f.transform.T@h@f.transform
    np.testing.assert_allclose(np.einsum('a,aij->ij',f.coefficients(.15,.1),family),normal,atol=1e-14)
    r=expm(np.array([[0.,.3],[-.3,0.]])) if nodes==2 else np.eye(1)
    t=np.kron(r,np.eye(4));full=np.block([[t,t*0],[t*0,t]])
    moved=ChargedModes(r@s.mass_squared@r.T,JointSourceAction(*[r@x@r.T for x in [s.source.c,s.source.d,s.source.e]]),s.momentum)
    np.testing.assert_allclose(moved.hessian(.15,.1),full@h@full.T,atol=3e-14)
    if nodes==2: assert np.linalg.norm(s.source.c@s.source.e-s.source.e@s.source.c)>1e-3


def test_free_quantum_phase_and_full_highest_shell_projection():
    s=model();f=FockSourceControl(s,3);w=f.frequencies;h=f.matrix(0,0).toarray()
    energy=np.array([np.dot(w,np.array(n)+.5) for n in f.states])
    np.testing.assert_allclose(h,np.diag(energy),atol=2e-14)
    evolved=f.evolve(8,0);np.testing.assert_allclose(evolved,np.exp(-8j*energy)[:,None]*f.initial,atol=2e-9)
    g=gaussian_state(s,8,0)
    assert g['vacuum_overlap']==pytest.approx(np.exp(-4j*sum(w)),abs=2e-9)
    assert g['one_particle_overlap']==pytest.approx(np.exp(-8j*(sum(w)/2+w[0])),abs=2e-9)
    # Multiplying already projected Q/P loses top-shell terms.
    wrong=sum(.5*(p@p+wi**2*(q@q)) for q,p,wi in zip(f.phase_operators[:4],f.phase_operators[4:],w))
    assert np.linalg.norm(wrong.toarray()-h)>1.


@pytest.mark.parametrize('nodes',[1,2])
def test_quantum_noether_charge_momentum_and_initial_information(nodes):
    s=model(nodes);f=FockSourceControl(s,3)
    np.testing.assert_allclose(f.initial.conj().T@f.charge@f.initial,np.diag([0,1]),atol=2e-14)
    np.testing.assert_allclose(f.initial.conj().T@f.momentum@f.initial,np.diag([0,s.momentum]),atol=2e-14)
    h=f.matrix(.15,.1)
    for conserved in [f.charge,f.momentum]:
        assert np.linalg.norm((h@conserved-conserved@h).data)<1e-12
    np.testing.assert_allclose(f.covariance(f.initial[:,1]),initial_covariance(f.frequencies,charged_excitation(s)),atol=2e-14)


def test_quantum_refinement_full_wavefunction_and_dynamic_pair_response():
    s=model();w,p=normal_transport(s,.15);scale=np.r_[np.sqrt(w),1/np.sqrt(w)]
    g=gaussian_state(s,8,.15);q=np.array([.5,-.3,.7,-.2])/np.sqrt(w)
    cov=p@initial_covariance(w,charged_excitation(s))@p.T;errors=[]
    for shell in [4,8,12,16]:
        f=FockSourceControl(s,shell);v=f.evolve(8,.15)
        overlap=np.diag(f.initial.conj().T@v)
        errors.append([np.max(abs(overlap-[g['vacuum_overlap'],g['one_particle_overlap']])),
            abs(f.wavefunction(v[:,1],q)-gaussian_wavefunction(g,q,one_particle=True)),
            np.linalg.norm(scale[:,None]*(f.covariance(v[:,1])-cov)*scale[None,:])])
    assert np.all(np.diff(np.array(errors),axis=0)<0)
    np.testing.assert_array_less(errors[-1],[1e-8,1e-7,1e-7])
    np.testing.assert_allclose(v.conj().T@v,np.eye(2),atol=2e-9)
    exact=gaussian_fock_coefficients(g,f.states)
    np.testing.assert_allclose(exact[0,0],g['vacuum_overlap'],atol=1e-15)
    assert np.vdot(f.initial[:,1],exact[:,1])==pytest.approx(g['one_particle_overlap'],abs=1e-14)
    assert np.linalg.norm(v-exact)<2e-5
    np.testing.assert_allclose(exact.conj().T@exact,np.eye(2),atol=2e-9)
    assert f.wavefunction(exact[:,1],q)==pytest.approx(gaussian_wavefunction(g,q,one_particle=True),abs=1e-7)
    for j in range(2):
        assert np.vdot(v[:,j],f.charge@v[:,j]).real==pytest.approx(j,abs=2e-9)
        assert np.vdot(v[:,j],f.momentum@v[:,j]).real==pytest.approx(j*s.momentum,abs=2e-9)
        # A spatially uniform source supplies energy, not net charge/momentum.
        assert sum(abs(v[f.total_excitation>j,j])**2)>.05
        assert sum(abs(v[f.total_excitation%2!=j,j])**2)<1e-25


def test_two_node_quantum_response_preserves_mixed_matrices():
    s=model(2);w,p=normal_transport(s,.1);g=gaussian_state(s,8,.1)
    errors=[]
    for shell in [4,6,8]:
        f=FockSourceControl(s,shell);v=f.evolve(8,.1)
        errors.append(np.max(abs(np.diag(f.initial.conj().T@v)-[g['vacuum_overlap'],g['one_particle_overlap']])))
    assert errors[2]<errors[1]<errors[0]
    assert errors[-1]<2e-7
    # Gaussian normalization retains all node phases and source vacuum energy.
    width=g['width'];norm=np.exp(2*g['log_normalization'].real)*np.pi**(len(w)/2)/np.sqrt(np.linalg.det(width.real))
    assert norm==pytest.approx(1,abs=2e-9)
    norm1=norm*np.vdot(g['polynomial'],np.linalg.solve(2*width.real,g['polynomial'])).real
    assert norm1==pytest.approx(1,abs=2e-9)


def test_time_dependent_linked_quantum_source_derivative():
    s=model();f=FockSourceControl(s,10);amplitude=.08;step=1e-4
    quantum=[];independent=[]
    for a in [amplitude-step,amplitude+step]:
        quantum.append(np.vdot(f.initial[:,1],f.evolve(8,a)[:,1]))
        independent.append(gaussian_state(s,8,a)['one_particle_overlap'])
    assert (quantum[1]-quantum[0])/(2*step)==pytest.approx((independent[1]-independent[0])/(2*step),abs=2e-7)
    assert abs((quantum[1]-quantum[0])/(2*step))>.1


def lf_matrices():
    return (np.array([[.2,.02+.03j],[.02-.03j,.7]]),np.array([[.7,.1j],[-.1j,.4]]),
        np.array([[.1,.2],[.2,-.1]]),np.array([[.1,.03j],[-.03j,.1]]))


@pytest.mark.parametrize('omega',[-.3,.08,.31+.12j])
def test_LF_noncommuting_generator_endpoints_and_direct_contact(omega):
    h,d,b,c=lf_matrices();exact,parts=lf_response_parts(h,d,omega,b,c)
    np.testing.assert_allclose(exact,sum(parts),atol=5e-14)
    for omitted in range(3): assert np.linalg.norm(exact-sum(x for i,x in enumerate(parts) if i!=omitted))>1e-2
    shift,endpoint,contact=lf_weak_source(h,d,b,c)
    np.testing.assert_allclose(shift,shift.conj().T,atol=1e-15)
    errors=[]
    for eps in [1e-3,5e-4,2.5e-4]:
        direct=np.linalg.inv(h-omega*np.eye(2)-eps*(d*omega**2+b*omega+c))
        W=np.eye(2)+eps*endpoint
        mapped=W@np.linalg.inv(h+eps*shift-omega*np.eye(2))@W.conj().T+eps*contact
        errors.append(np.linalg.norm(mapped-direct))
    np.testing.assert_allclose(np.array(errors[:-1])/errors[1:],4,rtol=.03)


def test_LF_pole_residue_and_cross_mode_normalization():
    h,d,b,c=lf_matrices();shift,endpoint,_=lf_weak_source(h,d,b,c);errors=[]
    for eps in [1e-3,5e-4,2.5e-4]:
        values,u=np.linalg.eigh(h+eps*shift);psi=(np.eye(2)+eps*endpoint)@u
        inner=np.empty((2,2),complex);residual=[]
        for n in range(2):
            K=h-values[n]*np.eye(2)-eps*(d*values[n]**2+b*values[n]+c)
            residual.append(np.linalg.norm(K@psi[:,n]))
            for m in range(2):
                norm=np.eye(2)+eps*((values[n]+values[m])*d+b)
                inner[n,m]=np.vdot(psi[:,n],norm@psi[:,m])
        errors.append([max(residual),np.linalg.norm(inner-np.eye(2))])
    np.testing.assert_allclose(np.array(errors[:-1])/np.array(errors[1:]),4,rtol=.01)


@pytest.mark.parametrize('omega',[.07,.2,.33])
def test_actual_E_action_contact_requires_LF_quadratic_energy_map(omega):
    s=model(2);P=.8;photon_plus=.3;eps=np.array([0.,1,0,0]);e=s.source.e;source=JointSourceAction(e*0,e*0,e)
    p=np.array([(P+omega)/np.sqrt(2),.1,.2,(P-omega)/np.sqrt(2)])
    q=np.array([photon_plus/np.sqrt(2),0,0,photon_plus/np.sqrt(2)])
    for i in range(2):
        for j in range(2):
            actual=action_contact(source,p,p,q,-q,eps,eps,np.eye(2)[i],np.eye(2)[j])
            reduced=stress_contact(p,p,q,-q,eps,eps,e[i,j])
            assert actual==pytest.approx(4*e[i,j]*photon_plus**2*omega**2,abs=2e-14)
            assert actual==pytest.approx(reduced,abs=2e-14)
    h=(s.mass_squared+.05*np.eye(2))/(2*P);d=4*e*photon_plus**2/(2*P)
    direct,parts=lf_response_parts(h,d,omega)
    np.testing.assert_allclose(direct,sum(parts),atol=1e-12)


def test_invalid_generator_domains_do_not_silently_change_physics():
    s=model()
    with pytest.raises(ValueError): ChargedModes(s.mass_squared,s.source,0.)
    with pytest.raises(ValueError): FockSourceControl(s,0)
    with pytest.raises(ValueError): FockSourceControl(s,1.5)
    with pytest.raises(ValueError): pulse(1,0,.2)
    with pytest.raises(ValueError): lf_weak_source(np.eye(2),np.array([[1.,2.],[0.,1.]]))
    bad=ChargedModes(s.mass_squared,JointSourceAction(s.source.c,s.source.d,-s.source.e),.23)
    with pytest.raises(ValueError): bad.hessian(0.,2.)


def test_saved_quantum_information_and_LF_claim_boundaries():
    root=Path(__file__).resolve().parents[1]
    record=json.loads((root/'docs/next_level/quantum_source_generator_results.json').read_text())
    assert [x['real_oscillators'] for x in record['quantum_controls']]==[4,8]
    charged=record['quantum_controls'][1]['refinements'][-1]
    assert charged['projected_full_amplitude_vector_errors'][1]>100*charged['survival_errors'][1]
    assert charged['infinite_Gaussian_outside_shell_norm_deficits'][1]>1e-6
    assert record['reassessment']['retained_modes_quantized_once']
    assert not record['reassessment']['complete_LF_time_dependent_or_zero_plus_map']
    assert not record['reassessment']['generated_physical_proton_neutron_or_nucleus']
    assert not record['reassessment']['already_computed_quark_loops_requantized']
    assert not record['reassessment']['production_binding_model_selected']
    assert not record['reassessment']['public_push']

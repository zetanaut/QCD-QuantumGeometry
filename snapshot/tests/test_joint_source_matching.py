import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.linalg import expm

from deuteron_wigner.quantum.joint_source_matching import (
    JointSourceAction,stress_density,stress_contact,photon_contact,
    projected_observables,match_projected_observables,compass_dipole_response,
    conditional_projection,match_conditional_observables,action_contact,dot,METRIC,ALPHA_EM,HBARC,
)
from deuteron_wigner.quantum.transverse_source_matching import source_density,action_vertex,curvature
from deuteron_wigner.quantum.quantum_collective_compton import compton_kinematics

MODE=np.array([.8,.6])
MP=.13957039


def model():
    return JointSourceAction(np.array([[.7,.1],[.1,.4]]),np.array([[-.2,.07],[.07,.3]]),np.array([[.5,.15],[.15,.8]]))


def jets(seed,n=2):
    r=np.random.default_rng(seed)
    f=r.normal(0,.2,(n,4));df=r.normal(0,.3,(4,n,4))
    a=r.normal(0,.1,(4,4,4));a-=a.transpose(0,2,1)
    da=r.normal(0,.2,(4,4,4,4));da-=da.transpose(0,1,3,2)
    return f,df,a,da


def pure_stress():
    zero=np.zeros((2,2));return JointSourceAction(zero,zero,model().e)


@pytest.mark.parametrize('seed',[13,27,49])
def test_full_local_nonAbelian_covariance_and_missing_connection_negative(seed):
    f,df,a,da=jets(seed);rng=np.random.default_rng(seed+1)
    z=rng.normal(size=(4,4));r=expm(.2*(z-z.T))
    omega=rng.normal(0,.1,(4,4,4));omega-=omega.transpose(0,2,1)
    b=r@a@r.T;ma=b-omega;mda=np.empty_like(da)
    for mu in range(4):
        for nu in range(4):
            mda[mu,nu]=r@da[mu,nu]@r.T+omega[mu]@b[nu]-b[nu]@omega[mu]-.5*(omega[mu]@omega[nu]-omega[nu]@omega[mu])
    mf=f@r.T;mdf=np.einsum('ab,mib->mia',r,df)+np.einsum('mab,ib->mia',omega,mf)
    np.testing.assert_allclose(stress_density(mf,mdf,ma,mda,model().e),stress_density(f,df,a,da,model().e),atol=3e-14)
    np.testing.assert_allclose(curvature(ma,mda),r@curvature(a,da).numpy()@r.T,atol=3e-14)
    # Rotating fields but deleting the inhomogeneous connection on their
    # derivatives cannot satisfy local symmetry even with correct F.
    wrong=np.einsum('ab,mib->mia',r,df)
    assert abs(stress_density(mf,wrong,ma,mda,model().e)-stress_density(f,df,a,da,model().e))>1e-6


@pytest.mark.parametrize('beta',[[.3,-.2,.1],[-.1,.2,.3]])
def test_action_and_reduced_contact_Lorentz_covariance(beta):
    gen=np.zeros((4,4));gen[0,1:]=gen[1:,0]=beta;upper=expm(gen);lower=METRIC[:,None]*upper*METRIC[None,:]
    f,df,a,da=jets(31)
    moved=(f,np.einsum('mn,nia->mia',lower,df),np.einsum('mn,nab->mab',lower,a),np.einsum('mr,ns,rsab->mnab',lower,lower,da))
    np.testing.assert_allclose(model().density(*moved),model().density(f,df,a,da),atol=3e-14)
    p1,p0,k,kp=compton_kinematics(.02,1.1,MP);e1=np.array([0.,1,0,0]);e2=np.array([0.,np.cos(1.1),0.,-np.sin(1.1)])
    source=pure_stress();actual=action_contact(source,p1,p0,k,-kp,e1,e2,MODE,MODE)
    transformed=action_contact(source,upper@p1,upper@p0,upper@k,-upper@kp,upper@e1,upper@e2,MODE,MODE)
    assert actual==pytest.approx(transformed,abs=2e-14)


@pytest.mark.parametrize('seed',[12,34,56])
def test_two_source_contact_is_independent_fourth_action_derivative(seed):
    p0,q1,q2,e1,e2=np.random.default_rng(seed).normal(0,.15,(5,4));p1=p0+q1+q2
    source=pure_stress();e=source.projected(MODE)[2]
    actual=action_contact(source,p1,p0,q1,q2,e1,e2,MODE,MODE)
    reduced=stress_contact(p1,p0,q1,q2,e1,e2,e)
    assert actual==pytest.approx(reduced,abs=2e-14)
    assert photon_contact(source,p1,p0,[q1,q2],[e1,e2],MODE,MODE)==pytest.approx(actual,abs=2e-14)
    assert abs(action_contact(source,p1,p0,q1,q2,q1,e2,MODE,MODE))<2e-14
    assert abs(action_contact(source,p1,p0,q1,q2,e1,q2,MODE,MODE))<2e-14
    assert action_contact(source,p1,p0,q2,q1,e2,e1,MODE,MODE)==pytest.approx(actual,abs=2e-14)


def test_three_and_four_photon_Ward_require_actual_covariant_derivative_seagulls():
    source=pure_stress();p0=np.array([.2,.02,.04,.03])
    q1=np.array([.04,.02,.05,-.01]);q2=np.array([.03,.06,-.02,.01]);q3=np.array([.02,-.01,.04,.03]);q4=np.array([.02,.04,-.03,.01])
    e1=np.array([.1,.3,-.2,.4]);e2=np.array([.15,-.1,.4,.2]);e3=np.array([.2,.1,.05,-.2])
    p1=p0+q1+q2+q3
    three=photon_contact(source,p1,p0,[q1,q2,q3],[e1,e2,q3],MODE,MODE)
    difference=action_contact(source,p1,p0+q3,q1,q2,e1,e2,MODE,MODE)-action_contact(source,p1-q3,p0,q1,q2,e1,e2,MODE,MODE)
    assert three==pytest.approx(difference,abs=2e-14)
    assert abs(three)>1e-6
    p1+=q4
    four=photon_contact(source,p1,p0,[q1,q2,q3,q4],[e1,e2,e3,q4],MODE,MODE)
    difference=photon_contact(source,p1,p0+q4,[q1,q2,q3],[e1,e2,e3],MODE,MODE)-photon_contact(source,p1-q4,p0,[q1,q2,q3],[e1,e2,e3],MODE,MODE)
    assert four==pytest.approx(difference,abs=2e-14)
    assert abs(four)>1e-7


def test_zero_source_dynamics_and_one_photon_current_are_not_changed_by_E():
    f,df,_,_=jets(32);a=np.zeros((4,4,4));da=np.zeros((4,4,4,4))
    fields=torch.tensor(f,requires_grad=True)
    density=stress_density(fields,df,a,da,model().e).real
    assert density==0
    np.testing.assert_array_equal(torch.autograd.grad(density,fields)[0],0)
    p0=np.array([.2,.01,.02,.03]);p1=np.array([.3,-.02,.04,.01])
    np.testing.assert_allclose(action_vertex(pure_stress(),p1,p0,MODE,MODE),0,atol=1e-15)


@pytest.mark.parametrize('angle',[0.,.8,np.pi])
def test_electric_magnetic_low_energy_map_from_action_not_observable_heads(angle):
    c,d,e=model().projected(MODE);mass=MP;values=projected_observables(c,d,e,mass)
    energy=1e-5;p1,p0,k,kp=compton_kinematics(energy,angle,mass);eps=np.array([0.,0,1,0])
    actual=action_contact(model(),p1,p0,k,-kp,eps,eps,MODE,MODE).real
    inferred=ALPHA_EM*HBARC**3/(2*mass)*actual/(k[0]*kp[0])
    assert inferred==pytest.approx(values[1]+np.cos(angle)*values[2],abs=1e-12)
    stress=projected_observables(0,0,e,mass)
    assert stress[2]==0
    assert stress[1]==pytest.approx(2*ALPHA_EM*HBARC**3*mass*e)


@pytest.mark.parametrize('n',[1,2,3])
def test_general_node_space_and_all_matrix_gradients(n):
    rng=np.random.default_rng(n);f,df,a,da=jets(45,n)
    # Use an orthogonal node-coordinate transformation, independent of O(4).
    z=rng.normal(size=(n,n));r=expm(.2*(z-z.T))
    matrices=[]
    for _ in range(3):
        x=rng.normal(0,.2,(n,n));matrices.append((x+x.T)/2)
    full=JointSourceAction(*matrices);moved=JointSourceAction(*(r@x@r.T for x in matrices))
    np.testing.assert_allclose(full.density(f,df,a,da),moved.density(r@f,np.einsum('ij,mja->mia',r,df),a,da),atol=3e-14)
    e=torch.tensor(matrices[2],requires_grad=True);value=stress_density(f,df,a,da,e).real
    gradient=torch.autograd.grad(value,e)[0].numpy()
    for i in range(n):
        for j in range(n):
            unit=np.zeros((n,n));unit[i,j]=1.
            assert gradient[i,j]==pytest.approx(stress_density(f,df,a,da,unit).real.item(),abs=1e-14)


def test_subtraction_and_parameterized_joint_inverse_preserve_the_loop_once():
    full=model();kept=JointSourceAction(.2*full.c,.3*full.d,.4*full.e);rest=full.residual(kept)
    f,df,a,da=jets(55)
    assert full.density(f,df,a,da)==pytest.approx(kept.density(f,df,a,da)+rest.density(f,df,a,da),abs=2e-14)
    loop=np.array([.3556496377716894,.0005816340049767354,-.0005576610646489389])
    parameters=np.array([.4,-.12,.8]);target=projected_observables(*parameters,MP,loop=loop)
    np.testing.assert_allclose(match_projected_observables(target,MP,loop=loop),parameters,atol=3e-15)
    duplicated=projected_observables(*match_projected_observables(target,MP),MP,loop=loop)
    np.testing.assert_allclose(duplicated-target,loop,atol=3e-15)


def test_three_projected_observables_leave_six_open_matrix_directions():
    # Nine independent symmetric C,D,E entries, only one physical mode.
    weights=np.array([MODE[0]**2,2*MODE[0]*MODE[1],MODE[1]**2]);k=2*ALPHA_EM*HBARC**3/MP
    response=np.array([np.r_[6*HBARC**2*weights,0*weights,0*weights],np.r_[k*weights,2*k*weights,k*MP*MP*weights],np.r_[-k*weights,-2*k*weights,0*weights]])
    assert np.linalg.matrix_rank(response)==3
    # Dimensionless row normalization, not an unphysical precision demand
    # on observables with different units or a claimed inference metric.
    scaled=response/np.linalg.norm(response,axis=1)[:,None]
    _,_,vh=np.linalg.svd(scaled,full_matrices=True)
    np.testing.assert_allclose(scaled@vh[3:].T,0,atol=2e-14)
    h=np.array([-MODE[1],MODE[0]]);null=np.outer(MODE,h)+np.outer(h,MODE)
    changed=JointSourceAction(model().c,model().d,model().e+.6*null)
    np.testing.assert_allclose(changed.projected(MODE),model().projected(MODE),atol=2e-15)
    p1,p0,k,kp=compton_kinematics(.02,.9,MP);eps=np.array([0.,0,1,0])
    assert abs(action_contact(changed,p1,p0,k,-kp,eps,eps,h,MODE)-action_contact(model(),p1,p0,k,-kp,eps,eps,h,MODE))>1e-6


def test_primary_cross_section_two_distinct_templates_and_conditional_limit():
    mass=MP;s=4*mass*mass
    backward=compass_dipole_response(mass,s,-1.);forward=compass_dipole_response(mass,s,1.)
    assert backward[0]<0 and backward[1]==0
    assert forward[0]==0 and forward[1]<0
    z=-.5;response=compass_dipole_response(mass,s,z)
    assert response[1]/response[0]==pytest.approx(s*s/mass**4*((1+z)/(1-z))**2)
    alpha=.0002/HBARC**3
    assert response@np.array([2*alpha,0.])==pytest.approx(response[0]*2*alpha)


def test_conditional_summary_projection_profiles_normalization_and_loses_information():
    r=np.array([[-1.,-.1],[-2.,-.4],[-3.,-1.4],[-4.,-3.5]])
    v=np.diag([.4,.5,.6,.7])**2;b=np.ones((4,1))
    result=conditional_projection(r,v,nuisance=b)
    true=np.array([.8,.2]);data=r@true+.3
    design=np.column_stack([2*r[:,0],b]);w=np.linalg.inv(v)
    fit=np.linalg.solve(design.T@w@design,design.T@w@data)
    assert fit[0]==pytest.approx(.5*true[0]+result['sum_weight']*true[1])
    assert abs(result['sum_weight']-conditional_projection(r,v)['sum_weight'])>.1
    # Scaling measurement units must not change the mixing coefficient.
    scaled=conditional_projection(1e-12*r,1e-24*v,nuisance=1e-12*b)
    assert scaled['sum_weight']==pytest.approx(result['sum_weight'])
    assert scaled['conditional_variance']==pytest.approx(result['conditional_variance'])
    assert np.linalg.matrix_rank(result['information_matrix'])==2
    # One scalar conditional result leaves a line of possible (diff,sum).
    same=np.array([true[0]-2*result['sum_weight']*.3,true[1]+.3])
    assert .5*same[0]+result['sum_weight']*same[1]==pytest.approx(fit[0])


def test_input_guards_and_unidentified_conditional_fit():
    with pytest.raises(ValueError): JointSourceAction(np.eye(2),np.eye(3),np.eye(2))
    with pytest.raises(ValueError): JointSourceAction(np.eye(2),np.eye(2),np.array([[1.,.1],[0.,1.]]))
    with pytest.raises(ValueError): match_projected_observables([1.,2.],MP)
    with pytest.raises(ValueError): projected_observables(0,0,0,0.)
    with pytest.raises(ValueError): compass_dipole_response(MP,MP*MP,-1.)
    with pytest.raises(ValueError): conditional_projection(np.ones((3,2)),np.eye(3),nuisance=np.ones((3,1)))
    with pytest.raises(ValueError): conditional_projection(np.ones((3,2)),-np.eye(3))
    with pytest.raises(ValueError): match_conditional_observables(.4,.0002,MP,sum_weight=np.nan,e=1.)


@pytest.mark.parametrize('kappa',[-.5,0.,.8])
def test_conditional_matching_retains_unmeasured_source_direction(kappa):
    # Synthetic observations, known loop counted once. Even known kappa and
    # a radius cannot identify E or turn a conditional alpha into true alpha.
    loop=np.array([.3556496377716894,.0005816340049767354,-.0005576610646489389])
    radius=.45;conditional=.00023;predictions=[]
    for e in [-1.2,0.,2.4]:
        c,d,kept_e=match_conditional_observables(radius,conditional,MP,sum_weight=kappa,e=e,loop=loop)
        assert kept_e==e
        values=projected_observables(c,d,e,MP,loop=loop);predictions.append(values)
        assert values[0]==pytest.approx(radius,abs=2e-15)
        assert .5*(values[1]-values[2])+kappa*(values[1]+values[2])==pytest.approx(conditional,abs=2e-17)
    predictions=np.array(predictions)
    assert np.ptp(predictions[:,1]+predictions[:,2])>1e-5
    # Independent of kappa, the unconstrained physical sum is not erased.
    k=2*ALPHA_EM*HBARC**3/MP
    response=np.array([[6*HBARC**2,0.,0.],[k,2*k,k*MP*MP*(.5+kappa)]])
    assert np.linalg.matrix_rank(response)==2
    null=np.array([0.,-.5*MP*MP*(.5+kappa),1.])
    np.testing.assert_allclose(response@null,0,atol=2e-17)


def test_saved_scientific_replay():
    from scripts.evaluate_joint_source_matching import evaluate
    saved=json.loads(Path('docs/next_level/joint_source_matching_results.json').read_text())
    assert evaluate()==saved

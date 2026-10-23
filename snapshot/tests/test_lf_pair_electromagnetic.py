import numpy as np
import pytest
import torch

from deuteron_wigner.quantum.finite_lf_coupling import collective_pair_leaf,reference_inputs
from deuteron_wigner.quantum.lf_pair_electromagnetic import (
    good_yukawa_pair,fermion_source,full_spinor_source,mixed_fermion_pair,
    covariant_instant_pair,tree_response,phi_source_coefficients,parent_h,
    PulseTransition,pulse_parameter_features,torch_pulse_response,
    cart,
    PHI_CHARGE,CHARGE,BILINEARS,
)


def inputs(charge=1):
    cell,m,g,_=reference_inputs()
    _,u=np.linalg.eigh(m[1])
    seed=np.zeros((2,4),complex)
    if charge:
        seed[:,1]=u[:,0]/np.sqrt(2);seed[:,2]=1j*charge*seed[:,1]
    else:
        _,u=np.linalg.eigh(m[0]);seed[:,0]=u[:,0]
    return cell,m,g,seed,np.array([[.14,.037],[.037,-.06]])


@pytest.mark.parametrize('pt',[(.2,-.2),(.31,.14),(-.4,.1)])
def test_actual_good_field_yukawa_constraint(pt):
    a=np.array([.125,*pt]);b=np.array([.375,-pt[0],-pt[1]])
    np.testing.assert_allclose(good_yukawa_pair(a,b,a+b,.27),
                               collective_pair_leaf(a,b,a+b,.27),atol=4e-15)


@pytest.mark.parametrize('anti',[False,True])
def test_four_free_current_components_from_constraint(anti):
    a=np.array([.125,.4,.2]);b=np.array([.125,.2,.4])
    for eps in np.eye(4):
        np.testing.assert_allclose(fermion_source(a,b,.27,eps,anti=anti),
                                   full_spinor_source(a,b,.27,eps,anti=anti),atol=3e-15)


@pytest.mark.parametrize('mu',range(4))
def test_mixed_instantaneous_constraint_vs_covariant_gamma_plus(mu):
    cell,*_=reference_inputs();a=np.array([.125,.4,.2]);b=np.array([.375,-.2,-.4]);p=np.array([.5,0,0])
    actual=mixed_fermion_pair(a,b,p,.27,np.eye(4)[mu],cell.volume)
    independent=covariant_instant_pair(a,b,p,.27,np.eye(4)[mu],cell.volume)
    np.testing.assert_allclose(actual,independent,atol=4e-17)
    if mu==1:
        assert np.linalg.norm(actual)<1e-16  # no mixed J+ term
    else:
        assert np.linalg.norm(actual)>.01


@pytest.mark.parametrize('charge',[-1,0,1])
def test_all_components_frequency_tree_and_ward(charge):
    cell,m,g,seed,c=inputs(charge)
    a=np.array([.125,.4,.2]);b=np.array([.375,-.2,-.4]);p=np.array([.5,0,0])
    for omega in (.11,.17,.27):
        currents=[]
        for mu in range(4):
            result=tree_response(a,b,p,.27,m,c,g,seed,np.eye(4)[mu],omega,cell.volume)
            np.testing.assert_allclose(result['covariant'],result['light_front'],rtol=3e-12,atol=3e-15)
            currents.append(result['light_front'])
        # Stored columns contract the UPPER source A, so transverse columns
        # already equal -J^T. Substituting A=q requires no second metric sign.
        q=a+b-p;ward=omega*currents[1]+q[1]*currents[2]+q[2]*currents[3]
        assert np.linalg.norm(ward)<2e-14


def test_omitted_instantaneous_and_scalar_endpoint_are_detected():
    cell,m,g,seed,c=inputs()
    a=np.array([.125,.4,.2]);b=np.array([.375,-.2,-.4]);p=np.array([.5,0,0])
    result=tree_response(a,b,p,.27,m,c,g,seed,[1,0,0,0],.17,cell.volume)
    assert np.linalg.norm(result['fermion_contact'])>.005
    assert np.linalg.norm(result['collective_endpoint'])>.005
    # Each term is tested separately: their sum can cancel in special channels.
    for key in ('quark_after','anti_after','collective_before','fermion_contact','collective_endpoint'):
        assert np.linalg.norm(result['light_front']-result[key]-result['covariant'])>1e-4


def test_transverse_C_not_determined_by_ward():
    cell,m,g,seed,c=inputs()
    a=np.array([.125,.4,.2]);b=np.array([.375,-.2,-.4]);p=np.array([.5,0,0])
    changes=[]
    for eps in np.eye(4):
        full=tree_response(a,b,p,.27,m,c,g,seed,eps,.17,cell.volume)['light_front']
        minimal=tree_response(a,b,p,.27,m,c*0,g,seed,eps,.17,cell.volume)['light_front']
        changes.append(full-minimal)
    assert np.linalg.norm(changes)>.001
    np.testing.assert_allclose(.17*changes[1]+.2*changes[2]-.2*changes[3],0,atol=1e-16)


def test_source_polynomial_hermiticity_with_reverse_frequency():
    cell,m,g,seed,c=inputs()
    a=np.array([.5,.2,-.2]);b=np.array([.5,0,0]);omega=.17
    eps=np.array([.3,.4,-.2,.1j])
    left=phi_source_coefficients(m,c,a,b,eps)
    right=phi_source_coefficients(m,c,b,a,eps.conj())
    np.testing.assert_allclose(left[0]+omega*left[1]+omega**2*left[2],
                               (right[0]-omega*right[1]+omega**2*right[2]).conj().T,atol=2e-16)
    np.testing.assert_allclose(left[3],right[3].conj().T,atol=2e-16)


def pulse_inputs(charge):
    cell,m,g,v=reference_inputs()
    if charge:
        v=np.zeros((2,4),complex);v[:,1]=[1,.3j];v[:,2]=1j*charge*v[:,1];v/=np.linalg.norm(v)
    return cell,m,g,v,np.array([[.14,.037],[.037,-.06]])


@pytest.mark.parametrize('charge',[-1,0,1])
def test_full_time_wave_independent_dirac_ode_and_charge(charge):
    cell,m,g,v,c=pulse_inputs(charge)
    args=(cell,m,c,g,.27,[.5,.2,0],[0,-.2,.2],1,12.,[.4,.2,.3,-.1])
    model=PulseTransition(*args)
    exact=model.response(v)
    independent=PulseTransition(*args,representation='dirac')
    numerical,_=independent.ode_response(v)
    np.testing.assert_allclose(exact,numerical,rtol=3e-8,atol=3e-10)
    np.testing.assert_allclose(exact,model.response(v,nodes=96),atol=3e-15)
    q=np.tile(np.repeat([2/3,-1/3],3),2)
    qpair=(q[:,None]-q[None,:]).reshape(144)
    assert np.linalg.norm((qpair-charge)*exact)<1e-15
    assert exact.shape==(12,144)
    assert model.outside_intermediates()==8


@pytest.mark.parametrize('charge',[-1,0,1])
def test_pure_gauge_endpoint_zero_whole_wave(charge):
    cell,m,g,v,c=pulse_inputs(charge)
    for sign in (-1,1):
        q=sign*np.array([0,-.2,.2])
        model=PulseTransition(cell,m,c,g,.27,[.5,.2,0],q,1,12.,[.5,0,0,0],pure_gauge=True)
        assert np.linalg.norm(model.response(v))<2e-15
        assert np.linalg.norm(model.response(v,omit='fermion_contact'))>1e-4
    assert np.linalg.norm(model.response(v,omit='outside_intermediates'))>1e-4


def test_contacts_orders_and_derivatives_have_independent_information():
    cell,m,g,v,c=pulse_inputs(1)
    model=PulseTransition(cell,m,c,g,.27,[.5,.2,0],[0,-.2,.2],1,12.,[1,0,0,0])
    full=model.response(v)
    for omit in ('quark','anti','collective','fermion_contact','collective_endpoint',
                 'source_derivatives','outside_intermediates'):
        assert np.linalg.norm(full-model.response(v,omit=omit))>1e-4


def test_shared_C_and_yukawa_torch_wave_derivatives():
    cell,m,g,v,c=pulse_inputs(1)
    fixed=(cell,m,.27,[.5,.2,0],[0,-.2,.2],1,12.,[.5,.2,-.1,.3],v)
    features=pulse_parameter_features(*fixed,nodes=48)
    x=torch.tensor([*g,c[0,0],c[0,1],c[1,1]],dtype=torch.float64,requires_grad=True)
    wave=torch_pulse_response(features,x[:2],x[2:])
    direct=PulseTransition(cell,m,c,g,.27,fixed[3],fixed[4],1,12.,fixed[7]).response(v)
    np.testing.assert_allclose(wave.detach(),direct,atol=3e-15)
    probe=np.random.default_rng(37).normal(size=direct.shape)+1j*np.random.default_rng(38).normal(size=direct.shape)
    probe/=np.linalg.norm(probe);probe=torch.tensor(probe)
    objective=torch.real(torch.sum(probe.conj()*wave))
    gradient=torch.autograd.grad(objective,x)[0].numpy()
    def value(par):
        cc=np.array([[par[2],par[3]],[par[3],par[4]]])
        result=PulseTransition(cell,m,cc,par[:2],.27,fixed[3],fixed[4],1,12.,fixed[7]).response(v,nodes=48)
        return np.vdot(probe.numpy(),result).real
    base=x.detach().numpy();step=2e-4;fd=[]
    for i in range(5):
        delta=np.eye(5)[i]*step
        fd.append((value(base+delta)-value(base-delta))/(2*step))
    np.testing.assert_allclose(gradient,fd,rtol=1e-7,atol=2e-11)
    assert np.min(abs(gradient))>1e-6


def test_internal_frequency_pole_and_nonzero_plus_domains_not_silenced():
    cell,m,g,seed,c=inputs()
    with pytest.raises(ValueError,match=r'q\+=0'):
        mixed_fermion_pair([.125,.4,.2],[.375,-.2,-.4],[.75,0,0],.27,[1,0,0,0],cell.volume)
    with pytest.raises(ValueError,match='Collective pole'):
        tree_response([.125,.4,.2],[.375,-.2,-.4],[.5,0,0],.27,m,c,g,seed,
                      [1,0,0,0],.08,cell.volume)


def test_tree_components_under_kinematic_LF_boosts():
    cell,m,g,seed,c=inputs()
    a=np.array([.125,.4,.2]);b=np.array([.375,-.2,-.4]);p=np.array([.5,0,0])
    eps=np.array([.3,.2,-.1,.4]);omega=.17
    base=tree_response(a,b,p,.27,m,c,g,seed,eps,omega,cell.volume)['light_front']
    scale=1.7;boost=np.array([scale,1,1]);e=eps*np.array([scale,1/scale,1,1])
    moved=tree_response(a*boost,b*boost,p*boost,.27,m,c,g,seed,e,omega/scale,cell.volume/scale)['light_front']
    np.testing.assert_allclose(moved,base/scale,atol=2e-16)
    velocity=np.array([.19,-.27]);q=a+b-p
    def transverse(x): return x+np.array([0,*(x[0]*velocity)])
    e=eps.copy();e[2:]+=velocity*eps[0];e[1]+=velocity@eps[2:]+(velocity@velocity)*eps[0]/2
    moved=tree_response(transverse(a),transverse(b),transverse(p),.27,m,c,g,seed,e,
                        omega+velocity@q[1:],cell.volume)['light_front']
    # Unit determinant boost of the cell/measure, not a fixed rectangular cutoff.
    np.testing.assert_allclose(moved,base,atol=3e-16)


def test_complete_pulse_wave_longitudinal_boost():
    from deuteron_wigner.quantum.finite_lf_coupling import LFCell
    cell,m,g,v,c=pulse_inputs(1)
    eps=np.array([.4,.2,.3,-.1]);scale=1.7
    model=PulseTransition(cell,m,c,g,.27,[.5,.2,0],[0,-.2,.2],1,12.,eps)
    bc=LFCell(tuple(np.asarray(cell.lengths)/[scale,1,1]))
    moved=PulseTransition(bc,m,c,g,.27,[.5*scale,.2,0],[0,-.2,.2],1,12.*scale,
                          eps*np.array([scale,1/scale,1,1]))
    np.testing.assert_allclose(model.response(v),moved.response(v),atol=4e-15)


def test_scalar_source_and_endpoint_from_actual_C_action():
    from deuteron_wigner.quantum.joint_source_matching import JointSourceAction
    from deuteron_wigner.quantum.transverse_source_matching import action_vertex,dot
    cell,m,g,v,c=pulse_inputs(1)
    p0=np.array([.5,.2,0]);p1=np.array([.5,0,.2]);omega=.17
    eps=np.array([.4,.2,.3,-.1]);action=JointSourceAction(c,np.zeros((2,2)),np.zeros((2,2)))
    def density(w):
        left=cart([p1[0],w+omega,*p1[1:]]);right=cart([p0[0],w,*p0[1:]])
        matrix=np.empty((2,2),complex)
        for i in range(2):
            for j in range(2):
                current=action_vertex(action,left,right,np.eye(2)[i],np.eye(2)[j])
                matrix[i,j]=(dot(current,cart(eps))+(i==j)*dot(left+right,cart(eps)))/(2*p0[0])
        return matrix
    d=(density(.2)-density(-.2))/.4
    hin=(m[1]+p0[1:]@p0[1:]*np.eye(2))/(2*p0[0])
    hout=(m[1]+p1[1:]@p1[1:]*np.eye(2))/(2*p1[0])
    canonical=density(0)+(hout@d+d@hin)/2-omega*d/2
    j0,j1,j2,w=phi_source_coefficients(m,c,p1,p0,eps)
    charged=np.kron(np.eye(2),np.array([[0],[1],[1j],[0]])/np.sqrt(2))
    np.testing.assert_allclose(charged.conj().T@(j0+omega*j1+omega**2*j2)@charged,canonical,atol=2e-15)
    np.testing.assert_allclose(charged.conj().T@w@charged,d/2,atol=2e-15)


def test_common_EM_sign_map_and_local_yukawa_gauge_invariance():
    from scipy.linalg import expm
    from deuteron_wigner.quantum.transverse_source_matching import connection
    # Preserved external source v=-e Q A, not an independently chosen pion sign.
    lower=np.array([.3,-.2,.4,.1]);vector=np.zeros((4,3));vector[:,2]=-lower
    actual=connection(vector,np.zeros((4,3))).numpy()
    np.testing.assert_allclose(actual,1j*lower[:,None,None]*PHI_CHARGE,atol=1e-16)
    theta=.37;uq=expm(-1j*theta*CHARGE);up=expm(-1j*theta*PHI_CHARGE)
    fields=np.array([.3,.2,-.4,.7])
    original=np.einsum('A,Aij->ij',fields,BILINEARS)
    changed=np.einsum('A,Aij->ij',up@fields,BILINEARS)
    np.testing.assert_allclose(uq.conj().T@changed@uq,original,atol=3e-16)

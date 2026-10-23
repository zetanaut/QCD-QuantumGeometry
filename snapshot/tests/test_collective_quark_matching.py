from dataclasses import replace
import numpy as np
import pytest
import torch

from deuteron_wigner.quantum.common_action_pilot import calibrated_action
from deuteron_wigner.quantum.canonical_lf_fields import GAMMA
from deuteron_wigner.quantum.transverse_source_matching import SourceCurvature, match_projected, dot
from deuteron_wigner.quantum.collective_quark_matching import (
    bilinear_matrices, yukawa_map, exchange_weights, open_quark_tensor,
    one_source_matrix, routed_jet, response_derivatives, matched_response,
    torch_response, potential_derivatives, induced_many_quark_tensors,
    tree_series, stationary_tree, nonlinear_bilinear_response,
)


@pytest.fixture(scope='module')
def case():
    base=calibrated_action(.13957039,.0922,.515,.004,.6,1.4,.12,[3.,1.])
    s=base.solve(.004);e=s['pion_modes'][:,0]
    c,d=match_projected(.659,2e-4,.13957039)
    model=SourceCurvature(c*np.outer(e,e),d*np.outer(e,e))
    p0=np.array([.07,.3,.12,.11]);q1=np.array([.01,.04,-.01,.02]);q2=np.array([.02,-.03,.02,-.04])
    ea=np.array([.1,.3,-.2,.4]);eb=np.array([.2,-.1,.4,.3])
    hp=base.matrices(s['vacuum'])[0]
    momenta,jet=routed_jet(hp,model,p0,q1,q2,ea,eb)
    y=np.kron(np.eye(4),base.yukawa[:,None])
    return base,s,model,(p0,q1,q2,ea,eb),momenta,jet,y


def test_resolved_bilinears_are_actual_mass_derivatives_with_Dirac_adjoint(case):
    base=case[0];b=bilinear_matrices();zero=np.zeros((2,4))
    gamma0=np.kron(GAMMA[0],np.eye(6))
    for i in range(2):
        for a in range(4):
            f=zero.copy();f[i,a]=1
            np.testing.assert_allclose(base.quark_mass_matrix(f,0),base.yukawa[i]*b[a])
    for matrix in b:
        np.testing.assert_allclose(gamma0@matrix.conj().T@gamma0,matrix)
    np.testing.assert_allclose(np.einsum('Aab,Bab->AB',b.conj(),b),24*np.eye(4))


@pytest.mark.parametrize('q2',[0.,.03,.2])
def test_open_quark_tensor_preserves_all_S_P_indices_and_contractions(case,q2):
    base,s=case[:2];w=exchange_weights(base,s['vacuum'],q2);b=bilinear_matrices()
    tensor=open_quark_tensor(w)
    rng=np.random.default_rng(71);left=rng.normal(size=(24,24))+1j*rng.normal(size=(24,24))
    right=rng.normal(size=(24,24))+1j*rng.normal(size=(24,24))
    actual=np.einsum('abcd,ab,cd',tensor,left,right)
    independent=sum(w[a]*np.sum(b[a]*left)*np.sum(b[a]*right) for a in range(4))
    np.testing.assert_allclose(actual,independent,rtol=2e-14)
    assert abs(actual-w[0]*np.trace(left)*np.trace(right))>1
    np.testing.assert_allclose(w[1:],w[1])


def test_Gaussian_elimination_is_normalized_source_functional_not_new_dynamics(case):
    *_,jet,y=case;rng=np.random.default_rng(52);b=rng.normal(0,.002,4)
    for a,c in [(0.,0.),(.001,-.002)]:
        k,ka,kb,kab=jet;full=k+a*ka+c*kb+a*c*kab
        x=np.linalg.solve(full,-y@b)
        actual=.5*x@full@x+b@y.T@x
        expected=-.5*b@torch_response(jet,y,torch.tensor([a,c],dtype=torch.float64)).numpy()@b
        assert actual==pytest.approx(expected,abs=2e-17)
        assert np.linalg.eigvalsh(full).min()>0
    # No determinant is discarded from an interacting quantum theory: only
    # the B-independent determinant cancels in this normalized Gaussian.


def test_both_source_orders_follow_actual_inverse_with_required_seagull(case):
    *_,jet,y=case;sources=torch.zeros(2,dtype=torch.float64)
    function=lambda z:torch_response(jet,y,z)
    jac=torch.autograd.functional.jacobian(function,sources).numpy()
    hess=torch.autograd.functional.hessian(lambda z:function(z)[3,0],sources).numpy()
    w=response_derivatives(jet,y)
    np.testing.assert_allclose(w[1],jac[:,:,0],atol=1e-12)
    np.testing.assert_allclose(w[2],jac[:,:,1],atol=1e-12)
    assert w[3][3,0]==pytest.approx(hess[0,1],abs=1e-11)
    without=response_derivatives((*jet[:3],np.zeros_like(jet[3])),y)
    assert abs(w[3][3,0]-without[3][3,0])>1


@pytest.mark.parametrize('scale',[.7,1.,1.6])
def test_interaction_Ward_identities_survive_collective_elimination(case,scale):
    base,s,model,args,momenta,_,y=case;p0,q1,q2,ea,eb=args
    q1=scale*q1
    hp=base.matrices(s['vacuum'])[0]
    momenta,jet=routed_jet(hp,model,p0,q1,q2,q1,eb)
    w=response_derivatives(jet,y);r=[np.linalg.inv(hp-dot(p,p)*np.eye(2)) for p in momenta]
    g=base.yukawa
    assert w[1][1,0]==pytest.approx(g@(r[1]-r[0])@g,abs=1e-12)
    boundary=g@(r[3]@one_source_matrix(model,momenta[3],momenta[1],eb)@r[1]
                -r[2]@one_source_matrix(model,momenta[2],momenta[0],eb)@r[0])@g
    assert w[3][3,0]==pytest.approx(boundary,abs=1e-11)
    wrong=response_derivatives((*jet[:3],np.zeros_like(jet[3])),y)[3][3,0]
    assert abs(wrong-boundary)>.01


def test_retained_subtraction_is_on_induced_functionals_including_currents(case):
    *_,jet,y=case;k,ka,kb,kab=jet
    # Synthetic retained content is explicitly declared; it is not inferred
    # from agreeing dimensions or asserted to be canonical QCD matching.
    kept=(k+.07*np.eye(8),.6*ka,.7*kb,.4*kab);yk=.4*y
    full=response_derivatives(jet,y);ret=response_derivatives(kept,yk)
    residual=matched_response(jet,y,kept,yk)
    for f,r,d in zip(full,ret,residual): np.testing.assert_allclose(r+d,f,atol=2e-12)
    naive=y.T@np.linalg.inv(jet[0]-kept[0])@y
    assert np.linalg.norm(naive-residual[0])>100
    assert np.linalg.norm(full[3]+ret[3]-full[3])>1


def test_source_dependent_field_coordinates_require_endpoint_derivatives(case):
    *_,jet,y=case;k,ka,kb,kab=[torch.tensor(z) for z in jet];yt=torch.tensor(y)
    rng=np.random.default_rng(75);t=torch.tensor(rng.normal(0,.1,(8,8)));s=t.T
    def response(z,freeze=False):
        a,b=z;u=torch.eye(8)+a*t+b*s
        km=u.T@(k+a*ka+b*kb+a*b*kab)@u
        ym=yt if freeze else u.T@yt
        return (ym.T@torch.linalg.solve(km,ym))[3,0]
    z=torch.zeros(2,dtype=torch.float64)
    expected=torch.autograd.functional.hessian(lambda x:torch_response(jet,y,x)[3,0],z)
    np.testing.assert_allclose(torch.autograd.functional.hessian(response,z),expected,atol=1e-11)
    wrong=torch.autograd.functional.hessian(lambda x:response(x,True),z)
    assert np.linalg.norm(wrong.numpy()-expected.numpy())>1


def test_pion_data_and_tree_mass_leave_heavy_quark_exchange_unidentified(case):
    base,s,model,*_=case;v=s['vacuum'];e=s['pion_modes'][:,0]
    perpendicular=np.array([-v[1],v[0]])/np.linalg.norm(v)
    changed=replace(base,yukawa=base.yukawa+1.2*perpendicular)
    other=changed.solve(.004)
    for key in ['vacuum','pion_mass_squared','scalar_mass_squared','quark_mass']:
        np.testing.assert_allclose(other[key],s[key],atol=2e-13)
    assert changed.yukawa@e==pytest.approx(base.yukawa@e)
    old=exchange_weights(base,v,.1);new=exchange_weights(changed,v,.1)
    assert abs(new[0]-old[0])>1 and abs(new[1]-old[1])>1


def test_H_T_U_are_derivatives_of_the_original_eight_field_potential(case):
    base,s=case[:2];h,t,u=potential_derivatives(base,s['vacuum'])
    x=torch.zeros(8,dtype=torch.float64);x[::4]=torch.tensor(s['vacuum'])
    def potential(z):
        f=z.reshape(2,4)
        return (torch.tensor(base.quartic)/4*((f*f).sum(1)-torch.tensor(base.radial_squared))**2).sum()+base.coupling/2*((f[0]-f[1])**2).sum()-.004*torch.tensor(base.scalar_source)@f[:,0]
    hfun=lambda z:torch.func.hessian(potential)(z)
    tfun=torch.func.jacfwd(hfun)
    np.testing.assert_allclose(h,hfun(x),atol=1e-13)
    np.testing.assert_allclose(t,tfun(x),atol=1e-12)
    np.testing.assert_allclose(u,torch.func.jacfwd(tfun)(x),atol=1e-11)


def test_many_quark_coefficients_preserve_isovector_parity_and_pairings(case):
    base,s=case[:2];h,t,u=potential_derivatives(base,s['vacuum']);y=yukawa_map(base.yukawa)
    w2,w3,w4=induced_many_quark_tensors(h,t,u,y)
    np.testing.assert_allclose(w2,np.diag(exchange_weights(base,s['vacuum'],0)),atol=1e-10)
    assert w3[0,0,0]<0 and w3[0,1,1]<0  # actual +g phi mass convention
    np.testing.assert_allclose([w3[0,i,i] for i in [1,2,3]],w3[0,1,1])
    assert w3[1,1,1]==0 and w3[0,0,1]==0 and w4[0,1,1,1]==0
    np.testing.assert_allclose(w4,w4.transpose(2,1,0,3),atol=1e-4,rtol=2e-14)
    ry=np.linalg.solve(h,y);bare=np.einsum('abcd,ai,bj,ck,dl->ijkl',u,ry,ry,ry,ry)
    assert np.linalg.norm(w4-bare)>np.linalg.norm(w4)*.5


@pytest.mark.parametrize('scale',[1e-5,3e-5,1e-4])
def test_small_bilinear_expansion_compared_to_independent_full_stationarity(case,scale):
    base,s=case[:2];h,t,u=potential_derivatives(base,s['vacuum']);y=yukawa_map(base.yukawa)
    b=scale*np.array([-1.,.2,-.1,.3])
    exact=nonlinear_bilinear_response(h,t,u,y,b)
    terms=tree_series(induced_many_quark_tensors(h,t,u,y),b)
    error=abs(np.cumsum(terms)-exact['action_density'])
    assert error[2]<error[1]<error[0]
    assert error[2]/abs(exact['action_density'])<.004
    # Independent full potential value, with +g*B*phi coupling and the
    # vacuum constant and linear +(g.v)*B_scalar explicitly subtracted.
    f=np.zeros((2,4));f[:,0]=s['vacuum'];shift=exact['displacement'].reshape(2,4)
    direct=base.potential(f+shift,.004)-base.potential(f,.004)+b@y.T@shift.ravel()
    assert direct==pytest.approx(exact['action_density'],abs=1e-16)


def test_tree_remainder_scaling_and_breakdown_are_not_hidden(case):
    base,s=case[:2];h,t,u=potential_derivatives(base,s['vacuum']);y=yukawa_map(base.yukawa)
    tensors=induced_many_quark_tensors(h,t,u,y);errors=[]
    for scale in [1e-5,2e-5,1e-3]:
        b=scale*np.array([-1.,.2,-.1,.3]);actual=nonlinear_bilinear_response(h,t,u,y,b)['action_density']
        errors.append(abs(np.cumsum(tree_series(tensors,b))-actual))
    np.testing.assert_allclose(errors[1]/errors[0],[8.,16.,32.],rtol=.06)
    assert errors[2][2]>errors[2][0]  # More terms are not valid everywhere.


def test_untruncated_branch_derivatives_are_linked_and_do_not_use_pairwise_cutoff(case):
    base,s=case[:2];h,t,u=potential_derivatives(base,s['vacuum']);y=yukawa_map(base.yukawa)
    b=3e-4*np.array([-1.,.2,-.1,.3]);result=nonlinear_bilinear_response(h,t,u,y,b)
    delta=1e-8;grad=[];hess=[]
    for direction in np.eye(4):
        plus=nonlinear_bilinear_response(h,t,u,y,b+delta*direction)
        minus=nonlinear_bilinear_response(h,t,u,y,b-delta*direction)
        grad.append((plus['action_density']-minus['action_density'])/(2*delta))
        hess.append((plus['bilinear_gradient']-minus['bilinear_gradient'])/(2*delta))
    np.testing.assert_allclose(result['bilinear_gradient'],grad,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(result['bilinear_hessian'],np.array(hess).T,rtol=1e-8,atol=1e-6)
    pairwise=-y.T@np.linalg.solve(h,y)
    assert np.linalg.norm(result['bilinear_hessian']-pairwise)>10


def test_invalid_inputs(case):
    with pytest.raises(ValueError): yukawa_map([1])
    with pytest.raises(ValueError): exchange_weights(case[0],case[1]['vacuum'],-1)
    with pytest.raises(ValueError): routed_jet(np.eye(2),case[2],np.array([2,0,0,0]),*case[3][1:])
    p,q1,q2,ea,eb=case[3]
    with pytest.raises(ValueError): routed_jet(np.eye(2),case[2],p,q1,q1,ea,eb)


def test_nonlinear_fluctuation_Hessian_is_not_source_independent(case):
    base,s=case[:2];h,t,u=potential_derivatives(base,s['vacuum']);y=yukawa_map(base.yukawa)
    b=1e-4*np.array([-1.,.2,-.1,.3]);result=nonlinear_bilinear_response(h,t,u,y,b)
    ratio=np.linalg.slogdet(result['collective_curvature'])[1]-np.linalg.slogdet(h)[1]
    assert abs(ratio)>.1
    gaussian=nonlinear_bilinear_response(h,0*t,0*u,y,b)
    np.testing.assert_array_equal(gaussian['collective_curvature'],h)
    # A local determinant witness is not the regulated spacetime loop trace.


def test_replay_labels_limits_and_preserves_actual_failures():
    import json
    from scripts.evaluate_collective_quark_matching import evaluate
    result=evaluate();json.dumps(result,allow_nan=False)
    current=result['gauged_exchange'];info=result['nonidentifiability']
    assert current['two_source_Ward_error']<1e-10
    assert current['missing_contact_Ward_error']>.01
    assert current['retained_reconstruction_max_error']<1e-10
    assert abs(info['tree_mass_change_GeV'])<1e-12
    # This is a sensitivity witness, not a physical minimum-response cut.
    assert info['null_direction_change_two_source_norm_GeV_minus4']==pytest.approx(.05187670779613103,rel=1e-9)
    probes=result['nonlinear_tree']['probes']
    assert probes[0]['relative_errors_through_4q_6q_8q'][2]<4e-6
    assert probes[-1]['relative_errors_through_4q_6q_8q'][2]>2

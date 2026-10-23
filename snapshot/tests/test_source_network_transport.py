import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pytest
import torch
from scipy.linalg import expm

from deuteron_wigner.quantum.source_network_transport import (
    symmetric_basis,accessibility,insertion_design,diagonal_cycle_null,
    contact_coefficients,source_jet,response,source_derivatives,one_source,
    open_bilinear_tensor,two_E_return,canonical_coordinates,
)
from deuteron_wigner.quantum.joint_source_matching import JointSourceAction,action_contact,stress_contact
from deuteron_wigner.quantum.collective_quark_matching import bilinear_matrices,routed_jet
from deuteron_wigner.quantum.transverse_source_matching import SourceCurvature,dot


def fixture(n=3):
    rng=np.random.default_rng(22+n);z=rng.normal(size=(n,n));u=expm(.2*(z-z.T))
    h=u@np.diag(np.linspace(.2,1.1,n))@u.T;g=u@np.linspace(1.,.5,n)
    matrices=[]
    for scale in [.4,.2,.7]:
        z=rng.normal(size=(n,n));matrices.append(scale*(z+z.T)/2)
    p0=np.array([.07,.3,.12,.11]);q1=np.array([.01,.04,-.01,.02]);q2=np.array([.02,-.03,.02,-.04])
    e1=np.array([.1,.3,-.2,.4]);e2=np.array([.2,-.1,.4,.3])
    args=(p0,q1,q2,e1,e2)
    return h,g,matrices,args


@pytest.mark.parametrize('n',[1,2,3,4])
def test_distinct_poles_diagonal_rank_and_independent_route_recovery(n):
    h=np.diag(np.geomspace(.1,1.5,n));g=np.linspace(1.,.7,n);x=np.geomspace(.002,8.,40)
    diagonal=insertion_design(h,g,x,diagonal=True);full=insertion_design(h,g,x[::5])
    assert np.linalg.matrix_rank(diagonal)==2*n-1
    assert np.linalg.matrix_rank(full)==n*(n+1)//2
    basis=symmetric_basis(n)
    np.testing.assert_allclose(np.einsum('aij,bij->ab',basis,basis),np.eye(len(basis)),atol=2e-15)
    rng=np.random.default_rng(5);e=rng.normal(size=(n,n));e=(e+e.T)/2
    coeff=np.einsum('kij,ij->k',basis,e);z=np.linalg.solve(h+x[0]*np.eye(n),g)
    assert (diagonal@coeff)[0]==pytest.approx(z@e@z)


@pytest.mark.parametrize('cycle',[(0,1,2),(0,1,3),(0,2,3)])
def test_exact_response_graph_cycle_is_diagonally_hidden_but_not_physically_zero(cycle):
    poles=np.array([.15,.4,.9,1.7]);g=np.array([1.,.8,.7,.5])
    e=diagonal_cycle_null(poles,g,cycle);h=np.diag(poles)
    for x in [.001,.03,.2,1.3,10.]:
        v=g/(poles+x)
        assert abs(v@e@v)<3e-14
    left=g/(poles+.02);right=g/(poles+.8)
    assert abs(left@e@right)>.01
    # Edges are source matrix entries; setting them all to zero is not a
    # coordinate change and loses a finite independent-route response.
    assert np.linalg.norm(e)>0
    tree=e.copy();i,j=cycle[-1],cycle[0];tree[i,j]=tree[j,i]=0.
    v=g/(poles+.03)
    assert abs(v@tree@v)>.1


def test_source_free_and_full_source_accessibility_are_different():
    h=np.diag([.2,.6,1.2]);g=np.array([1.,0,0])
    e=np.array([[0.,.4,0.],[.4,0.,.3],[0.,.3,0.]])
    assert accessibility(h,g).shape[1]==1
    assert accessibility(h,g,source_matrices=[e]).shape[1]==3
    assert np.linalg.matrix_rank(insertion_design(h,g,[.02,.1,.7]))==1
    # Degenerate H projects a single g into one vector within that eigenspace.
    degenerate=np.diag([.2,.2,.8]);g=np.array([1.,1.,1.])
    assert accessibility(degenerate,g).shape[1]==2
    assert accessibility(degenerate,g,source_matrices=[np.diag([.1,.4,.9])]).shape[1]==3
    assert accessibility(degenerate,np.eye(3)).shape[1]==3


def test_multiple_physical_endpoints_can_resolve_equal_virtuality_source_matrix():
    h=np.diag([.2,.2,.8]);x=[.1]
    assert np.linalg.matrix_rank(insertion_design(h,np.eye(3),x,diagonal=True))==6
    assert np.linalg.matrix_rank(insertion_design(h,np.array([1.,1.,1.]),x,diagonal=True))==1


def test_near_degenerate_is_ill_conditioned_not_the_exact_degenerate_model():
    x=np.geomspace(.01,3.,16);g=np.array([1.,.8]);ratios=[]
    for split in [.1,.01,.001]:
        h=np.diag([.2,.2+split]);design=insertion_design(h,g,x)
        assert np.linalg.matrix_rank(design)==3
        values=np.linalg.svd(design,compute_uv=False);ratios.append(values[-1]/values[0])
    assert ratios[2]<ratios[1]/50<ratios[0]/2500
    assert np.linalg.matrix_rank(insertion_design(.2*np.eye(2),g,x))==1


def test_node_rotations_preserve_reachability_and_kernel_response():
    h,g,matrices,_=fixture();e=matrices[2]
    gen=np.array([[0.,.3,-.1],[-.3,0.,.2],[.1,-.2,0.]]);u=expm(gen)
    old=accessibility(h,g,source_matrices=[e]);new=accessibility(u@h@u.T,u@g,source_matrices=[u@e@u.T])
    np.testing.assert_allclose(new@new.T,u@old@old.T@u.T,atol=3e-14)
    x=np.array([.02,.1,.4,.8]);basis=symmetric_basis(3)
    direct=insertion_design(h,g,x)@np.einsum('kij,ij->k',basis,e)
    moved=insertion_design(u@h@u.T,u@g,x)@np.einsum('kij,ij->k',basis,u@e@u.T)
    np.testing.assert_allclose(moved,direct,atol=2e-14)
    np.testing.assert_allclose(np.linalg.svd(insertion_design(u@h@u.T,u@g,x),compute_uv=False),np.linalg.svd(insertion_design(h,g,x),compute_uv=False),atol=3e-14)


@pytest.mark.parametrize('weak',[1e-2,1e-4])
def test_weak_coupling_conditioning_is_not_an_exact_missing_mode(weak):
    h=np.diag([.2,.8]);x=np.geomspace(.01,3.,12)
    full=insertion_design(h,[1.,weak],x);s=np.linalg.svd(full,compute_uv=False)
    assert np.linalg.matrix_rank(full)==3
    assert s[-1]/s[0]<weak*weak
    assert accessibility(h,[1.,weak]).shape[1]==2
    assert accessibility(h,[1.,0.]).shape[1]==1


@pytest.mark.parametrize('n',[1,2,3])
def test_actual_action_contact_is_carried_with_all_nodes_and_both_orders(n):
    h,g,matrices,args=fixture(n);momenta,jet=source_jet(h,*matrices,*args)
    p0,q1,q2,e1,e2=args;model=JointSourceAction(*matrices);eye=np.eye(n)
    direct=np.array([[action_contact(model,momenta[3],p0,q1,q2,e1,e2,eye[i],eye[j]).real for j in range(n)] for i in range(n)])
    np.testing.assert_allclose(jet[3][3*n:4*n,:n],-2*dot(e1,e2)*np.eye(n)-direct,atol=2e-14)
    for block in jet: np.testing.assert_allclose(block,block.T,atol=2e-14)
    y=np.kron(np.eye(4),g[:,None]);z=torch.zeros(2,dtype=torch.float64)
    exact=source_derivatives(jet,y)
    automatic=torch.autograd.functional.hessian(lambda a:response(jet,y,a)[3,0],z)
    assert exact[3][3,0]==pytest.approx(automatic[0,1],abs=3e-13)
    lost=source_derivatives((*jet[:3],torch.zeros_like(jet[3])),y)
    assert abs(exact[3][3,0]-lost[3][3,0])>.01
    if n==2:
        old_model=SourceCurvature(*matrices[:2]);_,old=routed_jet(h,old_model,*args)
        _,zero_e=source_jet(h,*matrices[:2],matrices[2]*0,*args)
        for a,b in zip(old,zero_e): np.testing.assert_allclose(a,b,atol=2e-14)


@pytest.mark.parametrize('n',[1,3])
def test_eliminated_Ward_includes_endpoint_changes_and_nonminimal_sources(n):
    h,g,matrices,args=fixture(n);p0,q1,q2,e1,e2=args
    ps,jet=source_jet(h,*matrices,p0,q1,q2,q1,e2);y=np.kron(np.eye(4),g[:,None])
    actual=source_derivatives(jet,y);r=[np.linalg.inv(h-dot(p,p)*np.eye(n)) for p in ps]
    assert actual[1][1,0]==pytest.approx(g@(r[1]-r[0])@g,abs=3e-13)
    c=torch.tensor(matrices[0]);v31=one_source(n,c,ps[3],ps[1],e2).numpy();v20=one_source(n,c,ps[2],ps[0],e2).numpy()
    boundary=g@(r[3]@v31@r[1]-r[2]@v20@r[0])@g
    assert actual[3][3,0]==pytest.approx(boundary,abs=3e-13)


def test_E_response_is_independent_route_insertion_not_projection_on_one_mode():
    h,g,matrices,args=fixture();p0,q1,q2,e1,e2=args;n=len(h)
    ps,full=source_jet(h,*matrices,*args);_,none=source_jet(h,*matrices[:2],matrices[2]*0,*args)
    y=np.kron(np.eye(4),g[:,None]);actual=(source_derivatives(full,y)[3]-source_derivatives(none,y)[3])[3,0]
    left=np.linalg.solve(h-dot(ps[3],ps[3])*np.eye(n),g);right=np.linalg.solve(h-dot(p0,p0)*np.eye(n),g)
    reduced=stress_contact(ps[3],p0,q1,q2,e1,e2,left@matrices[2]@right)
    assert actual==pytest.approx(reduced,abs=2e-13)
    assert abs(reduced)>1e-5


def test_joint_interaction_source_gradient_includes_propagator_endpoints_and_contact():
    h,g,m,args=fixture(2);basis=torch.tensor(symmetric_basis(2));zero=torch.zeros(3,dtype=torch.float64)
    def build(theta):
        ht=torch.tensor(h)+theta[0]*basis[0];gt=torch.tensor(g)+theta[1]*torch.tensor([.2,-.1],dtype=torch.float64)
        et=torch.tensor(m[2])+theta[2]*basis[1]
        _,jet=source_jet(ht,m[0],m[1],et,*args)
        return jet,torch.kron(torch.eye(4,dtype=torch.float64),gt[:,None])
    def mixed(theta):
        jet,y=build(theta);return source_derivatives(jet,y)[3][3,0]
    gradient=torch.autograd.functional.jacobian(mixed,zero).numpy()
    # These coordinates have different units/sensitivity. Compare the E
    # derivative to its independent insertion, not a shared size threshold.
    p0,q1,q2,eps1,eps2=args;p1=p0+q1+q2
    left=np.linalg.solve(h-dot(p1,p1)*np.eye(2),g);right=np.linalg.solve(h-dot(p0,p0)*np.eye(2),g)
    expected_e=stress_contact(p1,p0,q1,q2,eps1,eps2,left@basis[1].numpy()@right)
    assert gradient[2]==pytest.approx(expected_e,abs=2e-15)
    assert expected_e!=0
    numeric=[]
    for i in range(3):
        step=torch.zeros_like(zero);step[i]=1e-5
        numeric.append(float((mixed(step)-mixed(-step))/(2e-5)))
    np.testing.assert_allclose(gradient,numeric,rtol=1e-7,atol=2e-9)
    # Independent ordering: differentiate interaction first on the full
    # source-dependent inverse, then twice in external sources.
    def parameter_first(sources):
        return torch.autograd.functional.jacobian(lambda theta:response(*build(theta),sources)[3,0],zero,create_graph=True)[0]
    source_hessian=torch.autograd.functional.hessian(parameter_first,torch.zeros(2,dtype=torch.float64))
    assert source_hessian[0,1]==pytest.approx(gradient[0],abs=2e-12)
    # Freeze propagators/endpoints: only the E derivative remains. This is
    # exactly the missing feedback one must not discard in a joint fit.
    assert abs(gradient[0])>.01 and abs(gradient[1])>.01


def test_full_minus_retained_is_functional_including_mixed_derivatives():
    h,g,m,args=fixture(2);_,full=source_jet(h,*m,*args);_,ret=source_jet(h+.12*np.eye(2),*(.4*a for a in m),*args)
    y=np.kron(np.eye(4),g[:,None]);yr=.6*y;s=torch.zeros(2,dtype=torch.float64)
    difference=source_derivatives(full,y)[3]-source_derivatives(ret,yr)[3]
    direct=torch.autograd.functional.hessian(lambda z:(response(full,y,z)-response(ret,yr,z))[3,0],s)
    assert direct[0,1]==pytest.approx(difference[3,0],abs=3e-13)
    wrong=y.T@np.linalg.inv(full[0].numpy()-ret[0].numpy())@y
    assert np.linalg.norm(wrong-(source_derivatives(full,y)[0]-source_derivatives(ret,yr)[0]).numpy())>1


def test_nonorthogonal_field_coordinates_require_kinetic_metric_and_all_source_partners():
    h,g,m,args=fixture(2);t=np.array([[1.2,.3],[-.1,.8]]);z=t.T@t
    hm=t.T@h@t;gm=t.T@g;mm=[t.T@a@t for a in m]
    _,old=source_jet(h,*m,*args);_,moved=source_jet(hm,*mm,*args,metric=z)
    y=np.kron(np.eye(4),g[:,None]);ym=np.kron(np.eye(4),gm[:,None])
    original=source_derivatives(old,y);transformed=source_derivatives(moved,ym)
    for a,b in zip(original,transformed): np.testing.assert_allclose(a,b,atol=3e-13)
    hc,gc,mc,pullback=canonical_coordinates(hm,gm,z,source_matrices=mm)
    np.testing.assert_allclose(pullback.T@z@pullback,np.eye(2),atol=3e-15)
    _,canonical=source_jet(hc,*mc,*args);yc=np.kron(np.eye(4),gc)
    for a,b in zip(original,source_derivatives(canonical,yc)): np.testing.assert_allclose(a,b,atol=3e-13)
    _,wrong=source_jet(hm,*mm,*args)
    assert abs(source_derivatives(wrong,ym)[3][3,0]-original[3][3,0])>.01
    # Keeping H,G transport but freezing a source matrix is equally wrong.
    _,wrong_source=source_jet(hc,mc[0],mc[1],mm[2],*args)
    assert abs(source_derivatives(wrong_source,yc)[3][3,0]-original[3][3,0])>1e-6


def test_resolved_quark_tensor_keeps_charged_flavor_phases_spin_and_color():
    plus=np.array([0.,1.,1j,0.])/np.sqrt(2);minus=plus.conj();w=.73*np.outer(plus,minus)
    tensor=open_bilinear_tensor(w);b=bilinear_matrices();a=np.einsum('A,Aij->ij',plus,b);c=np.einsum('A,Aij->ij',minus,b)
    np.testing.assert_allclose(tensor,.73*np.einsum('ij,kl->ijkl',a,c),atol=2e-15)
    rng=np.random.default_rng(91);left=rng.normal(size=(24,24))+1j*rng.normal(size=(24,24));right=rng.normal(size=(24,24))+1j*rng.normal(size=(24,24))
    actual=np.einsum('ijkl,ij,kl',tensor,left,right)
    assert actual==pytest.approx(.73*np.sum(a*left)*np.sum(c*right),abs=2e-12)
    assert abs(actual-np.einsum('ijkl,ij,kl',open_bilinear_tensor(w.real),left,right))>1
    charge=np.kron(np.eye(4),np.kron(np.diag([2/3,-1/3]),np.eye(3)))
    np.testing.assert_allclose(charge@a-a@charge,a,atol=2e-15)
    np.testing.assert_allclose(charge@c-c@charge,-c,atol=2e-15)


def four_case():
    h=np.diag([.2,.7]);e=np.array([[0.,.6],[.6,0.]]);g=np.array([1.,0.]);p=np.array([.03,.35,.12,.1])
    q=np.array([[.01,.02,.01,0.],[.02,-.01,.02,.01],[.01,.01,-.02,.01],[.01,-.02,.01,-.01]])
    eps=np.array([[.1,.3,-.2,.4],[.2,-.1,.4,.3],[.1,.2,.3,-.1],[-.1,.4,.1,.2]])
    return h,e,g,p,q,eps


def test_hidden_mode_returns_at_two_E_insertions_with_gauge_and_permutation_checks():
    h,e,g,p,q,eps=four_case();result,parts=two_E_return(h,e,g,p,q,eps)
    assert abs(result[0,0])>1e-10
    assert parts.shape==(6,1,1)
    np.testing.assert_allclose(insertion_design(h,g,[.01,.1,.5])@np.einsum('kij,ij->k',symmetric_basis(2),e),0,atol=1e-15)
    for i in range(4):
        shifted=eps.copy();shifted[i]+=.7*q[i]
        assert two_E_return(h,e,g,p,q,shifted)[0][0,0]==pytest.approx(result[0,0],abs=1e-20)
        longitudinal=eps.copy();longitudinal[i]=q[i]
        assert abs(two_E_return(h,e,g,p,q,longitudinal)[0][0,0])<1e-23
    perm=[2,0,3,1]
    np.testing.assert_allclose(two_E_return(h,e,g,p,q[perm],eps[perm])[0],result,atol=1e-20)
    assert two_E_return(h[:1,:1],e[:1,:1],g[:1],p,q,eps)[0][0,0]==0
    generator=np.zeros((4,4));generator[0,1:]=generator[1:,0]=[.25,-.2,.1];boost=expm(generator)
    np.testing.assert_allclose(two_E_return(h,e,g,boost@p,q@boost.T,eps@boost.T)[0],result,atol=1e-20)


def test_two_E_return_from_independent_source_polynomial_inverse_derivative():
    h,e,g,p,q,eps=four_case();n=len(h);route=[p+sum((q[i] for i in range(4) if mask>>i&1),start=np.zeros(4)) for mask in range(16)]
    k=np.zeros((16*n,16*n));pair_blocks={pair:np.zeros_like(k) for pair in combinations(range(4),2)}
    for mask,momentum in enumerate(route):
        row=slice(n*mask,n*(mask+1));k[row,row]=h-dot(momentum,momentum)*np.eye(n)
        for pair in pair_blocks:
            bits=sum(1<<i for i in pair)
            if mask&bits: continue
            end=mask|bits;out=slice(n*end,n*(end+1));i,j=pair
            # Independent ACTUAL action coefficient, not stress_contact.
            coefficient=contact_coefficients(route[end],momentum,q[i],q[j],eps[i],eps[j])[2]
            pair_blocks[pair][out,row]=-coefficient*e
    kt=torch.tensor(k);terms={key:torch.tensor(value) for key,value in pair_blocks.items()};gt=torch.tensor(g)
    def kernel(a):
        matrix=kt+sum(a[i]*a[j]*value for (i,j),value in terms.items())
        return gt@torch.linalg.solve(matrix,torch.cat([gt,torch.zeros(15*n,dtype=torch.float64)]))[-n:]
    z=torch.zeros(4,dtype=torch.float64,requires_grad=True);value=kernel(z)
    for i in range(4): value=torch.autograd.grad(value,z,create_graph=True)[0][i]
    expected=two_E_return(h,e,g,p,q,eps)[0][0,0]
    assert float(value.detach())==pytest.approx(expected,rel=2e-13,abs=1e-22)


def test_input_guards():
    with pytest.raises(ValueError): accessibility(-np.eye(2),[1.,0.])
    with pytest.raises(ValueError): insertion_design(np.eye(2),[1.,1.],[-.1])
    with pytest.raises(ValueError): diagonal_cycle_null([.2,.2,.3],[1.,1.,1.],(0,1,2))
    with pytest.raises(ValueError): diagonal_cycle_null([.2,.4,.8],[1.,0.,1.],(0,1,2))
    with pytest.raises(ValueError): open_bilinear_tensor(np.eye(3))
    with pytest.raises(ValueError): canonical_coordinates(np.eye(2),[1.,1.],-np.eye(2))


def test_saved_scientific_replay():
    from scripts.evaluate_source_network_transport import evaluate
    saved=json.loads(Path('docs/next_level/source_network_transport_results.json').read_text())
    assert evaluate()==saved

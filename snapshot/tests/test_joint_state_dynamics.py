import numpy as np
import pytest
torch = pytest.importorskip('torch')
from deuteron_wigner.quantum.joint_state_dynamics import (
    GraphState,SourceFamily,AmplitudeMatching,state_metric,REAL,COMPLEX,
)


def family():
    h = np.diag([0.,.7,1.3,2.])
    t = np.zeros((3,4,4),complex)
    for i in range(3):t[0,i,i+1]=t[0,i+1,i]=-.4
    t[1] = np.diag([.1,-.2,.3,-.1])
    t[2,0,3]=.2j;t[2,3,0]=-.2j
    return SourceFamily(h,t,['hopping','diagonal','loop'],units='dimensionless_fixture')


@pytest.mark.parametrize('edges', [[(0,1),(1,2),(2,3)],[(0,1),(0,2),(0,3)]])
def test_quantum_state_and_full_gradient_match_independent_Torch_rotations(edges):
    pytest.importorskip('pennylane')
    graph = GraphState(['a','b','c','d'],edges)
    angles = torch.tensor([[.4,.2],[-.7,.6],[.9,-.5]],dtype=REAL,requires_grad=True)
    oracle = graph.pennylane_state()
    np.testing.assert_allclose(oracle(angles).detach(),graph.torch_state(angles).detach(),atol=3e-16)
    operator = family().matrix(torch.tensor([.6,-.4,.3],dtype=REAL))
    gradients = []
    for statefn in (oracle,graph.torch_state):
        psi=statefn(angles)
        value=torch.vdot(psi,operator@psi).real
        gradients.append(torch.autograd.grad(value,angles)[0].detach().numpy())
    np.testing.assert_allclose(*gradients,atol=3e-16)


def test_source_derivatives_local_gauge_and_continuity():
    model = family(); phi=torch.tensor([.8,.2,.4],dtype=REAL,requires_grad=True)
    currents=model.currents(phi); h=model.matrix(phi)
    for i in range(4):
        density=torch.diag(torch.eye(4,dtype=COMPLEX)[i])
        divergence=torch.einsum('e,eij->ij',model.incidence[i].to(COMPLEX),currents)
        torch.testing.assert_close(divergence,1j*(density@h-h@density),atol=1e-14,rtol=0)
    chi=torch.tensor([.2,-.3,.5,-.1],dtype=REAL)
    u=torch.diag(torch.exp(1j*chi))
    torch.testing.assert_close(model.matrix(phi,model.incidence.T@chi),u@h@u.conj().T,atol=1e-14,rtol=0)
    for e in range(len(model.edges)):
        da=torch.eye(len(model.edges),dtype=REAL)[e]*1e-5
        torch.testing.assert_close((model.matrix(phi,da)-model.matrix(phi,-da))/(2e-5),currents[e],atol=1e-11,rtol=0)
    # Dynamics derivative includes the current partner, not detached/fixed weights.
    derivative=torch.autograd.grad(currents.abs().square().sum(),phi)[0]
    assert torch.linalg.norm(derivative)>0


def test_complex_correlated_matching_subtracts_retained_content_and_reports_nullspace():
    a=np.array([[1+2j,0,2+4j],[0,1j,0],[2,1,4]],complex)
    truth=np.array([.4,-.7,.2]); retained=np.array([2+.3j,-1j,3.])
    covariance=np.eye(6)+.1*np.ones((6,6))
    match=AmplitudeMatching(a,retained+a@truth,retained,covariance,specification='finite amplitude/operator matching fixture')
    assert match.rank==2 and match.nullspace.shape==(3,1)
    np.testing.assert_allclose(a@match.representative,a@truth,atol=1e-14)
    np.testing.assert_allclose(a@match.nullspace,0,atol=1e-14)
    assert float(match.loss(torch.tensor(truth,dtype=REAL)))<1e-26
    # Omitting the retained subtraction does not implement this matching.
    wrong=AmplitudeMatching(a,retained+a@truth,np.zeros(3,complex),covariance,specification='negative_control')
    assert float(wrong.loss(torch.tensor(truth,dtype=REAL)))>1


def test_state_metric_is_projective_and_not_the_coefficient_matching_metric():
    graph=GraphState(['a','b','c'],[(0,1),(1,2)])
    parameters=torch.tensor([[.7,.3],[-.8,.4]],dtype=REAL,requires_grad=True)
    metric=state_metric(graph.torch_state,parameters)
    rotated=state_metric(lambda p:torch.exp(1j*p.sum())*graph.torch_state(p),parameters)
    torch.testing.assert_close(metric,rotated,atol=1e-14,rtol=0)
    assert torch.linalg.eigvalsh(metric).min()>1e-5
    assert metric.shape==(4,4) # State coordinates; dynamics has a different parameter count.


def test_eigenstate_residual_rejects_a_state_with_the_correct_energy_expectation_only():
    model=family();phi=torch.tensor([.8,.2,.4],dtype=REAL)
    eig,vec=torch.linalg.eigh(model.matrix(phi))
    psi=(vec[:,0]+vec[:,2])/np.sqrt(2)
    energy,residual=model.eigenstate_residual(phi,psi)
    torch.testing.assert_close(energy,(eig[0]+eig[2])/2)
    assert residual>.1
    assert model.eigenstate_residual(phi,vec[:,0])[1]<1e-28


def test_invalid_carriers_and_covariance_rejected():
    with pytest.raises(ValueError):GraphState(['a','b','c'],[(0,1)])
    with pytest.raises(ValueError):AmplitudeMatching(np.eye(2),[1,2],[0,0],np.zeros((2,2)),specification='fixture')
    with pytest.raises(ValueError):SourceFamily(np.eye(2),[np.ones((2,2))],['x'],units='')
    with pytest.raises(ValueError):family().matrix(torch.ones(3,dtype=COMPLEX))
    with pytest.raises(ValueError):family().matrix(torch.ones(3,dtype=REAL),torch.ones(4,dtype=COMPLEX))


def test_zero_state_not_a_solution_and_diagonal_family_has_no_link_current():
    model=SourceFamily(np.diag([0.,1.]),[np.eye(2)],['offset'],units='fixture')
    phi=torch.tensor([.4],dtype=REAL)
    assert model.currents(phi).shape==(0,2,2)
    for state in (torch.zeros(2,dtype=COMPLEX),torch.ones(2,dtype=COMPLEX)):
        with pytest.raises(ValueError):model.eigenstate_residual(phi,state)


def test_conjugate_ambiguity_and_phase_sensitive_matching_resolution():
    from scripts.evaluate_joint_state_dynamics import joint_fit
    left=joint_fit(initial_loop=.2)
    right=joint_fit(initial_loop=-.2)
    for result in (left,right):
        assert result['training_probe_max_error']<1e-8
        assert result['state_equation_norm']<1e-8
    assert right['learned'][2]<-.5
    assert right['withheld_current_operator_error']>.3
    resolved=joint_fit(initial_loop=-.2,phase_matching=True)
    assert resolved['matching_rank']==3
    assert resolved['withheld_current_operator_error']<1e-8
    negative=joint_fit(use_state_equation=False)
    assert negative['training_probe_max_error']<1e-8
    assert negative['state_equation_norm']>.1


def test_shared_neural_coefficients_and_quantum_state_both_receive_current_gradients():
    pytest.importorskip('pennylane')
    torch.manual_seed(47)
    # The input is explicitly a numerical feature fixture, not a physical
    # invariant construction. Physical feature qualification is upstream.
    network=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Tanh(),torch.nn.Linear(4,3)).double()
    phi=network(torch.tensor([.2,.7],dtype=REAL))
    statefn=GraphState(['a','b','c','d'],[(0,1),(1,2),(2,3)]).pennylane_state()
    angles=torch.tensor([[.6,.3],[.8,-.2],[.4,.5]],dtype=REAL,requires_grad=True)
    state=statefn(angles)
    response=torch.einsum('i,eij,j->e',state.conj(),family().currents(phi),state).real
    gradients=torch.autograd.grad(response.square().sum(),(angles,*network.parameters()))
    assert all(torch.isfinite(g).all() and torch.linalg.norm(g)>1e-9 for g in gradients)

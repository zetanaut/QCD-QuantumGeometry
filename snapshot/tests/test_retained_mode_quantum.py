import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import eigh

from deuteron_wigner.quantum.common_action_pilot import calibrated_action
from deuteron_wigner.quantum.retained_mode_quantum import (
    oscillator_states,polynomial_cell,chiral_cell,ground_state,
    connected_response,rotation_generator,eliminate_exact,
    eliminated_ground_energy,eliminated_response,
)


@pytest.fixture(scope='module')
def reference():
    a=calibrated_action(.13957039,.0922,.515,.004,.6,1.4,.12,[3.,1.])
    cell=chiral_cell(a,.004,volume=1000.,shell=3,frequency=.5)
    return a,cell,ground_state(cell.hamiltonian)


def source(cell,couplings,component):
    weights=np.zeros((2,4));weights[:,component]=couplings
    return cell.linear_source(weights)


def retained(cell):
    return np.array([i for i,s in enumerate(cell.states) if sum(s[4:])==0])


@pytest.mark.parametrize('shell,size',[(0,1),(2,45),(4,495),(6,3003)])
def test_complete_total_shell_basis_and_prefix(shell,size):
    states=oscillator_states(shell)
    assert len(states)==size and len(set(states))==size
    assert all(sum(s)<=shell for s in states)
    assert oscillator_states(shell+2)[:size]==states


def test_quartic_polynomial_is_projected_after_not_before_multiplication():
    volume=1000.;omega=.5;mu=np.array([.2,.3]);lam=np.array([4.,7.]);k=.03
    cell=polynomial_cell(mu,lam,k,np.zeros((2,4)),volume=volume,shell=0,frequency=omega)
    expected=2*omega+(mu.sum()+2*k)/omega+1.5*lam.sum()/(volume*omega**2)
    assert cell.hamiltonian[0,0]==pytest.approx(expected,abs=1e-13)
    assert all(x.nnz==0 for x in cell.fields)
    # Squaring these projected coordinates would erase the vacuum moments,
    # despite each retained operator separately being Hermitian.


@pytest.mark.parametrize('nu',[0.,.15,.4])
def test_exact_harmonic_response_and_frequency_normalization(nu):
    omega=np.array([.3,.4]);g=np.array([3.,1.]);v=700.
    c=polynomial_cell(omega**2,[0.,0.],0.,np.zeros((2,4)),volume=v,shell=2,frequency=omega)
    state=ground_state(c.hamiltonian)
    assert state['energy']==pytest.approx(2*omega.sum(),abs=1e-13)
    chi=connected_response(c,state,[source(c,g,k) for k in range(4)],nu)['susceptibility']
    expected=np.sum(g*g/(omega*omega+nu*nu))
    np.testing.assert_allclose(chi,expected*np.eye(4),atol=1e-10)


def test_coupled_harmonic_vacuum_and_shift_are_independent_controls():
    mu=np.array([.12,.2]);k=.03;v=500.;forcing=np.zeros((2,4));forcing[:,0]=[.0003,.0002]
    c=polynomial_cell(mu,[0.,0.],k,forcing,volume=v,shell=8,frequency=.4)
    state=ground_state(c.hamiltonian);h=np.diag(mu)+k*np.array([[1,-1],[-1,1]])
    exact=2*np.sqrt(np.linalg.eigvalsh(h)).sum()-.5*v*forcing[:,0]@np.linalg.solve(h,forcing[:,0])
    assert state['energy']==pytest.approx(exact,abs=2e-7)
    actual=np.array([state['state']@c.fields[i]@state['state'] for i in [0,4]])
    np.testing.assert_allclose(actual,np.linalg.solve(h,forcing[:,0]),rtol=3e-5,atol=1e-9)


def test_internal_vector_and_axial_spurion_generator_relations(reference):
    a,c,s=reference;h=c.hamiltonian
    vector=rotation_generator(c.states,1,2)+rotation_generator(c.states,5,6)
    axial=rotation_generator(c.states,0,1)+rotation_generator(c.states,4,5)
    assert sparse.linalg.norm(vector@h-h@vector)<1e-11
    np.testing.assert_allclose((axial@h-h@axial+1j*.004*source(c,a.scalar_source,1)).toarray(),0,atol=1e-12)
    assert np.linalg.norm(vector@s['state'])<1e-10


@pytest.mark.parametrize('theta',[.2,.8])
def test_finite_chiral_spurion_rotation_preserves_quantum_energy(reference,theta):
    a,c,s=reference
    rotated=chiral_cell(a,.004*np.cos(theta),volume=c.volume,shell=3,frequency=.5,
                        pseudoscalar=[.004*np.sin(theta),0,0])
    assert ground_state(rotated.hamiltonian)['energy']==pytest.approx(s['energy'],abs=1e-11)


def test_quantum_PCAC_static_susceptibility_and_all_pion_directions(reference):
    a,c,s=reference
    ops=[source(c,a.scalar_source,k) for k in [1,2,3]]
    chi=connected_response(c,s,ops)['susceptibility']
    expectation=float(s['state']@source(c,a.scalar_source,0)@s['state'])/c.volume
    np.testing.assert_allclose(chi,expectation/.004*np.eye(3),atol=2e-10)


@pytest.mark.parametrize('component',[0,1])
def test_quantum_source_derivatives_match_independent_eigenvalue_differences(reference,component):
    a,c,s=reference;op=source(c,a.yukawa,component);delta=2e-6
    plus=ground_state(c.hamiltonian+delta*op)['energy'];minus=ground_state(c.hamiltonian-delta*op)['energy']
    result=connected_response(c,s,[op]);first=(plus-minus)/(2*delta)
    second=-(plus+minus-2*s['energy'])/(delta*delta*c.volume)
    assert first==pytest.approx(result['operator_means'][0],rel=1e-6,abs=2e-7)
    assert second==pytest.approx(result['susceptibility'][0,0],rel=2e-5,abs=1e-5)


def test_exact_elimination_energy_embedding_norm_and_source_partners(reference):
    a,c,s=reference;h=c.hamiltonian.toarray();p=retained(c);e=eliminated_ground_energy(h,p)
    assert e==pytest.approx(s['energy'],abs=2e-11)
    reduced=eliminate_exact(h,p,e);phi=s['state'][p];wave=reduced['embedding']
    np.testing.assert_allclose(wave@phi,s['state'],atol=1e-10)
    assert phi@reduced['norm']@phi==pytest.approx(1.,abs=1e-11)
    assert phi@phi<.99
    step=1e-5
    derivative=(eliminate_exact(h,p,e+step)['hamiltonian']-eliminate_exact(h,p,e-step)['hamiltonian'])/(2*step)
    np.testing.assert_allclose(reduced['norm'],np.eye(len(p))-derivative,atol=2e-8)
    op=source(c,a.yukawa,0).toarray()
    step=1e-6
    source_derivative=(eliminate_exact(h+step*op,p,e)['hamiltonian']-eliminate_exact(h-step*op,p,e)['hamiltonian'])/(2*step)
    np.testing.assert_allclose(source_derivative,wave.T@op@wave,rtol=1e-5,atol=1e-4)
    correct=phi@source_derivative@phi/(phi@reduced['norm']@phi)
    assert correct==pytest.approx(s['state']@op@s['state'],rel=1e-6)
    wrong=phi@op[np.ix_(p,p)]@phi/(phi@phi)
    assert abs(wrong-correct)>.1


@pytest.mark.parametrize('nu',[0.,.2,.5])
def test_eliminated_quantum_response_requires_direct_term_and_both_endpoints(reference,nu):
    a,c,s=reference;ops=[source(c,a.yukawa,k) for k in [0,1]]
    full=connected_response(c,s,ops,nu)['susceptibility']
    eliminated=eliminated_response(c,s,ops,retained(c),nu)
    np.testing.assert_allclose(eliminated['susceptibility'],full,atol=2e-9)
    assert np.linalg.norm(full-eliminated['mediated'])>1


def test_second_source_response_from_independent_reduced_pole_equation(reference):
    a,c,s=reference;h=c.hamiltonian.toarray();op=source(c,a.yukawa,1);p=retained(c);delta=3e-6
    ep=eliminated_ground_energy(h+delta*op.toarray(),p)
    em=eliminated_ground_energy(h-delta*op.toarray(),p)
    e0=eliminated_ground_energy(h,p)
    chi=-(ep+em-2*e0)/(delta**2*c.volume)
    expected=connected_response(c,s,[op])['susceptibility'][0,0]
    assert chi==pytest.approx(expected,rel=2e-5)


def test_adding_eliminated_exchange_to_explicit_retained_modes_double_counts(reference):
    a,c,s=reference;op=source(c,a.yukawa,1);delta=3e-6
    chi=connected_response(c,s,[op])['susceptibility'][0,0]
    w_tree=a.yukawa@np.linalg.solve(a.matrices(a.solve(.004)['vacuum'])[0],a.yukawa)
    identity=sparse.eye(len(s['state']))
    plus=ground_state(c.hamiltonian+delta*op-.5*c.volume*delta**2*w_tree*identity)['energy']
    minus=ground_state(c.hamiltonian-delta*op-.5*c.volume*delta**2*w_tree*identity)['energy']
    wrong=-(plus+minus-2*s['energy'])/(delta**2*c.volume)
    assert wrong==pytest.approx(chi+w_tree,rel=2e-5)
    assert wrong>2*chi


def test_invalid_cell_and_projection_inputs(reference):
    with pytest.raises(ValueError): oscillator_states(-1)
    with pytest.raises(ValueError): polynomial_cell([1,1],[1,1],1,np.zeros((2,4)),volume=0,shell=2)
    with pytest.raises(ValueError): eliminate_exact(np.eye(3),[0,0],0)
    with pytest.raises(ValueError): connected_response(reference[1],reference[2],[],euclidean_frequency=-1)


def test_positive_spectral_response_decreases_with_Euclidean_frequency(reference):
    a,c,s=reference;ops=[source(c,a.yukawa,1),source(c,a.scalar_source,1)]
    chi0=connected_response(c,s,ops,0)['susceptibility']
    chi1=connected_response(c,s,ops,.2)['susceptibility']
    assert np.linalg.eigvalsh(chi1).min()>0
    assert np.linalg.eigvalsh(chi0-chi1).min()>0


def test_small_replay_keeps_quantum_scope_and_elimination_failures_explicit():
    import json
    from scripts.evaluate_retained_mode_quantum import evaluate
    result=evaluate(full_resolution=False);json.dumps(result,allow_nan=False)
    assert len(result['quantum_cases'])==4
    for case in result['quantum_cases']:
        assert case['source_PCAC_Ward_error_GeV2']<1e-8
    elim=result['exact_elimination']
    assert elim['reconstructed_state_error']<1e-10
    assert abs(elim['full_energy_GeV']-elim['independent_reduced_energy_GeV'])<1e-10
    assert abs(elim['bare_projected_source_means'][0]-elim['full_source_means'][0])>.1
    assert sum(elim['small_cell_node_schmidt_probabilities'])==pytest.approx(1.)
    assert elim['small_cell_node_schmidt_probabilities'][0]<.999

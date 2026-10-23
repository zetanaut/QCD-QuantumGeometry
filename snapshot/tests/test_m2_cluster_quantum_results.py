"""Independently recheck preserved quantum evidence; no source-current rebuild."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from deuteron_wigner.quantum.m2_coupled_pilot import symmetry_basis

from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit
from deuteron_wigner.quantum.m2_cluster_quantum_readout import controlled_transition_readout

ROOT=Path(__file__).resolve().parents[1]/'docs/next_level'
CASES=('baseline','Q5_shell1','Q5_shell2','Gaussian_reference12','reference12_degree2',
       'p_minus','n_plus','n_minus','p_flip','n_flip','zero_transfer')


def saved(case):
    info=json.loads((ROOT/f'm2_cluster_quantum_{case}.json').read_text())
    with np.load(ROOT/info['full_matrix_file'],allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    return info,arrays


@pytest.mark.parametrize('case',CASES)
def test_saved_every_order_measurement_norm_energy_and_state_bound(case):
    info,a=saved(case)
    assert not info['physical'] and not info['quantum_advantage'] and info['shots'] is None
    assert not info['exact_state_seed'] and not info['controlled_readout_stateprep']
    assert info['oracle_executions']>0 and info['controlled_readout_executions']>0
    states=[];whiteners=[];reference=[];bounds=[]
    for label,record in zip(('out','in'),info['endpoints']):
        n=a[label+'_N'];h=a[label+'_H'];e=info['endpoint_current_metadata']['energy_'+label+'_GeV2']
        val,u=np.linalg.eigh(n);assert min(val)>0
        w=(u/np.sqrt(val))@u.conj().T;whiteners.append(w)
        linearized=w@(h+e*(n-np.eye(len(n))))@w
        np.testing.assert_allclose(a[label+'_L'],linearized,atol=4e-14)
        state=a[label+'_state'][:len(n)]
        assert np.linalg.norm(a[label+'_state'][len(n):])==0
        np.testing.assert_allclose(np.vdot(state,state),1,atol=2e-14)
        rayleigh=float(np.vdot(state,linearized@state).real)
        residual=float(np.linalg.norm(linearized@state-rayleigh*state))
        assert residual<info['residual_tolerance_GeV2']
        np.testing.assert_allclose(residual,record['residual_GeV2'],atol=2e-14,rtol=0)
        values,vectors=np.linalg.eigh(linearized)
        ground=vectors[:,0];overlap=np.vdot(ground,state);ground*=overlap/abs(overlap)
        gap=values[1]-rayleigh;assert gap>0
        bound=np.sqrt(2)*residual/gap
        assert np.linalg.norm(state-ground)<=bound+5e-13
        states.append(state);reference.append(ground);bounds.append(bound)
        np.testing.assert_allclose(a[label+'_reference'],ground,atol=3e-14)
        # The raw-coordinate bases are preserved, not inferred from D=37.
        for j,(sector,hg) in enumerate((('qqq',0),('qqqg',-1),('qqqg',1))):
            prefix=f'{label}_basis_{j}_'
            basis=sparse.csr_matrix((a[prefix+'data'],a[prefix+'indices'],a[prefix+'indptr']),
                                    shape=tuple(a[prefix+'shape']))
            meta=info['endpoint_current_metadata']
            expected=symmetry_basis(sector,meta['total_k2'],meta['retained_cutoff'],
                meta['target'],meta['jz_'+label+'_twice'],hg)
            np.testing.assert_array_equal(basis.data,expected.data)
            np.testing.assert_array_equal(basis.indices,expected.indices)
            np.testing.assert_array_equal(basis.indptr,expected.indptr)
    operators={}
    for power in range(5):
        for j,mu in enumerate(('plus','minus','x','y')):
            operators[f'g{power}_{mu}']=whiteners[0]@a['current_coefficients'][power,j]@whiteners[1]
    for j,mu in enumerate(('plus','minus','x','y')):
        total=sum(info['model_parameters']['coupling']**k*operators[f'g{k}_{mu}'] for k in range(5))
        np.testing.assert_allclose(total,a['normalized_current'][j],atol=3e-14)
        operators['total_'+mu]=total
    assert set(operators)==set(info['measurements']) and len(operators)==24
    for name,operator in operators.items():
        record=info['measurements'][name];measured=complex(*record['measured'])
        direct=np.vdot(states[0],operator@states[1])
        exact=np.vdot(reference[0],operator@reference[1])
        np.testing.assert_allclose(measured,direct,atol=record['roundoff_budget'],rtol=0)
        bound=np.linalg.norm(operator,2)*sum(bounds)
        assert abs(measured-exact)<=bound+record['roundoff_budget']
        np.testing.assert_allclose(record['residual_gap_bound'],bound,atol=2e-13,rtol=2e-6)


@pytest.mark.parametrize('case',CASES)
def test_saved_parameters_reexecute_same_coherent_gate_transition(case):
    info,a=saved(case)
    circuits=[]
    for label,record in zip(('out','in'),info['endpoints']):
        c=SymmetryGraphCircuit(a[label+'_L'],root=record['root'],support_threshold=record['support_threshold'])
        actual=c.state(a[label+'_parameters'])
        np.testing.assert_allclose(actual,a[label+'_state'],atol=4e-14)
        circuits.append(c)
    whiteners=[]
    for label in ('out','in'):
        val,u=np.linalg.eigh(a[label+'_N'])
        whiteners.append((u/np.sqrt(val))@u.conj().T)
    operators={f'g{k}_{mu}':whiteners[0]@a['current_coefficients'][k,j]@whiteners[1]
        for k in range(5) for j,mu in enumerate(('plus','minus','x','y'))}
    operators.update({'total_'+mu:a['normalized_current'][j]
        for j,mu in enumerate(('plus','minus','x','y'))})
    measured=controlled_transition_readout(circuits[0],a['out_parameters'],circuits[1],a['in_parameters'],operators)
    np.testing.assert_allclose(measured['state'],a['executed_flagged_state'],atol=5e-14)
    for name,value in measured['transitions'].items():
        np.testing.assert_allclose(value,complex(*info['measurements'][name]['measured']),atol=8e-13,rtol=0)


@pytest.mark.parametrize('case',('p_flip','n_flip'))
def test_independent_reverse_orders_and_executed_gate_quadratures(case):
    info,a=saved(case)
    assert info['independent_reversal']['independent_assembly']
    assert info['independent_reversal']['endpoint_energies_exchanged']
    backward=a['independent_reverse_coefficients']
    np.testing.assert_allclose(backward,a['current_coefficients'].conj().transpose(0,1,3,2),
                               atol=5e-12,rtol=0)
    circuits=[];w=[]
    for label,record in zip(('out','in'),info['endpoints']):
        circuits.append(SymmetryGraphCircuit(a[label+'_L'],root=record['root'],
                                            support_threshold=record['support_threshold']))
        val,u=np.linalg.eigh(a[label+'_N']);w.append((u/np.sqrt(val))@u.conj().T)
    operators={f'g{k}_{mu}':w[1]@backward[k,j]@w[0]
        for k in range(5) for j,mu in enumerate(('plus','minus','x','y'))}
    for mu in ('plus','minus','x','y'):
        operators['total_'+mu]=sum(info['model_parameters']['coupling']**k*operators[f'g{k}_{mu}']
                                   for k in range(5))
    measured=controlled_transition_readout(circuits[1],a['in_parameters'],
        circuits[0],a['out_parameters'],operators)
    for name,value in measured['transitions'].items():
        np.testing.assert_allclose(value,complex(*info['measurements'][name]['measured']).conjugate(),
                                   atol=2e-12,rtol=0)


def test_target_spin_and_zero_transfer_coverage_is_explicit():
    channels={tuple(saved(case)[0]['endpoint_current_metadata'][key]
        for key in ('target','jz_out_twice','jz_in_twice')) for case in CASES}
    assert channels=={(target,f,i) for target in ('PROTON','NEUTRON')
                      for f,i in ((1,1),(-1,-1),(-1,1))}
    zero=saved('zero_transfer')[0]
    assert zero['endpoint_current_metadata']['transfer_GeV']==[0.,0.]
    assert max(zero['Ward_coefficient_Frobenius'])<5e-12
    assert zero['all_24_measurements_within_bounds']


def test_quantum_member_changes_are_not_labeled_cutoff_convergence():
    reports=[saved(case)[0] for case in CASES]
    assert len({r['model_parameters']['regulator']['width_GeV'] for r in reports})==2
    assert any(r['model_parameters']['regulator']['reference_shape'] is not None for r in reports)
    assert {r['model_parameters']['excluded_cutoffs']['qqqgg'] for r in reports}=={0,1,2}
    assert all(not r['physical'] for r in reports)

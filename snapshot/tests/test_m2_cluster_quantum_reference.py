"""Independent quantum-encoding checks on the saved common-regulated source."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_cluster_quantum_reference import (
    load_saved_cluster_reference, bind_cluster_reference, COMPONENTS)
from deuteron_wigner.quantum.m2_cluster_quantum_readout import (
    ClusterCurrentOracle, controlled_transition_readout, endpoint_error_certificate)
from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'docs/next_level/m2_gaussian_cluster_four_current_baseline.json'
LOAD = dict(charge='electromagnetic',loop_flavor_charges=(2/3,-1/3),
            mean_total_transverse_GeV=(0.,0.),quadrature_order=8)


@pytest.fixture(scope='module')
def reference():
    return load_saved_cluster_reference(REPORT,**LOAD)


def binding(reference, **changes):
    info=reference.metadata
    values=dict(transfer_GeV=info['transfer_GeV'],
        mean_total_transverse_GeV=info['mean_total_transverse_GeV'],
        charge=info['charge'],loop_flavor_charges=info['loop_flavor_charges'],
        prescription=info['physical_leg_prescription'],quadrature_order=info['quadrature_order'],
        smooth_prescription=info['smooth_prescription'],
        field_envelope_prescription=info['field_envelope_prescription'],
        isospin_prescription=info['isospin_prescription'])
    values.update(changes)
    return values


def test_actual_saved_model_has_both_linearized_hamiltonians_and_full_norms(reference):
    r=reference
    assert r.outgoing.energy_GeV2==1.39 and r.incoming.energy_GeV2==1.47
    assert r.outgoing.owner is r.incoming.owner
    assert not r.physical and not r.coefficients.flags.writeable
    with np.load(REPORT.with_suffix('.npz'),allow_pickle=False) as a:
        np.testing.assert_array_equal(r.raw_current,a['raw_current'])
        for name,s in (('out',r.outgoing),('in',r.incoming)):
            np.testing.assert_allclose(s.effective_hamiltonian,a['H_'+name],atol=3e-14)
            np.testing.assert_allclose(s.metric,a['N_'+name],atol=3e-15)
            ev,u=np.linalg.eigh(a['N_'+name]);w=(u/np.sqrt(ev))@u.conj().T
            direct=w@(a['H_'+name]+s.energy_GeV2*(a['N_'+name]-np.eye(37)))@w
            np.testing.assert_allclose(s.matrix,direct,atol=3e-14)
            assert np.linalg.norm(s.matrix-w@a['H_'+name]@w)>1e-5
        normed=np.stack([r.operators['total_'+mu] for mu in COMPONENTS])
        np.testing.assert_allclose(normed,a['normalized_current'],atol=5e-14)
    assert np.linalg.norm(r.outgoing.metric-r.incoming.metric)>1e-6
    assert max(r.ward_residuals)<1e-13
    for j,mu in enumerate(COMPONENTS):
        for n in range(5):
            np.testing.assert_allclose(r.operators[f'g{n}_{mu}'],
                r.outgoing.whitener@r.coefficients[n,j]@r.incoming.whitener,atol=3e-14)


@pytest.mark.parametrize('key,value',[
    ('prescription',None),('smooth_prescription',None),
    ('field_envelope_prescription',None),('isospin_prescription',None),
    ('loop_flavor_charges',(1.,0.))])
def test_no_prescription_or_charge_default(reference,key,value):
    with pytest.raises(ValueError):
        bind_cluster_reference(reference.outgoing,reference.incoming,reference.coefficients,
                               **binding(reference,**{key:value}))


@pytest.mark.parametrize('fault',['norm','hamiltonian','coefficient','dimension','nonfinite'])
def test_bad_endpoint_or_operator_is_not_a_valid_reference(reference,fault):
    f=reference.outgoing;c=reference.coefficients.copy()
    if fault=='norm':f=replace(f,metric=-f.metric)
    if fault=='hamiltonian':f=replace(f,matrix=f.matrix+.01*np.eye(37))
    if fault=='coefficient':c[4,2,0,0]+=.01
    if fault=='dimension':c=c[:4]
    if fault=='nonfinite':c[0,1,0,0]=np.nan
    with pytest.raises(ValueError):
        bind_cluster_reference(f,reference.incoming,c,**binding(reference))


@pytest.fixture(scope='module')
def oracle(reference):
    pytest.importorskip('pennylane')
    return ClusterCurrentOracle(reference)


def vectors(n):
    rng=np.random.default_rng(708)
    x=rng.normal(size=(2,n))+1j*rng.normal(size=(2,n))
    return x/np.linalg.norm(x,axis=1)[:,None]


@pytest.mark.parametrize('pair',['random','same','phase','orthogonal'])
def test_flagged_complex_endpoint_transitions(oracle,pair):
    import pennylane as qml
    f,i=vectors(oracle.df)
    if pair=='same':f=i
    if pair=='phase':f=1j*i
    if pair=='orthogonal':
        f=f-i*np.vdot(i,f);f/=np.linalg.norm(f)
    with qml.Tracker(oracle.device) as tracker:
        measured=oracle.transitions(f,i)
    assert tracker.totals['executions']>0
    assert oracle.endpoint_qubits==6 and oracle.qubits==7
    for name,a in oracle.reference.operators.items():
        np.testing.assert_allclose(measured[name],np.vdot(f,a@i),rtol=0,atol=8e-13)


def test_every_entry_of_twenty_coefficients_and_four_totals_is_measured(oracle):
    # Complete input basis, not just one state or diagonal expectations.
    for k in range(oracle.di):
        ket=np.eye(1,oracle.di,k=k,dtype=complex)[0]
        measured=oracle.actions(ket)
        for name,a in oracle.reference.operators.items():
            np.testing.assert_allclose(measured[name],a[:,k],rtol=0,atol=8e-13)


@pytest.mark.parametrize('flag',[0,1])
def test_padding_penalty_and_current_isolation(oracle,flag):
    state=np.zeros(2*oracle.size,complex)
    state[flag*oracle.size+oracle.df]=1
    measured=oracle.measure_padded(state)
    assert measured['padding_probability']==pytest.approx(1,abs=2e-15)
    assert measured['hamiltonian']==pytest.approx(oracle.padding_penalty,abs=2e-13)
    assert all(z==0 for z in measured['transitions'].values())
    for s in (oracle.reference.outgoing,oracle.reference.incoming):
        assert np.linalg.eigvalsh(s.matrix)[-1]<oracle.padding_penalty


@pytest.mark.parametrize('bad',[np.zeros(37),np.ones(36),np.ones(37),np.full(37,np.nan)])
def test_oracle_never_normalizes_or_truncates_invalid_states(oracle,bad):
    with pytest.raises(ValueError):oracle.transitions(bad,vectors(37)[0])


def test_actual_controlled_gate_readout_preserves_two_distinct_states(reference):
    circuits=[SymmetryGraphCircuit(s.matrix,root=int(np.argmin(s.matrix.diagonal().real)),
        support_threshold=1e-12) for s in (reference.outgoing,reference.incoming)]
    rng=np.random.default_rng(48)
    theta=[c.uniform_parameters()+rng.normal(0,.1,c.parameter_count) for c in circuits]
    a,b=[np.asarray(c.state(x)) for c,x in zip(circuits,theta)]
    result=controlled_transition_readout(circuits[0],theta[0],circuits[1],theta[1],reference.operators)
    np.testing.assert_allclose(result['state'],np.r_[a,b]/np.sqrt(2),atol=4e-14)
    assert result['executions']>=2 and not result['stateprep_used']
    assert result['padding_probability']<1e-26
    for name,op in reference.operators.items():
        np.testing.assert_allclose(result['transitions'][name],np.vdot(a[:37],op@b[:37]),atol=8e-13,rtol=0)


def test_two_endpoint_phase_aligned_spectral_error_bound():
    rng=np.random.default_rng(952)
    certificates=[]
    for k in range(2):
        z=rng.normal(size=(5,5))+1j*rng.normal(size=(5,5));h=z+z.conj().T
        e,u=np.linalg.eigh(h)
        state=(u[:,0]+.001j*u[:,1])*np.exp(.4j+k*.3j);state/=np.linalg.norm(state)
        cert=endpoint_error_certificate(h,state)
        assert cert['phase_aligned_distance']<=cert['state_distance_bound']+1e-13
        certificates.append((state,cert))
    a=rng.normal(size=(5,5))+1j*rng.normal(size=(5,5))
    (vf,f),(vi,i)=certificates
    error=abs(np.vdot(vf,a@vi)-np.vdot(f['reference'],a@i['reference']))
    assert error<=np.linalg.norm(a,2)*(f['state_distance_bound']+i['state_distance_bound'])
    with pytest.raises(ValueError,match='isolated'):endpoint_error_certificate(np.eye(5),vf)
    with pytest.raises(ValueError,match='separation'):
        endpoint_error_certificate(np.diag([1.,2.,3.]),[0,0,1])


@pytest.mark.parametrize('coupling',[0.,.001,-.001])
def test_zero_and_small_coupling_recombination_keeps_new_endpoint_metrics(reference,coupling):
    from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import build_cluster_pilot,GaussianClusterH2
    p=reference.incoming.owner.pilot
    new=build_cluster_pilot(total_k2=p.total_k2,cutoff=p.cutoff,target=p.target,jz_twice=p.jz_twice,
        b_GeV=p.width_GeV,quark_mass_GeV=p.quark_mass_GeV,gluon_mass_GeV=p.gluon_mass_GeV,
        coupling=coupling,regulator=p.field_regulator,total_pplus_GeV=p.total_pplus_GeV,
        wick_operators=p.wick_operators)
    model=GaussianClusterH2(new,excluded_cutoff=0,six_cutoff=0)
    r=bind_cluster_reference(model.snapshot(1.39),model.snapshot(1.47),reference.coefficients,
                             **binding(reference))
    # A coupling recombination test of the SAME source coefficients, not an
    # independently rebuilt physical current or replacement finite conditions.
    np.testing.assert_allclose(r.raw_current,sum(coupling**n*x for n,x in enumerate(r.coefficients)),atol=0,rtol=0)
    if coupling==0:
        np.testing.assert_array_equal(r.incoming.metric,np.eye(37))
        np.testing.assert_array_equal(r.raw_current,r.coefficients[0])
    else:
        assert np.linalg.norm(r.incoming.metric-np.eye(37))>0
        assert np.linalg.norm(r.incoming.metric-reference.incoming.metric)>1e-4
    oracle=ClusterCurrentOracle(r)
    f,i=vectors(37)
    measured=oracle.transitions(f,i)
    for name,a in r.operators.items():
        np.testing.assert_allclose(measured[name],np.vdot(f,a@i),rtol=0,atol=8e-13)


def test_rectangular_independent_coordinate_frames_in_actual_controlled_readout():
    rng=np.random.default_rng(1302)
    circuits=[];params=[];states=[]
    for dimension in (3,4):
        h=rng.normal(size=(dimension,dimension))+1j*rng.normal(size=(dimension,dimension))
        h=h+h.conj().T
        c=SymmetryGraphCircuit(h,root=1,support_threshold=0.)
        x=c.uniform_parameters()+rng.normal(0,.1,c.parameter_count)
        circuits.append(c);params.append(x);states.append(np.array(c.state(x)))
    a=rng.normal(size=(3,4))+1j*rng.normal(size=(3,4))
    result=controlled_transition_readout(circuits[0],params[0],circuits[1],params[1],{'rectangular':a})
    np.testing.assert_allclose(result['transitions']['rectangular'],np.vdot(states[0][:3],a@states[1]),atol=2e-14)
    assert result['readout_qubits']==3 and result['padding_probability']<1e-28


@pytest.mark.parametrize('key,value',[
    ('charge','baryon'),('loop_flavor_charges',(1.,0.)),
    ('mean_total_transverse_GeV',(.1,0)),('quadrature_order',4)])
def test_saved_export_refuses_unrecorded_prescription_override(key,value):
    with pytest.raises(ValueError):load_saved_cluster_reference(REPORT,**{**LOAD,key:value})


def test_saved_export_reports_cannot_redirect_array_path(tmp_path):
    info=json.loads(REPORT.read_text());info['full_matrix_file']='../elsewhere.npz'
    path=tmp_path/'report.json';path.write_text(json.dumps(info))
    with pytest.raises(ValueError,match='basename'):
        load_saved_cluster_reference(path,**LOAD)

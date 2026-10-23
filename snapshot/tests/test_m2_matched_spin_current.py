import numpy as np
import pytest

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source
from deuteron_wigner.quantum.m2_excluded_fock_matching import ExcludedFockReference
from deuteron_wigner.quantum.m2_spin_resolved_current import (
    parity_aligned_partner, excluded_parent_density_between, matched_good_current_between,
    good_form_factor_diagnostic, normalized_matched_good_current_between,
)


@pytest.fixture(scope="module")
def refs():
    pilots=[build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=j,b_GeV=.4,
                               quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3) for j in (1,-1)]
    parents=[ExcludedFockReference(p,[build_excluded_source(p,s,out_cutoff=1)
                                    for s in ("qqqgg","qqqq-qbar")]) for p in pilots]
    state=parents[0].reduction.lowest_state()
    vectors=[state.retained,parity_aligned_partner(pilots[1],pilots[0],state.retained,intrinsic_parity=1)]
    return parents,vectors,state.energy_GeV2


@pytest.mark.parametrize("charge",["baryon","electromagnetic","up_number","down_number"])
def test_matched_both_helicities_charge_and_existing_diagonal_api(refs,charge):
    parents,vectors,energy=refs
    q=dict(baryon=1.,electromagnetic=1.,up_number=2.,down_number=1.)[charge]
    matrix=np.array([[vf.conj()@matched_good_current_between(pf,pi,energy,energy,(0.,0.),charge=charge)@vi
                      for pi,vi in zip(parents,vectors)] for pf,vf in zip(parents,vectors)])
    np.testing.assert_allclose(matrix,q*np.eye(2),atol=3e-14)
    for ref in parents:
        actual=excluded_parent_density_between(ref,ref,(.12,.16),charge=charge)
        np.testing.assert_allclose(actual,ref.density((.12,.16),charge=charge),atol=2e-14)


def test_matched_spin_current_two_energy_reversal_and_reconstruction(refs):
    (up,down),vectors,energy=refs
    op=excluded_parent_density_between(down,up,(.12,.16),charge="electromagnetic")
    rev=excluded_parent_density_between(up,down,(-.12,-.16),charge="electromagnetic")
    np.testing.assert_allclose(op,rev.conj().T,atol=2e-14)
    out,inc=1.46,1.32
    matched=matched_good_current_between(down,up,out,inc,(.12,.16),charge="electromagnetic")
    reverse=matched_good_current_between(up,down,inc,out,(-.12,-.16),charge="electromagnetic")
    np.testing.assert_allclose(matched,reverse.conj().T,atol=2e-14)
    np.testing.assert_allclose(vectors[1].conj()@matched@vectors[0],
        (down.reduction.wave_operator(out)@vectors[1]).conj()@op
        @(up.reduction.wave_operator(inc)@vectors[0]),atol=2e-14)
    full=down.reduction.wave_operator(energy)@vectors[1]
    assert np.linalg.norm(down.matrix@full-energy*full)<2e-14


def test_excluded_current_induces_pauli_signal_absent_from_retained_zero_shell(refs):
    parents,vectors,e=refs;delta=(.12,.16)
    matched=np.array([[vf.conj()@matched_good_current_between(pf,pi,e,e,delta,charge="electromagnetic")@vi
                       for pi,vi in zip(parents,vectors)] for pf,vf in zip(parents,vectors)])
    bare=good_current_density_between(parents[1].pilot,parents[0].pilot,delta,charge="electromagnetic")
    assert np.linalg.norm(bare)==0.
    fit=good_form_factor_diagnostic(matched,delta,mass_GeV=np.sqrt(e),total_pplus=3.)
    assert np.linalg.norm(fit.residual_matrix)<4e-14
    assert abs(fit.pauli.real)>1e-9
    assert abs(fit.pauli.imag)<2e-14


def test_actual_seven_qubit_matched_helicity_interference(refs):
    import pennylane as qml
    from scipy import sparse
    from deuteron_wigner.quantum.m2_matched_h2_reduction import _inverse_sqrt
    parents,vectors,e=refs;delta=np.array([.12,.16]);q=np.linalg.norm(delta)
    states=[np.linalg.solve(_inverse_sqrt(r.reduction.norm_metric(e)),v) for r,v in zip(parents,vectors)]
    rho=np.block([[normalized_matched_good_current_between(rf,ri,e,e,delta,charge="electromagnetic")
                   for ri in parents] for rf in parents])
    op=(rho-rho.conj().T)/(2j)
    wave=np.r_[states[0],(delta[1]-1j*delta[0])/q*states[1]]/np.sqrt(2)
    assert len(wave)==74
    padded=np.pad(wave,(0,128-len(wave)))
    full=sparse.block_diag((sparse.csr_matrix(op),sparse.csr_matrix((54,54))),format="csr")
    device=qml.device("lightning.qubit",wires=7)
    @qml.qnode(device)
    def circuit():
        qml.StatePrep(padded,wires=range(7))
        return qml.expval(qml.SparseHamiltonian(full,wires=range(7)))
    value=float(circuit())
    assert value==pytest.approx(np.vdot(wave,op@wave).real,abs=3e-14)
    assert abs(value)>1e-10
    for r,v in zip(parents,states):
        assert np.linalg.norm(r.reduction.linearized_model(e).matrix@v-e*v)<3e-14
    charge=np.block([[normalized_matched_good_current_between(rf,ri,e,e,(0.,0.),charge="baryon")
                      for ri in parents] for rf in parents])
    np.testing.assert_allclose(charge,np.eye(74),atol=3e-14)

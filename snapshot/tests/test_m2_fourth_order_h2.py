import numpy as np
import pytest

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_canonical_qq import CanonicalQQSourceAction,CanonicalQQInsertion
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_fourth_order_h2 import FourthOrderH2,current_between
from deuteron_wigner.quantum.m2_spin_resolved_current import transverse_parity_map


def build(jz=1,g=.3):
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=jz,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=g)
    r=build_order_resolved_reference(p,out_cutoff=0)
    qq=CanonicalQQInsertion(r,action=CanonicalQQSourceAction(r))
    six=CanonicalSixBodyExcursion(r,out_cutoff=0)
    return FourthOrderH2(r,QQ_insertion=qq,six_body=six)


@pytest.fixture(scope="module")
def model():return build()


@pytest.fixture(scope="module")
def snap(model):return model.snapshot(1.4)


def test_coherent_H_norm_against_separate_insertion_implementations(model,snap):
    g=model.pilot.coupling;e=1.4
    h=model.pilot.matrix+sum(g**n*model.reference.return_coefficient(n,e) for n in (2,3,4))
    h+=g**4*(model.qq.return_power_coefficient(4,e)+model.six.hamiltonian_coefficient(e))
    n=np.eye(len(h))+sum(g**p*model.reference.return_coefficient(p,e,norm=True) for p in (2,3,4))
    n+=g**4*(model.qq.return_power_coefficient(4,e,norm=True)+model.six.norm_coefficient(e))
    np.testing.assert_allclose(snap.effective_hamiltonian,h,atol=1e-17)
    np.testing.assert_allclose(snap.metric,n,atol=3e-16)
    np.testing.assert_allclose(snap.whitener@snap.metric@snap.whitener,np.eye(len(h)),atol=5e-15)
    np.testing.assert_allclose(snap.matrix,snap.whitener@(h+e*(n-np.eye(len(h))))@snap.whitener,atol=2e-15)


def test_independent_snapshot_energy_derivative(model,snap):
    eps=2e-5
    derivative=(model.snapshot(1.4+eps).effective_hamiltonian-model.snapshot(1.4-eps).effective_hamiltonian)/(2*eps)
    np.testing.assert_allclose(np.eye(len(derivative))-snap.metric,derivative,atol=3e-11)


@pytest.mark.parametrize("charge,q",[("baryon",1),("electromagnetic",1),("up_number",2),("down_number",1)])
def test_full_current_zero_transfer_charge(snap,charge,q):
    np.testing.assert_allclose(snap.current((0,0),charge=charge),q*np.eye(len(snap.matrix)),atol=7e-15)


def test_finite_transfer_current_against_separate_matched_sources(model,snap):
    g=model.pilot.coupling;delta=(.12,.16);e=1.4
    expected=model.reference.current_coefficient(0,e,e,delta,charge="electromagnetic")
    expected+=sum(g**n*model.reference.current_coefficient(n,e,e,delta,charge="electromagnetic") for n in (2,3,4))
    expected+=g**4*(model.qq.current_power_coefficient(4,e,e,delta,charge="electromagnetic")
                   +model.six.current_coefficient(e,e,delta,charge="electromagnetic"))
    np.testing.assert_allclose(snap.current(delta,charge="electromagnetic"),snap.whitener@expected@snap.whitener,atol=3e-16)


def test_parity_covariance_and_two_energy_opposite_spin_current(model,snap):
    down=build(-1);sf=down.snapshot(1.1);same=down.snapshot(1.4)
    t=transverse_parity_map(down.pilot,model.pilot)
    np.testing.assert_allclose(same.effective_hamiltonian,t@snap.effective_hamiltonian@t.conj().T,atol=2e-14)
    np.testing.assert_allclose(same.metric,t@snap.metric@t.conj().T,atol=1e-14)
    x=current_between(sf,snap,(.12,.16),charge="electromagnetic",normalized=True)
    y=current_between(snap,sf,(-.12,-.16),charge="electromagnetic",normalized=True)
    np.testing.assert_allclose(x,y.conj().T,atol=5e-17)
    # At cutoff0 no orbital can carry this one-unit Jz change.
    np.testing.assert_allclose(x,0.,atol=0.)


def test_exact_zero_coupling_control_and_explicit_scope():
    m=build(g=0.);s=m.snapshot(1.4)
    np.testing.assert_allclose(s.effective_hamiltonian,m.pilot.matrix,atol=0.)
    np.testing.assert_allclose(s.metric,np.eye(len(s.metric)),atol=0.)
    assert not m.physical and not m.common_regulator_renormalization_complete


def test_reject_different_source_reference(model):
    other=build()
    with pytest.raises(ValueError):
        FourthOrderH2(model.reference,QQ_insertion=model.qq,six_body=other.six)


def test_near_identity_whitening_does_not_reconstruct_identity_from_eigenvectors():
    from deuteron_wigner.quantum.m2_fourth_order_h2 import _norm_inverse_sqrt
    rng=np.random.default_rng(836)
    u=np.linalg.qr(rng.normal(size=(37,37))+1j*rng.normal(size=(37,37)))[0]
    delta=np.r_[np.zeros(12),np.linspace(1e-12,1e-3,25)]
    metric=np.eye(37)+(u*delta)@u.conj().T
    w=_norm_inverse_sqrt(metric)
    error=w@metric@w-np.eye(37)
    # Diagonal matrix products accumulate a few ulps at unit scale;
    # off-diagonal reconstruction must retain the small-correction accuracy.
    assert np.max(abs(np.diag(error)))<8*np.finfo(float).eps
    np.testing.assert_allclose(error-np.diag(np.diag(error)),0.,atol=8e-16)
    np.testing.assert_allclose(w,w.conj().T,atol=1e-18)


def test_nonpositive_truncated_metric_is_not_clipped():
    from deuteron_wigner.quantum.m2_fourth_order_h2 import _norm_inverse_sqrt
    for value in (0.,-1e-6):
        with pytest.raises(ArithmeticError):_norm_inverse_sqrt(np.diag([1.,value]))

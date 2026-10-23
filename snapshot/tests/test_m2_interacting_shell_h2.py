import numpy as np
import pytest

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_shell_canonical_qq import FactorizedCanonicalQQSourceAction,ShellCanonicalQQInsertion
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_fourth_order_h2 import FourthOrderH2


@pytest.fixture(scope="module")
def model():
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    r=build_shell_fock_reference(p,out_cutoff=2)
    qq=ShellCanonicalQQInsertion(r,action=FactorizedCanonicalQQSourceAction(r))
    six=CanonicalSixBodyExcursion(r,out_cutoff=2)
    return FourthOrderH2(r,QQ_insertion=qq,six_body=six)


@pytest.fixture(scope="module")
def snapshot(model):return model.snapshot(1.4)


def test_second_shell_combined_H_norm_and_whitening(model,snapshot):
    e=1.4;g=model.pilot.coupling;r=model.reference
    h=model.pilot.matrix+sum(g**p*r.return_coefficient(p,e) for p in (2,3,4))
    h+=g**4*(model.qq.return_power_coefficient(4,e)+model.six.hamiltonian_coefficient(e))
    norm=np.eye(len(h))+sum(g**p*r.return_coefficient(p,e,norm=True) for p in (2,3,4))
    norm+=g**4*(model.qq.return_power_coefficient(4,e,norm=True)+model.six.norm_coefficient(e))
    np.testing.assert_allclose(h,snapshot.effective_hamiltonian,atol=3e-17)
    np.testing.assert_allclose(norm,snapshot.metric,atol=4e-16)
    np.testing.assert_allclose(snapshot.whitener@norm@snapshot.whitener,np.eye(len(h)),atol=5e-15)
    assert np.linalg.eigvalsh(norm)[0]>0


def test_second_shell_combined_norm_is_the_energy_derivative(model,snapshot):
    eps=2e-5
    derivative=(model.snapshot(1.4+eps).effective_hamiltonian
                -model.snapshot(1.4-eps).effective_hamiltonian)/(2*eps)
    np.testing.assert_allclose(np.eye(len(derivative))-snapshot.metric,derivative,atol=3e-11)


def test_second_shell_current_from_same_H2_matches_separate_finite_transfer_sources(model,snapshot):
    e=1.4;g=model.pilot.coupling;delta=(.12,.16);r=model.reference
    np.testing.assert_allclose(snapshot.current((0,0),charge="baryon"),np.eye(len(snapshot.matrix)),atol=7e-15)
    rho=r.current_coefficient(0,e,e,delta,charge="electromagnetic")
    rho+=sum(g**p*r.current_coefficient(p,e,e,delta,charge="electromagnetic") for p in (2,3,4))
    rho+=g**4*(model.qq.current_power_coefficient(4,e,e,delta,charge="electromagnetic")
              +model.six.current_coefficient(e,e,delta,charge="electromagnetic"))
    np.testing.assert_allclose(snapshot.current(delta,charge="electromagnetic"),
                               snapshot.whitener@rho@snapshot.whitener,atol=4e-16)
    assert not model.physical and not model.common_regulator_renormalization_complete

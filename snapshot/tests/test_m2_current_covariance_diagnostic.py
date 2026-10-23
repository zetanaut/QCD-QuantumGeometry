import numpy as np
import pytest

from deuteron_wigner.quantum.m2_spin_resolved_current import dirac_pauli_good_structures
from deuteron_wigner.quantum.m2_current_covariance_diagnostic import (
    transverse_dirac_pauli_structures,current_covariance_diagnostic,
)


@pytest.mark.parametrize("mass,P",[(.7,.31),(1.2,1.3),(1.8,3.7)])
@pytest.mark.parametrize("delta",[(.12,.16),(.2,0.)])
def test_Gordon_identity_and_joint_components_resolve_identical_form_factors(mass,P,delta):
    plus=dirac_pauli_good_structures(delta,mass_GeV=mass,total_pplus=P)
    trans=transverse_dirac_pauli_structures(delta,mass_GeV=mass,total_pplus_GeV=P)
    np.testing.assert_allclose(trans[0],trans[1],atol=8e-16)
    for f1,f2 in [(1.,1.7),(0.,-1.2),(.7+.1j,-.2+.04j)]:
        q=current_covariance_diagnostic(f1*plus[0]+f2*plus[1],f1*trans[0]+f2*trans[1],delta,
                                       mass_GeV=mass,total_pplus_GeV=P)
        assert q.F1_from_plus==pytest.approx(f1,abs=3e-15)
        assert q.F2_from_plus==pytest.approx(f2,abs=3e-15)
        assert q.magnetic_from_transverse==pytest.approx(f1+f2,abs=3e-15)
        np.testing.assert_allclose(q.transverse_covariance_residual,0.,atol=2e-15)
        np.testing.assert_allclose(q.elastic_continuity_defect,0.,atol=2e-15)


def test_conservation_does_not_detect_divergence_free_magnetic_mismatch():
    d=(.12,.16);m=1.2;P=1.3
    plus=dirac_pauli_good_structures(d,mass_GeV=m,total_pplus=P)[0]
    trans=transverse_dirac_pauli_structures(d,mass_GeV=m,total_pplus_GeV=P)[0]
    q=current_covariance_diagnostic(plus,3*trans,d,mass_GeV=m,total_pplus_GeV=P)
    np.testing.assert_allclose(q.elastic_continuity_defect,0.,atol=2e-15)
    np.testing.assert_allclose(q.transverse_nonmagnetic_residual,0.,atol=2e-15)
    assert q.magnetic_from_plus==pytest.approx(1.)
    assert q.magnetic_from_transverse==pytest.approx(3.)
    assert np.linalg.norm(q.transverse_covariance_residual)>.1


def test_zero_transfer_cannot_resolve_Pauli_structure():
    with pytest.raises(ValueError):
        transverse_dirac_pauli_structures((0,0),mass_GeV=1.,total_pplus_GeV=1.)


@pytest.mark.parametrize("target,charge,magnetic",[("PROTON",1.,3.),("NEUTRON",0.,-2.)])
def test_unphysical_equal_share_free_control_exposes_analytic_spin_current_mismatch(target,charge,magnetic):
    """This is a negative covariance control, NOT a physical acceptance target."""
    from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
    from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
    from deuteron_wigner.quantum.m2_transverse_current_projection import free_transverse_current_between
    from deuteron_wigner.quantum.m2_spin_resolved_current import transverse_parity_map
    def build(jz):
        return build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,b_GeV=.4,
            quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=0.)
    pilots=[build(1),build(-1)]
    e,u=np.linalg.eigh(pilots[0].matrix)
    v=[u[:,0],1j*transverse_parity_map(pilots[1],pilots[0])@u[:,0]]
    delta=np.array([.12,.16]);P=1.3
    plus=np.zeros((2,2),complex);trans=np.zeros((2,2,2),complex)
    for hf in range(2):
        for hi in range(2):
            plus[hf,hi]=np.vdot(v[hf],good_current_density_between(pilots[hf],pilots[hi],delta,
                                            charge="electromagnetic")@v[hi])
            trans[:,hf,hi]=[np.vdot(v[hf],j@v[hi]) for j in free_transverse_current_between(
                pilots[hf],pilots[hi],delta,total_pplus_GeV=P,charge="electromagnetic")]
    q=current_covariance_diagnostic(plus,trans,delta,mass_GeV=np.sqrt(e[0]),total_pplus_GeV=P)
    density=np.exp(-delta@delta/(2*.4**2))
    assert e[0]==pytest.approx(9*.35**2+2*.4**2)
    assert q.F1_from_plus==pytest.approx(charge*density,abs=3e-15)
    assert q.F2_from_plus==pytest.approx(0.,abs=3e-15)
    assert q.magnetic_from_transverse==pytest.approx(magnetic*density,abs=5e-15)
    np.testing.assert_allclose(q.elastic_continuity_defect,0.,atol=2e-15)
    assert np.linalg.norm(q.transverse_covariance_residual)>.1

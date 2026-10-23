import numpy as np
import pytest

from deuteron_wigner.quantum.m2_orientation_current_diagnostic import (
    orientation_current_structures,orientation_current_diagnostic,
)
from deuteron_wigner.quantum.m2_current_covariance_diagnostic import transverse_dirac_pauli_structures
from deuteron_wigner.quantum.m2_spin_resolved_current import dirac_pauli_good_structures


@pytest.mark.parametrize("m,P",[(.7,.31),(1.2,1.3),(1.8,3.7)])
@pytest.mark.parametrize("delta",[(.12,.16),(.2,0.)])
def test_independent_external_tensors_and_plus_contamination_identity(m,P,delta):
    b=orientation_current_structures(delta,mass_GeV=m,total_pplus_GeV=P)
    plus=dirac_pauli_good_structures(delta,mass_GeV=m,total_pplus=P)
    transverse=transverse_dirac_pauli_structures(delta,mass_GeV=m,total_pplus_GeV=P)
    np.testing.assert_allclose(b[:2,0],plus,atol=8e-16)
    np.testing.assert_allclose(b[:2,2:],transverse,atol=8e-16)
    eta=np.dot(delta,delta)/(4*m*m)
    np.testing.assert_allclose(b[2,0],2*(eta*plus[0]+plus[1])/(1+eta),atol=8e-16)
    np.testing.assert_array_equal(b[2:,2:],0.)
    np.testing.assert_array_equal(b[3:,0],0.)
    assert np.linalg.matrix_rank(b.reshape(5,-1))==5
    assert np.linalg.matrix_rank(b[:,[0,2,3]].reshape(5,-1))==3
    np.testing.assert_allclose(np.einsum("j,kjab->kab",delta,b[:,2:]),0.,atol=2e-15)


@pytest.mark.parametrize("m,P",[(.7,.31),(1.2,1.3),(1.8,3.7)])
def test_plus_transverse_separation_retains_unknown_minus_and_recovers_complex_coefficients(m,P):
    delta=(.12,.16)
    b=orientation_current_structures(delta,mass_GeV=m,total_pplus_GeV=P)
    coefficients=np.array([.7+.02j,1.2-.04j,-.3+.06j,4.,-7.])
    current=np.einsum("k,kmab->mab",coefficients,b)
    q=orientation_current_diagnostic(current[0],current[2:],delta,mass_GeV=m,total_pplus_GeV=P)
    np.testing.assert_allclose([q.F1_separated,q.F2_separated,q.B1],coefficients[:3],atol=4e-15)
    assert q.B2 is None and q.B3 is None
    assert q.unresolved_coefficients==("B2","B3")
    assert not q.full_five_structure_tested and not q.covariance_repaired and not q.physical
    assert q.electric_from_plus==pytest.approx(q.electric_separated,abs=3e-15)
    np.testing.assert_allclose(q.structure_residual,0.,atol=3e-15)
    full=orientation_current_diagnostic(current[0],current[2:],delta,
        mass_GeV=m,total_pplus_GeV=P,minus=current[1])
    np.testing.assert_allclose([full.F1_separated,full.F2_separated,full.B1,full.B2,full.B3],
                               coefficients,atol=5e-13)
    np.testing.assert_allclose(full.structure_residual,0.,atol=5e-13)
    assert full.full_five_structure_tested and not full.unresolved_coefficients
    assert not full.covariance_repaired and not full.physical


def test_out_of_basis_current_is_never_discarded_or_claimed_repaired():
    delta=(.12,.16);m=1.2;P=1.3
    b=orientation_current_structures(delta,mass_GeV=m,total_pplus_GeV=P)
    value=b[0].copy();value[2,0,1]+=.01
    q=orientation_current_diagnostic(value[0],value[2:],delta,mass_GeV=m,total_pplus_GeV=P)
    assert np.linalg.norm(q.structure_residual)>.005
    assert not q.covariance_repaired


@pytest.mark.parametrize("delta,m,P",[((0,0),1.,1.),((.1,0),0.,1.),((np.nan,0),1.,1.)])
def test_invalid_kinematics_are_rejected(delta,m,P):
    with pytest.raises(ValueError):
        orientation_current_structures(delta,mass_GeV=m,total_pplus_GeV=P)


@pytest.mark.parametrize("charge,magnetic",[(1.,3.),(0.,-2.)])
def test_analytic_equal_share_negative_covariance_control(charge,magnetic):
    delta=(.12,.16);m=np.sqrt(9*.35**2+2*.4**2);P=1.3
    b=orientation_current_structures(delta,mass_GeV=m,total_pplus_GeV=P)
    density=np.exp(-np.dot(delta,delta)/(2*.4**2))
    q=orientation_current_diagnostic(charge*density*b[0,0],magnetic*density*b[0,2:],
                                      delta,mass_GeV=m,total_pplus_GeV=P)
    assert q.B1==pytest.approx((charge-magnetic)*density/2,abs=3e-15)
    assert q.electric_separated==pytest.approx(charge*density,abs=3e-15)
    assert q.F1_separated+q.F2_separated==pytest.approx(magnetic*density,abs=3e-15)
    np.testing.assert_allclose(q.structure_residual,0.,atol=3e-15)
    assert not q.physical and not q.covariance_repaired

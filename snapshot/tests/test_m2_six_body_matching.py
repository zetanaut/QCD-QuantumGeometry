import numpy as np
import pytest
from deuteron_wigner.quantum.m2_six_body_matching import SixBodyExcursion


def model(seed=716):
    rng=np.random.default_rng(seed)
    def z(m,n):
        return (rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))/4
    a,b=z(5,5),z(4,4)
    return SixBodyExcursion(H5=3*np.eye(5)+a@a.conj().T,
        H6=5*np.eye(4)+b@b.conj().T,V5P=z(5,3),U65=z(4,5),W6P=z(4,3))


def exact(m,g,e):
    h=np.block([[m.H5,g*m.U.conj().T],[g*m.U,m.H6]])
    v=np.vstack((g*m.V,g*g*m.W))
    wave=np.linalg.solve(e*np.eye(len(h))-h,v)
    return v.conj().T@wave,wave


def test_norm_derivative_and_backflow_is_necessary():
    m=model();e=.7;eps=1e-5
    derivative=(m.hamiltonian_coefficient(e+eps)-m.hamiltonian_coefficient(e-eps))/(2*eps)
    np.testing.assert_allclose(m.norm_coefficient(e),-derivative,atol=3e-11)
    a=m.waves(e)
    assert np.linalg.norm(m.norm_coefficient(e)-a.six_leading.conj().T@a.six_leading)>1e-3
    np.testing.assert_allclose(m.hamiltonian_coefficient(e),m.hamiltonian_coefficient(e).conj().T,atol=1e-15)


@pytest.mark.parametrize("quantity",["hamiltonian","norm","current"])
def test_against_independent_full_block_resolvent(quantity):
    m=model();ei=.7;ef=1.1
    rng=np.random.default_rng(854)
    r5=rng.normal(size=(5,5))+1j*rng.normal(size=(5,5))
    r6=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    r=np.block([[r5,np.zeros((5,4))],[np.zeros((4,5)),r6]])
    ai,af=m.waves(ei).five_leading,m.waves(ef).five_leading
    desired={"hamiltonian":m.hamiltonian_coefficient(ei),
        "norm":m.norm_coefficient(ei),
        "current":m.current_coefficient(ef,ei,rho5=r5,rho6=r6)}[quantity]
    errors=[]
    for g in (.08,.04,.02):
        h,wi=exact(m,g,ei);_,wf=exact(m,g,ef)
        if quantity=="hamiltonian":
            actual=h-g*g*m.V.conj().T@ai
        elif quantity=="norm":
            actual=wi.conj().T@wi-g*g*ai.conj().T@ai
        else:
            actual=wf.conj().T@r@wi-g*g*af.conj().T@r5@ai
        errors.append(np.linalg.norm(actual/g**4-desired))
    assert 3.9<errors[0]/errors[1]<4.1
    assert 3.9<errors[1]/errors[2]<4.1
    assert errors[-1]<5e-5


@pytest.mark.parametrize("charge",[0.,1.,2.,-1.])
def test_charge_norm_identity(charge):
    m=model()
    np.testing.assert_allclose(
        m.current_coefficient(.7,.7,rho5=charge*np.eye(5),rho6=charge*np.eye(4)),
        charge*m.norm_coefficient(.7),atol=3e-16)


def test_two_energy_reversal_and_interference():
    m=model();rng=np.random.default_rng(723)
    a=rng.normal(size=(5,5))+1j*rng.normal(size=(5,5))
    b=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    x=m.current_coefficient(.7,1.1,rho5=a,rho6=b)
    y=m.current_coefficient(1.1,.7,rho5=a.conj().T,rho6=b.conj().T)
    np.testing.assert_allclose(x,y.conj().T,atol=3e-16)
    left=SixBodyExcursion(H5=m.H5,H6=m.H6,V5P=m.V,U65=m.U,W6P=np.zeros_like(m.W))
    right=SixBodyExcursion(H5=m.H5,H6=m.H6,V5P=m.V,U65=np.zeros_like(m.U),W6P=m.W)
    assert np.linalg.norm(m.hamiltonian_coefficient(.7)-left.hamiltonian_coefficient(.7)
                          -right.hamiltonian_coefficient(.7))>1e-3


def test_independent_internal_unitary_coordinates():
    m=model();rng=np.random.default_rng(167)
    a=np.linalg.qr(rng.normal(size=(5,5))+1j*rng.normal(size=(5,5)))[0]
    b=np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0]
    n=SixBodyExcursion(H5=a.conj().T@m.H5@a,H6=b.conj().T@m.H6@b,
        V5P=a.conj().T@m.V,U65=b.conj().T@m.U@a,W6P=b.conj().T@m.W)
    np.testing.assert_allclose(n.hamiltonian_coefficient(.7),m.hamiltonian_coefficient(.7),atol=1e-15)
    np.testing.assert_allclose(n.norm_coefficient(.7),m.norm_coefficient(.7),atol=1e-15)


def test_explicit_missing_input_and_poles_rejected():
    m=model()
    with pytest.raises(TypeError):
        SixBodyExcursion(H5=m.H5,H6=m.H6,V5P=m.V,U65=m.U)
    with pytest.raises(ValueError):
        SixBodyExcursion(H5=m.H5,H6=m.H6,V5P=m.V,U65=m.U,W6P=np.zeros((5,3)))
    with pytest.raises(ValueError):m.waves(np.linalg.eigvalsh(m.H6)[0])
    with pytest.raises(ValueError):m.waves(float("nan"))
    assert not m.physical and not m.canonical_six_body_vertices_supplied

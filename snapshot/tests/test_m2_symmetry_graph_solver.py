import numpy as np
import pytest
from pennylane import numpy as pnp
from scipy import sparse
from deuteron_wigner.quantum.m2_symmetry_graph_solver import (
    SymmetryGraphCircuit,graph_tree,lanczos_ground,
)
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot


@pytest.mark.parametrize("dimension,root",[(3,2),(7,0),(9,8)])
def test_gray_code_complex_rotations_padding_energy_and_adjoint_gradient(dimension,root):
    rng=np.random.default_rng(dimension)
    a=rng.normal(size=(dimension,dimension))+1j*rng.normal(size=(dimension,dimension))
    h=a+a.conj().T+5*np.eye(dimension)
    circuit=SymmetryGraphCircuit(h,root=root,support_threshold=0.)
    theta=pnp.array(rng.uniform(-1,1,circuit.parameter_count),requires_grad=True)
    direct=circuit.classical_state(theta)
    measured=np.array(circuit.state(theta))
    np.testing.assert_allclose(measured[:dimension],direct,atol=2e-14)
    np.testing.assert_allclose(measured[dimension:],0,atol=2e-14)
    assert float(circuit.energy(theta))==pytest.approx(np.vdot(direct,h@direct).real,abs=3e-13)
    assert circuit.padding_penalty>np.linalg.eigvalsh(h)[-1]
    np.testing.assert_allclose(circuit.classical_state(circuit.uniform_parameters()),
                               np.ones(dimension)/np.sqrt(dimension),atol=3e-14)
    gradient=circuit._gradient(theta)
    assert float(circuit.centered_energy(theta))+circuit.optimizer_energy_origin==pytest.approx(
        float(circuit.energy(theta)),abs=5e-13)
    np.testing.assert_allclose(circuit._centered_gradient(theta),gradient,atol=3e-13)
    for k in (0,circuit.parameter_count-1):
        plus=np.array(theta);minus=np.array(theta);plus[k]+=1e-5;minus[k]-=1e-5
        expected=(float(circuit.energy(plus))-float(circuit.energy(minus)))/2e-5
        assert gradient[k]==pytest.approx(expected,abs=3e-9)


def test_no_disconnected_component_is_silently_dropped():
    with pytest.raises(ValueError,match="disconnected"):
        graph_tree(np.diag([1.,2.,3.]),0,support_threshold=0.)


def test_coupled_37_state_quantum_circuit_is_not_stateprep():
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    c=SymmetryGraphCircuit(p.matrix,root=int(np.argmin(p.matrix.diagonal().real)),
                            support_threshold=1e-12)
    theta=np.random.default_rng(912).uniform(-.3,.3,c.parameter_count)
    direct=c.classical_state(theta)
    np.testing.assert_allclose(c.state(theta)[:37],direct,atol=4e-14)
    assert float(c.energy(theta))==pytest.approx(np.vdot(direct,p.matrix@direct).real,abs=4e-13)
    names=[o.name for o in c.state.qtape.operations]
    assert "StatePrep" not in names and "QubitUnitary" not in names
    e,v=lanczos_ground(p.matrix)
    assert np.linalg.norm(p.matrix@v-e*v)<2e-10
    assert e==pytest.approx(np.linalg.eigvalsh(p.matrix)[0],abs=2e-12)


def test_actual_simulator_optimization_small_complex_problem():
    h=np.array([[1.,.3+.2j,-.1j],[.3-.2j,2.,.4], [.1j,.4,3.]])
    c=SymmetryGraphCircuit(h,root=0,support_threshold=0.)
    result=c.fit(np.full(c.parameter_count,.1),max_iterations=160)
    assert result.fun==pytest.approx(np.linalg.eigvalsh(h)[0],abs=2e-9)
    state=c.state(result.x)
    assert np.sum(abs(state[3:])**2)<1e-24


@pytest.mark.parametrize("options",[dict(gradient_tolerance=0.),
                                    dict(function_tolerance=-1.),dict(function_tolerance=np.nan)])
def test_explicit_optimizer_tolerances_are_validated(options):
    c=SymmetryGraphCircuit([[1.,.2],[.2,2.]],root=0,support_threshold=0.)
    with pytest.raises(ValueError,match="tolerances"):
        c.fit(c.uniform_parameters(),**options)


def test_executed_state_residual_not_small_energy_error_is_the_eigenpair_check():
    h=np.array([[1.,.03+.01j,0.],[.03-.01j,2.,.02],[0.,.02,4.]])
    c=SymmetryGraphCircuit(h,root=0,support_threshold=0.)
    theta=c.uniform_parameters()
    result=c.state_diagnostics(theta)
    psi=c.classical_state(theta); energy=np.vdot(psi,h@psi).real
    residual=np.linalg.norm(h@psi-energy*psi)
    assert result["energy_GeV2"]==pytest.approx(energy,abs=3e-14)
    assert result["residual_GeV2"]==pytest.approx(residual,abs=3e-14)
    assert result["variance_GeV4"]==pytest.approx(residual**2,abs=3e-14)
    assert result["norm"]==pytest.approx(1.,abs=3e-14)
    assert result["padding_probability"]<1e-25
    fit=c.fit(theta,max_iterations=300,gradient_tolerance=1e-10,function_tolerance=0.)
    assert c.state_diagnostics(fit.x)["residual_GeV2"]<2e-7

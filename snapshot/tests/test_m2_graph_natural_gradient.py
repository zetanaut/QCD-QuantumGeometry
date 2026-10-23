import numpy as np
import pytest
from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit
from deuteron_wigner.quantum.m2_graph_natural_gradient import (
    graph_state_jacobian,graph_projective_metric,quantum_natural_fit,
)


@pytest.mark.parametrize("dimension",[3,7,9])
def test_tree_jacobian_and_projective_metric_against_executed_complex_states(dimension):
    rng=np.random.default_rng(dimension+32)
    a=rng.normal(size=(dimension,dimension))+1j*rng.normal(size=(dimension,dimension))
    h=a+a.conj().T
    circuit=SymmetryGraphCircuit(h,root=dimension-1,support_threshold=0.)
    theta=circuit.uniform_parameters()+rng.uniform(-.1,.1,circuit.parameter_count)
    psi,jac=graph_state_jacobian(circuit,theta)
    np.testing.assert_allclose(psi,circuit.state(theta)[:dimension],atol=3e-14)
    for k in range(circuit.parameter_count):
        plus,minus=theta.copy(),theta.copy();plus[k]+=1e-5;minus[k]-=1e-5
        derivative=(circuit.state(plus)[:dimension]-circuit.state(minus)[:dimension])/2e-5
        np.testing.assert_allclose(jac[:,k],derivative,atol=8e-11)
    metric=graph_projective_metric(circuit,theta)
    assert np.linalg.eigvalsh(metric)[0]>1e-5
    energy=np.vdot(psi,h@psi).real
    residual=h@psi-energy*psi
    gradient=2*(jac.conj().T@residual).real
    direction=-.5*np.linalg.solve(metric,gradient)
    projected=jac-psi[:,None]*(psi.conj()@jac)[None,:]
    # Complete tree tangent reproduces imaginary-time descent, not an exact-state seed.
    np.testing.assert_allclose(projected@direction,-residual,atol=3e-13)


def test_natural_fit_uses_actual_gates_and_stops_on_executed_state_residual():
    h=np.array([[1.,.3+.2j,-.1j],[.3-.2j,2.,.4],[.1j,.4,3.]])
    circuit=SymmetryGraphCircuit(h,root=0,support_threshold=0.)
    fit=quantum_natural_fit(circuit,circuit.uniform_parameters(),metric_rcond=1e-14,
                           residual_tolerance=1e-7,max_iterations=300)
    assert fit.success,fit.message
    assert fit.diagnostics["residual_GeV2"]<1e-7
    assert fit.fun==pytest.approx(np.linalg.eigvalsh(h)[0],abs=3e-13)
    assert np.max(np.diff(fit.history[:,0]))<3e-15
    circuit.state(fit.x)
    assert all(o.name not in ("StatePrep","QubitUnitary") for o in circuit.state.qtape.operations)


def test_actual_matched_37_state_convergence_and_current_error_bound():
    from scipy import linalg
    from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
    from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source
    from deuteron_wigner.quantum.m2_excluded_fock_matching import ExcludedFockReference
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    ref=ExcludedFockReference(p,[build_excluded_source(p,s,out_cutoff=1)
                                 for s in ("qqqgg","qqqq-qbar")])
    model=ref.reduction.linearized_model(1.4)
    c=SymmetryGraphCircuit(model.matrix,root=int(np.argmin(model.matrix.diagonal().real)),
                           support_threshold=1e-12)
    initial=c.uniform_parameters()
    initial[1::2]=.03*np.random.default_rng(4901).normal(size=len(c.edges))
    fit=quantum_natural_fit(c,initial,metric_rcond=1e-18,residual_tolerance=1e-8,max_iterations=200)
    assert fit.success,fit.message
    assert fit.diagnostics["residual_GeV2"]<1e-8
    # Independent exact reference only AFTER the actual circuit optimization.
    e,v=linalg.eigh(model.matrix)
    rho=model.operator(ref.density((.12,.16),charge="electromagnetic"))
    current=(rho+rho.conj().T)/2
    measured=c.measure_operators(fit.x,[current,np.eye(c.dimension)])
    exact=np.vdot(v[:,0],current@v[:,0]).real
    bound=2*np.linalg.norm(current,2)*fit.diagnostics["residual_GeV2"]/(e[1]-fit.fun)
    assert abs(measured[0]-exact)<=bound+3e-14
    assert measured[1]==pytest.approx(1.,abs=3e-14)
    assert fit.fun==pytest.approx(e[0],abs=3e-14)


@pytest.mark.parametrize("rcond",[0.,1.,np.nan])
def test_natural_metric_cutoff_is_explicit_and_validated(rcond):
    circuit=SymmetryGraphCircuit([[1.,.2],[.2,2.]],root=0,support_threshold=0.)
    with pytest.raises(ValueError,match="controls"):
        quantum_natural_fit(circuit,circuit.uniform_parameters(),metric_rcond=rcond,
                            residual_tolerance=1e-8)

from types import SimpleNamespace

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_tree_projective_geometry import (
    tree_state_weights,tree_state_tangent,tree_parameters_from_state,tree_inverse_metric_action,
)
from deuteron_wigner.quantum.m2_graph_natural_gradient import graph_state_jacobian


def tree(shape):
    edges=tuple((i-1 if shape=="chain" else (0 if shape=="star" else (i-1)//2),i) for i in range(1,11))
    return SimpleNamespace(dimension=11,root=0,edges=edges,parameter_count=20)


@pytest.mark.parametrize("shape",["chain","star","binary"])
def test_linear_storage_tangent_against_dense_derivative_and_finite_difference(shape):
    c=tree(shape);rng=np.random.default_rng(399)
    x=rng.normal(size=c.parameter_count);v=rng.normal(size=x.shape)
    psi,tangent=tree_state_tangent(c,x,v);p,jac=graph_state_jacobian(c,x)
    np.testing.assert_allclose(psi,p,atol=3e-16)
    expected=jac@v-p*np.vdot(p,jac@v)
    np.testing.assert_allclose(tangent,expected,atol=2e-15)
    step=1e-5
    finite=(tree_state_weights(c,x+step*v)[0]-tree_state_weights(c,x-step*v)[0])/(2*step)
    finite-=psi*np.vdot(psi,finite)
    np.testing.assert_allclose(tangent,finite,atol=2e-9)
    assert abs(np.vdot(psi,tangent))<1e-15


@pytest.mark.parametrize("shape",["chain","star","binary"])
@pytest.mark.parametrize("boundary",["interior","zeros","tiny"])
def test_inverse_gate_coordinates_preserve_complex_states_including_boundary(shape,boundary):
    c=tree(shape);rng=np.random.default_rng(294)
    state=rng.normal(size=c.dimension)+1j*rng.normal(size=c.dimension)
    if boundary=="zeros":state[[0,1,4,8]]=0.
    if boundary=="tiny":state[[0,1,4,8]]*=1e-10
    state/=np.linalg.norm(state)
    x=tree_parameters_from_state(c,state)
    actual,_=tree_state_weights(c,x)
    expected=state*np.exp(-1j*np.angle(state[c.root]))
    np.testing.assert_allclose(actual,expected,atol=8e-16)
    if boundary=="tiny":
        assert np.all(abs(actual[[0,1,4,8]])>0)


def test_tree_retraction_is_first_order_natural_direction_without_H_in_implementation():
    c=tree("binary");rng=np.random.default_rng(715)
    state=rng.normal(size=c.dimension)+1j*rng.normal(size=c.dimension);state/=np.linalg.norm(state)
    x=tree_parameters_from_state(c,state);psi,jac=graph_state_jacobian(c,x)
    a=rng.normal(size=(c.dimension,c.dimension))+1j*rng.normal(size=(c.dimension,c.dimension))
    h=a+a.conj().T;residual=h@psi-np.vdot(psi,h@psi).real*psi
    gradient=2*(jac.conj().T@residual).real
    direction=-.5*tree_inverse_metric_action(c,x,gradient,probability_regularization=0.)
    _,tangent=tree_state_tangent(c,x,direction)
    np.testing.assert_allclose(tangent,-residual,atol=2e-14)
    step=1e-5
    candidate=tree_parameters_from_state(c,psi+step*tangent)
    actual,_=tree_state_weights(c,candidate)
    actual*=np.exp(-1j*np.angle(np.vdot(psi,actual)))
    np.testing.assert_allclose((actual-psi)/step,tangent,atol=8e-4)


@pytest.mark.parametrize("epsilon",[0.,1e-12])
def test_actual_quantum_retraction_keeps_gates_and_uses_executed_gradient(monkeypatch,epsilon):
    import deuteron_wigner.quantum.m2_graph_natural_gradient as natural
    from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit
    h=np.array([[1.,.3+.2j,-.1j],[.3-.2j,2.,.4],[.1j,.4,3.]])
    c=SymmetryGraphCircuit(h,root=0,support_threshold=0.)
    counts=[0];old=c._centered_gradient
    def gradient(x):counts[0]+=1;return old(x)
    c._centered_gradient=gradient
    def forbidden(*args,**kw):raise AssertionError("dense Jacobian/eigenstate forbidden during optimization")
    monkeypatch.setattr(natural,"graph_state_jacobian",forbidden)
    with monkeypatch.context() as patch:
        for name in ("eigh","eigvalsh","svd"):patch.setattr(np.linalg,name,forbidden)
        progress=[]
        fit=natural.quantum_natural_fit(c,c.uniform_parameters(),geometry="tree",metric_rcond=None,
            probability_regularization=epsilon,residual_tolerance=1e-8,max_iterations=300,
            step_rule="tree_state_retraction",progress=progress.append)
    assert fit.success,fit.message
    assert counts[0]==len(progress)==fit.nit+1
    assert fit.fun==pytest.approx(np.linalg.eigvalsh(h)[0],abs=3e-13)
    assert fit.diagnostics["padding_probability"]<3e-28
    assert fit.step_rule=="tree_state_retraction"
    c.state(fit.x)
    assert all(o.name not in ("StatePrep","QubitUnitary") for o in c.state.qtape.operations)


def test_large_retraction_path_uses_no_dense_geometry(monkeypatch):
    def forbidden(*args,**kw):raise AssertionError("dense geometry forbidden")
    for name in ("eigh","eigvalsh","svd","solve","inv"):monkeypatch.setattr(np.linalg,name,forbidden)
    d=32768;c=SimpleNamespace(dimension=d,root=0,edges=tuple(((i-1)//2,i) for i in range(1,d)),
                              parameter_count=2*(d-1))
    state=np.exp(.1j*np.arange(d))/np.sqrt(d)
    x=tree_parameters_from_state(c,state)
    p,t=tree_state_tangent(c,x,np.sin(np.arange(c.parameter_count)))
    y=tree_parameters_from_state(c,p+1e-3*t)
    actual,_=tree_state_weights(c,y)
    assert np.linalg.norm(actual)==pytest.approx(1.,abs=2e-14)


@pytest.mark.parametrize("value",[np.zeros(11),np.ones(10),np.full(11,np.nan)])
def test_invalid_inverse_states_rejected(value):
    with pytest.raises(ValueError):tree_parameters_from_state(tree("binary"),value)

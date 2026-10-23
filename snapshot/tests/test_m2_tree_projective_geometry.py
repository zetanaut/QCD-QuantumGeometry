from types import SimpleNamespace
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_graph_natural_gradient import graph_state_jacobian,graph_projective_metric
from deuteron_wigner.quantum.m2_tree_projective_geometry import tree_state_weights,tree_inverse_metric_action


def tree(dimension,shape):
    edges=tuple((i-1 if shape=="chain" else (0 if shape=="star" else (i-1)//2),i)
                for i in range(1,dimension))
    return SimpleNamespace(dimension=dimension,edges=edges,root=0,parameter_count=2*(dimension-1))


@pytest.mark.parametrize("shape",["chain","star","binary"])
@pytest.mark.parametrize("dimension",[3,7,19])
def test_tree_geometry_against_independent_dense_projective_jacobian(shape,dimension):
    c=tree(dimension,shape);rng=np.random.default_rng(dimension+417)
    # Uniform conditional split probabilities keep all interior metrics resolved.
    sizes=np.ones(dimension,dtype=int)
    for a,b in reversed(c.edges):sizes[a]+=sizes[b]
    left=sizes.copy();x=rng.uniform(-.4,.4,c.parameter_count)
    for k,(a,b) in enumerate(c.edges):
        x[2*k]=2*np.arcsin(np.sqrt(sizes[b]/left[a]))+.01*rng.normal()
        left[a]-=sizes[b]
    psi,w=tree_state_weights(c,x)
    direct,jac=graph_state_jacobian(c,x);metric=graph_projective_metric(c,x)
    np.testing.assert_allclose(psi,direct,atol=3e-16)
    np.testing.assert_allclose(metric[0::2,0::2],np.diag(w/4),atol=5e-16)
    np.testing.assert_allclose(metric[0::2,1::2],0.,atol=5e-16)
    g=rng.normal(size=c.parameter_count)
    direction=tree_inverse_metric_action(c,x,g,probability_regularization=0.)
    np.testing.assert_allclose(direction,np.linalg.solve(metric,g),atol=5e-10,rtol=3e-12)
    a=rng.normal(size=(dimension,dimension))+1j*rng.normal(size=(dimension,dimension));h=a+a.conj().T
    residual=h@psi-np.vdot(psi,h@psi).real*psi
    energy_gradient=2*(jac.conj().T@residual).real
    natural=-.5*tree_inverse_metric_action(c,x,energy_gradient,probability_regularization=0.)
    tangent=jac-psi[:,None]*(psi.conj()@jac)[None,:]
    np.testing.assert_allclose(tangent@natural,-residual,atol=3e-13)


def test_declared_regularization_is_not_a_state_or_amplitude_cutoff():
    c=tree(8,"binary");x=np.ones(c.parameter_count)*.2
    x[0]=0.;g=np.arange(c.parameter_count,dtype=float)/10
    with pytest.raises(ValueError,match="singular"):
        tree_inverse_metric_action(c,x,g,probability_regularization=0.)
    before,_=tree_state_weights(c,x)
    value=tree_inverse_metric_action(c,x,g,probability_regularization=1e-8)
    assert np.all(np.isfinite(value)) and g@value>0
    after,_=tree_state_weights(c,x)
    np.testing.assert_array_equal(after,before)
    assert np.count_nonzero(after)==np.count_nonzero(before)


@pytest.mark.parametrize("epsilon",[-1.,np.nan,True])
def test_invalid_geometry_controls_rejected(epsilon):
    c=tree(3,"star")
    with pytest.raises(ValueError):
        tree_inverse_metric_action(c,np.ones(4),np.ones(4),probability_regularization=epsilon)


def test_large_tree_action_has_no_dense_jacobian_or_linear_algebra_dependency(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError("dense geometry forbidden")
    for name in ("svd","eigh","solve","inv"):monkeypatch.setattr(np.linalg,name,forbidden)
    c=tree(32768,"binary")
    x=np.ones(c.parameter_count);g=np.sin(np.arange(c.parameter_count))
    result=tree_inverse_metric_action(c,x,g,probability_regularization=1e-12)
    assert result.shape==g.shape and np.all(np.isfinite(result))
    assert g@result>0


@pytest.mark.parametrize("epsilon",[0.,1e-12])
def test_actual_quantum_fit_tree_path_never_constructs_a_dense_geometry(monkeypatch,epsilon):
    import deuteron_wigner.quantum.m2_graph_natural_gradient as natural
    from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit
    h=np.array([[1.,.3+.2j,-.1j],[.3-.2j,2.,.4],[.1j,.4,3.]])
    c=SymmetryGraphCircuit(h,root=0,support_threshold=0.)
    def forbidden(*args,**kw):raise AssertionError("dense state Jacobian forbidden")
    monkeypatch.setattr(natural,"graph_state_jacobian",forbidden)
    fit=natural.quantum_natural_fit(c,c.uniform_parameters(),geometry="tree",metric_rcond=None,
        probability_regularization=epsilon,residual_tolerance=1e-7,max_iterations=300)
    assert fit.success,fit.message
    assert fit.diagnostics["residual_GeV2"]<1e-7
    # An exact eigensolve is only an independent post-optimization reference.
    assert fit.fun==pytest.approx(np.linalg.eigvalsh(h)[0],abs=3e-13)
    assert fit.geometry=="tree" and fit.metric_rcond is None
    assert fit.diagnostics["padding_probability"]<3e-28
    measured=c.measure_operators(fit.x,[np.eye(c.dimension)])
    assert measured.shape==(1,)
    np.testing.assert_allclose(measured,[1.],atol=3e-14)
    c.state(fit.x)
    assert all(o.name not in ("StatePrep","QubitUnitary") for o in c.state.qtape.operations)


def test_tree_optimizer_does_not_invent_a_metric_regularization():
    from deuteron_wigner.quantum.m2_graph_natural_gradient import quantum_natural_fit
    from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit
    c=SymmetryGraphCircuit([[1.,.2],[.2,2.]],root=0,support_threshold=0.)
    with pytest.raises(ValueError,match="tree controls"):
        quantum_natural_fit(c,c.uniform_parameters(),geometry="tree",metric_rcond=None,
                            residual_tolerance=1e-8)

import numpy as np
import pytest
from scipy.linalg import expm

from deuteron_wigner.quantum.chiral_soft_matching import G5
from deuteron_wigner.quantum.modular_gauge_network import GaugeGraph,Path,su3,generators
from deuteron_wigner.quantum.momentum_chiral_matching import (
    dirac_basis,vector_fierz,color_fierz,flavor_fierz,two_quark_exchange,SpacelikeKernel,
    SharedInvariantFunctional,FunctionalRemainder,vector_map,vector_map_direct,
)


def fixture(nonlinear=.8):
    rng = np.random.default_rng(110926)
    graph = GaugeGraph({(0,1):su3(rng.normal(0,.3,8)),
                        (1,2):su3(rng.normal(0,.3,8)),
                        (0,2):su3(rng.normal(0,.3,8))})
    paths = {(i,i):Path((i,)) for i in range(3)}
    for i,j,vertices in [(0,1,(0,1)),(1,2,(1,2)),(0,2,(0,1,2))]:
        paths[i,j] = Path(vertices,"FINITE_SOURCE_REFERENCE")
        paths[j,i] = paths[i,j].reversed()
    transports = np.array([[graph.transport(paths[i,j]) for j in range(3)] for i in range(3)])
    kernel = SpacelikeKernel(1.2,.8,(.3,-.2,.1),(.01,100.))
    points = np.array([[0,0,0,0],[.8,.1,-.3,.2],[-.4,.2,.5,.1]])
    qs = np.array([[.3,.1,.2,.4],[.5,-.2,.1,.3],[.2,.6,-.3,.1]])
    qs = np.r_[qs,-qs]
    pieces = kernel.fourier(points,qs,np.full(6,.03),.4)
    f = SharedInvariantFunctional(pieces["total"],np.array([.6,.8,.7]),transports,.8,nonlinear)
    s = rng.normal(0,.13,(3,3,24,24))+1j*rng.normal(0,.13,(3,3,24,24))
    ds = rng.normal(0,.08,s.shape)+1j*rng.normal(0,.08,s.shape)
    return f,s,ds,graph,paths,kernel,pieces


def lf(cart):
    return np.array([(cart[0]+cart[3])/np.sqrt(2),cart[1],cart[2]])


def scattering_case(theta=.7,boost=None,masses=(.13,.21)):
    m1,m2 = masses
    inc = np.array([.12,.15,.5]); magnitude=np.linalg.norm(inc)
    out = magnitude*np.array([np.sin(theta)*np.cos(.4),np.sin(theta)*np.sin(.4),np.cos(theta)])
    carts = [np.r_[np.sqrt(m1*m1+magnitude**2),out],
             np.r_[np.sqrt(m2*m2+magnitude**2),-out],
             np.r_[np.sqrt(m1*m1+magnitude**2),inc],
             np.r_[np.sqrt(m2*m2+magnitude**2),-inc]]
    if boost is not None: carts=[boost@x for x in carts]
    return two_quark_exchange(*map(lf,carts),m1,m2,1.2)


def test_complete_dirac_color_recoupling_and_projection_losses():
    _,b = dirac_basis()
    np.testing.assert_allclose(np.einsum("aij,bij->ab",b.conj(),b)/4,np.eye(16),atol=1e-15)
    f = vector_fierz()
    np.testing.assert_allclose(f["rebuilt"],f["tensor"],atol=1e-15)
    assert np.linalg.norm(f["scalar_pseudoscalar"]-f["tensor"])/np.linalg.norm(f["tensor"]) == pytest.approx(1/np.sqrt(2))
    c,s,o = color_fierz()
    np.testing.assert_allclose(c,s+o,atol=1e-15)
    assert np.linalg.norm(c-s)/np.linalg.norm(c) == pytest.approx(1/3)
    assert np.max(abs(f["coefficients"][10:])) < 1e-15
    direct,scalar,vector=flavor_fierz()
    np.testing.assert_allclose(direct,scalar+vector,atol=1e-15)


def test_replay_is_serializable_and_keeps_its_claim_boundary():
    import json
    from scripts.evaluate_momentum_chiral_matching import evaluate
    result=evaluate()
    json.dumps(result,allow_nan=False)
    assert result["bare_maxima"]["raw_error_over_epsilon_condition"] < 40
    assert result["bare_maxima"]["LF_combined_relative"] < 1e-10
    assert result["curvature_negative"]["frozen_curvature_chiral_max_entry"] < 1e-12
    assert result["curvature_negative"]["omitted_curvature_radial_relative"] > .05
    assert result["nonlinear_subtraction"]["naive_kernel_relative"] > .01
    assert result["inputs"]["physical_soft_coefficients"].startswith("unknown")


@pytest.mark.parametrize("theta",[.45,.8,1.2,2.1])
@pytest.mark.parametrize("masses",[(0.,0.),(.13,.21)])
def test_canonical_LF_exchange_full_matrix_and_boost(theta,masses):
    a = scattering_case(theta,masses=masses)
    assert a["current_divergence"] < 2e-14
    np.testing.assert_allclose(a["light_front"],a["covariant"],rtol=1e-11,atol=2e-11)
    np.testing.assert_allclose(a["light_front_combined"],a["covariant"],rtol=1e-11,atol=2e-11)
    assert np.linalg.norm(a["propagating"]-a["covariant"])/np.linalg.norm(a["covariant"]) > .01
    generator=np.zeros((4,4));generator[0,1]=generator[1,0]=.31
    b = scattering_case(theta,expm(generator),masses)
    np.testing.assert_allclose(b["light_front"],b["covariant"],rtol=1e-11,atol=2e-11)
    assert np.linalg.norm(b["covariant"]) == pytest.approx(np.linalg.norm(a["covariant"]),rel=2e-13)


@pytest.mark.parametrize("fraction",[0.,.4,1.])
def test_momentum_tree_matching_and_no_double_counting(fraction):
    kernel = SpacelikeKernel(1.2,.8,(.3,-.2,.1),(.01,100.))
    qs = np.geomspace(.02,50,17)
    p = kernel.pieces(qs,fraction)
    np.testing.assert_allclose(p["retained"]+p["residual"],p["total"],atol=2e-14)
    np.testing.assert_allclose(p["tree_remainder"],(1-fraction)*p["canonical"],atol=2e-14)
    if fraction==1: np.testing.assert_array_equal(p["tree_remainder"],0)
    # Tree matching says nothing about the unknown nonperturbative coefficient.
    assert np.linalg.norm(p["diagnostic_soft"])>.01
    if fraction>0:
        assert np.linalg.norm(p["retained"]+p["total"]-p["total"])>.01
    assert abs(qs[-1]*p["diagnostic_soft"][-1]) < abs(qs[-2]*p["diagnostic_soft"][-2])


def test_independent_color_map_all_indices():
    f,s,_,_,_,_,_ = fixture()
    for i,j in [(0,0),(0,1),(2,0)]:
        np.testing.assert_allclose(vector_map(s[i,j],f.w[i,j]),vector_map_direct(s[i,j],f.w[i,j]),atol=3e-15)


@pytest.mark.parametrize("nonlinear",[0.,.8])
def test_first_and_second_variations_and_torch_autodiff(nonlinear):
    import torch
    f,s,ds,_,_,_,_ = fixture(nonlinear)
    # Phi is quartic along this direction: a five-point derivative removes
    # its cubic truncation term without subtracting nearly equal tiny steps.
    h=1e-3
    sigma=f.self_energy(s)
    first=(-f.value(s+2*h*ds)+8*f.value(s+h*ds)-8*f.value(s-h*ds)+f.value(s-2*h*ds))/(12*h)
    assert first == pytest.approx(f.pairing(sigma,ds),abs=2e-10)
    second=-(-f.self_energy(s+2*h*ds)+8*f.self_energy(s+h*ds)-8*f.self_energy(s-h*ds)+f.self_energy(s-2*h*ds))/(12*h)
    np.testing.assert_allclose(second,f.kernel_action(s,ds),atol=2e-10)
    epsilon=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    st=torch.tensor(s,dtype=torch.complex128)+epsilon*torch.tensor(ds,dtype=torch.complex128)
    phi=f.torch_value(st)
    assert phi.detach().numpy() == pytest.approx(f.value(s),abs=2e-13)
    for component,selector in [(phi.real,np.real),(phi.imag,np.imag)]:
        first_t=torch.autograd.grad(component,epsilon,create_graph=True,retain_graph=True)[0]
        second_t=torch.autograd.grad(first_t,epsilon,retain_graph=True)[0]
        assert first_t.item() == pytest.approx(selector(f.pairing(sigma,ds)),abs=2e-12)
        assert second_t.item() == pytest.approx(selector(-f.pairing(f.kernel_action(s,ds),ds)),abs=2e-12)


def test_local_gauge_covariance_and_frozen_transport_negative():
    f,s,_,graph,_,_,_ = fixture()
    rng=np.random.default_rng(9126)
    gauges={i:su3(rng.normal(0,.6,8)) for i in range(3)}
    big=[np.kron(np.eye(8),gauges[i]) for i in range(3)]
    moved=np.array([[big[i]@s[i,j]@big[j].conj().T for j in range(3)] for i in range(3)])
    transports=np.array([[gauges[i]@f.w[i,j]@gauges[j].conj().T for j in range(3)] for i in range(3)])
    other=SharedInvariantFunctional(f.f,f.weights,transports,f.scale,f.nonlinear)
    assert other.value(moved) == pytest.approx(f.value(s),abs=2e-13)
    expected=np.array([[big[i]@f.self_energy(s)[i,j]@big[j].conj().T for j in range(3)] for i in range(3)])
    np.testing.assert_allclose(other.self_energy(moved),expected,atol=5e-14)
    assert abs(f.value(moved)-f.value(s)) > .001


def test_local_nonsinglet_chiral_and_EM_identities_do_not_test_all_curvature():
    f,s,_,_,_,_,_=fixture()
    tau=np.array([[0,1],[1,0]],complex)
    a=np.kron(np.kron(G5,tau/2),np.eye(3))
    axial=[c*a for c in [.2,-.6,.9]]
    rot=[expm(1j*x) for x in axial]
    moved=np.array([[rot[i]@s[i,j]@rot[j] for j in range(3)] for i in range(3)])
    assert f.value(moved) == pytest.approx(f.value(s),abs=2e-13)
    sig=f.self_energy(s)
    ds=np.array([[axial[i]@s[i,j]+s[i,j]@axial[j] for j in range(3)] for i in range(3)])
    target=np.array([[axial[i]@sig[i,j]+sig[i,j]@axial[j] for j in range(3)] for i in range(3)])
    for include in [True,False]:
        np.testing.assert_allclose(f.kernel_action(s,ds,include_curvature=include),target,atol=5e-14)
    q=np.kron(np.kron(np.eye(4),np.diag([2/3,-1/3])),np.eye(3))
    charge=[c*q for c in [.3,-.2,.7]]
    dv=np.array([[charge[i]@s[i,j]-s[i,j]@charge[j] for j in range(3)] for i in range(3)])
    vt=np.array([[-charge[i]@sig[i,j]+sig[i,j]@charge[j] for j in range(3)] for i in range(3)])
    np.testing.assert_allclose(f.kernel_action(s,dv),vt,atol=5e-14)
    # Yet a radial response loses the indispensable u'' contribution.
    full=f.kernel_action(s,s)
    wrong=f.kernel_action(s,s,include_curvature=False)
    assert np.linalg.norm(full-wrong)/np.linalg.norm(full)>.05


def test_source_derivative_and_mixed_state_source_variations():
    import torch
    f,s,ds,graph,paths,_,_=fixture()
    edge=(0,1);t=generators()[2]
    dw=np.array([[graph.path_tangent(paths[i,j],edge,t) for j in range(3)] for i in range(3)])
    source,sigsource=f.path_source(s,dw)
    def moved(e):
        g=GaugeGraph({key:(expm(1j*e*t)@u if key==edge else u) for key,u in graph.links.items()})
        w=np.array([[g.transport(paths[i,j]) for j in range(3)] for i in range(3)])
        return SharedInvariantFunctional(f.f,f.weights,w,f.scale,f.nonlinear)
    h=2e-6
    assert (moved(h).value(s)-moved(-h).value(s))/(2*h) == pytest.approx(source,abs=2e-10)
    np.testing.assert_allclose((moved(h).self_energy(s)-moved(-h).self_energy(s))/(2*h),sigsource,atol=2e-10)
    e=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    v=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    links={key:torch.tensor(u,dtype=torch.complex128) for key,u in graph.links.items()}
    links[edge]=torch.matrix_exp(1j*e*torch.tensor(t))@links[edge]
    def transport(path):
        out=torch.eye(3,dtype=torch.complex128)
        for a,b in zip(path.vertices[:-1],path.vertices[1:]):
            out=out@(links[a,b] if (a,b) in links else links[b,a].conj().T)
        return out
    wt=torch.stack([torch.stack([transport(paths[i,j]) for j in range(3)]) for i in range(3)])
    phi=f.torch_value(torch.tensor(s)+v*torch.tensor(ds),transports=wt)
    for comp,selector in [(phi.real,np.real),(phi.imag,np.imag)]:
        derivative=torch.autograd.grad(comp,e,create_graph=True,retain_graph=True)[0]
        mixed=torch.autograd.grad(derivative,v,retain_graph=True)[0]
        assert derivative.item() == pytest.approx(selector(source),abs=3e-12)
        assert mixed.item() == pytest.approx(selector(f.pairing(sigsource,ds)),abs=3e-12)


def test_nonlinear_retained_subtraction_must_precede_variations():
    f,s,ds,graph,paths,_,pieces=fixture()
    ret=SharedInvariantFunctional(pieces["retained"],f.weights,f.w,f.scale,f.nonlinear)
    naive=SharedInvariantFunctional(pieces["residual"],f.weights,f.w,f.scale,f.nonlinear)
    correct=FunctionalRemainder(f,ret)
    assert correct.value(s)+ret.value(s) == pytest.approx(f.value(s),abs=1e-14)
    np.testing.assert_allclose(correct.self_energy(s)+ret.self_energy(s),f.self_energy(s),atol=1e-14)
    np.testing.assert_allclose(correct.kernel_action(s,ds)+ret.kernel_action(s,ds),f.kernel_action(s,ds),atol=1e-14)
    dw=np.array([[graph.path_tangent(paths[i,j],(0,1),generators()[2]) for j in range(3)] for i in range(3)])
    assert correct.path_source(s,dw)[0]+ret.path_source(s,dw)[0] == pytest.approx(f.path_source(s,dw)[0],abs=1e-14)
    assert abs(correct.value(s)-naive.value(s))>.001
    assert np.linalg.norm(correct.self_energy(s)-naive.self_energy(s))>.01
    assert np.linalg.norm(correct.kernel_action(s,ds)-naive.kernel_action(s,ds))>.01


def test_disconnected_blocks_are_additive_not_a_square_of_total_action():
    f,s,_,_,_,_,_=fixture()
    block_f=f.f.copy();block_f[0,1:]=block_f[1:,0]=0
    separated=SharedInvariantFunctional(block_f,f.weights,f.w,f.scale,f.nonlinear)
    left=SharedInvariantFunctional(block_f[:1,:1],f.weights[:1],f.w[:1,:1],f.scale,f.nonlinear)
    right=SharedInvariantFunctional(block_f[1:,1:],f.weights[1:],f.w[1:,1:],f.scale,f.nonlinear)
    assert separated.value(s) == pytest.approx(left.value(s[:1,:1])+right.value(s[1:,1:]),abs=1e-14)


def test_gamma5_hermitian_propagator_gives_real_action_and_covariant_self_energy():
    f,s,_,_,_,_,_=fixture()
    g5=np.kron(G5,np.eye(6))
    s=(s+np.array([[g5@s[j,i].conj().T@g5 for j in range(3)] for i in range(3)]))/2
    assert abs(f.value(s).imag)<1e-13
    sig=f.self_energy(s)
    np.testing.assert_allclose(sig,np.array([[g5@sig[j,i].conj().T@g5 for j in range(3)] for i in range(3)]),atol=2e-13)


def test_momentum_coefficient_gradients_reach_the_same_state_response():
    import torch
    f,s,ds,_,_,kernel,_=fixture()
    points=np.array([[0,0,0,0],[.8,.1,-.3,.2],[-.4,.2,.5,.1]])
    qs=np.array([[.3,.1,.2,.4],[.5,-.2,.1,.3],[.2,.6,-.3,.1]])
    qs=np.r_[qs,-qs]; q2=np.sum(qs*qs,axis=1)
    phase=np.cos(np.einsum("qk,xyk->xyq",qs,points[:,None,:]-points[None,:,:]))
    design=np.einsum("xyq,qa->xya",phase,.03*kernel.features(q2))
    canonical=np.einsum("xyq,q->xy",phase,.03*kernel.gs_squared/q2)
    theta=torch.tensor(kernel.coefficients,dtype=torch.float64,requires_grad=True)
    e=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    ft=torch.tensor(canonical)+torch.einsum("xya,a->xy",torch.tensor(design),theta)
    phi=f.torch_value(torch.tensor(s)+e*torch.tensor(ds),fourier_kernel=ft)
    for component,selector in [(phi.real,np.real),(phi.imag,np.imag)]:
        gradient=torch.autograd.grad(component,theta,create_graph=True,retain_graph=True)[0]
        for j in range(3):
            mixed=torch.autograd.grad(gradient[j],e,retain_graph=True)[0]
            h=1e-3
            plus=SharedInvariantFunctional(f.f+h*design[:,:,j],f.weights,f.w,f.scale,f.nonlinear)
            minus=SharedInvariantFunctional(f.f-h*design[:,:,j],f.weights,f.w,f.scale,f.nonlinear)
            expected=(plus.value(s)-minus.value(s))/(2*h)
            expected_mixed=f.pairing((plus.self_energy(s)-minus.self_energy(s))/(2*h),ds)
            assert gradient[j].item()==pytest.approx(selector(expected),abs=2e-11)
            assert mixed.item()==pytest.approx(selector(expected_mixed),abs=2e-11)


@pytest.mark.parametrize("q",[-.1,0.,.001,101.,np.nan])
def test_matching_domain_is_not_silent_extrapolation(q):
    k=SpacelikeKernel(1.2,.8,(.3,-.2,.1),(.01,100.))
    with pytest.raises(ValueError):k.pieces(q,.4)

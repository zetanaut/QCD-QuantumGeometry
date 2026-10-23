"""Independent continuous-wavefunction derivatives, not a Ward definition of J-."""
from math import sqrt
from dataclasses import replace
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_smooth_longitudinal_projector import (
    SMOOTH_PRESCRIPTION, smooth_minus_projector_commutator,
    straight_path_smooth_minus_projector_vertex, exact_smooth_minus_projector_boundary)
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_shell
from deuteron_wigner.quantum.m2_coupled_pilot import free_orbital
from deuteron_wigner.quantum.m2_gaussian_field_regulator import gaussian_shell_integral
from deuteron_wigner.quantum.m2_galerkin_projector_current import (
    straight_path_projector_vertex, _rectangular_displacement)
from deuteron_wigner.quantum.m2_gaussian_cluster_projector_kernel import (
    GaussianHamiltonianOrbital, exact_transverse_projector_boundary)
from test_m2_gaussian_cluster_projector_kernel import CASES, data

B,PP = .4,1.3
D = np.array([.17,-.11])
MEAN = np.array([.07,-.13])
ARGS = dict(b_GeV=B,total_pplus_GeV=PP,prescription=SMOOTH_PRESCRIPTION)
FRAMES = [((1,3,5),(0,(1,2))),
          ((1,1,3,4),((0,3),(1,2))),
          ((1,1,1,2,4),(((0,3),4),(1,2))),
          ((1,1,1,2,2,2),((((0,3),4),5),(1,2)))]


def continuous_frame(plus,tree):
    # Independent real-valued wavefunction family, not a finite-mode basis.
    x = np.asarray(plus)/sum(plus)
    rows = []
    def visit(node):
        if not isinstance(node,tuple):
            return [node]
        left,right = visit(node[0]),visit(node[1])
        a,b = x[left].sum(),x[right].sum()
        row = np.zeros(len(x))
        row[left] = sqrt(b/(a*(a+b)))
        row[right] = -sqrt(a/(b*(a+b)))
        rows.append(row)
        return left+right
    visit(tree)
    return np.vstack(rows+[np.ones(len(x))])


def projector_in_original_coordinates(plus,tree,active,epsilon,N,outer,pt):
    f = continuous_frame(plus,tree)
    changed = np.array(plus,float)
    changed[active] += epsilon
    g = continuous_frame(changed,tree)
    base_inverse = np.linalg.inv(f)
    A = g[:-1]@base_inverse[:,:-1]
    shift = (g[:-1]@base_inverse[:,-1,None])*pt
    d = len(plus)-1
    overlap = abs(np.linalg.det(A))*gaussian_shell_integral(
        out_cutoff=outer,in_cutoff=N,out_map=np.eye(d),in_map=A,
        in_shift=shift,b_GeV=B,field_precision=0.,field_maps=np.zeros((0,d)),
        field_shifts=np.zeros((0,2)))
    return overlap@overlap.conj().T


def independent_minus_derivative(plus,tree,active,N,pt,step=2e-5):
    values = {k:projector_in_original_coordinates(plus,tree,active,k*step,N,N+2,pt)
              for k in (-2,-1,1,2)}
    return -(values[-2]-8*values[-1]+8*values[1]-values[2])/(12*step)


@pytest.mark.parametrize("modes,tree",FRAMES)
@pytest.mark.parametrize("N",[0,1])
def test_minus_commutator_is_independent_fixed_physical_wavefunction_derivative(modes,tree,N):
    plus = np.array(modes)*PP/sum(modes)
    d = len(plus)-1
    f = continuous_frame(plus,tree)
    for active in (0,len(modes)-2):
        comm = smooth_minus_projector_commutator(d,N,f[:-1,active],
            total_transverse_GeV=MEAN,**ARGS)
        ref = independent_minus_derivative(plus,tree,active,N,MEAN)
        np.testing.assert_allclose(comm,ref,atol=8e-10,rtol=0.)
        np.testing.assert_allclose(comm,comm.conj().T,atol=2e-15)
        n = len(oscillator_shell(d,N))
        np.testing.assert_array_equal(comm[:n,:n],0.)
        np.testing.assert_array_equal(comm[n:,n:],0.)
        assert np.linalg.norm(comm)>1e-2


@pytest.mark.parametrize("N,outer",[(0,2),(1,3)])
def test_oblique_path_derivative_reversal_boost_idempotency_and_refinement(N,outer):
    modes,tree = FRAMES[0]
    plus = np.array(modes)*PP/sum(modes)
    w = continuous_frame(plus,tree)[:-1,0]
    args = dict(mean_total_transverse_GeV=MEAN,quadrature_order=12,**ARGS)
    result = straight_path_smooth_minus_projector_vertex(2,N,outer,w,D,**args)
    reverse = straight_path_smooth_minus_projector_vertex(2,N,outer,w,-D,**args)
    np.testing.assert_allclose(result,reverse.conj().T,atol=4e-15)
    refined = straight_path_smooth_minus_projector_vertex(2,N,outer,w,D,
        **{**args,"quadrature_order":18})
    np.testing.assert_allclose(result,refined,atol=4e-15)
    beta = np.array([.14,-.08])
    boosted = straight_path_smooth_minus_projector_vertex(2,N,outer,w,D,
        **{**args,"mean_total_transverse_GeV":MEAN+PP*beta})
    transverse = straight_path_projector_vertex(2,N,outer,w,D,b_GeV=B,quadrature_order=12)
    np.testing.assert_allclose(boosted-result,np.einsum("j,jab->ab",beta,transverse),atol=5e-15)
    n = len(oscillator_shell(2,N))
    ref = np.zeros_like(result)
    nodes,weights = np.polynomial.legendre.leggauss(14)
    for t,weight in zip((nodes+1)/2,weights/2):
        c = independent_minus_derivative(plus,tree,0,N,MEAN+(t-.5)*D)
        left = _rectangular_displacement(2,outer,N+2,w[:,None]*D*(1-t),B)
        right = _rectangular_displacement(2,N+2,outer,w[:,None]*D*t,B)
        ref += weight*left@c@right
    ref[:n,:n] = 0
    ref[n:,n:] = 0
    np.testing.assert_allclose(result,ref,atol=1e-9,rtol=0.)
    np.testing.assert_array_equal(result[:n,:n],0.)
    np.testing.assert_array_equal(result[n:,n:],0.)
    assert np.linalg.norm(result)>1e-2


def free_kernel(d,constant=.37):
    return GaussianHamiltonianOrbital(out_map=np.eye(d),in_map=np.eye(d),
        field_maps=np.zeros((0,d)),field_shifts=np.zeros((0,2)),b_GeV=B,
        field_precision=0.,measure=1.,constant=constant,quadratic=2*np.eye(2*d))


@pytest.mark.parametrize("d,N",[(1,0),(2,0),(2,1),(3,1)])
def test_exact_free_minus_boundary_equals_independent_finite_support_product(d,N):
    w = np.arange(1,d+1)*.6
    h = free_kernel(d)
    args = dict(out_cutoff=N,in_cutoff=N,out_weights=[w],in_weights=[w],
        out_charges=[.7],in_charges=[.7],transfer_GeV=D,total_pplus_GeV=PP,
        mean_total_transverse_GeV=MEAN,quadrature_order=12,prescription=SMOOTH_PRESCRIPTION)
    result = exact_smooth_minus_projector_boundary(h,**args)
    k = straight_path_smooth_minus_projector_vertex(d,N,N+2,w,D,
        mean_total_transverse_GeV=MEAN,quadrature_order=12,**ARGS)
    hf = free_orbital(d,N+2,B).toarray()
    hf += .37*np.eye(len(hf))
    n = len(oscillator_shell(d,N))
    ref = .7*(k[:n]@hf[:,:n]+hf[:n]@k[:,:n])/(2*PP)
    np.testing.assert_allclose(result,ref,atol=6e-15)
    assert np.linalg.norm(result)>1e-4
    if N==0:
        zero = exact_smooth_minus_projector_boundary(h,
            **{**args,"transfer_GeV":(0.,0.),"mean_total_transverse_GeV":(0.,0.)})
        np.testing.assert_allclose(zero,[[.7*B**2*(d-w@w)/(2*PP**2)]],atol=2e-15)


@pytest.mark.parametrize("case",CASES)
def test_exact_physical_pair_minus_boundaries_reverse_and_boost(case):
    h,wi,wf = data(case)
    ci,co = 0,1
    args = dict(out_cutoff=co,in_cutoff=ci,out_weights=wf,in_weights=wi,
        out_charges=case[-1],in_charges=case[-2],transfer_GeV=D,total_pplus_GeV=PP,
        mean_total_transverse_GeV=MEAN,quadrature_order=12,prescription=SMOOTH_PRESCRIPTION)
    result = exact_smooth_minus_projector_boundary(h,**args)
    inc,out,pair,si,so,ei,ef = case
    hr,_,_ = data((out,inc,pair,so,si,ef,ei))
    reverse = exact_smooth_minus_projector_boundary(hr,**{**args,
        "out_cutoff":ci,"in_cutoff":co,"out_weights":wi,"in_weights":wf,
        "out_charges":ei,"in_charges":ef,"transfer_GeV":-D})
    np.testing.assert_allclose(result,reverse.conj().T,atol=5e-15)
    beta = np.array([.14,-.08])
    boosted = exact_smooth_minus_projector_boundary(h,
        **{**args,"mean_total_transverse_GeV":MEAN+PP*beta})
    targs = {k:v for k,v in args.items() if k not in ("prescription","mean_total_transverse_GeV")}
    transverse = exact_transverse_projector_boundary(h,**targs)
    np.testing.assert_allclose(boosted-result,np.einsum("j,jab->ab",beta,transverse),atol=5e-15)
    refined = exact_smooth_minus_projector_boundary(h,**{**args,"quadrature_order":18})
    np.testing.assert_allclose(result,refined,atol=5e-15)
    assert np.linalg.norm(result)>1e-5


def test_smooth_minus_requires_explicit_scope_and_valid_parameters():
    args = dict(dimension=2,cutoff=0,outer_cutoff=2,weights=[.3,.7],
        transfer_GeV=D,mean_total_transverse_GeV=MEAN,quadrature_order=8,**ARGS)
    for change in (dict(prescription=None),dict(outer_cutoff=-1),dict(dimension=True),
                   dict(quadrature_order=True),dict(weights=[.3]),dict(total_pplus_GeV=0.)):
        with pytest.raises(ValueError):
            straight_path_smooth_minus_projector_vertex(**{**args,**change})
    # A reported Pi-only block is zero by tangent idempotency, not absent physics.
    np.testing.assert_array_equal(straight_path_smooth_minus_projector_vertex(
        **{**args,"outer_cutoff":0}),0.)


@pytest.mark.parametrize("case",CASES[:2])
def test_zero_transfer_exact_pair_boundary_is_derivative_of_original_projected_H(case):
    # Change the physical normalized bra/ket families inside exact Gaussian H
    # integrals. No current or oscillator-ladder formula enters the reference.
    h,wi,wf = data(case)
    inc,out,pair,si,so,ei,ef = case
    ci,co = 0,1
    result = exact_smooth_minus_projector_boundary(h,out_cutoff=co,in_cutoff=ci,
        out_weights=wf,in_weights=wi,out_charges=ef,in_charges=ei,
        transfer_GeV=(0.,0.),total_pplus_GeV=PP,mean_total_transverse_GeV=MEAN,
        quadrature_order=8,prescription=SMOOTH_PRESCRIPTION)
    reference = np.zeros_like(result)
    eps = 2e-5
    for side,(modes,charges,N) in enumerate(((out,ef,co),(inc,ei,ci))):
        d = len(modes)-1
        tree = FRAMES[d-2][1]
        plus = np.array(modes)*PP/sum(modes)
        original = continuous_frame(plus,tree)
        inv = np.linalg.inv(original)
        for active,charge in enumerate(charges):
            if charge==0:
                continue
            values = {}
            for k in (-2,-1,1,2):
                shifted = plus.copy()
                shifted[active] += k*eps
                f = continuous_frame(shifted,tree)
                A = f[:-1]@inv[:,:-1]
                shift = (f[:-1]@inv[:,-1,None])*MEAN
                norm = abs(np.linalg.det(A))
                overlap = norm*gaussian_shell_integral(out_cutoff=N,in_cutoff=N,
                    out_map=np.eye(d),in_map=A,in_shift=shift,b_GeV=B,
                    field_precision=0.,field_maps=np.zeros((0,d)),field_shifts=np.zeros((0,2)))
                if side==0:
                    new_h = replace(h,out_map=A@h.out_map)
                    values[k] = overlap@(norm*new_h.integrate(co,ci,out_shift=shift))
                else:
                    new_h = replace(h,in_map=A@h.in_map)
                    values[k] = (norm*new_h.integrate(co,ci,in_shift=shift))@overlap.conj().T
            reference -= charge*(values[-2]-8*values[-1]+8*values[1]-values[2])/(12*eps*2*PP)
    np.testing.assert_allclose(result,reference,atol=8e-11,rtol=0.)

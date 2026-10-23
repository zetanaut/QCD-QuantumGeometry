"""Exact projector/H products: independent finite Ward and parent-tail limits."""
from dataclasses import replace
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_projector_kernel import (
    GaussianHamiltonianOrbital,pair_hamiltonian_orbital,exact_transverse_projector_boundary)
from deuteron_wigner.quantum.m2_gaussian_cluster_interactions import cluster_pair_contact,_frame
from deuteron_wigner.quantum.m2_gaussian_cluster_pair_current import cluster_pair_current_orbitals
from deuteron_wigner.quantum.m2_galerkin_projector_current import (
    _rectangular_displacement,straight_path_projector_vertex,free_galerkin_boundary)
from deuteron_wigner.quantum.m2_coupled_pilot import free_orbital
from test_m2_gaussian_cluster_current import regulator
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION

D=np.array([.17,-.11]);B=.4;PP=1.3
CASES=[
 ((3,1,5),(1,3,5),(0,1),("q",)*3,("q",)*3,(.7,-.2,.3),(.7,-.2,.3)),
 ((3,1,3,2),(1,1,3,4),(0,3),("q",)*3+("g",),("q",)*3+("g",),(.7,-.2,.3,0.),(.7,-.2,.3,0.)),
 ((1,1,1,2,2),(1,1,1,1,3),(3,4),("q",)*3+("g","g"),("q",)*4+("qbar",),
  (.7,-.2,.3,0.,0.),(.7,-.2,.3,.7,-.7)),
]


def data(case):
    inc,out,pair,si,so,ei,ef=case
    h=pair_hamiltonian_orbital(inc,out,pair,species=si,out_species=so,
        b_GeV=B,regulator=regulator(),total_transverse_GeV=(0.,0.))
    fi,fo=_frame(inc,si),_frame(out,so)
    return h,fi.forward[:-1].T,fo.forward[:-1].T


def density(w,e,co,ci,delta=D):
    return sum(ch*_rectangular_displacement(w.shape[1],co,ci,a[:,None]*delta,B) for a,ch in zip(w,e))


def boundary(case,ci,co,delta=D,order=12):
    h,wi,wf=data(case)
    return exact_transverse_projector_boundary(h,out_cutoff=co,in_cutoff=ci,
        out_weights=wf,in_weights=wi,out_charges=case[-1],in_charges=case[-2],
        transfer_GeV=delta,total_pplus_GeV=PP,quadrature_order=order)


@pytest.mark.parametrize("case",CASES)
@pytest.mark.parametrize("ci,co",[(0,0),(0,1),(1,0),(1,1)])
def test_exact_pair_boundary_closes_physical_four_field_Ward_without_outer_cutoff(case,ci,co):
    inc,out,pair,si,so,ei,ef=case;h,wi,wf=data(case)
    projected=h.integrate(co,ci)
    np.testing.assert_allclose(projected,cluster_pair_contact(inc,out,pair,B,ci,co,
        regulator=regulator(),species=si,out_species=so),atol=3e-16)
    physical=cluster_pair_current_orbitals(inc,out,pair,ci,co,B,species=si,out_species=so,
        charges=tuple(ef[j] for j in pair)+tuple(ei[j] for j in pair),regulator=regulator(),
        total_pplus_GeV=PP,transfer_GeV=D,mean_total_transverse_GeV=(0.,0.),
        hamiltonian_coefficient=1.,canonical_current_coefficient=(0.,0.,0.,0.),
        prescription=PRESCRIPTION,quadrature_order=12)
    k=boundary(case,ci,co)
    expected=projected@density(wi,ei,ci,ci)-density(wf,ef,co,co)@projected
    np.testing.assert_allclose(2*PP*np.einsum("j,jab->ab",D,physical.total[2:]+k),expected,atol=3e-15)
    # An independent full H-rho composition shifts HO wavefunctions only.
    full=sum(ch*h.integrate(co,ci,in_shift=-w[:,None]*D) for w,ch in zip(wi,ei))
    full-=sum(ch*h.integrate(co,ci,out_shift=w[:,None]*D) for w,ch in zip(wf,ef))
    np.testing.assert_allclose(2*PP*np.einsum("j,jab->ab",D,physical.total[2:]),full,atol=3e-15)
    rev=(out,inc,pair,so,si,ef,ei)
    np.testing.assert_allclose(k,boundary(rev,co,ci,delta=-D).conj().transpose(0,2,1),atol=4e-15)
    np.testing.assert_allclose(k,boundary(case,ci,co,order=8),atol=4e-15)


@pytest.mark.parametrize("d,cutoff",[(1,0),(1,1),(2,0),(2,1)])
def test_exact_Gaussian_free_polynomial_boundary_equals_independent_Nplus2_construction(d,cutoff):
    h=GaussianHamiltonianOrbital(out_map=np.eye(d),in_map=np.eye(d),
        field_maps=np.zeros((0,d)),field_shifts=np.zeros((0,2)),b_GeV=B,field_precision=0.,
        measure=1.,constant=.37,quadratic=2*np.eye(2*d))
    expected=free_orbital(d,cutoff,B).toarray()+.37*np.eye(len(free_orbital(d,cutoff,B).toarray()))
    np.testing.assert_allclose(h.integrate(cutoff,cutoff),expected,atol=4e-15)
    w=np.arange(1,d+1)*.3
    result=exact_transverse_projector_boundary(h,out_cutoff=cutoff,in_cutoff=cutoff,
        out_weights=[w],in_weights=[w],out_charges=[.7],in_charges=[.7],
        transfer_GeV=D,total_pplus_GeV=PP,quadrature_order=12)
    reference=.7*free_galerkin_boundary(d,cutoff,w,D,b_GeV=B,total_pplus_GeV=PP,quadrature_order=12)
    np.testing.assert_allclose(result,reference,atol=4e-15)


def test_finite_reported_parent_tails_converge_to_exact_boundary_but_shell_one_is_zero():
    case=CASES[0];h,wi,wf=data(case);exact=boundary(case,0,0)
    errors=[]
    for N in (1,2,4,6,8):
        def tangent(w,e):
            return sum(ch*straight_path_projector_vertex(2,0,N,a,D,b_GeV=B,quadrature_order=12)
                       for a,ch in zip(w,e))
        ki,kf=tangent(wi,case[-2]),tangent(wf,case[-1])
        inc,out,pair,si,so,_,_=case
        hi=cluster_pair_contact(inc,out,pair,B,N,0,regulator=regulator(),species=si,out_species=so)
        hf=cluster_pair_contact(inc,out,pair,B,0,N,regulator=regulator(),species=si,out_species=so)
        finite=np.array([kf[j,:1]@hf+hi@ki[j,:,:1] for j in range(2)])/(2*PP)
        if N==1:np.testing.assert_array_equal(finite,0.)
        errors.append(np.linalg.norm(finite-exact))
    assert errors[0]>1e-4
    assert all(a>b for a,b in zip(errors,errors[1:]))
    assert errors[-1]<2e-7


def test_exact_boundary_requires_explicit_physical_recoil_and_no_silent_plus_minus():
    h,wi,wf=data(CASES[0])
    args=dict(out_cutoff=0,in_cutoff=0,out_weights=wf,in_weights=wi,out_charges=CASES[0][-1],
        in_charges=CASES[0][-2],transfer_GeV=D,total_pplus_GeV=PP,quadrature_order=8)
    for change in (dict(out_cutoff=True),dict(quadrature_order=True),
                   dict(out_charges=(1.,)),dict(total_pplus_GeV=0.)):
        with pytest.raises(ValueError):exact_transverse_projector_boundary(h,**{**args,**change})
    assert exact_transverse_projector_boundary(h,**args).shape==(2,1,1)

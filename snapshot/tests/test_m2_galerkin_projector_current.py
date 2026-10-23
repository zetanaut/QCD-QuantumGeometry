"""Noncommuting projector Ward identities, regularity and actual spin bases."""
import numpy as np
import pytest
from scipy import sparse

from deuteron_wigner.quantum.m2_coupled_pilot import free_orbital,free_sector
from deuteron_wigner.quantum.m2_transverse_current_projection import (
    displacement_and_moments,free_transverse_current_between,
)
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_galerkin_projector_current import (
    straight_path_projector_vertex,free_galerkin_boundary,retained_free_boundary_between,
)
from test_m2_transverse_current_projection import endpoint


@pytest.mark.parametrize("d,cutoff",[(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(3,1)])
def test_projector_vertex_ward_idempotent_tangent_and_reversal(d,cutoff):
    w=np.linspace(.3,1.1,d);delta=np.array([.29,-.21]);b=.4
    k=straight_path_projector_vertex(d,cutoff,cutoff+2,w,delta,b_GeV=b,quadrature_order=20)
    reverse=straight_path_projector_vertex(d,cutoff,cutoff+2,w,-delta,b_GeV=b,quadrature_order=20)
    density,_=displacement_and_moments(d,cutoff+2,w[:,None]*delta,b_GeV=b)
    n=len(free_orbital(d,cutoff,b).toarray())
    pi=np.diag(np.arange(len(density))<n)
    np.testing.assert_allclose(np.einsum("j,joi->oi",delta,k),pi@density-density@pi,atol=3e-15)
    np.testing.assert_allclose(k,pi@k+k@pi,atol=0.)
    np.testing.assert_allclose(k,reverse.conj().swapaxes(1,2),atol=3e-15)
    # Q=1-Pi has the opposite current; gauging Pi Q=0 obeys the product rule.
    np.testing.assert_allclose(k@(np.eye(len(pi))-pi)-pi@k,0.,atol=0.)


@pytest.mark.parametrize("d,cutoff",[(1,0),(1,1),(1,2),(2,1),(2,2),(3,2)])
def test_regular_boundary_closes_independently_projected_free_current(d,cutoff):
    w=np.linspace(.4,1.2,d);delta=np.array([.12,.16]);b=.4;pp=1.3
    density,m=displacement_and_moments(d,cutoff,w[:,None]*delta,b_GeV=b)
    h=free_orbital(d,cutoff,b).toarray()
    # Full continuum scalar current, with the outside momentum ladder retained.
    canonical=(np.einsum("a,ajoi->joi",w,m)+np.sum(w*w)*delta[:,None,None]*density/2)/pp
    boundary=free_galerkin_boundary(d,cutoff,w,delta,b_GeV=b,total_pplus_GeV=pp,quadrature_order=20)
    raw=h@density-density@h-2*pp*np.einsum("j,joi->oi",delta,canonical)
    np.testing.assert_allclose(raw,2*pp*np.einsum("j,joi->oi",delta,boundary),atol=4e-15)
    if cutoff:assert np.linalg.norm(raw)>1e-3
    else:np.testing.assert_allclose(boundary,0.,atol=2e-16)


@pytest.mark.parametrize("delta",[(0.,0.),(1e-10,-2e-10),(.81,-.47)])
def test_no_transfer_division_and_path_quadrature_convergence(delta):
    args=(2,2,4,[.4,1.2],delta)
    a=straight_path_projector_vertex(*args,b_GeV=.4,quadrature_order=24)
    c=straight_path_projector_vertex(*args,b_GeV=.4,quadrature_order=48)
    assert np.all(np.isfinite(a))
    np.testing.assert_allclose(a,c,atol=5e-15)
    if not np.linalg.norm(delta):
        # Independent displacement finite difference supplies the tangent at zero.
        pi=np.diag(np.arange(len(a[0]))<15);epsilon=1e-5
        for j in (0,1):
            shift=np.zeros((2,2));shift[:,j]=np.array([.4,1.2])*epsilon
            plus,_=displacement_and_moments(2,4,shift,b_GeV=.4)
            minus,_=displacement_and_moments(2,4,-shift,b_GeV=.4)
            derivative=(plus-minus)/(2*epsilon)
            np.testing.assert_allclose(a[j],pi@derivative-derivative@pi,atol=3e-8)


def test_outer_reporting_cutoff_does_not_truncate_internal_translations():
    args=(1,1)
    small=straight_path_projector_vertex(*args,3,[.7],(.3,-.2),b_GeV=.4,quadrature_order=24)
    large=straight_path_projector_vertex(*args,5,[.7],(.3,-.2),b_GeV=.4,quadrature_order=24)
    n=small.shape[-1]
    np.testing.assert_allclose(small,large[:,:n,:n],atol=2e-15)


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_actual_second_shell_symmetry_current_and_flavor(target,jz):
    i=endpoint(2,target,1);f=endpoint(2,target,jz);delta=np.array([.12,.16]);pp=1.3
    def partner(out,inc,delta,charge="electromagnetic",pplus=pp):
        return retained_free_boundary_between(out,inc,delta,total_pplus_GeV=pplus,
                                              charge=charge,quadrature_order=16)
    boundary=partner(f,i,delta)
    np.testing.assert_allclose(boundary,partner(i,f,-delta).conj().swapaxes(1,2),atol=2e-14)
    np.testing.assert_allclose((2*partner(f,i,delta,"up_number")-
                               partner(f,i,delta,"down_number"))/3,boundary,atol=2e-14)
    np.testing.assert_allclose(partner(f,i,delta,pplus=2*pp),boundary/2,atol=0.)
    def h(p):
        return sparse.block_diag([(u.getH()@free_sector(s,9,2,.4,.35,0.)@u).toarray()
             for s,u in zip(("qqq","qqqg","qqqg"),p.bases)]).toarray()
    density=good_current_density_between(f,i,delta,charge="electromagnetic")
    canonical=free_transverse_current_between(f,i,delta,total_pplus_GeV=pp,charge="electromagnetic")
    raw=h(f)@density-density@h(i)-2*pp*np.einsum("j,joi->oi",delta,canonical)
    np.testing.assert_allclose(raw,2*pp*np.einsum("j,joi->oi",delta,boundary),atol=6e-14)
    if target=="PROTON":assert np.linalg.norm(raw)>1e-3


def test_boundary_does_not_repair_ground_shell_magnetic_covariance():
    # Exact g=0 ground-shell counterexample: projector current is spin scalar
    # and its orbital block vanishes. The order-one magnetic discrepancy stays.
    for d in (2,3):
        result=free_galerkin_boundary(d,0,np.ones(d),(.12,.16),b_GeV=.4,
                                    total_pplus_GeV=1.3,quadrature_order=20)
        np.testing.assert_allclose(result,0.,atol=3e-16)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        straight_path_projector_vertex(1,2,1,[1.],(0.,0.),b_GeV=.4,quadrature_order=20)
    with pytest.raises(ValueError):
        straight_path_projector_vertex(1,0,2,[1.],(0.,0.),b_GeV=.4,quadrature_order=1)
    with pytest.raises(ValueError):
        straight_path_projector_vertex(1,0,2,[1.,2.],(0.,0.),b_GeV=.4,quadrature_order=20)
    with pytest.raises(ValueError):
        free_galerkin_boundary(1,0,[1.],(0.,0.),b_GeV=.4,total_pplus_GeV=0.,quadrature_order=20)

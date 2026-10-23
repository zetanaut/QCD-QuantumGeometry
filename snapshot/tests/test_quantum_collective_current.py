import json
import numpy as np
import pytest
from scipy.linalg import expm

from deuteron_wigner.quantum.quantum_collective_current import (
    triangle_unit,photon_attachments,numerator_coefficients,bubble_vertex_norm,
    normalized_loop_shape,elastic_momenta,lf_spin_numerator,lf_overlap_shape,
    lf_parameter_shape,loop_slope,dot,
)
from deuteron_wigner.quantum.quantum_pair_threshold import feynman_nonlocal

M=.2843478671310942
S=.13957039**2


@pytest.mark.parametrize('seed',[11,27,31])
def test_independent_original_Dirac_trace_and_angular_numerator(seed):
    a,b,c=np.random.default_rng(seed).normal(0,.2,(3,4))
    direct=numerator_coefficients(a,b,c,M,method='trace')
    reduced=numerator_coefficients(a,b,c,M)
    np.testing.assert_allclose(direct,reduced,atol=2e-15,rtol=2e-13)


@pytest.mark.parametrize('s1,s0',[(.02,.01),(-.03,-.05),(.06,.04)])
def test_offshell_triangle_Ward_from_independent_inverse(s1,s0):
    p1=np.array([.31,np.sqrt(.31**2-s1),0,0]);p0=np.array([.25,-np.sqrt(.25**2-s0),0,0])
    a=photon_attachments(M,p1,p0,order=36)
    difference=feynman_nonlocal(M,s0,'pion')-feynman_nonlocal(M,s1,'pion')
    assert dot(p1-p0,a['total'])==pytest.approx(difference,rel=3e-11,abs=1e-14)
    assert abs(dot(p1-p0,a['quark'])-difference)>1e-6
    np.testing.assert_allclose(a['antiquark'],a['quark']/2,atol=2e-16)


def test_actual_flavor_factors_and_neutral_pair_cancellation():
    tp=np.sqrt(2)*np.array([[0,1],[0,0]]);tm=tp.T;q=np.diag([2/3,-1/3])
    assert np.trace(tm@q@tp)/2==pytest.approx(2/3)
    assert np.trace(tm@tp@q)/2==pytest.approx(-1/3)
    p1,p0=elastic_momenta(S,.1)
    np.testing.assert_allclose(photon_attachments(M,p1,p0,charges=(.4,.4))['total'],0,atol=1e-16)
    np.testing.assert_allclose(triangle_unit(M,-p1,-p0),-triangle_unit(M,p1,p0),atol=1e-16)


@pytest.mark.parametrize('s',[.01,S,.06])
def test_zero_transfer_norm_is_linked_inverse_derivative_not_charge_fitting(s):
    p=np.array([np.sqrt(s+.08),.2,0,.2])
    vertex=triangle_unit(M,p,p)
    norm=bubble_vertex_norm(M,s)
    np.testing.assert_allclose(vertex,2*p*norm,atol=3e-15,rtol=1e-12)
    h=2e-6
    derivative=-(feynman_nonlocal(M,s+h,'pion')-feynman_nonlocal(M,s-h,'pion'))/(2*h)
    assert norm==pytest.approx(derivative,rel=2e-9)


@pytest.mark.parametrize('q2',[0.,.02,.1,.5])
def test_independent_triangle_and_LF_overlaps(q2):
    triangle=normalized_loop_shape(M,S,q2)
    parameter=lf_parameter_shape(M,S,q2,order=64)
    overlap=lf_overlap_shape(M,S,q2,order=64)
    assert triangle==pytest.approx(parameter,rel=2e-12,abs=5e-16)
    assert overlap==pytest.approx(parameter,rel=2e-10,abs=5e-16)
    if q2:
        assert triangle<0
        # Not dividing by the linked pole norm leaves a spurious charge term.
        assert abs(triangle-(triangle+bubble_vertex_norm(M,S)))>.005


def test_direct_trace_loop_four_components_boost_and_reversal():
    p1=np.array([.2,.05,.03,.02]);p0=np.array([.13,-.07,.02,-.01])
    reduced=triangle_unit(M,p1,p0,order=12)
    trace=triangle_unit(M,p1,p0,order=12,method='trace')
    np.testing.assert_allclose(trace,reduced,atol=2e-15,rtol=1e-12)
    generator=np.zeros((4,4));generator[0,1:]=generator[1:,0]=[.3,-.2,.25];boost=expm(generator)
    changed=triangle_unit(M,boost@p1,boost@p0,order=12,method='trace')
    np.testing.assert_allclose(changed,boost@trace,atol=2e-15,rtol=1e-12)
    np.testing.assert_allclose(triangle_unit(M,p0,p1,order=12),reduced,atol=2e-16)
    # Separate on-shell component extractions after a boost with nonzero sum.
    p1,p0=elastic_momenta(S,.1);p1,p0=boost@p1,boost@p0
    j=triangle_unit(M,p1,p0)
    np.testing.assert_allclose(j/(p1+p0),j[0]/(p1+p0)[0],atol=2e-15)


@pytest.mark.parametrize('x,plus',[(.2,1.),(.7,2.3)])
def test_LF_numerator_from_canonical_spinors_and_retained_orbitals(x,plus):
    kx,ky=.2,.1
    n=lf_spin_numerator(x,kx,ky,M,plus=plus)
    expected=1j*np.array([[kx+1j*ky,M],[-M,kx-1j*ky]])
    np.testing.assert_allclose(n,expected,atol=3e-16)
    kxp,kyp=.31,-.09
    changed=lf_spin_numerator(x,kxp,kyp,M,plus=plus)
    assert np.trace(changed.conj().T@n)==pytest.approx(2*(M*M+kxp*kx+kyp*ky),abs=2e-15)
    assert np.linalg.norm(n-np.diag(np.diag(n)))>.3


def test_loop_slope_derivative_and_zero_external_mass_limit():
    slope=loop_slope(M,S)
    h=1e-5
    coarse=-lf_parameter_shape(M,S,h)/h;fine=-lf_parameter_shape(M,S,h/2)/(h/2)
    assert 2*fine-coarse==pytest.approx(slope,rel=2e-9)
    limiting=loop_slope(M,1e-10)
    assert limiting==pytest.approx(3/(24*np.pi**2*M*M),rel=1e-9)


def test_orbital_erasure_changes_transverse_loop_despite_charge_subtraction():
    complete=lf_overlap_shape(M,S,.1)
    erased=lf_overlap_shape(M,S,.1,orbital=False)
    assert abs(erased-complete)/abs(complete)>.1
    assert lf_overlap_shape(M,S,0.,orbital=False)==0


def test_multiplet_mixing_retains_offdiagonal_loop_and_shared_norm():
    g=np.array([3.,1.]);p1,p0=elastic_momenta(S,.1);unit=triangle_unit(M,p1,p0)
    current=np.einsum('i,j,m->mij',g,g,unit)
    angle=.43;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    rotated_g=rotation@g
    expected=np.einsum('i,j,m->mij',rotated_g,rotated_g,unit)
    np.testing.assert_allclose(np.array([rotation@a@rotation.T for a in current]),expected,atol=1e-16)
    assert np.linalg.norm(current[:,0,1])>.001
    zero=np.einsum('i,j,m->mij',g,g,triangle_unit(M,p0,p0))
    np.testing.assert_allclose(zero,2*p0[:,None,None]*np.outer(g,g)*bubble_vertex_norm(M,S),atol=1e-14)


def test_invalid_domains_and_no_timelike_or_threshold_extrapolation():
    with pytest.raises(ValueError): triangle_unit(M,[1.,0,0,0],[1.,0,0,0])
    with pytest.raises(ValueError): bubble_vertex_norm(M,4*M*M)
    with pytest.raises(ValueError): normalized_loop_shape(M,S,-.1)
    with pytest.raises(ValueError): lf_spin_numerator(0.,.2,.1,M)


def test_replay_separates_conditional_matching_from_physical_refit():
    from scripts.evaluate_quantum_collective_current import evaluate
    data=evaluate();json.dumps(data,allow_nan=False)
    assert not data['claims']['local_C_actually_refitted']
    assert not data['claims']['independent_quantum_LF_four_current']
    assert data['checks']['maximum_overlap_difference']<1e-10

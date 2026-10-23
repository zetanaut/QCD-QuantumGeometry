"""Finite-reference shape operators against independent continuum integrals."""
from dataclasses import replace
from math import factorial,pi
import numpy as np
import pytest
from scipy.integrate import quad
from deuteron_wigner.quantum.m2_gaussian_reference_shape import GaussianReferenceShape
from deuteron_wigner.quantum.m2_gaussian_field_regulator import gaussian_shell_integral
from deuteron_wigner.quantum.m2_gaussian_cluster_projector_kernel import GaussianHamiltonianOrbital


def shape(degree):
    return GaussianReferenceShape(base_width_GeV=.4,reference_width_GeV=.8,degree=degree)


@pytest.mark.parametrize("degree",[0,1,3,9])
def test_scalar_kernel_derivative_and_positive_reference_remainder(degree):
    s=shape(degree);q=np.linspace(0,20,71);z=(s.a-s.a_reference)*q
    expected=np.exp(-s.a*q)*sum(z**m/factorial(m) for m in range(degree+1))
    value,derivative=s.value_and_derivative(q)
    np.testing.assert_allclose(value,expected,atol=3e-16)
    previous=sum(z**m/factorial(m) for m in range(degree))
    ref=np.exp(-s.a*q)*((s.a-s.a_reference)*previous-s.a*sum(z**m/factorial(m) for m in range(degree+1)))
    np.testing.assert_allclose(derivative,ref,atol=2e-15)
    assert np.all(value>=0) and np.all(value<=np.exp(-s.a_reference*q)*(1+1e-15))
    assert np.all(value<=shape(degree+1).value_and_derivative(q)[0]*(1+1e-15))
    # The gauged kernel has the SAME integrable reference majorant. This
    # verifies the domination used before finite H2 loop/current products;
    # it does not assert a resolution-uniform bound as cutoffs are removed.
    assert np.all(derivative<=0)
    assert np.all(abs(derivative)<=s.a*np.exp(-s.a_reference*q)*(1+1e-15))
    assert s.value_and_derivative(np.array([1e10]))[0][0]==0.


@pytest.mark.parametrize("degree",[1,2,4])
def test_exact_ground_loop_integrals_and_derivative_against_independent_radial_quadrature(degree):
    s=shape(degree);b=.37;f=.7
    args=dict(out_cutoff=0,in_cutoff=0,out_map=np.eye(1),in_map=np.eye(1),
        b_GeV=b,field_precision=2*s.a,field_maps=np.array([[f]]),field_shifts=np.zeros((1,2)))
    def kernel(q):
        z=(s.a-s.a_reference)*q
        return np.exp(-s.a*q)*sum(z**m/factorial(m) for m in range(degree+1))
    # Two-dimensional normalized ground HO: integral dx exp(-x/b²)/b²,
    # where x=|qT|². The insertion includes a nonconstant bare numerator.
    ref=quad(lambda x:np.exp(-x/b**2)/b**2*(.3+.2*x)*kernel(f*f*x),0,np.inf,epsabs=1e-13)[0]
    got=s.integrate(**args,constant=.3,quadratic=.4*np.eye(2))[0,0]
    np.testing.assert_allclose(got,ref,atol=2e-14)
    # Independent integration-by-parts reference for integral F'(f²x).
    unit=s.integrate(**args)[0,0]
    reference_derivative=(unit-1)/(f*f*b*b)
    got_derivative=s.integrate(**args,gauged=True,constant=-s.a)[0,0]
    np.testing.assert_allclose(got_derivative,reference_derivative,atol=2e-14)


def test_whole_vertex_expansion_does_not_factor_into_independent_auxiliary_halves():
    s=shape(2);x,y=.23,.51
    whole=s.value_and_derivative(x+y)[0]
    incorrectly_factorized=s.value_and_derivative(x)[0]*s.value_and_derivative(y)[0]
    assert abs(whole-incorrectly_factorized)>.01
    # Even M=1 has a missing mixed term if expanded halves are multiplied.
    a=shape(1);delta=a.a-a.a_reference
    exact_difference=a.value_and_derivative(x)[0]*a.value_and_derivative(y)[0]-a.value_and_derivative(x+y)[0]
    np.testing.assert_allclose(exact_difference,np.exp(-a.a*(x+y))*delta**2*x*y,atol=2e-16)


def test_shifted_complex_orbital_uses_joint_insertions_and_preserves_reversal():
    args=dict(out_map=np.eye(1),in_map=np.eye(1),field_maps=np.array([[.7],[-.2]]),
        field_shifts=np.array([[.12,-.09],[-.04,.07]]),b_GeV=.37,
        field_precision=2*shape(2).a,measure=.23,constant=.3+.2j,
        linear=np.array([[.1j],[.2]]),quadratic=np.array([[.3,.2j],[.2j,-.1]]),
        reference_shape=shape(2))
    h=GaussianHamiltonianOrbital(**args);out=np.array([[.03,-.01]]);inc=np.array([[-.07,.04]])
    a=h.integrate(2,1,out_shift=out,in_shift=inc)
    rev=replace(h,constant=h.constant.conjugate(),linear=h.linear.conj(),quadratic=h.quadratic.conj())
    b=rev.integrate(1,2,out_shift=inc,in_shift=out)
    np.testing.assert_allclose(a,b.conj().T,atol=3e-16)
    bare=replace(h,reference_shape=None)
    old=replace(h,reference_shape=shape(0))
    np.testing.assert_array_equal(old.integrate(2,1),bare.integrate(2,1))


def test_nonconstant_matrix_shape_converges_to_independent_reference_gaussian():
    args=dict(out_cutoff=2,in_cutoff=2,out_map=np.eye(1),in_map=np.eye(1),
        out_shift=np.array([[.02,-.01]]),in_shift=np.array([[-.03,.01]]),
        b_GeV=.37,field_precision=2*shape(0).a,field_maps=np.array([[.7]]),
        field_shifts=np.array([[.06,-.04]]),constant=.3+.1j,linear=np.array([[.1],[.2j]]))
    reference=gaussian_shell_integral(**{**args,"field_precision":2*shape(0).a_reference})
    errors=[np.linalg.norm(shape(m).integrate(**args)-reference) for m in (0,1,2,4,8)]
    assert all(b<a for a,b in zip(errors,errors[1:]))
    assert errors[-1]<errors[0]*.001
    print("finite-reference full-matrix degree 0,1,2,4,8 errors",errors,flush=True)


def test_shape_rejects_unproved_reverse_series_and_mixed_width():
    for kwargs in (dict(degree=True),dict(degree=1.5),dict(degree=-1),
                   dict(reference_width_GeV=.2),dict(base_width_GeV=0)):
        with pytest.raises(ValueError):GaussianReferenceShape(**{**dict(base_width_GeV=.4,reference_width_GeV=.8,degree=2),**kwargs})
    with pytest.raises(ValueError):shape(2).value_and_derivative(-.1)
    with pytest.raises(ValueError):shape(2).integrate(field_precision=1.)


@pytest.mark.parametrize("p,k",[(1.,2.),(5.,2.),(2.,7.)])
@pytest.mark.parametrize("degree",[1,3])
def test_shaped_Wick_paths_against_independent_uncontracted_physical_leg_integrals(p,k,degree):
    from test_m2_gaussian_cluster_wick_current import direct_paths,regulator
    from deuteron_wigner.quantum.m2_gaussian_cluster_wick_current import cluster_contraction_paths
    s=shape(degree);reg=regulator(.4);delta=np.array([.12,.16])
    a=cluster_contraction_paths(regulator=reg,external_mode=p,loop_mode=k,transfer_GeV=delta,reference_shape=s)
    direct,transverse=direct_paths(p,k,delta,(.23,-.17),reg,reference_shape=s)
    np.testing.assert_allclose([a.external_mode_path_GeV2,a.loop_mode_path_GeV2],direct,atol=7e-14,rtol=3e-11)
    np.testing.assert_allclose(transverse,0.,atol=5e-14)
    coincident=reg.contraction_trace_GeV2(external_mode=p,loop_mode=k)
    ratio=1-s.a_reference/s.a
    np.testing.assert_allclose(a.coincident_trace_GeV2,coincident*sum(ratio**m for m in range(degree+1)),atol=3e-16)
    base=cluster_contraction_paths(regulator=reg,external_mode=p,loop_mode=k,transfer_GeV=delta)
    assert abs(a.base_trace_GeV2-base.base_trace_GeV2)>1e-5
    zero=cluster_contraction_paths(regulator=reg,external_mode=p,loop_mode=k,transfer_GeV=delta,reference_shape=shape(0))
    np.testing.assert_allclose(list(vars(zero).values()),list(vars(base).values()),atol=3e-16)


@pytest.mark.parametrize("species,p,e",[("q",1,2/3),("qbar",3,-2/3),("g",2,0.)])
def test_all_shaped_ordered_Wick_H_and_current_subtraction_partners_remain_paired(species,p,e):
    from test_m2_gaussian_cluster_wick import regulator
    from deuteron_wigner.quantum.m2_gaussian_cluster_wick import cluster_wick_summands
    from deuteron_wigner.quantum.m2_gaussian_cluster_wick_current import ordered_cluster_wick_terms
    from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION
    s=shape(3);reg=regulator(.4)
    h=cluster_wick_summands(species,p,regulator=reg,quark_flavors=2,reference_shape=s)
    terms=ordered_cluster_wick_terms(species,p,regulator=reg,quark_flavors=2,total_k2=9,
        total_pplus_GeV=1.3,transfer_GeV=(.12,.16),external_charge=e,
        loop_flavor_charges=(2/3,-1/3),prescription=PRESCRIPTION,reference_shape=s)
    for name,(modes,values) in h.items():
        for k,v in zip(modes,values):
            actual=sum(t.hamiltonian_per_gs2_GeV2 for t in terms if t.source==name and t.loop_mode==k)
            np.testing.assert_allclose(actual,9/pi*v,atol=3e-15)
    assert np.linalg.norm([t.minus for t in terms])>1e-5
    # Same composite subtraction removes all three current pieces, not
    # merely the canonical squared-denominator piece. No finite fit is made.
    for t in terms:
        assert t.minus+(-t.canonical_minus-t.external_leg_minus-t.internal_leg_minus)==pytest.approx(0.,abs=2e-17)

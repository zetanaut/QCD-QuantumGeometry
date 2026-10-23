import json
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import expm

from deuteron_wigner.quantum.quantum_collective_compton import (
    box_numerator,box_unit,compton_unit,compton_kinematics,
    scalar_photon_coefficient,scalar_photon_trace,photon_vacuum_kernel,
    scalar_exchange,scalar_exchange_matrix,dipole_estimate,
    absorption_kinematics,absorption_parts,absorption_cross_section,absorption_dipole_sum,
    dot,G5,slash,bubble_vertex_norm,triangle_unit,ALPHA_EM,HBARC,
)
from deuteron_wigner.quantum.common_action_pilot import calibrated_action
from deuteron_wigner.quantum.canonical_lf_fields import lf_spinor,antiquark_spinor,GAMMA
from deuteron_wigner.quantum.nonzero_momentum_current import lf_coordinates

M=.2843478671310942
MP=.13957039


def reference():
    a=calibrated_action(MP,.0922,.515,.004,.6,1.4,.12,[3.,1.])
    state=a.solve(.004)
    return a,state,state['pion_modes'][:,0]


@pytest.mark.parametrize('seed',[11,27,33])
def test_original_trace_Wick_contractions_vs_independent_angular_cubature(seed):
    rng=np.random.default_rng(seed);shifts=rng.normal(0,.2,(5,4,4))
    eps=rng.normal(size=(2,4))
    for vertices in [[1j*G5,slash(eps[0]),1j*G5,slash(eps[1])],
                     [1j*G5,1j*G5,slash(eps[0]),slash(eps[1])]]:
        a,b,c=box_numerator(shifts,M,vertices)
        aa,bb,cc=box_numerator(shifts,M,vertices,method='angular')
        np.testing.assert_allclose([a,b],[aa,bb],atol=2e-13,rtol=2e-12)
        np.testing.assert_allclose(cc,c,atol=2e-13,rtol=2e-12)


@pytest.mark.parametrize('scale',[.5,1.,1.4])
def test_all_six_box_orderings_satisfy_independent_triangle_Ward(scale):
    p0=np.array([.15,.01,.02,.03]);q1=scale*np.array([.04,.06,.01,.02])
    q2=np.array([.03,-.02,.04,.01]);p1=p0+q1+q2;eps=np.array([.1,.3,-.2,.4])
    b=box_unit(M,p1,p0,q1,q2,q1,eps)
    difference=dot(eps,triangle_unit(M,p1,p0+q1)-triangle_unit(M,p1-q1,p0))
    assert b==pytest.approx(difference,rel=2e-11,abs=2e-14)
    assert abs(b)>1e-4
    # A neutral q/antiquark pair has no equal-mass one-photon current but
    # still has a two-photon response. Its Ward contraction vanishes.
    assert abs(box_unit(M,p1,p0,q1,q2,q1,eps,charges=(.4,.4)))<2e-14


def test_soft_box_comes_from_inverse_second_derivative_not_a_contact_fit():
    p=np.array([.23,.04,.02,.03]);zero=np.zeros(4)
    e1=np.array([.3,.4,-.1,.2]);e2=np.array([.1,-.2,.3,.15]);s=dot(p,p)
    h=1e-6
    derivative=(bubble_vertex_norm(M,s+h)-bubble_vertex_norm(M,s-h))/(2*h)
    expected=2*dot(e1,e2)*bubble_vertex_norm(M,s)+4*dot(p,e1)*dot(p,e2)*derivative
    assert box_unit(M,p,p,zero,zero,e1,e2)==pytest.approx(expected,rel=2e-9,abs=1e-14)
    assert abs(box_unit(M,p,p,zero,zero,e1,e2,charges=(.4,.4)))<2e-14


def test_full_box_integral_independent_trace_and_resolution():
    p1,p0,k,kp=compton_kinematics(.04,1.1,MP);eps=np.array([0.,0,1,0])
    wick=box_unit(M,p1,p0,k,-kp,eps,eps,order=8)
    angular=box_unit(M,p1,p0,k,-kp,eps,eps,order=8,method='angular')
    refined=box_unit(M,p1,p0,k,-kp,eps,eps,order=16)
    assert wick==pytest.approx(angular,abs=4e-14)
    assert wick==pytest.approx(refined,abs=2e-11)


@pytest.mark.parametrize('angle',[.4,1.1,2.2])
def test_full_real_Compton_gauge_crossing_and_negative_Born_control(angle):
    p1,p0,k,kp=compton_kinematics(.025,angle,MP)
    e1=np.array([0.,1,0,0]);e2=np.array([0.,np.cos(angle),0.,-np.sin(angle)])
    original=compton_unit(M,p1,p0,k,-kp,e1,e2)
    shifted=compton_unit(M,p1,p0,k,-kp,e1+.7*k,e2-.9*kp)
    crossed=compton_unit(M,p1,p0,-kp,k,e2,e1)
    assert original['normalized_loop']==pytest.approx(shifted['normalized_loop'],abs=3e-14)
    assert original['normalized_loop']==pytest.approx(crossed['normalized_loop'],abs=3e-14)
    assert original['tree']==pytest.approx(shifted['tree'],abs=3e-14)
    # The box alone is NOT gauge invariant. Neither Ward nor the convenient
    # initial-rest radiation gauge authorizes dropping the Born diagrams.
    assert abs(original['box']-shifted['box'])>1e-5
    for ea,eb in [(k,e2),(e1,-kp)]:
        gauge=compton_unit(M,p1,p0,k,-kp,ea,eb)
        assert abs(gauge['normalized_loop'])<2e-14
        assert abs(gauge['box'])>1e-5


def test_Lorentz_boost_of_complete_loop_and_normalization():
    p1,p0,k,kp=compton_kinematics(.02,1.2,MP)
    e1=np.array([0.,1,0,0]);e2=np.array([0.,np.cos(1.2),0.,-np.sin(1.2)])
    gen=np.zeros((4,4));gen[0,1:]=gen[1:,0]=[.3,-.25,.15];boost=expm(gen)
    original=compton_unit(M,p1,p0,k,-kp,e1,e2)
    moved=compton_unit(M,boost@p1,boost@p0,boost@k,-boost@kp,boost@e1,boost@e2)
    assert original['normalized_loop']==pytest.approx(moved['normalized_loop'],abs=4e-14)
    assert original['tree']==pytest.approx(moved['tree'],abs=4e-14)
    assert abs(original['box'])>100*abs(original['normalized_loop'])
    assert abs(original['normalized_loop'])<1e-4


@pytest.mark.parametrize('energy,angle',[(.015,.4),(.04,1.2),(.05,2.)])
def test_scalar_real_photon_triangle_original_trace_vs_reduced(energy,angle):
    p1,p0,k,kp=compton_kinematics(energy,angle,MP);eps=np.array([0.,0,1,0])
    t=dot(k-kp,k-kp);invariant=dot(k,kp)*dot(eps,eps)-dot(k,eps)*dot(kp,eps)
    expected=4*scalar_photon_coefficient(M,t)*invariant
    assert scalar_photon_trace(M,k,-kp,eps,eps)==pytest.approx(expected,rel=2e-11,abs=2e-14)
    assert scalar_photon_trace(M,-kp,k,eps,eps)==pytest.approx(expected,abs=2e-14)
    e2=np.array([0.,np.cos(angle),0.,-np.sin(angle)])
    assert abs(scalar_photon_trace(M,k,-kp,k,e2))<2e-14
    assert abs(scalar_photon_trace(M,k,-kp,eps,-kp))<2e-14


def test_scalar_triangle_low_energy_theorem_from_fixed_subtraction_mass_derivative():
    a0=3*(5/9)/(24*np.pi**2*M)
    assert scalar_photon_coefficient(M,0.)==pytest.approx(a0,rel=2e-14)
    q=np.array([.04,.1,.03,.07]);e1=np.array([0.,1,0,0]);e2=np.array([0.,.7,0.,-.3]);h=1e-6
    independent=(photon_vacuum_kernel(M+h,M,q,e1,e2)-photon_vacuum_kernel(M-h,M,q,e1,e2))/(2*h)
    direct=scalar_photon_trace(M,q,-q,e1,e2)
    assert direct==pytest.approx(independent,rel=1e-9)
    wrong=(photon_vacuum_kernel(M+h,M+h,q,e1,e2)-photon_vacuum_kernel(M-h,M-h,q,e1,e2))/(2*h)
    assert abs(direct-wrong)>10*abs(wrong)


def test_scalar_potential_third_derivative_full_modes_and_basis_transport():
    a,state,e=reference();v=state['vacuum'];hessian=a.matrices(v)[1]
    h=2e-4;third=[]
    for i in range(2):
        values=[]
        for sign in [-1,1]:
            fields=np.zeros((2,4));fields[:,0]=v;fields[i,0]+=sign*h
            z=a.potential(fields,.004)
            plus=fields.copy();minus=fields.copy();plus[:,1]+=h*e;minus[:,1]-=h*e
            values.append((a.potential(plus,.004)+a.potential(minus,.004)-2*z)/h**2)
        third.append((values[1]-values[0])/(2*h))
    expected=2*a.quartic*v*e*e
    np.testing.assert_allclose(third,expected,rtol=3e-9,atol=3e-9)
    mass=state['quark_mass'];t=-.04;vertex=expected
    result=scalar_exchange(a,v,e,e,mass,t)
    r=np.array([[.6,-.8],[.8,.6]])
    moved=scalar_photon_coefficient(mass,t)*(r@a.yukawa)@np.linalg.solve(r@hessian@r.T-t*np.eye(2),r@vertex)
    assert moved==pytest.approx(result,rel=2e-14)
    matrix=scalar_exchange_matrix(a,v,mass,t)
    assert e@matrix@e==pytest.approx(result)
    assert np.linalg.matrix_rank(matrix)==2
    poles,modes=np.linalg.eigh(hessian)
    pieces=scalar_photon_coefficient(mass,t)*(a.yukawa@modes)*(vertex@modes)/(poles-t)
    assert pieces.sum()==pytest.approx(result)
    assert min(abs(pieces))>.1*abs(result)


def test_quantum_matching_adds_nonzero_alpha_plus_beta_not_encoded_by_C_D():
    a,state,e=reference();g2=(e@a.yukawa)**2
    coarse=dipole_estimate(M,MP,.005,yukawa_squared=g2)
    fine=dipole_estimate(M,MP,.0025,yukawa_squared=g2)
    assert fine['alpha_plus_beta']==pytest.approx(2.3972971166e-5,rel=3e-8)
    assert fine['alpha_plus_beta']==pytest.approx(coarse['alpha_plus_beta'],rel=4e-6)
    b=scalar_exchange(a,state['vacuum'],e,e,M,0.)
    scalar_alpha=2*ALPHA_EM*HBARC**3/MP*b
    assert scalar_alpha==pytest.approx(.0006305848505083,rel=2e-13)
    assert scalar_alpha+fine['alpha']>0
    assert -scalar_alpha+fine['beta']<0
    # C and D change alpha-beta but neither can erase alpha+beta in the
    # retained local derivative basis. Do not promote the old conditional
    # COMPASS alpha=-beta extraction to two independent measured inputs.
    assert fine['alpha']+fine['beta']>2e-5


def test_neutral_pair_two_photon_response_is_not_inferred_from_its_zero_current():
    p1,p0,k,kp=compton_kinematics(.025,1.3,MP);eps=np.array([0.,0,1,0])
    assert abs(box_unit(M,p1,p0,k,-kp,eps,eps,charges=(.4,.4)))>1e-5


def test_input_guards():
    p1,p0,k,kp=compton_kinematics(.02,1.,MP);eps=np.array([0.,0,1,0])
    with pytest.raises(ValueError): box_unit(-M,p1,p0,k,-kp,eps,eps)
    with pytest.raises(ValueError): box_unit(M,p1+.1,p0,k,-kp,eps,eps)
    with pytest.raises(ValueError): box_unit(M,3*p1,3*p0,3*k,-3*kp,eps,eps)
    with pytest.raises(ValueError): scalar_photon_coefficient(M,4*M*M)
    with pytest.raises(ValueError): scalar_photon_trace(M,np.array([.1,0,0,0]),-kp,eps,eps)
    with pytest.raises(ValueError): photon_vacuum_kernel(M,0.,k,eps,eps)
    with pytest.raises(ValueError): dipole_estimate(M,MP,.01,yukawa_squared=-1)
    with pytest.raises(ValueError): compton_kinematics(0.,1.,MP)
    with pytest.raises(ValueError): absorption_cross_section(M,MP,4*M*M)


@pytest.mark.parametrize('s,cosine',[(.5,-.4),(1.2,.7)])
def test_absorption_independent_canonical_spinors_and_all_source_attachments(s,cosine):
    p,k,q,a=absorption_kinematics(M,MP,s,cosine)
    u=np.column_stack([lf_spinor(*lf_coordinates(q),M,h) for h in [-1,1]])
    v=np.column_stack([antiquark_spinor(*lf_coordinates(a),M,h) for h in [-1,1]])
    for eps in [np.array([0.,1,0,0]),np.array([0.,0,1,0])]:
        parts=absorption_parts(M,MP,s,cosine,eps);vertex=parts['total']
        resolved=u.conj().T@GAMMA[0]@vertex@v
        traced=np.trace((slash(q)+M*np.eye(4))@vertex@(slash(a)-M*np.eye(4))@GAMMA[0]@vertex.conj().T@GAMMA[0])
        assert np.sum(abs(resolved)**2)==pytest.approx(traced,abs=2e-13)
        moved=u.conj().T@GAMMA[0]@absorption_parts(M,MP,s,cosine,eps+.7*k)['total']@v
        np.testing.assert_allclose(moved,resolved,atol=2e-14)
    gauge=absorption_parts(M,MP,s,cosine,k)
    np.testing.assert_allclose(u.conj().T@GAMMA[0]@gauge['total']@v,0,atol=2e-14)
    assert np.linalg.norm(u.conj().T@GAMMA[0]@(gauge['quark']+gauge['antiquark'])@v)>.5


def test_forward_dipole_sum_from_independent_positive_absorption_not_box_reuse():
    g2=9.245548275808696
    coarse=absorption_dipole_sum(M,MP,order=48,angular_order=96,yukawa_squared=g2)
    fine=absorption_dipole_sum(M,MP,order=144,angular_order=288,yukawa_squared=g2)
    assert fine>0
    assert abs(coarse-fine)<2e-11
    first=dipole_estimate(M,MP,.005,yukawa_squared=g2)['alpha_plus_beta']
    second=dipole_estimate(M,MP,.0025,yukawa_squared=g2)['alpha_plus_beta']
    extrapolated=(4*second-first)/3
    assert extrapolated==pytest.approx(fine,abs=2e-12)
    for s in [.35,.5,1.,3.]: assert absorption_cross_section(M,MP,s)>0


def test_saved_scientific_replay():
    from scripts.evaluate_quantum_collective_compton import evaluate
    saved=json.loads(Path('docs/next_level/quantum_collective_compton_results.json').read_text())
    assert evaluate()==saved

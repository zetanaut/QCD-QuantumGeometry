"""Independent algebra, current and inference controls for spinful selection."""

import numpy as np
import pytest

from deuteron_wigner.quantum.spinful_selection_controls import (
    ETA,GAMMA,IDENTITY,slash,dot,spinors,momenta,bilinear,extract,loop_vertex,
    parameter_form_factors,lightfront_form_factors,lf_amplitudes,
    finite_cloud_probability,inverse_propagator,ward_vertex,
)
from scripts.evaluate_spinful_selection import preferred_frame_vertex,inference_experiment


@pytest.mark.parametrize("boost", [(0.,0.,0.),(.2,-.1,.3),(-.15,.3,-.2)])
def test_spinors_dirac_norm_completeness_and_metric(boost):
    po,pi = momenta(1.3,boost,.7)
    for p in (po,pi):
        u = spinors(p)
        assert dot(p,p) == pytest.approx(1,abs=2e-14)
        assert np.linalg.norm((slash(p)-IDENTITY)@u) < 1e-13
        assert np.linalg.norm(u.conj().T@GAMMA[0]@u-2*np.eye(2)) < 1e-13
        assert np.linalg.norm(u@u.conj().T@GAMMA[0]-(slash(p)+IDENTITY)) < 1e-13
    for mu in range(4):
        for nu in range(4):
            expected = 2*ETA[mu]*IDENTITY if mu==nu else np.zeros((4,4))
            assert np.linalg.norm(GAMMA[mu]@GAMMA[nu]+GAMMA[nu]@GAMMA[mu]-expected) < 1e-14


@pytest.mark.parametrize("mu,q2", [(.35,.05),(.35,4.),(.8,.25),(1.6,1.)])
def test_loop_numerator_all_components_and_independent_overlap(mu,q2):
    po,pi = momenta(q2,(.23,-.12,.17))
    current = bilinear(loop_vertex(po,pi,mu),po,pi)
    independent = lightfront_form_factors(mu,q2,order=96)
    reduced = parameter_form_factors(mu,q2)
    for components in ((0,),(1,),(2,3),(0,1,2,3)):
        f,residual,_ = extract(current,po,pi,components)
        assert np.max(abs(f-independent)) < 1e-8
        assert np.max(abs(f-reduced)) < 1e-11
        assert residual < 1e-12
    assert np.linalg.norm(np.einsum("i,ijk->jk",(po-pi)*ETA,current)) < 1e-12


def test_frame_independent_form_factors_without_fitting_them_into_current():
    values = []
    for boost in ((.23,-.12,.17),(-.15,.22,-.1),(.31,.11,.27)):
        po,pi = momenta(1.7,boost,.83)
        j = bilinear(loop_vertex(po,pi,.6),po,pi)
        values.append(extract(j,po,pi)[0])
    assert np.max(abs(np.array(values)-values[0])) < 1e-12


@pytest.mark.parametrize("mu", [.35,.8,1.,1.6])
def test_charge_subtraction_and_exact_equal_mass_magnetic_anchor(mu):
    f = lightfront_form_factors(mu,0,order=96)
    assert f[0] == 1
    assert abs(f[1]-parameter_form_factors(mu,0)[1]) < 1e-12
    po,pi = momenta(0,(.2,.1,.3))
    corrected = bilinear(loop_vertex(po,pi,mu),po,pi)
    assert np.linalg.norm(corrected-bilinear(GAMMA,po,pi)) < 1e-12
    if mu==1:
        assert f[1] == pytest.approx(.01,abs=1e-13)


def test_resolution_and_zero_coupling():
    assert np.max(abs(lightfront_form_factors(.35,4,order=64)-lightfront_form_factors(.35,4,order=96))) < 1e-7
    po,pi = momenta(4,(.23,-.12,.17))
    assert np.linalg.norm(loop_vertex(po,pi,.35,order=48)-loop_vertex(po,pi,.35,order=80)) < 1e-12
    assert np.max(abs(loop_vertex(po,pi,.35,kappa=0)-np.array(GAMMA))) == 0
    assert np.max(abs(lightfront_form_factors(.35,4,kappa=0)-[1,0])) == 0


@pytest.mark.parametrize("x,kx,ky", [(.2,.3,-.4),(.7,-.5,.2),(.5,.1,.2)])
def test_amplitudes_match_local_scalar_dirac_vertex_and_Jz(x,kx,ky):
    # Parent at rest, daughter on shell with p+ fraction x; LF energy need not conserve at vertex.
    parent = np.array([1.,0.,0.,0.])
    pp = x/np.sqrt(2)
    pm = (1+kx*kx+ky*ky)/(2*pp)
    daughter = np.array([(pp+pm)/np.sqrt(2),kx,ky,(pp-pm)/np.sqrt(2)])
    scalar_vertex = spinors(daughter).conj().T@GAMMA[0]@spinors(parent)
    energy_den = 1-(kx*kx+ky*ky+1)/x-(kx*kx+ky*ky+.5**2)/(1-x)
    direct = scalar_vertex/(np.sqrt(x*(1-x))*energy_den)
    reference = lf_amplitudes(x,kx,ky,.5)
    assert np.max(abs(direct-reference)) < 1e-12
    theta = .73
    rotated = lf_amplitudes(x,kx*np.cos(theta)-ky*np.sin(theta),kx*np.sin(theta)+ky*np.cos(theta),.5)
    orbital = np.array([[0,-1],[1,0]])
    assert np.max(abs(rotated-np.exp(1j*orbital*theta)*reference)) < 1e-12


def test_charge_and_Jz_alone_do_not_retain_magnetic_interference():
    assert lightfront_form_factors(.5,0,orbital=False)[0] == 1
    missing = lightfront_form_factors(.5,1,orbital=False)
    full = lightfront_form_factors(.5,1)
    assert missing[1] == 0
    assert full[1].real > .01
    assert abs(missing[0]-full[0]) > 1e-4


def test_current_hermiticity_under_reversal():
    po,pi = momenta(1,(.2,-.1,.3),.8)
    j = bilinear(loop_vertex(po,pi,.5),po,pi)
    reversed_j = bilinear(loop_vertex(pi,po,.5),pi,po)
    assert np.linalg.norm(j-reversed_j.conj().transpose(0,2,1)) < 1e-12


@pytest.mark.parametrize("transverse", [-.03,0.,.03])
def test_off_shell_Ward_vertex_and_transverse_freedom(transverse):
    po,pi = np.array([.7,.2,-.4,.1]),np.array([1.3,-.2,.1,.4])
    target = inverse_propagator(po)-inverse_propagator(pi)
    actual = np.einsum("i,ijk->jk",(po-pi)*ETA,ward_vertex(po,pi,transverse=transverse))
    assert np.linalg.norm(actual-target) < 1e-13
    assert np.linalg.norm(slash(po-pi)-target) > .01
    po,pi = momenta(1,(.2,-.1,.3))
    f,residual,_ = extract(bilinear(ward_vertex(po,pi,transverse=transverse),po,pi),po,pi)
    assert np.max(abs(f-[1,transverse])) < 1e-12
    assert residual < 1e-12


def test_Ward_equal_virtuality_limit_is_nonsingular():
    pi = np.array([.8,.2,.1,.3])
    po = np.array([.8,-.2,-.1,-.3])
    vertex = ward_vertex(po,pi)
    assert np.all(np.isfinite(vertex))
    assert np.linalg.norm(np.einsum("i,ijk->jk",(po-pi)*ETA,vertex)-inverse_propagator(po)+inverse_propagator(pi)) < 1e-13


def test_conserved_non_covariant_negative_is_detected():
    po,pi = momenta(1,(.23,-.12,.17))
    j = bilinear(loop_vertex(po,pi,.5)+preferred_frame_vertex(po,pi),po,pi)
    assert np.linalg.norm(np.einsum("i,ijk->jk",(po-pi)*ETA,j)) < 1e-12
    plus = extract(j,po,pi,(0,))[0]
    transverse = extract(j,po,pi,(2,3))[0]
    assert np.max(abs(plus-transverse)) > 1e-3
    assert extract(j,po,pi)[1] > 1e-3


def test_two_body_norm_is_cutoff_dependent_not_normalized_continuum_state():
    values = [finite_cloud_probability(.5,k) for k in (2,8,32,128)]
    assert all(b>a for a,b in zip(values,values[1:]))
    assert values[-1]-values[-2] == pytest.approx(.01*np.log(4),abs=1e-5)


def test_synthetic_closure_and_magnetic_information_not_physical_fit():
    result = inference_experiment()
    assert result["physical_data_used"] is False
    assert result["charge_only_response_rank"] == 0
    electric,joint = result["results"]["F1_only"],result["results"]["joint_F1_F2"]
    assert electric["noiseless_parameter_error"] < 1e-9
    assert joint["noiseless_parameter_error"] < 1e-10
    assert electric["condition"] > 100*joint["condition"]
    assert joint["withheld_F2_RMSE"] < electric["withheld_F2_RMSE"]


def test_invalid_or_unidentifiable_inputs():
    with pytest.raises(ValueError):
        parameter_form_factors(0,1)
    with pytest.raises(ValueError):
        lightfront_form_factors(.5,-1)
    po,pi = momenta(0)
    with pytest.raises(ValueError):
        extract(bilinear(GAMMA,po,pi),po,pi)
    with pytest.raises(ValueError):
        loop_vertex(po+np.array([.1,0,0,0]),pi,.5)

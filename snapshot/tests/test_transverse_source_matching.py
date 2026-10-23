import numpy as np
import pytest
import torch
from scipy.linalg import expm

from deuteron_wigner.quantum.transverse_source_matching import (
    SourceCurvature, connection, curvature, source_density, tensor,
    mixed_derivative, action_vertex, action_contact, transverse_vertex,
    dot, METRIC, HBARC, ALPHA_EM, radius_coefficient, polarizability, match_projected,
)
from deuteron_wigner.quantum.common_action_pilot import calibrated_action


E = np.array([.8, .6])


def model():
    return SourceCurvature(np.array([[.7, .1], [.1, .4]]), np.array([[-.2, .07], [.07, .3]]))


def jets(seed):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, .2, (2, 4)); df = rng.normal(0, .3, (4, 2, 4))
    a = rng.normal(0, .1, (4, 4, 4)); a -= a.transpose(0, 2, 1)
    da = rng.normal(0, .2, (4, 4, 4, 4)); da -= da.transpose(0, 1, 3, 2)
    return f, df, a, da


def boost(beta):
    beta = np.asarray(beta); b2 = beta@beta; gamma = 1/np.sqrt(1-b2)
    result = np.eye(4); result[0, 0] = gamma
    result[0, 1:] = result[1:, 0] = gamma*beta
    result[1:, 1:] += (gamma-1)*np.outer(beta, beta)/b2
    return result


@pytest.mark.parametrize('seed', [23, 43, 67])
def test_full_local_connection_covariance_not_partial_curvature(seed):
    f, df, a, da = jets(seed); rng = np.random.default_rng(seed+1)
    z = rng.normal(size=(4, 4)); r = expm(.2*(z-z.T))
    omega = rng.normal(0, .1, (4, 4, 4)); omega -= omega.transpose(0, 2, 1)
    b = r@a@r.T; moved_a = b-omega
    moved_da = np.empty_like(da)
    for mu in range(4):
        for nu in range(4):
            moved_da[mu, nu] = r@da[mu, nu]@r.T+omega[mu]@b[nu]-b[nu]@omega[mu]-.5*(omega[mu]@omega[nu]-omega[nu]@omega[mu])
    moved_f = f@r.T
    moved_df = np.einsum('ab,mib->mia', r, df)+np.einsum('mab,ib->mia', omega, moved_f)
    np.testing.assert_allclose(curvature(moved_a, moved_da), r@curvature(a, da).numpy()@r.T, atol=2e-14)
    np.testing.assert_allclose(model().density(moved_f, moved_df, moved_a, moved_da), model().density(f, df, a, da), atol=2e-14)
    # Omitting the non-Abelian commutator loses covariance, even though
    # an Abelian-only electromagnetic test would not expose this omission.
    naive = moved_da-moved_da.transpose(1, 0, 2, 3)
    old_naive = da-da.transpose(1, 0, 2, 3)
    assert np.linalg.norm(naive-r@old_naive@r.T) > .01


@pytest.mark.parametrize('beta', [[.2, -.1, .15], [-.1, .25, .2]])
def test_action_density_and_derived_four_current_transform_covariantly(beta):
    f, df, a, da = jets(40); upper = boost(beta)
    lower = METRIC[:, None]*upper*METRIC[None, :]
    moved = (f, np.einsum('mn,nia->mia', lower, df),
             np.einsum('mn,nab->mab', lower, a),
             np.einsum('mr,ns,rsab->mnab', lower, lower, da))
    np.testing.assert_allclose(model().density(*moved), model().density(f, df, a, da), atol=2e-14)
    p0 = np.array([.3, .1, .03, .02]); p1 = np.array([.4, -.02, .15, -.05])
    j = action_vertex(model(), p1, p0, E, E)
    np.testing.assert_allclose(action_vertex(model(), upper@p1, upper@p0, E, E), upper@j, atol=2e-13)


@pytest.mark.parametrize('p1', [[.4, -.2, .1, .2], [.3, .2, -.1, .1], [.2, 0., .1, .3]])
def test_offshell_current_is_action_derivative_and_exactly_transverse(p1):
    p0 = np.array([.25, .02, .04, .12]); p1 = np.array(p1)
    c, _ = model().projected(E)
    actual = action_vertex(model(), p1, p0, E, E)
    np.testing.assert_allclose(actual, transverse_vertex(p1, p0, c), atol=2e-14)
    assert abs(dot(p1-p0, actual)) < 2e-14
    # An on-shell form-factor multiplier used off shell misses the q^mu term.
    wrong = c*dot(p1-p0, p1-p0)*(p1+p0)
    assert abs(dot(p1-p0, wrong)) > 1e-5


@pytest.mark.parametrize('q2', [.0, .02, .1])
def test_radius_matching_and_D_invisible_to_one_photon(q2):
    c, d = match_projected(.659, 2e-4, .13957039)
    a = SourceCurvature(c*np.outer(E, E), d*np.outer(E, E))
    q = np.sqrt(q2); energy = np.sqrt(.13957039**2+q2/4)
    p0 = np.array([energy, -q/2, 0, 0]); p1 = np.array([energy, q/2, 0, 0])
    correction = action_vertex(a, p1, p0, E, E)
    assert 1+correction[0]/(2*energy) == pytest.approx(1-c*q2, abs=2e-14)
    only_d = SourceCurvature(np.zeros((2, 2)), a.d)
    np.testing.assert_allclose(action_vertex(only_d, p1, p0, E, E), 0, atol=1e-15)
    assert 6*HBARC**2*c == pytest.approx(.659**2)


@pytest.mark.parametrize('scale', [.6, 1., 1.7])
def test_two_source_Ward_relation_requires_actual_contact(scale):
    p0 = np.array([.3, .1, -.02, .04])
    q1 = scale*np.array([.08, -.03, .04, .05]); q2 = np.array([.04, .07, -.01, .03])
    p1 = p0+q1+q2; eps2 = np.array([.1, .3, -.2, .4])
    contact = action_contact(model(), p1, p0, q1, q2, q1, eps2, E, E)
    c, _ = model().projected(E)
    difference = transverse_vertex(p1, p0+q1, c)-transverse_vertex(p1-q1, p0, c)
    np.testing.assert_allclose(contact, dot(eps2, difference), atol=2e-14)
    assert abs(contact) > 1e-5  # Dropping the induced contact would fail.


def compton_kinematics(w, theta, mass=.13957039):
    k = np.array([w, 0, 0, w])
    wp = w/(1+w/mass*(1-np.cos(theta)))
    kp = wp*np.array([1, np.sin(theta), 0, np.cos(theta)])
    p0 = np.array([mass, 0, 0, 0]); p1 = p0+k-kp
    return p1, p0, k, kp


@pytest.mark.parametrize('w,theta', [(.01, .3), (.03, .8), (.04, 1.4)])
def test_real_Compton_response_independent_of_radius_alone(w, theta):
    p1, p0, k, kp = compton_kinematics(w, theta)
    eps = np.array([0, 0, 1, 0]); c, d = model().projected(E)
    assert dot(p1, p1) == pytest.approx(dot(p0, p0))
    invariant = dot(k, kp)*dot(eps, eps)-dot(k, eps)*dot(kp, eps)
    actual = action_contact(model(), p1, p0, k, -kp, eps, eps, E, E)
    np.testing.assert_allclose(actual, -4*(c+2*d)*invariant, atol=2e-14)
    gauge = action_contact(model(), p1, p0, k, -kp, k, eps, E, E)
    assert abs(gauge) < 1e-14
    # The transverse correction at a real physical photon vanishes even
    # when its intermediate pion leg is off shell; mixed Born terms vanish.
    assert abs(dot(eps, transverse_vertex(p0+k, p0, c))) < 1e-14
    missing_d = SourceCurvature(model().c, np.zeros((2, 2)))
    assert abs(actual-action_contact(missing_d, p1, p0, k, -kp, eps, eps, E, E)) > 1e-6


def test_polarizability_normalization_and_restricted_physical_failure():
    c = radius_coefficient(.659); mass = .13957039
    assert polarizability(c, 0, mass) == pytest.approx(.001493509818747316)
    c, d = match_projected(.659, 2e-4, mass)
    assert polarizability(c, d, mass) == pytest.approx(2e-4)
    # F^2=2(B^2-E^2), pi_rel=exp(-imt)*psi/sqrt(2m).
    # L_NR=2*pi*(alpha E^2+beta B^2)*psi^dagger psi.
    e_squared = 4*np.pi*ALPHA_EM
    assert e_squared*(c+2*d)/mass == pytest.approx(2*np.pi*2e-4/HBARC**3)


def test_zero_source_vacuum_scalar_and_PCAC_are_unchanged():
    a = calibrated_action(.1348, .0922, .515, .004, .6, 1.4, .12, [3., 1.])
    s = a.solve(.004); f, df, _, _ = jets(82)
    zero = np.zeros((4, 4, 4)); dzero = np.zeros((4, 4, 4, 4))
    x = torch.tensor(f, requires_grad=True)
    density = model().density(x, df, zero, dzero).real
    assert density.item() == 0
    np.testing.assert_array_equal(torch.autograd.grad(density, x)[0], 0)
    assert a.scalar_radius_squared(s['vacuum'], s['pion_modes'][:, 0]) == pytest.approx(.515)
    np.testing.assert_allclose(s['pion_mass_squared']*s['decay_constants'], .008*s['pseudoscalar_residues'], atol=1e-12)


def test_functional_subtraction_transports_both_derivative_orders():
    full = model(); kept = SourceCurvature(.35*full.c, .6*full.d); rest = full.residual(kept)
    f, df, a, da = jets(80)
    np.testing.assert_allclose(full.density(f, df, a, da), kept.density(f, df, a, da)+rest.density(f, df, a, da), atol=2e-14)
    p1,p0,k,kp = compton_kinematics(.04, .8); eps=np.array([0,0,1,0])
    for operator in [lambda m: action_vertex(m,p1,p0,E,E), lambda m: action_contact(m,p1,p0,k,-kp,eps,eps,E,E)]:
        np.testing.assert_allclose(operator(full),operator(kept)+operator(rest),atol=2e-14)
    wrong = SourceCurvature(rest.c, full.d)  # Double counts retained D.
    assert abs(action_contact(wrong,p1,p0,k,-kp,eps,eps,E,E)-action_contact(rest,p1,p0,k,-kp,eps,eps,E,E)) > 1e-5


def test_inter_node_basis_transport_and_unidentified_transitions():
    f, df, a, da = jets(91); r = np.array([[.6,-.8],[.8,.6]])
    m=model(); moved=SourceCurvature(r@m.c@r.T,r@m.d@r.T)
    np.testing.assert_allclose(moved.density(r@f,np.einsum('ij,mja->mia',r,df),a,da),m.density(f,df,a,da),atol=2e-14)
    np.testing.assert_allclose(moved.projected(r@E),m.projected(E),atol=1e-14)
    h=np.array([-E[1],E[0]]); null=np.outer(E,h)+np.outer(h,E)
    different=SourceCurvature(m.c+.4*null,m.d)
    np.testing.assert_allclose(different.projected(E),m.projected(E),atol=1e-14)
    p0=np.array([.3,.1,.02,.04]);p1=np.array([.5,-.2,.03,.06])
    assert np.linalg.norm(action_vertex(different,p1,p0,h,E)-action_vertex(m,p1,p0,h,E)) > .01
    # In the fixed two-multiplet basis, radius and alpha constrain rank 2
    # among six symmetric C,D coefficients. Coordinate changes are distinct.
    weights=np.array([E[0]**2,2*E[0]*E[1],E[1]**2])
    response=np.array([np.r_[weights,0*weights],np.r_[weights,2*weights]])
    assert np.linalg.matrix_rank(response)==2


def test_Torch_parameter_gradients_of_the_full_source_action():
    f,df,a,da=jets(13)
    x=torch.tensor([.7,.1,.4,-.2,.07,.3],dtype=torch.float64,requires_grad=True)
    def expression(y):
        c=torch.stack([y[0],y[1],y[1],y[2]]).reshape(2,2)
        d=torch.stack([y[3],y[4],y[4],y[5]]).reshape(2,2)
        return source_density(f,df,a,da,c,d).real
    gradient=torch.autograd.grad(expression(x),x)[0].numpy()
    independent=np.array([expression(torch.eye(6,dtype=torch.float64)[i]).item() for i in range(6)])
    np.testing.assert_allclose(gradient,independent,atol=2e-14)
    assert np.linalg.norm(gradient)>1e-4


def test_invalid_coefficient_or_momentum_inputs_are_not_silently_accepted():
    with pytest.raises(ValueError): SourceCurvature([[1,1],[0,1]],np.eye(2))
    with pytest.raises(ValueError): polarizability(1,1,0)
    with pytest.raises(ValueError): mixed_derivative(lambda x:x[0],5)
    with pytest.raises(ValueError): action_contact(model(),np.ones(4),np.zeros(4),np.zeros(4),np.zeros(4),E,E,E,E)


def test_replay_distinguishes_restricted_failure_calibration_and_remaining_rank():
    import json
    from scripts.evaluate_transverse_source_matching import evaluate
    result = evaluate()
    json.dumps(result, allow_nan=False)
    assert result['radius_only']['alpha_fm3'] > .0014
    assert result['joint_matching']['alpha_fm3'] == pytest.approx(2e-4)
    assert result['matching_rank'] == 2 and result['unidentified_directions'] == 4
    for case in result['cases']:
        assert case['action_vertex_vs_analytic_max_error_GeV'] < 1e-12
        assert case['real_Compton_action_vs_reduced_max_error'] < 1e-12
        assert case['alternative_same_light_projection_max_error'] < 1e-12
        assert case['heavy_light_offshell_current_change_norm_GeV'] > .04

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import block_diag, expm
from scipy.optimize import minimize, minimize_scalar

from deuteron_wigner.quantum.chiral_soft_matching import G5
from deuteron_wigner.quantum.modular_gauge_network import generators, su3
from deuteron_wigner.quantum.stable_connection_sea import (
    BoundedKineticGate, EGAMMA, PAULI, StableConnectionSea, chiral_matrix,
    connection_profile, dirac_jet, face_action_jet, full_logdet,
    negative_logdet_jet, scalar_texture_matrix, sea_kernel,
    retained_dirac_jet, dirac_determinant_jet,
    texture_hamiltonian,
)


@pytest.mark.parametrize('eta', [-8.,-1.,0.,1.,8.])
def test_uniform_electric_and_magnetic_bounds(eta):
    gate = BoundedKineticGate(eta=eta)
    lo, hi = gate.bounds()
    assert gate.jet(gate.rho_reference)[0] == pytest.approx(1.)
    rng = np.random.default_rng(91)
    p, b = rng.normal(size=(2,3,8))
    norm = np.sum(p*p+b*b)
    for rho in np.r_[0.,np.logspace(-8,8,21)]:
        k = gate.jet(rho)[0]
        assert lo <= k <= hi
        e = gate.energy(rho,p,b)
        assert .5*min(1/hi,lo)*norm <= e <= .5*max(1/lo,hi)*norm


def test_unconstrained_negative_gate_rejected_by_energy_not_rescaled():
    # kappa=1-rho: either electric or magnetic energy runs to -infinity.
    kappa = 1.-2.
    assert kappa*100**2/2 < kappa*10**2/2 < 0
    assert BoundedKineticGate().energy(2.,np.zeros(8),np.ones(8)*100) > 0


def test_gate_derivatives_and_canonical_electric_source():
    gate, rho, h = BoundedKineticGate(), .7, 1e-4
    k, dk, ddk = gate.jet(rho)
    a, c = gate.jet(rho-h)[0], gate.jet(rho+h)[0]
    assert dk == pytest.approx((c-a)/(2*h),abs=3e-9)
    assert ddk == pytest.approx((c-2*k+a)/h**2,abs=8e-8)
    p, b = np.ones((3,8))*.8, np.ones((3,8))*.3
    derivative = .5*dk*(np.sum(b*b)-np.sum(p*p)/k**2)
    finite = (gate.energy(rho+h,p,b)-gate.energy(rho-h,p,b))/(2*h)
    assert derivative == pytest.approx(finite,abs=3e-8)
    assert abs(derivative-.5*dk*np.sum(b*b)) > .5  # magnetic-only is wrong


def test_full_dirac_tensor_algebra():
    for i in range(4):
        assert np.linalg.norm(EGAMMA[i]-EGAMMA[i].conj().T) == 0
        for j in range(4):
            assert np.linalg.norm(EGAMMA[i]@EGAMMA[j]+EGAMMA[j]@EGAMMA[i]
                                  -2*(i==j)*np.eye(4)) < 1e-14
    d, dt, dtt = dirac_jet(connection_profile(.43))
    g5 = np.kron(np.eye(8),np.kron(G5,np.eye(3)))
    for matrix in (d,dt,dtt):
        assert np.linalg.norm(matrix+matrix.conj().T) < 1e-14
        assert np.linalg.norm(matrix@g5+g5@matrix) < 1e-14


@pytest.mark.parametrize('theta,seed', [(.1,2),(.43,17),(-1.3,41)])
def test_full_flavor_determinant_not_extra_flavor_factor(theta,seed):
    d = dirac_jet(connection_profile(theta,seed))[0]
    phi, mass, y = [.47,.13,-.29,.07], .06, .83
    full = chiral_matrix(d,phi,mass,y)
    kernel = sea_kernel(d,phi,mass,y)
    assert np.linalg.norm(full.conj().T@full-np.kron(kernel,np.eye(2))) < 1e-14
    logabs, phase = full_logdet(full)
    sea = negative_logdet_jet(kernel)[0]
    assert logabs == pytest.approx(-sea,abs=6e-14)
    assert phase == pytest.approx(1.,abs=2e-13)
    assert abs(logabs+2*sea) > 1  # catches double-counting Nf


def test_local_color_gauge_covariance_of_sea_and_open_quark_line():
    graph = connection_profile(.43)
    rng = np.random.default_rng(41)
    gauges = {n:su3(rng.normal(size=8)) for n in graph.nodes}
    q = block_diag(*(np.kron(np.eye(4),gauges[n]) for n in graph.nodes))
    d = dirac_jet(graph)[0]
    transformed = dirac_jet(graph.transformed(gauges))[0]
    assert np.linalg.norm(transformed-q@d@q.conj().T) < 5e-15
    k, kt = [sea_kernel(x,[.53,.12,0.,0.],.04) for x in (d,transformed)]
    assert np.linalg.norm(kt-q@k@q.conj().T) < 7e-15
    assert negative_logdet_jet(k)[0] == pytest.approx(negative_logdet_jet(kt)[0],abs=5e-14)
    s, st = [np.linalg.inv(x+.6*np.eye(len(x))) for x in (d,transformed)]
    assert np.linalg.norm(st-q@s@q.conj().T) < 2e-14
    assert face_action_jet(graph)[0] == pytest.approx(face_action_jet(graph.transformed(gauges))[0],abs=2e-15)


@pytest.mark.parametrize('generator', range(8))
def test_all_link_color_sources_have_second_order_contact(generator):
    graph = connection_profile(.43)
    d, dt, dtt = dirac_jet(graph,generator=generator)
    t = generators()[generator]
    from deuteron_wigner.quantum.modular_gauge_network import GaugeGraph
    h = 2e-4
    def shifted(epsilon):
        links = dict(graph.links)
        links[0,1] = expm(1j*epsilon*t)@links[0,1]
        return GaugeGraph(links)
    minus, plus = shifted(-h), shifted(h)
    dm, dp = dirac_jet(minus)[0], dirac_jet(plus)[0]
    coarse = (dp-dm)/(2*h)
    fine = (dirac_jet(shifted(h/2))[0]-dirac_jet(shifted(-h/2))[0])/h
    # Verify the expected O(h²) truncation, then remove it independently.
    assert np.linalg.norm(fine-dt)/np.linalg.norm(coarse-dt) == pytest.approx(.25,abs=.001)
    assert np.linalg.norm((4*fine-coarse)/3-dt) < 4e-12
    assert np.linalg.norm((dp-2*d+dm)/h**2-dtt) < 3e-8
    b, bt, btt = face_action_jet(graph,generator=generator)
    bm, bp = face_action_jet(minus)[0], face_action_jet(plus)[0]
    assert bt == pytest.approx((bp-bm)/(2*h),abs=2e-9)
    assert btt == pytest.approx((bp-2*b+bm)/h**2,abs=6e-8)


def test_chiral_orbit_and_non_singlet_mean_field_ward_identity():
    action = StableConnectionSea(mass=0.)
    theta = .43
    radius = minimize_scalar(lambda r:action.homogeneous_value([r,0,0,0],theta),
                             bounds=(.05,2.),method='bounded',options={'xatol':1e-12}).x
    rng = np.random.default_rng(92)
    base = action.homogeneous_value([radius,0,0,0],theta)
    vectors = rng.normal(size=(12,4))
    vectors *= radius/np.linalg.norm(vectors,axis=1)[:,None]
    for phi in vectors:
        assert action.homogeneous_value(phi,theta) == pytest.approx(base,abs=5e-14)
    h = 2e-4
    pion_curvature = (action.homogeneous_value([radius,h,0,0],theta)
                     -2*base+action.homogeneous_value([radius,-h,0,0],theta))/h**2
    assert abs(action.jet(radius,theta)[1][0]) < 2e-6
    assert abs(pion_curvature) < 3e-6
    # Finite-volume orientation average is zero: radius != physical condensate.
    assert np.linalg.norm(np.mean(np.r_[vectors,-vectors],axis=0)) < 1e-16


def test_explicit_mass_chiral_curvature_has_the_same_sea_source():
    action, sigma, theta = StableConnectionSea(), .63, .43
    d = dirac_jet(connection_profile(theta))[0]
    k = sea_kernel(d,[sigma,0,0,0],action.mass,action.yukawa)
    trace = np.trace(np.linalg.inv(k)).real
    expected = (action.jet(sigma,theta)[1][0]
                +2*action.sea_weight*action.yukawa*action.mass*trace)/sigma
    h = 2e-4
    base = action.homogeneous_value([sigma,0,0,0],theta)
    curvature = 2*(action.homogeneous_value([sigma,h,0,0],theta)-base)/h**2
    assert curvature == pytest.approx(expected,abs=3e-6)


@pytest.mark.parametrize('sigma,theta', [(.22,-1.),(.63,.43),(1.5,2.)])
def test_linked_action_gradient_hessian_and_mixed_sea_response(sigma,theta):
    action = StableConnectionSea()
    x = np.array([sigma,theta])
    value, gradient, hessian = action.jet(*x)
    h = 2e-4
    for i in range(2):
        delta = np.eye(2)[i]*h
        vm, gm, _ = action.jet(*(x-delta))
        vp, gp, _ = action.jet(*(x+delta))
        assert gradient[i] == pytest.approx((vp-vm)/(2*h),abs=6e-7)
        np.testing.assert_allclose(hessian[:,i],(gp-gm)/(2*h),atol=6e-6,rtol=0)
    assert np.linalg.norm(hessian-hessian.T) < 1e-13
    quenched = replace(action,sea_weight=0.)
    assert abs(hessian[0,1]-quenched.jet(*x)[2][0,1]) > 1e-5
    assert np.isfinite(value)


def test_sea_changes_coupled_stationary_point_not_just_constant_normalization():
    action = StableConnectionSea()
    fits = []
    for candidate in (action,replace(action,sea_weight=0.)):
        result = minimize(lambda x:candidate.jet(*x)[:2],[.5,.4],jac=True,
                          method='L-BFGS-B',bounds=[(0,3),(-2*np.pi,2*np.pi)],
                          options={'ftol':1e-14,'gtol':1e-9,'maxiter':200})
        assert result.success
        fits.append(result.x)
    assert fits[0][0] > .1
    assert fits[1][0] == pytest.approx(0.,abs=1e-10)
    assert abs(fits[0][1]-fits[1][1]) > 1e-3
    assert np.linalg.norm(action.jet(*fits[0])[1]) < 2e-6
    assert np.linalg.eigvalsh(action.jet(*fits[0])[2]).min() > 1e-4
    for radius in (4.,8.,16.):
        assert action.jet(radius,.4)[0] > .1*radius**4


def test_inhomogeneous_scalar_sea_and_linked_mass_sources():
    d = dirac_jet(connection_profile(.43))[0]
    texture = scalar_texture_matrix(d,np.linspace(.2,.8,8))
    g5 = np.kron(np.eye(8),np.kron(G5,np.eye(3)))
    assert np.linalg.norm(texture.conj().T-g5@texture@g5) < 1e-14
    logabs, phase = full_logdet(texture)
    assert phase**2 == pytest.approx(1.,abs=3e-13)
    source = np.diag(np.repeat([1.,0,0,0,0,0,0,0],12))
    inv = np.linalg.inv(texture)
    first = -2*np.trace(inv@source).real
    second = 2*np.trace(inv@source@inv@source).real
    h = 1e-4
    minus, plus = [full_logdet(texture+s*h*source)[0] for s in (-1,1)]
    assert first == pytest.approx(-(plus-minus)/h,abs=2e-6)
    assert second == pytest.approx(-2*(plus-2*logabs+minus)/h**2,abs=5e-6)


def test_bilocal_open_line_and_sea_use_the_same_full_matrix():
    d = chiral_matrix(dirac_jet(connection_profile(.43))[0],[.53,.12,0.,0.],.04)
    inv = np.linalg.inv(d)
    # A single spin-color-flavor bilocal insertion, with no scalar projection.
    source = np.zeros_like(d); source[7,113] = .7+.2j
    analytic = -inv@source@inv
    h = 1e-4
    finite = (np.linalg.inv(d+h*source)-np.linalg.inv(d-h*source))/(2*h)
    assert np.linalg.norm(finite-analytic) < 2e-9
    a, phase_a = full_logdet(d+h*source)
    b, phase_b = full_logdet(d-h*source)
    finite_sea = -((a-b)+1j*np.angle(phase_a/phase_b))/(2*h)
    assert finite_sea == pytest.approx(-np.trace(inv@source),abs=5e-10)


def test_disconnected_modules_factorize_and_have_no_cross_source():
    d1 = dirac_jet(connection_profile(.43,2))[0]
    d2 = dirac_jet(connection_profile(-.6,41))[0]
    k1, k2 = [sea_kernel(d,[.53,0,0,0],.04) for d in (d1,d2)]
    zero = np.zeros_like(k1)
    joint = block_diag(k1,k2)
    first = [block_diag(np.eye(96),zero),block_diag(zero,np.eye(96))]
    value, gradient, hessian = negative_logdet_jet(joint,first)
    assert value == pytest.approx(negative_logdet_jet(k1)[0]+negative_logdet_jet(k2)[0],abs=1e-13)
    assert hessian[0,1] == 0
    assert np.all(gradient < 0)
    # This algebraic factorization is not a nuclear force or binding proof.


@pytest.mark.parametrize('kwargs', [{'eta':np.inf},{'eta':51},{'scale':0},{'rho_reference':-1}])
def test_invalid_gate_parameters(kwargs):
    with pytest.raises(ValueError):
        BoundedKineticGate(**kwargs)


def test_invalid_action_and_chiral_carrier_are_not_silently_accepted():
    with pytest.raises(ValueError):
        StableConnectionSea(quartic=0)
    with pytest.raises(ValueError):
        StableConnectionSea(sea_weight=-1)
    with pytest.raises(ValueError):
        sea_kernel(np.eye(96),[1.,0,0,0])
    with pytest.raises(ValueError):
        scalar_texture_matrix(np.zeros((96,96)),[-1]*8)


@pytest.mark.parametrize('source', ['mass','color_link','mixed'])
def test_exact_retained_sea_matching_keeps_induced_source_partners(source):
    d,dt,dtt = dirac_jet(connection_profile(.43))
    full = chiral_matrix(d,[.53,.12,-.09,.18],.04)
    ds = np.kron(dt,np.eye(2)) if source != 'mass' else np.eye(192)
    dss = np.kron(dtt,np.eye(2)) if source != 'mass' else np.zeros_like(full)
    if source == 'mixed':
        ds = ds+.3*np.eye(192)  # same source changes both link and quark mass
    jets = (full,ds,dss)
    retained = np.concatenate([np.arange(24*n,24*(n+1)) for n in (0,2,4,6)])
    effective, residual = retained_dirac_jet(jets,retained)
    direct, eff, rest = [dirac_determinant_jet(j) for j in (jets,effective,residual)]
    assert direct[0] == pytest.approx(eff[0]+rest[0],abs=8e-14)
    assert direct[1] == pytest.approx(eff[1]*rest[1],abs=3e-13)
    assert direct[2] == pytest.approx(eff[2]+rest[2],abs=6e-14)
    assert direct[3] == pytest.approx(eff[3]+rest[3],abs=1e-13)
    np.testing.assert_allclose(np.linalg.inv(effective[0]),np.linalg.inv(full)[np.ix_(retained,retained)],atol=1e-15)
    # Attaching full sea determinant alongside retained loops double counts.
    assert abs(eff[2]) > .001
    raw = tuple(j[np.ix_(retained,retained)] for j in jets)
    raw_result = dirac_determinant_jet(raw)
    assert abs(raw_result[2]-eff[2])+abs(raw_result[3]-eff[3]) > .001
    h = 2e-4
    em,_ = retained_dirac_jet((full-h*ds+h*h*dss/2,ds,dss),retained)
    ep,_ = retained_dirac_jet((full+h*ds+h*h*dss/2,ds,dss),retained)
    np.testing.assert_allclose((ep[0]-em[0])/(2*h),effective[1],atol=2e-7,rtol=0)
    np.testing.assert_allclose((ep[0]-2*effective[0]+em[0])/h**2,effective[2],atol=7e-7,rtol=0)


@pytest.mark.parametrize('amplitude', [.1,1.,4.])
def test_arbitrary_pion_texture_hermitian_hamiltonian_and_fock_lower_bound(amplitude):
    rng = np.random.default_rng(151)
    fields = amplitude*rng.normal(size=(8,4))
    h = texture_hamiltonian(connection_profile(.43),fields,mass=.06)
    assert np.linalg.norm(h-h.conj().T) < 1e-14
    eigenvalues = np.linalg.eigvalsh(h)
    radius = np.linalg.norm(fields,axis=1).max()
    one_body_bound = 3*.4+.06+radius
    assert np.max(np.abs(eigenvalues)) < one_body_bound
    rho = np.sum(fields**2,axis=1)/2
    potential = np.sum(.25*rho+.6*rho**2/2)
    # All finite occupations bounded; filled negative modes give the smallest
    # background energy before gauge/charge projection, not a physical vacuum.
    background_minimum = potential+np.minimum(eigenvalues,0.).sum()
    coercive_lower_bound = .6*radius**4/8-4*abs(.25)*radius**2-192*one_body_bound
    assert background_minimum > coercive_lower_bound


def test_non_singlet_chiral_covariance_for_inhomogeneous_texture_hamiltonian():
    rng = np.random.default_rng(151)
    fields = .7*rng.normal(size=(8,4))
    transformed = fields.copy()
    angle = .38
    transformed[:,0] = np.cos(angle)*fields[:,0]-np.sin(angle)*fields[:,3]
    transformed[:,3] = np.sin(angle)*fields[:,0]+np.cos(angle)*fields[:,3]
    graph = connection_profile(.43)
    h, ht = [texture_hamiltonian(graph,p) for p in (fields,transformed)]
    q5 = np.kron(np.eye(8),np.kron(np.kron(G5,np.eye(3)),PAULI[2]))
    axial = expm(.5j*angle*q5)
    assert np.linalg.norm(ht-axial.conj().T@h@axial) < 5e-15

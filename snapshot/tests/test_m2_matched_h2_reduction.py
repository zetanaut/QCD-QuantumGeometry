import numpy as np
import pytest
from scipy import linalg

from deuteron_wigner.quantum.m2_coupled_pilot import (
    build_coupled_pilot, mass_flow_current,
)
from deuteron_wigner.quantum.m2_matched_h2_reduction import (
    FiniteParentReduction, nested_pilot_isometry,
)


@pytest.fixture
def reduction():
    rng = np.random.default_rng(83114)
    x = rng.normal(size=(7, 7)) + 1j*rng.normal(size=(7, 7))
    h = np.diag(np.arange(1, 8)) + .09*(x+x.conj().T)
    # Deliberately non-coordinate P, not an eigenvector-derived retained space.
    z = rng.normal(size=(7, 3)) + 1j*rng.normal(size=(7, 3))
    p = linalg.qr(z, mode="economic")[0]
    return FiniteParentReduction(h, p, scope="complex finite-matrix diagnostic")


def test_exact_lowest_root_reconstruction_and_normalization(reduction):
    r = reduction
    result = r.lowest_state()
    values, vectors = linalg.eigh(r.h)
    assert result.energy_GeV2 == pytest.approx(values[0], abs=5e-13)
    assert result.parent_residual_GeV2 < 2e-12
    assert abs(np.vdot(result.reconstructed, vectors[:, 0]))**2 == pytest.approx(1., abs=2e-13)
    pstate = result.retained
    assert np.vdot(pstate, r.norm_metric(values[0])@pstate) == pytest.approx(1., abs=2e-13)
    assert np.vdot(pstate, pstate).real + result.excluded_probability == pytest.approx(1., abs=2e-13)
    assert result.excluded_probability > .01


def test_effective_self_energy_and_derivative_against_direct_solve(reduction):
    r = reduction
    e = .2
    direct = r.hpp+r.hqp.conj().T@linalg.solve(e*np.eye(len(r.q.T))-r.hqq,r.hqp)
    np.testing.assert_allclose(r.effective_hamiltonian(e), direct, atol=1e-13)
    step = 1e-5
    difference = (r.effective_hamiltonian(e+step)-r.effective_hamiltonian(e-step))/(2*step)
    np.testing.assert_allclose(r.energy_derivative(e), difference, atol=2e-9)
    assert linalg.eigvalsh(r.energy_derivative(e))[-1] <= 1e-13
    omega = r.wave_operator(e)
    np.testing.assert_allclose(r.norm_metric(e), omega.conj().T@omega, atol=3e-13)


def test_two_energy_operator_includes_all_four_blocks(reduction):
    r = reduction
    rng = np.random.default_rng(73)
    op = rng.normal(size=r.h.shape)+1j*rng.normal(size=r.h.shape)
    eo, ei = .1, .7
    wo, wi = r.wave_operator(eo), r.wave_operator(ei)
    xo, xi = r.q.conj().T@wo, r.q.conj().T@wi
    expected = r.p.conj().T@op@r.p + r.p.conj().T@op@r.q@xi
    expected += xo.conj().T@r.q.conj().T@op@r.p
    expected += xo.conj().T@r.q.conj().T@op@r.q@xi
    actual = r.matched_operator(op,eo,ei)
    np.testing.assert_allclose(actual,expected,atol=5e-13)
    np.testing.assert_allclose(actual.conj().T,
        r.matched_operator(op.conj().T,ei,eo),atol=5e-13)
    assert np.linalg.norm(actual-r.p.conj().T@op@r.p) > .1


def test_two_energy_off_shell_current_identity(reduction):
    r = reduction
    rng = np.random.default_rng(343)
    rho = rng.normal(size=r.h.shape)+1j*rng.normal(size=r.h.shape)
    eo, ei = .2, .6
    wo, wi = r.wave_operator(eo), r.wave_operator(ei)
    rop = r.matched_operator(rho,eo,ei)
    do = r.effective_hamiltonian(eo)-eo*np.eye(3)
    di = r.effective_hamiltonian(ei)-ei*np.eye(3)
    lhs = r.matched_operator(r.h@rho-rho@r.h,eo,ei)
    rhs = (eo-ei)*rop + do@r.p.conj().T@rho@wi - wo.conj().T@rho@r.p@di
    np.testing.assert_allclose(lhs,rhs,atol=8e-13)


def test_on_shell_transition_and_charge_orthogonality(reduction):
    r = reduction
    energies, states = linalg.eigh(r.h)
    rng = np.random.default_rng(232)
    rho = rng.normal(size=r.h.shape)+1j*rng.normal(size=r.h.shape)
    # Distinct energies: do not replace two-energy metric by N(E_in).
    for a,b in ((0,0),(0,1),(1,0),(1,2)):
        va,vb = r.p.conj().T@states[:,a],r.p.conj().T@states[:,b]
        matched = r.matched_operator(rho,energies[a],energies[b])
        assert np.vdot(va,matched@vb) == pytest.approx(
            np.vdot(states[:,a],rho@states[:,b]),abs=1e-12)
        charge = r.matched_operator(np.eye(7),energies[a],energies[b])
        assert np.vdot(va,charge@vb) == pytest.approx(float(a==b),abs=1e-12)
        current = r.matched_operator(r.h@rho-rho@r.h,energies[a],energies[b])
        assert np.vdot(va,current@vb) == pytest.approx(
            (energies[a]-energies[b])*np.vdot(va,matched@vb),abs=2e-12)


def test_linearization_is_parent_rayleigh_ritz_with_unit_charge(reduction):
    r = reduction
    e0, e = .2, .21
    model = r.linearized_model(e0)
    u = model.parent_isometry
    np.testing.assert_allclose(u.conj().T@u,np.eye(3),atol=8e-14)
    np.testing.assert_allclose(model.matrix,u.conj().T@r.h@u,atol=5e-13)
    np.testing.assert_allclose(model.operator(np.eye(7)),np.eye(3),atol=8e-14)
    assert linalg.eigvalsh(model.matrix)[0] >= linalg.eigvalsh(r.h)[0]-1e-13
    remainder = r.effective_hamiltonian(e)-r.effective_hamiltonian(e0)-(e-e0)*r.energy_derivative(e0)
    assert np.linalg.norm(remainder,2) <= r.linearization_remainder_bound(e,e0)*(1+1e-9)


def test_retained_basis_rotation_preserves_reconstructed_ground(reduction):
    r = reduction
    rotation = linalg.expm(1j*np.array([[0,1,2],[1,0,.4],[2,.4,1]]))
    other = FiniteParentReduction(r.h,r.p@rotation,scope=r.scope)
    a,b = r.lowest_state(),other.lowest_state()
    assert a.energy_GeV2 == pytest.approx(b.energy_GeV2,abs=1e-12)
    assert abs(np.vdot(a.reconstructed,b.reconstructed))**2 == pytest.approx(1.,abs=1e-12)


def test_decoupled_space_is_not_an_implicit_missing_block():
    h = np.diag([1.,2.,4.])
    r = FiniteParentReduction(h,np.eye(3)[:,:2],scope="explicit zero coupling control")
    state = r.lowest_state()
    assert state.energy_GeV2 == pytest.approx(1.)
    assert state.excluded_probability == 0.
    np.testing.assert_array_equal(r.norm_metric(.5),np.eye(2))


@pytest.mark.parametrize("offset",[0.,1e-13,-1e-13])
def test_poles_are_rejected_not_regularized_away(reduction,offset):
    with pytest.raises(ValueError,match="pole"):
        reduction.effective_hamiltonian(reduction.poles[0]+offset)


def test_excluded_ground_intruder_is_not_misreported_as_retained_ground():
    r = FiniteParentReduction(np.diag([2.,0.]),np.eye(2)[:,:1],scope="intruder diagnostic")
    with pytest.raises(ValueError,match="enlarge retained"):
        r.lowest_state()


@pytest.mark.parametrize("energy",[np.nan,np.inf,1+0j,[1.]])
def test_invalid_energy(reduction,energy):
    with pytest.raises(ValueError):
        reduction.wave_operator(energy)


def test_invalid_parent_map_scope_and_operator(reduction):
    r = reduction
    with pytest.raises(ValueError,match="isometry"):
        FiniteParentReduction(r.h,2*r.p,scope=r.scope)
    with pytest.raises(ValueError,match="Hermitian"):
        FiniteParentReduction(r.h+1j*np.eye(7),r.p,scope=r.scope)
    with pytest.raises(ValueError,match="scope"):
        FiniteParentReduction(r.h,r.p,scope="")
    with pytest.raises(ValueError,match="parent space"):
        r.matched_operator(np.eye(3),.1,.1)
    with pytest.raises(ValueError,match="Taylor disk"):
        r.linearization_remainder_bound(100.,.1)


@pytest.fixture(scope="module")
def nested():
    args = dict(total_k2=9,target="PROTON",jz_twice=1,b_GeV=.4,
                quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    low,high = [build_coupled_pilot(cutoff=n,**args) for n in (0,1)]
    u = nested_pilot_isometry(low,high)
    r = FiniteParentReduction(high.matrix,u,scope="higher HO shells ONLY, same H2 Fock sectors")
    return low,high,u,r


def test_h2_nested_map_preserves_each_hamiltonian_component(nested):
    low,high,u,r = nested
    assert u.shape == (204,37)
    for name in ("h0","cross_pairs_per_gs2","emission_per_gs",
                 "instantaneous_fermion_per_gs2","matrix"):
        np.testing.assert_allclose(u.conj().T@getattr(high,name)@u,
                                   getattr(low,name),atol=4e-12)
    with pytest.raises(ValueError,match="larger oscillator"):
        nested_pilot_isometry(high,low)


def test_h2_matched_state_density_and_current_against_parent(nested):
    low,high,u,r = nested
    solved = r.lowest_state()
    eigenvalues, vectors = linalg.eigh(high.matrix)
    assert solved.energy_GeV2 == pytest.approx(eigenvalues[0],abs=1e-12)
    assert solved.parent_residual_GeV2 < 1e-11
    rho,jx,jy = mass_flow_current(high,(.12,.16))
    np.testing.assert_allclose(u.conj().T@rho@u,mass_flow_current(low,(.12,.16))[0],atol=1e-12)
    a = 0
    # Pick a NONZERO transition, not merely a vanishing diagonal commutator.
    amplitudes = vectors[:,a].conj()@jx@vectors
    b = int(np.argmax(np.abs(amplitudes)))
    assert b != a and abs(amplitudes[b]) > 1e-5
    va,vb = u.conj().T@vectors[:,a],u.conj().T@vectors[:,b]
    for op in (rho,jx,jy):
        matched = r.matched_operator(op,eigenvalues[a],eigenvalues[b])
        assert np.vdot(va,matched@vb) == pytest.approx(
            np.vdot(vectors[:,a],op@vectors[:,b]),abs=2e-11)


def test_staged_elimination_composes_wave_operators_and_norms(reduction):
    r = reduction
    # Independent nested subspaces, not parent eigenvectors.
    outer = FiniteParentReduction(r.h,np.eye(7)[:,:5],scope="outer diagnostic")
    p_inner = np.eye(5)[:,:3]
    direct = FiniteParentReduction(r.h,np.eye(7)[:,:3],scope="direct diagnostic")
    e = .2
    inner = FiniteParentReduction(outer.effective_hamiltonian(e),p_inner,
                                 scope="inner at fixed outer energy")
    combined = outer.wave_operator(e)@inner.wave_operator(e)
    np.testing.assert_allclose(inner.effective_hamiltonian(e),
                               direct.effective_hamiltonian(e),atol=2e-13)
    np.testing.assert_allclose(combined,direct.wave_operator(e),atol=2e-13)
    # The inner metric ALONE omits the first elimination's normalization.
    w = inner.wave_operator(e)
    np.testing.assert_allclose(w.conj().T@outer.norm_metric(e)@w,
                               direct.norm_metric(e),atol=2e-13)
    assert np.linalg.norm(inner.norm_metric(e)-direct.norm_metric(e)) > 1e-4


def test_static_projection_current_has_explicit_leakage_remainder(reduction):
    r = reduction
    model = r.linearized_model(.2)
    u = model.parent_isometry
    rng = np.random.default_rng(21)
    rho = rng.normal(size=r.h.shape)+1j*rng.normal(size=r.h.shape)
    matched_rho = model.operator(rho)
    matched_flow = model.operator(r.h@rho-rho@r.h)
    simple_commutator = model.matrix@matched_rho-matched_rho@model.matrix
    excluded = np.eye(7)-u@u.conj().T
    remainder = u.conj().T@r.h@excluded@rho@u-u.conj().T@rho@excluded@r.h@u
    np.testing.assert_allclose(matched_flow-simple_commutator,remainder,atol=5e-13)
    assert np.linalg.norm(remainder) > .01


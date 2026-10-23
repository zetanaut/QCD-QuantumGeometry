"""Independent determinant/loop and broken-symmetry controls, not QCD fits."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.linalg import expm

from deuteron_wigner.quantum.canonical_lf_fields import GAMMA, slash
from deuteron_wigner.quantum.chiral_soft_matching import G5, PionPole, square
from deuteron_wigner.quantum.nonlocal_source import ETA
from deuteron_wigner.quantum.shared_chiral_dynamics import (
    CovariantPV, RadialInteraction, SharedChiralAction,
)


def reference(m=.004, a=.1, cutoff=.8):
    return SharedChiralAction(CovariantPV(cutoff), RadialInteraction(a), m, 3)


@pytest.mark.parametrize("cutoff,mass", [(.65,.21), (.8,.32), (1.05,.4)])
def test_common_regulator_integral_derivative_and_independent_routing(cutoff, mass):
    r = CovariantPV(cutoff)
    h = 1e-6
    derivative = (r.i1(mass*mass+h)-r.i1(mass*mass-h))/(2*h)
    assert derivative == pytest.approx(-r.bubble(mass, 0), rel=2e-9)
    for q2 in [0., .08, .37]:
        for eta in [0., .23, .5, 1.]:
            tadpole, bubble = r.direct_euclidean_integrals(mass, q2, routing=eta)
            assert tadpole == pytest.approx(r.i1(mass*mass), abs=1e-13)
            assert bubble == pytest.approx(r.bubble(mass, -q2), abs=2e-12)


def test_potential_gradient_and_independent_constant_field_hessians():
    for lam in [0., 1.3]:
        a = replace(reference(), interaction=RadialInteraction(.1, lam, .03))
        mass = a.solve_mass(2.)
        sigma, h = mass-a.current_mass, 2e-5
        value = a.potential(sigma)
        fd_gap = (a.potential(sigma+h)-a.potential(sigma-h))/(2*h)
        sigma_hessian = (a.potential(sigma+h)+a.potential(sigma-h)-2*value)/h**2
        pion_hessian = (a.potential(sigma,h)+a.potential(sigma,-h)-2*value)/h**2
        assert abs(fd_gap-a.gap(mass)) < 4e-10
        assert sigma_hessian == pytest.approx(a.scalar_inverse(mass, 0), abs=3e-7)
        assert pion_hessian == pytest.approx(a.pion_inverse(mass, 0), abs=3e-7)


def test_determinant_hessian_and_axial_loop_dirac_traces():
    # Euclidean gamma matrices, mu=0 denotes Euclidean fourth direction.
    gamma = np.array([GAMMA[0], *[1j*g for g in GAMMA[1:]]])
    rng = np.random.default_rng(100926)
    a = reference()
    mass = a.solve_mass(2.)
    for _ in range(12):
        q, p = rng.normal(0,.15,(2,4))
        qslash = np.einsum("m,mij->ij", q, gamma)
        # Tr_D,F deltaH(q)deltaH(-q), flavor Tr(tau^a tau^a)=2.
        hessian_trace = 2*np.trace((qslash @ G5) @ (-qslash @ G5))
        assert hessian_trace == pytest.approx(8*(q@q), abs=1e-14)
        plus = mass*np.eye(4)-1j*np.einsum("m,mij->ij", p+q/2, gamma)
        minus = mass*np.eye(4)-1j*np.einsum("m,mij->ij", p-q/2, gamma)
        # One tau/2 and one tau: flavor trace is 1, fermion loop sign -Nc.
        numerator = -a.colors*np.array([np.trace(g @ G5 @ plus @ (1j*G5) @ minus) for g in gamma])
        assert np.max(abs(numerator-4*a.colors*mass*q)) < 2e-15
        i1, i2 = a.regulator.direct_euclidean_integrals(mass, q@q, nodes=80)
        determinant_hessian = a.interaction.a-8*a.colors*i1 + a.colors/2*hessian_trace*i2
        assert determinant_hessian == pytest.approx(a.pion_inverse(mass, -q@q), abs=2e-12)
        assert np.max(abs(numerator*i2-q*a.axial_mixing(mass,-q@q))) < 2e-13


@pytest.mark.parametrize("a_value", [.08, .10, .12])
def test_spontaneous_mass_goldstone_normalization_and_quark_gt(a_value):
    a = reference(m=0., a=a_value)
    mass = a.solve_mass(2.)
    pion = a.solve_pion(mass)
    assert mass > .1 and pion.mass == 0 and pion.norm > 0
    assert a.potential(mass) < a.potential(0)
    assert abs(a.gap(mass)) < 2e-14
    assert abs(a.pion_inverse(mass, 0)) < 2e-14
    assert pion.decay_constant*pion.quark_coupling == pytest.approx(mass, abs=1e-14)
    assert pion.decay_constant*pion.pseudoscalar_residue == pytest.approx(-pion.condensate_per_flavor, abs=1e-14)


def test_subcritical_phase_and_threshold_are_not_fake_pions():
    a = reference(m=0., a=.15)
    assert a.solve_mass(2.) == 0
    with pytest.raises(ValueError, match="nonzero"):
        a.solve_pion(0.)
    a = replace(a, current_mass=.004)
    mass = a.solve_mass(2.)
    with pytest.raises(ValueError, match="No real bound"):
        a.solve_pion(mass)
    with pytest.raises(ValueError, match="threshold"):
        a.regulator.direct_euclidean_integrals(mass, -4*mass*mass)
    with pytest.raises(ValueError, match="s <"):
        a.regulator.bubble(mass, 4*mass*mass)


@pytest.mark.parametrize("m", [.0001, .001, .004, .008])
def test_finite_mass_pole_pcac_offshell_identity_and_independent_norm(m):
    a = reference(m=m)
    mass = a.solve_mass(2.)
    p = a.solve_pion(mass)
    assert 0 < p.mass < 2*mass
    assert abs(a.pion_inverse(mass,p.mass_squared)) < 1e-13
    assert p.mass_squared*p.decay_constant == pytest.approx(2*m*p.pseudoscalar_residue, abs=1e-13)
    for s in [-.2, -.03, 0., p.mass_squared]:
        assert (2*m*a.pseudoscalar_mixing(mass,s)-s*a.axial_mixing(mass,s)
                -(mass-m)*a.pion_inverse(mass,s)) == pytest.approx(0., abs=1e-14)
    # Direct 4D continuation, independently differentiated at the pole.
    h = 2e-5
    r = a.regulator
    i1, i2 = r.direct_euclidean_integrals(mass, -p.mass_squared)
    d_i2 = (r.direct_euclidean_integrals(mass, -p.mass_squared-h)[1]
            -r.direct_euclidean_integrals(mass, -p.mass_squared+h)[1])/(2*h)
    norm = 4*a.colors*(i2+p.mass_squared*d_i2)
    assert norm == pytest.approx(p.norm, rel=2e-9)
    assert p.decay_constant == pytest.approx(4*a.colors*mass*i2/np.sqrt(norm), rel=2e-9)
    derivative = (a.pion_inverse(mass,p.mass_squared+h)-a.pion_inverse(mass,p.mass_squared-h))/(2*h)
    assert -derivative == pytest.approx(p.norm, rel=3e-9)
    # Unit pole residue, not a full EM form factor.
    for delta in [1e-5, 1e-6]:
        assert delta/a.normalized_inverse(p,p.mass_squared-delta) == pytest.approx(1., abs=2e-5)


def test_gmor_is_a_chiral_expansion_not_an_exact_finite_mass_fit():
    errors = []
    for m in [1e-3, 1e-4, 1e-5]:
        a = reference(m=m)
        p = a.solve_pion(a.solve_mass(2.))
        ratio = p.mass_squared*p.decay_constant**2/(-2*m*p.condensate_per_flavor)
        errors.append(abs(ratio-1))
    assert errors[2] < errors[1]/8 < errors[0]/64
    assert errors[0] > 1e-5


def test_invariant_radial_deformation_is_not_fixed_by_pion_data():
    a = reference()
    mass = a.solve_mass(2.)
    p = a.solve_pion(mass)
    rho = (mass-a.current_mass)**2/2
    for lam in [.4, 2.]:
        b = replace(a, interaction=RadialInteraction(a.interaction.a,lam,rho))
        q = b.solve_pion(b.solve_mass(2.))
        assert q.quark_mass == pytest.approx(p.quark_mass, abs=1e-12)
        assert q.mass == pytest.approx(p.mass, abs=1e-12)
        assert q.decay_constant == pytest.approx(p.decay_constant, abs=1e-12)
        assert q.norm == pytest.approx(p.norm, abs=1e-12)
        assert q.scalar_curvature-p.scalar_curvature == pytest.approx(2*lam*rho, abs=1e-12)
        for sigma, pion in [(.1,.2), (.4,.01)]:
            expected = lam/2*((sigma*sigma+pion*pion)/2-rho)**2
            assert b.potential(sigma,pion)-a.potential(sigma,pion) == pytest.approx(expected, abs=1e-14)


def test_wrong_kernel_or_regulator_breaks_goldstone_without_changing_gap():
    a = reference(m=0.)
    mass = a.solve_mass(2.)
    wrong_kernel = replace(a, interaction=RadialInteraction(.101))
    wrong_regulator = replace(a, regulator=CovariantPV(.84))
    assert abs(wrong_kernel.pion_inverse(mass,0)) > 9e-4
    assert abs(wrong_regulator.pion_inverse(mass,0)) > .01
    for wrong in [wrong_kernel, wrong_regulator]:
        with pytest.raises(ValueError, match="stationary"):
            wrong.solve_pion(mass)


def test_protected_pole_matches_residue_not_entire_loop_response():
    a = reference()
    p = a.solve_pion(a.solve_mass(2.))
    pole = PionPole(p.mass,p.decay_constant)
    for s in [-.01, -.1, -.25]:
        full = a.axial_mixing(p.quark_mass,s)**2/a.pion_inverse(p.quark_mass,s)
        residue_part = p.decay_constant**2*pole.propagator(s)
        remainder = a.pole_subtracted_axial_exchange(p,s)
        assert full == pytest.approx(residue_part+remainder, abs=1e-14)
        assert abs(remainder) > .001
    with pytest.raises(ValueError, match="coincident"):
        a.pole_subtracted_axial_exchange(p,p.mass_squared)


@pytest.mark.parametrize("m", [0., .004])
def test_generated_resummed_quark_vertices_not_an_inserted_ward_ansatz(m):
    a = reference(m=m)
    mass = a.solve_mass(2.)
    rng = np.random.default_rng(92611)
    for _ in range(12):
        p0 = rng.normal(0,.13,4)
        q = rng.normal(0,.15,4)
        q[0] = .2*np.linalg.norm(q[1:])
        p1 = p0+q
        axial, weighted_p = a.quark_source_vertices(mass,p1,p0)
        expected = (slash(p1)-mass*np.eye(4))@G5+G5@(slash(p0)-mass*np.eye(4))+weighted_p
        defect = np.einsum("m,mij->ij",q @ ETA,axial)-expected
        # A finite stationary-point residual is amplified near the massless
        # pole: defect=2*(M K(0)-m U')/K(s). Do not repair the raw kernel
        # with the gap identity to hide it, or insist on uniform machine eps.
        slope = a.interaction.slope((mass-m)**2/2)
        predicted = 2*(mass*a.pion_inverse(mass,0)-m*slope)/a.pion_inverse(mass,square(q))*G5
        assert np.max(abs(defect-predicted)) < 2e-14
        assert np.max(abs(defect)) < 2e-13
        # Omitting the dynamically generated channel leaves the mass defect.
        bare = np.einsum("m,mij->ij",q @ ETA,np.array([g@G5 for g in GAMMA]))
        bare_expected = (slash(p1)-mass*np.eye(4))@G5+G5@(slash(p0)-mass*np.eye(4))+2*m*G5
        assert np.max(abs(bare-bare_expected)) == pytest.approx(2*(mass-m), abs=1e-14)
        generator = np.zeros((4,4))
        generator[0,2] = generator[2,0] = .37
        boost = expm(generator)
        spin = expm(.37/2*GAMMA[0]@GAMMA[2])
        moved,moved_p = a.quark_source_vertices(mass,boost@p1,boost@p0)
        np.testing.assert_allclose(moved,np.einsum("mn,nij->mij",boost,spin@axial@np.linalg.inv(spin)),rtol=2e-14,atol=2e-14)
        assert np.max(abs(moved_p-spin@weighted_p@np.linalg.inv(spin))) < 2e-14


def test_conditional_pion_calibration_does_not_identify_dressing_across_cutoffs():
    from scripts.evaluate_shared_chiral_dynamics import conditional_calibration
    results = [conditional_calibration(cutoff) for cutoff in [.7,.85,1.]]
    for result in results:
        p = result["forward"]
        assert p["mass"] == pytest.approx(.14,abs=1e-11)
        assert p["decay_constant"] == pytest.approx(.0924,abs=1e-11)
        assert p["norm"] > 0
    masses = [result["forward"]["quark_mass"] for result in results]
    assert max(masses)-min(masses) > .08


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf])
def test_invalid_cutoffs(bad):
    with pytest.raises(ValueError):
        CovariantPV(bad)

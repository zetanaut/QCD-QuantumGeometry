"""Independent analytic controls; no physical positive-quark-spectrum assumption."""
import numpy as np
import pytest
from scipy.optimize import brentq

from deuteron_wigner.quantum.canonical_lf_fields import GAMMA
from deuteron_wigner.quantum.nonlocal_binding_analyticity import (
    CRITICAL_RATIO, GaussianMass, contour_count, lf_momentum, lf_pole, lf_residue,
)


@pytest.mark.parametrize('ratio', [.3, .45, .6])
def test_all_sampled_lambert_branches_solve_full_equation(ratio):
    model = GaussianMass(.8*ratio, .8)
    for branch in range(-5, 5):
        s = model.chiral_pole(branch)
        assert abs(model.denominator(s))/(abs(s)+model.m0**2) < 2e-14
    assert abs(model.chiral_pole(1).conjugate()-model.chiral_pole(-2)) < 2e-15


def test_real_roots_and_critical_double_pole_without_lambert():
    model = GaussianMass(.24, .8)
    # y exp(-2y)=r^2 has its unique maximum 1/(2e) at y=1/2.
    f = lambda y: y*np.exp(-2*y)-(.24/.8)**2
    roots = [.8**2*brentq(f, 0., .5), .8**2*brentq(f, .5, 10.)]
    np.testing.assert_allclose(roots, [model.chiral_pole(0).real, model.chiral_pole(-1).real], rtol=2e-13)
    residues = [1/model.derivative(s) for s in roots]
    assert residues[0] > 0 and residues[1] < 0
    critical = GaussianMass(.8*CRITICAL_RATIO, .8)
    s = .8**2/2
    assert abs(critical.denominator(s)) < 2e-16
    assert abs(critical.derivative(s)) < 5e-16
    assert critical.chiral_pole(0) == critical.chiral_pole(-1)
    with pytest.raises(ValueError):
        lf_residue(critical, s, .8, (.2, -.1))


@pytest.mark.parametrize('offset', [-1e-6, 1e-6])
def test_near_critical_poles_are_not_collapsed_to_the_double_pole(offset):
    model = GaussianMass(.8*CRITICAL_RATIO*(1+offset), .8)
    first, second = model.chiral_pole(0), model.chiral_pole(-1)
    assert abs(first-second) > 1e-4
    if offset < 0:
        assert first.imag == second.imag == 0
    else:
        assert first.imag < 0 < second.imag


@pytest.mark.parametrize('ratio', [.45, .6])
def test_independent_complex_roots_and_closed_contour(ratio):
    model = GaussianMass(.8*ratio, .8)
    for sign, branch in [(1, -1), (-1, 0)]:
        root = model.numerical_pole(.8**2*(.4+sign*.4j))
        assert abs(root-model.chiral_pole(branch)) < 2e-13
    bounds = tuple(.8**2*x for x in (-.2, 1.8, -1.1, 1.1))
    for order in (64, 96):
        count = contour_count(model.denominator, model.derivative, bounds, order)
        assert abs(count-2) < 3e-12


@pytest.mark.parametrize('ratio, expected', [(.3, 0), (.45, 1), (.6, 1)])
def test_upper_frequency_singularities_not_real_feynman_poles(ratio, expected):
    model = GaussianMass(.8*ratio, .8)
    # This rectangle excludes the real-frequency boundary and all negative frequencies.
    f = lambda w: model.denominator(w*w)
    df = lambda w: 2*w*model.derivative(w*w)
    bounds = tuple(.8*x for x in (.1, 1.1, .02, .75))
    for order in (96, 144):
        assert abs(contour_count(f, df, bounds, order)-expected) < 4e-12


@pytest.mark.parametrize('current_mass', [0., .004])
def test_poles_are_in_actual_full_dirac_lf_kernel_and_have_nonzero_residue(current_mass):
    model = GaussianMass(.36, .8, current_mass)
    s = model.numerical_pole(.8**2*(.4+.3j))
    pp, pt = .8, (.2, -.1)
    pm = lf_pole(s, pp, pt)
    assert pm.imag > 0
    matrix = model.inverse_dirac(lf_momentum(pp, pm, pt))
    singular = np.linalg.svd(matrix, compute_uv=False)
    assert max(singular[-2:]) < 4e-14
    assert min(singular[:2]) > .5
    residue = lf_residue(model, s, pp, pt)
    assert np.linalg.norm(residue) > .1
    # Independent numerical inversion near the pole retains every Dirac entry.
    errors = []
    for delta in (1e-5*(1+.4j), 1e-6*(1+.4j)):
        numerical = delta*np.linalg.inv(model.inverse_dirac(lf_momentum(pp, pm+delta, pt)))
        errors.append(np.linalg.norm(numerical-residue)/np.linalg.norm(residue))
    assert errors[1] < .12*errors[0] and errors[1] < 2e-5
    # Det(slash p-M)=(p^2-M^2)^2, away from the singularity as well.
    p = lf_momentum(pp, pm+.03+.01j, pt)
    invariant = p[0]**2-np.dot(p[1:],p[1:])
    np.testing.assert_allclose(np.linalg.det(model.inverse_dirac(p)), model.denominator(invariant)**2, rtol=2e-13)


def test_real_current_mass_does_not_remove_the_conjugate_pair():
    model = GaussianMass(.36, .8, .004)
    plus = model.numerical_pole(.8**2*(.4+.3j))
    minus = model.numerical_pole(.8**2*(.4-.3j))
    assert plus.imag > .1
    assert abs(plus.conjugate()-minus) < 2e-14


def test_contour_refinement_resolves_nearby_exterior_real_pole_without_rounding():
    model = GaussianMass(.24,.8)
    f = lambda w:model.denominator(w*w)
    df = lambda w:2*w*model.derivative(w*w)
    bounds = tuple(.8*x for x in (.1,1.1,.02,.75))
    coarse = contour_count(f,df,bounds,96,subdivisions=1)
    fine = contour_count(f,df,bounds,96,subdivisions=4)
    assert abs(coarse) > 1e-5
    assert abs(fine) < 4e-12


def test_lf_pole_location_respects_longitudinal_boost_but_stays_complex():
    model = GaussianMass(.36, .8)
    s = model.chiral_pole(-1)
    for factor in (.4, 3.):
        pm = lf_pole(s, .8*factor, (.2,-.1))
        assert abs(pm*factor-lf_pole(s, .8, (.2,-.1))) < 2e-16
        p = lf_momentum(.8*factor, pm, (.2,-.1))
        assert abs(p[0]**2-np.dot(p[1:],p[1:])-s) < 3e-15


def test_local_hermitian_reference_has_only_real_energy_eigenvalues():
    p = np.array([.2,-.1,.3]); mass = .36
    h = sum(GAMMA[0]@GAMMA[j+1]*p[j] for j in range(3)) + GAMMA[0]*mass
    np.testing.assert_allclose(h, h.conj().T, atol=1e-16)
    np.testing.assert_allclose(np.linalg.eigvalsh(h), np.array([-1,-1,1,1])*np.sqrt(mass**2+p@p), atol=2e-15)
    w = np.sqrt(GaussianMass(.36,.8).chiral_pole(-1))
    assert w.imag > 0
    assert np.linalg.svd(w*np.eye(4)-h, compute_uv=False)[-1] >= w.imag-1e-15


@pytest.mark.parametrize('params', [(0., .8), (.3, 0.), (-.3,.8), (.3,.8,-.1), (.3,np.inf)])
def test_bad_parameters(params):
    with pytest.raises(ValueError):
        GaussianMass(*params)

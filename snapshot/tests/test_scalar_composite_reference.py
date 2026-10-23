"""Equation-level tests of a reference control, not nucleon qualification."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from deuteron_wigner.quantum.scalar_composite_reference import (
    charge_integral, charge_radius_squared, longitudinal_form_factor,
    overlap_form_factor, triangle_form_factor, wavefunction,
)


@pytest.mark.parametrize("mass", [1.0, 1.5, 1.9])
@pytest.mark.parametrize("q2", [0.0, 0.25, 1.0, 4.0])
def test_three_routes_and_refinement(mass, q2):
    reference = triangle_form_factor(mass, q2, 128)
    for order in (64, 128):
        assert abs(triangle_form_factor(mass, q2, order) - reference) < 1e-10
        assert abs(overlap_form_factor(mass, q2, order) - reference) < 1e-10
        parts = longitudinal_form_factor(mass, q2, order)
        assert abs(parts.total - reference) < 1e-10
        assert mass**2 * (1 - parts.beta)**2 / parts.beta == pytest.approx(q2, abs=1e-13)


@pytest.mark.parametrize("mass", [1.0, 1.5, 1.9])
def test_shared_continuous_charge_normalization_and_wavefunction(mass):
    # Direct quadrature of |psi|^2, including the continuum measure.
    z, w = leggauss(96)
    z, w = (z + 1) / 2, w / 2
    r = z / (1 - z)
    probability = wavefunction(mass, z[:, None], r[None, :]**2)**2
    norm = np.sum(w[:, None] * (w * r / (1-z)**2)[None, :] * probability) / (8*np.pi**2)
    assert norm == pytest.approx(1.0, abs=1e-10)
    assert triangle_form_factor(mass, 0) == pytest.approx(1.0, abs=1e-10)
    assert overlap_form_factor(mass, 0) == pytest.approx(1.0, abs=1e-10)
    parts = longitudinal_form_factor(mass, 0)
    assert parts.valence == pytest.approx(1.0, abs=1e-10)
    assert parts.pair == 0.0


def test_negative_control_is_charge_correct_but_frame_wrong():
    # This is a deliberate missing-current control, not a rescaling recipe.
    parts = longitudinal_form_factor(1.5, 1.0)
    reference = triangle_form_factor(1.5, 1.0)
    assert abs(parts.valence / reference - 1) > 0.10
    assert abs(parts.total - reference) < 1e-10
    assert parts.pair > 0.1
    assert longitudinal_form_factor(1.5, 0).valence == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_infinite_radial_map_is_a_numerical_choice(scale):
    reference = triangle_form_factor(1.9, 4.0, 128)
    assert abs(overlap_form_factor(1.9, 4.0, 128, scale) - reference) < 1e-10
    assert abs(longitudinal_form_factor(1.9, 4.0, 128, scale).total - reference) < 1e-10


@pytest.mark.parametrize("mass", [1.0, 1.5, 1.9])
def test_radius_is_same_form_factor_derivative(mass):
    h = 1e-5
    f0, f1, f2 = [triangle_form_factor(mass, q2, 128) for q2 in (0, h, 2*h)]
    radius_fd = -6 * (-3*f0 + 4*f1 - f2) / (2*h)
    assert radius_fd == pytest.approx(charge_radius_squared(mass), abs=2e-7)


@pytest.mark.parametrize("mass", [1.0, 1.5, 1.9])
def test_spacelike_positive_monotone_and_known_ultraviolet_tail(mass):
    values = [triangle_form_factor(mass, q2) for q2 in (0, .25, 1, 4)]
    assert np.all(np.diff(values) < 0)
    assert values[-1] > 0
    # psi ~ 1/k^2: normalizable, but this does NOT give a finite <kT^2>.
    k2 = np.array([1e6, 2e6])
    tail = k2 * wavefunction(mass, .3, k2)
    assert tail[1] / tail[0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("mass", [0, 2, 2.1, -1, np.nan, np.inf])
def test_threshold_and_nonfinite_masses_rejected(mass):
    with pytest.raises(ValueError):
        charge_integral(mass)


@pytest.mark.parametrize("q2", [-1, np.nan, np.inf])
def test_only_declared_spacelike_domain(q2):
    for evaluator in (triangle_form_factor, overlap_form_factor, longitudinal_form_factor):
        with pytest.raises(ValueError):
            evaluator(1.5, q2)


def test_invalid_resolution_and_wavefunction_inputs():
    for order in (0, 7, 8.5, True):
        with pytest.raises(ValueError):
            triangle_form_factor(1.5, 1, order)
    for scale in (0, -1, np.nan):
        with pytest.raises(ValueError):
            overlap_form_factor(1.5, 1, radial_scale=scale)
    with pytest.raises(ValueError):
        wavefunction(1.5, -.1, 0)
    with pytest.raises(ValueError):
        wavefunction(1.5, .5, -1)

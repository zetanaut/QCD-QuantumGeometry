"""Full tensor and continuum-displacement checks of the Wick current projection."""
from dataclasses import replace

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_six_body_space import SixBodySpace
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import SPECIES as FIVE, frame as frame5
from deuteron_wigner.quantum.m2_six_body_kinematics import SPECIES as SIX, frame as frame6
from deuteron_wigner.quantum.m2_transverse_current_projection import displacement_and_moments
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_wick_current_projection import (
    retained_wick_minus_between, shell_wick_minus_between,
)
from test_m2_wick_one_body import operators
from test_m2_interacting_projector_current import pilot

SECTORS = ("qqqgg", "qqqq-qbar", "qqqggg", "qqqq-qbar-g")


def _tensor_reference(f, s, a, delta, op):
    species, frame = (FIVE[s.sector], frame5) if s.sector in FIVE else (SIX[s.sector], frame6)
    n = len(s.orbitals)
    result = np.zeros((f.dimension, a.shape[1]), complex)
    for li, modes in enumerate(s.longitudinal):
        rho = np.zeros((n, n, s.nspin), complex)
        for particle, kind in enumerate(species):
            weights = op.minus_particle_weight(kind, modes[particle], s.spins[:, particle],
                total_k2=9, total_pplus_GeV=1.3, charge="electromagnetic",
                loop_flavor_charges=[2/3, -1/3])
            shift = frame(s.sector, modes).forward[:-1, particle, None]*delta[None, :]
            d, _ = displacement_and_moments(len(species)-1, s.cutoff, shift, b_GeV=.4)
            rho += d[:, :, None]*weights[None, None, :]
        rows = np.flatnonzero(s.coordinates[:, 0] == li)
        orb, col, spin = s.coordinates[rows, 1:].T
        full = np.zeros((n, s.ncolor, s.nspin, a.shape[1]), complex)
        full[orb, col, spin] = a[rows]
        value = np.einsum("ois,icsk->ocsk", rho, full, optimize=True)
        target = np.flatnonzero(f.coordinates[:, 0] == li)
        orb, col, spin = f.coordinates[target, 1:].T
        result[target] = value[orb, col, spin]
    return result


@pytest.mark.parametrize("sector", SECTORS)
@pytest.mark.parametrize("out_jz", [1, -1])
def test_raw_current_against_full_orbital_color_spin_tensor(sector, out_jz):
    cls = IntermediateSpace if sector in FIVE else SixBodySpace
    s, f = cls(sector, 9, 1, "PROTON", 1), cls(sector, 9, 1, "PROTON", out_jz)
    rng = np.random.default_rng(612)
    a = (rng.normal(size=(s.dimension, 2))+1j*rng.normal(size=(s.dimension, 2)))/np.sqrt(s.dimension)
    op, delta = operators(), np.array([.12, .16])
    args = dict(operators=op, b_GeV=.4, total_pplus_GeV=1.3,
                charge="electromagnetic", loop_flavor_charges=[2/3, -1/3])
    actual = shell_wick_minus_between(f, s, a, delta, **args)
    np.testing.assert_allclose(actual, _tensor_reference(f, s, a, delta, op), atol=8e-17)
    assert np.linalg.norm(actual) > 1e-4
    # A one-body longitudinal scalar commutes even between different Jz fibers.
    lhs = op.shell_mass_diagonal(f)[:, None]*actual
    rhs = shell_wick_minus_between(f, s, op.shell_mass_diagonal(s)[:, None]*a, delta, **args)
    np.testing.assert_allclose(lhs, rhs, atol=2e-16)


@pytest.mark.parametrize("target,jz", [("PROTON", 1), ("PROTON", -1), ("NEUTRON", -1)])
def test_retained_adjoint_flavor_and_longitudinal_boost(target, jz):
    s, f = pilot(1, target, 1), pilot(1, target, jz)
    op, delta = operators(), np.array([.12, .16])
    def current(f, s, delta, charge="electromagnetic", loop=(2/3, -1/3), P=1.3):
        return retained_wick_minus_between(f, s, delta, operators=op, total_pplus_GeV=P,
                                           charge=charge, loop_flavor_charges=loop)
    value = current(f, s, delta)
    np.testing.assert_allclose(value, current(s, f, -delta).conj().T, atol=7e-15)
    np.testing.assert_allclose(value, (2*current(f, s, delta, "up_number", (1, 0))
                                      -current(f, s, delta, "down_number", (0, 1)))/3, atol=6e-15)
    np.testing.assert_allclose(current(f, s, delta, P=2.6), value/4, atol=0.)
    rho = good_current_density_between(f, s, delta, charge="electromagnetic")
    np.testing.assert_allclose(op.retained_mass_coefficient(f)@rho,
                               rho@op.retained_mass_coefficient(s), atol=2e-14)


def test_zero_finite_normal_order_current_is_zero_in_both_projection_paths():
    op = operators(subtract=True, muq=0, mug=0)
    args = dict(operators=op, total_pplus_GeV=1.3, charge="electromagnetic",
                loop_flavor_charges=[2/3, -1/3])
    p = pilot(0)
    np.testing.assert_array_equal(retained_wick_minus_between(p, p, (.12, .16), **args), 0.)
    s = IntermediateSpace("qqqq-qbar", 9, 1, "PROTON", 1)
    a = np.ones((s.dimension, 1), complex)
    np.testing.assert_array_equal(shell_wick_minus_between(s, s, a, (.12, .16), b_GeV=.4, **args), 0.)


def test_invalid_projection_inputs_are_not_repaired():
    p = pilot(0)
    args = dict(operators=operators(), total_pplus_GeV=1.3, charge="electromagnetic",
                loop_flavor_charges=[2/3, -1/3])
    with pytest.raises(ValueError):
        retained_wick_minus_between(p, p, (np.nan, 0), **args)
    with pytest.raises(ValueError):
        retained_wick_minus_between(replace(p, target="NEUTRON"), p, (0, 0), **args)
    s = IntermediateSpace("qqqgg", 9, 0, "PROTON", 1)
    with pytest.raises(ValueError):
        shell_wick_minus_between(s, s, np.zeros(s.dimension), (0, 0), b_GeV=.4, **args)

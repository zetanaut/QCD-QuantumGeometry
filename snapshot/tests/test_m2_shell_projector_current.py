"""Packed Q5/Q6 projector identities independent of any source-image reduction."""
from functools import lru_cache

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_six_body_space import SixBodySpace
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import SPECIES as FIVE, frame as frame5
from deuteron_wigner.quantum.m2_six_body_kinematics import SPECIES as SIX, frame as frame6
from deuteron_wigner.quantum.m2_canonical_good_current import parton_charge
from deuteron_wigner.quantum.m2_galerkin_projector_current import straight_path_projector_vertex
from deuteron_wigner.quantum.m2_shell_projector_current import ShellProjectorVertex, inner_mask

SECTORS = ("qqqgg", "qqqq-qbar", "qqqggg", "qqqq-qbar-g")


@lru_cache(maxsize=8)
def space(sector, cutoff, jz):
    return (IntermediateSpace if sector in FIVE else SixBodySpace)(sector, 9, cutoff, "PROTON", jz)


def random_columns(s, seed, count=1):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(s.dimension, count))+1j*rng.normal(size=(s.dimension, count)))/np.sqrt(s.dimension)


def vertex(f, s, delta=(.12, .16), inner=0, order=16, charge="electromagnetic"):
    return ShellProjectorVertex(f, s, inner_cutoff=inner, transfer=delta,
                                b_GeV=.4, charge=charge, quadrature_order=order)


@pytest.mark.parametrize("sector", SECTORS)
@pytest.mark.parametrize("jz", [1, -1])
@pytest.mark.parametrize("cutoff", [1, 2])
def test_full_shell_divergence_idempotency_and_reversal(sector, jz, cutoff):
    s, f = space(sector, cutoff, 1), space(sector, cutoff, jz)
    k = vertex(f, s)
    reverse = vertex(s, f, (-.12, -.16))
    a, b = random_columns(s, 532), random_columns(f, 751)
    pi, pf = k.mask_in[:, None], k.mask_out[:, None]
    actions = [k.apply(a, component=j) for j in (0, 1)]
    expected = pf*k.density(a)-k.density(pi*a)
    np.testing.assert_allclose(.12*actions[0]+.16*actions[1], expected, atol=2e-17)
    for j, value in enumerate(actions):
        np.testing.assert_allclose(value, pf*value+k.apply(pi*a, component=j), atol=2e-17)
        rev = reverse.apply(b, component=j)
        bound = 64*np.finfo(float).eps*max(np.linalg.norm(b)*np.linalg.norm(value),
                                           np.linalg.norm(a)*np.linalg.norm(rev))
        assert np.vdot(b, value) == pytest.approx(np.vdot(rev, a), abs=bound, rel=0)
    if cutoff == 1 and jz == 1:
        np.testing.assert_array_equal(actions, 0.)
    else:
        assert sum(np.linalg.norm(v) for v in actions) > 1e-3


def tensor_reference(f, s, a, delta, component):
    """Unpack ALL color/spin entries, then gather; no packed block routing."""
    species, frame = (FIVE[s.sector], frame5) if s.sector in FIVE else (SIX[s.sector], frame6)
    n, d = len(s.orbitals), len(species)-1
    result = np.zeros((f.dimension, a.shape[1]), complex)
    for li, modes in enumerate(s.longitudinal):
        kernel = np.zeros((n, n, s.nspin), complex)
        for active, kind in enumerate(species):
            if kind == "g":
                continue
            k = straight_path_projector_vertex(d, 0, s.cutoff,
                frame(s.sector, modes).forward[:-1, active], delta,
                b_GeV=.4, quadrature_order=16)
            q = parton_charge(s.spins[:, active], kind, "electromagnetic")
            kernel += k[component, :, :, None]*q[None, None, :]
        rows = np.flatnonzero(s.coordinates[:, 0] == li)
        orb, col, spin = s.coordinates[rows, 1:].T
        raw = np.zeros((n, s.ncolor, s.nspin, a.shape[1]), complex)
        raw[orb, col, spin] = a[rows]
        value = np.einsum("ois,icsk->ocsk", kernel, raw, optimize=True)
        rows = np.flatnonzero(f.coordinates[:, 0] == li)
        orb, col, spin = f.coordinates[rows, 1:].T
        result[rows] = value[orb, col, spin]
    return result


@pytest.mark.parametrize("sector", SECTORS)
def test_full_tensor_and_identical_particle_statistics(sector):
    s, f = space(sector, 1, 1), space(sector, 1, -1)
    k, a = vertex(f, s), random_columns(s, 859)
    value = k.apply(a, component=0)
    np.testing.assert_allclose(value, tensor_reference(f, s, a, (.12, .16), 0), atol=2e-17)
    np.testing.assert_allclose(f.statistics_project(value),
        k.apply(s.statistics_project(a), component=0), atol=2e-16)


@pytest.mark.parametrize("delta", [(0., 0.), (1e-10, -2e-10), (.81, -.47)])
def test_regular_zero_transfer_and_path_convergence(delta):
    s, f = space("qqqgg", 2, 1), space("qqqgg", 2, -1)
    a = random_columns(s, 464)
    k, refined = vertex(f, s, delta, order=24), vertex(f, s, delta, order=48)
    for j in (0, 1):
        value = k.apply(a, component=j)
        np.testing.assert_allclose(value, refined.apply(a, component=j), atol=6e-17)
        if delta == (0., 0.):
            eps = 2e-5
            plus, minus = np.zeros(2), np.zeros(2)
            plus[j], minus[j] = eps, -eps
            dk = vertex(f, s, plus)
            dm = vertex(f, s, minus)
            derivative = (dk.density(a)-dm.density(a))/(2*eps)
            inner_derivative = (dk.density(k.mask_in[:, None]*a)-dm.density(k.mask_in[:, None]*a))/(2*eps)
            np.testing.assert_allclose(value, k.mask_out[:, None]*derivative-inner_derivative, atol=2e-10)


def test_exact_nested_outer_reporting_and_flavor_linearity():
    s1, f1 = space("qqqgg", 1, 1), space("qqqgg", 1, -1)
    s2, f2 = space("qqqgg", 2, 1), space("qqqgg", 2, -1)
    rows_in = s2.lookup[np.ravel_multi_index(s1.coordinates.T, s2.shape)]
    rows_out = f2.lookup[np.ravel_multi_index(f1.coordinates.T, f2.shape)]
    a = random_columns(s1, 134)
    embedded = np.zeros((s2.dimension, 1), complex)
    embedded[rows_in] = a
    small, large = vertex(f1, s1), vertex(f2, s2)
    np.testing.assert_allclose(small.apply(a, component=0),
        large.apply(embedded, component=0)[rows_out], atol=3e-17)
    up, down = vertex(f2, s2, charge="up_number"), vertex(f2, s2, charge="down_number")
    np.testing.assert_allclose(large.apply(embedded, component=0),
        (2*up.apply(embedded, component=0)-down.apply(embedded, component=0))/3, atol=3e-17)


@pytest.mark.parametrize("cutoff", [-1, True, 1, 1.5])
def test_invalid_inner_cutoff_is_not_coerced(cutoff):
    with pytest.raises(ValueError):
        inner_mask(space("qqqgg", 1, 1), cutoff)


def test_explicit_component_and_endpoint_coordinates_required():
    s = space("qqqgg", 1, 1)
    k = vertex(s, s)
    for j in (True, None, 2, 0.5):
        with pytest.raises(ValueError):
            k.apply(random_columns(s, 1), component=j)
    with pytest.raises(ValueError):
        vertex(s, space("qqqgg", 2, 1))
    with pytest.raises(ValueError):
        vertex(s, s, order=1)

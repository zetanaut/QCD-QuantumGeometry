"""Independent non-Abelian boson-Fock and spinor checks of Wick terms."""
from functools import lru_cache
from itertools import product

import numpy as np
import pytest
from scipy import sparse

from deuteron_wigner.bridge.modes.core import GAMMA, GAMMA_PLUS
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor
from deuteron_wigner.quantum.m2_canonical_gluon_self import (
    CA, TF, gluon_wick_sums, canonical_wick_self, gluon_mass_squared_direction,
)
from deuteron_wigner.quantum.m2_canonical_quark_self import (
    gaussian_coincident_trace, quark_mass_squared_direction,
)
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot


@lru_cache(None)
def _bosons(modes):
    """Complete total-occupation <=3, sufficient for quadratic self witnesses."""
    def compositions(n, d):
        if d == 1:
            yield (n,)
        else:
            for k in range(n+1):
                for rest in compositions(n-k, d-1):
                    yield (k,)+rest
    states = [s for n in range(4) for s in compositions(n, modes)]
    lookup = {s: i for i, s in enumerate(states)}
    dim = len(states)
    ops = []
    for k in range(modes):
        rows, cols, vals = [], [], []
        for col, s in enumerate(states):
            if s[k]:
                out = list(s); out[k] -= 1
                rows.append(lookup[tuple(out)]); cols.append(col); vals.append(np.sqrt(s[k]))
        ops.append(sparse.csr_matrix((vals, (rows, cols)), shape=(dim, dim)))
    return states, lookup, ops


def _f(a, b, c):
    return (a-b)*(b-c)*(c-a)/2  # SU(2) epsilon, C_A=2


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adjoint_current_wick_terms_against_explicit_boson_fock(n):
    states, lookup, ann = _bosons(3*n)
    dim = len(states)
    currents = {w: [sparse.csr_matrix((dim, dim), dtype=complex) for _ in range(3)]
                for w in range(-4*n, 4*n+1, 2)}
    modes = list(range(2, 2*n+1, 2))
    for a, b, c in product(range(3), repeat=3):
        if not _f(a, b, c):
            continue
        for i, p in enumerate(modes):
            for j, r in enumerate(modes):
                ab, ac = ann[3*i+b], ann[3*j+c]
                factor = 1j*_f(a, b, c)*r/(2*np.sqrt(p*r))
                currents[-p-r][a] -= factor*(ab@ac)
                currents[-p+r][a] += factor*(ab@ac.T)
                currents[p-r][a] -= factor*(ab.T@ac)
                currents[p+r][a] += factor*(ab.T@ac.T)
    h = sum(.5*c@currents[-w][a]/w**2 for w, cs in currents.items() if w for a, c in enumerate(cs))
    for i, p in enumerate(modes):
        state = [0]*(3*n); state[3*i] = 1
        row = lookup[tuple(state)]
        derived = gluon_wick_sums(p, boson_modes=n, fermion_modes=n, quark_flavors=0)[1]/CA
        # H convention g²/2, SU(2) C_A=2 cancel; numerical reference is direct.
        assert h[row, row]-h[0, 0] == pytest.approx(derived, abs=3e-14)


def test_local_four_gluon_tadpole_against_explicit_six_oscillator_hamiltonian():
    # Three SU(2) colors, two real transverse polarizations, one p2=2 mode.
    p = 2.
    states, lookup, ann = _bosons(6)
    fields = [(a+a.T)/np.sqrt(2*p) for a in ann]
    h = sparse.csr_matrix((len(states), len(states)))
    for a, b, c, d, e in product(range(3), repeat=5):
        color = _f(a, b, c)*_f(a, d, e)
        if not color:
            continue
        for i, j in product(range(2), repeat=2):
            h += color/4 * fields[2*b+i]@fields[2*d+i]@fields[2*c+j]@fields[2*e+j]
    for mode in range(6):
        state = [0]*6; state[mode] = 1
        row = lookup[tuple(state)]
        # delta P-=C_A*C_trace/(2p), C_trace=1/(2p), C_A=2.
        assert h[row, row]-h[0, 0] == pytest.approx(1/(2*p*p), abs=1e-14)
    sum_a4 = gluon_wick_sums(2, boson_modes=1, fermion_modes=1, quark_flavors=0)[2]
    assert sum_a4/CA == pytest.approx(1/(2*p*p))


@pytest.mark.parametrize("hg", [-1, 1])
@pytest.mark.parametrize("pplus,px,py,mass", [(1., 0., 0., 0.), (.7, .31, -.4, .35)])
def test_antiquark_trace_spin_and_field_normalization(hg, pplus, px, py, mass):
    eps = -np.array([hg, 1j])/np.sqrt(2)
    left = sum(GAMMA[i+1]*eps[i].conjugate() for i in range(2))
    right = sum(GAMMA[i+1]*eps[i] for i in range(2))
    result = 0j
    for h in (-1, 1):
        v = antiquark_spinor(pplus, px, py, mass, h)
        result += v.conj()@GAMMA[0]@left@GAMMA_PLUS@right@v/(2*pplus)
    assert result == pytest.approx(2., abs=2e-15)


@pytest.mark.parametrize("p", [2, 4, 6])
def test_gluon_field_sums_and_flavor_linearity(p):
    r = np.arange(1, 16, 2)
    one = np.array(gluon_wick_sums(p, boson_modes=8, fermion_modes=8, quark_flavors=1))
    two = np.array(gluon_wick_sums(p, boson_modes=8, fermion_modes=8, quark_flavors=2))
    assert one[0] == pytest.approx(TF/p*np.sum(2*r/(p*p-r*r)))
    np.testing.assert_allclose(two, one*np.array([2, 1, 1]), atol=2e-15)
    assert one[1] >= 0 and one[2] > 0


@pytest.fixture(scope="module")
def pilot():
    return build_coupled_pilot(total_k2=9, cutoff=1, target="PROTON", jz_twice=1,
                              b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)


def test_complete_one_body_trace_and_kinetic_subtraction_at_204_states(pilot):
    objects = [canonical_wick_self(pilot, boson_modes=nb, fermion_modes=nb, quark_flavors=2,
                                  coincident_trace_GeV2=gaussian_coincident_trace(width))
               for nb, width in ((8, .6), (12, 1.2))]
    reference = pilot.matrix + .011*quark_mass_squared_direction(pilot) - .023*gluon_mass_squared_direction(pilot)
    for obj in objects:
        s = obj.total_per_gs2
        assert np.linalg.norm(s-s.conj().T) < 1e-12
        assert np.linalg.norm(s) > 1
        subtracted = pilot.matrix + pilot.coupling**2*s + obj.normal_order_counterterm(
            finite_quark_mass_squared_GeV2=.011, finite_gluon_mass_squared_GeV2=-.023)
        np.testing.assert_allclose(subtracted, reference, atol=3e-14)
        nq = pilot.bases[0].shape[1]
        np.testing.assert_allclose(obj.gluon.total_per_gs2[:nq], 0., atol=1e-14)
        for charge in ("baryon", "electromagnetic", "up_number", "down_number"):
            rho = good_current_density(pilot, (.12, .16), charge=charge)
            assert np.linalg.norm(s@rho-rho@s) < 2e-12
    # No silent finite condition is selected by calling the subtraction API.
    with pytest.raises(TypeError):
        objects[0].normal_order_counterterm()


@pytest.mark.parametrize("p,nb,nf,flavors", [(1, 3, 3, 2), (8, 3, 3, 2), (2, 3, 3, True), (2, 3, 3, -1)])
def test_invalid_gluon_trace_inputs(p, nb, nf, flavors):
    with pytest.raises(ValueError):
        gluon_wick_sums(p, boson_modes=nb, fermion_modes=nf, quark_flavors=flavors)

"""Independent signed-field/CAR checks and common one-body H/current pairing."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from deuteron_wigner.quantum.m2_canonical_quark_self import (
    CF, quark_wick_sums, antiquark_wick_sums, gaussian_coincident_trace,
)
from deuteron_wigner.quantum.m2_canonical_gluon_self import (
    canonical_wick_self, gluon_wick_sums, quark_mass_squared_direction, gluon_mass_squared_direction,
)
from deuteron_wigner.quantum.m2_wick_one_body import WickOneBody
from test_m2_ordered_current_wick import _field_norm
from test_m2_canonical_quark_self import _annihilators
from test_m2_interacting_projector_current import pilot


def operators(*, subtract=False, width=.8, muq=.011, mug=-.007):
    return WickOneBody(boson_modes=8, fermion_modes=8, quark_flavors=2,
        coincident_trace_GeV2=gaussian_coincident_trace(width),
        normal_order_subtraction=subtract,
        finite_quark_mass2_per_gs2_GeV2=muq, finite_gluon_mass2_per_gs2_GeV2=mug)


@pytest.mark.parametrize("nb,nf", [(1, 1), (2, 2), (3, 3)])
def test_all_IF_energy_descendants_from_unreordered_signed_field(nb, nf):
    occ, bos = [0]*(4*nf), [0]*(2*nb)
    vacuum = _field_norm(occ, bos, nb, nf, r_power=1)
    for species, fn in ((0, quark_wick_sums), (1, antiquark_wick_sums)):
        for fi in range(nf):
            for spin in range(2):
                state = occ.copy()
                state[(species*nf+fi)*2+spin] = 1
                value = _field_norm(state, bos, nb, nf, r_power=1)-vacuum
                expected = fn(2*fi+1, boson_modes=nb, fermion_modes=nf)[0]
                assert value == pytest.approx(expected, abs=2e-14)
    for bi in range(nb):
        state = bos.copy()
        state[2*bi] = 1
        value = _field_norm(occ, state, nb, nf, r_power=1)-vacuum
        expected = gluon_wick_sums(2*bi+2, boson_modes=nb, fermion_modes=nf, quark_flavors=1)[0]/.5
        assert value == pytest.approx(expected, abs=2e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_antiquark_IG_from_full_unreordered_current_square(n):
    ops = _annihilators(2*n)
    b, d = ops[:n], ops[n:]
    size = 2**(2*n)
    currents = {w: sparse.csr_matrix((size, size)) for w in range(-4*n, 4*n+1, 2)}
    for i, p in enumerate(range(1, 2*n, 2)):
        for j, r in enumerate(range(1, 2*n, 2)):
            currents[p-r] += b[i].T@b[j]-d[j].T@d[i]
            currents[p+r] += b[i].T@d[j].T
            currents[-p-r] += d[j]@b[i]
    h = sum(.5*c@currents[-w]/w**2 for w, c in currents.items() if w)
    for i, p in enumerate(range(1, 2*n, 2)):
        anti = h[1 << (n+i), 1 << (n+i)]-h[0, 0]
        quark = h[1 << i, 1 << i]-h[0, 0]
        assert anti == pytest.approx(quark, abs=2e-15)
        assert anti == pytest.approx(antiquark_wick_sums(p, boson_modes=n, fermion_modes=n)[1]/2, abs=2e-15)


@pytest.mark.parametrize("subtract", [False, True])
def test_retained_common_operator_matches_preceding_Hamiltonian_sources(subtract):
    p = pilot(0)
    o = operators(subtract=subtract)
    previous = canonical_wick_self(p, boson_modes=o.boson_modes, fermion_modes=o.fermion_modes,
        quark_flavors=o.quark_flavors, coincident_trace_GeV2=o.coincident_trace_GeV2)
    expected = (0 if subtract else previous.total_per_gs2)
    expected = expected + .011*quark_mass_squared_direction(p)-.007*gluon_mass_squared_direction(p)
    np.testing.assert_allclose(o.retained_mass_coefficient(p), expected, atol=2e-15)
    assert np.linalg.norm(expected) > .01


@pytest.mark.parametrize("species,mode", [("q", 1), ("qbar", 3), ("g", 2)])
def test_explicit_normal_order_cancels_composite_and_preserves_finite_terms(species, mode):
    for width in (.4, 1.6):
        o = operators(subtract=True, width=width)
        wick, counter, finite = o.mass_components(species, mode, total_k2=9)
        assert wick != 0 and wick+counter == 0
        mu = -.007 if species == "g" else .011
        assert finite == pytest.approx(mu*9/mode)
        current = o.minus_particle_weight(species, mode, np.array([0, 2]),
            total_k2=9, total_pplus_GeV=1.3, charge="electromagnetic",
            loop_flavor_charges=[2/3, -1/3])
        physical = np.array([2/3, -1/3])*(1 if species == "q" else -1)
        expected = 0 if species == "g" else physical*.011/(2*(mode*1.3/9)**2)
        np.testing.assert_allclose(current, expected, atol=3e-16)


@pytest.mark.parametrize("change", [
    {"normal_order_subtraction": 1}, {"quark_flavors": True},
    {"boson_modes": 0}, {"fermion_modes": 0}, {"coincident_trace_GeV2": -.1},
    {"finite_quark_mass2_per_gs2_GeV2": np.nan}, {"finite_gluon_mass2_per_gs2_GeV2": np.inf},
])
def test_invalid_controls(change):
    with pytest.raises(ValueError):
        replace(operators(), **change)


def test_current_flavor_trace_cannot_disagree_with_Hamiltonian():
    with pytest.raises(ValueError):
        operators().minus_particle_weight("q", 1, [0], total_k2=9,
            total_pplus_GeV=1.3, charge="electromagnetic", loop_flavor_charges=[1.])

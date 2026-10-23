"""Connected ordered Wick coefficients checked before any Wick reordering."""
from collections import defaultdict

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_constrained_fermion_current import ALPHA, GOOD_AXES, ANTI_AXES
from deuteron_wigner.quantum.m2_current_wick_self import (
    quark_minus_current_wick, antiquark_minus_current_wick, gluon_minus_current_wick,
)
from deuteron_wigner.quantum.m2_canonical_quark_self import CF

ARGS = dict(total_k2=9, total_pplus_GeV=1.3, coincident_trace_GeV2=.17)


def _field_norm(occupied, bosons, nb, nf, *, r_power=0):
    """Direct sum ||chi_r |Fock>||², chi=(alpha A psi)_r/(sqrt(2)r).

    Exact occupation transitions, including CAR signs, stimulated emission
    and Pauli blocking. No contraction formula or denominator routing is
    imported from the tested implementation. Abelian color, stripped 2pi,
    transverse trace and cell spacing; two physical real polarizations.
    """
    pieces = defaultdict(lambda: np.zeros(4, complex))
    for species in (0, 1):
        axes = GOOD_AXES if species == 0 else ANTI_AXES
        for fi in range(nf):
            p = 2*fi+1
            for spin in range(2):
                index = (species*nf+fi)*2+spin
                create = species == 1
                if occupied[index] == create:
                    continue
                sign = (-1)**sum(occupied[:index])
                fout = list(occupied)
                fout[index] += 1 if create else -1
                signed_p = -p if create else p
                for bi in range(nb):
                    k = 2*(bi+1)
                    for pol in range(2):
                        mode = 2*bi+pol
                        for boson_create in (False, True):
                            if not boson_create and not bosons[mode]:
                                continue
                            bout = list(bosons)
                            amplitude = np.sqrt(bout[mode]+int(boson_create))
                            bout[mode] += 1 if boson_create else -1
                            signed_k = -k if boson_create else k
                            r = signed_p+signed_k
                            key = (r, tuple(fout), tuple(bout))
                            pieces[key] += sign*amplitude/np.sqrt(2*k)/r * (ALPHA[pol]@axes[:, spin])
    return float(sum(key[0]**r_power*np.vdot(v, v).real for key, v in pieces.items()))


@pytest.mark.parametrize("nb,nf", [(1, 1), (1, 2), (3, 2), (4, 3)])
def test_all_one_body_descendants_against_unreordered_fock_field_norm(nb, nf):
    occ = [0]*(4*nf)
    bos = [0]*(2*nb)
    vacuum = _field_norm(occ, bos, nb, nf)
    assert vacuum > 0
    scale = ARGS["coincident_trace_GeV2"]*9**2/(2*np.pi*1.3**2)
    args = dict(ARGS, boson_modes=nb, fermion_modes=nf)
    for fi in range(nf):
        p = 2*fi+1
        for spin in range(2):
            for species, kernel in ((0, quark_minus_current_wick), (1, antiquark_minus_current_wick)):
                state = occ.copy()
                state[(species*nf+fi)*2+spin] = 1
                connected = _field_norm(state, bos, nb, nf)-vacuum
                physical_charge = 1 if species == 0 else -1
                expected = physical_charge*kernel(p, **args)/(CF*scale)
                assert connected == pytest.approx(expected, abs=2e-14)
    for bi in range(nb):
        for pol in range(2):
            state = bos.copy()
            state[2*bi+pol] = 1
            connected = _field_norm(occ, state, nb, nf)-vacuum
            expected = gluon_minus_current_wick(2*(bi+1), **args,
                loop_flavor_charges=[1.])/(.5*scale)
            assert connected == pytest.approx(expected, abs=2e-14)


@pytest.mark.parametrize("p", [1, 3, 7])
def test_antiquark_is_not_silently_equated_to_quark(p):
    args = dict(ARGS, boson_modes=9, fermion_modes=4)
    q = quark_minus_current_wick(p, **args)
    anti = antiquark_minus_current_wick(p, **args)
    assert 0 < anti < q
    # Gauge derivative of the corresponding ORDERED anti IF self energy.
    k = 2*np.arange(1, 10)
    delta = 1.3/9
    def energy(pplus):
        return CF*.17/(2*np.pi)*sum(1/(k*(pplus+k*delta)))
    pp = p*delta
    h = 1e-4*delta
    minus_derivative = -(energy(pp-2*h)-8*energy(pp-h)+8*energy(pp+h)-energy(pp+2*h))/(12*h)
    assert anti == pytest.approx(minus_derivative, rel=2e-10)


@pytest.mark.parametrize("k", [2, 4, 6])
def test_gluon_current_is_internal_charge_derivative_not_gluon_charge(k):
    args = dict(ARGS, boson_modes=4, fermion_modes=5)
    r = 2*np.arange(5)+1
    delta = 1.3/9
    def energy(internal_shift):
        return .5*.17/(2*np.pi*k)*sum(
            1/(k*delta-(r*delta+internal_shift))-1/(k*delta+(r*delta+internal_shift)))
    h = 1e-4*delta
    derivative = (energy(-2*h)-8*energy(-h)+8*energy(h)-energy(2*h))/(12*h)
    result = gluon_minus_current_wick(k, **args, loop_flavor_charges=[1.])
    assert result == pytest.approx(derivative, rel=2e-10)
    assert gluon_minus_current_wick(k, **args, loop_flavor_charges=[2/3, -1/3]) == pytest.approx(result/3)
    assert gluon_minus_current_wick(k, **args, loop_flavor_charges=[1., -1.]) == 0
    assert gluon_minus_current_wick(k, **args, loop_flavor_charges=[]) == 0


@pytest.mark.parametrize("function,mode,extra", [
    (antiquark_minus_current_wick, 2, {}),
    (antiquark_minus_current_wick, 9, {}),
    (gluon_minus_current_wick, 3, {"loop_flavor_charges": [1.]}),
    (gluon_minus_current_wick, 8, {"loop_flavor_charges": [1.]}),
    (gluon_minus_current_wick, 2, {"loop_flavor_charges": [[1.]]}),
    (gluon_minus_current_wick, 2, {"loop_flavor_charges": [float("nan")]}),
])
def test_invalid_or_unowned_field_inputs(function, mode, extra):
    with pytest.raises(ValueError):
        function(mode, **dict(ARGS, boson_modes=3, fermion_modes=5), **extra)

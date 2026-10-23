"""Current/Hamiltonian common-regulator contraction and derivative checks."""
import numpy as np
import pytest

from deuteron_wigner.bridge.modes.core import polarization
from deuteron_wigner.quantum.m2_canonical_quark_self import CF,quark_wick_sums
from deuteron_wigner.quantum.m2_constrained_fermion_current import (
    two_gluon_current_spin,discrete_gluon_field_factor,
)
from deuteron_wigner.quantum.m2_current_wick_self import quark_minus_current_wick


@pytest.mark.parametrize("p2",[1,3,7])
@pytest.mark.parametrize("nb",[2,6,17])
@pytest.mark.parametrize("pplus",[.7,2.3])
def test_wick_sum_against_ordered_spin_polarization_contraction_and_H_derivative(p2,nb,pplus):
    total=9;trace=.17
    value=quark_minus_current_wick(p2,total_k2=total,total_pplus_GeV=pplus,
        boson_modes=nb,fermion_modes=5,coincident_trace_GeV2=trace)
    direct=np.zeros((4,2,2),complex)
    for k2 in range(2,2*nb+1,2):
        signed=(p2-k2)*pplus/total
        for hg in (-1,1):
            eps=polarization(k2*pplus/total,0.,0.,hg)[2:]
            direct+=CF*trace*discrete_gluon_field_factor(k2)**2*two_gluon_current_spin(
                left_polarization=eps,right_polarization=eps.conj(),
                left_plus=signed,right_plus=signed)
    np.testing.assert_allclose(direct[1],value*np.eye(2),atol=5e-15,rtol=2e-15)
    np.testing.assert_array_equal(direct[[0,2,3]],0.)
    k=2*np.arange(1,nb+1,dtype=float)
    # Hold physical field modes and the cell fixed while varying the
    # external momentum locally. This is not a boost or regulator change.
    def energy(pq):
        return CF*trace*np.sum(1/(2*np.pi*k*(pq-k*pplus/total)))
    pq=p2*pplus/total
    h=1e-4*pplus/total
    derivative=(energy(pq-2*h)-8*energy(pq-h)+8*energy(pq+h)-energy(pq+2*h))/(12*h)
    assert value==pytest.approx(-derivative,rel=8e-11,abs=1e-12)
    old_if,_=quark_wick_sums(p2,boson_modes=nb,fermion_modes=5)
    assert energy(pq)==pytest.approx(CF*total/np.pi*trace*old_if/(2*pplus),abs=2e-15)


def test_negative_constraint_modes_contribute_and_field_trace_is_not_external_shell():
    args=dict(total_k2=9,total_pplus_GeV=1.3,fermion_modes=5,coincident_trace_GeV2=.17)
    first=quark_minus_current_wick(1,boson_modes=1,**args)
    more=quark_minus_current_wick(1,boson_modes=9,**args)
    assert more>first>0  # every internal mode is above this external quark
    assert quark_minus_current_wick(1,boson_modes=9,**dict(args,coincident_trace_GeV2=.34))==pytest.approx(2*more)
    assert quark_minus_current_wick(1,boson_modes=9,**dict(args,total_pplus_GeV=2.6))==pytest.approx(more/4)


@pytest.mark.parametrize("override",[
    {"boson_modes":0},{"fermion_modes":1},{"coincident_trace_GeV2":-.1},
    {"total_pplus_GeV":0.},{"total_k2":3},
])
def test_invalid_contraction_controls_are_not_defaulted(override):
    args=dict(total_k2=9,total_pplus_GeV=1.3,boson_modes=4,fermion_modes=5,
              coincident_trace_GeV2=.17)
    with pytest.raises(ValueError):
        quark_minus_current_wick(3,**dict(args,**override))

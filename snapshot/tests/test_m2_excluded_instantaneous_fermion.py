from itertools import product
from math import pi,sqrt
import numpy as np
import pytest
from scipy import sparse

from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_canonical_instantaneous_fermion import transverse_spin_factor
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_excluded_current_scattering import current_pair_color
from deuteron_wigner.quantum.m2_excluded_instantaneous_fermion import (
    ordered_five_qg_color,instantaneous_fermion_QQ_per_gs2,
)


def test_full_six_color_ordered_contacts_and_commutator_charge():
    for q,g in product(range(3),(3,4)):
        direct,cross=(ordered_five_qg_color(q,g,c) for c in (False,True))
        np.testing.assert_allclose(direct,direct.conj().T,atol=3e-15)
        np.testing.assert_allclose(cross,cross.conj().T,atol=3e-15)
        np.testing.assert_allclose(direct-cross,-current_pair_color("qqqgg",q,g),atol=3e-15)


def test_constant_contact_all_four_physical_spinors_and_signed_cell_denominator():
    for ho,hi,go,gi in product((-1,1),repeat=4):
        u=lf_spinor(.7,.21,-.12,.6,hi)/sqrt(1.4)
        v=lf_spinor(1.3,-.18,.33,.6,ho)/sqrt(2.6)
        eo=(-np.array([go,1j])/sqrt(2)).conj()
        ei=-np.array([gi,1j])/sqrt(2)
        for crossed in (False,True):
            left,right=(ei,eo) if crossed else (eo,ei)
            a=sum(GAMMA[i+1]*left[i] for i in range(2))
            b=sum(GAMMA[i+1]*right[i] for i in range(2))
            spin=v.conj()@GAMMA[0]@a@GAMMA_PLUS@b@u
            expected=transverse_spin_factor(hi,gi,crossed=crossed) if ho==hi and go==gi else 0
            assert spin==pytest.approx(expected,abs=3e-15)
    for half_length,r in product((.4,3.,20.),(-3,-1,3,5)):
        total,kin,kout=9,2,4
        pp=pi*total/(2*half_length);pg1=pi*kin/(2*half_length);pg2=pi*kout/(2*half_length)
        constrained=pi*r/(2*half_length)
        direct=2*pp*.5/(2*half_length)/(2*sqrt(pg1*pg2))/constrained
        assert direct==pytest.approx(total/(2*pi*sqrt(kin*kout)*r))
        if r<0:assert direct<0  # magnitude clipping would fail this physical normalization


@pytest.mark.parametrize("cutoff,target,jz",[(0,"PROTON",1),(1,"NEUTRON",-1)])
def test_actual_QQ_IF_matrix_Hermiticity_quark_and_gluon_statistics(cutoff,target,jz):
    space=IntermediateSpace("qqqgg",9,cutoff,target,jz)
    w=instantaneous_fermion_QQ_per_gs2(space,b_GeV=.4)
    assert sparse.linalg.norm(w)>1e-3
    assert sparse.linalg.norm(w-w.getH())<3e-14
    assert sparse.linalg.norm(instantaneous_fermion_QQ_per_gs2(space,b_GeV=.8)-4*w)<3e-14
    same=np.flatnonzero(space.coordinates[:,0]==0)
    assert sparse.linalg.norm(w[same][:,same])>1e-4
    rng=np.random.default_rng(522)
    a=rng.normal(size=(space.dimension,2))+1j*rng.normal(size=(space.dimension,2))
    for p in ((1,0,2,3,4),(0,1,2,4,3)):
        np.testing.assert_allclose(w@space.permute(a,p),space.permute(w@a,p),atol=3e-14)


def test_invalid_slots_and_sector_rejected():
    for args in ((True,3,False),(0,2,True),(0,3,0)):
        with pytest.raises(ValueError):ordered_five_qg_color(*args)
    with pytest.raises(ValueError):
        instantaneous_fermion_QQ_per_gs2(IntermediateSpace("qqqq-qbar",9,0,"PROTON",1),b_GeV=.4)


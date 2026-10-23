from itertools import product
from math import pi,sqrt
import numpy as np
import pytest
from scipy import sparse
from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_canonical_instantaneous_fermion import (
    instantaneous_fermion_mass_per_gs2,ordered_qg_color,transverse_spin_factor,
)
from deuteron_wigner.quantum.m2_c117_pair_current import color_pair_matrix,color_permutation_matrix
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import (
    S3,longitudinal_orbital_action,total_sign_project,
)
from deuteron_wigner.quantum.m2_spin_flavor_s3_completion import spin_flavor_permutation_representation


def maxabs(a):
    return float(np.max(abs(a.data))) if a.nnz else 0.


def test_all_ordered_color_matrices_and_commutator_charge_identity():
    for q in range(3):
        d,c=ordered_qg_color(q,False),ordered_qg_color(q,True)
        np.testing.assert_allclose(d,d.conj().T,atol=3e-15)
        np.testing.assert_allclose(c,c.conj().T,atol=3e-15)
        # [T^a,T^b]=i f_abc T^c = - (T^c F^c)_ab.
        np.testing.assert_allclose(d-c,-color_pair_matrix("qqqg",q,3),atol=3e-15)


@pytest.mark.parametrize("hq,hg,crossed",list(product((-1,1),(-1,1),(False,True))))
def test_spin_contact_from_arbitrary_physical_spinors(hq,hg,crossed):
    pq=(.7,.2,-.1); po=(1.3,-.4,.3)
    u=lf_spinor(*pq,.6,hq)/sqrt(2*pq[0])
    v=lf_spinor(*po,.6,hq)/sqrt(2*po[0])
    eps=-np.array([hg,1j])/sqrt(2)
    left,right=(eps,eps.conj()) if crossed else (eps.conj(),eps)
    a=sum(GAMMA[i+1]*left[i] for i in range(2))
    b=sum(GAMMA[i+1]*right[i] for i in range(2))
    expected=v.conj()@GAMMA[0]@a@GAMMA_PLUS@b@u
    assert transverse_spin_factor(hq,hg,crossed=crossed)==pytest.approx(expected,abs=3e-15)
    # Source SB -gamma+ gamma.A has the BPP plus ordering used above.
    sb=-v.conj()@GAMMA[0]@GAMMA_PLUS@a@b@u
    assert sb==pytest.approx(expected,abs=3e-15)


def test_finite_cell_mass_factor_both_fermion_denominator_signs():
    for length in (.4,3.,20.):
        total=9; kg=2; ko=4
        for r in (3,-1):
            pn=pi*total/(2*length)
            pgin=pi*kg/(2*length);pgout=pi*ko/(2*length);pf=pi*r/(2*length)
            # 2P+, source +1/2, integrated four cell modes, gluon factors, inverse.
            direct=2*pn*.5/(2*length)/(2*sqrt(pgin*pgout))/pf
            assert direct==pytest.approx(total/(2*pi*sqrt(kg*ko)*r))


@pytest.mark.parametrize("hg",[-1,1])
def test_retained_contact_hermitian_s3_covariant_and_nonzero(hg):
    h=instantaneous_fermion_mass_per_gs2(9,.4,0,hg)
    assert h.nnz>0 and maxabs(h-h.getH())<5e-15
    for p in S3:
        lc=sparse.kron(longitudinal_orbital_action("qqqg",9,0,p),
                        color_permutation_matrix("qqqg",p),format="csr")
        u=sparse.kron(lc,spin_flavor_permutation_representation(p),format="csr")
        assert maxabs(u@h-h@u)<5e-15
    rng=np.random.default_rng(55)
    v=total_sign_project(rng.normal(size=(h.shape[0]//128,2,64)),"qqqg",9,0)
    hv=(h@v.ravel()).reshape(v.shape)
    np.testing.assert_allclose(total_sign_project(hv,"qqqg",9,0),hv,atol=2e-14)

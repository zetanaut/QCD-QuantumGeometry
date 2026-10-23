from itertools import product
from math import sqrt

import numpy as np
import pytest
from scipy import sparse

from deuteron_wigner.microscopic.h0.color import ColorSingletBasis, representation_generators
from deuteron_wigner.pilot.color import structure_constants
from deuteron_wigner.quantum.m2_c117_pair_current import color_permutation_matrix
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import (
    S3,longitudinal_basis,longitudinal_orbital_action,oscillator_shell,total_sign_project,
)
from deuteron_wigner.quantum.m2_coupled_pair_action import pair_contact_block
from deuteron_wigner.quantum.m2_canonical_mixed_current import (
    canonical_gluon_current_factor,qg_cross_mass_per_gs2,
)
from test_m2_coupled_pair_action import direct_coordinate_contact,maxabs


def test_current_sign_from_minkowski_contraction_and_ordered_field_derivative():
    f=structure_constants()
    for ko,ki in ((2,2),(2,6),(8,4)):
        for a in range(8):
            # Euclidean +f A_b dA_c: creation-annihilation -i ki f_abc,
            # reordered annihilation-creation +i ko f_acb.
            direct=(-1j*ki*f[a]+1j*ko*f[a].T)/(2*sqrt(ki*ko))
            expected=canonical_gluon_current_factor(ko,ki)*representation_generators("8")[a]
            np.testing.assert_allclose(direct,expected,atol=2e-15)


def test_quark_plus_gluon_charge_annihilates_both_complete_c7_singlets():
    v=ColorSingletBasis.construct("qqqg").tensors
    tensor=v.reshape(3,3,3,8,2)
    worst_wrong=0.
    for a in range(8):
        tq=representation_generators("3")[a]
        fg=representation_generators("8")[a]
        quarks=sum(np.moveaxis(np.tensordot(tq,tensor,axes=(1,i)),0,i) for i in range(3))
        gluon=np.moveaxis(np.tensordot(fg,tensor,axes=(1,3)),0,3)
        np.testing.assert_allclose(quarks+gluon,0,atol=3e-15)
        worst_wrong=max(worst_wrong,np.linalg.norm(quarks-gluon))
    assert worst_wrong>1.


@pytest.mark.parametrize("pair,out",[
    ((0,3),(5,7,9,6)),((1,3),(3,11,9,4)),((2,3),(3,7,15,2)),
])
def test_qg_excited_contact_against_physical_position_quadrature(pair,out):
    incoming=(3,7,9,8)
    expected=direct_coordinate_contact(incoming,out,pair,.4,2)
    np.testing.assert_allclose(pair_contact_block(incoming,out,pair,.4,2),expected,atol=3e-15)


@pytest.mark.parametrize("cutoff",[0,1,2])
def test_mixed_matrix_hermitian_and_s3_covariant(cutoff):
    h=qg_cross_mass_per_gs2(9,.4,cutoff)
    assert maxabs(h-h.getH())<1e-14
    assert h.nnz>0
    for p in S3:
        u=sparse.kron(longitudinal_orbital_action("qqqg",9,cutoff,p),
                       color_permutation_matrix("qqqg",p),format="csr")
        assert maxabs(u@h-h@u)<1e-14
    np.testing.assert_allclose(qg_cross_mass_per_gs2(9,.8,cutoff).toarray(),
                               4*h.toarray(),atol=1e-14)


def test_mixed_total_sign_and_no_gluon_helicity_or_flavor_mixing():
    h=qg_cross_mass_per_gs2(9,.4,1)
    shape=(h.shape[0]//2,2,64)
    rng=np.random.default_rng(192)
    v=total_sign_project(rng.normal(size=shape)+1j*rng.normal(size=shape),"qqqg",9,1)
    hv=(h@v.reshape(h.shape[0],64)).reshape(shape)
    np.testing.assert_allclose(total_sign_project(hv,"qqqg",9,1),hv,atol=2e-14)


def test_gluon_factor_is_not_one_constant_legacy_rescaling():
    assert canonical_gluon_current_factor(2,2)==1
    assert canonical_gluon_current_factor(2,6)==pytest.approx(2/sqrt(3))
    for ko,ki in product((2,4,6),repeat=2):
        assert canonical_gluon_current_factor(ko,ki)==canonical_gluon_current_factor(ki,ko)

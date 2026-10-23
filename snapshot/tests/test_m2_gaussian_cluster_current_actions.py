"""Actual coherent raw and symmetry-projected retained physical-leg currents."""
from dataclasses import replace
from functools import lru_cache
import numpy as np
import pytest
from scipy import sparse
from deuteron_wigner.quantum.m2_gaussian_cluster_current_actions import (
    cluster_emission_current,cluster_cubic_current_between)
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import build_cluster_pilot
from deuteron_wigner.quantum.m2_gaussian_cluster_wick import GaussianClusterWick
from deuteron_wigner.quantum.m2_transverse_contact_current import emission_transverse_contact
from deuteron_wigner.quantum.m2_one_gluon_minus_contact import emission_minus_contact
from deuteron_wigner.quantum.m2_c117_pair_current import color_permutation_matrix
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import longitudinal_orbital_action
from deuteron_wigner.quantum.m2_spin_flavor_s3_completion import spin_flavor_permutation_representation
from test_m2_gaussian_cluster_current import regulator


@lru_cache(maxsize=8)
def raw(hg,ci,co,width=.8,charge="electromagnetic"):
    return cluster_emission_current(9,ci,co,(.12,.16),b_GeV=.4,mass_GeV=.35,
        total_pplus_GeV=1.3,gluon_helicity=hg,charge=charge,regulator=regulator(width),
        mean_total_transverse_GeV=(.03,.07),prescription=PRESCRIPTION,quadrature_order=8)


@pytest.mark.parametrize("hg",[-1,1])
@pytest.mark.parametrize("ci,co",[(0,0),(0,1),(1,0)])
def test_full_raw_bare_limit_matches_both_existing_canonical_contact_assemblies(hg,ci,co):
    a = raw(hg,ci,co,1e7)
    jt = emission_transverse_contact(9,ci,co,(.12,.16),b_GeV=.4,total_pplus_GeV=1.3,
        gluon_helicity=hg,charge="electromagnetic")
    jm = emission_minus_contact(9,ci,co,(.12,.16),b_GeV=.4,mass_GeV=.35,total_pplus_GeV=1.3,
        gluon_helicity=hg,charge="electromagnetic",mean_total_transverse=(.03,.07))
    assert a.total[0].nnz==0
    for actual,expected in zip(a.total[1:],(jm,*jt)):
        difference = actual-expected
        assert np.linalg.norm(difference.data)<3e-14
    assert all(np.linalg.norm(x.data)<3e-14 for x in a.regulator)


@pytest.mark.parametrize("hg",[-1,1])
def test_regulated_all_component_raw_current_preserves_quark_permutation_covariance(hg):
    a = raw(hg,1,1)
    rng = np.random.default_rng(759)
    v = rng.normal(size=(a.total[0].shape[1],2))+1j*rng.normal(size=(a.total[0].shape[1],2))
    for permutation in ((1,0,2),(1,2,0)):
        def perm(sector):
            return sparse.kron(longitudinal_orbital_action(sector,9,1,permutation),
                sparse.kron(color_permutation_matrix(sector,permutation),
                    spin_flavor_permutation_representation(permutation)),format="csr")
        left,right = perm("qqqg"),perm("qqq")
        for current in (*a.canonical,*a.regulator):
            np.testing.assert_allclose(left@(current@v),current@(right@v),atol=5e-15)
    assert sum(np.linalg.norm(x.data) for x in a.regulator)>1e-3


def test_resolved_flavor_currents_remain_linear_before_any_target_projection():
    up,down,em,baryon = [raw(1,0,0,charge=q) for q in ("up_number","down_number","electromagnetic","baryon")]
    for u,d,e,b in zip(up.total,down.total,em.total,baryon.total):
        assert np.linalg.norm((e-(2*u-d)/3).data)<3e-15
        assert np.linalg.norm((b-(u+d)/3).data)<3e-15


@lru_cache(maxsize=4)
def pilot(target,jz):
    r = regulator()
    w = GaussianClusterWick(regulator=r,quark_flavors=2,normal_order_subtraction=True,
        finite_quark_mass2_per_gs2_GeV2=.011,finite_gluon_mass2_per_gs2_GeV2=-.007)
    return build_cluster_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,regulator=r,total_pplus_GeV=1.3,wick_operators=w)


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
def test_actual_cluster_pilots_both_spin_endpoints_and_reversed_absorption(target):
    incoming,outgoing = pilot(target,1),pilot(target,-1)
    args = dict(charge="electromagnetic",mean_total_transverse_GeV=(.03,.07),
                prescription=PRESCRIPTION,quadrature_order=8)
    a = cluster_cubic_current_between(outgoing,incoming,(.12,.16),**args)
    b = cluster_cubic_current_between(incoming,outgoing,(-.12,-.16),**args)
    assert a.total.shape==(4,len(outgoing.matrix),len(incoming.matrix))
    for forward,backward in ((a.canonical,b.canonical),(a.regulator,b.regulator)):
        np.testing.assert_allclose(forward,backward.conj().transpose(0,2,1),atol=2e-15)
    assert np.linalg.norm(a.total)>1e-4 and np.linalg.norm(a.regulator)>1e-5
    assert not a.physical and not a.complete_gauged_contractions


def test_different_field_current_definitions_cannot_be_combined():
    p = pilot("PROTON",1)
    other = replace(p,field_regulator=regulator(.4))
    with pytest.raises(ValueError,match="identical cluster"):
        cluster_cubic_current_between(other,p,(.12,.16),charge="electromagnetic",
            mean_total_transverse_GeV=(.03,.07),prescription=PRESCRIPTION,quadrature_order=8)

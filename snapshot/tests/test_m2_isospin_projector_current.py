"""Finite isospin projector: independent flavor algebra and exact Ward endpoints."""
from itertools import combinations
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_isospin_projector_current import (
    retained_isospin_projector,exchange_path_orbital,isospin_outside_tangent)
from deuteron_wigner.quantum.m2_source_vectors import source_tensor
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import jacobi_frame,longitudinal_basis
from deuteron_wigner.quantum.m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS
from deuteron_wigner.quantum.m2_canonical_good_current import parton_charge
from deuteron_wigner.quantum.m2_galerkin_projector_current import _rectangular_displacement
from deuteron_wigner.quantum.m2_gaussian_cluster_Q_projector import _apply_excluded_raw
from test_m2_gaussian_cluster_h2 import model

D=np.array([.12,.16]);B=.4


@pytest.mark.parametrize("cutoff",[0,1,2])
def test_exchange_path_has_independent_displacement_endpoints_reversal_and_refinement(cutoff):
    wa=(.2,.3,-.6);wb=(-.3,.1,.5)
    x=exchange_path_orbital(3,cutoff,wa,wb,tuple(D),B,8)
    a,b=[_rectangular_displacement(3,cutoff,cutoff,np.array(w)[:,None]*D,B) for w in (wa,wb)]
    np.testing.assert_allclose(np.einsum("j,jab->ab",D,x),a-b,atol=3e-15)
    reverse=exchange_path_orbital(3,cutoff,wa,wb,tuple(-D),B,8)
    np.testing.assert_allclose(x,-reverse.conj().transpose(0,2,1),atol=3e-15)
    np.testing.assert_allclose(x,exchange_path_orbital(3,cutoff,wa,wb,tuple(D),B,12),atol=3e-15)


def test_three_quark_I_projector_equals_exact_flavor_swaps_not_spin_permutations():
    sf=np.array(SPIN_FLAVOR_BASIS);direct=np.eye(64)
    for a,b in combinations(range(3),2):
        swapped=sf.copy()
        swapped[:,a]=2*(sf[:,b]//2)+sf[:,a]%2
        swapped[:,b]=2*(sf[:,a]//2)+sf[:,b]%2
        rows=np.ravel_multi_index(swapped.T,(4,4,4))
        direct[rows,np.arange(64)]-=1/3
    iso=retained_isospin_projector()
    np.testing.assert_allclose(iso,direct,atol=5e-16)
    np.testing.assert_allclose(iso@iso,iso,atol=5e-16)


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
def test_full_I_complement_tangent_closes_charge_commutator_and_keeps_raw_H(target):
    m=model(target=target);p=m.pilot
    v=source_tensor(p,"qqqg",np.eye(len(p.matrix)))
    iso=retained_isospin_projector();sf=np.array(SPIN_FLAVOR_BASIS)
    def ip(x):return np.einsum("st,loathk->loashk",iso,x,optimize=True)
    rho=np.zeros_like(v)
    for li,modes in enumerate(longitudinal_basis("qqqg",p.total_k2)):
        fr=jacobi_frame(modes)
        for q in range(3):
            orb=_rectangular_displacement(3,p.cutoff,p.cutoff,fr.forward[:-1,q,None]*D,B)
            weight=parton_charge(sf[:,q],"q","electromagnetic")
            rho[li]+=np.einsum("oi,iasnk,s->oasnk",orb,v[li],weight,optimize=True)
    k=isospin_outside_tangent(p,v,D,charge="electromagnetic",quadrature_order=8)
    np.testing.assert_allclose(np.einsum("j,jloashk->loashk",D,k),ip(rho)-rho,atol=4e-15)
    assert np.linalg.norm(k)>1e-3
    for a in k:np.testing.assert_allclose(ip(a),0.,atol=5e-16)
    baryon=isospin_outside_tangent(p,v,D,charge="baryon",quadrature_order=8)
    np.testing.assert_array_equal(baryon,0.)
    for n,s in m.spaces5.items():
        np.testing.assert_allclose(_apply_excluded_raw(p,s,v),m.qq.sources_by_power[1][n],atol=3e-15)


def test_I_source_requires_the_actual_incoming_internal_projector():
    p=model().pilot;v=source_tensor(p,"qqqg",np.eye(len(p.matrix)))
    v=v.copy();v.flat[0]+=1.
    with pytest.raises(ValueError,match="I=1/2"):
        isospin_outside_tangent(p,v,D,charge="electromagnetic",quadrature_order=8)

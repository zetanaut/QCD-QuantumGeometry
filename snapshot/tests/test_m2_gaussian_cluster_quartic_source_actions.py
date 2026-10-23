"""Exact retained→Q source projection, independent slot enumeration and bare actions."""
from dataclasses import replace
from functools import lru_cache
from itertools import product
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_source_current import (
    PairSourceRoute,build_cluster_quartic_source_current,quartic_source_coefficients)
from deuteron_wigner.quantum.m2_quadratic_minus_actions import build_pair_minus_endpoint,build_six_minus_endpoint
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_six_body_space import SixBodySpace
from deuteron_wigner.quantum.m2_six_body_quartic import COMPONENTS,direct_routes
from deuteron_wigner.quantum.m2_canonical_good_current import parton_charge
from deuteron_wigner.quantum.m2_source_vectors import source_tensor
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import longitudinal_basis
from test_m2_gaussian_cluster_excluded_current import pilot
from test_m2_gaussian_cluster_quartic_source_current import ward_orbital,D,MEAN
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION

FAMILIES = [(5,c) for c in ("valence_IF","valence_IG","valence_sea")]+[(6,c) for c in COMPONENTS]


@lru_cache(maxsize=16)
def space(n,component,target="PROTON",jz=1,cutoff=0):
    sea = component in ("valence_sea","gluon_IF_pair","gluon_IG_pair")
    if n==5:return IntermediateSpace("qqqq-qbar" if sea else "qqqgg",9,cutoff,target,jz)
    return SixBodySpace("qqqq-qbar-g" if sea else "qqqggg",9,cutoff,target,jz)


def action(p,s,c,*,charge="electromagnetic",mean=MEAN,order=8):
    return build_cluster_quartic_source_current(p,s,D,component=c,charge=charge,
        mean_total_transverse_GeV=mean,prescription=PRESCRIPTION,quadrature_order=order)


@pytest.mark.parametrize("n,c",[(5,"valence_IF"),(6,"valence_IF"),(6,"gluon_IF_pair")])
@pytest.mark.parametrize("target,jz", [("PROTON",1),("NEUTRON",-1)])
def test_actual_bare_all_IF_source_actions_and_creation_statistics(n,c,target,jz):
    p = pilot(target,1e7);s = space(n,c,target,jz,1)
    a = action(p,s,c,order=2)
    old = (build_pair_minus_endpoint(p,s,D,total_pplus_GeV=1.3,charge="electromagnetic") if n==5
           else build_six_minus_endpoint(p,s,D,component=c,total_pplus_GeV=1.3,charge="electromagnetic"))
    np.testing.assert_array_equal(a.retained_columns,old.retained_columns)
    np.testing.assert_allclose(a.total[1],old.current_per_gs2,atol=3e-15)
    np.testing.assert_allclose(a.regulator,0.,atol=3e-15)
    np.testing.assert_array_equal(a.total[0],0.)
    assert np.linalg.norm(a.canonical)>1e-5


def slot_ward(p,s,c):
    """Individually enumerated full spin/color slots; no production gather."""
    n = len(s.spin_dims)
    parent_sector = "qqq" if n==5 else "qqqg"
    columns = np.arange(p.bases[0].shape[1]) if n==5 else np.arange(p.bases[0].shape[1],len(p.matrix))
    source = source_tensor(p,parent_sector,np.eye(len(p.matrix))[:,columns])
    incoming = longitudinal_basis(parent_sector,p.total_k2)
    if n==6:
        routes = direct_routes(9,c)
    else:
        routes = []
        for out in s.longitudinal:
            for q in range(3):
                inc = list(out[:3]);inc[q] += out[3]+out[4]
                routes.append(PairSourceRoute(c,tuple(inc),out,q))
    result = np.zeros((s.dimension,len(columns)),complex)
    @lru_cache(maxsize=None)
    def endpoint(route,e,f):
        return ward_orbital(route,ci=0,co=0,e=e,f=f)[0,0]
    sf = list(product(range(4),repeat=3))
    for route in routes:
        li,lo = incoming.index(route.incoming),s.longitudinal.index(route.outgoing)
        for hs in product((-1,1),repeat=4):
            h,_ = quartic_source_coefficients(route,hs,total_pplus_GeV=1.3,emitter_charge=1.,pair_charge=1.)
            if not np.any(h):continue
            for spin,state in enumerate(sf):
                if c.startswith("valence"):
                    ho,hi,h1,h2 = hs;q = route.emitter
                    if 2*(state[q]%2)-1!=hi:continue
                    new = list(state);new[q] = 2*(state[q]//2)+(ho+1)//2
                    e = float(parton_charge(state[q],"q","electromagnetic"))
                    for old in (range(1) if n==5 else range(2)):
                        value = source[li,0,:,spin] if n==5 else source[li,0,:,spin,old]
                        for flavor in (range(2) if c=="valence_sea" else range(1)):
                            if c=="valence_sea":
                                final = new+[2*flavor+(h1+1)//2,2*flavor+(h2+1)//2]+([] if n==5 else [old])
                                f = float(parton_charge(2*flavor,"q","electromagnetic"))
                            else:
                                final = new+([] if n==5 else [old])+[(h1+1)//2,(h2+1)//2];f=0.
                            color = endpoint(route,e,f)*(h@value)
                            spin_index = np.ravel_multi_index(final,s.spin_dims)
                            ids = np.flatnonzero((s.coordinates[:,0]==lo)&(s.coordinates[:,3]==spin_index))
                            result[ids] += color[s.coordinates[ids,2]]
                elif c.endswith("_pair"):
                    hq,ha,hg,hin = hs
                    value = source[li,0,:,spin,(hin+1)//2]
                    for flavor in range(2):
                        f = float(parton_charge(2*flavor,"q","electromagnetic"))
                        final = list(state)+[2*flavor+(hq+1)//2,2*flavor+(ha+1)//2,(hg+1)//2]
                        color = endpoint(route,0.,f)*(h@value)
                        spin_index = np.ravel_multi_index(final,s.spin_dims)
                        ids = np.flatnonzero((s.coordinates[:,0]==lo)&(s.coordinates[:,3]==spin_index))
                        result[ids] += color[s.coordinates[ids,2]]
    return s.statistics_project(result)


@pytest.mark.parametrize("n,c",FAMILIES)
def test_independent_projected_full_Hamiltonian_Ward_endpoints_for_every_quartic_source(n,c):
    p = pilot("NEUTRON");s = space(n,c,"NEUTRON")
    a = action(p,s,c)
    expected = slot_ward(p,s,c)
    np.testing.assert_allclose(np.einsum("j,joi->oi",D,a.total[2:]),expected,atol=3e-16,rtol=8e-11)
    for part in ("canonical","regulator"):
        for matrix in getattr(a,part):
            np.testing.assert_allclose(s.statistics_project(matrix),matrix,atol=8e-16)
    if c in ("gluon_A4","gluon_IG_ggg"):
        np.testing.assert_array_equal(a.total,0.)
    elif c not in ("valence_IF","gluon_IF_pair"):
        np.testing.assert_array_equal(a.canonical,0.)
    assert not a.physical and not a.complete_H2_matched_current


@pytest.mark.parametrize("n,c",[(5,"valence_sea"),(6,"valence_IG"),(6,"valence_sea"),(6,"gluon_IG_pair")])
def test_regulator_only_quartics_keep_created_flavor_both_boost_laws_and_opposite_Jz(n,c):
    p = pilot();s = space(n,c,jz=-1,cutoff=1)
    a,u,d = [action(p,s,c,charge=q) for q in ("electromagnetic","up_number","down_number")]
    beta = np.array([.06,-.08])
    boost = action(p,s,c,mean=tuple(np.array(MEAN)+1.3*beta))
    scaled = action(replace(p,total_pplus_GeV=2.6),s,c)
    np.testing.assert_allclose(a.total,(2*u.total-d.total)/3,atol=4e-15)
    np.testing.assert_allclose(boost.total[2:],a.total[2:],atol=4e-15)
    np.testing.assert_allclose(boost.total[1],a.total[1]+np.einsum("j,joi->oi",beta,a.total[2:]),atol=4e-15)
    np.testing.assert_allclose(scaled.total[1],a.total[1]/4,atol=4e-15)
    np.testing.assert_allclose(scaled.total[2:],a.total[2:]/2,atol=4e-15)
    np.testing.assert_array_equal(a.canonical,0.)
    if c=="valence_IG":
        # At K2=9 all three physical gluons have mode2. The actual IG
        # source has (k_new1-k_new2)=0, independently of regulator gauging.
        assert all(r.outgoing[4]==r.outgoing[5] for r in direct_routes(9,c))
        np.testing.assert_array_equal(a.regulator,0.)
    else:
        assert np.linalg.norm(a.regulator)>1e-6


def test_source_definition_guards():
    p = pilot();s = space(5,"valence_IF")
    with pytest.raises(ValueError,match="actual"):
        action(object(),s,"valence_IF")
    with pytest.raises(ValueError,match="target"):
        action(p,space(5,"valence_IF",target="NEUTRON"),"valence_IF")
    with pytest.raises(ValueError,match="matching"):
        action(p,s,"valence_sea")
    with pytest.raises(ValueError,match="component"):
        action(p,s,"gluon_A4")

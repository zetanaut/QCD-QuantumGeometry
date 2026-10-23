"""Actual common-regulated cubic P/Q actions, independent bare and slot references."""
from dataclasses import replace
from functools import lru_cache
from itertools import product
from math import sqrt
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_excluded_current import (
    build_cluster_excluded_cubic_current,ClusterSixCubicCurrent)
from deuteron_wigner.quantum.m2_gaussian_cluster_current import (
    PRESCRIPTION,cluster_cubic_current_orbitals)
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import build_cluster_pilot
from deuteron_wigner.quantum.m2_gaussian_cluster_wick import GaussianClusterWick
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_excluded_minus_contact import build_excluded_minus_contact,SixBodyMinusContact
from deuteron_wigner.quantum.m2_excluded_transverse_contact import build_excluded_transverse_contact
from deuteron_wigner.quantum.m2_six_body_transverse_contact import SixBodyTransverseContact
from deuteron_wigner.quantum.m2_six_body_cubic import five_statistics_project
from deuteron_wigner.quantum.m2_six_body_vertices import split_routes,color_source
from test_m2_six_body_transverse_contact import spaces,CHANNELS
from test_m2_gaussian_cluster_current import regulator,endpoint_ward_reference


DELTA = (.12,.16)
MEAN = (.03,.07)
ARGS = dict(charge="electromagnetic",mean_total_transverse_GeV=MEAN,
            prescription=PRESCRIPTION,quadrature_order=8)


@lru_cache(maxsize=4)
def pilot(target="PROTON",width=.8):
    r = regulator(width)
    w = GaussianClusterWick(regulator=r,quark_flavors=2,normal_order_subtraction=True,
        finite_quark_mass2_per_gs2_GeV2=.011,finite_gluon_mass2_per_gs2_GeV2=-.007)
    return build_cluster_pilot(total_k2=9,cutoff=0,target=target,jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,regulator=r,
        total_pplus_GeV=1.3,wick_operators=w)


def make(inc,out,*,width=.8,charge="electromagnetic",mean=MEAN,pp=1.3,order=8):
    return ClusterSixCubicCurrent(inc,out,DELTA,regulator=regulator(width),
        b_GeV=.4,mass_GeV=.35,total_pplus_GeV=pp,charge=charge,
        mean_total_transverse_GeV=mean,prescription=PRESCRIPTION,quadrature_order=order)


def raw(spaces,seed=795):
    rng = np.random.default_rng(seed)
    return {s:(rng.normal(size=(sp.dimension,1))+1j*rng.normal(size=(sp.dimension,1)))/sqrt(sp.dimension)
            for s,sp in spaces.items()}


@pytest.mark.parametrize("sector",["qqqgg","qqqq-qbar"])
@pytest.mark.parametrize("target,jz,cutoff",[("PROTON",1,0),("NEUTRON",-1,1)])
def test_actual_P_to_Q_bare_limit_matches_all_established_contact_components(sector,target,jz,cutoff):
    p = pilot(target,1e7);s = IntermediateSpace(sector,9,cutoff,target,jz)
    new = build_cluster_excluded_cubic_current(p,s,DELTA,**{**ARGS,"quadrature_order":2})
    jt = build_excluded_transverse_contact(p,s,DELTA,total_pplus_GeV=1.3,charge="electromagnetic")
    jm = build_excluded_minus_contact(p,s,DELTA,total_pplus_GeV=1.3,charge="electromagnetic",
                                     mean_total_transverse=MEAN)
    np.testing.assert_array_equal(new.retained_columns,jt.retained_columns)
    np.testing.assert_array_equal(new.total[0],0.)
    np.testing.assert_allclose(new.total[1],jm.current_per_gs,atol=3e-14)
    np.testing.assert_allclose(new.total[2:],jt.current_per_gs,atol=3e-14)
    np.testing.assert_allclose(new.regulator,0.,atol=3e-14)
    assert np.linalg.norm(new.total)>1e-4


@pytest.mark.parametrize("source,channel",CHANNELS)
def test_every_Q5_Q6_channel_bare_limit_including_antiquark_charge_once(source,channel):
    inc,out = spaces(0,"PROTON",1);x = raw(inc)
    a = make(inc,out,width=1e7,order=2)
    oldt = SixBodyTransverseContact(inc,out,DELTA,b_GeV=.4,total_pplus_GeV=1.3,charge="electromagnetic")
    oldm = SixBodyMinusContact(inc,out,DELTA,b_GeV=.4,total_pplus_GeV=1.3,
        mass_GeV=.35,charge="electromagnetic",mean_total_transverse=MEAN)
    for action in (a,oldt,oldm):
        action.routes = tuple(t for t in action.routes if t[0]==source and t[2].channel==channel)
    for mu in (1,2,3):
        actual = a.apply(x,component=mu,part="total")
        expected = oldm.apply(x) if mu==1 else oldt.apply(x,component=mu-2)
        for s in out:
            np.testing.assert_allclose(actual[s],expected[s],atol=2e-15)
    assert sum(np.linalg.norm(v) for v in a.apply(x,component=1,part="total").values())>1e-7


@pytest.mark.parametrize("sector",["qqqgg","qqqq-qbar"])
@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_regulated_P_to_Q_color_statistics_charge_and_kinematic_boosts(sector,target,jz):
    p = pilot(target);s = IntermediateSpace(sector,9,1,target,jz)
    def value(charge="electromagnetic",mean=MEAN,pp=1.3):
        return build_cluster_excluded_cubic_current(replace(p,total_pplus_GeV=pp),s,DELTA,
            **{**ARGS,"charge":charge,"mean_total_transverse_GeV":mean})
    base = value();u,d = value("up_number"),value("down_number")
    beta = np.array([.04,-.07])
    shifted = value(mean=tuple(np.array(MEAN)+1.3*beta));longitudinal = value(pp=2.6)
    for part in ("canonical","regulator","total"):
        a = getattr(base,part)
        np.testing.assert_allclose(a,(2*getattr(u,part)-getattr(d,part))/3,atol=3e-15)
        for mu in range(4):
            np.testing.assert_allclose(s.statistics_project(a[mu]),a[mu],atol=4e-15)
        np.testing.assert_allclose(getattr(shifted,part)[2:],a[2:],atol=4e-15)
        np.testing.assert_allclose(getattr(shifted,part)[1],a[1]+np.einsum("j,joi->oi",beta,a[2:]),atol=6e-15)
        np.testing.assert_allclose(getattr(longitudinal,part)[1],a[1]/4,atol=4e-15)
        np.testing.assert_allclose(getattr(longitudinal,part)[2:],a[2:]/2,atol=4e-15)
    assert np.linalg.norm(base.regulator)>1e-5
    assert not base.physical and not base.longitudinal_envelope_gauged


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_regulated_Q5_Q6_adjoint_full_raw_statistics_and_separate_parts(target,jz):
    inc,out = spaces(1,target,jz);x,y = raw(inc),raw(out,796)
    a = make(inc,out)
    for mu in range(4):
        pieces = {}
        for part in ("canonical","regulator","total"):
            actual,adjoint = a.apply(x,component=mu,part=part),a.adjoint(y,component=mu,part=part)
            np.testing.assert_allclose(sum(np.vdot(y[s],actual[s]) for s in out),
                sum(np.vdot(adjoint[s],x[s]) for s in inc),atol=3e-16)
            for s,sp in out.items():
                np.testing.assert_allclose(sp.statistics_project(actual[s]),actual[s],atol=3e-16)
            for s,sp in inc.items():
                np.testing.assert_allclose(five_statistics_project(sp,adjoint[s]),adjoint[s],atol=3e-16)
            pieces[part] = actual
        for s in out:
            np.testing.assert_allclose(pieces["total"][s],
                pieces["canonical"][s]+pieces["regulator"][s],atol=5e-17)


def slot_reference(action,x,source,channel,*,ward=False):
    """Ground shell: direct occupation/color enumeration, no packed production gather."""
    incoming,outgoing = action.incoming,action.outgoing
    v = {s:five_statistics_project(sp,x[s]) for s,sp in incoming.items()}
    expected = {s:np.zeros((sp.dimension,1),complex) for s,sp in outgoing.items()}
    for route in split_routes(9,source,channel):
        target,origin = outgoing[route.sector],incoming[source]
        li,lo = origin.longitudinal.index(route.incoming),target.longitudinal.index(route.outgoing)
        fock = 2*sqrt(2) if channel=="g_to_pair" else sqrt(3) if source=="qqqgg" else 1.
        color = color_source(route)
        for hs in product((-1,1),repeat=3):
            if ward:
                # Independent full shifted H endpoints use e=.7;
                # rescale to the actual underlying baryon flavor charge.
                scalar = endpoint_ward_reference(route,hs,DELTA,MEAN,action.regulator,0,0)[0,0]/(.7*3)
            else:
                orb = cluster_cubic_current_orbitals(route,0,0,.4,.35,hs,
                    regulator=action.regulator,total_pplus_GeV=1.3,transfer_GeV=DELTA,
                    mean_total_transverse_GeV=MEAN,flavor_charge=1/3,
                    prescription=PRESCRIPTION,quadrature_order=8)
                scalar = orb.regulator[1,0,0]
            for si,state in enumerate(origin.spins):
                parent = 2*state[4]-1 if channel=="g_to_pair" else 2*(state[route.parent]%2)-1
                if parent!=hs[2]:continue
                if channel=="g_to_pair":
                    targets = [tuple(state[:3])+(2*f+(hs[0]+1)//2,2*f+(hs[1]+1)//2,int(state[3]))
                               for f in range(2)]
                else:
                    changed = list(state);changed[route.parent]=2*(state[route.parent]//2)+(hs[0]+1)//2
                    targets = [tuple(changed)+((hs[1]+1)//2,)]
                for spin in targets:
                    so = np.ravel_multi_index(spin,target.spin_dims)
                    for co,ci in zip(*np.nonzero(color)):
                        ii = origin.lookup[(li*origin.ncolor+ci)*origin.nspin+si]
                        oo = target.lookup[(lo*target.ncolor+co)*target.nspin+so]
                        if min(ii,oo)>=0:
                            expected[route.sector][oo] += fock*scalar*color[co,ci]*v[source][ii]
    return {s:sp.statistics_project(expected[s]) for s,sp in outgoing.items()}


@pytest.mark.parametrize("source,channel",CHANNELS)
def test_regulator_minus_and_full_Ward_by_independent_Fock_slots(source,channel):
    inc,out = spaces(0,"PROTON",1);x = raw(inc);a = make(inc,out,charge="baryon")
    a.routes = tuple(t for t in a.routes if t[0]==source and t[2].channel==channel)
    actual = a.apply(x,component=1,part="regulator")
    expected = slot_reference(a,x,source,channel)
    for s in out:np.testing.assert_allclose(actual[s],expected[s],atol=2e-16)
    jx,jy = [a.apply(x,component=m,part="total") for m in (2,3)]
    ward = slot_reference(a,x,source,channel,ward=True)
    for s in out:
        np.testing.assert_allclose(DELTA[0]*jx[s]+DELTA[1]*jy[s],ward[s],atol=3e-16)


def test_invalid_prescription_or_unbound_P_is_not_dimension_matched():
    p = pilot();s = IntermediateSpace("qqqgg",9,0,"PROTON",1)
    with pytest.raises(ValueError):
        build_cluster_excluded_cubic_current(object(),s,DELTA,**ARGS)
    with pytest.raises(ValueError):
        build_cluster_excluded_cubic_current(p,s,DELTA,**{**ARGS,"prescription":"bare"})
    inc,out = spaces(0,"PROTON",1);a = make(inc,out)
    for component,part in ((True,"total"),(4,"total"),(1,"auto")):
        with pytest.raises(ValueError):a.apply(raw(inc),component=component,part=part)

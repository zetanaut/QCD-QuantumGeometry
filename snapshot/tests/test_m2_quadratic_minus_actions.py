"""Independent old Fock enumerators, endpoint statistics and raw QQ adjoints."""
from math import sqrt
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_six_body_space import SixBodySpace
from deuteron_wigner.quantum.m2_six_body_cubic import five_statistics_project
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum import m2_instantaneous_pair_exit as old_pair
from deuteron_wigner.quantum import m2_six_body_quartic as old_six
from deuteron_wigner.quantum import m2_excluded_pair_conversion as old_conversion
from deuteron_wigner.quantum.m2_quadratic_minus_actions import (
    build_pair_minus_endpoint,build_six_minus_endpoint,FiveBodyQuadraticMinus,
)
from deuteron_wigner.quantum.m2_quadratic_minus_fock import (
    pair_emission_minus_kernel,direct_six_minus_kernel,conversion_minus_kernel,
)
from test_m2_excluded_transverse_contact import pilot


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
@pytest.mark.parametrize("cutoff",[0,1])
def test_pair_current_matches_independent_existing_Fock_enumerator(target,cutoff):
    p=pilot(target);s=IntermediateSpace("qqqgg",9,cutoff,target,1);pp=1.3
    actual=build_pair_minus_endpoint(p,s,(0.,0.),total_pplus_GeV=pp,charge="baryon")
    expected=old_pair._build_pair_source(p,cutoff,
        lambda *args:pair_emission_minus_kernel(*args,total_pplus_GeV=pp)/3,
        "independent normal-ordered current Fock enumeration")
    np.testing.assert_allclose(actual.current_per_gs2,expected.matrix_per_gs2,atol=2e-15)
    np.testing.assert_array_equal(actual.retained_columns,expected.retained_columns)
    assert np.linalg.norm(actual.current_per_gs2)>1e-4


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
@pytest.mark.parametrize("component,sector",[("valence_IF","qqqggg"),("gluon_IF_pair","qqqq-qbar-g")])
def test_six_current_matches_independent_existing_Fock_enumerator(monkeypatch,target,component,sector):
    p=build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    s=SixBodySpace(sector,9,1,target,1);pp=1.3
    actual=build_six_minus_endpoint(p,s,(0.,0.),component=component,total_pplus_GeV=pp,charge="baryon")
    monkeypatch.setattr(old_six,"mass_kernel",lambda route,hs:direct_six_minus_kernel(route,hs,total_pplus_GeV=pp)/3)
    expected=old_six.build_direct_source(p,out_cutoff=1,component=component)
    np.testing.assert_allclose(actual.current_per_gs2,expected.matrix_per_gs2,atol=2e-15)
    np.testing.assert_array_equal(actual.retained_columns,expected.retained_columns)
    assert np.linalg.norm(actual.current_per_gs2)>1e-4


def make_spaces(cutoff,target,jz):
    return {s:IntermediateSpace(s,9,cutoff,target,jz) for s in ("qqqgg","qqqq-qbar")}


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
def test_pair_conversion_against_independent_old_Fock_enumerator(monkeypatch,target):
    spaces=make_spaces(1,target,1);rng=np.random.default_rng(942)
    raw={s:five_statistics_project(sp,(rng.normal(size=(sp.dimension,1))+
            1j*rng.normal(size=(sp.dimension,1)))/sqrt(sp.dimension)) for s,sp in spaces.items()}
    monkeypatch.setattr(old_conversion,"instantaneous_conversion_kernel",
        lambda inc,out,hs:conversion_minus_kernel(inc,out,hs,total_pplus_GeV=1.3)/3)
    old=old_conversion.FiveBodyPairConversion(spaces,b_GeV=.4,channel="IF")
    new=FiveBodyQuadraticMinus(spaces,spaces,(0.,0.),b_GeV=.4,total_pplus_GeV=1.3,charge="baryon")
    for source,destination in (("qqqgg","qqqq-qbar"),("qqqq-qbar","qqqgg")):
        v={s:raw[s] if s==source else np.zeros_like(raw[s]) for s in spaces}
        np.testing.assert_allclose(new.apply(v)[destination],old.apply(v)[destination],atol=3e-16)


@pytest.mark.parametrize("target,jz",[("PROTON",1),("PROTON",-1),("NEUTRON",1),("NEUTRON",-1)])
def test_QQ_quadratic_minus_reversal_flavor_statistics_and_boost(target,jz):
    inc=make_spaces(1,target,1);out=make_spaces(1,target,jz);rng=np.random.default_rng(943)
    v={s:(rng.normal(size=(sp.dimension,1))+1j*rng.normal(size=(sp.dimension,1)))/sqrt(sp.dimension) for s,sp in inc.items()}
    w={s:(rng.normal(size=(sp.dimension,1))+1j*rng.normal(size=(sp.dimension,1)))/sqrt(sp.dimension) for s,sp in out.items()}
    delta=(.12,.16)
    def make(f=out,i=inc,d=delta,P=1.3,charge="electromagnetic"):
        return FiveBodyQuadraticMinus(f,i,d,b_GeV=.4,total_pplus_GeV=P,charge=charge)
    actual=make().apply(v);back=make(inc,out,tuple(-x for x in delta)).apply(w)
    np.testing.assert_allclose(sum(np.vdot(w[s],actual[s]) for s in out),
                               sum(np.vdot(back[s],v[s]) for s in inc),atol=3e-16)
    boosted=make(P=2.6).apply(v);up=make(charge="up_number").apply(v);down=make(charge="down_number").apply(v)
    for s,sp in out.items():
        np.testing.assert_allclose(boosted[s],actual[s]/4,atol=0.)
        np.testing.assert_allclose((2*up[s]-down[s])/3,actual[s],atol=4e-16)
        np.testing.assert_allclose(five_statistics_project(sp,actual[s]),actual[s],atol=4e-16)
    assert sum(np.linalg.norm(a) for a in actual.values())>1e-4


def test_second_shell_raw_QQ_action_and_reversed_endpoint():
    inc=make_spaces(2,"PROTON",1);out=make_spaces(2,"PROTON",-1);rng=np.random.default_rng(944)
    v={s:(rng.normal(size=(sp.dimension,1))+1j*rng.normal(size=(sp.dimension,1)))/sqrt(sp.dimension) for s,sp in inc.items()}
    w={s:(rng.normal(size=(sp.dimension,1))+1j*rng.normal(size=(sp.dimension,1)))/sqrt(sp.dimension) for s,sp in out.items()}
    f=FiveBodyQuadraticMinus(out,inc,(.12,.16),b_GeV=.4,total_pplus_GeV=1.3,charge="electromagnetic").apply(v)
    b=FiveBodyQuadraticMinus(inc,out,(-.12,-.16),b_GeV=.4,total_pplus_GeV=1.3,charge="electromagnetic").apply(w)
    np.testing.assert_allclose(sum(np.vdot(w[s],f[s]) for s in out),
                               sum(np.vdot(b[s],v[s]) for s in inc),atol=3e-16)


def test_invalid_quadratic_endpoint():
    p=pilot();s=IntermediateSpace("qqqq-qbar",9,0,"PROTON",1)
    with pytest.raises(ValueError):build_pair_minus_endpoint(p,s,(0,0),total_pplus_GeV=1.3,charge="baryon")

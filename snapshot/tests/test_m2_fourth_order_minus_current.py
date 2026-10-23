"""Whole-parent polynomial and common four-component LF boost identities."""
from dataclasses import replace
import numpy as np
import pytest
from deuteron_wigner.quantum import m2_fourth_order_minus_current as implementation
from deuteron_wigner.quantum.m2_fourth_order_minus_current import minus_coefficients_between,minus_current_between
from deuteron_wigner.quantum.m2_fourth_order_transverse_current import transverse_current_between
from deuteron_wigner.quantum.m2_fourth_order_h2 import current_between as plus_current_between
from deuteron_wigner.quantum.m2_free_minus_current import free_minus_current_between
from deuteron_wigner.quantum.m2_one_gluon_minus_contact import minus_contact_between
from deuteron_wigner.quantum.m2_shell_minus_current import shell_minus_current_between
from deuteron_wigner.quantum.m2_excluded_minus_contact import build_excluded_minus_contact,SixBodyMinusContact
from deuteron_wigner.quantum.m2_two_gluon_minus_current import retained_two_gluon_minus_between
from deuteron_wigner.quantum.m2_quadratic_minus_actions import (
    build_pair_minus_endpoint,build_six_minus_endpoint,FiveBodyQuadraticMinus,
)
from test_m2_fourth_order_transverse_current import model


@pytest.fixture(scope="module",params=[("PROTON",1),("PROTON",-1),("NEUTRON",1),("NEUTRON",-1)])
def endpoints(request):
    target,jz=request.param
    f=model(target,jz).snapshot(1.39);i=model(target,1).snapshot(1.47);delta=(.12,.16);P=1.3
    c=minus_coefficients_between(f,i,delta,total_pplus_GeV=P,charge="electromagnetic")
    return f,i,delta,P,c


def test_all_orders_against_full_parent_wave_products_including_degree_eight(endpoints):
    f,i,delta,P,c=endpoints;charge="electromagnetic";reverse=tuple(-x for x in delta)
    rng=np.random.default_rng(983);n=len(i.matrix)
    vi=rng.normal(size=n)+1j*rng.normal(size=n);vf=rng.normal(size=n)+1j*rng.normal(size=n)
    vi/=np.linalg.norm(vi);vf/=np.linalg.norm(vf)
    fw=[{s:a@vf[:,None] for s,a in order.items()} for order in f.five_wave_coefficients]
    iw=[{s:a@vi[:,None] for s,a in order.items()} for order in i.five_wave_coefficients]
    f6={s:a@vf[:,None] for s,a in f.six_wave_coefficient.items()}
    i6={s:a@vi[:,None] for s,a in i.six_wave_coefficient.items()}
    p0=free_minus_current_between(f.owner.pilot,i.owner.pilot,delta,total_pplus_GeV=P,charge=charge)
    p1=minus_contact_between(f.owner.pilot,i.owner.pilot,delta,total_pplus_GeV=P,charge=charge)
    p2=retained_two_gluon_minus_between(f.owner.pilot,i.owner.pilot,delta,total_pplus_GeV=P,charge=charge)
    c1={s:(build_excluded_minus_contact(i.owner.pilot,f.owner.spaces5[s],delta,total_pplus_GeV=P,charge=charge),
           build_excluded_minus_contact(f.owner.pilot,i.owner.spaces5[s],reverse,total_pplus_GeV=P,charge=charge))
        for s in i.owner.spaces5}
    p5=(build_pair_minus_endpoint(i.owner.pilot,f.owner.spaces5["qqqgg"],delta,total_pplus_GeV=P,charge=charge),
        build_pair_minus_endpoint(f.owner.pilot,i.owner.spaces5["qqqgg"],reverse,total_pplus_GeV=P,charge=charge))
    p6={}
    for s in i.owner.spaces6:
        component="valence_IF" if s=="qqqggg" else "gluon_IF_pair"
        p6[s]=(build_six_minus_endpoint(i.owner.pilot,f.owner.spaces6[s],delta,component=component,total_pplus_GeV=P,charge=charge),
               build_six_minus_endpoint(f.owner.pilot,i.owner.spaces6[s],reverse,component=component,total_pplus_GeV=P,charge=charge))
    c6=SixBodyMinusContact(i.owner.spaces5,f.owner.spaces6,delta,b_GeV=.4,total_pplus_GeV=P,mass_GeV=.35,charge=charge)
    r6=SixBodyMinusContact(f.owner.spaces5,i.owner.spaces6,reverse,b_GeV=.4,total_pplus_GeV=P,mass_GeV=.35,charge=charge)
    qq=FiveBodyQuadraticMinus(f.owner.spaces5,i.owner.spaces5,delta,b_GeV=.4,total_pplus_GeV=P,charge=charge)
    nodes=np.linspace(-.8,.8,9);curve=[]
    for g in nodes:
        wf={s:sum(g**(k+1)*a[s] for k,a in enumerate(fw)) for s in f.owner.spaces5}
        wi={s:sum(g**(k+1)*a[s] for k,a in enumerate(iw)) for s in i.owner.spaces5}
        value=np.vdot(vf,(p0+g*p1+g*g*p2)@vi)
        qvalue=qq.apply(wi)
        for s in i.owner.spaces5:
            free=shell_minus_current_between(f.owner.spaces5[s],i.owner.spaces5[s],wi[s],delta,
                b_GeV=.4,total_pplus_GeV=P,mass_GeV=.35,charge=charge)
            value+=np.vdot(wf[s],free+g*g*qvalue[s])
            forward,backward=c1[s]
            value+=g*(np.vdot(wf[s][:,0],forward.current_per_gs@vi[forward.retained_columns])+
                       np.vdot(vf[backward.retained_columns],backward.current_per_gs.conj().T@wi[s][:,0]))
        forward,backward=p5
        value+=g*g*(np.vdot(wf["qqqgg"][:,0],forward.current_per_gs2@vi[forward.retained_columns])+
                     np.vdot(vf[backward.retained_columns],backward.current_per_gs2.conj().T@wi["qqqgg"][:,0]))
        to6=c6.apply(wi);from6=r6.apply(wf)
        for s in i.owner.spaces6:
            free=shell_minus_current_between(f.owner.spaces6[s],i.owner.spaces6[s],g*g*i6[s],delta,
                b_GeV=.4,total_pplus_GeV=P,mass_GeV=.35,charge=charge)
            value+=np.vdot(g*g*f6[s],free)+g*(np.vdot(g*g*f6[s],to6[s])+np.vdot(from6[s],g*g*i6[s]))
            forward,backward=p6[s]
            value+=g**4*(np.vdot(f6[s][:,0],forward.current_per_gs2@vi[forward.retained_columns])+
                         np.vdot(vf[backward.retained_columns],backward.current_per_gs2.conj().T@i6[s][:,0]))
        curve.append(value)
    polynomial=np.polynomial.polynomial.polyfit(nodes,np.array(curve),8)
    expected=np.array([np.vdot(vf,c[k]@vi) for k in range(5)])
    np.testing.assert_allclose(polynomial[:5],expected,atol=8e-14)
    assert np.linalg.norm(polynomial[5:])>1e-13


def test_every_order_adjoint_and_both_norm_endpoints(endpoints):
    f,i,delta,P,c=endpoints
    reverse=minus_coefficients_between(i,f,tuple(-x for x in delta),total_pplus_GeV=P,charge="electromagnetic")
    for k in range(5):np.testing.assert_allclose(c[k],reverse[k].conj().T,atol=1e-14)
    raw=sum(.3**k*a for k,a in c.items())
    actual=minus_current_between(f,i,delta,total_pplus_GeV=P,charge="electromagnetic",normalized=True)
    np.testing.assert_allclose(actual,f.whitener@raw@i.whitener,atol=0.)


def test_matched_plus_transverse_minus_satisfy_the_same_kinematic_boost(endpoints):
    f,i,delta,P,c=endpoints;beta=np.array([.06,-.09])/P
    raw=sum(.3**k*a for k,a in c.items())
    boosted=minus_current_between(f,i,delta,total_pplus_GeV=P,charge="electromagnetic",
                                  normalized=False,mean_total_transverse=beta*P)
    jt=transverse_current_between(f,i,delta,total_pplus_GeV=P,charge="electromagnetic",normalized=False)
    jp=plus_current_between(f,i,delta,charge="electromagnetic",normalized=False)
    expected=np.einsum("j,joi->oi",beta,jt)+np.dot(beta,beta)/2*jp
    np.testing.assert_allclose(boosted-raw,expected,atol=1e-14)


def test_quadratic_current_blocks_are_not_silently_lost(monkeypatch,endpoints):
    f,i,delta,P,c=endpoints
    monkeypatch.setattr(implementation,"retained_two_gluon_minus_between",lambda *a,**k:np.zeros_like(c[0]))
    def omit_endpoint(fn):
        def wrapped(*a,**k):
            result=fn(*a,**k)
            return replace(result,current_per_gs2=np.zeros_like(result.current_per_gs2))
        return wrapped
    monkeypatch.setattr(implementation,"build_pair_minus_endpoint",omit_endpoint(build_pair_minus_endpoint))
    monkeypatch.setattr(implementation,"build_six_minus_endpoint",omit_endpoint(build_six_minus_endpoint))
    class OmittedQQ:
        def __init__(self,*a,**k):pass
        def apply(self,columns):return {s:np.zeros_like(a) for s,a in columns.items()}
    monkeypatch.setattr(implementation,"FiveBodyQuadraticMinus",OmittedQQ)
    omitted=minus_coefficients_between(f,i,delta,total_pplus_GeV=P,charge="electromagnetic")
    for k in (0,1):np.testing.assert_array_equal(omitted[k],c[k])
    if f.owner.pilot.jz_twice==i.owner.pilot.jz_twice:
        assert np.linalg.norm(c[2]-omitted[2])>1e-5
        assert np.linalg.norm(c[4]-omitted[4])>1e-8


def test_explicit_snapshot_and_normalization():
    with pytest.raises(ValueError):
        minus_coefficients_between(None,None,(0,0),total_pplus_GeV=1.3,charge="electromagnetic")
    with pytest.raises(ValueError):
        minus_current_between(None,None,(0,0),total_pplus_GeV=1.3,charge="electromagnetic",normalized=None)

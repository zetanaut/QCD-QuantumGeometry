from dataclasses import replace
from functools import lru_cache
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_canonical_qq import CanonicalQQSourceAction,CanonicalQQInsertion
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_fourth_order_h2 import FourthOrderH2
from deuteron_wigner.quantum.m2_fourth_order_transverse_current import (
    transverse_coefficients_between,transverse_current_between,
)
from deuteron_wigner.quantum.m2_transverse_current_projection import free_transverse_current_between
from deuteron_wigner.quantum.m2_transverse_contact_current import transverse_contact_between
from deuteron_wigner.quantum.m2_shell_transverse_current import shell_transverse_current_between
from deuteron_wigner.quantum.m2_excluded_transverse_contact import build_excluded_transverse_contact
from deuteron_wigner.quantum.m2_six_body_transverse_contact import SixBodyTransverseContact


@lru_cache(maxsize=4)
def model(target,jz):
    p=build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    r=build_order_resolved_reference(p,out_cutoff=0)
    qq=CanonicalQQInsertion(r,action=CanonicalQQSourceAction(r))
    return FourthOrderH2(r,QQ_insertion=qq,six_body=CanonicalSixBodyExcursion(r,out_cutoff=0))


@pytest.fixture(scope="module",params=[("PROTON",1),("PROTON",-1),("NEUTRON",1),("NEUTRON",-1)])
def endpoints(request):
    target,jz=request.param
    f=model(target,jz).snapshot(1.39);i=model(target,1).snapshot(1.47)
    delta=(.12,.16);P=1.3;charge="electromagnetic"
    coefficients=transverse_coefficients_between(f,i,delta,total_pplus_GeV=P,charge=charge)
    return f,i,delta,P,charge,coefficients


def test_all_coefficients_against_untruncated_raw_parent_wave_polynomial(endpoints):
    f,i,delta,P,charge,coefficients=endpoints
    rng=np.random.default_rng(735);n=len(i.matrix)
    vi=rng.normal(size=n)+1j*rng.normal(size=n);vf=rng.normal(size=n)+1j*rng.normal(size=n)
    vi/=np.linalg.norm(vi);vf/=np.linalg.norm(vf)
    fw=[{s:a@vf[:,None] for s,a in order.items()} for order in f.five_wave_coefficients]
    iw=[{s:a@vi[:,None] for s,a in order.items()} for order in i.five_wave_coefficients]
    f6={s:a@vf[:,None] for s,a in f.six_wave_coefficient.items()}
    i6={s:a@vi[:,None] for s,a in i.six_wave_coefficient.items()}
    pp0=free_transverse_current_between(f.owner.pilot,i.owner.pilot,delta,total_pplus_GeV=P,charge=charge)
    pp1=transverse_contact_between(f.owner.pilot,i.owner.pilot,delta,total_pplus_GeV=P,charge=charge)
    reverse=tuple(-x for x in delta)
    c5={s:(build_excluded_transverse_contact(i.owner.pilot,f.owner.spaces5[s],delta,total_pplus_GeV=P,charge=charge),
           build_excluded_transverse_contact(f.owner.pilot,i.owner.spaces5[s],reverse,total_pplus_GeV=P,charge=charge))
        for s in i.owner.spaces5}
    c6=SixBodyTransverseContact(i.owner.spaces5,f.owner.spaces6,delta,b_GeV=.4,total_pplus_GeV=P,charge=charge)
    r6=SixBodyTransverseContact(f.owner.spaces5,i.owner.spaces6,reverse,b_GeV=.4,total_pplus_GeV=P,charge=charge)
    nodes=np.linspace(-1,1,7);curve=[]
    for g in nodes:
        wf={s:sum(g**(k+1)*a[s] for k,a in enumerate(fw)) for s in f.owner.spaces5}
        wi={s:sum(g**(k+1)*a[s] for k,a in enumerate(iw)) for s in i.owner.spaces5}
        values=[]
        for j in (0,1):
            value=np.vdot(vf,(pp0[j]+g*pp1[j])@vi)
            for s in i.owner.spaces5:
                jq=shell_transverse_current_between(f.owner.spaces5[s],i.owner.spaces5[s],wi[s],delta,
                    b_GeV=.4,total_pplus_GeV=P,charge=charge,component=j)
                forward,backward=c5[s]
                value+=np.vdot(wf[s],jq)
                value+=g*np.vdot(wf[s][:,0],forward.current_per_gs[j]@vi[forward.retained_columns])
                value+=g*np.vdot(vf[backward.retained_columns],backward.current_per_gs[j].conj().T@wi[s][:,0])
            to6=c6.apply(wi,component=j);from6=r6.apply(wf,component=j)
            for s in i.owner.spaces6:
                jq=shell_transverse_current_between(f.owner.spaces6[s],i.owner.spaces6[s],g*g*i6[s],delta,
                    b_GeV=.4,total_pplus_GeV=P,charge=charge,component=j)
                value+=np.vdot(g*g*f6[s],jq)
                value+=g*np.vdot(g*g*f6[s],to6[s])+g*np.vdot(from6[s],g*g*i6[s])
            values.append(value)
        curve.append(values)
    # Whole-parent wave products include orders five/six. Extract all seven
    # coefficients before comparing only the requested zero-to-four subset.
    polynomial=np.polynomial.polynomial.polyfit(nodes,np.array(curve),6)
    expected=np.array([[np.vdot(vf,a@vi) for a in coefficients[k]] for k in range(5)])
    np.testing.assert_allclose(expected,polynomial[:5],atol=2e-14)
    assert np.linalg.norm(polynomial[5:])>1e-12


def test_every_order_reversed_transfer_and_both_snapshot_whiteners(endpoints):
    f,i,delta,P,charge,coefficients=endpoints
    reverse=transverse_coefficients_between(i,f,tuple(-x for x in delta),total_pplus_GeV=P,charge=charge)
    for power in range(5):
        np.testing.assert_allclose(coefficients[power],reverse[power].conj().swapaxes(1,2),atol=8e-15)
    expected=np.array([f.whitener@j@i.whitener for j in sum(.3**k*a for k,a in coefficients.items())])
    actual=transverse_current_between(f,i,delta,total_pplus_GeV=P,charge=charge,normalized=True)
    np.testing.assert_allclose(actual,expected,atol=5e-16)
    assert not f.owner.physical


def test_backflow_and_six_body_current_are_not_discarded(endpoints):
    f,i,delta,P,charge,coefficients=endpoints
    def omit(s):
        return replace(s,five_wave_coefficients=(*s.five_wave_coefficients[:2],
            {n:np.zeros_like(a) for n,a in s.five_wave_coefficients[2].items()}),
            six_wave_coefficient={n:np.zeros_like(a) for n,a in s.six_wave_coefficient.items()})
    incomplete=transverse_coefficients_between(omit(f),omit(i),delta,total_pplus_GeV=P,charge=charge)
    for k in (0,1,2,3):np.testing.assert_allclose(coefficients[k],incomplete[k],atol=0.)
    assert np.linalg.norm(coefficients[4]-incomplete[4])>1e-8


def test_no_implicit_snapshot_or_normalization_choice():
    with pytest.raises(ValueError):
        transverse_coefficients_between(None,None,(0,0),total_pplus_GeV=1.,charge="baryon")
    with pytest.raises(ValueError):
        transverse_current_between(None,None,(0,0),total_pplus_GeV=1.,charge="baryon",normalized=None)

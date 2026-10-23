"""Independent Fock-slot enumeration and complete charged minus-current actions."""
from itertools import product
from math import sqrt,pi
import numpy as np
import pytest
from deuteron_wigner.bridge.modes.core import polarization
from deuteron_wigner.quantum.m2_constrained_fermion_current import (
    one_gluon_current_spin,pair_creation_current_spin,discrete_gluon_field_factor,
)
from deuteron_wigner.quantum.m2_antiquark_contact_current import antiquark_one_gluon_current_spin
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_excluded_minus_contact import build_excluded_minus_contact,SixBodyMinusContact
from deuteron_wigner.quantum.m2_excluded_transverse_contact import build_excluded_transverse_contact
from deuteron_wigner.quantum.m2_six_body_transverse_contact import SixBodyTransverseContact
from deuteron_wigner.quantum.m2_six_body_cubic import five_statistics_project
from deuteron_wigner.quantum.m2_six_body_vertices import split_routes,color_source
from deuteron_wigner.quantum.m2_six_body_kinematics import frame
from test_m2_six_body_transverse_contact import spaces,CHANNELS
from test_m2_excluded_transverse_contact import pilot


@pytest.mark.parametrize("sector",["qqqgg","qqqq-qbar"])
@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
@pytest.mark.parametrize("jz",[-1,1])
def test_Q5_minus_contacts_flavor_statistics_and_boost(sector,target,jz):
    p=pilot(target);s=IntermediateSpace(sector,9,1,target,jz);delta=(.12,.16);pp=1.3
    boost=np.array([.04,-.07])
    def j(charge="electromagnetic",P=pp,mean=(0.,0.)):
        return build_excluded_minus_contact(p,s,delta,total_pplus_GeV=P,
            charge=charge,mean_total_transverse=mean).current_per_gs
    actual=j();transverse=build_excluded_transverse_contact(p,s,delta,total_pplus_GeV=pp,
        charge="electromagnetic").current_per_gs
    np.testing.assert_allclose(j(P=2*pp),actual/4,atol=0.)
    np.testing.assert_allclose(j(mean=boost)-actual,np.einsum("j,joi->oi",boost/pp,transverse),atol=3e-15)
    np.testing.assert_allclose((2*j("up_number")-j("down_number"))/3,actual,atol=3e-15)
    np.testing.assert_allclose(s.statistics_project(actual),actual,atol=3e-15)
    np.testing.assert_allclose(s.permute(actual,(1,0,2,3,4)),-actual,atol=3e-15)
    assert np.linalg.norm(actual)>1e-4


@pytest.mark.parametrize("source,channel",CHANNELS)
def test_ground_shell_Q6_minus_against_independent_slot_enumeration(source,channel):
    inc,out=spaces(0,"PROTON",1);P=1.3;b=.4;mass=.35
    action=SixBodyMinusContact(inc,out,(0.,0.),b_GeV=b,total_pplus_GeV=P,mass_GeV=mass,charge="baryon")
    action.routes=[term for term in action.routes if term[0]==source and term[2].channel==channel]
    rng=np.random.default_rng(482)
    raw={s:rng.normal(size=(sp.dimension,1))+1j*rng.normal(size=(sp.dimension,1)) for s,sp in inc.items()}
    v={s:five_statistics_project(sp,raw[s]) for s,sp in inc.items()}
    expected={s:np.zeros((sp.dimension,1),complex) for s,sp in out.items()}
    for route in split_routes(9,source,channel):
        target=out[route.sector];origin=inc[source];color=color_source(route)
        li=origin.longitudinal.index(route.incoming);lo=target.longitudinal.index(route.outgoing)
        x,y=frame(route.sector,route.outgoing).fractions[list(route.daughters)]
        fock=2*sqrt(2) if channel=="g_to_pair" else sqrt(3) if source=="qqqgg" else 1.
        charge=(-1 if channel=="anti_to_antig" else 1)/3
        for hg in (-1,1):
            if channel=="g_to_pair":
                eps=polarization((x+y)*P,0,0,hg)[2:]
                spin=pair_creation_current_spin((x*P,0,0),(y*P,0,0),mass_GeV=mass,
                    polarization=eps,right_plus=x*P,left_plus=-y*P)[1]
                field=discrete_gluon_field_factor(route.incoming[4])
            else:
                eps=polarization(y*P,0,0,hg).conj()[2:]
                fn=antiquark_one_gluon_current_spin if channel=="anti_to_antig" else one_gluon_current_spin
                rp,lp=(-P*(x+y),-P*x) if channel=="anti_to_antig" else (P*x,P*(x+y))
                spin=fn((x*P,0,0),((x+y)*P,0,0),mass_GeV=mass,polarization=eps,right_plus=rp,left_plus=lp)[1]
                field=discrete_gluon_field_factor(route.outgoing[5])
            spin*=fock*charge*field*b/sqrt(pi)*sqrt(x*y/(x+y))
            for si,state in enumerate(origin.spins):
                if channel=="g_to_pair" and 2*state[4]-1!=hg:continue
                for hf,hother in product(range(2),repeat=2):
                    if channel=="g_to_pair":
                        spin_targets=[tuple(state[:3])+(2*f+hf,2*f+hother,int(state[3])) for f in range(2)]
                    else:
                        if state[route.parent]%2!=hother:continue
                        changed=list(state);changed[route.parent]=2*(changed[route.parent]//2)+hf
                        spin_targets=[tuple(changed)+((hg+1)//2,)]
                    for out_spin in spin_targets:
                        so=np.ravel_multi_index(out_spin,target.spin_dims)
                        for co,ci in zip(*np.nonzero(color)):
                            ii=origin.lookup[(li*origin.ncolor+ci)*origin.nspin+si]
                            oo=target.lookup[(lo*target.ncolor+co)*target.nspin+so]
                            if min(ii,oo)>=0:expected[route.sector][oo]+=spin[hf,hother]*color[co,ci]*v[source][ii]
    actual=action.apply(raw);expected={s:sp.statistics_project(expected[s]) for s,sp in out.items()}
    for s in out:np.testing.assert_allclose(actual[s],expected[s],atol=5e-15)
    assert sum(np.linalg.norm(a) for a in actual.values())>1e-4


@pytest.mark.parametrize("target,jz",[("PROTON",1),("PROTON",-1),("NEUTRON",1),("NEUTRON",-1)])
def test_Q6_minus_adjoint_flavor_statistics_and_boost(target,jz):
    inc,out=spaces(1,target,jz);rng=np.random.default_rng(485);pp=1.3;delta=(.12,.16);boost=np.array([.04,-.07])
    v={s:(rng.normal(size=(sp.dimension,1))+1j*rng.normal(size=(sp.dimension,1)))/sqrt(sp.dimension) for s,sp in inc.items()}
    w={s:(rng.normal(size=(sp.dimension,1))+1j*rng.normal(size=(sp.dimension,1)))/sqrt(sp.dimension) for s,sp in out.items()}
    def make(charge="electromagnetic",P=pp,mean=(0.,0.)):
        return SixBodyMinusContact(inc,out,delta,b_GeV=.4,total_pplus_GeV=P,mass_GeV=.35,
                                  charge=charge,mean_total_transverse=mean)
    actual=make().apply(v);adjoint=make().adjoint(w)
    np.testing.assert_allclose(sum(np.vdot(w[s],actual[s]) for s in out),
        sum(np.vdot(adjoint[s],v[s]) for s in inc),atol=3e-16)
    longitudinal=make(P=2*pp).apply(v);shifted=make(mean=boost).apply(v)
    up=make("up_number").apply(v);down=make("down_number").apply(v)
    t=SixBodyTransverseContact(inc,out,delta,b_GeV=.4,total_pplus_GeV=pp,charge="electromagnetic")
    jt=[t.apply(v,component=j) for j in (0,1)]
    for s,sp in out.items():
        np.testing.assert_allclose(longitudinal[s],actual[s]/4,atol=0.)
        np.testing.assert_allclose(shifted[s]-actual[s],sum(boost[j]/pp*jt[j][s] for j in (0,1)),atol=8e-16)
        np.testing.assert_allclose((2*up[s]-down[s])/3,actual[s],atol=8e-16)
        np.testing.assert_allclose(sp.statistics_project(actual[s]),actual[s],atol=8e-16)
    for s,sp in inc.items():
        np.testing.assert_allclose(five_statistics_project(sp,adjoint[s]),adjoint[s],atol=8e-16)


def test_complete_second_shell_Q6_minus_adjoint():
    inc,out=spaces(2,"PROTON",-1);rng=np.random.default_rng(486)
    v={s:(rng.normal(size=(sp.dimension,1))+1j*rng.normal(size=(sp.dimension,1)))/sqrt(sp.dimension) for s,sp in inc.items()}
    w={s:(rng.normal(size=(sp.dimension,1))+1j*rng.normal(size=(sp.dimension,1)))/sqrt(sp.dimension) for s,sp in out.items()}
    action=SixBodyMinusContact(inc,out,(.12,.16),b_GeV=.4,total_pplus_GeV=1.3,mass_GeV=.35,charge="electromagnetic")
    actual=action.apply(v);adjoint=action.adjoint(w)
    np.testing.assert_allclose(sum(np.vdot(w[s],actual[s]) for s in out),
        sum(np.vdot(adjoint[s],v[s]) for s in inc),atol=3e-16)
    assert all(np.linalg.norm(v)>0 for v in actual.values())


def test_invalid_minus_endpoint():
    inc,out=spaces(0,"PROTON",1)
    with pytest.raises(ValueError):
        SixBodyMinusContact(inc,out,(0.,0.),b_GeV=.4,total_pplus_GeV=1.3,mass_GeV=-.1,charge="baryon")

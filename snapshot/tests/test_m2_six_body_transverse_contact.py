from functools import lru_cache
from itertools import product
from math import pi,sqrt
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from deuteron_wigner.bridge.modes.core import ho_momentum
from deuteron_wigner.quantum.m2_coupled_pair_action import circular_to_c45
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_shell
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import frame as five_frame
from deuteron_wigner.quantum.m2_six_body_space import SixBodySpace
from deuteron_wigner.quantum.m2_six_body_kinematics import frame
from deuteron_wigner.quantum.m2_six_body_vertices import split_routes,split_frame,color_source,cubic_mass_kernel
from deuteron_wigner.quantum.m2_six_body_cubic import five_statistics_project
from deuteron_wigner.quantum.m2_six_body_transverse_contact import six_contact_orbital,SixBodyTransverseContact


CHANNELS=[("qqqgg","q_to_qg"),("qqqgg","g_to_pair"),
          ("qqqq-qbar","q_to_qg"),("qqqq-qbar","anti_to_antig")]


@pytest.mark.parametrize("source,channel",CHANNELS)
def test_contact_orbital_against_independent_ten_dimensional_polar_quadrature(source,channel):
    route=split_routes(9,source,channel)[0];b=.4;delta=np.array([.12,-.16])
    fi,fo=five_frame(source,route.incoming),frame(route.sector,route.outgoing)
    x,y=fo.fractions[list(route.daughters)]
    shift=fi.forward[:-1,route.parent,None]*delta
    nodes,weights=hermgauss(3);indices=np.array(list(product(range(3),repeat=10)))
    t=nodes[indices].reshape(-1,5,2)
    weight=2*b**10*np.prod(weights[indices],axis=1)*np.exp(np.sum(t*t,axis=(1,2)))
    parent=b*t[:,:4]-shift/2
    adapted=np.concatenate((parent+shift,sqrt(2)*b*t[:,4:]),axis=1)
    global_out=np.einsum("ab,nbj->naj",split_frame(route).T,adapted)
    def waves(d,q):
        result=[]
        for occupation in oscillator_shell(d,1):
            v=np.ones(len(q),complex)
            for a in range(d):
                n,m,phase=circular_to_c45(occupation[a],occupation[d+a])
                v*=phase*ho_momentum(n,m,q[:,a,0],q[:,a,1],b)/(2*pi)
            result.append(v)
        return np.array(result)
    wi,wo=waves(4,parent),waves(5,global_out)
    expected=sqrt(x*y/(x+y))/(2*pi)*(wo.conj()*weight)@wi.T
    actual=six_contact_orbital(route,1,1,b,tuple(delta))
    np.testing.assert_allclose(actual,expected,atol=5e-15)


@lru_cache(maxsize=8)
def spaces(cutoff,target,jz):
    return ({s:IntermediateSpace(s,9,cutoff,target,1) for s in ("qqqgg","qqqq-qbar")},
            {s:SixBodySpace(s,9,cutoff,target,jz) for s in ("qqqggg","qqqq-qbar-g")})


@pytest.mark.parametrize("source,channel",CHANNELS)
@pytest.mark.parametrize("component",[0,1])
def test_zero_transfer_action_against_independent_Hamiltonian_vertex_derivative(source,channel,component):
    inc,out=spaces(0,"PROTON",-1);P=1.3;b=.4
    action=SixBodyTransverseContact(inc,out,(0.,0.),b_GeV=b,total_pplus_GeV=P,charge="baryon")
    action.routes=[term for term in action.routes if term[0]==source and term[2].channel==channel]
    rng=np.random.default_rng(738)
    raw={s:rng.normal(size=(space.dimension,1))+1j*rng.normal(size=(space.dimension,1)) for s,space in inc.items()}
    v={s:five_statistics_project(space,raw[s]) for s,space in inc.items()}
    expected={s:np.zeros((space.dimension,1),complex) for s,space in out.items()}
    # Independently enumerate distinguished Fock slots and differentiate the
    # Hamiltonian spinor vertex, not the current kernel or its term iterator.
    for route in split_routes(9,source,channel):
        target=out[route.sector];origin=inc[source]
        li=origin.longitudinal.index(route.incoming);lo=target.longitudinal.index(route.outgoing)
        x,y=frame(route.sector,route.outgoing).fractions[list(route.daughters)]
        recoil=1. if channel=="g_to_pair" else y/(x+y)
        fock=(2*sqrt(2) if channel=="g_to_pair" else sqrt(3) if source=="qqqgg" else 1.)
        charge=(-1 if channel=="anti_to_antig" else 1)/3
        color=color_source(route);direction=np.eye(2)[component]
        for h1,h2,h0 in product((-1,1),repeat=3):
            dv=(cubic_mass_kernel(route,direction,.35,(h1,h2,h0),total_pplus=P)
                -cubic_mass_kernel(route,-direction,.35,(h1,h2,h0),total_pplus=P))/2
            coefficient=fock*charge*b/sqrt(pi)*recoil*dv/(2*P)
            if coefficient==0:continue
            for si,state in enumerate(origin.spins):
                parent=(2*state[route.parent]-1 if channel=="g_to_pair" else 2*(state[route.parent]%2)-1)
                if parent!=h0:continue
                if channel=="g_to_pair":
                    spin_targets=[tuple(state[:3])+(2*f+(h1+1)//2,2*f+(h2+1)//2,int(state[3])) for f in range(2)]
                else:
                    changed=list(state);changed[route.parent]=2*(changed[route.parent]//2)+(h1+1)//2
                    spin_targets=[tuple(changed)+((h2+1)//2,)]
                for target_spin in spin_targets:
                    so=np.ravel_multi_index(target_spin,target.spin_dims)
                    for co,ci in zip(*np.nonzero(color)):
                        ii=origin.lookup[(li*origin.ncolor+ci)*origin.nspin+si]
                        oo=target.lookup[(lo*target.ncolor+co)*target.nspin+so]
                        if min(ii,oo)>=0:expected[route.sector][oo]+=coefficient*color[co,ci]*v[source][ii]
    expected={s:space.statistics_project(expected[s]) for s,space in out.items()}
    actual=action.apply(raw,component=component)
    for s in out:np.testing.assert_allclose(actual[s],expected[s],atol=5e-15)
    assert sum(np.linalg.norm(a) for a in actual.values())>1e-4


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
@pytest.mark.parametrize("jz",[-1,1])
def test_exact_adjoint_flavor_boost_and_both_statistics_endpoints(target,jz):
    inc,out=spaces(1,target,jz);rng=np.random.default_rng(524)
    v={s:(rng.normal(size=(space.dimension,1))+1j*rng.normal(size=(space.dimension,1)))/sqrt(space.dimension)
       for s,space in inc.items()}
    w={s:(rng.normal(size=(space.dimension,1))+1j*rng.normal(size=(space.dimension,1)))/sqrt(space.dimension)
       for s,space in out.items()}
    def make(P=1.3,charge="electromagnetic"):
        return SixBodyTransverseContact(inc,out,(.12,.16),b_GeV=.4,total_pplus_GeV=P,charge=charge)
    a=make()
    for j in (0,1):
        actual=a.apply(v,component=j);adjoint=a.adjoint(w,component=j)
        lhs=sum(np.vdot(w[s],actual[s]) for s in out)
        rhs=sum(np.vdot(adjoint[s],v[s]) for s in inc)
        assert lhs==pytest.approx(rhs,abs=3e-16)
        boosted=make(P=2.6).apply(v,component=j)
        up=make(charge="up_number").apply(v,component=j);down=make(charge="down_number").apply(v,component=j)
        for s,space in out.items():
            np.testing.assert_allclose(actual[s],2*boosted[s],atol=2e-15)
            np.testing.assert_allclose(actual[s],(2*up[s]-down[s])/3,atol=2e-15)
            np.testing.assert_allclose(actual[s],space.statistics_project(actual[s]),atol=2e-15)
        for s,space in inc.items():
            np.testing.assert_allclose(adjoint[s],five_statistics_project(space,adjoint[s]),atol=2e-15)
    assert not a.physical and not a.complete_H2_matched_current


def test_explicit_invalid_endpoint_controls():
    inc,out=spaces(0,"PROTON",-1)
    with pytest.raises(ValueError):
        SixBodyTransverseContact(inc,out,(0,0),b_GeV=.4,total_pplus_GeV=0.,charge="baryon")
    with pytest.raises(ValueError):
        SixBodyTransverseContact(inc,spaces(0,"NEUTRON",-1)[1],(0,0),b_GeV=.4,total_pplus_GeV=1.,charge="baryon")


def test_complete_second_shell_action_retains_all_raw_coordinates():
    inc,out=spaces(2,"PROTON",-1);rng=np.random.default_rng(365)
    v={s:(rng.normal(size=(space.dimension,1))+1j*rng.normal(size=(space.dimension,1)))/sqrt(space.dimension)
       for s,space in inc.items()}
    w={s:(rng.normal(size=(space.dimension,1))+1j*rng.normal(size=(space.dimension,1)))/sqrt(space.dimension)
       for s,space in out.items()}
    a=SixBodyTransverseContact(inc,out,(.12,.16),b_GeV=.4,total_pplus_GeV=1.3,charge="electromagnetic")
    for j in (0,1):
        actual=a.apply(v,component=j);adjoint=a.adjoint(w,component=j)
        lhs=sum(np.vdot(w[s],actual[s]) for s in out)
        rhs=sum(np.vdot(adjoint[s],v[s]) for s in inc)
        assert lhs==pytest.approx(rhs,abs=3e-16)
        assert all(np.linalg.norm(value)>1e-5 for value in actual.values())
        assert all(actual[s].shape==(space.dimension,1) for s,space in out.items())

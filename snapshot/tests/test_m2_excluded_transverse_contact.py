"""Excluded electromagnetic contacts: continuum projection and H-vertex oracle."""
from functools import lru_cache
from itertools import product
from math import pi,sqrt
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from deuteron_wigner.bridge.modes.core import ho_momentum
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_shell
from deuteron_wigner.quantum.m2_coupled_pair_action import circular_to_c45
from deuteron_wigner.quantum.m2_coupled_pilot import symmetry_basis
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import split_routes,split_frame,frame
from deuteron_wigner.quantum.m2_excluded_fock_vertices import cubic_mass_kernel
from deuteron_wigner.quantum.m2_excluded_transverse_contact import (
    split_contact_orbital,build_excluded_transverse_contact,
)


@lru_cache(maxsize=4)
def pilot(target="PROTON",jz=1):
    return SimpleNamespace(total_k2=9,cutoff=0,target=target,jz_twice=jz,
        width_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,
        bases=tuple(symmetry_basis(s,9,0,target,jz,h)
                    for s,h in (("qqq",0),("qqqg",-1),("qqqg",1))))


@pytest.mark.parametrize("channel,parent",[("q_to_qg",i) for i in range(3)]+[("g_to_pair",3)])
def test_displaced_local_contact_against_eight_dimensional_polar_wave_integration(channel,parent):
    r=next(r for r in split_routes(9,channel) if r.parent==parent)
    b=.4;delta=np.array([.12,-.16])
    fi,fo=frame("qqqg",r.incoming),frame(r.sector,r.outgoing)
    x,y=fo.fractions[list(r.daughters)]
    shift=fi.forward[:-1,parent,None]*delta[None,:]
    nodes,w=hermgauss(3)
    indices=np.array(list(product(range(3),repeat=8)))
    t=nodes[indices].reshape(-1,4,2)
    qin=b*t[:,:3]-shift/2
    adapted=np.concatenate((qin+shift,sqrt(2)*b*t[:,3:]),axis=1)
    qout=np.einsum("ab,nbj->naj",split_frame(r).T,adapted)
    weight=np.prod(w[indices],axis=1)*2*b**8*np.exp(np.sum(t*t,axis=(1,2)))
    def waves(d,cutoff,q):
        values=[]
        for occupation in oscillator_shell(d,cutoff):
            v=np.ones(len(q),complex)
            for a in range(d):
                n,m,phase=circular_to_c45(occupation[a],occupation[d+a])
                v*=phase*ho_momentum(n,m,q[:,a,0],q[:,a,1],b)/(2*pi)
            values.append(v)
        return np.array(values)
    wi,wo=waves(3,1,qin),waves(4,2,qout)
    expected=sqrt(x*y/(x+y))/(2*pi)*(wo.conj()*weight)@wi.T
    np.testing.assert_allclose(split_contact_orbital(r,1,2,b,tuple(delta)),expected,atol=3e-15)


@pytest.mark.parametrize("sector",["qqqgg","qqqq-qbar"])
@pytest.mark.parametrize("jz",[-1,3])
@pytest.mark.parametrize("component",[0,1])
def test_zero_transfer_normalization_against_existing_Hamiltonian_vertex_derivative(
        sector,jz,component,monkeypatch):
    import deuteron_wigner.quantum.m2_excluded_fock_action as old
    p=pilot();s=IntermediateSpace(sector,9,0,"PROTON",jz);P=1.3
    actual=build_excluded_transverse_contact(p,s,(0.,0.),total_pplus_GeV=P,
                                            charge="baryon").current_per_gs[component]
    # Reuse the independently tested HAMILTONIAN color/statistics/creation
    # implementation, replacing only its local scalar vertex by the
    # derivative fixed by the unprojected plane-wave Ward identity.
    class CurrentEndpoint(IntermediateSpace):
        def add(self,output,li,spins,amp):
            raw=(((li*len(self.orbitals)+np.arange(len(self.orbitals))[:,None,None])
                *self.ncolor+np.arange(self.ncolor)[None,:,None])*self.nspin
                +np.asarray(spins)[None,None,:])
            rows=self.lookup[raw];allowed=rows>=0
            output[rows[allowed]]+=amp[allowed]
    def factory(sector,total,cutoff,target,ignored_jz):
        return CurrentEndpoint(sector,total,cutoff,target,jz)
    def derivative(route,inc,out,b,mass,helicities):
        assert inc==out==0
        if route.channel=="g_to_gg":
            return np.zeros((1,1),complex)  # neutral tree gluon vertex
        d=np.eye(2)[component]
        dv=(cubic_mass_kernel(route,d,mass,helicities,total_pplus=P)
             -cubic_mass_kernel(route,-d,mass,helicities,total_pplus=P))/2
        x,y=frame(route.sector,route.outgoing).fractions[list(route.daughters)]
        recoil=y/(x+y) if route.channel=="q_to_qg" else 1.
        return np.array([[b/sqrt(pi)*recoil*dv/(2*P)]])
    monkeypatch.setattr(old,"IntermediateSpace",factory)
    monkeypatch.setattr(old,"cubic_orbital",derivative)
    expected=old.build_excluded_source(p,sector,out_cutoff=0).matrix_per_gs/3
    np.testing.assert_allclose(actual,expected,atol=3e-15)
    assert np.linalg.norm(actual)>1e-3


@pytest.mark.parametrize("sector",["qqqgg","qqqq-qbar"])
@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
@pytest.mark.parametrize("jz",[-1,1])
def test_complete_contact_flavor_boost_and_fermion_boson_statistics(sector,target,jz):
    p=pilot(target);s=IntermediateSpace(sector,9,1,target,jz)
    def contact(charge="electromagnetic",P=1.3):
        return build_excluded_transverse_contact(p,s,(.12,.16),total_pplus_GeV=P,charge=charge)
    c=contact();a=c.current_per_gs
    assert np.linalg.norm(a)>1e-4
    assert a.shape==(2,s.dimension,len(c.retained_columns))
    np.testing.assert_allclose(contact(P=2.6).current_per_gs,a/2,atol=2e-15)
    np.testing.assert_allclose((2*contact("up_number").current_per_gs
                               -contact("down_number").current_per_gs)/3,a,atol=2e-15)
    for value in a:
        np.testing.assert_allclose(s.statistics_project(value),value,atol=3e-15)
        np.testing.assert_allclose(s.permute(value,(1,0,2,3,4)),-value,atol=3e-15)


@pytest.mark.parametrize("sector",["qqqgg","qqqq-qbar"])
def test_second_shell_and_spin_vector_selection(sector):
    p=pilot()
    same=IntermediateSpace(sector,9,2,"PROTON",1)
    flip=IntermediateSpace(sector,9,2,"PROTON",-1)
    a=build_excluded_transverse_contact(p,same,(0.,0.),total_pplus_GeV=1.3,
                                       charge="electromagnetic").current_per_gs
    b=build_excluded_transverse_contact(p,flip,(0.,0.),total_pplus_GeV=1.3,
                                       charge="electromagnetic").current_per_gs
    np.testing.assert_allclose(a,0.,atol=2e-15)
    np.testing.assert_allclose(b[0]+1j*b[1],0.,atol=2e-15)
    assert np.linalg.norm(b)>.01


def test_mismatched_physical_endpoints_are_not_defaulted():
    p=pilot();s=IntermediateSpace("qqqgg",9,0,"NEUTRON",1)
    with pytest.raises(ValueError):
        build_excluded_transverse_contact(p,s,(0,0),total_pplus_GeV=1.,charge="baryon")

"""Independent vertex Ward identity, six-dimensional integration and symmetry."""
from functools import lru_cache
from itertools import product
from math import pi,sqrt
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy import sparse

from deuteron_wigner.bridge.modes.core import ho_momentum,polarization
from deuteron_wigner.quantum.m2_canonical_emission import emission_mass_kernel_per_gs
from deuteron_wigner.quantum.m2_canonical_good_current import parton_charge
from deuteron_wigner.quantum.m2_constrained_fermion_current import (
    one_gluon_current_spin,discrete_gluon_field_factor,
)
from deuteron_wigner.quantum.m2_c117_pair_current import color_permutation_matrix
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import (
    emission_routes,emission_coordinate_map,jacobi_frame,oscillator_shell,
    longitudinal_orbital_action,
)
from deuteron_wigner.quantum.m2_coupled_pair_action import circular_to_c45
from deuteron_wigner.quantum.m2_coupled_pilot import symmetry_basis
from deuteron_wigner.quantum.m2_spin_flavor_s3_completion import spin_flavor_permutation_representation
from deuteron_wigner.quantum.m2_transverse_contact_current import (
    contact_route_orbital,emission_transverse_contact,transverse_contact_between,
)


@pytest.mark.parametrize("hg",[-1,1])
@pytest.mark.parametrize("mass",[0.,.35])
@pytest.mark.parametrize("pplus",[.7,2.3])
def test_plane_vertex_ward_identity_fixes_contact_sign_and_normalization(hg,mass,pplus):
    routes=emission_routes(9)
    # Every longitudinal route, not just one favored emitter or denominator.
    for route in routes:
        frame=jacobi_frame(route.outgoing)
        xq,xg=frame.fractions[[route.emitter,3]];xp=xq+xg
        scale=sqrt(xq*xg/xp)
        pq=np.array([.11,-.18]);pg=np.array([.21,.08]);delta=np.array([.12,.16])
        q1=(pq-xq/xp*(pq+pg))/scale
        q2=(pq-delta-xq/xp*(pq+pg-delta))/scale
        eps=polarization(xg*pplus,*pg,hg).conj()[2:]
        current=one_gluon_current_spin((xq*pplus,*pq),(xp*pplus,*(pq+pg-delta)),
            mass_GeV=mass,polarization=eps,right_plus=xq*pplus,left_plus=xp*pplus)[2:]
        current*=discrete_gluon_field_factor(route.outgoing[3])
        diff=np.array([[emission_mass_kernel_per_gs(route,q1,mass,ho,hi,hg,
                total_pplus=pplus,parent_transverse=pq+pg)
            -emission_mass_kernel_per_gs(route,q2,mass,ho,hi,hg,
                total_pplus=pplus,parent_transverse=pq+pg-delta)
            for hi in (-1,1)] for ho in (-1,1)])
        np.testing.assert_allclose(diff,2*pplus*np.einsum("j,joi->oi",delta,current),atol=8e-16)


@pytest.mark.parametrize("emitter",range(3))
@pytest.mark.parametrize("cutoffs",[(0,0),(1,2)])
def test_local_contact_against_six_dimensional_polar_wave_quadrature(emitter,cutoffs):
    route=next(r for r in emission_routes(9) if r.emitter==emitter)
    inc,out=cutoffs;b=.4;delta=np.array([.12,-.16])
    frame=jacobi_frame(route.outgoing);xp=jacobi_frame(route.incoming)
    xq,xg=frame.fractions[[emitter,3]]
    shift=xp.forward[:-1,emitter,None]*delta[None,:]
    nodes,w=hermgauss(5)
    indices=np.array(list(product(range(len(nodes)),repeat=6)))
    t=nodes[indices].reshape(-1,3,2)
    weight=np.prod(w[indices],axis=1)*2*b**6*np.exp(np.sum(t*t,axis=(1,2)))
    parent=b*t[:,:2]-shift/2
    adapted=np.concatenate((parent+shift,sqrt(2)*b*t[:,2:]),axis=1)
    global_out=np.einsum("ab,nbj->naj",emission_coordinate_map(route).T,adapted)
    def waves(d,cutoff,q):
        result=[]
        for occupation in oscillator_shell(d,cutoff):
            value=np.ones(len(q),complex)
            for a in range(d):
                n,m,phase=circular_to_c45(occupation[a],occupation[d+a])
                value*=phase*ho_momentum(n,m,q[:,a,0],q[:,a,1],b)/(2*pi)
            result.append(value)
        return np.array(result)
    wi,wo=waves(2,inc,parent),waves(3,out,global_out)
    expected=sqrt(xq*xg/(xq+xg))/(2*pi)*(wo.conj()*weight)@wi.T
    actual=contact_route_orbital(route,inc,out,b,tuple(delta))
    np.testing.assert_allclose(actual,expected,atol=3e-15)


@pytest.mark.parametrize("hg",[-1,1])
def test_raw_contact_preserves_joint_quark_permutation_covariance(hg):
    currents=emission_transverse_contact(9,1,1,(.12,.16),b_GeV=.4,
        total_pplus_GeV=1.3,gluon_helicity=hg,charge="electromagnetic")
    def perm(sector,p):
        return sparse.kron(longitudinal_orbital_action(sector,9,1,p),
            sparse.kron(color_permutation_matrix(sector,p),
                        spin_flavor_permutation_representation(p)),format="csr")
    rng=np.random.default_rng(23)
    v=rng.normal(size=(currents[0].shape[1],2))+1j*rng.normal(size=(currents[0].shape[1],2))
    for p in [(1,0,2),(1,2,0)]:
        pi_,pf=perm("qqq",p),perm("qqqg",p)
        for current in currents:
            np.testing.assert_allclose(pf@(current@v),current@(pi_@v),atol=2e-15)


@lru_cache(maxsize=8)
def endpoint(cutoff,target,jz):
    return SimpleNamespace(total_k2=9,cutoff=cutoff,target=target,jz_twice=jz,
        width_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,
        bases=tuple(symmetry_basis(s,9,cutoff,target,jz,hg)
                    for s,hg in (("qqq",0),("qqqg",-1),("qqqg",1))))


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
@pytest.mark.parametrize("jz",[1,-1])
def test_contact_reversal_flavor_boost_and_offdiagonal_support(target,jz):
    pi_=endpoint(1,target,1);pf=endpoint(1,target,jz)
    delta=np.array([.12,.16])
    def current(out,inc,d,charge="electromagnetic",pplus=1.3):
        return transverse_contact_between(out,inc,d,total_pplus_GeV=pplus,charge=charge)
    value=current(pf,pi_,delta)
    np.testing.assert_allclose(value,current(pi_,pf,-delta).conj().swapaxes(1,2),atol=2e-15)
    np.testing.assert_allclose(current(pf,pi_,delta,pplus=2.6),value/2,atol=2e-15)
    np.testing.assert_allclose((2*current(pf,pi_,delta,"up_number")
                               -current(pf,pi_,delta,"down_number"))/3,value,atol=2e-15)
    nf=pf.bases[0].shape[1];ni=pi_.bases[0].shape[1]
    assert np.linalg.norm(value[:,nf:,:ni])>1e-4
    np.testing.assert_array_equal(value[:,:nf,:ni],0.)
    np.testing.assert_array_equal(value[:,nf:,ni:],0.)


def test_zero_transfer_vector_selection_and_nonzero_spin_transition():
    p=endpoint(0,"PROTON",1);m=endpoint(0,"PROTON",-1)
    same=transverse_contact_between(p,p,(0.,0.),total_pplus_GeV=1.3,charge="electromagnetic")
    flip=transverse_contact_between(m,p,(0.,0.),total_pplus_GeV=1.3,charge="electromagnetic")
    np.testing.assert_allclose(same,0.,atol=2e-15)
    assert np.linalg.norm(flip)>1e-3
    # J_x+i J_y raises Jz, hence vanishes between +1/2 and -1/2.
    np.testing.assert_allclose(flip[0]+1j*flip[1],0.,atol=2e-15)


def test_invalid_current_controls_fail():
    with pytest.raises(ValueError):
        emission_transverse_contact(9,0,0,(0,0),b_GeV=.4,total_pplus_GeV=0,
                                    gluon_helicity=1,charge="baryon")
    p=endpoint(0,"PROTON",1)
    with pytest.raises(ValueError):
        transverse_contact_between(p,p,(0.,),total_pplus_GeV=1.,charge="baryon")

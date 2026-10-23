"""Independent field/CCR, complete-color, Jacobi and statistics checks."""
from itertools import combinations, product
from math import sqrt
import numpy as np
import pytest
from scipy import sparse

from deuteron_wigner.microscopic.h0.color import representation_generators
from deuteron_wigner.pilot.color import structure_constants
from deuteron_wigner.quantum.m2_canonical_mixed_current import canonical_gluon_current_factor
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_excluded_current_scattering import (
    current_pair_color,five_body_contact,current_scattering_per_gs2,
)
from test_m2_excluded_fermion_scattering import direct_contact


def test_all_pair_color_Casimir_on_complete_six_singlets_and_adjoint_field_sign():
    total=sum(current_pair_color("qqqgg",i,j) for i,j in combinations(range(5),2))
    np.testing.assert_allclose(total,-5*np.eye(6),atol=6e-15)
    f=structure_constants()
    for ki,ko,a in product((2,4),(2,6),range(8)):
        # Two independent number-preserving contractions in +f A dA.
        direct=(-1j*ki*f[a]+1j*ko*f[a].T)/(2*sqrt(ki*ko))
        np.testing.assert_allclose(direct,
            canonical_gluon_current_factor(ko,ki)*representation_generators("8")[a],atol=3e-15)


def test_boson_normal_order_pair_count_against_independent_occupation_CCR():
    # :[dGamma(T)]²:/2 on the normalized two-boson sector equals T_1 T_2.
    # The single-contraction dGamma(T²) is subtracted, not retained as a pair.
    modes=3
    labels=list(product(range(3),repeat=modes));lookup={s:i for i,s in enumerate(labels)}
    def annihilator(k):
        a=np.zeros((len(labels),len(labels)),complex)
        for col,s in enumerate(labels):
            if s[k]:
                out=list(s);out[k]-=1;a[lookup[tuple(out)],col]=sqrt(s[k])
        return a
    a=[annihilator(k) for k in range(modes)]
    t=np.array([[.4,.2j,.13],[-.2j,-.1,.17j],[.13,-.17j,.3]])
    second=lambda m:sum(m[i,j]*a[i].conj().T@a[j] for i,j in product(range(modes),repeat=2))
    direct=(second(t)@second(t)-second(t@t))/2
    pair_labels=[(i,j) for i in range(modes) for j in range(i,modes)]
    u=np.zeros((9,len(pair_labels)),complex);v=np.zeros((len(labels),len(pair_labels)),complex)
    for col,(i,j) in enumerate(pair_labels):
        if i==j:u[3*i+j,col]=1
        else:u[3*i+j,col]=u[3*j+i,col]=1/sqrt(2)
        s=[0]*modes;s[i]+=1;s[j]+=1;v[lookup[tuple(s)],col]=1
    np.testing.assert_allclose(v.conj().T@direct@v,u.conj().T@np.kron(t,t)@u,atol=3e-15)


@pytest.mark.parametrize("incoming,outgoing,pair",[
    ((3,1,1,2,2),(1,1,1,4,2),(0,3)),
    ((1,1,1,2,4),(1,1,1,4,2),(3,4)),
])
def test_mixed_and_gluon_contact_against_independent_physical_quadrature(incoming,outgoing,pair):
    np.testing.assert_allclose(five_body_contact("qqqgg",incoming,outgoing,pair,.4,1),
                               direct_contact("qqqgg",incoming,outgoing,pair,.4),atol=4e-15)


@pytest.mark.parametrize("channel,cutoff,target",[
    ("qg",0,"PROTON"),("qg",1,"NEUTRON"),
    ("gg",0,"NEUTRON"),("gg",1,"PROTON"),
])
def test_actual_number_current_QQ_matrices_Hermiticity_Q0_and_statistics(channel,cutoff,target):
    space=IntermediateSpace("qqqgg",9,cutoff,target,1)
    w=current_scattering_per_gs2(space,b_GeV=.4,channel=channel)
    assert sparse.linalg.norm(w)>1e-3
    assert sparse.linalg.norm(w-w.getH())<4e-14
    assert sparse.linalg.norm(current_scattering_per_gs2(space,b_GeV=.8,channel=channel)-4*w)<4e-14
    for li in range(len(space.longitudinal)):
        indices=np.flatnonzero(space.coordinates[:,0]==li)
        assert w[indices][:,indices].nnz==0
    rng=np.random.default_rng(492)
    v=rng.normal(size=(space.dimension,2))+1j*rng.normal(size=(space.dimension,2))
    for p in ((1,0,2,3,4),(0,1,2,4,3)):
        np.testing.assert_allclose(w@space.permute(v,p),space.permute(w@v,p),atol=4e-14)


def test_no_scattering_at_kinematic_minimum_and_no_silent_unknown_component():
    space=IntermediateSpace("qqqgg",7,0,"PROTON",1)
    for c in ("qg","gg"):assert current_scattering_per_gs2(space,b_GeV=.4,channel=c).nnz==0
    with pytest.raises(ValueError):
        current_scattering_per_gs2(space,b_GeV=.4,channel="complete QQ")


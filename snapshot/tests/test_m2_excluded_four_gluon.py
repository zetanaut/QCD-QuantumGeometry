from itertools import combinations,product
from math import pi,sqrt
import numpy as np
import pytest
from scipy import sparse

from deuteron_wigner.pilot.color import levi_civita3,structure_constants
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import color_basis
from deuteron_wigner.quantum.m2_excluded_four_gluon import (
    distinguished_quartic_tensor,quartic_spin_color,four_gluon_QQ_per_gs2,
)


def test_normal_ordered_SU2_A4_against_independent_two_boson_occupation_algebra():
    # Six modes: three adjoint colors times two Cartesian polarizations.
    # Direct normal ordering uses six choices of creation positions; ladder
    # amplitudes supply the contractions. It does not reuse 24 leg permutations.
    f=levi_civita3().real
    pairs=[(i,j) for i in range(6) for j in range(i,6)]
    labels=[]
    for i,j in pairs:
        s=[0]*6;s[i]+=1;s[j]+=1;labels.append(tuple(s))
    lookup={s:i for i,s in enumerate(labels)}
    direct=np.zeros((21,21),complex)
    for adj,b,c,d,e,ip,jp in product(range(3),range(3),range(3),range(3),range(3),range(2),range(2)):
        coef=.25*f[adj,b,c]*f[adj,d,e]
        if coef==0:continue
        slots=(2*b+ip,2*c+jp,2*d+ip,2*e+jp)
        for creation_positions in combinations(range(4),2):
            annih=[slots[i] for i in range(4) if i not in creation_positions]
            create=[slots[i] for i in creation_positions]
            for col,initial in enumerate(labels):
                s=list(initial);amplitude=coef
                for k in annih:
                    if s[k]==0:amplitude=0;break
                    amplitude*=sqrt(s[k]);s[k]-=1
                if amplitude==0:continue
                for k in create:
                    s[k]+=1;amplitude*=sqrt(s[k])
                direct[lookup[tuple(s)],col]+=amplitude
    raw=np.zeros((3,2,3,2,3,2,3,2),complex)
    for pols in product(range(2),repeat=4):
        tensor=distinguished_quartic_tensor(f,np.eye(2)[list(pols)])/8
        raw[:,pols[0],:,pols[1],:,pols[2],:,pols[3]]=tensor
    u=np.zeros((36,21),complex)
    for col,(i,j) in enumerate(pairs):
        if i==j:u[6*i+j,col]=1
        else:u[6*i+j,col]=u[6*j+i,col]=1/sqrt(2)
    np.testing.assert_allclose(u.conj().T@raw.reshape(36,36)@u,direct,atol=6e-15)


def test_complete_SU3_color_projection_and_all_physical_helicity_assignments():
    f=structure_constants();u=color_basis("qqqgg")
    for hs in product((-1,1),repeat=4):
        eps=[-np.array([h,1j])/sqrt(2) for h in hs]
        eps[:2]=[a.conj() for a in eps[:2]]
        tensor=distinguished_quartic_tensor(f,eps)
        acted=np.einsum("abcd,ijkcdn->ijkabn",tensor,u.reshape(3,3,3,8,8,6)).reshape(u.shape)
        np.testing.assert_allclose(quartic_spin_color(*hs),u.conj().T@acted,atol=2e-14)
        if hs[0]+hs[1]!=hs[2]+hs[3]:
            np.testing.assert_allclose(quartic_spin_color(*hs),0.,atol=2e-15)


def test_field_cell_source_and_normalized_two_boson_prefactor():
    for length in (.4,3.,20.):
        total=9;k=np.array([2,4,4,2])
        pp=pi*total/(2*length);p=pi*k/(2*length)
        direct=2*pp/(2*length)*.25/np.sqrt(np.prod(2*p))*.5
        assert direct==pytest.approx(total/(16*pi*np.sqrt(np.prod(k))))


@pytest.mark.parametrize("cutoff,target",[(0,"PROTON"),(1,"NEUTRON")])
def test_actual_A4_QQ_Hermitian_statistics_spin_and_nonzero_same_longitudinal_modes(cutoff,target):
    space=IntermediateSpace("qqqgg",9,cutoff,target,1)
    w=four_gluon_QQ_per_gs2(space,b_GeV=.4)
    assert sparse.linalg.norm(w)>1e-3
    assert sparse.linalg.norm(w-w.getH())<5e-14
    assert sparse.linalg.norm(four_gluon_QQ_per_gs2(space,b_GeV=.8)-4*w)<5e-14
    same=np.flatnonzero(space.coordinates[:,0]==0)
    assert sparse.linalg.norm(w[same][:,same])>1e-4
    rng=np.random.default_rng(448)
    a=rng.normal(size=(space.dimension,2))+1j*rng.normal(size=(space.dimension,2))
    for p in ((1,0,2,3,4),(0,1,2,4,3)):
        np.testing.assert_allclose(w@space.permute(a,p),space.permute(w@a,p),atol=5e-14)


def test_A4_remains_at_minimum_K_where_Q0_scattering_is_empty():
    w=four_gluon_QQ_per_gs2(IntermediateSpace("qqqgg",7,0,"PROTON",1),b_GeV=.4)
    assert sparse.linalg.norm(w)>1e-3
    with pytest.raises(ValueError):
        four_gluon_QQ_per_gs2(IntermediateSpace("qqqq-qbar",9,0,"PROTON",1),b_GeV=.4)


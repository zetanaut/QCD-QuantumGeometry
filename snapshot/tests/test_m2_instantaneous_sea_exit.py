from itertools import permutations,product
from math import factorial,pi,sqrt
import numpy as np
import pytest
from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_instantaneous_sea_exit import (
    sea_pair_spin_factor,sea_pair_color,sea_pair_mass_kernel,build_instantaneous_sea_source,
)


@pytest.mark.parametrize("hq,ha",list(product((-1,1),repeat=2)))
def test_pair_current_and_complete_finite_cell_normalization(hq,ha):
    inc,out=(5,1,1),(1,1,1,1,3)
    for pp,mass in product((.7,3.2),(.16,.72)):
        cell=pi*7/pp
        p0,pf,pq,pa=pp*np.array([5,1,1,3])/7
        u0=lf_spinor(p0,.2,-.31,mass,1);uf=lf_spinor(pf,-.17,.19,mass,1)
        uq=lf_spinor(pq,.32,.21,mass,hq);va=antiquark_spinor(pa,-.12,.04,mass,ha)
        j1=uf.conj()@GAMMA[0]@GAMMA_PLUS@u0
        j2=uq.conj()@GAMMA[0]@GAMMA_PLUS@va
        spin=j2/sqrt(4*pq*pa)
        assert spin==pytest.approx(sea_pair_spin_factor(hq,ha),abs=4e-15)
        field_norm=sqrt((2*p0)*(2*pf)*(2*pq)*(2*pa))
        expected=2*pp/cell*j1*j2/field_norm/(pq+pa)**2*sea_pair_color(0)
        actual=sea_pair_mass_kernel(inc,out,0,(1,1,hq,ha))
        np.testing.assert_allclose(actual,expected,atol=8e-16)
        np.testing.assert_allclose(sea_pair_mass_kernel(inc,out,0,(-1,1,hq,ha)),0.,atol=0.)


def test_four_quark_creation_factor_and_normal_order_sign_against_independent_CAR():
    # Five quark modes plus one antiquark; exact occupation-bit CAR, no color
    # or symmetry projector reused from the physics implementation.
    n=5;anti=n;dim=2**(n+1);rng=np.random.default_rng(934)
    m=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    a=rng.normal(size=n)+1j*rng.normal(size=n)
    initial=np.zeros(dim,complex);initial[7]=1
    def ladder(values,k,creation):
        result=np.zeros_like(values)
        for label,value in enumerate(values):
            occupied=(label>>k)&1
            if bool(occupied)==creation:continue
            sign=(-1)**((label&((1<<k)-1)).bit_count())
            result[label^(1<<k)]+=sign*value
        return result
    bstate=sum(m[r,s]*ladder(ladder(initial,s,False),r,True) for r,s in product(range(n),repeat=2))
    # Normal ordering removes the one-contraction term from B P; the two
    # normal-ordered current orders give the same quartic P B operator.
    expected=sum(a[q]*ladder(ladder(bstate,anti,True),q,True) for q in range(n))
    tensor=np.zeros((n,n,n),complex)
    def sign(p):return (-1)**sum(p[i]>p[j] for i in range(len(p)) for j in range(i+1,len(p)))
    for p in permutations(range(3)):tensor[p]=sign(p)/sqrt(6)
    valence=sum(np.moveaxis(np.tensordot(m,tensor,axes=(1,i)),0,i) for i in range(3))
    appended=valence[:,:,:,None]*a[None,None,None,:]
    projected=sum(sign(p)*appended.transpose(p) for p in permutations(range(4)))/factorial(4)
    image=2*projected
    for omitted in range(n):
        labels=tuple(i for i in range(n) if i!=omitted)
        amplitude=sum(sign(p)*image[tuple(labels[i] for i in p)] for p in permutations(range(4)))/sqrt(24)
        index=sum(1<<i for i in labels)+(1<<anti)
        assert amplitude==pytest.approx(expected[index],abs=2e-14)


@pytest.mark.parametrize("target,jz,cutoff",[("PROTON",1,0),("NEUTRON",-1,0),("PROTON",1,1)])
def test_actual_sea_source_four_quark_statistics_and_nonzero_free_return(target,jz,cutoff):
    p=build_coupled_pilot(total_k2=9,cutoff=cutoff,target=target,jz_twice=jz,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    s=build_instantaneous_sea_source(p,out_cutoff=1);v=s.matrix_per_gs2
    assert np.linalg.norm(v)>1e-4
    assert s.space.ncolor==3
    assert np.array_equal(s.retained_columns,np.arange(p.bases[0].shape[1]))
    for perm in ((1,0,2,3,4),(3,1,2,0,4)):
        np.testing.assert_allclose(s.space.permute(v,perm),-v,atol=4e-14)
    energies=s.space.free_energies(.4,.35,0.)
    sigma=v.conj().T@(v/(1.4-energies[:,None]))
    assert np.linalg.eigvalsh(sigma)[-1]<2e-15
    assert np.linalg.norm(sigma)>1e-6
    if cutoff==0:
        low=build_instantaneous_sea_source(p,out_cutoff=0)
        rows=s.space.coordinates[:,1]==0
        np.testing.assert_allclose(v[rows],low.matrix_per_gs2,atol=2e-16)
        np.testing.assert_allclose(v[~rows],0.,atol=2e-16)


@pytest.mark.parametrize("args",[(True,-1),(0,1),(1,np.nan)])
def test_invalid_physical_pair_helicities_rejected(args):
    with pytest.raises(ValueError):sea_pair_spin_factor(*args)

from itertools import product
from math import sqrt
import numpy as np
import pytest
from scipy import linalg,sparse

from deuteron_wigner.pilot.color import structure_constants
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_excluded_annihilation import annihilation_per_gs2
from deuteron_wigner.quantum.m2_excluded_fock_matching import intermediate_density_action
from deuteron_wigner.quantum.m2_excluded_pair_current import (
    build_pair_current_factorization,adjoint_pair_current_factor,CoupledPairCurrentInsertion,
)


def build(cutoff=0):
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    ref=build_order_resolved_reference(p,out_cutoff=cutoff)
    f=build_pair_current_factorization(ref)
    zeros={name:(space,sparse.csr_matrix((space.dimension,space.dimension)))
           for name,space in f.spaces.items()}
    return CoupledPairCurrentInsertion(ref,factorization=f,sector_operators=zeros,
                                       label="canonical pair-current square only")


@pytest.fixture(scope="module")
def insertion():
    return build()


def test_pair_current_normalization_and_phase_from_two_full_field_contractions():
    f=structure_constants()
    for k1,k2,h1,h2 in product((2,4),(2,6),(-1,1),(-1,1)):
        e1=(-np.array([h1,1j])/sqrt(2)).conj()
        e2=(-np.array([h2,1j])/sqrt(2)).conj()
        # The derivative acts on either created gluon, reversing its f slots.
        direct=(1j*k2*f+1j*k1*f.transpose(0,2,1))*np.dot(e1,e2)/(2*sqrt(k1*k2)*sqrt(2))
        np.testing.assert_allclose(direct,adjoint_pair_current_factor(k1,k2,h1,h2)*f,atol=2e-15)
    assert adjoint_pair_current_factor(2,2,1,-1)==0
    assert adjoint_pair_current_factor(2,4,1,1)==0


def test_boson_pair_current_Bose_factor_and_both_orders_against_independent_CCR():
    # P† = (1/2) M_ij a_i† a_j† for symmetric M. The normalized
    # ordered two-boson wave is M/sqrt(2), not M or sqrt(2) M.
    labels=list(product(range(5),repeat=3));lookup={s:i for i,s in enumerate(labels)}
    def annihilator(k):
        a=np.zeros((len(labels),len(labels)),complex)
        for col,s in enumerate(labels):
            if s[k]:
                out=list(s);out[k]-=1;a[lookup[tuple(out)],col]=sqrt(s[k])
        return a
    a=[annihilator(k) for k in range(3)]
    m=np.array([[.3,.2j,.11],[.2j,-.4,.07j],[.11,.07j,.2]],complex)
    create=sum(m[i,j]*a[i].conj().T@a[j].conj().T/2 for i,j in product(range(3),repeat=2))
    one=m@m.conj().T
    wick=sum(one[i,j]*a[i].conj().T@a[j] for i,j in product(range(3),repeat=2))
    wick+=.5*np.linalg.norm(m)**2*np.eye(len(labels))
    normal=(create@create.conj().T+create.conj().T@create-wick)/2
    two=[s for s in labels if sum(s)==2]
    indices=[lookup[s] for s in two]
    wave=create[:,lookup[(0,0,0)]][indices]
    expected=[]
    for s in two:
        slots=[i for i,n in enumerate(s) for _ in range(n)]
        i,j=slots
        expected.append(m[i,j]/sqrt(2) if i==j else m[i,j])
    np.testing.assert_allclose(wave,expected,atol=2e-15)
    np.testing.assert_allclose(normal[np.ix_(indices,indices)],np.outer(wave,wave.conj()),atol=3e-15)


def test_auxiliary_is_current_not_physical_gluon_and_weight_preserves_its_range(insertion):
    f=insertion.factorization;a=f.auxiliary
    np.testing.assert_allclose((a.basis.getH()@a.basis).toarray(),np.eye(a.basis.shape[1]),atol=4e-15)
    assert a.basis.shape[1]>0
    assert np.linalg.eigvalsh(a.weight)[0]>0
    for name,b in f.matrices.items():
        s=f.spaces[name]
        np.testing.assert_allclose(s.statistics_project(b),b,atol=3e-14)
        np.testing.assert_allclose(s.permute(b,(1,0,2,3,4)),-b,atol=3e-14)
    assert np.linalg.norm(f.matrices["qqqgg"])>1e-4
    assert np.linalg.norm(f.matrices["qqqq-qbar"])>1e-4


@pytest.mark.parametrize("cutoff",[0,1])
def test_sea_factorization_matches_independent_all_four_annihilator_action(cutoff):
    obj=build(cutoff);f=obj.factorization;s=f.spaces["qqqq-qbar"]
    w=annihilation_per_gs2(s,b_GeV=.4)
    # Both coupling-order images have I=1/2 and full four-quark statistics.
    for power in (1,2):
        sector=next(x for x in obj.reference.sectors if x.source.space is s)
        v=sector.lift(sector.coupling_by_power[power])
        b=f.matrices[s.sector]
        np.testing.assert_allclose(w@v,b@f.auxiliary.weight@b.conj().T@v,atol=3e-14)


def test_coupled_square_positive_and_nonzero_interference_not_two_incoherent_channels(insertion):
    obj=insertion;f=obj.factorization
    v={name:obj.wave_coefficients(name,1.4)[0] for name in obj.sources}
    acted=f.apply(v)
    full=sum(a.conj().T@acted[name] for name,a in v.items())
    incoherent=sum(a.conj().T@f.matrices[name]@f.auxiliary.weight@f.matrices[name].conj().T@a
                   for name,a in v.items())
    np.testing.assert_allclose(full,full.conj().T,atol=3e-15)
    assert np.linalg.eigvalsh(full)[0]>-3e-15
    assert np.linalg.norm(full-incoherent)>1e-7
    np.testing.assert_allclose(full,obj.hamiltonian_coefficient(1.4),atol=3e-15)


@pytest.mark.parametrize("charge,total",[("baryon",1),("electromagnetic",1),("up_number",2),("down_number",1)])
def test_coupled_pair_current_two_energies_reversal_and_norm(insertion,charge,total):
    obj=insertion
    a=obj.current_coefficient(1.46,1.31,(.12,.16),charge=charge)
    r=obj.current_coefficient(1.31,1.46,(-.12,-.16),charge=charge)
    np.testing.assert_allclose(a,r.conj().T,atol=3e-15)
    np.testing.assert_allclose(obj.current_coefficient(1.4,1.4,(0.,0.),charge=charge),
                               total*obj.norm_coefficient(1.4),atol=3e-15)


def test_independent_low_rank_interacting_Q_resolvent_and_two_energy_current(insertion):
    obj=insertion;f=obj.factorization;names=list(f.spaces)
    b=np.vstack([f.matrices[n] for n in names]);d=f.auxiliary.weight
    energies=np.concatenate([obj.energies[n] for n in names])
    source=np.vstack([obj.sources[n] for n in names])
    cuts=np.cumsum([0]+[f.spaces[n].dimension for n in names])
    def resolvent(energy,lam):
        rb=b/(energy-energies[:,None]);rv=source/(energy-energies[:,None])
        return rv+lam*rb@linalg.solve(linalg.inv(d)-lam*b.conj().T@rb,b.conj().T@rv)
    step=1e-3;hs=[];currents=[]
    for lam in (-step,step):
        a=resolvent(1.4,lam);hs.append(source.conj().T@a)
        af,ai=resolvent(1.46,lam),resolvent(1.31,lam)
        rho=0.
        for k,name in enumerate(names):
            sl=slice(cuts[k],cuts[k+1])
            rho+=af[sl].conj().T@intermediate_density_action(f.spaces[name],ai[sl],(.12,.16),
                                                             b_GeV=.4,charge="electromagnetic")
        currents.append(rho)
    np.testing.assert_allclose((hs[1]-hs[0])/(2*step),obj.hamiltonian_coefficient(1.4),atol=3e-11)
    np.testing.assert_allclose((currents[1]-currents[0])/(2*step),
        obj.current_coefficient(1.46,1.31,(.12,.16),charge="electromagnetic"),atol=3e-11)
    numerical=(obj.hamiltonian_coefficient(1.4+1e-5)-obj.hamiltonian_coefficient(1.4-1e-5))/2e-5
    np.testing.assert_allclose(numerical,-obj.norm_coefficient(1.4),atol=3e-12)

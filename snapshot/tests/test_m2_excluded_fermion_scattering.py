"""Independent five-body contact, CAR/color, symmetry and power-insertion checks."""
from itertools import combinations, product
from math import pi, sqrt

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import roots_hermite

from deuteron_wigner.bridge.modes.core import GAMMA, GAMMA_PLUS, ho_coordinate
from deuteron_wigner.microscopic.h0.color import representation_generators
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_shell
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import frame
from deuteron_wigner.quantum.m2_excluded_fock_matching import intermediate_density_action
from deuteron_wigner.quantum.m2_excluded_fermion_scattering import (
    fermion_pair_color, fermion_contact_block, fermion_scattering_per_gs2,
)
from deuteron_wigner.quantum.m2_excluded_interaction_insertion import ExcludedInteractionInsertion
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference


def direct_contact(sector, incoming, outgoing, pair, b, *, out_sector=None):
    # Integrate the physical r_i=r_j surface, not the adapted Jacobi contact.
    d=4; i,j=pair
    constraint=np.zeros(d)
    if i<d: constraint[i]+=1
    if j<d: constraint[j]-=1
    pivot=np.flatnonzero(constraint)[0]
    free=[k for k in range(d) if k!=pivot]
    z=np.eye(d)[:,free]; z[pivot]=-constraint[free]/constraint[pivot]
    frames=[frame(sector,incoming),frame(sector if out_sector is None else out_sector,outgoing)]
    maps=[f.inverse[:-1,:-1].T@z for f in frames]
    precision=b*b*sum(a.T@a for a in maps)
    transform=sqrt(2)*np.linalg.inv(np.linalg.cholesky(precision).T)
    t,w=roots_hermite(3)
    labels=np.array(list(product(range(3),repeat=6)))
    h=t[labels].reshape(-1,3,2)
    v=np.einsum("ij,njk->nik",transform,h)
    points=[np.einsum("ij,njk->nik",a,v) for a in maps]
    weight=np.prod(w[labels],axis=1)*8/np.linalg.det(precision)
    weight*=np.exp(np.sum(h*h,axis=(1,2)))
    modes=[]
    for f,r in zip(frames,points):
        shell=[]
        for s in oscillator_shell(4,1):
            value=np.ones(len(r),complex)*sqrt(f.transverse_measure_factor)
            for k in range(d):
                n=min(s[k],s[d+k]); m=s[k]-s[d+k]
                value*=(-1)**n*ho_coordinate(n,m,r[:,k,0],r[:,k,1],b)
            shell.append(value)
        modes.append(np.array(shell))
    return (modes[1].conj()*weight)@modes[0].T


@pytest.mark.parametrize("sector,incoming,outgoing,pair",[
    ("qqqgg",(3,1,1,2,2),(1,3,1,2,2),(0,1)),
    ("qqqq-qbar",(3,1,1,1,3),(1,1,1,1,5),(0,4)),
    ("qqqq-qbar",(1,1,1,3,3),(1,1,1,5,1),(3,4)),
])
def test_contact_against_independent_six_dimensional_coordinate_quadrature(sector,incoming,outgoing,pair):
    actual=fermion_contact_block(sector,incoming,outgoing,pair,.4,1)
    np.testing.assert_allclose(actual,direct_contact(sector,incoming,outgoing,pair,.4),atol=4e-15)
    np.testing.assert_allclose(actual.T,fermion_contact_block(sector,outgoing,incoming,pair,.4,1),atol=4e-15)


def test_full_spinor_field_norm_and_antiquark_CAR_charge_sign():
    # Both good bilinears have positive diagonal spin norm. The anti charge
    # minus sign comes from normal ordering d d†, not a negative vbar gamma+ v.
    for fn,hout,hin in product((lf_spinor,antiquark_spinor),(-1,1),(-1,1)):
        p1,p2=.7,1.3
        a=fn(p1,.21,-.12,.35,hout); b=fn(p2,-.11,.38,.35,hin)
        j=a.conj()@GAMMA[0]@GAMMA_PLUS@b/sqrt(4*p1*p2)
        assert j==pytest.approx(float(hout==hin),abs=3e-15)
    n=6; dim=2**n
    def annihilator(k):
        a=np.zeros((dim,dim),complex)
        for col in range(dim):
            if (col>>k)&1:
                a[col^(1<<k),col]=(-1)**((col&((1<<k)-1)).bit_count())
        return a
    ops=[annihilator(i) for i in range(n)]
    # qqbar ordered normalized ket b_i† d_j† |0>.
    vacuum=np.eye(dim)[:,0]
    u=np.column_stack([ops[i].conj().T@ops[3+j].conj().T@vacuum for i,j in product(range(3),repeat=2)])
    actual=np.zeros((dim,dim),complex)
    expected=np.zeros((9,9),complex)
    for t in representation_generators("3"):
        jq=sum(t[i,j]*ops[i].conj().T@ops[j] for i,j in product(range(3),repeat=2))
        # Raw antiquark current d_i T_ij d_j†, with its trace vacuum removed.
        ja=sum(t[i,j]*ops[3+i]@ops[3+j].conj().T for i,j in product(range(3),repeat=2))
        actual+=jq@ja
        expected+=np.kron(t,-t.T)
    np.testing.assert_allclose(u.conj().T@actual@u,expected,atol=2e-15)
    total=sum(fermion_pair_color("qqqq-qbar",i,j) for i,j in combinations(range(5),2))
    np.testing.assert_allclose(total,-10/3*np.eye(3),atol=4e-15)


@pytest.mark.parametrize("sector,cutoff,target",[
    ("qqqgg",0,"PROTON"),("qqqgg",1,"NEUTRON"),
    ("qqqq-qbar",0,"NEUTRON"),("qqqq-qbar",1,"PROTON"),
])
def test_sparse_QQ_Hermitian_charge_statistics_Q0_and_width(sector,cutoff,target):
    space=IntermediateSpace(sector,9,cutoff,target,1)
    w=fermion_scattering_per_gs2(space,b_GeV=.4)
    assert sparse.linalg.norm(w)>1e-3
    assert sparse.linalg.norm(w-w.getH())<4e-14
    assert sparse.linalg.norm(fermion_scattering_per_gs2(space,b_GeV=.8)-4*w)<4e-14
    # No zero-longitudinal-transfer block, including transverse changes.
    for li in range(len(space.longitudinal)):
        rows=np.flatnonzero(space.coordinates[:,0]==li)
        assert w[rows][:,rows].nnz==0
    rng=np.random.default_rng(448)
    a=rng.normal(size=(space.dimension,2))+1j*rng.normal(size=(space.dimension,2))
    for permutation in ((1,0,2,3,4), (0,1,2,4,3) if sector=="qqqgg" else (3,1,2,0,4)):
        np.testing.assert_allclose(w@space.permute(a,permutation),space.permute(w@a,permutation),atol=3e-14)
    q=1 if target=="PROTON" else 0
    np.testing.assert_allclose(intermediate_density_action(space,w@a,(0.,0.),b_GeV=.4,charge="electromagnetic"),
                               q*(w@a),atol=3e-14)


@pytest.fixture(scope="module")
def insertion():
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    ref=build_order_resolved_reference(p,out_cutoff=1)
    operators={s.source.space.sector:(s.source.space,fermion_scattering_per_gs2(s.source.space,b_GeV=.4))
               for s in ref.sectors}
    return ExcludedInteractionInsertion(ref,sector_operators=operators,label="canonical fermion t-channel only")


def test_actual_insertion_nonzero_H_norm_energy_derivative_and_new_wave_support(insertion):
    obj=insertion;e=1.4
    h=obj.hamiltonian_coefficient(e);n=obj.norm_coefficient(e)
    assert np.linalg.norm(h)>1e-6
    np.testing.assert_allclose(h,h.conj().T,atol=4e-15)
    derivative=(obj.hamiltonian_coefficient(e+1e-5)-obj.hamiltonian_coefficient(e-1e-5))/2e-5
    np.testing.assert_allclose(derivative,-n,atol=2e-12)
    assert max(obj.free_source_leakage(e).values())>1e-6
    for s in obj.reference.sectors:
        name=s.source.space.sector
        a1,a3=obj.wave_coefficients(name,e)
        # H can use projected W; current cannot in general.
        projected=s.lift(s.project(a3))
        np.testing.assert_allclose(obj.sources[name].conj().T@(a3-projected),0.,atol=3e-15)


@pytest.mark.parametrize("charge,total",[("baryon",1),("electromagnetic",1),("up_number",2),("down_number",1)])
def test_inserted_current_both_endpoints_reversal_and_norm_charge(insertion,charge,total):
    obj=insertion
    f=obj.current_coefficient(1.46,1.31,(.12,.16),charge=charge)
    r=obj.current_coefficient(1.31,1.46,(-.12,-.16),charge=charge)
    np.testing.assert_allclose(f,r.conj().T,atol=4e-15)
    np.testing.assert_allclose(obj.current_coefficient(1.4,1.4,(0.,0.),charge=charge),
                               total*obj.norm_coefficient(1.4),atol=4e-15)


def test_single_insertion_against_independent_interacting_raw_Q_resolvent(insertion):
    obj=insertion;e=1.4;step=1e-3;charge="electromagnetic";delta=(.12,.16)
    h=[];current=[]
    # Finite-difference in an independent QQ multiplier lambda, NOT in g:
    # this isolates the first W insertion without mixing cubic/quartic orders.
    for lam in (-step,step):
        hv=np.zeros_like(obj.reference.pilot.matrix);cv=hv.copy()
        for s in obj.reference.sectors:
            name=s.source.space.sector
            w=obj.operators[name];v=obj.sources[name]
            r=splu((sparse.diags(e-obj.energies[name])-lam*w).tocsc())
            a=r.solve(v)
            hv+=v.conj().T@a
            cv+=a.conj().T@intermediate_density_action(s.source.space,a,delta,b_GeV=.4,charge=charge)
        h.append(hv);current.append(cv)
    np.testing.assert_allclose((h[1]-h[0])/(2*step),obj.hamiltonian_coefficient(e),atol=2e-11)
    np.testing.assert_allclose((current[1]-current[0])/(2*step),
                               obj.current_coefficient(e,e,delta,charge=charge),atol=2e-11)


def test_old_free_source_current_projection_loses_same_order_finite_transfer_terms(insertion):
    obj=insertion;e=1.4;delta=(.12,.16)
    projected=np.zeros_like(obj.reference.pilot.matrix)
    for s in obj.reference.sectors:
        name=s.source.space.sector
        a1,a3=obj.wave_coefficients(name,e)
        a3=s.lift(s.project(a3))
        rho1=intermediate_density_action(s.source.space,a1,delta,b_GeV=.4,charge="electromagnetic")
        rho3=intermediate_density_action(s.source.space,a3,delta,b_GeV=.4,charge="electromagnetic")
        projected+=a3.conj().T@rho1+a1.conj().T@rho3
    actual=obj.current_coefficient(e,e,delta,charge="electromagnetic")
    assert np.linalg.norm(actual-projected)>1e-8


def test_invalid_insertion_basis_Hermiticity_and_energy_rejected(insertion):
    obj=insertion
    with pytest.raises(ValueError,match="each"):
        ExcludedInteractionInsertion(obj.reference,sector_operators={},label="missing")
    spaces={s.source.space.sector:s.source.space for s in obj.reference.sectors}
    wrong={name:(spaces[name],w) for name,w in obj.operators.items()}
    name=next(iter(wrong));wrong[name]=(spaces[name],wrong[name][1]+1j*sparse.eye(len(obj.energies[name])))
    with pytest.raises(ValueError,match="Hermitian"):
        ExcludedInteractionInsertion(obj.reference,sector_operators=wrong,label="wrong")
    for e in (np.nan,obj.energies[name][0]):
        with pytest.raises(ValueError,match="off-pole"):obj.wave_coefficients(name,e)
    # A same-dimension independent space is not an implicit basis map.
    wrong={name:(spaces[name],w) for name,w in obj.operators.items()}
    s=spaces[name]
    other=IntermediateSpace(s.sector,s.total_k2,s.cutoff,s.target,s.jz_twice)
    wrong[name]=(other,obj.operators[name])
    with pytest.raises(ValueError,match="exact raw"):
        ExcludedInteractionInsertion(obj.reference,sector_operators=wrong,label="unmapped")

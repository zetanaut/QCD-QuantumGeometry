"""Independent color Fierz, occupation CAR and five-body annihilation checks."""
from itertools import product
from math import sqrt, pi
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import eigsh

from deuteron_wigner.microscopic.h0.color import representation_generators
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import color_basis
from deuteron_wigner.quantum.m2_excluded_annihilation import annihilation_color, annihilation_per_gs2
from deuteron_wigner.quantum.m2_instantaneous_sea_exit import sea_pair_spin_factor
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor
from deuteron_wigner.bridge.modes.core import GAMMA, GAMMA_PLUS


def test_pair_color_from_independent_octet_projector_Fierz_and_positivity():
    singlet=np.eye(3).ravel()/sqrt(3)
    octet=(np.eye(9)-np.outer(singlet,singlet))/2
    t=np.asarray(representation_generators("3")).reshape(8,9)
    np.testing.assert_allclose(t.T@t.conj(),octet,atol=2e-16)
    u=color_basis("qqqq-qbar").reshape((3,)*6)
    for slot in range(4):
        rest=tuple(i for i in range(4) if i!=slot)
        order=rest+(slot,4,5)
        adapted=u.transpose(order).reshape(27,9,3)
        direct=np.einsum("ran,ab,rbm->nm",adapted.conj(),octet,adapted)
        np.testing.assert_allclose(annihilation_color(slot),direct,atol=2e-15)
        assert np.linalg.eigvalsh(direct)[0]>-2e-15


def test_number_preserving_pair_annihilation_no_extra_factor_against_independent_CAR():
    nq,na=3,2;dim=2**(nq+na)
    rng=np.random.default_rng(477)
    c=rng.normal(size=(nq,na))+1j*rng.normal(size=(nq,na))
    psi=rng.normal(size=(nq,nq,na))+1j*rng.normal(size=(nq,nq,na))
    psi=(psi-psi.transpose(1,0,2))/2;psi/=np.linalg.norm(psi)
    def ladder(k):
        a=np.zeros((dim,dim),complex)
        for col in range(dim):
            if (col>>k)&1:a[col^(1<<k),col]=(-1)**((col&((1<<k)-1)).bit_count())
        return a
    a=[ladder(k) for k in range(nq+na)]
    vac=np.eye(dim)[:,0]
    def embed(tensor):
        return sum(tensor[i,j,b]/sqrt(2)*a[i].conj().T@a[j].conj().T@a[nq+b].conj().T@vac
                   for i,j,b in product(range(nq),range(nq),range(na)))
    create=sum(c[i,b]*a[i].conj().T@a[nq+b].conj().T for i,b in product(range(nq),range(na)))
    expected=create@create.conj().T@embed(psi)
    first=np.einsum("ia,kb,kjb->ija",c,c.conj(),psi)
    second=np.einsum("ja,kb,ikb->ija",c,c.conj(),psi)
    np.testing.assert_allclose(embed(first+second),expected,atol=4e-15)
    assert np.linalg.norm(embed(psi))==pytest.approx(1.,abs=2e-15)


@pytest.mark.parametrize("mass,pplus",[(.16,.7),(.72,3.2)])
def test_annihilation_total_pair_transfer_full_field_normalization(mass,pplus):
    total_k2=9;inc=(1,1,1,1,5);out=(3,1,1,1,3)
    ell=pi*total_k2/pplus
    for hf,hi in product((-1,1),repeat=2):
        po,ao,pi_,ai=pplus*np.array([out[0],out[4],inc[0],inc[4]])/total_k2
        uf=lf_spinor(po,.2,-.1,mass,hf);vf=antiquark_spinor(ao,-.1,.3,mass,-hf)
        ui=lf_spinor(pi_,.12,.14,mass,hi);vi=antiquark_spinor(ai,-.21,.07,mass,-hi)
        jo=uf.conj()@GAMMA[0]@GAMMA_PLUS@vf
        ji=ui.conj()@GAMMA[0]@GAMMA_PLUS@vi
        coefficient=2*pplus/ell*jo*ji.conjugate()/sqrt((2*po)*(2*ao)*(2*pi_)*(2*ai))/(pi_+ai)**2
        assert coefficient==pytest.approx(2*total_k2/(pi*6**2)*
            sea_pair_spin_factor(hf,-hf)*sea_pair_spin_factor(hi,-hi).conjugate(),abs=2e-15)


@pytest.mark.parametrize("cutoff,target",[(0,"PROTON"),(1,"NEUTRON")])
def test_actual_QQ_annihilation_Hermitian_nonzero_same_tuple_and_quark_covariance(cutoff,target):
    space=IntermediateSpace("qqqq-qbar",9,cutoff,target,1)
    w=annihilation_per_gs2(space,b_GeV=.4)
    assert sparse.linalg.norm(w)>1e-3
    assert sparse.linalg.norm(w-w.getH())<3e-14
    assert sparse.linalg.norm(annihilation_per_gs2(space,b_GeV=.8)-4*w)<3e-14
    # Pair sum remains positive even with zero momentum rearrangement.
    li=np.flatnonzero(space.coordinates[:,0]==0)
    assert sparse.linalg.norm(w[li][:,li])>1e-4
    rng=np.random.default_rng(997)
    a=rng.normal(size=(space.dimension,2))+1j*rng.normal(size=(space.dimension,2))
    for p in ((1,0,2,3,4),(3,1,2,0,4)):
        np.testing.assert_allclose(w@space.permute(a,p),space.permute(w@a,p),atol=3e-14)
    # Each annihilation component is a regulated positive pair-current square.
    assert np.vdot(a[:,0],w@a[:,0]).real>0
    if cutoff==0:
        assert eigsh(w,k=1,which="SA",return_eigenvectors=False,tol=1e-10)[0]>-3e-14


def test_invalid_annihilating_slot_or_sector_rejected():
    for value in (True,4,-1):
        with pytest.raises((ValueError,TypeError)):annihilation_color(value)
    with pytest.raises(ValueError):
        annihilation_per_gs2(IntermediateSpace("qqqgg",9,0,"PROTON",1),b_GeV=.4)


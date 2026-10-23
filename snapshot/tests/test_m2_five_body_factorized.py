"""Factorized contacts against assembled controls, without a source-range shortcut."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse

from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_canonical_qq import CanonicalQQSourceAction
from deuteron_wigner.quantum.m2_five_body_factorized import FiveBodyContactAction,apply_terms
from deuteron_wigner.quantum.m2_pair_current_factorized import (
    AuxiliaryCurrentSpace,FactorizedPairCurrentAction,
)
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import S3,longitudinal_orbital_action
from deuteron_wigner.quantum.m2_c117_pair_current import color_permutation_matrix
from deuteron_wigner.quantum.m2_spin_flavor_s3_completion import spin_flavor_permutation_representation


def spaces(cutoff,target="PROTON",jz=1):
    return {n:IntermediateSpace(n,9,cutoff,target,jz) for n in ("qqqgg","qqqq-qbar")}


def columns(ss,seed=817):
    rng=np.random.default_rng(seed)
    return {n:(rng.normal(size=(s.dimension,2))+1j*rng.normal(size=(s.dimension,2)))
              /np.sqrt(s.dimension) for n,s in ss.items()}


def reference(ss):
    s=ss["qqqgg"]
    return SimpleNamespace(pilot=SimpleNamespace(total_k2=s.total_k2,width_GeV=.4,
        target=s.target,jz_twice=s.jz_twice),
        sectors=tuple(SimpleNamespace(source=SimpleNamespace(space=s)) for s in ss.values()))


@pytest.mark.parametrize("cutoff",[0,1])
def test_each_contact_conversion_and_pair_square_on_arbitrary_raw_columns(cutoff):
    ss=spaces(cutoff);v=columns(ss)
    old=CanonicalQQSourceAction(reference(ss))
    contacts=FiveBodyContactAction(ss,b_GeV=.4)
    pair=FactorizedPairCurrentAction(ss,b_GeV=.4)
    for label,(name,terms) in contacts.components.items():
        old_name,matrix=old.block_components[label]
        assert old_name==name
        np.testing.assert_allclose(apply_terms(terms,v[name],ss[name].dimension),
                                   matrix@v[name],atol=3e-15,rtol=3e-13)
    # Arbitrary raw columns, not only an already projected source image.
    before=old.apply(v);after=contacts.apply(v);square=pair.apply(v)
    square_old=old.pair_current.apply(v)
    for name in ss:
        np.testing.assert_allclose(square[name],square_old[name],atol=3e-15,rtol=3e-13)
        np.testing.assert_allclose(after[name]+square[name],before[name],atol=3e-15,rtol=3e-13)
    aux=pair.auxiliary;u=old.pair_current.auxiliary.basis[aux.rows]
    rng=np.random.default_rng(914)
    a=rng.normal(size=(aux.dimension,2))+1j*rng.normal(size=(aux.dimension,2))
    np.testing.assert_allclose(aux.project(a),u@(u.getH()@a),atol=5e-14)


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_second_shell_auxiliary_against_independent_full_tensor_projector(target,jz):
    a=AuxiliaryCurrentSpace(9,2,target,jz)
    rng=np.random.default_rng(431);v=rng.normal(size=(a.dimension,2))+1j*rng.normal(size=(a.dimension,2))
    expected=np.zeros_like(v,dtype=complex)
    for p in S3:
        sign=(-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
        lc=sparse.kron(longitudinal_orbital_action("qqqg",9,2,p),
                       color_permutation_matrix("qqqg",p),format="csr")
        full=sparse.kron(lc,spin_flavor_permutation_representation(p)@a.isospin_projector,format="csr")
        expected+=sign/6*(full[a.rows][:,a.rows]@v)
    projected=a.project(v)
    np.testing.assert_allclose(projected,expected,atol=5e-14)
    np.testing.assert_allclose(a.project(projected),projected,atol=5e-14)
    np.testing.assert_allclose(v.conj().T@projected,projected.conj().T@v,atol=5e-12)
    np.testing.assert_allclose(a.project(a.weight[:,None]*v),a.weight[:,None]*projected,atol=5e-14)


def test_second_shell_full_contacts_and_pair_are_hermitian_and_preserve_statistics():
    ss=spaces(2);v={n:s.statistics_project(a) for (n,s),a in zip(ss.items(),columns(ss).values())}
    contacts=FiveBodyContactAction(ss,b_GeV=.4);pair=FactorizedPairCurrentAction(ss,b_GeV=.4)
    c,p=contacts.apply(v),pair.apply(v)
    for action in (c,p):
        form=sum(v[n].conj().T@action[n] for n in ss)
        np.testing.assert_allclose(form,form.conj().T,atol=3e-15)
        for n,s in ss.items():
            np.testing.assert_allclose(s.statistics_project(action[n]),action[n],atol=3e-15)
    positive=sum(v[n].conj().T@p[n] for n in ss)
    assert np.linalg.eigvalsh(positive).min()>-3e-15
    assert np.linalg.norm(positive)>1e-7
    assert np.linalg.norm(sum(v[n].conj().T@c[n] for n in ss))>1e-7


def test_factorized_pair_and_contact_reject_mismatched_spaces_and_columns():
    ss=spaces(0)
    for action in (FiveBodyContactAction,FactorizedPairCurrentAction):
        with pytest.raises(ValueError):action({"qqqgg":ss["qqqgg"]},b_GeV=.4)
        obj=action(ss,b_GeV=.4)
        with pytest.raises(ValueError):obj.apply({"qqqgg":np.zeros((ss["qqqgg"].dimension,1))})
    with pytest.raises(TypeError):AuxiliaryCurrentSpace(9,True,"PROTON",1)
    with pytest.raises(ValueError):AuxiliaryCurrentSpace(9,2,"PROTON",True)
